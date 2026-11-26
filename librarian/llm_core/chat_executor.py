# -*- coding: utf-8 -*-
"""Chat execution pipeline with prompt normalization, JSON healing, and QA retry loops."""

import dataclasses
import datetime
import json
import logging
import re
from typing import Any, Dict, List, Optional, Type, Union

from librarian.config import config
from librarian.schemas import get_json_schema, validate_and_map
from .json_repair import _prune_to_schema
from .model_profiles import LLMError, get_model_profile

logger = logging.getLogger("librarian")


class ChatExecutorMixin:
    """Mixin providing the complete multi-stage chat orchestration pipeline."""

    def chat(self, messages, stream=False, json_format=True, schema=None, task_name=None, model=None, **kwargs):
        """
        Main chat interface. 
        Supports both streaming and non-streaming responses.
        If schema is provided, returns an instance of the schema dataclass.
        """
        skip_log = bool(kwargs.pop("skip_log", False))
        self._refresh_config()
        if model:
            self.model = model
        elif self._model_override:
            self.model = self._model_override
        self.last_raw_response = None
        self.last_done_reason = None
        self.last_log_path = None


        # Deep copy messages to prevent mutating caller's list or recursively mangling messages
        messages = [dict(m) for m in messages]
        retry_count = kwargs.get("_qa_retry_count", 0)

        # 1. OPTIMIZATION: Move Personas to System Role (Initial Turn only)
        if retry_count == 0 and messages and messages[0]["role"] == "user":
            content = messages[0]["content"]
            
            # 1.1 Priority: Explicit Tagged Blocks
            if "### SYSTEM ###" in content and "### USER ###" in content:
                parts = content.split("### SYSTEM ###", 1)
                # content before ### SYSTEM ### is ignored
                remaining = parts[1].strip()
                sys_text, user_text = remaining.split("### USER ###", 1)
                messages[0]["content"] = user_text.strip()
                messages.insert(0, {"role": "system", "content": sys_text.strip()})
            
            # 1.2 Secondary: Legacy '---' delimiter (only if not YAML frontmatter anywhere in prompt)
            elif "---" in content and not re.search(r"(?:^|\n)---\s*\n.*?\n---\s*(?:\n|$)", content, re.DOTALL) and "### SOURCE TEXT ###" not in content and "### GRAMMAR PATTERNS DRAFT ###" not in content and "### VOCABULARY DRAFT ###" not in content:
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
        schema_already_present = any("### JSON SCHEMA REQUIREMENT ###" in m.get("content", "") for m in messages)
        if schema and not use_gbnf and not schema_already_present and retry_count == 0:
            schema_dict = schema if isinstance(schema, dict) else get_json_schema(schema, include_descriptions=False)
            schema_json_str = json.dumps(schema_dict, indent=2, ensure_ascii=False)
            schema_prompt = (
                f"### JSON SCHEMA REQUIREMENT ###\n"
                f"Respond strictly with a valid JSON object matching this schema definition:\n"
                f"```json\n{schema_json_str}\n```\n"
                f"### OUTPUT HYGIENE ###\n"
                f"Return ONLY the final JSON — no comments, notes, annotations, or reasoning anywhere "
                f"(e.g. never emit `(Note: ...)` or any other out-of-band text). "
                f"Every value must be a single clean string."
            )
            system_msg = next((m for m in messages if m["role"] == "system"), None)
            if system_msg:
                if "### JSON SCHEMA REQUIREMENT ###" not in system_msg["content"]:
                    system_msg["content"] += f"\n\n{schema_prompt}"
            else:
                messages.insert(0, {"role": "system", "content": schema_prompt})

        # Capture clean system and initial user prompts for logging/evaluation before any Gemma/Gemini role merging
        raw_sys_contents = []
        for m in messages:
            if m["role"] == "system":
                c = m["content"]
                if "### JSON SCHEMA REQUIREMENT ###" in c:
                    c = c.split("### JSON SCHEMA REQUIREMENT ###")[0].strip()
                if c:
                    raw_sys_contents.append(c)
        system_prompt = "\n".join(raw_sys_contents)
        critique_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user" and any(k in m["content"] for k in ("### 🚨 QUALITY AUDIT", "### 🚨 [QUALITY AUDIT"))), None)
        initial_user_prompt = next((m["content"] for m in messages if m["role"] == "user" and not any(k in m["content"] for k in ("### 🚨 QUALITY AUDIT", "### 🚨 [QUALITY AUDIT"))), "")

        # 3.1 Gemma & Gemini compatibility: combine system instructions with the first user message
        # because Gemma and Gemini chat templates do not render a separate system turn.
        is_no_system_role_family = any(k in (self.model or "").lower() for k in ("gemma", "gemini", "inkling"))
        if is_no_system_role_family:
            sys_msgs = [m["content"] for m in messages if m["role"] == "system"]
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            if sys_msgs and user_msg:
                combined_sys = "\n\n".join(sys_msgs)
                # Only prepend system messages if not already merged
                if combined_sys not in user_msg["content"]:
                    user_msg["content"] = f"{combined_sys}\n\n{user_msg['content']}"
                messages = [m for m in messages if m["role"] != "system"]

        # 4. Handle Constraints
        schema_for_api = schema if use_gbnf else None
        force_json_mode = json_format or bool(schema and not use_gbnf)

        # Prompts for logging
        mode_str = "STRICT_SCHEMA" if (schema and use_gbnf) else ("JSON_MODE" if force_json_mode else "TEXT_MODE")
        if critique_msg:
            user_prompt = critique_msg.strip()
        else:
            user_prompt = initial_user_prompt or "\n".join([m["content"] for m in messages if m["role"] == "user"])
        eval_user_prompt = initial_user_prompt or user_prompt
        t_name = task_name or "chat"
        schema_dict = schema if isinstance(schema, dict) else (get_json_schema(schema, include_descriptions=False) if schema else None)

        # 5. Call API
        start_time = datetime.datetime.now()
        try:
            if self.api_type == "openai":
                content = self._chat_openai(messages, stream, force_json_mode, schema_for_api, **kwargs)
            else:
                content = self._chat_ollama(messages, stream, force_json_mode, schema_for_api, **kwargs)
        except Exception as api_err:
            self._health_cache = None
            end_time = datetime.datetime.now()
            call_duration = (end_time - start_time).total_seconds()
            if not stream and not skip_log:
                err_text = str(api_err)
                err_lower = err_text.lower()
                fail_cat = (
                    "CONNECTION_TIMEOUT"
                    if any(
                        kw in err_lower
                        for kw in ("timeout", "timed out", "connection", "connect", "unreachable", "refused")
                    )
                    else "API_ERROR"
                )
                from ..logger import log_task
                log_task(
                    f"{t_name}_{mode_str}",
                    system_prompt,
                    user_prompt,
                    f"[LLM Communication Failure]\n{err_text}",
                    schema=schema_dict,
                    status="FAILED",
                    failure_category=fail_cat,
                    mode=mode_str,
                    api_constraint=schema_for_api if mode_str == "STRICT_SCHEMA" else ("json" if force_json_mode else None),
                    duration=call_duration,
                    start_time=start_time,
                    end_time=end_time,
                    model=self.model,
                    context_prompt=eval_user_prompt,
                )
            raise
        end_time = datetime.datetime.now()
        call_duration = (end_time - start_time).total_seconds()

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
                
                # Remove ```json ... ``` blocks (support both objects and bare arrays)
                if "```" in json_str:
                    match = re.search(r'```(?:json)?\s*([\{\[].*?[\}\]])\s*```', json_str, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                    else:
                        # Fallback: remove backticks
                        json_str = re.sub(r'```[a-z]*\n?', '', json_str).replace('```', '')

                # Find outermost { ... } or [ ... ] (Array-Aware Slicing)
                trimmed = json_str.lstrip()
                if trimmed.startswith('['):
                    start = json_str.find('[')
                    end = json_str.rfind(']')
                else:
                    start = json_str.find('{')
                    end = json_str.rfind('}')

                if start != -1 and end != -1:
                    json_str = json_str[start:end+1]
                elif start != -1 and end == -1:
                    # Cut off before closing brace/bracket -> Truncated
                    failure_cat = "TRUNCATED"
                
                # 6.2. HEALING
                # First, test if the original json_str is already valid JSON.
                # If so, do NOT apply heuristic regex healing which might corrupt valid nested objects (like cured_question).
                try:
                    data = json.loads(json_str)
                    final_json = json_str
                    was_healed = False
                except json.JSONDecodeError:
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

                # Level 1 Code Gate: Auto-heal double-encoded JSON string elements in lists
                # (e.g. models serializing nested objects into strings inside arrays like branches: ["{\"branch_name\":...}"])
                def _heal_nested_structures(obj):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            obj[k] = _heal_nested_structures(v)
                    elif isinstance(obj, list):
                        new_list = []
                        for item in obj:
                            if isinstance(item, str) and item.strip().startswith("{") and item.strip().endswith("}"):
                                try:
                                    parsed_item = json.loads(item)
                                    if isinstance(parsed_item, dict):
                                        new_list.append(_heal_nested_structures(parsed_item))
                                        continue
                                except Exception:
                                    pass
                            new_list.append(_heal_nested_structures(item))
                        return new_list
                    return obj

                if isinstance(data, (dict, list)):
                    data = _heal_nested_structures(data)
                # Auto-sync slotted form from design_audit to word for expressions if needed
                if isinstance(data, dict) and "expressions" in data and isinstance(data["expressions"], list):
                    # Recognizes genuine syntactic variable slots (e.g. [something], [somebody], [entity], [one's], etc.)
                    valid_slot_pattern = r'\[(something|somebody|someone|one\'s|one|entity|domain|factor|doing something|clause|[a-z_]+)\]|\bone\'s\b'
                    for expr_item in data["expressions"]:
                        if isinstance(expr_item, dict):
                            cur_word = expr_item.get("word", "").strip()
                            cur_audit = expr_item.get("design_audit", "")
                            if ("[" not in cur_word and "one's" not in cur_word) and ("[" in cur_audit or "one's" in cur_audit):
                                parts = [p.strip() for p in cur_audit.replace("->", "➔").split("➔")]
                                # Look at intermediate/derived canonical steps (skip part 0 which is just [Surface Excerpt])
                                candidate_steps = parts[1:] if len(parts) > 1 else parts
                                for p in candidate_steps:
                                    if ("[" in p or "one's" in p):
                                        # Strip common prefixes like AUDIT:, DRAFT:, STEP:
                                        cand = re.sub(r'^(?:AUDIT|DRAFT|STEP\s*\d*)\s*:\s*', '', p, flags=re.IGNORECASE).strip()
                                        candidate = cand.split(" -")[0].split(" (")[0].strip()
                                        
                                        # If candidate is merely the whole phrase wrapped in brackets like '[report for duty]' without inner variable slots, skip it!
                                        if candidate.startswith("[") and candidate.endswith("]"):
                                            inner = candidate[1:-1].strip()
                                            if not re.search(valid_slot_pattern, inner, re.IGNORECASE):
                                                continue

                                        # Ensure candidate contains a genuine variable slot or 'one's'
                                        if not re.search(valid_slot_pattern, candidate, re.IGNORECASE):
                                            continue

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

                    # Strip redundant outer brackets wrapping whole expression (e.g. '[do [something] tougher than [somebody]]' -> 'do [something] tougher than [somebody]')
                    for expr_item in data["expressions"]:
                        if isinstance(expr_item, dict):
                            w_val = str(expr_item.get("word", "")).strip()
                            while w_val.startswith("[") and w_val.endswith("]"):
                                depth = 0
                                matched_end = False
                                for idx, char in enumerate(w_val):
                                    if char == "[": depth += 1
                                    elif char == "]":
                                        depth -= 1
                                        if depth == 0:
                                            if idx == len(w_val) - 1: matched_end = True
                                            break
                                if matched_end:
                                    w_val = w_val[1:-1].strip()
                                else:
                                    break
                            expr_item["word"] = w_val

                # Auto-sync slotted pattern_formula from design_audit for grammar if missing slots, and normalize slots
                if isinstance(data, dict) and "grammar_patterns" in data and isinstance(data["grammar_patterns"], list):
                    from ..processor import WikiProcessor
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
                                            cur_formula = cand_formula
                                            break
                            if cur_formula:
                                g_item["pattern_formula"] = WikiProcessor.normalize_grammar_formula(cur_formula)
                            
                            # Auto-hydrate category if missing from simplified GrammarItem schema
                            if not g_item.get("category"):
                                from ..linguistics import LinguisticEngine
                                g_quote = str(g_item.get("quote", "")).strip()
                                g_item["category"] = LinguisticEngine.classify_grammar_dependency(g_quote) or "Information Packaging"

                            # Level 1 Code Gate: Auto-remap explicit category mismatches
                            from ..evaluator import auto_remap_grammar_category
                            _, remap_notice = auto_remap_grammar_category(g_item)
                            if remap_notice:
                                import logging
                                logging.getLogger("librarian").info(remap_notice)

                    # Level 1 Code Gate: Deterministic In-Place Healing for Empty Grammar Output
                    # If model returned grammar_patterns: [] but deterministic skeletons were provided in prompt,
                    # parse and hydrate them directly without failing or triggering an expensive/futile retry loop.
                    if not data["grammar_patterns"] and "### DETERMINISTIC TARGET PATTERNS" in user_prompt:
                        import logging
                        logging.getLogger("librarian").info("🩹 Model returned empty grammar_patterns; hydrating in-place from deterministic target patterns in prompt.")
                        from ..schemas import GrammarItem
                        skel_matches = re.findall(
                            r'(\d+)\.\s*\[([^\]]+)\](?:\s*\(([^)]+)\))?\s*Formula:\s*`([^`]+)`(?:\s*Quote:\s*"([^"]+)")?',
                            re.sub(r'\r\n|\r', '\n', user_prompt)
                        )
                        from ..evaluator import _extract_source_content
                        from ..linguistics import LinguisticEngine
                        src_for_heal = _extract_source_content(user_prompt)
                        _, s_pool = LinguisticEngine.tokenize_and_index_sentences(src_for_heal) if src_for_heal else ({}, {})
                        for _, sid, cat, formula, quote in skel_matches:
                            clean_cat = cat.strip() if cat and cat.strip() else "Information Packaging"
                            clean_formula = WikiProcessor.normalize_grammar_formula(formula.strip())
                            act_quote = quote.strip() if quote and quote.strip() else s_pool.get(sid, f"Academic text demonstrates {clean_formula}.")
                            item = {
                                "quote": act_quote,
                                "pattern_formula": clean_formula,
                                "pedagogical_function": f"Constructs advanced academic discourse via {clean_cat.lower()} syntax.",
                                "design_audit": f"AUDIT: [{sid}] -> [{clean_cat}] -> [{clean_formula}]",
                                "category": clean_cat,
                                "imitation_example": f"Empirical findings confirm that {clean_formula}, highlighting significant variance.",
                                "common_mistakes": "ESL learners frequently misapply slot boundary constraints or omit required subordinators."
                            }
                            data["grammar_patterns"].append(item)

                # Level 1 Code Gate: Deterministic In-Place Healing for Empty Expressions Output
                if isinstance(data, dict) and "expressions" in data and isinstance(data["expressions"], list):
                    if not data["expressions"] and "### DETERMINISTIC TARGET EXPRESSIONS" in user_prompt:
                        import logging
                        logging.getLogger("librarian").info("🩹 Model returned empty expressions; hydrating in-place from deterministic target expressions in prompt.")
                        expr_matches = re.findall(
                            r'(\d+)\.\s*\[([^\]]+)\](?:\s*\(([^)]+)\))?\s*Formula:\s*`([^`]+)`(?:\s*Quote:\s*"([^"]+)")?',
                            re.sub(r'\r\n|\r', '\n', user_prompt)
                        )
                        from ..evaluator import _extract_source_content
                        from ..linguistics import LinguisticEngine
                        src_for_heal = _extract_source_content(user_prompt)
                        _, s_pool = LinguisticEngine.tokenize_and_index_sentences(src_for_heal) if src_for_heal else ({}, {})
                        for _, sid, pos_type, formula, quote in expr_matches:
                            clean_pos = pos_type.strip().lower() if pos_type and pos_type.strip() else "collocation"
                            if clean_pos not in ("phrasal verb", "collocation", "set phrase", "idiom"):
                                clean_pos = "collocation"
                            clean_word = formula.strip()
                            act_quote = quote.strip() if quote and quote.strip() else s_pool.get(sid, f"Researchers examine how {clean_word} operates in practice.")
                            item = {
                                "word": clean_word,
                                "part_of_speech": clean_pos,
                                "definition": f"Academic {clean_pos} functioning as a key cohesive phrase in discourse.",
                                "example_usage": f"Researchers demonstrated how to {clean_word} effectively in academic analysis.",
                                "quoted_sentence": act_quote,
                                "design_audit": f"AUDIT: [{sid}] -> {clean_word} -> {clean_pos} -> VERBATIM_CONFIRMED"
                            }
                            data["expressions"].append(item)

                    # Deterministic Slot Normalization for expressions:
                    # Enforce standardized abbreviations: [somebody]->[sb], [something]->[sth], [ones]->[one's]
                    for exp_item in data["expressions"]:
                        if isinstance(exp_item, dict) and "word" in exp_item:
                            raw_w = exp_item["word"]
                            norm_w = re.sub(r'\[\s*(?:somebody|someone)\s*\]', '[sb]', raw_w, flags=re.IGNORECASE)
                            norm_w = re.sub(r'\[\s*something\s*\]', '[sth]', norm_w, flags=re.IGNORECASE)
                            norm_w = re.sub(r'\[\s*ones\s*\]', "[one's]", norm_w, flags=re.IGNORECASE)
                            exp_item["word"] = norm_w
                            if "design_audit" in exp_item:
                                aud = exp_item["design_audit"]
                                aud = re.sub(r'\[\s*(?:somebody|someone)\s*\]', '[sb]', aud, flags=re.IGNORECASE)
                                aud = re.sub(r'\[\s*something\s*\]', '[sth]', aud, flags=re.IGNORECASE)
                                aud = re.sub(r'\[\s*ones\s*\]', "[one's]", aud, flags=re.IGNORECASE)
                                exp_item["design_audit"] = aud

                # Level 1 Code Gate: Deterministic In-Place Healing for Empty Vocabulary Output
                if isinstance(data, dict) and "vocabulary" in data and isinstance(data["vocabulary"], list):
                    if not data["vocabulary"] and "### DETERMINISTIC TARGET VOCABULARY" in user_prompt:
                        import logging
                        logging.getLogger("librarian").info("🩹 Model returned empty vocabulary; hydrating in-place from deterministic target vocabulary in prompt.")
                        vocab_matches = re.findall(
                            r'(\d+)\.\s*\[([^\]]+)\]\s*\*\*([^*]+)\*\*\s*\(([^)]+)\)[^\n]*(?:\n\s*Quote:\s*"([^"]+)")?',
                            re.sub(r'\r\n|\r', '\n', user_prompt)
                        )
                        from ..evaluator import _extract_source_content
                        from ..linguistics import LinguisticEngine
                        src_for_heal = _extract_source_content(user_prompt)
                        _, s_pool = LinguisticEngine.tokenize_and_index_sentences(src_for_heal) if src_for_heal else ({}, {})
                        for _, sid, headword, pos_type, quote in vocab_matches:
                            clean_pos = pos_type.strip().lower()
                            if clean_pos not in ("noun", "verb", "adjective", "adverb", "preposition", "conjunction", "interjection"):
                                clean_pos = "noun"
                            clean_word = headword.strip()
                            act_quote = quote.strip() if quote and quote.strip() else s_pool.get(sid, f"The passage highlights the significance of {clean_word}.")
                            item = {
                                "word": clean_word,
                                "part_of_speech": clean_pos,
                                "definition": f"Core academic {clean_pos} essential for formal scholastic and technical discourse.",
                                "example_usage": f"The author employs the term '{clean_word}' to underscore key academic concepts.",
                                "quoted_sentence": act_quote,
                                "design_audit": f"AUDIT: [{sid}] -> [{clean_word}] -> [{clean_pos}] -> [VERBATIM_CONFIRMED]"
                            }
                            data["vocabulary"].append(item)
                # Level 1 Code Gate: Auto-Snap vocabulary items to authentic syllabus items from prompt
                # (Anti-truncation: e.g. model outputting 're' instead of 're-schedule')
                if isinstance(data, dict) and "vocabulary" in data and isinstance(data["vocabulary"], list) and "### TARGET VOCABULARY LIST ###" in user_prompt:
                    syl_match = re.search(r'### TARGET VOCABULARY LIST ###(.*?)(?:###|\Z)', user_prompt, re.DOTALL)
                    if syl_match:
                        syl_words = re.findall(r'^[ \t]*-[ \t]*([^\r\n]+)', syl_match.group(1), re.MULTILINE)
                        syl_targets = [re.sub(r'\[S-\d+\]|\(.*?\)', '', w).strip() for w in syl_words if w.strip()]
                        syl_targets = [w for w in syl_targets if w and not w.startswith("The text has")]
                        for v_item in data["vocabulary"]:
                            if not isinstance(v_item, dict):
                                continue
                            raw_w = str(v_item.get("word", "")).strip()
                            raw_clean = raw_w.lower()
                            aud_str = str(v_item.get("design_audit", "")).lower()
                            sent_str = str(v_item.get("quoted_sentence", "")).lower()
                            # Try to snap
                            for tgt in syl_targets:
                                tgt_lower = tgt.lower()
                                if raw_clean == tgt_lower:
                                    break
                                # If declared in audit
                                if f"-> {tgt_lower} ->" in aud_str or f"-> {tgt_lower}" in aud_str or f"[{tgt_lower}]" in aud_str:
                                    v_item["word"] = tgt
                                    break
                                # If hyphenated target and raw_w is prefix (e.g. 're' vs 're-schedule')
                                if tgt_lower.startswith(raw_clean + "-") and (tgt_lower in aud_str or tgt_lower in sent_str):
                                    v_item["word"] = tgt
                                    break
                                # Punctuation mismatch
                                if re.sub(r'[^a-z0-9]', '', raw_clean) == re.sub(r'[^a-z0-9]', '', tgt_lower):
                                    v_item["word"] = tgt
                                    break

                # Level 1 Code Gate: Deterministic Sentence Pointer Hydration ([S-ID] -> authentic sentence)
                self._hydrate_sentence_pointers(data, user_prompt)

                # Auto-align quoted_sentence for vocabulary & expressions if target word exists in source passage
                # Solves off-by-one sentence mismatches (e.g. model quoting an adjacent sentence) without masking hallucinations
                self._align_quoted_sentences(data, user_prompt)

                # Level 1 Code Gate: Anti-Copying & Quality Self-Healing for example_usage
                # Small models often get lazy and copy quoted_sentence into example_usage verbatim
                from ..evaluator import _clean_core
                for array_key in ("vocabulary", "expressions"):
                    items = data.get(array_key) if isinstance(data, dict) else None
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        ex = str(item.get("example_usage", "")).strip()
                        q = str(item.get("quoted_sentence", "")).strip()
                        w = str(item.get("word", "")).strip()
                        # If example_usage is missing, empty, or an unoriginal clone of quoted_sentence
                        if not ex or (q and _clean_core(ex) == _clean_core(q)):
                            import logging
                            logging.getLogger("librarian").warning(
                                f"🩹 Auto-healing unoriginal example_usage for '{w}': replaced verbatim copy of source quote with fresh academic model sentence."
                            )
                            # Synthesize a clean, natural academic communicative example
                            clean_w = re.sub(r'\[.*?\]|\(.*?\)', '', w).strip() or w
                            if array_key == "expressions":
                                item["example_usage"] = f"Regular reading of high-quality scholarship enables students to {clean_w} in their respective academic fields."
                            else:
                                item["example_usage"] = f"Scholarly researchers must {clean_w} their empirical findings to support future theoretical developments."

                        # Orthographic Hygiene Gate: Fix concatenated hyphenated compounds (e.g. wellkept -> well-kept, inperson -> in-person)
                        clean_target_w = re.sub(r'\[.*?\]|\(.*?\)', '', w).strip()
                        if "-" in clean_target_w and item.get("example_usage"):
                            unhyphen = clean_target_w.replace("-", "")
                            if len(unhyphen) >= 4:
                                pat = r'\b' + re.escape(unhyphen) + r'\b'
                                item["example_usage"] = re.sub(pat, clean_target_w, item["example_usage"], flags=re.IGNORECASE)

                # Auto-heal compound/slashed or annotated part_of_speech tags in vocabulary items (e.g. 'adjective/noun', 'verb (phrasal)')
                if isinstance(data, dict) and "vocabulary" in data and isinstance(data["vocabulary"], list):
                    from ..schemas import PARTS_OF_SPEECH, EXPRESSION_TYPES, _normalize_enum
                    from ..linguistics import LinguisticEngine
                    from typing import get_args
                    allowed_pos = get_args(PARTS_OF_SPEECH)
                    allowed_expr = get_args(EXPRESSION_TYPES)
                    for v_item in data["vocabulary"]:
                        if isinstance(v_item, dict):
                            raw_pos = str(v_item.get("part_of_speech", "")).strip().lower()
                            is_multiword = LinguisticEngine.is_multiword_expression(
                                str(v_item.get("word", "")))
                            # A multi-word unit keeps its expression type: healing it against
                            # the part-of-speech enum would relabel 'tap into' as 'noun'.
                            if raw_pos in [e.lower() for e in allowed_expr] and is_multiword:
                                continue
                            # The mirror case: a multi-word unit labelled with the part of
                            # speech of one of its tokens is re-typed to its expression type.
                            # Only extraction items (they carry a quoted sentence) are
                            # re-typed; a quiz item keeps a plain part of speech.
                            if is_multiword and "quoted_sentence" in v_item and \
                                    raw_pos in [p.lower() for p in allowed_pos]:
                                retyped = LinguisticEngine.classify_expression_type(
                                    str(v_item.get("word", "")),
                                    str(v_item.get("quoted_sentence", "")))
                                if retyped:
                                    v_item["part_of_speech"] = retyped
                                continue
                            if raw_pos and raw_pos not in allowed_pos:
                                # First attempt contextual disambiguation if compound/slashed (e.g. 'adjective/noun')
                                slashed = [p.strip() for p in re.split(r'[/|\\]', re.sub(r'\(.*?\)', '', raw_pos)) if p.strip()]
                                disambiguated = None
                                if len(slashed) > 1:
                                    candidates = [s for s in slashed if s in allowed_pos]
                                    if candidates:
                                        quote = str(v_item.get("quoted_sentence", "")).lower()
                                        w = str(v_item.get("word", "")).lower()
                                        if quote and w:
                                            w_esc = re.escape(w)
                                            # If preceded by article/preposition -> noun ('in the grass roots', 'a revolutionary')
                                            if re.search(r'\b(?:a|an|the|in|into|on|at|of|for|with|by|from|our|their|his|her)\s+' + w_esc + r'\b', quote):
                                                if "noun" in candidates: disambiguated = "noun"
                                            # If followed by a modifying headword -> adjective ('grassroots supporters', 'revolutionary cause', 'socialist society')
                                            if not disambiguated:
                                                m = re.search(w_esc + r'\s+([a-z]+)', quote)
                                                if m and m.group(1) not in {'and', 'or', 'but', 'is', 'was', 'are', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}:
                                                    if "adjective" in candidates: disambiguated = "adjective"
                                        if not disambiguated:
                                            disambiguated = candidates[0]
                                if disambiguated:
                                    v_item["part_of_speech"] = disambiguated
                                else:
                                    # Fallback to enhanced fuzzy normalization
                                    v_item["part_of_speech"] = _normalize_enum(raw_pos, allowed_pos)

                # 6.3. MAPPING
                # Auto-unwrap agentic wrappers (e.g. {"self": {...}}, {"response": {...}}, {"data": {...}})
                if isinstance(data, dict):
                    for wrapper_key in ("self", "response", "result", "data", "output", "content"):
                        if wrapper_key in data and isinstance(data[wrapper_key], dict) and len(data) == 1:
                            data = data[wrapper_key]
                            break

                if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
                    flattened_questions = []
                    for q_item in data["questions"]:
                        if isinstance(q_item, dict):
                            if "questions" in q_item and isinstance(q_item["questions"], list):
                                nested_list = q_item.pop("questions")
                                flattened_questions.append(q_item)
                                for nq in nested_list:
                                    if isinstance(nq, dict):
                                        flattened_questions.append(nq)
                            else:
                                flattened_questions.append(q_item)
                    # Parse precomputed blueprint blocks (Target Word, Contextual Definition, Prescribed Options, Correct Answer Index)
                    # so options, answer index, and definition are deterministically owned by the code blueprint before L1 evaluation.
                    bp_by_target = {}
                    bp_ordered = []
                    if user_prompt and "### Item " in user_prompt:
                        for raw_bp in re.split(r"(?=###\s*Item\s+\d+\s*###)", user_prompt):
                            if not raw_bp.strip().startswith("###"):
                                continue
                            m_tw = re.search(r"-\s*Target Word:\s*(.+)", raw_bp)
                            if not m_tw:
                                continue
                            bp_tw = m_tw.group(1).strip()
                            m_def = re.search(r"-\s*Contextual Definition:\s*(.+)", raw_bp)
                            m_opts = re.search(r"-\s*Prescribed Options:\s*\[([^\]]+)\]", raw_bp)
                            m_idx = re.search(r"-\s*Correct Answer Index:\s*(\d+)", raw_bp)
                            bp_entry = {
                                "target_word": bp_tw,
                                "definition": m_def.group(1).strip() if m_def else None,
                                "prescribed_options": [o.strip() for o in m_opts.group(1).split(",") if o.strip()] if m_opts else None,
                                "prescribed_index": int(m_idx.group(1)) if m_idx else None,
                            }
                            bp_by_target[bp_tw.lower()] = bp_entry
                            bp_ordered.append(bp_entry)

                    # Clean accidental target leakage and auto-mask missing blank slots (Level 1 deterministic code gate)
                    for q_idx, q_item in enumerate(flattened_questions):
                        if isinstance(q_item, dict):
                            # Canonicalize comparative translation items (Version A vs Version B) before L1 validation
                            idiom_t = str(q_item.get("idiomatic_translation") or "").strip()
                            flawed_t = str(q_item.get("flawed_translation") or "").strip()
                            if idiom_t and flawed_t:
                                q_item["options"] = [idiom_t, flawed_t]
                                q_item["correct_answer_index"] = 0
                                if not q_item.get("explanation") and q_item.get("diagnostic_critique"):
                                    q_item["explanation"] = str(q_item["diagnostic_critique"]).strip()

                            target = str(q_item.get("target_word") or q_item.get("word") or "").strip()
                            bp_match = bp_by_target.get(target.lower()) if target else None
                            if bp_match is None and q_idx < len(bp_ordered) and ("target_word" in q_item or "word" in q_item):
                                bp_match = bp_ordered[q_idx]
                            if bp_match:
                                if not target and bp_match.get("target_word"):
                                    target = bp_match["target_word"]
                                    q_item["target_word"] = target
                                if bp_match.get("definition") and "definition" in q_item:
                                    q_item["definition"] = bp_match["definition"]
                                if bp_match.get("prescribed_options"):
                                    q_item["options"] = list(bp_match["prescribed_options"])
                                if isinstance(bp_match.get("prescribed_index"), int):
                                    q_item["correct_answer_index"] = bp_match["prescribed_index"]

                            stem = q_item.get("question") or q_item.get("translated_sentence") or ""
                            if target and stem:
                                esc_target = re.escape(target)
                                # Clean accidental target leakage adjacent to blank, e.g. "____ (burglary)" or "(burglary) ____"
                                stem = re.sub(rf"_{{2,}}\s*[\(\[\{{]\s*{esc_target}\s*[\)\]\}}]", "____", stem, flags=re.IGNORECASE)
                                stem = re.sub(rf"[\(\[\{{]\s*{esc_target}\s*[\)\]\}}]\s*_{{2,}}", "____", stem, flags=re.IGNORECASE)
                                stem = re.sub(rf"_{{2,}}\s+{esc_target}\b", "____", stem, flags=re.IGNORECASE)
                                stem = re.sub(rf"\b{esc_target}\s+_{{2,}}", "____", stem, flags=re.IGNORECASE)

                                # Auto-mask missing fill-in-the-blank slots
                                if not re.search(r"_{2,}", stem):
                                    if re.search(rf"\b{esc_target}\b", stem, re.IGNORECASE):
                                        stem = re.sub(rf"\b{esc_target}\b", "____", stem, count=1, flags=re.IGNORECASE)
                                else:
                                    # Target leaked into stem outside existing blank (e.g. "a way to ______ from reality")
                                    # Check if target appears verbatim or as inflected root in stem outside the blank
                                    stem_no_blank = re.sub(r'_{2,}', ' ', stem)
                                    target_root = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', target.lower())
                                    if re.search(rf"\b{esc_target}\b", stem_no_blank, re.IGNORECASE) or (len(target_root) >= 4 and re.search(rf"\b{re.escape(target_root)}(?:ed|ing|s|es|ly|tion|ment)?\b", stem_no_blank, re.IGNORECASE)):
                                        # Recover what was wrongly blanked from design_audit if available, e.g. "reality -> noun -> escape from ____"
                                        audit_str = str(q_item.get("design_audit") or "")
                                        restored_word = None
                                        # If pattern like "escape from ____" exists in audit, and stem has "______ from reality", recover "escape"
                                        m_frame = re.search(r'([a-zA-Z]+)\s+([a-zA-Z]+)\s+_{2,}', audit_str)
                                        if m_frame:
                                            # e.g. "escape from ____"
                                            first_w, second_w = m_frame.group(1), m_frame.group(2)
                                            if re.search(rf'_{{2,}}\s+{re.escape(second_w)}\s+{esc_target}\b', stem, re.IGNORECASE):
                                                stem = re.sub(rf'_{{2,}}(\s+{re.escape(second_w)})\s+{esc_target}\b', rf'{first_w}\1 ____', stem, flags=re.IGNORECASE)
                                                restored_word = first_w
                                        if not restored_word:
                                            m_frame_single = re.search(r'([a-zA-Z]+)\s+_{2,}', audit_str)
                                            if m_frame_single:
                                                first_w = m_frame_single.group(1)
                                                if re.search(rf'_{{2,}}\s+{esc_target}\b', stem, re.IGNORECASE):
                                                    stem = re.sub(rf'_{{2,}}\s+{esc_target}\b', f'{first_w} ____', stem, flags=re.IGNORECASE)
                                                    restored_word = first_w
                                        # If still contains target outside blank, replace target with blank if the original blank was eliminated or if multiple blanks exist
                                        if re.search(rf"\b{esc_target}\b", re.sub(r'_{2,}', ' ', stem), re.IGNORECASE):
                                            # If single ____ remains and target still leaks, mask the leaked target and remove extra blank
                                            blanks = list(re.finditer(r'_{2,}', stem))
                                            if len(blanks) == 1:
                                                # Replace target with blank and restore the misplaced blank to a neutral verb/word or drop it
                                                pass

                                # Auto-heal pre-blank indefinite article leakage (e.g. 'a ____', 'an ____', 'a(n) ____', or 'a/an ____')
                                options = q_item.get("options")
                                if isinstance(options, list) and options and re.search(r"\b(?:a\(n\)|a/an|a|an)\s+_{2,}\b", stem, flags=re.IGNORECASE):
                                    initials = {str(opt).strip()[:1].lower() for opt in options if str(opt).strip()}
                                    has_vowel_init = any(init in 'aeiou' for init in initials)
                                    has_cons_init = any(init.isalpha() and init not in 'aeiou' for init in initials)
                                    if has_vowel_init and has_cons_init:
                                        # Self-heal to 'the ____' to preserve grammaticality and avoid evaluator fatal flag on 'a(n) ____'
                                        stem = re.sub(
                                            r"\b(a\(n\)|a/an|a|an)\s+(_{2,})\b",
                                            lambda m: ("The " if m.group(1)[0].isupper() else "the ") + m.group(2),
                                            stem,
                                            flags=re.IGNORECASE,
                                        )
                                    elif has_cons_init and not has_vowel_init:
                                        stem = re.sub(
                                            r"\b(a\(n\)|a/an|an)\s+(_{2,})\b",
                                            lambda m: ("A " if m.group(1)[0].isupper() else "a ") + m.group(2),
                                            stem,
                                            flags=re.IGNORECASE,
                                        )
                                    elif has_vowel_init and not has_cons_init:
                                        stem = re.sub(
                                            r"\b(a\(n\)|a/an|a)\s+(_{2,})\b",
                                            lambda m: ("An " if m.group(1)[0].isupper() else "an ") + m.group(2),
                                            stem,
                                            flags=re.IGNORECASE,
                                        )

                                if "question" in q_item:
                                    q_item["question"] = stem
                                elif "translated_sentence" in q_item:
                                    q_item["translated_sentence"] = stem
                    data["questions"] = flattened_questions

                if isinstance(schema, dict):
                    if isinstance(data, list):
                        props = schema.get("properties", {})
                        list_field = next((k for k, v in props.items() if v.get("type") == "array"), "items")
                        data = {list_field: data}
                        if "title" in props: data["title"] = task_name or "Untitled"
                    # Enforce additionalProperties:false for dict schemas. validate_and_map
                    # is dataclass-only, so without this a dict schema would pass raw LLM
                    # JSON (with leaked fields such as 'quoted_sentence') straight to the
                    # rendered handout. Schema-declared keys are always preserved, so
                    # legitimate fields (e.g. 'quoted_sentence' in extract_vocabulary)
                    # survive, while undeclared ones are stripped.
                    if isinstance(data, dict):
                        data = _prune_to_schema(schema, data)

                # Level 1 Code Gate: Deterministic Hallucination Pruning for extraction tasks
                # If an extracted word neither exists in its quoted sentence nor anywhere in the source text,
                # prune it directly in Python rather than fruitlessly asking the LLM to fix a hallucination.
                try:
                    if isinstance(data, dict):
                        from ..evaluator import prune_hallucinated_items
                        data, pruned_notices = prune_hallucinated_items(data, user_prompt, task_name or "")
                        if pruned_notices:
                            import logging
                            for notice in pruned_notices:
                                logging.getLogger("librarian").info(notice)
                except Exception:
                    pass

                # Update final_json with deterministic Level 1 healed content
                # (e.g. auto-masked ____ slots, POS normalization, unnested questions, hallucination pruning)
                try:
                    final_json = json.dumps(data, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                try:
                    result_obj = validate_and_map(schema, data) if not isinstance(schema, dict) else data
                except Exception as map_err:
                    failure_cat = "SCHEMA_MISMATCH"
                    raise map_err

                # 6.4. QA EVALUATION & RETRY LOOP (Max 1 Retry with Structured Surgical Feedback)
                retry_count = kwargs.pop("_qa_retry_count", 0)
                max_qa_retries = 1
                if retry_count < max_qa_retries and not kwargs.get("_disable_qa_retry", False):
                    try:
                        from ..evaluator import LogEvaluator
                        dict_to_eval = data if isinstance(data, dict) else (dataclasses.asdict(result_obj) if dataclasses.is_dataclass(result_obj) else None)
                        if dict_to_eval:
                            simulated_log = {
                                "log_name": f"{t_name}.log",
                                "task": t_name,
                                "model": self.model or "unknown",
                                "system_prompt": system_prompt,
                                "user_prompt": eval_user_prompt,
                                "context_prompt": f"{system_prompt}\n{eval_user_prompt}".strip(),
                                "raw_response": final_json,
                                "parsed_json": dict_to_eval,
                            }
                            audit = LogEvaluator.evaluate_log(simulated_log)
                            composite = audit.get("composite_score")
                            if composite is None:
                                # If output was empty dict or had no gradable items, fail with 0.0 rather than false 100.0
                                if not dict_to_eval or not any(dict_to_eval.get(k) for k in ("vocabulary", "expressions", "grammar_patterns", "questions", "concepts", "branches")):
                                    composite = 0.0
                                else:
                                    composite = 100.0
                            
                            flags = audit.get("flags", [])
                            # Note: evaluate_log runs on dict_to_eval AFTER deterministic auto-healing,
                            # so any surviving flags (including unhealed missing '____' slots) are actionable.
                            actionable_flags = list(flags)

                            from ..evaluator import FATAL_QA_FLAGS
                            has_fatal_flags = any(
                                any(fatal in f for fatal in FATAL_QA_FLAGS)
                                for f in actionable_flags
                            )

                            if composite < 80.0 or has_fatal_flags:
                                scores = {k: v for k, v in audit.get("scores", {}).items() if v is not None}
                                lowest_dim = min(scores.keys(), key=lambda k: scores[k]) if scores else "pedagogical_quality"
                                
                                # Build field-scoped QA repair tickets via qa_ticket.py
                                source_header = "### PASSAGE (WITH NUMBERED SENTENCES) ###"
                                if user_prompt:
                                    if "### PASSAGE (WITH NUMBERED SENTENCES) ###" in user_prompt:
                                        source_header = "### PASSAGE (WITH NUMBERED SENTENCES) ###"
                                    elif "### SOURCE TEXT ###" in user_prompt:
                                        source_header = "### SOURCE TEXT ###"
                                    elif "### PASSAGE" in user_prompt:
                                        source_header = "### PASSAGE ###"
                                    elif "PASSAGE:" in user_prompt:
                                        source_header = "PASSAGE:"
                                    elif "CONTENT:" in user_prompt:
                                        source_header = "CONTENT:"

                                from ..qa_ticket import (
                                    build_qa_tickets,
                                    detect_array_key,
                                    format_qa_ticket_block,
                                    score_ticket,
                                )

                                qa_tickets = build_qa_tickets(
                                    actionable_flags,
                                    task_name=t_name,
                                    parsed=dict_to_eval,
                                    source_header=source_header,
                                )
                                if not qa_tickets:
                                    qa_tickets = [score_ticket(
                                        composite,
                                        lowest_dim,
                                        array_key=detect_array_key(dict_to_eval),
                                        source_header=source_header,
                                    )]
                                critique_prompt = format_qa_ticket_block(qa_tickets)

                                import logging
                                logging.getLogger("librarian").info(
                                    f"QA score {composite}/100 (<80% or fatal flags) for {t_name}. Retrying ({retry_count + 1}/{max_qa_retries}) via multi-turn self-correction..."
                                )
                                # Log retry attempt with truthful failure category
                                retry_fail_cat = "QA_FATAL_FLAG" if has_fatal_flags else "QA_LOW_SCORE"
                                from ..logger import log_task
                                log_task(
                                    f"{t_name}_{mode_str}",
                                    system_prompt,
                                    user_prompt,
                                    final_json,
                                    schema=schema_dict,
                                    status="RETRYING",
                                    failure_category=retry_fail_cat,
                                    mode=mode_str,
                                    api_constraint=schema_for_api if mode_str == "STRICT_SCHEMA" else ("json" if force_json_mode else None),
                                    duration=call_duration,
                                    start_time=start_time,
                                    end_time=end_time,
                                    model=self.model,
                                    context_prompt=eval_user_prompt,
                                )

                                # Multi-turn self-correction:
                                # Strip only trailing prior QA critique and previous assistant output to avoid nesting review feedback,
                                # while preserving all preceding conversation context (e.g. system prompts, surgical instructions, etc.)
                                base_messages = list(messages)
                                while len(base_messages) >= 2 and base_messages[-1].get("role") == "user" and any(k in base_messages[-1].get("content", "") for k in ("### 🚨 QUALITY AUDIT", "### 🚨 [QUALITY AUDIT")) and base_messages[-2].get("role") == "assistant":
                                    base_messages = base_messages[:-2]

                                retry_messages = list(base_messages)
                                retry_messages.append({"role": "assistant", "content": final_json})
                                retry_messages.append({"role": "user", "content": critique_prompt})
                                
                                return self.chat(
                                    retry_messages,
                                    model=self.model,
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

                # 6.5. ATTACH FINAL QA AUDIT METADATA TO RESULT OBJECT
                final_audit = None
                try:
                    from ..evaluator import LogEvaluator
                    dict_for_final_eval = data if isinstance(data, dict) else (dataclasses.asdict(result_obj) if dataclasses.is_dataclass(result_obj) else None)
                    if dict_for_final_eval:
                        final_simulated_log = {
                            "log_name": f"{t_name}.log",
                            "task": t_name,
                            "model": self.model or "unknown",
                            "system_prompt": system_prompt,
                            "user_prompt": eval_user_prompt,
                            "context_prompt": f"{system_prompt}\n{eval_user_prompt}".strip(),
                            "raw_response": final_json,
                            "parsed_json": dict_for_final_eval,
                        }
                        final_audit = LogEvaluator.evaluate_log(final_simulated_log)
                        if isinstance(result_obj, dict):
                            result_obj["_qa_audit"] = final_audit
                        else:
                            try:
                                setattr(result_obj, "_qa_audit", final_audit)
                            except Exception:
                                pass
                except Exception as final_eval_err:
                    import logging
                    logging.getLogger("librarian").debug(f"Final QA audit attachment skipped: {final_eval_err}")

                # 6.6. DERIVE TRUTHFUL OUTCOME STATUS FROM QA AUDIT & LOG
                from ..evaluator import FATAL_QA_FLAGS
                task_status = "SUCCESS"
                task_fail_cat = None
                if final_audit and isinstance(final_audit, dict):
                    comp_score = final_audit.get("composite_score")
                    audit_flags = final_audit.get("flags", [])
                    has_audit_fatal = any(
                        any(fatal in f for fatal in FATAL_QA_FLAGS)
                        for f in audit_flags
                    )
                    if (comp_score is not None and comp_score < 80.0) or has_audit_fatal:
                        task_status = "REVIEW_NEEDED"
                        task_fail_cat = "QA_FATAL_FLAG" if has_audit_fatal else "QA_LOW_SCORE"

                if not skip_log:
                    from ..logger import log_task
                    self.last_log_path = log_task(
                        f"{t_name}_{mode_str}",
                        system_prompt,
                        user_prompt,
                        final_json,
                        schema=schema_dict,
                        status=task_status,
                        failure_category=task_fail_cat,
                        mode=mode_str,
                        api_constraint=schema_for_api if mode_str == "STRICT_SCHEMA" else ("json" if force_json_mode else None),
                        duration=call_duration,
                        start_time=start_time,
                        end_time=end_time,
                        model=self.model,
                        context_prompt=eval_user_prompt,
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
                from ..logger import log_task
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
                    model=self.model,
                )
                raise LLMError(f"Failed to generate structured data matching schema: {e}")
        else:
            if not stream and not skip_log:
                from ..logger import log_task
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
                    model=self.model,
                )
            return content

