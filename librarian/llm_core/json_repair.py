# -*- coding: utf-8 -*-
"""JSON repair, string lookahead, and sentence pointer alignment mixin."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("librarian")


def _prune_to_schema(schema: Any, data: Any) -> Any:
    """Recursively prune ``data`` to the structure declared by a JSON Schema dict.

    ``validate_and_map`` only accepts dataclass types, so dict-based schemas
    (e.g. the ``reading_quiz`` / ``extract_*`` task schemas) bypass it in
    :func:`LLM.chat`. Left unchecked, LLM-hallucinated extra fields (such as the
    ``quoted_sentence`` leak in reading quizzes) flow straight into rendered
    handouts. This helper enforces ``additionalProperties: false`` by dropping any
    key not declared in ``schema["properties"]`` and recursing into nested objects
    and arrays. Keys that ARE declared by the schema are always preserved, so
    legitimate fields (e.g. ``quoted_sentence`` in the vocabulary/expressions
    extraction schemas) are left intact.
    """
    if not isinstance(schema, dict):
        return data

    # Arrays: prune each element against the ``items`` schema.
    if isinstance(data, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            return [_prune_to_schema(items_schema, item) for item in data]
        return data

    if not isinstance(data, dict):
        return data

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return data

    additional = schema.get("additionalProperties", True)
    pruned = {}
    for key, prop_schema in properties.items():
        if key in data:
            pruned[key] = _prune_to_schema(prop_schema, data[key])
    if additional is not False:
        for key, value in data.items():
            if key not in properties:
                pruned[key] = value
    return pruned



def _is_real_string_closer(s: str, i: int) -> bool:
    """Decide whether s[i] (a '"' seen inside a string) genuinely closes it.

    Structural lookahead:
      - followed by  } ] :  or end-of-input           -> real closer
      - followed by  ,  then  { } ] [  (or EOF)        -> real closer
      - followed by  ,  then a string that is followed by ':' (a real key)
                                                          -> real closer
      - followed by a bare word / anything else         -> stray inner quote
    """
    n = len(s)
    j = i + 1
    while j < n and s[j].isspace():
        j += 1
    if j >= n:
        return True                       # string runs to end of input
    c = s[j]
    if c in '}]' or c == ':':
        return True
    if c == ',':
        k = j + 1
        while k < n and s[k].isspace():
            k += 1
        if k >= n:
            return True                   # dangling trailing comma (fixed later)
        if s[k] in '{}[]':
            return True
        if s[k] == '"':
            # Next token is a string. In valid JSON, that string is either:
            #   (a) an object key  -> followed by ':'
            #   (b) the next element in a string array -> followed by ',' or ']'
            m = k + 1
            while m < n and s[m] != '"':
                if s[m] == '\\':
                    m += 1
                m += 1
            p = m + 1
            while p < n and s[p].isspace():
                p += 1
            return p < n and s[p] in ':,]'
        return False                      # comma followed by bare word -> inner quote
    return False                          # followed by bare word -> inner quote



class JsonRepairMixin:
    """Mixin providing robust JSON repair, sentence alignment, and pointer hydration."""

    def _heal_json(self, json_str):
        """Attempts to fix common LLM structural errors with minimal intrusion."""
        if not json_str:
            return "{}"

        # 0. Strip <think>...</think> reasoning blocks if present in text output
        json_str = re.sub(r"<think>.*?</think>", "", json_str, flags=re.DOTALL).strip()

        # 0.1. Fix incomplete or invalid unicode escapes (e.g. truncated "\\u1" or "\\u123" at end of generation)
        json_str = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', json_str)

        # 0.5. Strip out-of-band parenthesized comments leaked OUTSIDE JSON strings
        # (small models sometimes emit e.g. `"value", (Note: 'x' is ...),`).
        # Safe: valid JSON never contains '(' outside of string values, so any such
        # group is junk; quotes inside it no longer threaten the parser once removed.
        if '(' in json_str:
            buf = []
            i, n, in_string, escaped = 0, len(json_str), False, False
            while i < n:
                ch = json_str[i]
                if in_string:
                    buf.append(ch)
                    if escaped:
                        escaped = False
                    elif ch == '\\':
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    i += 1
                    continue
                if ch == '"':
                    in_string = True
                    buf.append(ch)
                    i += 1
                    continue
                if ch == '(':
                    depth = 1
                    j = i + 1
                    while j < n and depth:
                        if json_str[j] == '(':
                            depth += 1
                        elif json_str[j] == ')':
                            depth -= 1
                        j += 1
                    if depth == 0:
                        # Remove the group; if both sides already carry commas, drop the left one
                        left = next((c for c in reversed(buf) if not c.isspace()), None)
                        right = json_str[j:j + 8].lstrip()[:1]
                        if left == ',' and right == ',':
                            while buf and buf[-1].isspace():
                                buf.pop()
                            buf.pop()  # left comma
                        i = j
                        continue
                buf.append(ch)
                i += 1
            json_str = re.sub(r",\s*,", ",", "".join(buf))

        # 0.6. Escape unescaped double quotes INSIDE string values.
        # Small models often cram several quoted source phrases into one value,
        # e.g.  "quoted_sentence": "ships burned" and "vessels went up in flames"
        # which is invalid JSON.  A quote inside a string is a genuine closer only
        # if what follows (past whitespace) is a JSON structural token: one of
        # } ] : , end-of-input, or a comma whose next token is structural (a new
        # key string must itself be followed by ':').  Anything else is a stray
        # content quote and gets escaped so the string keeps its true extent.
        # (Runs AFTER comment-stripping so a comma followed by junk-parens is
        # not misread as "stray inner quote".)
        out = []
        i, n = 0, len(json_str)
        in_string = False
        escaped = False
        while i < n:
            ch = json_str[i]
            if not in_string:
                if ch == '"':
                    in_string = True
                out.append(ch)
                i += 1
                continue
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == '\\':
                out.append(ch)
                escaped = True
            elif ch == '"':
                if _is_real_string_closer(json_str, i):
                    in_string = False
                    out.append(ch)
                else:
                    out.append('\\"')
            else:
                out.append(ch)
            i += 1
        json_str = "".join(out)

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

        # 2.3. Fix small-model attention drift: accidental dictionary keys inside array of objects
        # e.g., in a questions array, after "explanation": "..."\n  "target_word_key": { ... }
        # Instead of closing the prior item with '}' and starting '{', the model wrote ',\n "key": {'
        # NOTE: Do NOT match valid schema keys that legitimately hold objects (e.g. "cured_question": {)
        json_str = re.sub(r',\s*\n(\s*)(?!"(?:cured_question)"\s*:)"[^"]+":\s*\{', r'\n\1},\n\1{', json_str)

        # 2.3b. Fix premature closing brace before object property keys
        # e.g., after "lesson_hook": "..." the model wrote '},\n  "concepts": [' instead of ',\n  "concepts": ['
        json_str = re.sub(r'\}\s*,\s*(?:\r?\n\s*)*("[A-Za-z0-9_]+"\s*:)', r',\n  \1', json_str)

        # 2.4. Fix array wrongly closed with object braces
        # e.g., starts with "questions": [ but ends with }\n  }\n} instead of }\n  ]\n}
        if re.search(r'"(?:questions|vocabulary|expressions|grammar_patterns)":\s*\[', json_str):
            json_str = re.sub(r'\}\s*\}\s*\}\s*$', '}\n  ]\n}', json_str.strip())

        # 3. Fix unescaped newlines within values
        # This is a bit risky but common: "value": "line1\nline2"
        # We only escape newlines that are NOT followed by a potential key or object close
        # json_str = re.sub(r'\n(?!\s*["\}\]])', r'\\n', json_str)

        # 4. Ensure balanced braces and brackets (Stack-aware delimiter tracker)
        stack = []
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
            elif ch in '{[':
                stack.append(ch)
            elif ch in '}]':
                if stack:
                    if (stack[-1] == '{' and ch == '}') or (stack[-1] == '[' and ch == ']'):
                        stack.pop()

        # Close any dangling open string
        if in_string:
            json_str += '"'

        # Close open delimiters in exact reverse order of nesting
        for opener in reversed(stack):
            if opener == '{':
                json_str += '}'
            elif opener == '[':
                json_str += ']'

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

        from ..evaluator import _extract_source_content
        source_content = _extract_source_content(user_prompt)
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


    def _hydrate_sentence_pointers(self, data, user_prompt):
        """
        Deterministically hydrates model-returned sentence index tags (e.g. '[S-16]', 'S-126')
        in 'quoted_sentence' or 'quote' into full, authentic source sentences from the indexed pool.
        Saves ~70% completion tokens while guaranteeing 100% verbatim accuracy.
        """
        if not isinstance(data, dict) or not user_prompt:
            return

        from ..evaluator import _extract_source_content
        from ..linguistics import LinguisticEngine

        source = _extract_source_content(user_prompt)
        if not source:
            return

        _, sentence_pool = LinguisticEngine.tokenize_and_index_sentences(source)
        if not sentence_pool:
            return

        import logging
        logger = logging.getLogger("librarian")

        for array_key in ("vocabulary", "expressions", "grammar_patterns"):
            items = data.get(array_key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                quote_field = "quoted_sentence" if "quoted_sentence" in item else ("quote" if "quote" in item else None)
                if not quote_field or not item.get(quote_field):
                    continue
                raw_ref = str(item[quote_field]).strip()
                snapped = LinguisticEngine.snap_to_sentence_pool(raw_ref, sentence_pool)
                if snapped and snapped != raw_ref:
                    logger.info(f"⚡ Hydrated sentence pointer '{raw_ref}' -> full authentic sentence ({len(snapped)} chars)")
                    item[quote_field] = snapped

