import re
import json
import unicodedata
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, get_args

from .config import config
from .schemas import PARTS_OF_SPEECH, EXPRESSION_TYPES

# --- Dimension weights (max points per dimension). Single source of truth. ---
W_SCHEMA = 25.0
W_VERBATIM = 30.0
W_PEDAGOGY = 25.0
W_UNIQUENESS = 20.0

# Valid Part of Speech enum set derived directly from schemas.py
VALID_POS_SET = set(get_args(PARTS_OF_SPEECH))
VALID_EXPR_POS_SET = set(get_args(EXPRESSION_TYPES))

# Top-level arrays that may hold gradeable items, in priority order.
ITEM_KEYS = ["vocabulary", "expressions", "grammar_patterns", "questions", "concepts", "branches"]

# Module-level stop-slots set for verbatim checking (avoids re-allocation on every item)
STOP_SLOTS = frozenset({
    "something", "somebody", "ones", "someone", "sb", "sth", "entity",
    "field", "area", "role", "object", "domain", "type", "situation", "goal"
})

# Module-level irregular verbs mapping (avoids re-allocation on every item)
COMMON_IRREGULARS = {
    'lead': ('led',), 'led': ('lead',),
    'take': ('took', 'taken'), 'took': ('take',), 'taken': ('take',),
    'bring': ('brought',), 'brought': ('bring',),
    'lay': ('laid',), 'laid': ('lay',),
    'make': ('made',), 'made': ('make',),
    'give': ('gave', 'given'), 'gave': ('give',), 'given': ('give',),
    'come': ('came',), 'came': ('come',),
    'go': ('went', 'gone'), 'went': ('go',), 'gone': ('go',),
    'keep': ('kept',), 'kept': ('keep',),
    'hold': ('held',), 'held': ('hold',),
    'find': ('found',), 'found': ('find',),
    'break': ('broke', 'broken'), 'broke': ('break',), 'broken': ('break',),
    'choose': ('chose', 'chosen'), 'chose': ('choose',), 'chosen': ('choose',),
    'run': ('ran',), 'ran': ('run',),
    'see': ('saw', 'seen'), 'saw': ('see',), 'seen': ('see',),
    'speak': ('spoke', 'spoken'), 'spoke': ('speak',), 'spoken': ('speak',),
    'write': ('wrote', 'written'), 'wrote': ('write',), 'written': ('write',),
    'build': ('built',), 'built': ('build',),
    'lose': ('lost',), 'lost': ('lose',),
    'pay': ('paid',), 'paid': ('pay',),
    'say': ('said',), 'said': ('say',),
    'send': ('sent',), 'sent': ('send',),
    'spend': ('spent',), 'spent': ('spend',),
    'stand': ('stood',), 'stood': ('stand',),
    'tell': ('told',), 'told': ('tell',),
    'think': ('thought',), 'thought': ('think',),
    'understand': ('understood',), 'understood': ('understand',),
    'win': ('won',), 'won': ('win',),
    'catch': ('caught',), 'caught': ('catch',),
    'draw': ('drew', 'drawn'), 'drew': ('draw',), 'drawn': ('draw',),
    'grow': ('grew', 'grown'), 'grew': ('grow',), 'grown': ('grow',),
    'hear': ('heard',), 'heard': ('hear',),
    'hide': ('hid', 'hidden'), 'hid': ('hide',), 'hidden': ('hide',),
    'know': ('knew', 'known'), 'knew': ('know',), 'known': ('know',),
    'leave': ('left',), 'left': ('leave',),
    'meet': ('met',), 'met': ('meet',),
    'read': ('read',),
    'rise': ('rose', 'risen'), 'rose': ('rise',), 'risen': ('rise',),
    'wear': ('wore', 'worn'), 'wore': ('wear',), 'worn': ('wear',),
    'drive': ('drove', 'driven'), 'drove': ('drive',), 'driven': ('drive',),
    'fall': ('fell', 'fallen'), 'fell': ('fall',), 'fallen': ('fall',),
    'feel': ('felt',), 'felt': ('feel',),
}


def _safe_str(val: Any, default: str = "") -> str:
    """Safely converts a value to string, mapping None to default rather than 'None'."""
    if val is None:
        return default
    return str(val).strip()


def _normalize_text(value: Any) -> str:
    """Normalize unicode, strip markdown formatting, brackets, ellipses, quotes, and whitespace."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).lower()
    # Remove markdown bold/italics/code/strikethrough markers
    text = re.sub(r"[\*\_`~]+", " ", text)
    # Remove ellipses and dots
    text = re.sub(r"(\.{2,}|…)", " ", text)
    # Remove quotes and apostrophes directly to keep words intact
    text = re.sub(r"[“”‘’\"'«»]+", "", text)
    # Remove structural brackets/parentheses with whitespace separation
    text = re.sub(r"[\[\]\(\)\{\}\<\>]+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_core(text: str) -> str:
    """Strip all punctuation and non-alphanumeric chars (retaining spaces)."""
    norm = _normalize_text(text)
    return re.sub(r"[^\w\s]", "", norm).strip()


def _ngram_coverage(quote_clean: str, source_clean: str, n: int = 3) -> float:
    """Calculates n-gram overlap between cleaned quote and source text."""
    q_words = quote_clean.split()
    s_words = source_clean.split()
    if not q_words or not s_words:
        return 0.0
    if len(q_words) < n:
        if quote_clean in source_clean:
            return 1.0
        s_set = set(s_words)
        return sum(1 for w in q_words if w in s_set) / len(q_words)
    q_ngrams = set(tuple(q_words[i:i + n]) for i in range(len(q_words) - n + 1))
    s_ngrams = set(tuple(s_words[i:i + n]) for i in range(len(s_words) - n + 1))
    if not q_ngrams:
        return 0.0
    return len(q_ngrams.intersection(s_ngrams)) / len(q_ngrams)


def _is_hallucinated_quote(quote: str) -> bool:
    """Detect if model explicitly notes quote is inferred or missing from text."""
    q_lower = quote.lower()
    indicators = [
        "not present in text",
        "not in text",
        "not found in text",
        "not in source",
        "inferred from",
        "implied by",
        "constructed from",
        "not explicitly mentioned",
    ]
    return any(ind in q_lower for ind in indicators)


def _extract_source_content(user_prompt: str) -> str:
    """Extracts isolated source text from user prompt (under CONTENT:, # Source Material, etc.)
    Returns empty string if no authentic content block is found to prevent instruction text from
    being falsely matched as verbatim source.
    """
    if not user_prompt:
        return ""
    # Standard CONTENT: marker
    match = re.search(r"CONTENT:\s*\n(.*)", user_prompt, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Secondary headings if CONTENT: was omitted
    match_sec = re.search(r"(?:#+\s*(?:Source Material|Input Content|Text Context|Transcript))\s*\n(.*)", user_prompt, re.DOTALL | re.IGNORECASE)
    if match_sec:
        return match_sec.group(1).strip()
    return ""


def _extract_wordlist(user_prompt: str) -> List[str]:
    """Extract the supplied vocabulary headwords from the content block.

    Prefers Obsidian heading entries of the form `## [[word]]`; falls back to any
    `## heading` so that differently-formatted word lists still work.
    """
    source = _extract_source_content(user_prompt)
    if not source:
        return []
    # Match ## [[...]] where the inner content might contain nested slot brackets like [[pay [somebody] attention]]
    words = re.findall(r"^##\s*\[\[(.*?)(?:\]\]\s*$)", source, re.MULTILINE)
    if not words:
        words = re.findall(r"^##\s*\[\[([^\]]+)\]\]", source, re.MULTILINE)
    if not words:
        words = re.findall(r"^##\s+(.+?)\s*$", source, re.MULTILINE)
    return [w.strip() for w in words if w and w.strip()]


def _wordlist_matches(clean_target: str, wordlist: set) -> bool:
    """True if the (already cleaned) target word matches any headword in the list.

    Handles multi-word units and inflections via whole-token / shared-4-char-stem
    matching, while avoiding naive substring false positives (e.g. 'win' vs 'window').
    """
    if clean_target in wordlist:
        return True
    target_tokens = [t for t in clean_target.split() if t]
    if not target_tokens:
        return False
    for wl in wordlist:
        wl_tokens = [t for t in wl.split() if t]
        if all(
            any(
                t == w
                or (len(t) >= 4 and len(w) >= 4 and t[:4] == w[:4])
                for w in wl_tokens
            )
            for t in target_tokens
        ):
            return True
    return False


def _detect_task_type(task: str, parsed: Any) -> str:
    """Detect the logical task type from JSON structure first, task name as fallback."""
    t = (task or "").lower()
    # Turn 1 prose drafting is an intermediate natural language stage, not a final JSON output
    if "turn1_prose" in t or "prose_draft" in t:
        return "intermediate_prose"
    if "expert_audit" in t or "quality_audit" in t:
        return "expert_audit"
    if isinstance(parsed, dict):
        if "pass_audit" in parsed and "blind_solve_accuracy" in parsed:
            return "expert_audit"
        if isinstance(parsed.get("questions"), list):
            return "quiz"
        if isinstance(parsed.get("grammar_patterns"), list):
            return "grammar"
        if isinstance(parsed.get("expressions"), list):
            return "expressions"
        if isinstance(parsed.get("vocabulary"), list):
            return "vocabulary"
        if isinstance(parsed.get("concepts"), list):
            return "summary"
        if isinstance(parsed.get("branches"), list):
            return "mindmap"
    if "extract_grammar" in t:
        return "grammar"
    if "extract_expressions" in t:
        return "expressions"
    if "quiz" in t:
        return "quiz"
    if "extract_summary" in t or "summary" in t:
        return "summary"
    if "extract_mindmap" in t or "mindmap" in t:
        return "mindmap"
    if "vocabulary" in t:
        return "vocabulary"
    return "unknown"


def _extract_items(parsed: Any, task_type: str) -> List[Dict[str, Any]]:
    """Return the list of dict-items for the given task type (structure-aware)."""
    if not isinstance(parsed, dict):
        return []
    key_by_type = {
        "quiz": "questions",
        "grammar": "grammar_patterns",
        "expressions": "expressions",
        "vocabulary": "vocabulary",
        "summary": "concepts",
        "mindmap": "branches",
    }
    preferred = key_by_type.get(task_type)
    candidates = ([preferred] if preferred else []) + [k for k in ITEM_KEYS if k != preferred]
    # First pass: try to find a candidate that contains valid dict items
    for key in candidates:
        value = parsed.get(key)
        if isinstance(value, list) and value:
            valid_items = [it for it in value if isinstance(it, dict)]
            if valid_items:
                return valid_items
    return []


def _extract_json(text: Any) -> Any:
    """Best-effort JSON extraction: direct, fenced ```json, or first balanced brace."""
    if not text or not isinstance(text, str):
        return None
    # Strip <think>...</think> reasoning blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                continue
    return None


def _score_schema(parsed: Any, raw_response: str = "") -> Tuple[Optional[float], List[str]]:
    """Dimension 1 (0–25). Always applicable."""
    if isinstance(parsed, dict):
        if len(parsed) == 0:
            return 0.0, ["❌ Valid JSON but empty object"]
        
        flags = []
        deduction = 0.0
        if raw_response and isinstance(raw_response, str):
            trimmed = raw_response.strip()
            # If closing brace was missing or raw string didn't end properly
            if trimmed.count("{") > trimmed.count("}") or trimmed.count("[") > trimmed.count("]"):
                deduction += 5.0
                flags.append("⚠️ Structural repair: unclosed braces/brackets in raw output")
        
        score = max(0.0, W_SCHEMA - deduction)
        return score, flags
    return 0.0, ["❌ Invalid or missing JSON output"]


def _score_verbatim(items: List[Dict[str, Any]], task_type: str, user_prompt: str, context_prompt: str = "") -> Tuple[Optional[float], List[str]]:
    """Dimension 2 (0–30). Applies to extraction and quiz tasks; returns (None, []) otherwise.

    For extraction tasks this verifies quoted sentences are faithful to the source.
    For quiz tasks this verifies every target_word comes from the supplied word list
    (anti-hallucination: the assessment may not test fabricated vocabulary).
    """
    if task_type not in ("vocabulary", "expressions", "grammar", "quiz"):
        return None, []  # N/A -> normalized out of composite score
    if not items:
        return 0.0, ["❌ No items to evaluate for source faithfulness"]

    # --- Quiz: every target_word must be present in the supplied word list ---
    if task_type == "quiz":
        flags: List[str] = []
        effective_prompt = user_prompt if user_prompt.strip() else (context_prompt or "")
        wordlist = {_clean_core(w) for w in _extract_wordlist(effective_prompt)}
        wordlist = {w for w in wordlist if w}
        checks = matches = 0
        for item in items:
            target = str(item.get("target_word") or "").strip()
            if not target:
                continue
            clean_target = _clean_core(target)
            if not clean_target:
                continue
            checks += 1
            if wordlist and _wordlist_matches(clean_target, wordlist):
                matches += 1
            elif not wordlist:
                # Packaging / conversion phase: check if target word exists in the drafted prompt text
                clean_prompt = _clean_core(effective_prompt)
                if clean_target in clean_prompt:
                    matches += 1
                else:
                    flags.append(f"⚠️ Target '{target}' not found in draft content")
            else:
                flags.append(f"❌ Target word '{target}' not found in supplied word list (possible hallucination)")
        if checks == 0:
            # No target_word items (e.g. reading/translation quizzes) -> nothing to verify -> N/A
            return None, flags
        return max(0.0, round((matches / checks) * W_VERBATIM, 1)), flags

    flags: List[str] = []
    source = _extract_source_content(user_prompt)
    core_src = _clean_core(source)
    checks = matches = 0
    
    for item in items:
        quote = item.get("quoted_sentence") or item.get("quote")
        word = str(item.get("word") or item.get("pattern_formula") or "").strip()
        if not (quote and isinstance(quote, str)):
            continue
            
        checks += 1
        
        # Check 1: Explicit hallucination acknowledgment
        if _is_hallucinated_quote(quote):
            flags.append(f"❌ Hallucinated quote (explicitly inferred/absent): '{quote[:50]}...'")
            continue

        # Check 2: Target word must be present in the quoted sentence (for vocabulary & expressions)
        if word and task_type in ("vocabulary", "expressions"):
            clean_word_no_slots = re.sub(r"\[.*?\]|\(.*?\)", " ", word)
            clean_word = _clean_core(clean_word_no_slots)
            clean_quote = _clean_core(quote)
            word_tokens = [w for w in clean_word.split() if w and w not in STOP_SLOTS]
            
            if word_tokens:
                # Match token, stem/inflection (e.g. degrade -> degradation, took -> take, pose -> poses), or irregular forms
                def _tok_in_quote(tok: str) -> bool:
                    if tok in clean_quote or (len(tok) >= 4 and tok[:4] in clean_quote):
                        return True
                    if tok in COMMON_IRREGULARS and any(ir in clean_quote for ir in COMMON_IRREGULARS[tok]):
                        return True
                    return False

                matched_count = sum(1 for w in word_tokens if _tok_in_quote(w))
                min_needed = max(1, len(word_tokens) // 2 + (1 if len(word_tokens) % 2 == 1 else 0))
                word_in_quote = (matched_count >= min_needed)
            else:
                word_in_quote = (clean_word in clean_quote)
                
            if not word_in_quote:
                flags.append(f"❌ Target word '{word}' does not appear in quoted sentence: '{quote[:40]}...'")
                # Severe deduction: directly penalize verbatim score rather than neutral skip
                matches = max(0, matches - 1)
                continue

        # Check 3: Cleaned quote in source or high n-gram coverage
        core_quote = _clean_core(quote)
        is_verbatim = False
        if core_quote:
            if core_quote in core_src or _ngram_coverage(core_quote, core_src, n=3) >= 0.85:
                is_verbatim = True
            elif "..." in quote or "…" in quote:
                # If model used ellipsis to omit middle parts of a long sentence, check each segment
                segments = [s.strip() for s in re.split(r'\.{3,}|…', quote) if s.strip()]
                meaningful_segs = [s for s in segments if len(_clean_core(s).split()) >= 2]
                if meaningful_segs and all(_clean_core(seg) in core_src or _ngram_coverage(_clean_core(seg), core_src, n=3) >= 0.85 for seg in meaningful_segs):
                    is_verbatim = True

        if is_verbatim:
            matches += 1
        else:
            flags.append(f"⚠️ Non-verbatim quote detected: '{quote[:40]}...'")
            
    if checks == 0:
        return W_VERBATIM, flags
    return max(0.0, round((matches / checks) * W_VERBATIM, 1)), flags



def _score_pedagogy(items: List[Dict[str, Any]], task_type: str, user_prompt: str = "") -> Tuple[Optional[float], List[str]]:
    """Dimension 3 (0–25). Evaluates pedagogical quality across extraction and assessment types."""
    if task_type not in ("vocabulary", "expressions", "grammar", "quiz", "summary", "mindmap", "expert_audit"):
        return None, []
    if not items:
        return 0.0, [f"⚠️ Output list for '{task_type}' is empty."]
        
    flags: List[str] = []
    checks = passes = 0

    # Extract study list headwords from prompt if evaluating a quiz
    prompt_headwords = set()
    if task_type == "quiz" and user_prompt:
        for m in re.finditer(r"## \[\[(.*?)\]\]", user_prompt):
            prompt_headwords.add(m.group(1).strip().lower())
        if not prompt_headwords:
            for m in re.finditer(r"-\s*Target:\s*([^\n\r]+)", user_prompt):
                prompt_headwords.add(m.group(1).strip().lower())

    for item in items:
        if task_type == "vocabulary":
            word = _safe_str(item.get("word"))
            pos = _safe_str(item.get("part_of_speech")).lower()
            definition = _safe_str(item.get("definition"))
            example = _safe_str(item.get("example_usage"))
            quote = _safe_str(item.get("quoted_sentence"))
            checks += 1
            
            # Allow words/phrases with slots, hyphens, brackets, parentheses, apostrophes
            valid_word = bool(re.search(r"^[A-Za-z\s\-'\[\]\(\)]+$", word) and len(word) >= 2)
            valid_pos = pos in VALID_POS_SET
            has_example = bool(example)
            is_distinct_example = bool(has_example and _clean_core(example) != _clean_core(quote))
            
            if valid_word and valid_pos and definition and is_distinct_example:
                passes += 1
            else:
                reasons = []
                if not valid_word: reasons.append(f"invalid headword '{word}'")
                if not valid_pos: reasons.append(f"invalid PoS '{pos}'")
                if not definition: reasons.append("missing definition")
                if not has_example:
                    reasons.append("missing example_usage")
                elif not is_distinct_example:
                    reasons.append("example_usage is an unoriginal duplicate of quoted_sentence")
                flags.append(f"⚠️ Vocabulary item '{word}' failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "expressions":
            word = _safe_str(item.get("word"))
            checks += 1
            is_multiword = bool(re.search(r"\[.+?\]|one's", word, re.IGNORECASE) or len(word.split()) > 1)
            is_trivial = word.lower() in ("talk", "listen", "turn", "watch", "sit down", "talk to", "listen to", "look at")
            if is_multiword and not is_trivial:
                passes += 1
            else:
                flags.append(f"⚠️ Expression lacks multi-word/slot form or is too basic: '{word}'")
        elif task_type == "grammar":
            pattern = _safe_str(item.get("pattern_formula"))
            audit = _safe_str(item.get("design_audit"))
            checks += 1
            if re.search(r"\[.+?\]", pattern) and audit:
                passes += 1
            else:
                flags.append("⚠️ Grammar pattern missing slot formula or design audit")
        elif task_type == "quiz":
            options = item.get("options", [])
            idx = item.get("correct_answer_index")
            explanation = _safe_str(item.get("explanation") or item.get("diagnostic_critique"))
            question = _safe_str(item.get("question") or item.get("translated_sentence") or item.get("source_sentence"))
            target = _safe_str(item.get("target_word") or item.get("target_keyword") or item.get("word")).strip().lower()
            is_translation = bool(item.get("translated_sentence") or item.get("source_sentence") or item.get("english_skeleton") or item.get("target_keyword") or item.get("idiomatic_translation"))
            skeleton = _safe_str(item.get("english_skeleton"))
            is_comparative_translation = bool(item.get("idiomatic_translation") and item.get("flawed_translation"))
            expected_opt_count = 2 if is_comparative_translation else 4
            checks += 1
            
            # 1. Option count and distinctness (case/space-insensitive uniqueness check)
            is_list_valid = isinstance(options, list) and len(options) == expected_opt_count
            has_no_duplicates = is_list_valid and len(set(str(o).strip().lower() for o in options)) == expected_opt_count
            
            # 2. Correct answer index validity
            valid_idx = isinstance(idx, int) and 0 <= idx < expected_opt_count
            
            # 3. Target word must be present in options and match the correct answer slot options[idx]
            target_in_options = True
            if target and is_list_valid and valid_idx:
                clean_target = _clean_core(target)
                if is_comparative_translation:
                    # In comparative translation, target_keyword must be in idiomatic_translation
                    idiomatic_ans = _safe_str(item.get("idiomatic_translation")).lower()
                    target_in_options = clean_target in _clean_core(idiomatic_ans) or (
                        len(clean_target) >= 4 and clean_target[:4] in _clean_core(idiomatic_ans)
                    )
                elif is_translation:
                    # In legacy translation quizzes, target_keyword can be in the options slot,
                    # or in the fixed english_skeleton / correct_english_answer!
                    full_ans = _safe_str(item.get("correct_english_answer")).lower()
                    skel_ans = _safe_str(item.get("english_skeleton")).lower()
                    opt_ans = _safe_str(options[idx]).lower()
                    target_in_options = (
                        clean_target in _clean_core(full_ans) or
                        clean_target in _clean_core(skel_ans) or
                        clean_target in _clean_core(opt_ans) or
                        (len(clean_target) >= 4 and any(
                            clean_target[:4] in _clean_core(text)
                            for text in [full_ans, skel_ans, opt_ans]
                        ))
                    )
                else:
                    selected_opt = _clean_core(str(options[idx]))
                    # Match exact or valid inflections (e.g. assert -> asserted, marathon -> marathons)
                    target_in_options = (clean_target == selected_opt) or (
                        len(clean_target) >= 4 and len(selected_opt) >= 4 and (
                            selected_opt.startswith(clean_target[:4]) or clean_target.startswith(selected_opt[:4]) or (clean_target in selected_opt)
                        )
                    )

            # 4. Blank verification for fill-in-the-blank questions
            has_blank_when_expected = True
            if is_comparative_translation:
                # Comparative translation does not require blanks
                has_blank_when_expected = True
            elif is_translation:
                if skeleton:
                    has_blank_when_expected = bool(re.search(r"\[\s*_{2,}\s*\]|_{2,}", skeleton))
            elif target and question:
                # If target is specified in a standard quiz, it's a lexical/fill-in-the-blank item.
                # Must contain at least two consecutive underscores (____)
                has_blank_when_expected = bool(re.search(r"_{2,}", question))

            # 5. In-List Distractor Recycling check (Zero-Tolerance Level 1 Gate)
            recycled_in_distractors = []
            if prompt_headwords and is_list_valid and valid_idx:
                clean_target = _clean_core(target) if target else ""
                clean_hw_set = {_clean_core(hw) for hw in prompt_headwords if hw}

                for o_idx, opt in enumerate(options):
                    if o_idx == idx:
                        continue
                    opt_clean = str(opt).strip().lower()
                    opt_core = _clean_core(opt_clean)

                    # If this option is derived from the item's own target keyword (e.g. preserves, preserving, to preserve),
                    # it is a legitimate morphological/syntactic distractor trap, NOT recycling an external word!
                    if clean_target and (
                        opt_core == clean_target or 
                        clean_target in opt_clean or 
                        (len(clean_target) >= 4 and opt_clean.startswith(clean_target[:4]))
                    ):
                        continue

                    # Now check if it recycles another DIFFERENT headword from the study list
                    if opt_clean in prompt_headwords or (opt_core and opt_core in clean_hw_set and opt_core != clean_target):
                        recycled_in_distractors.append(opt_clean)
                    elif not is_translation:
                        # For vocabulary multiple-choice, check suffix-stripped stems
                        opt_stem = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', opt_clean)
                        for hw in prompt_headwords:
                            hw_clean = str(hw).strip().lower()
                            if clean_target and (hw_clean == clean_target or _clean_core(hw_clean) == clean_target):
                                continue
                            hw_stem = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', hw_clean)
                            if (len(hw_stem) >= 4 and len(opt_stem) >= 4 and hw_stem == opt_stem) or \
                               (len(hw_clean) >= 5 and opt_clean.startswith(hw_clean[:5])) or \
                               (len(opt_clean) >= 5 and hw_clean.startswith(opt_clean[:5])):
                                recycled_in_distractors.append(opt_clean)
                                break

            no_in_list_recycling = len(recycled_in_distractors) == 0

            if is_list_valid and has_no_duplicates and valid_idx and target_in_options and explanation and question and has_blank_when_expected and no_in_list_recycling:
                passes += 1
            else:
                reasons = []
                if not is_list_valid: reasons.append(f"options count != {expected_opt_count}")
                elif not has_no_duplicates: reasons.append("duplicate options detected")
                if not valid_idx: reasons.append("invalid answer index")
                if not target_in_options: reasons.append(f"target '{target}' not matching options[{idx}]")
                if not explanation: reasons.append("missing explanation")
                if not question: reasons.append("missing question text")
                if not has_blank_when_expected: reasons.append("missing fill-in-the-blank slot (____)")
                if not no_in_list_recycling:
                    reasons.append(f"recycles study list headwords in distractors {recycled_in_distractors}")
                    flags.append(f"❌ Quiz item '{target or question[:25]}' in-list distractor recycling: {recycled_in_distractors}")
                else:
                    flags.append(f"⚠️ Quiz question failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "summary":
            name = _safe_str(item.get("concept_name"))
            sig = _safe_str(item.get("educational_significance"))
            details = item.get("key_details", [])
            checks += 1
            if name and sig and isinstance(details, list) and len(details) > 0:
                passes += 1
            else:
                flags.append(f"⚠️ Concept '{name}' missing educational significance or key details")
        elif task_type == "mindmap":
            name = _safe_str(item.get("branch_name"))
            checks += 1
            if name:
                passes += 1
            else:
                flags.append("⚠️ MindMap branch missing branch name")
        elif task_type == "expert_audit":
            checks += 1
            distractors = item.get("distractors", [])
            bs_idx = item.get("blind_solved_index")
            feedback = _safe_str(item.get("diagnostic_feedback"))
            score = item.get("pedagogical_score", 100)
            single_valid = item.get("single_fit_valid", True)
            is_valid_dist = isinstance(distractors, list) and len(distractors) in (1, 2, 3, 4)
            is_valid_idx = isinstance(bs_idx, int) and 0 <= bs_idx <= 3
            # Feedback is required only if the item has defects (score < 90 or invalid single fit);
            # for flawless items (score >= 90), empty feedback is valid and expected.
            feedback_ok = bool(feedback) if (score < 90 or not single_valid) else True
            triage = item.get("triage_action", "PASS")
            cured_q = item.get("cured_question")
            is_valid_triage = triage in ("PASS", "REPAIR", "REWRITE")
            cured_ok = bool(cured_q) if triage in ("REPAIR", "REWRITE") else True
            if is_valid_dist and is_valid_idx and feedback_ok and is_valid_triage and cured_ok:
                passes += 1
            else:
                reasons = []
                if not is_valid_dist: reasons.append(f"distractor count != 1 to 4 (got {len(distractors) if isinstance(distractors, list) else 0})")
                if not is_valid_idx: reasons.append("invalid blind solve index")
                if not feedback_ok: reasons.append("missing diagnostic feedback for defective item")
                if not is_valid_triage: reasons.append(f"invalid triage_action '{triage}'")
                if not cured_ok: reasons.append(f"missing cured_question for triage '{triage}'")
                flags.append(f"⚠️ Audit item failed validation: {', '.join(reasons)}")
                
    if checks == 0:
        return 0.0, [f"⚠️ Output list for '{task_type}' is empty."]
    
    # Check for copy-pasted/homogenized definitions in vocabulary
    if task_type in ("vocabulary", "expressions"):
        definitions = [_clean_core(item.get("definition", "")) for item in items if item.get("definition")]
        if len(definitions) > 1:
            dup_defs = len(definitions) - len(set(definitions))
            if dup_defs > 0:
                flags.append(f"❌ Found {dup_defs} duplicate or copy-pasted definition(s) across different terms")
                passes = max(0, passes - dup_defs)

    return max(0.0, round((passes / checks) * W_PEDAGOGY, 1)), flags


def _score_uniqueness(items: List[Dict[str, Any]], task_type: str, user_prompt: str = "") -> Tuple[Optional[float], List[str]]:
    """Dimension 4 (0–20). Deduplicate by the task's primary identifying field."""
    if not items:
        return 0.0, ["❌ No items to evaluate for uniqueness"]
    key_by_type = {
        "vocabulary": ("word",),
        "expressions": ("word",),
        "grammar": ("pattern_formula", "quote"),
        "quiz": ("target_word", "question", "translated_sentence", "correct_english_answer"),
        "summary": ("concept_name",),
        "mindmap": ("branch_name",),
        "expert_audit": ("item_index",),
    }
    keys = key_by_type.get(task_type, ("word", "quote", "question"))
    headwords = []
    for item in items:
        for key in keys:
            value = item.get(key)
            if value:
                headwords.append(str(value).strip().lower())
                break
    if not headwords:
        return 0.0, ["⚠️ Missing identifying keys for uniqueness check"]

    unique_flags = []
    unique = len(set(headwords))
    earned = W_UNIQUENESS
    if unique < len(headwords):
        dup = len(headwords) - unique
        # Stricter deduction: duplicate items heavily reduce score
        ratio = unique / len(headwords)
        earned = 0.0 if ratio < 0.8 else round(ratio * W_UNIQUENESS, 1)
        unique_flags.append(f"❌ Found {dup} duplicate item(s)")

    # For quizzes: penalize identical distractors recycled repeatedly across different questions
    if task_type == "quiz":
        all_distractors = []
        for item in items:
            opts = item.get("options", [])
            idx = item.get("correct_answer_index")
            if isinstance(opts, list) and isinstance(idx, int):
                for o_i, o in enumerate(opts):
                    if o_i != idx:
                        all_distractors.append(str(o).strip().lower())
        if all_distractors:
            from collections import Counter
            counts = Counter(all_distractors)
            repeated = {w: c for w, c in counts.items() if c >= 3 and len(w) >= 3}
            if repeated:
                rep_details = ", ".join(f"'{w}' ({c}x)" for w, c in list(repeated.items())[:3])
                unique_flags.append(f"❌ Distractors recycled repeatedly across quiz: {rep_details}")
    return earned, unique_flags


def prune_hallucinated_items(parsed_data: Any, user_prompt: str, task_type: str = "") -> Tuple[Any, List[str]]:
    """Deterministic Level 1 Hallucination Pruning Gate.
    
    If the model extracts vocabulary/expressions where the headword neither appears
    in its quoted sentence nor in the source text, it is a pure hallucination.
    Rather than blindly looping retries that ask the model to fix nonexistent words,
    we surgically prune the hallucinated items deterministically.
    
    Returns:
        (pruned_parsed_data, list_of_pruned_flags)
    """
    if not isinstance(parsed_data, dict):
        return parsed_data, []
    
    if not task_type or task_type == "unknown":
        task_type = _detect_task_type("", parsed_data)
        
    if task_type not in ("vocabulary", "expressions"):
        return parsed_data, []

    items = _extract_items(parsed_data, task_type)
    if not items:
        return parsed_data, []

    source = _extract_source_content(user_prompt)
    core_src = _clean_core(source)
    if not core_src:
        return parsed_data, []

    key_by_type = {
        "vocabulary": "vocabulary",
        "expressions": "expressions",
    }
    array_key = key_by_type.get(task_type)
    if not array_key or array_key not in parsed_data or not isinstance(parsed_data[array_key], list):
        return parsed_data, []

    surviving_items = []
    pruned_flags = []

    for item in parsed_data[array_key]:
        if not isinstance(item, dict):
            surviving_items.append(item)
            continue

        word = str(item.get("word") or item.get("pattern_formula") or "").strip()
        quote = str(item.get("quoted_sentence") or item.get("quote") or "").strip()
        
        if not word:
            surviving_items.append(item)
            continue

        clean_word_no_slots = re.sub(r"\[.*?\]|\(.*?\)", " ", word)
        clean_word = _clean_core(clean_word_no_slots)
        clean_quote = _clean_core(quote)
        word_tokens = [w for w in clean_word.split() if w and w not in STOP_SLOTS]

        # Check A: Does the word appear in its claimed quote?
        def _tok_matches(tok: str, target_text: str) -> bool:
            if tok in target_text or (len(tok) >= 4 and tok[:4] in target_text):
                return True
            if tok in COMMON_IRREGULARS and any(ir in target_text for ir in COMMON_IRREGULARS[tok]):
                return True
            return False

        if word_tokens:
            min_needed = max(1, len(word_tokens) // 2 + (1 if len(word_tokens) % 2 == 1 else 0))
            word_in_quote = (sum(1 for w in word_tokens if _tok_matches(w, clean_quote)) >= min_needed)
            word_in_source = (sum(1 for w in word_tokens if _tok_matches(w, core_src)) >= min_needed)
        else:
            word_in_quote = (clean_word in clean_quote)
            word_in_source = (clean_word in core_src)

        # If it's absent from quote AND absent from entire source text -> pure fabrication
        if not word_in_quote and not word_in_source:
            pruned_flags.append(f"✂️ Pruned hallucinated item '{word}' (not present in quoted sentence or source text)")
            continue

        # If it's absent from its claimed quote but the quote itself is not in source
        if not word_in_quote:
            core_quote = _clean_core(quote)
            if core_quote and (core_quote not in core_src and _ngram_coverage(core_quote, core_src, n=3) < 0.85):
                pruned_flags.append(f"✂️ Pruned hallucinated item '{word}' (unmatched word in unverified quote)")
                continue

        surviving_items.append(item)

    if pruned_flags:
        parsed_data = dict(parsed_data)
        parsed_data[array_key] = surviving_items
    return parsed_data, pruned_flags


class LogEvaluator:
    """
    Evaluates LLM execution logs across 4 pedagogical quality dimensions:
    1. Schema & Structural Adherence (25%)
    2. Source Faithfulness & Verbatim Verification (30%)
    3. Pedagogical & Slot-Filling Quality (25%)
    4. Uniqueness & Deduplication (20%)

    Scoring convention:
    - Dimensions returning None (N/A) are excluded from composite calculation.
    - Composite Score is strictly normalized to 0–100 based on applicable dimensions.
    """

    # Cache & thread-lock for repeated audits (dashboard polls /api/hero-board).
    _AUDIT_LOCK = threading.Lock()
    _AUDIT_CACHE: Dict[str, Any] = {"key": None, "value": None}

    @classmethod
    def parse_log_file(cls, log_path: Path) -> Optional[Dict[str, Any]]:
        """Parses a single .log file and extracts metadata, status, prompt content, and JSON response."""
        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        task_match = re.search(r"=== TASK:\s*(.+?)\s*===", content)
        model_match = re.search(r"=== MODEL:\s*(.+?)\s*===", content)
        time_match = re.search(r"=== TIMESTAMP:\s*(.+?)\s*===", content)
        status_match = re.search(r"=== STATUS:\s*(.+?)\s*===", content)
        fail_cat_match = re.search(r"=== FAILURE_CATEGORY:\s*(.+?)\s*===", content)

        if not task_match or not model_match:
            return None

        task = task_match.group(1).strip()
        model = model_match.group(1).strip()
        timestamp = time_match.group(1).strip() if time_match else ""
        status = status_match.group(1).strip() if status_match else "SUCCESS"
        failure_category = fail_cat_match.group(1).strip() if fail_cat_match else None

        pre_cure_match = re.search(r"=== PRE_CURE_SCORE:\s*([\d\.]+)%\s*===", content)
        pre_cure_score = float(pre_cure_match.group(1)) if pre_cure_match else None

        post_cure_match = re.search(r"=== POST_CURE_SCORE:\s*([\d\.]+)%\s*===", content)
        post_cure_score = float(post_cure_match.group(1)) if post_cure_match else None

        # Extract SYSTEM PROMPT content block (fallback context)
        system_prompt = ""
        system_prompt_match = re.search(r"--- SYSTEM PROMPT ---\n(.*?)(?=\n--- USER PROMPT ---|\n--- RAW RESPONSE ---|\n===|\Z)", content, re.DOTALL)
        if system_prompt_match:
            system_prompt = system_prompt_match.group(1).strip()

        # Extract USER PROMPT content block
        user_prompt = ""
        user_prompt_match = re.search(r"--- USER PROMPT ---\n(.*?)(?=\n--- RAW RESPONSE ---|\n===|\Z)", content, re.DOTALL)
        if user_prompt_match:
            user_prompt = user_prompt_match.group(1).strip()

        # Extract RAW RESPONSE
        raw_response = ""
        response_match = re.search(r"--- RAW RESPONSE ---\n(.*?)(?=\n===|\Z)", content, re.DOTALL)
        if response_match:
            raw_response = response_match.group(1).strip()

        # Parse JSON (direct, fenced ```json, or first balanced brace)
        parsed_json = _extract_json(raw_response)

        return {
            "log_name": log_path.name,
            "task": task,
            "model": model,
            "status": status,
            "failure_category": failure_category,
            "timestamp": timestamp,
            "pre_cure_score": pre_cure_score,
            "post_cure_score": post_cure_score,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_json": parsed_json,
        }

    @classmethod
    def evaluate_log(cls, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the 4-dimension audit on a parsed log entry."""
        log_name = log_data.get("log_name", "")
        task = log_data.get("task", "")
        model = log_data.get("model", "")
        status = log_data.get("status", "SUCCESS")
        failure_category = log_data.get("failure_category")
        pre_cure_score = log_data.get("pre_cure_score")
        post_cure_score = log_data.get("post_cure_score")
        user_prompt = log_data.get("user_prompt", "")
        system_prompt = log_data.get("system_prompt", "")
        context_prompt = log_data.get("context_prompt", "") or system_prompt
        parsed = log_data.get("parsed_json")

        raw_response = log_data.get("raw_response", "")
        task_type = _detect_task_type(task, parsed)
        items = _extract_items(parsed, task_type)

        flags: List[str] = []
        schema_score, f1 = _score_schema(parsed, raw_response)
        verbatim_score, f2 = _score_verbatim(items, task_type, user_prompt, context_prompt=context_prompt)
        effective_prompt = user_prompt if user_prompt.strip() else context_prompt
        pedagogy_score, f3 = _score_pedagogy(items, task_type, user_prompt=effective_prompt)
        uniqueness_score, f4 = _score_uniqueness(items, task_type, user_prompt=effective_prompt)
        flags.extend(f1 + f2 + f3 + f4)

        # Dimension scores & weights map
        dim_results = [
            ("schema_adherence", schema_score, W_SCHEMA),
            ("verbatim_faithfulness", verbatim_score, W_VERBATIM),
            ("pedagogical_quality", pedagogy_score, W_PEDAGOGY),
            ("uniqueness", uniqueness_score, W_UNIQUENESS),
        ]

        applicable_weight = sum(w for _, score, w in dim_results if score is not None)
        earned_score = sum(score for _, score, _ in dim_results if score is not None)

        if applicable_weight > 0:
            composite_score = round((earned_score / applicable_weight) * 100.0, 1)
        else:
            composite_score = 0.0

        # Special handling for expert_audit task: align composite_score with deterministic code gate
        if task_type == "expert_audit" and isinstance(parsed, dict):
            try:
                from .expert_auditor import ExpertAuditor
                infer_type = "vocabulary"
                task_lower = (task or "").lower()
                prompt_lower = (user_prompt or "").lower()
                if "translation" in task_lower or "translation" in prompt_lower:
                    infer_type = "translation"
                elif "reading" in task_lower or "reading" in prompt_lower:
                    infer_type = "reading"
                
                divs = parsed.get("blind_solve_confident_divergences", 0)
                ExpertAuditor.reconcile_report_scores(parsed, quiz_type=infer_type, confident_divergences=divs)
                raw_audited_score = round(float(parsed.get("overall_quality_score", composite_score)), 1)
                
                if pre_cure_score is None:
                    pre_cure_score = raw_audited_score
                if post_cure_score is not None:
                    composite_score = post_cure_score
                else:
                    composite_score = raw_audited_score
                
                flaws = parsed.get("flawed_item_indices", [])
                if flaws:
                    flags.insert(0, f"❌ [FATAL FLAW] {len(flaws)} item(s) failed single-fit validation: #{', #'.join(str(i) for i in flaws)}")
                if not parsed.get("pass_audit", False):
                    base = parsed.get("base_quality_score", raw_audited_score)
                    if base > raw_audited_score:
                        flags.insert(1, f"⚠️ [L2_PENALTY] Capped from Base {base}% to {raw_audited_score}% due to fatal flaw(s)")
                if divs > 0:
                    flags.insert(0, f"❌ [CRITICAL FLAW] {divs} confident double-key divergence(s) detected in blind solve")
                if post_cure_score is not None and post_cure_score > raw_audited_score:
                    flags.insert(0, f"✅ [SURGICAL CURE] In-place healed from {pre_cure_score}% to Final {post_cure_score}%")
            except Exception as err:
                composite_score = round(float(parsed.get("overall_quality_score", composite_score)), 1)

        # Sort flags: critical errors (❌) first, then warnings (⚠️)
        flags.sort(key=lambda x: (0 if x.startswith("❌") else (1 if x.startswith("⚠️") else 2)))

        return {
            "log_name": log_name,
            "task": task,
            "task_type": task_type,
            "model": model,
            "status": status,
            "failure_category": failure_category,
            "n_items": len(items),
            "composite_score": composite_score,
            "pre_cure_score": pre_cure_score,
            "post_cure_score": post_cure_score,
            "scores": {
                "schema_adherence": schema_score,
                "verbatim_faithfulness": verbatim_score,
                "pedagogical_quality": pedagogy_score,
                "uniqueness": uniqueness_score,
            },
            "flags": flags,
        }

    @classmethod
    def audit_all_logs(cls, logs_dir: Path = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Scans and audits all .log files in the logs/ directory.
        Returns the overall Hero Board leaderboard and individual log audits.
        Results are cached by MD5 hash of entries' metadata so repeated calls stay fast and thread-safe.
        """
        if logs_dir is None:
            logs_dir = config.project_root / "logs"
        logs_dir = Path(logs_dir)

        if not logs_dir.exists():
            return {"hero_board": [], "audits": []}

        entries = []
        for path in logs_dir.glob("*.log"):
            try:
                st = path.stat()
            except OSError:
                continue
            entries.append((path, st.st_mtime, st.st_size))

        if not entries:
            return {"hero_board": [], "audits": []}

        # Build compact hash-based cache key using sha256 (FIPS compliant)
        sig_data = f"{logs_dir.as_posix()}:{len(entries)}:" + ":".join(f"{p.name}:{mt}:{sz}" for (p, mt, sz) in entries)
        cache_key = hashlib.sha256(sig_data.encode("utf-8")).hexdigest()

        with cls._AUDIT_LOCK:
            if use_cache and cls._AUDIT_CACHE["key"] == cache_key and cls._AUDIT_CACHE["value"] is not None:
                return cls._AUDIT_CACHE["value"]

        # Newest first
        entries.sort(key=lambda e: e[1], reverse=True)

        audits: List[Dict[str, Any]] = []
        model_stats: Dict[str, Dict[str, Any]] = {}

        # Worker for ThreadPoolExecutor
        def _process_one(log_file: Path) -> Optional[Dict[str, Any]]:
            parsed = cls.parse_log_file(log_file)
            if not parsed:
                return None
            evaluation = cls.evaluate_log(parsed)
            if evaluation.get("task_type") == "intermediate_prose":
                return None
            return evaluation

        # Parallelize log parsing and evaluation (preserves recency order)
        with ThreadPoolExecutor() as executor:
            raw_evaluations = list(executor.map(_process_one, [e[0] for e in entries]))

        for evaluation in raw_evaluations:
            if not evaluation:
                continue

            audits.append(evaluation)

            model = evaluation["model"]
            if model not in model_stats:
                model_stats[model] = {
                    "model": model,
                    "runs": 0,
                    "success_runs": 0,
                    "failed_runs": 0,
                    "total_score": 0.0,
                    "schema_sum": 0.0,
                    "schema_runs": 0,
                    "verbatim_sum": 0.0,
                    "verbatim_runs": 0,
                    "pedagogy_sum": 0.0,
                    "pedagogy_runs": 0,
                    "uniqueness_sum": 0.0,
                    "uniqueness_runs": 0,
                }

            s = model_stats[model]
            s["runs"] += 1

            # Exclude FAILED status logs from the quality leaderboard composite score
            if evaluation.get("status") == "FAILED":
                s["failed_runs"] += 1
                continue

            s["success_runs"] += 1
            s["total_score"] += evaluation["composite_score"]

            sc = evaluation["scores"]
            if sc["schema_adherence"] is not None:
                s["schema_sum"] += sc["schema_adherence"]
                s["schema_runs"] += 1
            if sc["verbatim_faithfulness"] is not None:
                s["verbatim_sum"] += sc["verbatim_faithfulness"]
                s["verbatim_runs"] += 1
            if sc["pedagogical_quality"] is not None:
                s["pedagogy_sum"] += sc["pedagogical_quality"]
                s["pedagogy_runs"] += 1
            if sc["uniqueness"] is not None:
                s["uniqueness_sum"] += sc["uniqueness"]
                s["uniqueness_runs"] += 1

        # Build Hero Board Leaderboard
        hero_board = []
        for model, s in model_stats.items():
            succ = s["success_runs"]
            if succ == 0:
                comp = 0.0
                sch_avg = vbt_avg = ped_avg = unq_avg = 0.0
            else:
                comp = round(s["total_score"] / succ, 1)
                sch_avg = round((s["schema_sum"] / (s["schema_runs"] * W_SCHEMA)) * 100.0, 1) if s["schema_runs"] else 100.0
                vbt_avg = round((s["verbatim_sum"] / (s["verbatim_runs"] * W_VERBATIM)) * 100.0, 1) if s["verbatim_runs"] else 100.0
                ped_avg = round((s["pedagogy_sum"] / (s["pedagogy_runs"] * W_PEDAGOGY)) * 100.0, 1) if s["pedagogy_runs"] else 100.0
                unq_avg = round((s["uniqueness_sum"] / (s["uniqueness_runs"] * W_UNIQUENESS)) * 100.0, 1) if s["uniqueness_runs"] else 100.0

            hero_board.append({
                "model": model,
                "runs": s["runs"],
                "success_runs": succ,
                "failed_runs": s["failed_runs"],
                "composite_score": comp,
                "schema_adherence_avg": sch_avg,
                "verbatim_faithfulness_avg": vbt_avg,
                "pedagogical_quality_avg": ped_avg,
                "uniqueness_avg": unq_avg,
            })

        # Sort Leaderboard descending by composite score
        hero_board.sort(key=lambda x: x["composite_score"], reverse=True)

        result = {"hero_board": hero_board, "audits": audits}
        with cls._AUDIT_LOCK:
            cls._AUDIT_CACHE["key"] = cache_key
            cls._AUDIT_CACHE["value"] = result
        return result
