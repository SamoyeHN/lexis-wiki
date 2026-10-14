"""
librarian/qa_ticket.py — Field-level Level-1 QA repair ticket builder.

Maps deterministic evaluator flags (`❌`) to targeted JSON field paths (e.g. `questions[2].options`)
and concise repair instructions without echoing raw source sentences, ensuring LLM retry turns
modify only the violated field while keeping all other items intact.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

ARRAY_KEYS: Tuple[str, ...] = (
    "questions", "vocabulary", "expressions", "grammar_patterns",
    "concepts", "branches", "items", "points", "rules"
)

_NAME_FIELDS: Tuple[str, ...] = (
    "target_word", "word", "headword", "term", "pattern_name",
    "pattern_formula", "topic", "branch_name"
)

ARRAY_LEVEL = "@array"

MANDATE_LINE = (
    "MANDATE: rewrite ONLY the field named in each ticket. Every other item and every other "
    "field must be output exactly as you already wrote it. After fixing a field, re-check it "
    "against the declared value before you answer."
)

# Maps evaluator flag substrings -> (target field name, repair instruction)
_FIELD_RULES: Tuple[Tuple[Tuple[str, ...], str, str], ...] = (
    (("prescribed options altered",),
     "options",
     "The blueprint prescribes this item's options. Restore the list to exactly the prescribed "
     "options in the prescribed order; do not invent, rename, or reorder an option."),
    (("prescribed answer index altered",),
     "correct_answer_index",
     "The blueprint declares which option is the answer. Set correct_answer_index back to the "
     "declared index and leave the options themselves untouched."),
    (("inflection discordance",),
     "target_word",
     "target_word does not carry the inflectional form the blueprint declares. Rewrite it in "
     "exactly the declared form and keep it the option at the declared index."),
    (("stem verbatim from",),
     "question",
     "The stem reuses wording from a corpus model. Write a new stem with the same structure and "
     "register but no shared wording; keep the options and the answer index untouched."),
    (("cross-target leakage",),
     "question",
     "The stem contains a headword that belongs to another item. Remove that word from the stem, "
     "keep the target inside the blank, and keep every other field untouched."),
    (("target word leaks verbatim",),
     "question",
     "The target word appears outside the blank. Rebuild the stem so the target appears only as "
     "the blank; keep every other field untouched."),
    (("indefinite article leakage", "indefinite article"),
     "question",
     "An indefinite article sits before the blank and leaks the answer's sound. Use 'the ____', "
     "a plural, or a possessive instead; keep every other field untouched."),
    (("multiple blanks",),
     "question",
     "The stem contains more than one blank. Rewrite it with exactly one '____'."),
    (("fill-in-the-blank slot",),
     "question",
     "The stem has no '____' slot. Add exactly one blank at the position the blueprint frame "
     "declares."),
    (("blank slot pos mismatch",),
     "question",
     "The stem does not match the part of speech the blueprint declares for this item. Rebuild "
     "the stem so the slot is the declared part of speech."),
    (("anchor missing in question stem",),
     "question",
     "The collocational anchor the blueprint declares is absent from the stem. Put the declared "
     "anchor into the stem and keep the options untouched."),
    (("explanation anchor not grounded",),
     "explanation",
     "The explanation claims a collocation the stem does not contain. Rewrite the explanation so "
     "it describes only what the stem actually says."),
    (("selection clustering",),
     "correct_answer_index",
     "This item's answer index repeats the position other items in the batch use. Move the answer "
     "to the index the blueprint assigns this item."),
    (("does not appear in quoted sentence", "non-verbatim quote", "ungrounded in quote",
      "hallucinated quote"),
     "quoted_sentence",
     "quoted_sentence is not a sentence of the source text. Copy the source sentence that "
     "actually contains this word, word for word, and change nothing else in the item."),
    (("not a recognized english word",),
     "word",
     "This word is neither a syllabus target nor a word of the source text. Replace it with the "
     "syllabus target it was meant to be."),
    (("copy-pasted definition",),
     "definition",
     "definition duplicates another item's definition. Restore the definition the dictionary "
     "entry for this word gives."),
    (("degenerate formula slot",),
     "formula_slot",
     "The formula slot is a degenerate fill. Restore the slot the declared pattern's formula "
     "requires."),
    (("deviates from the pre-extracted skeleton", "matches none"),
     "pattern_name",
     "pattern_name is not one of the patterns the source pre-extraction found. Set it to the "
     "pattern the blueprint assigns this item."),
    (("duplicates the source quote",),
     "imitation_example",
     "imitation_example repeats the source quote instead of imitating its structure. Write a new "
     "sentence with the same structure and different content."),
    (("invalid_schema", "invalid_item_type", "insufficient_items", "empty object",
      "invalid or missing json"),
     ARRAY_LEVEL,
     "The array itself is malformed. Rebuild it from the schema the task declares, one object "
     "per item, and leave the surrounding fields untouched."),
    (("distractors recycled repeatedly", "word-family collisions", "redundant headwords",
      "duplicate item"),
     ARRAY_LEVEL,
     "Two or more items reuse the same option or the same word family. Replace the repeated "
     "option with the distractor the blueprint assigns this item; do not touch the other fields."),
    (("in-list distractor recycling",),
     "options",
     "A distractor repeats a word that already appears in the item's own list. Replace it with "
     "the distractor the blueprint assigns this item; keep the answer option untouched."),
    (("not found in supplied word list",),
     "word",
     "This word is not on the syllabus target list the task supplies. Replace it with the target "
     "it was meant to be and change nothing else in the item."),
    (("missing quote/quoted_sentence", "prompt instruction leakage in quote"),
     "quoted_sentence",
     "quoted_sentence is missing or contains prompt wording rather than source wording. Copy the "
     "source sentence that actually contains this word, word for word."),
    (("incomplete target coverage", "incomplete_coverage"),
     ARRAY_LEVEL,
     "The array is missing syllabus targets. Append one item per missing target and leave every "
     "existing item untouched."),
)

_NOT_REPAIRABLE: Tuple[str, ...] = (
    "distractor slot illegality",
    "blueprint self-conflict",
)

_FALLBACK_FIELD = "question"
_FALLBACK_ACTION = (
    "This field does not match the value the blueprint declares for it. Restore the declared "
    "value and change nothing else."
)

_ITEM_NAME_RE = re.compile(
    r"(?:Quiz item|Item|item|target word|target|word|Headword|headword)\s+['\"\u2018]([^'\"\u2019]+)['\"\u2019]?"
)

_QUOTED_RE = re.compile(r"['\u2018\"]([^'’\"]{4,})['’\"]")

_CONTENT_FIELDS: Tuple[str, ...] = (
    "quoted_sentence", "quote", "source_sentence", "question",
    "pattern_formula", "definition", "explanation", "imitation_example",
)


def detect_array_key(parsed: Optional[Dict[str, Any]]) -> str:
    """Return the primary array key in the parsed payload."""
    if isinstance(parsed, dict):
        for key in ARRAY_KEYS:
            if isinstance(parsed.get(key), list):
                return key
    return "items"


def _snippet_from_flag(flag: str) -> Optional[str]:
    """Extract the longest quoted multi-word span in a flag for locating an item."""
    for span in sorted(_QUOTED_RE.findall(flag), key=len, reverse=True):
        candidate = span.strip().rstrip(".… ").lower()
        if len(candidate) >= 8 and " " in candidate:
            return candidate
    return None


def _is_source_like(span: str) -> bool:
    """Return True if a quoted span looks like a sentence rather than a short label."""
    cleaned = span.strip()
    if len(cleaned.split()) >= 4:
        return True
    return len(cleaned) >= 20


def _redact_source_text(flag: str) -> str:
    """Redact sentence-length quoted spans from a flag before emitting a repair ticket."""
    def _replace(match: "re.Match[str]") -> str:
        span = match.group(1)
        return "<source text redacted>" if _is_source_like(span) else match.group(0)

    return _QUOTED_RE.sub(_replace, flag).strip()


def item_name_from_flag(flag: str) -> Optional[str]:
    """Extract the headword/target name referenced by a flag, if present."""
    match = _ITEM_NAME_RE.search(flag)
    if not match:
        return None
    return match.group(1).strip() or None


def locate_item(flag: str, items: Sequence[Any]) -> Optional[int]:
    """Locate the 0-based index of the item referenced by an evaluator flag."""
    name = item_name_from_flag(flag)
    if not items:
        return None
    needle = (name or "").lower()
    if needle:
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            for field in _NAME_FIELDS:
                value = item.get(field)
                if isinstance(value, str) and value.strip().lower() == needle:
                    return index
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            for field in _NAME_FIELDS:
                value = item.get(field)
                if isinstance(value, str) and value.strip().lower()[:4] == needle[:4]:
                    return index
    snippet = _snippet_from_flag(flag)
    if snippet:
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            for field in _CONTENT_FIELDS:
                value = str(item.get(field) or "").lower()
                if snippet in value or (len(snippet) >= 15 and snippet[:15] in value):
                    return index
    return None


def rule_for_flag(flag: str) -> Tuple[str, str]:
    """Return `(field_name, repair_instruction)` for an evaluator flag."""
    lowered = flag.lower()
    for needles, field_name, action in _FIELD_RULES:
        if any(needle in lowered for needle in needles):
            return field_name, action
    return _FALLBACK_FIELD, _FALLBACK_ACTION


def build_qa_ticket(
    flag: str,
    *,
    array_key: str = "items",
    item_index: Optional[int] = None,
    source_header: str = "the blueprint declaration for this item",
) -> str:
    """Format a single field-level QA repair ticket."""
    field_name, action = rule_for_flag(flag)
    if field_name == ARRAY_LEVEL:
        target = f"`{array_key}` (array coverage — append the missing items)"
    else:
        target = f"`{array_key}[{item_index}].{field_name}`"
    return (
        f"- [FIELD]: {target}\n"
        f"  [ERROR]: {_redact_source_text(flag)}\n"
        f"  [FIX]: {action}\n"
        f"  [LOOKUP]: {source_header}"
    )


def build_qa_tickets(
    flags: Sequence[str],
    *,
    task_name: str = "",
    parsed: Optional[Dict[str, Any]] = None,
    array_key: Optional[str] = None,
    source_header: Optional[str] = None,
    max_tickets: int = 5,
) -> List[str]:
    """Convert fatal (`❌`) evaluator flags into deduplicated field-level repair tickets."""
    key = array_key or detect_array_key(parsed)
    items = parsed.get(key, []) if isinstance(parsed, dict) else []
    if not isinstance(items, list):
        items = []
    is_quiz = "quiz" in task_name.lower()
    fallback_header = source_header or "Turn 1 context sections"
    item_lookup = (
        "the blueprint declaration for this item (Turn 1)" if is_quiz else fallback_header
    )
    array_lookup = "the syllabus target list in Turn 1" if is_quiz else fallback_header

    tickets: List[str] = []
    seen = set()
    for flag in flags:
        if not flag or not flag.strip():
            continue
        lowered = flag.lower()
        if "❌" not in flag:
            continue
        if any(marker in lowered for marker in _NOT_REPAIRABLE):
            continue
        field_name, _ = rule_for_flag(flag)
        if field_name == ARRAY_LEVEL:
            item_index = None
            lookup = array_lookup
        else:
            item_index = locate_item(flag, items)
            if item_index is None:
                continue
            lookup = item_lookup
        ticket = build_qa_ticket(
            flag, array_key=key, item_index=item_index, source_header=lookup
        )
        if ticket in seen:
            continue
        seen.add(ticket)
        tickets.append(ticket)
        if len(tickets) >= max_tickets:
            break
    return tickets


def score_ticket(
    composite: Optional[float],
    lowest_dimension: str = "",
    *,
    array_key: str = "items",
    source_header: str = "Turn 1 context sections",
) -> str:
    """Fallback array-level ticket when a low composite score has no single locatable field defect."""
    dimension = lowest_dimension or "overall quality"
    return (
        f"- [FIELD]: `{array_key}` (re-audit every item against its declared values)\n"
        f"  [ERROR]: composite QA score {composite}/100; lowest dimension: {dimension}\n"
        f"  [FIX]: Do not rewrite items. Re-check each item's fields against the values declared "
        f"for it and correct only the fields that differ.\n"
        f"  [LOOKUP]: {source_header}"
    )


def mandate_for_tickets(tickets: Sequence[str]) -> str:
    """Build the closing mandate listing only the field paths referenced in `tickets`."""
    fields: List[str] = []
    for ticket in tickets:
        for line in ticket.splitlines():
            if line.strip().startswith("- [FIELD]:"):
                match = re.search(r"`([^`]+)`", line)
                path = match.group(1) if match else ""
                if path and path not in fields:
                    fields.append(path)
    if not fields:
        return MANDATE_LINE
    listed = ", ".join(f"`{path}`" for path in fields)
    return (
        f"🛑 MANDATE: the only fields you may change are {listed}. Every other item and every "
        "other field must be output exactly as you already wrote it. After fixing each field, "
        "re-check it against the declared value before you answer. Return ONLY the complete "
        "corrected JSON object."
    )


def format_qa_ticket_block(tickets: Sequence[str], header: str = "QUALITY AUDIT DEFECT TICKET") -> str:
    """Format the complete QA repair prompt block (tickets + mandate)."""
    if not tickets:
        return ""
    return (
        f"\n\n### 🚨 {header}\n"
        "Fix ONLY the field named in each ticket while keeping everything else exactly as "
        "written:\n\n"
        + "\n\n".join(tickets)
        + "\n\n" + mandate_for_tickets(tickets)
    )


def tickets_for_flags(
    flags: Sequence[str],
    *,
    task_name: str = "",
    parsed: Optional[Dict[str, Any]] = None,
    source_header: Optional[str] = None,
    max_tickets: int = 5,
) -> str:
    """Build and format the complete QA repair block from evaluator flags."""
    return format_qa_ticket_block(
        build_qa_tickets(
            flags,
            task_name=task_name,
            parsed=parsed,
            source_header=source_header,
            max_tickets=max_tickets,
        )
    )


def ticket_is_repairable(ticket: str) -> bool:
    """Verify that a ticket contains `[FIELD]`, `[FIX]`, and `[LOOKUP]` and no unredacted source sentences."""
    if "[FIELD]:" not in ticket or "[FIX]:" not in ticket or "[LOOKUP]:" not in ticket:
        return False
    if "`" not in ticket.splitlines()[0]:
        return False
    for span in _QUOTED_RE.findall(ticket):
        if _is_source_like(span):
            return False
    return True
