"""
librarian/qa_ticket.py — Level-1 repair tickets that name a field, never a sentence.

Backlog F10. A repair ticket is the only channel through which Level-1 QA talks to the model,
so its shape decides what the model is allowed to do. The old builder (the inline
_format_defect_ticket in librarian/llm.py) emitted a free-text error plus a vague "[LOOKUP]:
<source text>" hint, which left the model free to rewrite whole items — and in the
Book_2_Unit_3_Section_A run it rewrote the four options the blueprint had already fixed, then
repeated the same mistake. This module replaces it with a table: every flag maps to one field
path and one instruction of the form "this field does not equal the declared value; make it
equal to the declared value".

Three rules the table enforces:

1. A ticket names a field path (`questions[2].options`), never a sentence and never a whole item.
2. A ticket never carries source text — no quote, no example sentence, no definition. The model
   already has all of it in the first turn; repeating it is what taught the model to copy
   (Backlog B).
3. The mandate is emitted only for the declarations that were actually violated, and it always
   closes with "re-check the fixed field against the declared value".

Level-1 QA may only report "field X does not equal the declared value" (F10 rule 0), so the
table below contains only flags of that kind. Sentence-level findings belong to the Level-2
expert audit (librarian/expert_auditor.py, scripts/l2_expert_audit.py).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The array a task's items live in, in the order the schemas declare them.
ARRAY_KEYS: Tuple[str, ...] = ("questions", "vocabulary", "expressions", "items", "points", "rules")

# Fields that identify an item inside an array, in priority order.
_NAME_FIELDS: Tuple[str, ...] = ("target_word", "word", "headword", "term", "pattern_name")

# Sentinel field meaning "this flag is about the array as a whole", not about one item.
ARRAY_LEVEL = "@array"

MANDATE_LINE = (
    "MANDATE: rewrite ONLY the field named in each ticket. Every other item and every other "
    "field must be output exactly as you already wrote it. After fixing a field, re-check it "
    "against the declared value before you answer."
)

# ---------------------------------------------------------------------------
# The table: flag substrings -> (field name, instruction).
#
# A field name is filled into `{array_key}[{index}].<field>`; ARRAY_LEVEL means the flag is
# about coverage of the array rather than about one item. An instruction may only say which
# field is wrong and which declared value it must be matched against — it may never supply
# content, because supplying content is what taught the model to copy.
# ---------------------------------------------------------------------------
_FIELD_RULES: Tuple[Tuple[Tuple[str, ...], str, str], ...] = (
    # --- Prescribed Options Contract (Backlog D1) -------------------------------------
    (("prescribed options altered",),
     "options",
     "The blueprint prescribes this item's options. Restore the list to exactly the prescribed "
     "options in the prescribed order; do not invent, rename, or reorder an option."),
    (("prescribed answer index altered",),
     "correct_answer_index",
     "The blueprint declares which option is the answer. Set correct_answer_index back to the "
     "declared index and leave the options themselves untouched."),
    # --- Declared inflection (Backlog D1) ---------------------------------------------
    (("inflection discordance",),
     "target_word",
     "target_word does not carry the inflectional form the blueprint declares. Rewrite it in "
     "exactly the declared form and keep it the option at the declared index."),
    # --- Verbatim-copy and leakage gates ----------------------------------------------
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
    # --- Fields that must equal a declared value --------------------------------------
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
    # --- Extraction fields (Backlog B) -------------------------------------------------
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
    # --- Grammar skeleton gates (Backlog E) -------------------------------------------
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
    # --- Structural and cross-item defects (no single item to point at) ---------------
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
    # --- Array coverage ---------------------------------------------------------------
    (("incomplete target coverage", "incomplete_coverage"),
     ARRAY_LEVEL,
     "The array is missing syllabus targets. Append one item per missing target and leave every "
     "existing item untouched."),
)

# Flags the model cannot legitimately repair, so they never become tickets:
# 'distractor slot illegality' and 'blueprint self-conflict' are both defects in the blueprint's
# own declarations — the options and the anchor are blueprint-owned, so a regeneration would only
# invent a different defect. They stay ⚠️ notes for the human backlog (scripts/待修清单.md).
_NOT_REPAIRABLE: Tuple[str, ...] = (
    "distractor slot illegality",
    "blueprint self-conflict",
)

# A flag that matches no rule still gets a ticket, but the fallback keeps it honest rather than
# letting the table drift out of sync with the evaluator silently.
_FALLBACK_FIELD = "question"
_FALLBACK_ACTION = (
    "This field does not match the value the blueprint declares for it. Restore the declared "
    "value and change nothing else."
)

_ITEM_NAME_RE = re.compile(
    r"(?:Quiz item|Item|item|target word|target|word|Headword|headword)\s+['\"\u2018]([^'\"\u2019]+)['\"\u2019]?"
)

# Flags that quote offending content rather than naming an item. The span is used to locate the
# item, never to describe the defect.
_QUOTED_RE = re.compile(r"['\u2018\"]([^'’\"]{4,})['’\"]")

# Fields a quoted span can be searched in.
_CONTENT_FIELDS: Tuple[str, ...] = (
    "quoted_sentence", "quote", "source_sentence", "question",
    "pattern_formula", "definition", "explanation", "imitation_example",
)


def detect_array_key(parsed: Optional[Dict[str, Any]]) -> str:
    """The array key this task's items live in."""
    if isinstance(parsed, dict):
        for key in ARRAY_KEYS:
            if isinstance(parsed.get(key), list):
                return key
    return "items"


def _snippet_from_flag(flag: str) -> Optional[str]:
    """The longest quoted multi-word span in a flag, for locating an item only."""
    for span in sorted(_QUOTED_RE.findall(flag), key=len, reverse=True):
        candidate = span.strip().rstrip(".… ").lower()
        if len(candidate) >= 8 and " " in candidate:
            return candidate
    return None


def _is_source_like(span: str) -> bool:
    """True when a quoted span is a stretch of text rather than a name or a label.

    'manufacture' and 'past tense (VBD)' are declarations the model must match against; 'The
    children carried their own buckets through the mud.' is text the model already wrote.
    """
    cleaned = span.strip()
    if len(cleaned.split()) >= 4:
        return True
    return len(cleaned) >= 20


def _redact_source_text(flag: str) -> str:
    """Strip source-like quoted spans from a flag before it reaches a ticket.

    The evaluator quotes the offending text in its flag. The model already produced that text and
    already has the source in Turn 1; quoting it again inside a repair instruction is exactly what
    taught the model to copy (Backlog B).
    """
    def _replace(match: "re.Match[str]") -> str:
        span = match.group(1)
        return "<source text redacted>" if _is_source_like(span) else match.group(0)

    return _QUOTED_RE.sub(_replace, flag).strip()


def item_name_from_flag(flag: str) -> Optional[str]:
    """The headword a flag names, if it names one."""
    match = _ITEM_NAME_RE.search(flag)
    if not match:
        return None
    return match.group(1).strip() or None


def locate_item(flag: str, items: Sequence[Any]) -> Optional[int]:
    """Index of the item a flag refers to, by the name the flag carries.

    The name is preferred because it is what the field-level fix is applied to. When a flag only
    quotes content (the extraction gates quote the offending sentence), the quoted span is used to
    find the item — but it is used for locating only; it never reaches the ticket.
    """
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
        # An inflected target is named by its surface form in the flag but by its lemma in the
        # array, or the other way round, so a shared-stem prefix is the fallback.
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
    """(field name, instruction) for one evaluator flag."""
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
    """One ticket: the field that is wrong, how it differs from the declared value, and where
    the declared value can be read. Never a sentence, never source text."""
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
    """Turn evaluator flags into repair tickets.

    Only a ❌ flag drives a repair. A ⚠️ flag is a note for the human reviewer, and a flag listed
    in _NOT_REPAIRABLE is a defect in the blueprint's own declarations, which the model has no
    legitimate way to fix. A flag whose item cannot be located is dropped rather than turned into
    a vague instruction — a ticket the model cannot act on is worse than no ticket, because it
    invites the model to guess which item was meant.
    """
    key = array_key or detect_array_key(parsed)
    items = parsed.get(key, []) if isinstance(parsed, dict) else []
    if not isinstance(items, list):
        items = []
    is_quiz = "quiz" in task_name.lower()
    # Where the declared value lives. For a quiz the values are the blueprint's own declarations,
    # which the model wrote in Turn 1; for an extraction task they are the source section, whose
    # header the caller supplies.
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
    """The fallback ticket when no flag could be attributed to a field.

    A low composite score with no locatable defect is the one case where a ticket cannot name a
    field, so it names the array and the scored dimension instead — and it says plainly that the
    array must be re-checked against the declarations rather than rewritten.
    """
    dimension = lowest_dimension or "overall quality"
    return (
        f"- [FIELD]: `{array_key}` (re-audit every item against its declared values)\n"
        f"  [ERROR]: composite QA score {composite}/100; lowest dimension: {dimension}\n"
        f"  [FIX]: Do not rewrite items. Re-check each item's fields against the values declared "
        f"for it and correct only the fields that differ.\n"
        f"  [LOOKUP]: {source_header}"
    )


def mandate_for_tickets(tickets: Sequence[str]) -> str:
    """The mandate, built only from the declarations actually violated (F10 rule 4).

    The old builder appended a fixed per-task-type invariant naming rules the item may never
    have broken, which told the model to re-derive constraints instead of fixing the field.
    What remains is the list of fields this run is allowed to touch, and the instruction to
    re-check each one against the declared value.
    """
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
    """The full block handed to the model: the tickets first, then the mandate."""
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
    """One call, one block — the whole repair-prompt channel librarian/llm.py uses."""
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
    """Guard used by the tests: a ticket must name a field, a fix and a lookup location, and must
    never carry a source sentence — a ticket that quotes the text is what taught the model to copy
    (Backlog B)."""
    if "[FIELD]:" not in ticket or "[FIX]:" not in ticket or "[LOOKUP]:" not in ticket:
        return False
    if "`" not in ticket.splitlines()[0]:
        return False
    for span in _QUOTED_RE.findall(ticket):
        if _is_source_like(span):
            return False
    return True

