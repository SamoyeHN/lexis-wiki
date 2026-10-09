"""
Cloze prompt + deterministic gates for ONE pattern:

    SEPARABLE phrasal verb with a NOUN object AFTER the particle.
    e.g. "take away my book"   (the noun follows the particle)

The rule under test
-------------------
A separable phrasal verb takes a NOUN object either side of the particle; we lock
the AFTER order:  take away my book  (verb + particle + noun).  With a PRONOUN
object the object must sit in the middle ("take it away"), but the object in this
item is a NOUN, so it comes after the particle.

Consequence for the cloze
-------------------------
  * the blank covers the 2-word predicate "take away" (verb + particle);
  * a NOUN object is a FIXED part of the stem, placed AFTER the blank;
  * every option is a 2-word "verb + particle" (take away / back / off / up);
  * no option may carry the pronoun 'it' (that is the fronted form, not this).

LLM is local (Ollama) and only writes the stem + explanation; deterministic code
owns the option pool and the correct index.
"""

from __future__ import annotations

import re

import pytest

from librarian.linguistics import LinguisticEngine as L

__all__ = [
    "build_separable_blueprint",
    "format_separable_prompt",
]

_PRON = {"it", "me", "you", "him", "her", "us", "them"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _is_verb_particle(o: str, head: str = "take") -> bool:
    """True iff ``o`` is a 2-word 'verb + particle' (no pronoun object)."""
    toks = _norm(o).split()
    return len(toks) == 2 and toks[0] == head and toks[1] not in _PRON


def _near_duplicate(target: str, option: str) -> bool:
    """Two options are near-duplicates iff they differ ONLY by the pronoun
    (same verb + same particle).  Different particle = different meaning = allowed."""
    t = [w for w in _norm(target).split() if w not in _PRON]
    o = [w for w in _norm(option).split() if w not in _PRON]
    return t == o


def _assert_options(options: list[str], target: str, correct_index: int, head: str = "take") -> None:
    assert len(options) == 4, f"expected 4 options, got {options!r}"
    target = _norm(target)
    assert target in options, f"target {target!r} missing from {options!r}"
    for o in options:
        assert _is_verb_particle(o, head), f"option {o!r} is not a 2-word 'verb + particle'"
        assert "it" not in _norm(o).split(), f"option {o!r} carries the pronoun 'it' (fronted form)"
        assert (not _near_duplicate(target, o)) or o == target, f"near-duplicate {o!r} of {target!r}"
    assert options[correct_index] == target, "correct index must point at the target"


def _ollama_online(model_name: str, host: str | None = None) -> bool:
    """True iff ``model_name`` is on the SAME Ollama the LLMClient uses (config.api_url)."""
    if host is None:
        try:
            from librarian.config import config
            host = (config.get("api_url") or "http://localhost:11434").rstrip("/")
        except Exception:
            host = "http://localhost:11434"
    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return model_name in resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, Exception):
        return False


def build_separable_blueprint(
    head: str = "take",
    target_particle: str = "away",
    distractor_particles: tuple[str, ...] = ("back", "off", "up"),
    definition: str = "to remove something from somewhere, or take it with you",
) -> dict:
    """Deterministic blueprint: separable phrasal verb, NOUN object after the particle.

    Options = 'verb + particle' (2 words, no pronoun).  Target 'take away';
    distractors share the head and differ by particle (back / off / up).
    """
    pool = {_norm(p) for p in L.get_ldoce_phrasal_verbs(head)}
    target = f"{head} {target_particle}"
    options = [target]
    for p in distractor_particles:
        assert f"{head} {p}" in pool, f"'{head} {p}' not attested for head {head!r}"
        options.append(f"{head} {p}")
    bp = {
        "strategy": "separable_noun_object",
        "phrase": target,
        "head": head,
        "particle": target_particle,
        "definition": definition,
        "options": options,
        "correct_answer_index": 0,
    }
    _assert_options(options, target, 0, head)
    return bp


def format_separable_prompt(bp: dict) -> str:
    opts = "\n".join(
        f"  {i + 1}. {o}{'   (correct)' if i == bp['correct_answer_index'] else ''}"
        for i, o in enumerate(bp["options"])
    )
    return f"""You are an expert English-language test writer.
Create ONE fill-in-the-blank (cloze) multiple-choice question whose answer is the SEPARABLE PHRASAL VERB 'take away', used with a NOUN object AFTER the particle: 'take away my book'.

[Why the noun goes after the particle]
  - A separable phrasal verb takes a NOUN object either side of the particle; here we lock the AFTER order: 'take away my book'.
  - (With a PRONOUN object the object must sit in the MIDDLE -- 'take it away' -- but the object in THIS item is a NOUN, so it follows the particle.)

[Blanking rule - MUST obey]
  - The blank covers the 2-word predicate 'take away' (verb + particle).
  - A NOUN object is a FIXED part of the stem and comes AFTER the blank (e.g. 'the empty trays', 'my book').
  - Put exactly one blank '____'. Use a base-form slot (modal or imperative) before the blank.
  - GOOD stem: "Before we close, please ____ the empty trays from the tables."   (-> 'take away')
  - BAD stem:  "Could you ____ it?"     (a pronoun object -- that is the 'take it away' form, not this item)
  - The sentence must give a clue that forces 'take away' (sense: {bp['definition']}) and rules out the other particles.

[Options - use exactly these 4, no others, unchanged]
{opts}
  (Every option is exactly 'verb + particle'. Do NOT use the pronoun 'it' in any option.)

[Output fields]
  - design_audit: one sentence naming the strategy (separable phrasal verb; noun object after the particle).
  - question: the stem with exactly one '____'.
  - options: the 4 options above, unchanged.
  - correct_answer_index: the index of the option marked (correct) above.
  - explanation: why the answer fits and why each distractor does not.
"""



# ---------------------------------------------------------------------------
# Deterministic (no LLM)
# ---------------------------------------------------------------------------

def test_blueprint_noun_object_options():
    bp = build_separable_blueprint()
    assert bp["strategy"] == "separable_noun_object"
    assert bp["phrase"] == "take away"
    assert bp["options"] == ["take away", "take back", "take off", "take up"]
    assert bp["correct_answer_index"] == 0
    for o in bp["options"]:
        assert _is_verb_particle(o), f"option {o!r} is not 2-word 'verb + particle'"
        assert "it" not in o.split(), f"option {o!r} must not carry the pronoun 'it'"
    for o in bp["options"][1:]:
        assert not _near_duplicate("take away", o), f"near-duplicate {o!r}"


def test_prompt_locks_noun_object_rule():
    prompt = format_separable_prompt(build_separable_blueprint())
    assert "'take away'" in prompt
    assert "NOUN object" in prompt
    # the pronoun order must be called out as NOT this item
    assert "take it away" in prompt and "not this item" in prompt
    assert "Do NOT use the pronoun 'it'" in prompt
    for o in ["take away", "take back", "take off", "take up"]:
        assert o in prompt


# ---------------------------------------------------------------------------
# Live (real local LLM) -- skipped when the model is offline
# ---------------------------------------------------------------------------

# Tracks which model was most recently loaded, so that switching to a different
# model first evicts the previous one from Ollama VRAM (mirrors processor.py's
# `llm.unload_model(...)` between generator and judge models).
_LOADED_MODEL: str | None = None


@pytest.mark.parametrize("model_name", ["llama3.1:8b", "qwen3.5:9b"])
def test_llm_separable_noun_object_compliance(model_name: str):
    if not _ollama_online(model_name):
        pytest.skip(f"{model_name} not available")

    from librarian.llm import llm

    # When switching models, clear the previous one from Ollama first.
    global _LOADED_MODEL
    if _LOADED_MODEL and _LOADED_MODEL != model_name:
        llm.unload_model(_LOADED_MODEL)
        print(f"[{model_name}] unloaded previous model {_LOADED_MODEL!r} before switch")
    _LOADED_MODEL = model_name

    bp = build_separable_blueprint()
    prompt = format_separable_prompt(bp)
    schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
            "correct_answer_index": {"type": "integer"},
            "explanation": {"type": "string"},
        },
        "required": ["question", "options", "correct_answer_index"],
    }
    d = None
    last_raw = None
    for _attempt in range(2):
        # keep_alive=0 -> Ollama evicts the model from VRAM right after this call
        # (supported by _chat_ollama; see llm.py). This also guarantees the
        # previous model is gone before the next parametrized case loads a new one.
        res = llm.chat(messages=[{"role": "user", "content": prompt}], schema=schema, model=model_name, keep_alive=0)
        last_raw = getattr(llm, "last_raw_response", None)
        if isinstance(res, dict):
            d = res
        else:  # defensive: a dataclass instance
            d = {k: getattr(res, k, None) for k in ("question", "options", "correct_answer_index", "explanation")}
        if d and d.get("question") and d.get("options"):
            break

    stem = (d.get("question") or "").strip() if d else ""
    opts = [str(o).strip() for o in (d.get("options") or [])] if d else []
    idx = d.get("correct_answer_index") if d else None
    answer = opts[idx] if (isinstance(idx, int) and 0 <= idx < len(opts)) else None
    if not (stem and opts):
        print(f"[{model_name}] RAW RESPONSE (no usable JSON after retries):")
        print(repr(last_raw)[:2000])
    print("\n" + "=" * 78)
    print(f"[{model_name}] PROMPT (separable / noun object after particle)")
    print("=" * 78)
    print(prompt)
    print("=" * 78)
    print(f"[{model_name}] MODEL OUTPUT")
    print("=" * 78)
    print(f"  Q    : {stem}")
    print(f"  O    : {opts}")
    print(f"  ans  : {answer!r}")
    print("-" * 78)

    noun_after_blank = bool(re.search(
        r"____ (the|a|an|some|my|your|his|her|their|these|those|this|that) ", stem))
    checks = {
        "exactly ONE blank '____'": stem.count("____") == 1,
        "a NOUN object follows the blank (article/possessive + noun)": noun_after_blank,
        "every option is 2-word 'verb + particle'": all(_is_verb_particle(o) for o in opts),
        "no option carries the pronoun 'it'": not any("it" in o.split() for o in opts),
        "target 'take away' is offered": "take away" in opts,
        "correct index -> 'take away'": _norm(answer) == "take away",
        "no near-duplicate option": all(not _near_duplicate("take away", o) or o == "take away" for o in opts),
    }
    print(f"[{model_name}] COMPLIANCE (noun object after particle):")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
    print("=" * 78)
    assert all(checks.values()), "LLM did not fully comply with the noun-object cloze rule"

