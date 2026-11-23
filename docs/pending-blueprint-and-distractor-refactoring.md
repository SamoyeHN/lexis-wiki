# Pending: Blueprint `cloze_slot_frame` & Distractor Engine Gradual Refactoring

> **Status**: Completed (Phase 1 `SlotFrame` contract and Phase 3 `distractor_engine.py` modular decomposition shipped; Phase 2 linguistic discrimination tightening active as ongoing polish).

---

## 1. Why Defer Structural `cloze_slot_frame` Refactoring (Historical Context)
1. **Immediate LLM quality gains are already captured**:
   - `_render_llm_surface_frame()` in `librarian/processor_core/quiz_generator.py` already decouples internal symbolic frames from natural-English prompts sent to local LLMs, and L1 meta-placeholder leakage gates (`evaluator.py`, `quiz_integrity_gates.py`) catch literal copying (`to do something`, `is/are`, `(should)`, `[...]`) at zero token cost.
   - Verified on `Book_3_Unit_1_Section_A`: L2 base score improved from `48% -> 71%`, single-fit defects halved (`6 -> 3`), while preserving 100% compatibility with existing string-based unit tests (`301 passed, 1 skipped` in `tests/test_linguistics.py`).
2. **Low short-term ROI for immediate breaking rewrite**:
   - `cloze_slot_frame` was consumed as a `str` across `quiz_frames.py`, `quiz_vocab.py`, `quiz_generator.py`, and 300+ unit tests. Converting it to a structured dataclass with backward-compatible `__str__` and `__eq__` resolved this completely without breaking any tests.

---

## 2. Trigger Conditions to Start Refactoring
Initiate this gradual refactoring when either of the following becomes a concrete blocker:
1. **Automated L1 syntactic slot validation** requires structured access to left auxiliary, bound preposition, and number/countability constraints (instead of regex-parsing `cloze_slot_frame` strings).
2. **Modularization of oversized core files** (`distractor_engine.py` ~8.8k lines, `quiz_vocab.py` ~4.5k lines, `quiz_frames.py` ~2.4k lines) is scheduled to curb patch-on-patch regression coupling.

---

## 3. Phased Refactoring Plan

### Phase 1: Structured `SlotFrame` Dataclass (Completed)
- Introduced lightweight `SlotFrame` dataclass (`left_anchor`, `aux_lock`, `right_anchor`, `prep`, `ontology_hint`, `constraints`) with:
  - `.to_symbolic_str()` — preserves existing `cloze_slot_frame` string contract for legacy tests.
  - `.to_llm_surface()` — replaces regex post-processing in `_render_llm_surface_frame()` with direct structured rendering.
  - Implements `__str__`, `__eq__` against `str`, and `__hash__` for 100% backward compatibility.
- Verified with `tests/test_slot_frame.py` (8 passing tests) and `tests/audit_blueprints.py` (100% pass rate).

### Phase 2: Distractor Engine Hygiene & Discrimination Tightening (Ongoing Polish)
1. **Separate Preposition-Government vs. Semantic-Discrimination Distractor Modes**:
   - Prevent obvious grammatical giveaways (e.g., `alike` in `are ____ to ...`) from mixing into semantic-contrast items, and ensure polarity-sensitive antonyms (`opposite`/`contrasting` vs `identical`) are paired with explicit contextual polarity cues.
2. **Reduce `corpus_fallback` for Sparse Derived Adjectives (`habitable`, `renewable`)**:
   - Keep strict `double_key_collision` intact; add a domain-coherent, non-collocating fallback tier (e.g. PedLex domain/topic peers or lightweight 1-token candidate filtering) before falling back to generic `corpus_fallback`.
3. **Purify `_ocd_lexfile_cache` Noun Index**:
   - Exclude inflected plural forms (e.g. `flies` leaking into `noun.location` for singular `corner`) and ultra-rare specialist senses during cache construction.

### Phase 3: Decompose `distractor_engine.py` (Completed)
- Split `distractor_engine.py` (8,825 lines) along clean domain boundaries into 6 specialized mixins:
  - `distractor_guards.py`: Syntactic slot clash, double-key collision, countability contrast & light verb checks
  - `distractor_paronyms.py`: Orthographic paronym generation, morpho-phonetic distance & semantic fields
  - `distractor_phrases.py`: Multi-word phrase distractors, idiom mapping & phrase normalization
  - `distractor_inflections.py`: Inflection screening, voice tags & target reconciliation
  - `distractor_fallbacks.py`: Antonym/synonym retrieval, dynamic corpus fallback & construction contrast
  - `distractor_vocab.py`: Single-word vocabulary distractor generation and zero-collision orchestrator
  - `distractor_engine.py`: 42-line clean facade mixin preserving 100% backward compatibility.
