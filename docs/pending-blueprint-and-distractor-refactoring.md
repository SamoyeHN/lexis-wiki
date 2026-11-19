# Pending: Blueprint `cloze_slot_frame` & Distractor Engine Gradual Refactoring

> **Status**: Pending (Deferred by design — high-leverage de-metasyntaxification & L1 gates already shipped in commits `d5386af` and `c6dc1c7`).

---

## 1. Why Defer Structural `cloze_slot_frame` Refactoring Right Now
1. **Immediate LLM quality gains are already captured**:
   - `_render_llm_surface_frame()` in `librarian/processor_core/quiz_generator.py` already decouples internal symbolic frames from natural-English prompts sent to local LLMs, and L1 meta-placeholder leakage gates (`evaluator.py`, `quiz_integrity_gates.py`) catch literal copying (`to do something`, `is/are`, `(should)`, `[...]`) at zero token cost.
   - Verified on `Book_3_Unit_1_Section_A`: L2 base score improved from `48% -> 71%`, single-fit defects halved (`6 -> 3`), while preserving 100% compatibility with existing string-based unit tests (`301 passed, 1 skipped` in `tests/test_linguistics.py`).
2. **Low short-term ROI for immediate breaking rewrite**:
   - `cloze_slot_frame` is currently consumed as a `str` across `quiz_frames.py`, `quiz_vocab.py`, `quiz_generator.py`, and 300+ unit tests. Converting it to a structured dataclass immediately would carry high migration/regression cost without an equivalent jump in LLM item quality.

---

## 2. Trigger Conditions to Start Refactoring
Initiate this gradual refactoring when either of the following becomes a concrete blocker:
1. **Automated L1 syntactic slot validation** requires structured access to left auxiliary, bound preposition, and number/countability constraints (instead of regex-parsing `cloze_slot_frame` strings).
2. **Modularization of oversized core files** (`distractor_engine.py` ~8.7k lines, `quiz_vocab.py` ~4.5k lines, `quiz_frames.py` ~2.4k lines) is scheduled to curb patch-on-patch regression coupling.

---

## 3. Phased Refactoring Plan

### Phase 1: Structured `SlotFrame` Dataclass (Backward-Compatible)
- Introduce a lightweight `SlotFrame` dataclass (`left_anchor`, `aux_lock`, `right_anchor`, `prep`, `ontology_hint`, `constraints`) with:
  - `.to_symbolic_str()` — preserves existing `cloze_slot_frame` string contract for legacy tests.
  - `.to_llm_surface()` — replaces regex post-processing in `_render_llm_surface_frame()` with direct structured rendering.
- Keep `tests/audit_blueprints.py` as the mandatory 0-token pre-flight regression baseline before every change.

### Phase 2: Distractor Engine Hygiene & Discrimination Tightening
1. **Separate Preposition-Government vs. Semantic-Discrimination Distractor Modes**:
   - Prevent obvious grammatical giveaways (e.g., `alike` in `are ____ to ...`) from mixing into semantic-contrast items, and ensure polarity-sensitive antonyms (`opposite`/`contrasting` vs `identical`) are paired with explicit contextual polarity cues.
2. **Reduce `corpus_fallback` for Sparse Derived Adjectives (`habitable`, `renewable`)**:
   - Keep strict `double_key_collision` intact; add a domain-coherent, non-collocating fallback tier (e.g. PedLex domain/topic peers or lightweight 1-token candidate filtering) before falling back to generic `corpus_fallback`.
3. **Purify `_ocd_lexfile_cache` Noun Index**:
   - Exclude inflected plural forms (e.g. `flies` leaking into `noun.location` for singular `corner`) and ultra-rare specialist senses during cache construction.

### Phase 3: Decompose `distractor_engine.py` & `quiz_vocab.py`
- Split `distractor_engine.py` along clean boundaries (candidate harvesting, valency/slot clash gates, double-key screening, fallback pools) so rule updates stop causing cross-suite regressions.
