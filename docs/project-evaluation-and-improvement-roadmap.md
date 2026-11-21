# Project Evaluation & Engineering Improvement Roadmap

> **Snapshot Date**: 2026-11-19 (baseline commit `531b105`)  
> **Scope**: Architecture, maintainability, engineering guardrails, frontend/template structure, documentation alignment, and end-to-end quality evaluation of `lexis-wiki`.

---

## 1. Executive Summary

- **Core Strengths**: Strong neuro-symbolic architecture (`spaCy + PedLex + WordNet + cefrpy`), 0-token deterministic quality gates, SLA/psychometric rigor, and 30-unit corpus regression auditing (`tests/audit_blueprints.py`, `tests/audit_cloze.py`).
- **Primary Risks**: Structural maintainability debt (multi-thousand-line functions and flat Mixin God Classes), missing engineering guardrails (no lint/typecheck/CI, 333 broad `except Exception` catches, uneven test coverage), frontend/template duplication, weak internal dict contracts, and documentation drift.

---

## 2. Diagnosed Weaknesses (Evidence-Based)

### P0 — Monolithic Functions & Flat Mixin Inheritance ("Pseudo-Modularity")
- Codebase totals **100,228** Python lines across `librarian/`, `tests/`, and `scripts/`, with **85 functions $\ge 150$ lines** in `librarian/`.
- Largest single functions:
  - `build_precomputed_target_skeletons()` (`librarian/linguistics_core/quiz_vocab.py:1089`): **3,473 lines**
  - `generate_vocab_distractors()` (`librarian/linguistics_core/distractor_engine.py:230`): **3,129 lines**
  - `_lock_sense()` (`librarian/linguistics_core/wsd_engine.py:488`): **2,510 lines**
  - `generate_quiz()` (`librarian/processor_core/quiz_generator.py:1390`): **1,299 lines**
  - `determine_contextual_pos()` (`librarian/linguistics_core/cefr_and_morphology.py:2286`): **1,279 lines**
  - `generate_phrase_distractors()` (`librarian/linguistics_core/distractor_engine.py:6022`): **1,217 lines**
  - `_score_pedagogy()` (`librarian/evaluator.py:948`): **1,209 lines**
  - `chat()` (`librarian/llm.py:427`): **1,018 lines**
  - `main()` (`librarian/cli.py:10`): **849 lines**
  - `handle_api_post()` (`librarian/dashboard.py:1063`): **710 lines**
- `LinguisticEngine` (`librarian/linguistics.py`) composes **7 Mixins** (with nested sub-mixins under `QuizSkeletonsMixin`) and `WikiProcessor` (`librarian/processor.py`) composes **4 Mixins**. Implicit `self` sharing makes local refactoring and static analysis difficult and encourages patch-on-patch branching.

### P1 — Missing Engineering Guardrails & Uneven Test Coverage
- **Zero Static Analysis / CI**: No `.github`, `ruff`/`flake8`, `mypy`, or `pre-commit` configuration in the repository or `pyproject.toml`.
- **Silent Exception Swallowing**: **333** broad `except Exception` blocks across `librarian/`, risking silent fallbacks that hide schema/SQL/edge-case bugs.
- **Skewed Test Coverage**:
  - Linguistic core is heavily tested, but `tests/test_linguistics.py` alone has grown to **5,635 lines**.
  - Engineering/delivery modules have near-zero coverage: `exporter.py` (1,337 lines) and `tts_engine` have **0** references in `tests/`; `dashboard.py` (1,830 lines) is referenced in only **2** test files (`test_dashboard_api.py` is ~1KB).

### P1 — Frontend & Template Duplication
- `librarian/static/js/dashboard.js` is a **4,951-line** single-file vanilla JS monolith with no module boundaries or type checking.
- `librarian/dashboard.py` hand-rolls HTTP routing over `http.server` / `socketserver` (`handle_api_post` = 710 lines) with in-memory `DashboardState.jobs`.
- Seven interactive HTML templates (`video.html` 1,475 lines, `cloze.html` 1,249, `vocabulary.html` 1,153, `reading.html` 1,039, `mindmap.html` 979, `listening.html` 864, `translation.html` 856) inline large amounts of duplicated CSS/JS boilerplate (scoring, hints, theming, progress UI).

### P2 — Untyped Internal Contracts, Doc Drift & Repo Hygiene
- Core pipelines pass deeply nested untyped `dict`s across `quiz_patterns` $\rightarrow$ `quiz_frames` $\rightarrow$ `quiz_vocab` $\rightarrow$ `quiz_generator` $\rightarrow$ `quiz_integrity_gates` (see also [pending-blueprint-and-distractor-refactoring.md](pending-blueprint-and-distractor-refactoring.md)).
- **Documentation Drift**: `docs/cloze-discourse-architecture.md` and `GEMINI.md` still describe Cloze as Mode B (4-option fallback), whereas commit `531b105` transitioned Cloze to **Pure Open Cloze** (`options: []`, `accepted_answers`).
- **Stray File**: Repo root contains a 0-byte `./pedlex.db` while the real 389MB database lives at `./librarian/data/pedlex.db`.

### P2 — Missing End-to-End LLM Golden Benchmark
- Zero-token symbolic layers and L1 code gates are well audited, but LLM-generated quiz stems/distractors and `extractions/` artifacts lack a fixed Golden Set benchmark tracking L1/L2 score regressions across prompt/model changes (`re-audit --target extraction` remains unimplemented).

---

## 3. Phased Improvement Plan

### Phase 1: Short-Term (1–2 Weeks, High ROI, Zero Breaking Changes)
- [ ] **Add Minimal Engineering Guardrails**: Configure `ruff` (lint + format) in `pyproject.toml` (enforced incrementally on modified files) and add a lightweight CI/pre-commit check.
- [x] **Sync Doc Drift (Cloze & PedLex)**: Updated `docs/cloze-discourse-architecture.md`, `GEMINI.md`, `README.md`, `README.zh-CN.md`, `docs/pedlex-database-architecture.md`, and `docs/vocabulary-selection-and-distractor-methodology.md` to reflect Pure Open Cloze (`accepted_answers`) and actual `pedlex.db` table names/counts (`ldoce_phrase_index`: 189,410 rows).
- [ ] **Remove Stray Root File**: Remove root 0-byte `./pedlex.db` (real 389MB database lives at `./librarian/data/pedlex.db`).
- [ ] **Close Outer-Layer Test Blind Spots**: Add smoke/contract tests for `exporter.py`, `dashboard.py` core endpoints, and `cli.py`; split `tests/test_linguistics.py` (5,635 lines) into domain-focused files (`test_wsd.py`, `test_vocab_blueprint.py`, `test_cloze.py`, `test_morphology.py`).
- [ ] **Triage Broad Exceptions**: Audit the 333 `except Exception` blocks—narrow expected external/fallback exceptions with structured warnings, and let internal logic bugs fail fast.

### Phase 2: Mid-Term (1–2 Months, Structural Debt Reduction)
- [ ] **Typed Contracts First, Then Decompose Top-4 Monoliths**:
  - Introduce lightweight `@dataclass` contracts (`SlotFrame`, `VocabBlueprint`, `SenseLockResult`) per [pending-blueprint-and-distractor-refactoring.md](pending-blueprint-and-distractor-refactoring.md).
  - Decompose `build_precomputed_target_skeletons`, `generate_vocab_distractors`, `_lock_sense`, and `generate_quiz` into focused, composable service helpers while keeping `LinguisticEngine` / `WikiProcessor` as thin backward-compatible facades.
- [ ] **Decouple Dashboard & Shared Quiz Assets**:
  - Split `dashboard.py`'s `handle_api_post` into resource-based route handlers (or migrate to a lightweight ASGI framework if API surface grows).
  - Split `dashboard.js` (4,951 lines) into native browser ES Modules (`api.js`, `state.js`, `views/*.js`) without heavy build tooling.
  - Extract shared `quiz-base.css` and `quiz-core.js` across HTML quiz templates, and have `exporter.py` inline them at export time to preserve standalone offline HTML handouts.

### Phase 3: Long-Term (Scalability & Quality Loop)
- [ ] **Data-Driven Linguistic Rules**: Externalize static lexical tables and syntactic constraint maps from `distractor_engine.py`, `wsd_engine.py`, and `cefr_and_morphology.py` into declarative JSON/YAML or `pedlex.db` tables to curb code bloat.
- [ ] **Golden Set End-to-End Regression Suite**: Curate 3–5 representative units spanning CEFR levels and genres, pair with `re-audit --target extraction`, and track L1/L2 quality deltas across prompt, rule, and model updates.
- [ ] **Agent Skill & MCP Cloud Service Packaging (Pending / Cost Feasibility)**:
  - **Context**: Inbound interest (e.g., Agensi.io marketplace, Claude Code, Cursor/Windsurf tool calling) to package `lexis-wiki`'s neuro-symbolic quiz/wiki generator and psychometric gates as callable remote skills.
  - **Bottleneck & Cost Constraint**: Hosting cloud-accessible endpoints requires dedicated VPS/container servers with Python runtime, `spaCy` transformer models, and ~400MB `pedlex.db` SQLite memory mapping, entailing non-trivial ongoing server rental/maintenance expenses.
  - **Resolution**: Kept as a **Pending Exploration Track**. Prioritize local CLI, Docker self-hosting, and zero-cost local desktop execution first. Cloud marketplace deployment deferred until commercial viability, sponsorship, or subsidized server infrastructure is established.
