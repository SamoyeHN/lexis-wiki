# Lexis Wiki Project Instructions

> **Purpose**: Single source of truth for the Lexis Wiki project — a neuro-symbolic educational knowledge base and interactive assessment generator that transforms primary texts, audio, and video into Obsidian-style wiki nodes and psychometrically audited HTML quizzes.

---

## 1. Neuro-Symbolic Core Architecture

### 1.1 Quad-Tier Separation of Responsibilities
1. **Symbolic Grounding & Lexical Corpus (`LinguisticEngine` + `pedlex.db` + WordNet + `cefrpy`)**:
   - 100% authoritative semantic truth and constraint engine.
   - Combines offline computational NLP (`spaCy` dependency parsing, POS tagging, morphological inflection) with Pedagogical Lexicon (`pedlex.db` + 106k `pedlex_phrase_index` + `pedlex_cxn_index` + `pedlex_error_traps`), WordNet (`oewn:2025+`), and `cefrpy`.
   - Deterministically extracts vocabulary, multi-word expressions, and syntactic patterns; pre-computes single-fit collocational anchors, inflected options, and zero-collision distractors.
2. **Code Gate & Invariant Enforcement (`processor.py`, `evaluator.py`)**:
   - Enforces physical invariants at zero token cost (<5ms): verbatim quotation checks, single-blank constraints (`re.findall(r'_{2,}', stem) == 1`), option cardinality (exactly 4 options), CRC32 deterministic slot assignment, and CEFR lexical ceiling gates.
   - Executes instant programmatic self-healing and category auto-remapping without burning LLM retry tokens.
3. **Pedagogical Prompting (`librarian/prompts/*.md`)**:
   - Focuses strictly on Second Language Acquisition (SLA) standards, contextual stem crafting, and psychometric discrimination.
   - Executes pre-computed `🎯 Micro-Task for LLM` blueprints (for quizzes) or discourse synthesis (for summaries/mind maps) without distractor-inventing burdens.
4. **Schema Enforcement (`schemas.py`)**:
   - Defines strict `snake_case` JSON structures via Prompt-Guided JSON Mode (`format: "json"`).

### 1.2 Architectural Discipline & Theory-First Correction Mandate
- **Zero Tolerance for Band-Aid Bloat**: Never accumulate brittle word-specific regex hacks, ad-hoc `if-else` patches, or arbitrary exception branches.
- **First-Principles Escalation**: Proactively escalate linguistic edge cases to clean neuro-symbolic mechanisms (Top-$K$ symbolic candidate filtering + 1-token LLM arbitration).
- **Mandatory Correction of Flawed Methods (`违背常识与既有学科理论的方法必须主动纠偏`)**:
  - Never blindly execute requests that violate SLA, Conversation Analysis, cognitive load, or psychometric theory.
  - Proactively challenge flawed assumptions (e.g., plotline-stitching in listening) and enforce established theoretical models (Conversation Analysis Three-Part Turn Micro-Structure, Lexical Approach semi-open skeletons, Light/Heavy Turn Alternation, and Cambridge/IELTS `Bait → Real Key` listening signposts).

---

## 2. Sense-First Extraction & WSD Architecture

### 2.1 Deterministic Extraction & Evidence-Completeness Gate ("宁缺勿滥")
- **Deterministic Extraction (`extract_deterministic_vocabulary`, `extract_deterministic_expressions`, `extract_deterministic_grammar`)**:
  - Extracted deterministically from `sources/<Unit>.md` + syllabus via `spaCy` dependency trees and `pedlex.db`.
  - Multi-word expressions are strictly typed (`phrasal verb`, `collocation`, `set phrase`, `idiom`), never labeled with token-level POS. Slash alternatives (`get / feel / be lost`) are normalized prior to `/IPA/` stripping.
- **Three-Gate Quality Pipeline**:
  1. **Gate 1 — Sense Locking**: Locks the exact contextual `Sense` in `pedlex.db` (or WordNet) matching POS, syntactic frame, and semantics.
  2. **Gate 2 — Sense-Isolated Authentic Example Gate (`ensure_distinct_example`)**: Polysemous words must draw authentic examples exclusively from their locked sense's own `examples`/`pattern_bundles`. Monosemous words may use inflection-aware `search_corpus_examples`. Any item with 0 authentic examples for its locked sense is dropped (`[DROP_INCOMPLETE_LEXICON]`). Zero hardcoded Python example/sense dictionaries.
  3. **Gate 3 — Collocational Anchor & Zero Double-Key Gate**: Candidates lacking discriminative context anchors or prone to synonym collisions (e.g. `crossbreed` vs `hybridize`) are dropped before quiz generation.

### 2.2 Two-Stage Neuro-Symbolic Sense Locking (WSD)
1. **Stage 1 — Symbolic Candidate Pool (`pedlex.db` + WordNet, 0ms / 0 tokens)**:
   - Applies POS hard gates (cross-homograph resolution `word_1`..`word_6`), `<gram>` slot constraints (`[only before noun]`, `[uncountable]`), register/attitude gates (`[spoken]`, `[disapproving]`), conditional-subject ontology gates, and syntactic pattern intersection to build a Top-$K$ ($K \le 4$) candidate pool.
   - ~71% monosemous entries lock instantaneously at 0 token cost.
2. **Stage 2 — 1-Token Local LLM Arbitration (~29% Polysemous Cases)**:
   - Formats Top-$K$ ($K \le 4$) senses with lexicographic `[SIGNPOST]` cues and syntactic patterns alongside passage context into a multiple-choice prompt resolved via a 1-token (`A`/`B`/`C`/`D`) local LLM call. Zero free-generation authority.

---

## 3. Modular Sense-First Quiz Blueprint Pipeline

### 3.1 Downstream Markdown Decoupling
- `extractions/<Unit>_{vocabulary,grammar}.md` are user-facing presentation artifacts; assessment pipelines ground directly on `sources/<Unit>.md` via fresh deterministic extraction.

### 3.2 Modular Blueprint Mixins & Pure Five-Source LDOCE Pool
- **Modular Architecture (`librarian/linguistics_core/`)**:
  - `quiz_patterns.py`: Five-source authentic example pre-selection and atomic pattern bundles.
  - `quiz_frames.py`: First-principles verb patient ontology and example-derived cloze slot frame slicing.
  - `quiz_vocab.py`: Target skeleton assembly, option coordination, and universal double-key screening.
  - `quiz_translation.py`: Translation minimal-pair micro-task blueprints and error-injection specs.
  - `quiz_listening.py`: Functional dialogue trajectories and listening comprehension blueprints.
  - `quiz_skeletons.py`: Unified Facade Mixin exposing backward-compatible public APIs.
- **Five-Source Sense-Isolated LDOCE Pool (`select_optimal_five_source_example`)**:
  - Draws candidate examples strictly across 5 LDOCE compartments: (1) `grammar_boxes`, (2) locked `Sense.pattern_bundles` (`gramexa`) & `Sense.examples`, (3) `LDOCE Thesaurus`, (4) `Language Activator`, and (5) curated `collocations` (gated by single-sense or lexical overlap). Oxford Thesaurus and unedited BNC dumps are excluded.
  - Full finite clauses strongly outrank fragments. Attributive vs predicative slot constraints and valency patterns are strictly enforced.
- **Example-Derived Cloze Slot Frame Slicing (`derive_cloze_slot_frame_from_example`)**:
  - Sliced directly from the winning example's syntactic neighborhood (`... their ____ achievement`, `... ____ [concrete physical object] into`). Zero `None` frames, zero unexpanded slash choices.
- **Distractor Architecture & Universal Post-Assembly Double-Key Guard**:
  - Valency cascades: nouns use SLA context clues; transitive verbs prioritize patient object $\gg$ prep; intransitive verbs prioritize prep $\gg$ adv; adjectives enforce attributive (`[only before noun]`) vs predicative (`[not before noun]`) complements.
  - Multi-word phrases use symmetric tail-sharing, particle contrasts, and possessive pronoun & tense synchronization (`come to one's aid` $\rightarrow$ `came to her aid`).
  - Dynamic CEFR fallback pool (`_ensure_corpus_fallback_index`, 8,102 core words) replaces hardcoded fallback lists.
  - Post-assembly screening verifies `double_key_collision` and `evaluate_syntactic_slot_clash` across ALL options against the final context anchor, auto-refilling any colliding distractor.

---

## 4. Deterministic CEFR Calibration & QA Audit

### 4.1 Zero-LLM CEFR Calibration (`cefrpy`)
- Headword and passage CEFR levels (`A1`–`C2`) are computed deterministically via `cefrpy` + Zipf frequency gates; output schemas exclude CEFR fields so LLMs never guess difficulty.

### 4.2 Two-Level QA Quality Audit
1. **Level 1: Deterministic Code Gate (`evaluator.py`, <5ms, 0 tokens)**:
   - Structural invariants, option length bounds, Fisher-Yates shuffling with atomic answer-index remapping, deterministic grammar category remapping.
2. **Level 2: Expert Model Semantic Audit (`expert_auditor.py`, LLM-as-a-Judge)**:
   - Optional (`enable_expert_audit` in `wiki_config.json`). Blind solver resolution with 3-tier triage: Tier 3 `PASS` (80–100), Tier 2 `REPAIR` (60–79), Tier 1 `REWRITE` (20–59).
3. **Delivery Status**: Direct delivery to `extractions/` or `handouts/` with frontmatter `qa_status: "passed" | "review_needed" | "failed"`.

---

## 5. Stateless Storage & Hardware Conventions

### 5.1 Self-Contained Directory Layout
```text
wiki/<UnitName>/
├── sources/              # Primary text (<UnitName>.md) and media/
├── extractions/          # <UnitName>_{vocabulary,grammar,summary}.md & <UnitName>_mindmap.json
└── handouts/             # <UnitName>_{vocabulary,cloze,reading,translation,listening,video}_quiz.html
```
- Folder slug `<UnitName>` is the immutable physical key (`normalize_name()`). Display titles live in YAML frontmatter (`title: "..."`).

### 5.2 Runtime Configuration (`wiki_config.json`)
- **Hot-Reloading**: `config.py` monitors `st_mtime` of `wiki_config.json`.
- **VRAM Concurrency**: $\le 12\text{GB}$ VRAM uses `max_parallel: 1`; $\ge 24\text{GB}$ scales to `max_parallel: 2`.
- **Dual-Distribution Compliance**: All `.db`/`.sqlite` binaries are git-ignored; when `pedlex.db` is absent or `DISABLE_PEDLEX=1`, `LinguisticEngine` falls back to `spaCy` + `cefrpy` + WordNet (`oewn:2025+`).

---

## 6. Technical Documentation & Roadmap

### 6.1 Documentation Index (`docs/`)
- [docs/vocabulary-selection-and-distractor-methodology.md](docs/vocabulary-selection-and-distractor-methodology.md): Single source of truth for candidate extraction, POS resolution, optimal example selection principles across parts of speech, valency cascades, distractor generation, and the 8 lexico-syntactic construction families.
- [docs/pedlex-database-architecture.md](docs/pedlex-database-architecture.md): PedLex SQLite database schemas (`ldoce`, `pedlex_phrase_index`, `pedlex_cxn_index`, `pedlex_error_traps`), JSON payload specifications, and runtime query interfaces.
- [docs/listening-functional-dialogue-architecture.md](docs/listening-functional-dialogue-architecture.md): Four pragmatic interaction paradigms, 14 functional dialogue archetypes, conversation analysis turn structures, and listening blueprints.
- [docs/reading-discourse-architecture.md](docs/reading-discourse-architecture.md): Discourse graph extraction, 6 discourse relation types, 10 psychometric distractor fallacy operators, single-step inference entailment, and verbatim lexical leakage guard.
- [docs/audit-scores-guide.md](docs/audit-scores-guide.md): Level 1 deterministic code gate and Level 2 expert evaluator rubrics.
- [docs/digital-human-characters.md](docs/digital-human-characters.md): Digital human dialogue cast archetypes, facial geometry standards, and multi-character rendering pipelines.
- [docs/esp-trade-logistics-roadmap-and-validation.md](docs/esp-trade-logistics-roadmap-and-validation.md): Technical roadmap, corpus collection checklist, and 6-stage model validation specification for foreign trade and logistics specialized English (ESP).
- [docs/cli.md](docs/cli.md): CLI commands, compilation flags, and dashboard workflows.

### 6.2 Project Roadmap
- [x] **100% Deterministic Extraction & Neuro-Symbolic WSD**: Pure symbolic Top-$K$ + 1-token LLM arbitration.
- [x] **Sense-First Quiz Blueprint Closure**: Five-source LDOCE pool, modular architecture (`quiz_vocab.py` etc.), zero drift, universal post-assembly double-key collision screening verified across Books 1–4 (30 units, 300 items, 100% pass rate).
- [x] **Verb Patient Ontology & Dynamic Fallback Pool**: WordNet supersense + cognitive subdomains for patient slots; 8,102-word dynamic core lexicon fallback.
- [x] **Symbolic Translation & Listening Blueprints**: Minimal-pair error-injection (`translation_quiz`); 14 dialogue archetypes with pragmatic turn structures (`listening_quiz`).
- [x] **Symbolic Reading Discourse Blueprint**: Pre-assign rhetorical span anchors (`spaCy` transitions), 10 deterministic distractor fallacy operators (cross-paragraph bait, causal reversal, modality overgeneralization, explicit restatement), single-step inference entailment, and 3-gram verbatim lexical leakage code gate.
- [ ] **Paired Listening–Speaking Workbench**: Two-act receptive-to-productive pairing (`Act I: Listening` $\rightarrow$ `Act II: Speaking`) with 3-level progressive fading scaffolding (Cue Card $\rightarrow$ Acoustic Cloze $\rightarrow$ Blind Production) and cross-scenario transfer.
- [ ] **Prerequisite Extraction Gate**: Gate quiz generation on verified `extractions/` artifacts to prevent topic drifts.
- [ ] **ESP Specialized Domain Extension**: Multi-source trade/logistics corpus collection, Incoterms/UCP600 legal consistency gates, and domain-gated WSD (`pedlex_esp.db`).
