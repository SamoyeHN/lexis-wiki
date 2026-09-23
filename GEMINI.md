# Lexis Wiki Project Instructions

> **Purpose**: This document serves as the single source of truth for the Lexis Wiki project -- an AI-powered wiki generator that produces Obsidian-style educational content from source materials using a schema-first prompt architecture.

---
## 1. Architecture & Core Philosophy

### 1.1 Separation of Responsibilities
- **Schema = Structure**: `schemas.py` defines output format, required keys, JSON data types, and array constraints via native API structured outputs.
- **Prompt = Pedagogy & Quality**: `.md` prompts focus 100% on educational standards (CEFR/TOEFL), item-writing rules, and distractor engineering—completely free of mechanical JSON formatting instructions or code variable names.
- **Code = Invariant Enforcement**: Deterministic Python logic enforces physical invariants, normalization, and bounds at zero token cost.

### 1.2 Generation Pipeline Modes
- **Extraction (Vocabulary / Expressions / Grammar)**: Defaults to **One-Shot** (`enable_vocab_prose: false`, `enable_grammar_prose: false`). One-Shot provides 80% faster generation and superior verbatim grounding for text-extraction tasks.
- **Assessment / Quiz Generation**: Defaults to **Two-Turn Decoupled Pipeline (Prose-to-JSON)** (`enable_prose_pipeline: true`):
  - Two stateless, independent HTTP calls:
    1. $\text{Call}_1$: $\text{Source Text} + \text{Pedagogical Prompt} \longrightarrow \text{Prose Draft}$ (native thinking/reasoning without JSON constraints).
    2. $\text{Call}_2$: $\text{Packaging Prompt} + \text{Turn 1 Draft} + \text{Source Text} \longrightarrow \text{Strict JSON}$ (fast temperature-0 packaging).
  - *Benefits*: Prevents VRAM exhaustion and KV cache fragmentation on consumer GPUs ($\le$ 12GB), shields JSON structure from prose contamination, and purges reasoning tokens before structured serialization.
- **Structured Output Strategy**:
  - **Prompt-Guided JSON Mode (`format: "json"`, Default)**: Automatically injects programmatic JSON Schema derived from `schemas.py` into the system prompt. Eliminates GBNF grammar parser stalls, tokenizer conflicts, and CPU-bound token-masking timeouts while maintaining 100% schema fidelity.
  - **Strict Schema Mode (`enforce_gbnf: true`)**: Used for engines with hardware-accelerated grammar transducers. If empty tokens are returned, `llm.py` automatically falls back to `format: "json"`.
- **True Multi-Turn Sessions (Self-Correction Retries Only)**:
  - Multi-turn conversational appending (`conversation.append(...)`) is reserved exclusively for the QA Self-Correction Retry Loop in `llm.py` when validation fails or QA score $< 80$.

### 1.3 Linguistic Grounding & Extraction Mandates
- **Zero Concrete Examples in Definition Prompts (Eliminating Example Contamination)**:
  - Grammatical categories and schema fields are defined purely through abstract linguistic functions without concrete vocabulary examples.
  - Empirical finding: Concrete prompt examples hijack self-attention, prompting models to hallucinate or force-fit non-existent structures.
  - Prompts enforce structural pipeline auditing: `AUDIT: 'Physical Marker in Quote' -> [Category] -> [Syntactic Slot Formula]`.
- **Four Macro Functional Domains for Grammar**:
  1. **`Rhetoric & Emphasis`**: Symmetry, rhythmic balance, fronted inversion, or cleft focus.
  2. **`Cohesion & Framing`**: Abstract shell nouns, complement that-propositions, and anaphoric discourse encapsulation (including `, which + [interpretive verb: means/meant, suggests/suggested] + that`).
  3. **`Information Packaging`**: Non-finite participial adjuncts, dense nominalizations, evaluative dummy-it extrapositions, and elaborative relative clauses.
  4. **`Logic & Stance`**: Conditionals, concessive refutations, and calibrated epistemic stance/hedging.
- **Prohibition Outlet Principle & Quality Over Quota**:
  - Every strict negative constraint provides an explicit permissible outlet (traffic routing to legal categories, graceful skip, or surgical rewrite).
  - Quotas are evidence-driven: models never force artificial category distributions. Items cluster naturally based on source text evidence.

---

## 2. Stateless Storage & Decoupled Display Architecture

### 2.1 Self-Contained Unit Layout
Units reside in self-contained folders under `wiki/`:
```
wiki/<UnitName>/sources/<UnitName>.md | media/
              /extractions/<UnitName>_{vocab,grammar,summary,mindmap}.md
              /handouts/<UnitName>_[type]_quiz.html
```
All lookups use `normalize_name()` (case-insensitive, ignoring spaces and special characters).

### 2.2 Stateless Rules & Media Preservation
- **No Global Raw Dirs & Zero Background Scans**: `raw/` is deprecated. Zero background scans or self-healing file moves exist; operations occur strictly through explicit CLI commands (`lexis compile`) or dashboard actions.
- **Zero Metadata Files**: `unit.json` is completely eliminated.
- **Display Layer Decoupling**:
  - Physical folder name (`unit_name` / slug) acts as the immutable filesystem primary key for all routes, API endpoints, and artifacts.
  - Primary source frontmatter (`title: "..."`) provides the dynamic display title.
  - Renaming course titles or lesson headers never triggers cascading filesystem operations or broken media links.
- **Media Organization**:
  - Standalone video unit: video in `sources/media/<UnitName>.<ext>`, transcript at `sources/<UnitName>.md`.
  - Supplementary video: both video and transcript reside in `sources/media/` to protect the primary text source.

---

## 3. Configuration & Runtime Architecture

### 3.1 Live Configuration Hot-Reloading (`config.py`)
- `config.py` monitors the modification time (`st_mtime`) of `wiki_config.json`. Configuration changes apply immediately across CLI and Web Dashboard without restarting the server.

### 3.2 Audit & Generation Controls (`wiki_config.json`)
- `enable_expert_audit`: Master switch for Level 2 LLM-as-a-Judge semantic audit (default: `false`).
- `judge_model`: Independent evaluation model (should differ from generation `model`).
- `num_ctx`: Model context window size in tokens (default: `16384`).
- `max_tokens`: Maximum generation token budget (default: `8192`).
- `enable_prose_pipeline`: Multi-turn prose-to-JSON for quizzes (default: `true`).
- `enable_vocab_prose`: Prose pipeline for vocabulary (default: `false`).
- `enable_grammar_prose`: Prose pipeline for grammar (default: `false`).
- **Transparent Delivery (No Quarantine)**:
  - All content delivers directly to `extractions/` or `handouts/` under its canonical unit slug, eliminating isolated quarantine folders (`_quarantine/`).
  - **3-Tier Frontmatter Status**:
    - `qa_status: "passed"` ($\ge 80$): Complete delivery ready for classroom instruction.
    - `qa_status: "review_needed"` ($60–79$): Delivered with full content, flagged for optional teacher review.
    - `qa_status: "failed"` ($< 60$ or fatal defects): Frontmatter records score, failure status, and reasons, while defective content items are safely blocked from rendering to prevent curricular contamination.

### 3.3 Hardware Alignment & Concurrency Protocols
- **Default Serialization (`max_parallel: 1`)**:
  - Consumer GPUs ($\le$ 12GB VRAM, e.g., RTX 3060/4070): strictly `max_parallel = 1` on both client and engine (Ollama `OLLAMA_NUM_PARALLEL=1`, LM Studio 1 slot). Guarantees 100% layer GPU offload and allocates remaining VRAM exclusively to the active KV cache.
  - Large GPUs ($\ge$ 24GB VRAM): safely scale to `max_parallel = 2` with server slots set to 2.
- **TTS Service Defaults**:
  - Kokoro: `http://localhost:8880/v1/audio/speech` (default voices: `af_sarah` / `am_michael`).
  - Edge-TTS: `http://localhost:5050/v1/audio/speech` (default voices: `en-US-AriaNeural` / `en-GB-RyanNeural`).

---

## 4. Two-Level QA Quality Audit Architecture

```
[ Generation Phase (Fast Generation Model) ]
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Level 1: Deterministic Code Gate (Structural Audit)    │
│  - Zero token cost, instantaneous (<5ms) python logic  │
└────────────────────────────────────────────────────────┘
                    │
                    ├─► [❌ Defect: Schema violation, target desync, category mismatch]
                    │       └─► Instant programmatic repair / auto-remap
                    ▼ [✅ PASSED]
┌────────────────────────────────────────────────────────┐
│ Level 2: Expert Model Quality Audit (Semantic Audit)   │
│  - High-intelligence evaluator (LLM-as-a-Judge)        │
│  - Independent Blind Solver Resolution                 │
└────────────────────────────────────────────────────────┘
                    │
                    ├─► [❌ Defect: Double keys, ambiguous stems, trivial distractors]
                    │       └─► Targeted surgical cure or complete rewrite
                    ▼ [✅ PASSED]
          [ Final Content Delivery / HTML Handout ]
```

### 4.1 Level 1: Deterministic Code Gate (`evaluator.py`)
- **Physical Invariants**: Verbatim quotation verification, single-blank constraints (`re.findall(r'_{2,}', stem) <= 1`), option bounds (exactly 4 non-empty options), and headword-answer key synchronization.
- **Deterministic Category Auto-Remap & Soft Penalty**:
  - When sentences exhibit incontrovertible physical markers (e.g. Antithesis `not... but...` misclassified as `Logic & Stance`, or Propositional Encapsulation `, which + [interpretive verb: means/meant, suggests/suggested] + that` misclassified as `Information Packaging`), code deterministically remaps `category` to the correct macro domain.
  - Deducts a soft penalty (-2.0 pts per item) from the pedagogy score, recording an audit trace while avoiding expensive 30s LLM retry loops.
- **Option Sanitization & Fisher-Yates Shuffling**: Strips redundant prefixes (`A.`, `B.`) and shuffles options with atomic index remapping to eliminate positional key bias.

### 4.2 Level 2: Expert Model Semantic Audit (LLM-as-a-Judge)
- **Universal Constraints**:
  - **Cardinality Closure**: Judge evaluates all $N$ items without early truncation.
  - **3-Tier Triage Mapping**:
    - **Tier 3 (`PASS`, 80–100, `single_fit_valid: true`)**: `cured_question` is null.
    - **Tier 2 (`REPAIR`, 60–79, `single_fit_valid: true`)**: Minimal-invasive surgical repair preserving original stem and context while replacing flawed distractors.
    - **Tier 1 (`REWRITE`, 20–59, `single_fit_valid: false`)**: Unsolvable defect; item discarded and regenerated.
  - **High Token Budget**: `num_predict: 16384` allocated to prevent incomplete distractor evaluations.

---

## 5. Naming & Formatting Conventions

- **JSON Keys & Schema Fields**: Use `snake_case` strictly across all payloads (`schemas.py`).
  - Mandatory keys: `concepts` (not `items`), `part_of_speech` (not `pos`), `explanation`, `target_language`.
- **Markdown & Frontmatter**:
  - Extracted vocabulary `word` and grammar `name` wrapped in double brackets `[[ ]]`.
  - Concept connections formatted as `[[Linked Concept]]`.
  - YAML frontmatter must include `source: "[[filename.md]]"`, `category`, and title.

---

## 6. To-Do List

> ✅ Complete | 🔄 In Progress | ⬜ Pending

### Completed
- [x] Embed raw audio data (Base64) into HTML handouts
- [x] Unit Dependency Graph visualization
- [x] Video Quiz Extension (Bilibili/MP4 via Whisper)
- [x] pyproject.toml dependency management
- [x] Two-turn decoupled Prose-to-JSON quiz generation pipeline
- [x] Level 1 Deterministic Code Gate and Level 2 Expert Model Semantic Audit across all quiz modalities
- [x] Psychometric targeted healing (selective item regeneration and surgical splicing)
- [x] Display layer decoupling (display titles decoupled from physical slugs)
- [x] Live configuration hot-reloading via filesystem mtime tracking
- [x] Clean prompt & schema separation (removal of code variables and formatting from prompts)
- [x] Elimination of example contamination (pure syntactic definitions replacing concrete prompt examples)
- [x] Four Macro Functional Domains & Triad Quality Architecture for grammar extraction
- [x] Deterministic grammar category auto-remap with soft penalty scoring
- [x] Expanded tense and inflection support for interpretive verbs (`meant`, `suggested`, `indicated`, `showed`)
- [x] Dual-Track Assessment Track 1: Authentic Passage Cloze (0-token authentic stem masking and precomputed zero-collision distractors)
- [x] In-place deterministic slot standardization (`[sb]`, `[sth]`, `[one's]`) with spaCy `poss` dependency binding
- [x] Streamed intra-paragraph sentence tokenization (`[S-ID]`) preserving authentic markdown paragraph boundaries
- [x] Elimination of redundant candidate quotes in target skeletons; inverted cognitive flow (Passage first, Targets second)
- [x] Restored authentic full sentence requirement for `quoted_sentence` and `quote`, restricting `[S-ID]` exclusively to `design_audit` tag chains
- [x] Level 1 deterministic self-healing for unoriginal/copied `example_usage` and empty extraction fallback from sentence pool
- [x] Quiz prompt lean refactor: tag-chain `design_audit` across all 5 quiz modalities (reading, listening, vocab, translation, video)
- [x] Fixed grammar extraction prompt variable leakage (P0-1) across deterministic pattern targets
- [x] Deterministic Target Coverage Gate & Formula Grounding in Evaluator (P0-2: proportionate score scaling, incomplete coverage fatal flag, and formula-quote physical grounding)
- [x] Unified Transparent Delivery & Frontmatter Safety Blocking (P0-3: 3-tier status triage, frontmatter-only error reporting with body safety blocking on failed extractions, elimination of quarantine directory confusion)

### Pending
- **Primary Focus: Source-to-Wiki Extraction Quality**:
  - Maximize precision and pedagogical rigor across vocabulary, expressions, and grammar extractions prior to quiz generation updates.
  - Enforce single-word contextual discipline and authentic syntactic slot binding.
- Deterministic overall_cefr_level labelling
  - apply to word level(vocabulary/expression), sentence level(grammar/quiz/reading), passage level(reading)
- **Deterministic Code Substitution Roadmap**:
  - Prioritize code-based deterministic enforcement over LLM prompting for repetitive, rule-bound tasks:
    1. **COBUILD Slot Formula Normalization**: Enforce closed symbol mappings (e.g., normalize `[sb]` / `[someone]` to standard slot representations and auto-close brackets).
    2. **CEFR / Frequency Objective Lookup**: Integrate lightweight offline reference lexicons (CEFR-J, Oxford 3000/5000, AWL) for sub-millisecond, objective proficiency categorization.
    3. **Distractor Sanitization & Position Shuffling**: Automated option collision prevention, prefix stripping, and Fisher-Yates position randomization with atomic key synchronization.
    4. **Common Mistakes Template Injection**: Maintain curated pedagogical defect templates for core grammatical structures, enriching generic or repetitive model explanations.
    5. **Quote Boundary Magnetic Snapping**: Leverage the indexed sentence pool to auto-snap partial quotes to pristine, authentic source sentences.
- **Code-Driven Lexical Pipeline & Zero-Double-Key Distractor Assembly (WordNet & Collocation Base)**:
  - *Context*: Rather than burdening the LLM with complex prompt instructions to engineer distractors, prevent synonym collisions, and balance prepositions, the code directly pre-computes valid, mutually exclusive options using lexical databases (WordNet / Academic Collocation List).
  - *Core Capabilities*:
    1. **Pre-Computed Distractor Synthesis**: Code queries WordNet for strict antonyms, taxonomy siblings, and distinct semantic categories.
    2. **Dependent Preposition Collocational Clashing**: Code selects distractors that are grammatically incompatible with the target sentence's post-blank preposition (e.g., target `rely (+ on)` vs distractors `trust` (transitive), `believe (+ in)`), physically guaranteeing a zero double-key environment at zero token cost.
    3. **LLM Task Simplification (Context Generation Only)**: The LLM's role is stripped of distractor generation and reduced purely to its greatest strength: generating authentic academic context stems containing `____ [anchor prep]`.
- **Level 1 In-Place Self-Healing & Phase-Out of LLM Retry Loops**:
  - *Context*: Small models (8B–12B) exhibit confirmation bias and lack deep functional grammar reasoning in multi-turn dialogues (agreeing with whatever category is suggested in conversational turns). LLM retry loops are therefore eliminated.
  - *Design*: Level 1 is transformed into a deterministic self-healing gate:
    1. **Deterministic Category Auto-Remap**: Incontrovertible structural markers (inverted subject-aux, expletives, antithesis `not... but...`, shell nouns) trigger direct programmatic reassignment of `category` in memory (<0.01ms).
    2. **Canonical Lemmatization & Boundary Snapping**: In-place replacement of inflected words and partial quotes via spaCy dependency trees and indexed sentence pools.
    3. **Elimination of Multi-Turn Retries**: Generates content strictly in a single pass (One-Shot for extraction, single-pass generation for quiz stems).
- **Adjust functions of Level 2 Judge Model**:
  - *Context*: When distractors, single-fit validity, verbatim grounding, and category assignments are mathematically guaranteed by WordNet, spaCy, and Level 1 Code Gates, Level 2 LLM-as-a-Judge semantic audits become redundant.

- **Quality Tier Routing & Human Review UI**:
  - 90–100: Automatic delivery (Passed - High Quality).
  - 75–89: Automatic delivery flagged as Review Candidate.
  - 60–74: Routed to Human Review Queue.
  - < 60: Precision retry loop; permanent unresolve flag if retry limit exceeded.
- **Multilingual User Interface (i18n)**:
  - Implement lightweight client-side i18n dictionary decoupling UI strings.
  - Default to concise, professional academic English, with seamless one-click switching to Simplified Chinese and other languages.
- **UI/UX & Visualization**:
  - Knowledge graph enhancements: legend, zoom, filter, export, and clickable side-preview nodes.
  - Dashboard hub with category grouping and breadcrumb navigation (Bootstrap 5 CDN).