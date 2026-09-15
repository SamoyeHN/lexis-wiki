# Lexis Wiki Project Instructions

> **Purpose**: This document serves as the single source of truth for the Lexis Wiki project -- an AI-powered wiki generator that produces Obsidian-style educational content from source materials using a schema-first prompt architecture.

---
## 1. Prompt Architecture

Prompts drive pedagogical quality and assessment design; JSON schemas enforce output structure.

- **Clean Separation of Responsibilities**:
  - **Schema = Structure**: `schemas.py` defines output format, required keys, JSON data types, and array constraints via native API structured outputs.
  - **Prompt = Pedagogy & Quality**: `.md` prompts focus 100% on educational standards (CEFR/TOEFL), item-writing rules, distractor engineering, and `MANDATE:` rules—completely free of mechanical JSON formatting instructions or code variable names.
- **Dual-Mode Structured Output Architecture**:
  - **Prompt-Guided JSON Mode (`format: "json"`, Default)**: When calling local engines (Ollama), `format: "json"` is active (`"enforce_gbnf": false` in `wiki_config.json`). The system automatically injects the JSON Schema derived programmatically from `schemas.py` into the system role. This eliminates GBNF grammar parser stalls, tokenizer conflicts (e.g. 131k/248k tokenizers in Nemotron/Gemma/Qwen), and CPU-bound token-masking timeouts while maintaining 100% schema fidelity.
  - **Strict GBNF / Native Schema Mode**: Supported via `"enforce_gbnf": true` in `wiki_config.json` for engines with hardware-accelerated grammar transducers (e.g., OpenAI `json_schema` strict mode).
  - **Automatic Empty-Output Fallback**: If strict GBNF mode fails or returns empty tokens, `llm.py` automatically catches the failure, logs a warning, and retries seamlessly in `format: "json"`.
  - **Two-Turn Decoupled Generation Pipeline (Prose-to-JSON)**:
    - *Turn 1 (Prose Drafting)*: Unconstrained text generation with full reasoning and thinking capability, focusing 100% on linguistic depth, complex CEFR sentence framing, and 3-vector distractor traps without JSON token constraints.
    - *Turn 2 (Deterministic Packaging)*: Fast, temperature-0 packing that transforms the prose draft into strict JSON conforming to `schemas.py`.
  - **Expert / Judge Model Architecture**:
    - The generation model drafts content (via Prose-to-JSON or One-Shot).
    - The expert judge model executes independent blind solving, semantic auditing, and minimal-invasive surgical repair/rewriting. Keeping the generation model and the audit judge distinct preserves assessment rigor and catches double-key divergence.

- **Human-Readable Logging**:
  - `librarian/logger.py` automatically indents and formats `--- RAW RESPONSE ---` with 2-space pretty-printed JSON in all task logs under `logs/`.

---

## 2. Self-Contained Unit Structure

Each unit is a self-contained folder under `wiki/`. All lookups use `normalize_name()` (case-insensitive, ignores spaces/special chars).

### 2.1 Directory Layout

```
wiki/<UnitName>/sources/<UnitName>.md | media/ 
              /extractions/<UnitName>_{vocab,grammar,summary,mindmap}.md
              /handouts/<UnitName>_[type]_quiz.html
```

### 2.2 Stateless Rules

- **Deprecation of Global `raw/` and `raw/media/` Directories**.
- **Zero Background Scans / No Self-Healing**: There are no automatic, implicit, or startup scans running in the background to self-heal or move files. Content is managed and organized strictly through explicit compilation commands (`lexis compile`) or explicit dashboard upload/compile buttons.
- **No metadata files (Removal of `unit.json`).** The architecture is 100% stateless. `unit.json` has been completely eliminated from the system. Unit title = formatted folder name (e.g., `Book_4_Unit_1` → `Book 4 Unit 1`).
- **Active media** = newest file in `sources/media/` by mtime; dashboard click updates mtime.
- **Video Handout Validation**: The `/api/check-video-source` endpoint robustly validates video units by checking both (1) companion transcript markdown files in the unit's `sources/media/` directory, and (2) primary standalone transcript files directly inside the `sources/` directory (matching `sources/<UnitName>.md` which contains the video's YAML frontmatter).
- **Media & Supplementary Layout**:
  - **Standalone Video Unit**: Video file is saved under `sources/media/<UnitName>.<ext>`, and transcript `.md` is saved as the primary source at `sources/<UnitName>.md`.
  - **Supplementary Video (Existing Unit)**: Both the video file (e.g., `Three_Gorges_Dam.mkv`) and its transcript `.md` (e.g., `Three_Gorges_Dam.md`) are saved under `sources/media/` to protect the primary text source (`sources/<UnitName>.md` or `.txt`) from overwrite.
- **Library Sorting (Recency-First)**: The Document Library automatically displays files sorted by modification time (`mtime`) descending when the default `'recent'` sorting option is active. This places newly uploaded, created, or edited units first, directly next to the "Add Source" card.
- **Instant Workspace Transition on Completion**: The poller instantly opens the workspace and auto-switches to the **Raw Source** tab the moment background transcription completes.
- **Commands**: `lexis rename <old> <new>` renames unit + all internals.

### 2.3 Display Layer Decoupling (Display Title vs. Physical Slug)

- **Principle**: Decouple user-facing display titles (`f.title`) from the underlying physical filesystem identifiers (`unit_name` / `slug`).
- **Stable Physical Primary Keys**:
  - The folder name (e.g., `Book_4_Unit_1`) and primary filename (e.g., `Book_4_Unit_1.md`) serve as immutable physical anchors/slugs.
  - All internal linkages, API endpoints (`/api/compile`, `/api/generate-quiz`), frontend state routing (`currentUnit`), and derived artifacts (`extractions/`, `handouts/`) strictly use the physical slug. This eliminates broken wikilinks, browser 404s, stale cache issues, and media path desynchronization.
- **Dynamic Display Title Extraction**:
  - The primary source file (`sources/<UnitName>.md`) defines the display title via its YAML frontmatter `title: "..."` (e.g., `title: "Unit 1: The Road to Success"`).
  - The backend (`/api/raw-files`) automatically parses this frontmatter title and delivers it as `title`.
  - The Dashboard UI (card titles, list view, sidebar tree, breadcrumb nav, workspace header, name sorting) prioritizes `f.title`, gracefully falling back to the formatted folder name (`stem.replace(/_/g, ' ')`) when no YAML title is present.
- **Zero Renaming Side-Effects & Media Protection**:
  - Users can freely edit lesson titles, add subtitles, or adjust course names in Markdown without triggering risky cascading filesystem renames.
  - Media files (video and audio in `sources/media/`) remain permanently intact with their original filenames.

---

## 3. Configuration & Runtime Architecture

### 3.1 Live Configuration Hot-Reloading (`config.py`)
- **Zero-Restart Disk Synchronization**: `config.py` monitors the filesystem modification timestamp (`st_mtime`) of `wiki_config.json`.
- Any external edits (e.g. toggling `enable_prose_pipeline`, switching `model`, updating `judge_model`, or tuning `enable_expert_audit`) are instantly and silently hot-reloaded into memory across both CLI and the running Web Dashboard without requiring server restarts.

### 3.2 Audit & Generation Controls (`wiki_config.json`)
- `enable_expert_audit`  : master switch for Level 2 LLM-as-a-Judge semantic audit (default: `false`)
- `judge_model`          : L2 judge model; SHOULD differ from generation `model` (e.g., `gemma4:12b` or `phi4:14b`)
- `min_passing_items`    : per-quiz passing floor after triage and discards (default: `3`)
- `quarantine_on_fail`   : route failed artifacts to `_quarantine/` (default: `true`)
- `enable_prose_pipeline`: multi-turn prose→JSON generation toggle (default: `true`)
- `enable_vocab_prose`   : prose pipeline toggle for vocabulary extractions (default: `false`)

### 3.3 Reasoning & Thinking Parameter Protocols
- Models featuring internal reasoning channels (e.g., `gpt-oss:20b` dual-channel `<|channel|>analysis` vs `<|channel|>final`) operate under Ollama's native routing.
- The system avoids imposing artificial token masks (`think: false`) that trigger reasoning spillover into output text; clean channel separation is maintained natively.

### 3.4 Dashboard Input Configuration & TTS Defaults
- **Autofill Prevention**: Passwords require `autocomplete="new-password"`; URL fields require `autocomplete="url"`, `inputmode="url"`.
- **TTS API Endpoints**:
  - Kokoro: `http://localhost:8880/v1/audio/speech`
  - Edge-TTS: `http://localhost:5050/v1/audio/speech`
- **Voice Validation**: Invalid voices self-heal to defaults: Kokoro (`af_sarah`/`am_michael`), Edge-TTS (`en-US-AriaNeural`/`en-GB-RyanNeural`).

---

## 4. Naming Conventions

Use `snake_case` for all JSON keys and variable names.

### Must-use (per schemas.py)
- **`concepts`** (not `items`) -- semantic concept topics
- **`part_of_speech`** (not `pos`) -- lexical category
- **`explanation`** -- pedagogical reasoning across all quiz types
- **`target_language`** (not `language`) -- translation target lang

---

## 5. To-Do List

> ✅ Complete | 🔄 In Progress | ⬜ Pending

### Completed
- [x] Embed raw audio data (Base64) into HTML handouts
- [x] Unit Dependency Graph visualization
- [x] Video Quiz Extension (Bilibili/MP4 via Whisper)
- [x] pyproject.toml dependency management
- [x] QA: Evaluation schema (faithfulness/completeness/pedagogical/schema adherence) + retry loop at <80% threshold
- [x] Level 2 Expert Model Quality Audit (LLM-as-a-Judge pedagogical quality audit, blind quiz test solver, contextual appropriateness & distractor trap validation, multi-turn self-correction loop)
- [x] Multi-turn Prose-to-JSON quiz generation pipeline (Turn 1 unconstrained prose drafting with native reasoning/thinking parameters; Turn 2 deterministic packaging without token stalls)
- [x] Selective Item Regeneration & Surgical Splicing (Psychometric targeted healing: only flagged defective items are regenerated and spliced into untouched locked items)
- [x] Display Layer Decoupling (Human-facing display titles decoupled from immutable filesystem slugs while preserving zero-side-effect media protection)
- [x] Two-Level Auditing Across All Quiz Types: Completed Level 1 (Deterministic Code Gate) and Level 2 (LLM-as-a-Judge Expert Semantic Audit) verification pipeline across Vocabulary, Translation, Reading, Video, and Listening quizzes.
- [x] Expert Model Audit-Repair-Rewrite Pipeline: Direct surgical cure and rewriting by the expert model replacing slow multi-attempt blind regeneration loops.
- [x] Configuration Disk Hot-Reloading: Instant runtime synchronization with `wiki_config.json` via file mtime tracking without dashboard restarts.
- [x] Clean Prompt & Schema Separation: Stripped code variable names and JSON formatting instructions from prompt templates, guaranteeing 100% compatibility across Prose-to-JSON and One-Shot modes.

### Pending
- **Vocabulary Extraction Expert Audit & Item Filtering**:
  - *Context*: 在 JSON 模式下，模型偏向于根据材料中的所有词汇尽量穷尽提取（如从 10 题扩张到 20+ 题），覆盖面广但教学焦点容易发散。
  - *Design*: 结合 Expert Mode，在 `vocabulary extraction` 阶段引入 Expert Audit，由高智能专家模型从提取列表中筛选出最具代表性、教学权重最高的核心重点词汇（Curated Lexical Subset），从源头上精准供给 Quiz Generator，兼顾出题速度与核心教学覆盖。
- **Quality Tier Routing & Human Review UI**:
  - 90–100: 自动交付（Passed - High Quality）。
  - 75–89: 自动交付但标注审查候选（Review Candidate）。
  - 60–74: 进入待人工抽检队列（Needs Human Review）。
  - < 60: 触发精准回炉重试；重试超限后打上未解决标记。
- **Multilingual User Interface (i18n)**:
  - *Context*: 仪表板与各编辑模态框（如 Source & Syllabus Editor、工作区与配置面板）需支持国际化与本地化多语言切换。
  - *Design*: 建立轻量级客户端 i18n 资源字典与切换机制，将界面文字解耦键值化。默认界面全面采用简洁、地道的专业学术英语（Concise English UI），杜绝双语生硬混杂，并支持一键无感切换简体中文及其他语种。
- **UI/UX**: HTML HUB, wiki file list with categories, breadcrumb nav, Bootstrap 5 CDN only.
- **Visualization**: Knowledge graph legend/zoom/filter/export, clickable nodes with side preview.

---

## 6. Two-Level QA Quality Audit Architecture

The system enforces quality through a strict two-tier verification pipeline:

```
[ Generation Phase (Fast Generation Model) ]
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Level 1: Deterministic Code Gate (Structural Audit)    │
│  - Zero token cost, instantaneous (<5ms) python logic  │
└────────────────────────────────────────────────────────┘
                    │
                    ├─► [❌ FAILED: Schema violation, missing keys, target desync, duplicate options]
                    │       └─► Triggers immediate structural repair / normalization
                    ▼ [✅ PASSED]
┌────────────────────────────────────────────────────────┐
│ Level 2: Expert Model Quality Audit (Semantic Audit)   │
│  - High-intelligence evaluator (LLM-as-a-Judge)        │
│  - Independent Blind Solver Resolution                 │
└────────────────────────────────────────────────────────┘
                    │
                    ├─► [❌ FAILED: Hallucination, ambiguous stem, invalid distractors, double keys]
                    │       └─► Executes Minimal-Invasive Surgical Cure or Full Rewrite
                    ▼ [✅ PASSED]
          [ Final Content Delivery / HTML Handout ]
```

### Level 1: Deterministic Code Gate (Structure & Physical Truth)
Pure Python validation ensuring structural completeness and absolute invariants:
- **Schema & Array Bounds**: Strict enforcement of required keys, field types, and exact item counts (e.g., exactly 4 options per quiz question).
- **Physical Ground Truth Anchors**:
  - `target_word` strictly matches `options[correct_answer_index]`.
  - Quoted text/sentences exist verbatim in source material.
  - Option uniqueness (no duplicate options).
- **Answer Distribution & Option Integrity**:
  - Options automatically shuffled and randomized by deterministic Python logic.
  - Explanation option labels (`Option A/B/C/D`) are atomically remapped upon option shuffling or index repair.

### Level 2: Expert Model Quality Audit (Content & Pedagogical Correctness)
High-reasoning semantic evaluation focusing on linguistic rigor:
- **Blind Solver Test**: The expert model independently solves the item without access to declared answers. Divergence indicates ambiguous stems or insufficient textual evidence.
- **Absolute Single-Fit Validity**: Verifies that the correct answer is the ONLY defensible choice while all 3 distractors are objectively and conclusively eliminated.
- **Cognitive Distractor Trap Quality**: Validates that distractors represent authentic educational traps (e.g., Chinglish L1 negative transfer, Scope Shift, Speaker Attribution) rather than trivial or absurd giveaways.
- **Factuality & Faithfulness**: Verifies that conclusions are fully warranted by the provided context or timestamped transcript evidence.

### 6.1 Modality-Specific Two-Level Audit Coverage

- **Vocabulary Quiz**:
  - **Level 1**: Strict Single Blank Gate (`re.findall(r'_{2,}', stem) <= 1`), target word exact synchronization with `options[correct_answer_index]`, verbatim quotation integrity, single-word exclusivity.
  - **Level 2**: Blind solver resolution, collocational precision, POS distractor trap legitimacy, elimination of valid alternative near-synonyms.
- **Translation Quiz (Comparative Translation Appraisal)**:
  - **Level 1**: Target sentence presence, target keyword synchronization inside idiomatic translation, dual distinct candidate versions (Idiomatic vs Flawed), flaw type labeling, and correct option index bounds.
  - **Level 2**: Blind solver resolution, pragmatic and idiomatic academic naturalness, contrastive defect legitimacy (L1 Chinglish transfer, collocation/preposition clash, formula breakdown), and single defensible translation target.
- **Reading Comprehension Quiz**:
  - **Level 1**: Text snippet verbatim anchoring, option count invariants, complete explanation coverage across all 4 options.
  - **Level 2**: Blind solver inference validation, strict rejection of scope shifts / extreme modifiers / false attribution distractors, proof that the declared key is uniquely and incontrovertibly supported by passage evidence.
- **Video Comprehension Quiz**:
  - **Level 1**: Option bounds (strictly 4 options, non-empty, unique), correct index check [0-3], timestamp format validation (`[MM:SS]` or `[HH:MM:SS]`), and transcript chronological anchor verification.
  - **Level 2**: Blind solver validation with timestamped transcript, verification of genuine video evidence at target timestamp (rejection of timestamp hallucination and trivial number recall), single-fit proof, and video distractor trap analysis (cross-timestamp shift, rumor vs fact, over-generalization).
- **Listening Comprehension Quiz**:
  - **Level 1**: Dialogue script turn bounds (>= 4 speaker turns), option count invariants (strictly 4 options, non-empty, unique), correct index check [0-3], and category validation (`Detail`, `Main Idea`, `Inference`).
  - **Level 2**: Blind solver validation against dialogue script, strict speaker attribution verification (rejection of speaker role swaps between Speaker 1 and Speaker 2), verbatim catch trap analysis, and single-fit proof without subjective conjecture.

### 6.2 Universal Expert Audit Criteria & System Boundaries

To eliminate cognitive divergence, early truncations, and contradictory judging across models, all Level 2 Expert Auditing modules (Vocabulary, Reading Comprehension, Translation, Video, Listening) strictly adhere to four universal system boundaries:

1. **Exhaustive 100% Cardinality Mandate (`total_items` Closure)**:
   - The judge model receives exactly $N$ items and MUST output an evaluation array of length exactly $N$ (`questions[0...N-1]`). Early truncation or skipping to `summary_verdict` is strictly prohibited.
   - Deterministic Code Gate checks: `len(audit_questions) == len(questions)` and issues a warning if a cardinality deficit is detected.

2. **Unified 3-Tier Score-Action Mapping**:
   The judge must strictly align pedagogical scores with surgical triage actions:
   - **Tier 3 (`PASS`, Score 80–100, `single_fit_valid: true`)**:
     - Item is psychometrically robust with unambiguous evidence and authentic distractors.
     - `cured_question` MUST be `null`.
   - **Tier 2 (`REPAIR`, Score 60–79, `single_fit_valid: true`)**:
     - Question stem context and target anchors are fundamentally sound, but localized distractor defects exist (e.g. distractor recycling, weak plausibility, minor typo, misaligned index).
     - **Minimal Invasive Surgical Invariant**: The judge MUST keep the original stem, target keyword/anchor, and valid context completely intact. Only flawed distractors or surface options are refreshed in `cured_question`.
   - **Tier 1 (`REWRITE`, Score 20–59, `single_fit_valid: false`)**:
     - Fatal defect: unsolvable double-keys, ungrounded keys, grammatical breakdown, or core translation fidelity mismatch.
     - Entire item is discarded and rewritten from the target curriculum pool in `cured_question`.

3. **Field Disambiguation & Single Source of Truth**:
   - In Comparative Translation Appraisal, `translated_sentence` is strictly defined as the Non-English source prompt to be translated (e.g. Chinese stem), preventing LLMs from confusing it with English translations.
   - All options in Translation Appraisal are strictly plain English strings (`options: ["...", "..."]`).

4. **Budget & Parameter Safeguards**:
   - High-token budget (`num_predict: 16384`) allocated for local and remote judge models to ensure complete multi-item cognitive distractor justifications without token exhaustion.

---

## 7. Automatic Interlinking (Wikilinks)

The system generates an Obsidian-style wiki. The following interlinking rules apply:

- `librarian/processor.py` automatically wraps extracted vocabulary `word` and grammar `name` in double brackets `[[ ]]` during Markdown formatting.
- Concept `connections` are formatted as `[[Linked Concept]]`.

---

## 8. Frontmatter Traceability

All generated Markdown files must include properly structured YAML frontmatter:

- **Source tracking**: `source: "[[filename.md]]"` to trace back to the original material.
- **Semantic categorization**: Use meaningful categories (e.g., `category: ["vocabulary", "extraction"]`).
- **Consistent titling**: Vocabulary and grammar titles follow a uniform format (e.g, "Book 3 Unit 4").

---

## 9. Validation Over Cleaning

Enforce structure through the schema API rather than post-processing. If the returned JSON is malformed, investigate the schema or prompt alignment before adding cleanup logic.

---

## 10. MCP Tool Integration and Design Guidelines

To maintain visual excellence, security, and accuracy:
- **Bootstrap Reference**: Always use the `context7` MCP server to query and fetch the latest official specifications and best practices for Bootstrap 5 elements (e.g., spinners, classes, grid layout).
- **Web Verification**: Use the `playwright` MCP server to load, interact with, and view generated web pages or local HTML dashboards to verify that interfaces display correctly.
- **Web Design with Stitch**: Utilize the `stitch` MCP server for prototyping and executing advanced design systems and mockups to ensure a premium, modern, and highly polished visual aesthetic.