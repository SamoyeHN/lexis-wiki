# Lexis Wiki Project Instructions

> **Purpose**: This document serves as the single source of truth for the Lexis Wiki project -- an AI-powered wiki generator that produces Obsidian-style educational content from source materials using a schema-first prompt architecture.

---
## 1. Architecture & Core Philosophy

### 1.1 Separation of Responsibilities (Neuro-Symbolic Quad-Tier Architecture)
- **Symbolic Grounding & Lexical Corpus (NLP & Lexicons = Semantic Truth & Constraints)**:
  - `LinguisticEngine` combines offline computational NLP (spaCy syntactic dependency trees, POS tagging, morphological inflection lemmatization) with authoritative lexical databases (WordNet taxonomy, Longman Dictionary of Contemporary English 6th Edition / LDOCE).
  - **Absolute Grounding in LDOCE 6th Edition (Semantic Truth & Pedagogical Collocation Engine)**:
    - The engine is 100% grounded in LDOCE 6th Edition (`ldoce6_essential.db`), whose collocation boxes, syntactic patterns, and sense structures are semantically prioritized by authentic pedagogical salience and frequency. Obscure, physiologically biased, or non-pedagogical mechanical collocations are eliminated.
    - Deterministically pre-computes single-fit collocational anchors and 3 zero-collision distractors at zero token cost (<1ms).
    - Pre-binds parallel inflectional morphology and CRC32 deterministic option positions, generating itemized `🎯 Micro-Task for LLM` constraint blueprints. *(Details see: [docs/ldoce6-database-architecture.md](file:///E:/teacher-wiki/docs/ldoce6-database-architecture.md))*.
- **Code = Invariant Enforcement & Physical Gate**:
  - Deterministic Python logic (`evaluator.py`, `processor.py`) enforces physical invariants, string normalization, blank constraints (`re.findall(r'_{2,}', stem) == 1`), atomic option remapping, and bounds at zero token cost.
  - Acts as the Level 1 Deterministic Code Gate to instantly filter, remap, or self-heal structural defects without burning LLM retry tokens.
- **Prompt = Pedagogy & Context Generation**:
  - `.md` prompts focus 100% on educational standards (CEFR/TOEFL), cognitive depth, and executing embedded micro-tasks.
  - LLM acts as an academic sentence crafter and psychometric discriminator: crafts natural contextual stems fulfilling the exact syntactic constraints, and generates deep pedagogical explanations by quoting target and distractor wordings directly—completely free of mechanical JSON formatting or distractor generation burdens.
- **Schema = Structural Integrity**:
  - `schemas.py` defines output format, required keys, JSON data types, and array constraints via native API structured outputs. Eliminates syntactic drift and structural formatting instructions from system prompts.

### 1.2 Generation Pipeline Modes (Unified One-Shot Direct Structured Output)
- **Extraction & Assessment Pipeline (Unified One-Shot Direct JSON)**:
  - **Empirical Baseline**: Thoroughly validated across 11 benchmark models (8B–27B). One-Shot achieves 100% structural completion, 2.2× faster generation (~30s vs ~78s), and eliminates secondary turn semantic drift, target-word loss, and answer-key misalignment.
  - **Elimination of Prose-to-JSON**: The historical two-turn Prose-to-JSON mode (`enable_prose_pipeline`) has been completely removed. Generating free prose failed to enhance creativity and instead introduced severe token bloat, structural omissions, and packaging discrepancies.
  - **Architecture Strategy**: The LLM focuses exclusively on academic sentence crafting and psychometric discrimination in a single turn, while symbolic constraints (WordNet pre-computed distractors, CEFR physical ceilings, spaCy dependency trees) and Python-level key binding enforce structural validity at zero token cost.
- **Structured Output Strategy**:
  - **Prompt-Guided JSON Mode (`format: "json"`, Default)**: Automatically injects programmatic JSON Schema derived from `schemas.py` into the system prompt. Eliminates GBNF grammar parser stalls, tokenizer conflicts, and CPU-bound token-masking timeouts while maintaining 100% schema fidelity.
  - **Strict Schema Mode (`enforce_gbnf: true`)**: Used for engines with hardware-accelerated grammar transducers. If empty tokens are returned, `llm.py` automatically falls back to `format: "json"`.
- **True Multi-Turn Sessions (Self-Correction Retries Only)**:
  - Multi-turn conversational appending (`conversation.append(...)`) is reserved exclusively for the QA Self-Correction Retry Loop in `llm.py` when validation fails or QA score $< 80$.

### 1.3 Deterministic Code-Driven Extraction Architecture (Source-to-Wiki Paradigm Shift)
- **Vocabulary & Expressions: Complete Migration from LLM to Deterministic Code Extraction (已完成代码化提取)**:
  - **Historical Defect of LLM Extraction**: Prompting LLMs to extract vocabulary and multi-word expressions suffered from severe hallucinations, omission of core curricular headwords, misaligned part-of-speech labeling, and ungrounded CEFR ratings, while consuming thousands of unnecessary generation tokens.
  - **Deterministic Pipeline (`LinguisticEngine` + `cefrpy` + LDOCE 6th Edition)**:
    - **Single-Word Vocabulary**: spaCy POS-tagging, lemmatization, and syntax dependency trees extract candidate content headwords directly from the passage text. Headwords are filtered and calibrated against the CEFR lexical database and LDOCE 6th Edition at 0 token cost, guaranteeing 100% lexical fidelity and exact syntactic alignment.
    - **Multi-Word Expressions & Phrasal Idioms**: Deterministically parsed and mined using dependency trees (verb-particle combinations, prepositional verbs, fixed idioms) and dictionary index lookups, completely eliminating arbitrary or fragmented chunking.
    - **Expression-Type Labelling**: A multi-word headword is never labelled with the part of speech of one of its tokens ('tap into' is not a 'verb'). `LinguisticEngine.classify_expression_type` reads Longman's own filing first (a phrasal-verb block or a PHRASES entry named exactly that unit, slots and slash groups expanded), then a frame opened by a closed-class word, then verb + particle adjacency, and returns one of `phrasal verb`, `collocation`, `set phrase`, `idiom`. The evaluator, the LLM healing pass and the Markdown writer each re-type any plain part of speech that arrives on a multi-word row.
- **Grammar Extraction: Deterministic Code-Driven Extraction (方案 A：务实推荐已落地)**:
  - **Current Status**: Structural pattern matching and syntactic formulas (COBUILD formulas, clausal dependency trees, non-finite adjuncts, cleft sentences, inversion, complex transitives, and extraposition) are deterministically extracted at 0 token cost via `LinguisticEngine.extract_deterministic_grammar()`.
  - **Scheme A Implementation Policy (方案 A 务实落地)**:
    - 统一所有语法警告前缀为 `Grammar Warning (<TOPIC>): ...`，彻底清除与禁止任何带来源偏见或空洞的 `COBUILD Warning` / `LDOCE Grammar Alert` 区分与显示。
    - 公式字面词绝对优先于引文兜底词，避免 `want` 劫持 `but` 等并列转折结构；复合宾语消除假死条件，自适应动词匹配。
    - 保持结构公式与原句句式真实对应，剔除空洞/不符原句结构的无意义后缀。
  - **Ongoing Refinement**: 持续基于大样本回归测试积累句法模式规则，进一步完善复杂修辞层与篇章衔接边界的覆盖。

### 1.4 Linguistic Grounding & Extraction Mandates
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

### 1.5 Unified & Deterministic CEFR Calibration Architecture (`cefrpy` Physical Grounding)
- **Core Philosophy: Separation of Judgement (Code Calculates Semantic Ceiling, LLM Crafts Context)**:
  - Eliminates the systemic defect where LLMs hallucinate or distort CEFR ratings (e.g. forcing A1/A2 daily words like `print`, `program`, `team` into artificial B1+ labels due to prompt/schema biases, or burning token compute asking LLMs to guess reading levels).
  - **Zero CEFR Burden on LLM**: All output schemas (`schemas.py`) across both extraction (`VocabularyItem`, `ExpressionItem`, `GrammarItem`, `SummaryExtraction`, `MindMapExtraction`) and assessment generation (`VocabularyQuiz`, `ReadingQuiz`, `TranslationQuiz`, `ListeningQuiz`, `VideoQuiz`) completely strip out `cefr_level`, `word_cefr_level`, and `overall_cefr_level`.
- **Symbolic Grounding & Physical Computation Layer (`LinguisticEngine` + `cefrpy`)**:
  - **Text-Level Ceiling (`LinguisticEngine.calculate_text_cefr`)**: Statistically evaluates full-text vocabulary distribution, computing weighted frequencies against Oxford 3000/5000 and the CEFR lexical database to deterministically output the passage's overall level (`A1`–`C2`).
  - **Word-Level Exact Grounding (`LinguisticEngine.get_word_cefr`)**: Performs offline, zero-token, exact dictionary lookups for mined headwords, accurately tagging basic words (`print` -> `A2`) without prompt distortion.
  - **Stateless Frontmatter & Metadata Persistence**: Automatically writes `overall_cefr_level: "<level>"` to `extractions/<Unit>_{vocabulary,summary,grammar}.md` and `extractions/<Unit>_mindmap.json`.
- **Downstream Decoupled Dynamic Injection Pipeline (`processor.py`)**:
  - `_load_wiki_data` deterministically resolves the stored `overall_cefr_level` and injects it into downstream prompts as an authoritative design constraint `{cefr_level}`, rather than asking models to classify it.
  - **Vocabulary Quiz (`vocabulary_quiz`)**:
    - Pre-computed WordNet distractors pass through an offline `cefrpy` + Zipf frequency ceiling gate ($Zipf \ge 3.2$ for A1/A2, $3.0$ for B1), physically eliminating obscure or super-advanced distractors (e.g., `conceivableness` on basic words).
  - **Reading Quiz (`reading_quiz`)**:
    - **Dynamic Prompt Interpolation**: Injects `{cefr_descriptor}`, `{question_stem_guidance}`, `{option_complexity_guidance}`, `{skill_distribution_guidance}`, and `{vocab_target_guidance}`. For A1/A2, question stems are direct, options are concise (4–12 words), and skills focus on Detail/Recall & Main Idea.
    - **Level 1 Deterministic Code Gate (`audit_reading_integrity`)**: Enforces option length bounds ($\le 16$ words for A1/A2, $\le 20$ words for B1) and uses `cefrpy` to audit that non-passage option/stem vocabulary never exceeds the passage CEFR ceiling (intercepts C1/C2 words).
  - **Translation Quiz (`translation_quiz`)**:
    - **Dynamic Prompt Interpolation**: Injects `{sentence_complexity_guidance}`, `{target_grammar_guidance}`, and `{flaw_taxonomy_guidance}`. Scales smoothly down from B2–C1 idiomatic appraisal to A1–B1 foundational curriculum sentences with direct clause structures and high-utility vocabulary.
    - **Level 1 Deterministic Code Gate (`audit_translation_integrity`)**: Enforces translation length bounds ($\le 16$ words for A1/A2, $\le 24$ words for B1) and uses `cefrpy` to block C1/C2 obscure words on foundational curriculum items.
  - **Listening Quiz (`listening_quiz`)**:
    - **Dynamic Prompt Interpolation**: Injects `{dialogue_style_guidance}`, `{question_stem_guidance}`, `{option_complexity_guidance}`, and `{skill_distribution_guidance}`. Calibrates dialogue registers and comprehension questions to target level.
    - **Level 1 Deterministic Code Gate (`audit_listening_integrity`)**: Enforces option length bounds ($\le 12$ words for A1/A2, $\le 16$ words for B1) and uses `cefrpy` to prevent off-script C1/C2 distractors.
    - **Post-Processing Invariant Binding**: The deterministically computed `cefr_level` is attached directly onto `quiz_obj` during packaging for display on the interactive handout badge (`CEFR Level ${cefr}`).
  - **Video Quiz (`video_quiz`)**:
    - **Dynamic Prompt Interpolation**: Injects `{question_depth_guidance}` and `{option_complexity_guidance}`.
    - **Level 1 Deterministic Code Gate (`audit_video_integrity`)**: Enforces option length bounds ($\le 14$ words for A1/A2, $\le 18$ words for B1) and checks `cefrpy` difficulty ceilings.
- **Audit Layer (Expert Auditor LLM-as-a-Judge)**:
  - All quiz modalities inject curriculum level `{cefr_level}` and alignment guidelines into expert audit prompts (`expert_audit_*.md`), preventing high-tier evaluator models from penalizing foundational curriculum units with irrelevant academic criteria while ensuring evaluation adheres to the deterministically assigned level.

### 1.6 Pedagogical Lexical Selection, Collocation & Distractor Engine
- **Overview & Separation of Concerns**:
  - The lexical selection and distractor generation pipeline decouples pedagogical contextual constraints from mechanical syntactic enforcement.
  - LLM focuses purely on crafting authentic 1-2 sentence classroom scenarios embedding triangulated narrative context clues, while symbolic code pre-computes zero-collision distractors, valency bindings, and morphological symmetry at zero token cost.
- **Key Invariants**:
  - **Noun SLA Grounding**: Replaces rare dictionary verb slots with authentic narrative context clues (Roles, Physical Props, Observable Actions), cognitive polarity contrast, and mandatory plural morphological symmetry (`plurale tantum` invariant).
  - **Verb Valency Cascade**: Transitive verbs prioritize direct object patients (`object` $\gg$ `prep` $\gg$ `adv_mod`); intransitive verbs prioritize bound prepositions (`prep` $\gg$ `adv_mod` $\gg$ `verb_subject`).
  - **Adjective Valency Cascade**: Prepositional adjectives prioritize bound prepositions (`prep` $\gg$ `modified_noun` $\gg$ `adv_mod`); descriptive adjectives prioritize characteristic head nouns (`modified_noun` $\gg$ `adv_mod` $\gg$ `prep`).
  - **Adverb Modification Cascade**: Evaluative/degree adverbs bind scalar adjectives (`modifies_adj` $\gg$ `modifies_verb`); manner/frequency adverbs bind action verbs (`modifies_verb` $\gg$ `modifies_adj`).
  - **Closed-Class Function Words**: Syntactic complement contrast (clausal vs. prepositional complement constraint) guarantees mathematically absolute single-fit discrimination on connectors (*despite* vs. *although*).
- *(Details see: [docs/vocabulary-selection-and-distractor-methodology.md](file:///E:/teacher-wiki/docs/vocabulary-selection-and-distractor-methodology.md))*.

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
- `enable_authentic_cloze`: Target cloze assessment mode (default: `true`).
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
- [x] COBUILD slot formula deterministic generation via spaCy dependency trees (`generate_cobuild_formula`)
- [x] Distractor sanitization & position shuffling with atomic key synchronization (`_shuffle_quiz_options`)
- [x] Quote boundary magnetic snapping to indexed sentence pool (`snap_to_sentence_pool`)
- [x] Canonical headword lemmatization & in-place self-healing (`lemmatize_headword`)
- [x] Elimination of multi-turn QA retries in production mode (One-Shot default established)
- [x] Full Purge of Oxford Collocations Dictionary (OCD) and complete migration to LDOCE 6th Edition semantic ordering
- [x] Complete Migration of Vocabulary & Expressions Extraction from LLM to Deterministic Code Extraction (`LinguisticEngine` + `cefrpy` + LDOCE 6th Edition)
- [x] Adjective Distractor Quadruple Transformation (WordNet bipolar cluster harvesting, opposite satellite antonym tracing, modified noun collocation clash, preposition valency gate, and academic adjective families)
- [x] CEFR Distractor Difficulty Ceiling & Offline Frequency Gate (P0: eliminated super-advanced/obscure distractors on foundational texts via `cefrpy` + Zipf frequency filtering)
- [x] Reading Assessment CEFR Difficulty Adaptation & L1 Code Gate (P2 Step 1 & 2: dynamic prompt interpolation of difficulty matrix, option length limit, and C1/C2 difficulty ceiling gate)
- [x] LDOCE Zero-Collision Anchor Fallback & Semantic Valency Cascade (`find_ldoce_zero_collision_anchor`)
- [x] CEFR Assessment Difficulty Calibration across All Remaining Modalities (Translation, Listening, Video with CEFR matrices, option length limits, and L1 code gates)
- [x] Full-Pipeline Cross-Stage Defect Alignment (decoupled comprehension quiz prompts, evaluator retry logic, reading context sentences, and slotted expression cloze cleanup)
- [x] Fixed Phrase Artificial Decomposition Prevention & Vocabulary Fragmentation Auto-Pruning
- [x] Markdown Extraction Formatting Standardization (`Part of Speech` & `Word CEFR Level`)
- [x] Option Morphological Symmetry & Plurale Tantum Invariant (100% symmetric inflections across target and distractors)

### In Progress
- **Grammar Code-Driven Extraction Optimization (语法代码化提取完善中)**:
  - Migrate remaining syntactic pattern matching and COBUILD slot formulas from prompt inference to deterministic spaCy rule gates.
  - Strengthen discourse-level clause boundary detection, rhetorical functional scoping (Four Macro Functional Domains), and stance/hedging classification.

### Pending / Next Improvement Directions (后续演化与改善方向 - 优先攻坚)
- **0. 核心合规与双架构适配器（P0: Dual-Distribution Adapter Architecture - 最先解决，再 Push）**:
  - **版权安全与零门槛开源分发**：确立开源版（Public）与高精语料版（Corpus-Powered）的平滑共存架构，杜绝双分支（Two-Branch）维护分裂。
  - **可插拔特征探测与优雅降级（Graceful Fallback）**：
    - 在单分支代码中实现环境自动探测：当检测到本地存在私有 `ldoce6_essential.db` 时，自动激活毫秒级精准搭配骨架、4阶词典干扰项与 0-token 代码化提取；
    - 当未挂载词典数据库时，系统自动无缝降级为**纯 LLM 提示词工程 + WordNet + spaCy 开源库**的纯软件轻量模式。
    - 仓库彻底将所有 `*.db`, `*.sqlite`, `*.mdx` 列入 `.gitignore`，GitHub 公开仓库零侵权把柄、零受限数据文件。
- **1. 双通道语境与搭配融合（Context-First Dual-Channel Collocation Engine）**:
  - 解决“Corpus Blueprint 与课文 Quote 语境冲突”：从硬性机械覆盖转向“课文语境优先”。
  - 优先通过 spaCy 语法依存分析提取课文原句（Quote）中的真实搭配并去 LDOCE 校验，若课文搭配权威合规，100% 沿用课文语境，保证题目的课文代入感；仅在课文语境过于散乱口语时，回退至词典 Blueprint 规范化骨架。
- **2. 句型模式互斥干扰项升级（Syntactic Pattern Exclusion for Single-Fit Distractors）**:
  - 释放 LDOCE 6th Edition `patterns` 核心价值（如 `[+ to do]`, `[+ on doing]`, `[+ that]`，及物/不及物）。
  - 在生成干扰项时引入句法槽位互斥判定作为高优先级权重，不仅确保“主题/语义相关”，更在“语法槽位上形成绝对排斥”，从根源上杜绝双答案（Double-key）缺陷。
- **3. 多词表达式代码化提取务实演化（Expressions & Phrasal Verbs Pragmatic Grounding）**:
  - **废弃残缺索引表（方案 A 已采纳）**：物理废弃并注明 `ldoce_phrase_index` 为空洞残缺表（大量核心搭配与日常短语如 `worry about`, `keep silent` 缺失严重，且引擎无任何代码依赖）。
  - **基于内存与条目结构的轻量提取**：继续依托 `LinguisticEngine` 现有的 `_phrase_evidence_cache` 与条目 `data_json`（senses、collocations）原生结构，配合 spaCy 语法依存与词形还原，实现动介短语、习语的精准识别与类型推断；避免过早引入复杂度过高的多表重构（方案 B 需海量测试验证其规律，暂缓实施）。
- **4. 教学法与 Markdown 卡片演化（Source-to-Wiki Enrichment）**:
  - Markdown 词汇卡片自动注入 Longman 经典搭配列表（0 Token 消耗）。
  - 优质错题本模板与知识图谱交互可视化。