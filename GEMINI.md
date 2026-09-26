# Lexis Wiki Project Instructions

> **Purpose**: This document serves as the single source of truth for the Lexis Wiki project -- an AI-powered wiki generator that produces Obsidian-style educational content from source materials using a schema-first prompt architecture.

---
## 1. Architecture & Core Philosophy

### 1.1 Separation of Responsibilities (Neuro-Symbolic Quad-Tier Architecture)
- **Symbolic Grounding & Lexical Corpus (NLP & Lexicons = Semantic Truth & Constraints)**:
  - `LinguisticEngine` combines offline computational NLP (spaCy syntactic dependency trees, POS tagging, morphological inflection lemmatization) with authoritative lexical databases (WordNet taxonomy, Oxford Collocations Dictionary 2nd Ed.).
  - Deterministically pre-computes single-fit collocational anchors (preposition binding, adjective-noun collocations, verb-object valency) and 3 zero-collision, taxonomically distinct distractors at zero token cost (<1ms).
  - Pre-binds parallel inflectional morphology (e.g. all-VBN past participles) and CRC32 deterministic option positions, generating itemized `🎯 Micro-Task for LLM` constraint blueprints.
- **Code = Invariant Enforcement & Physical Gate**:
  - Deterministic Python logic (`evaluator.py`, `processor.py`) enforces physical invariants, string normalization, blank constraints (`re.findall(r'_{2,}', stem) == 1`), atomic option remapping, and bounds at zero token cost.
  - Acts as the Level 1 Deterministic Code Gate to instantly filter, remap, or self-heal structural defects without burning LLM retry tokens.
- **Prompt = Pedagogy & Context Generation**:
  - `.md` prompts focus 100% on educational standards (CEFR/TOEFL), cognitive depth, and executing embedded micro-tasks.
  - LLM acts as an academic sentence crafter and psychometric discriminator: crafts natural contextual stems fulfilling the exact syntactic constraints, and generates deep pedagogical explanations by quoting target and distractor wordings directly—completely free of mechanical JSON formatting or distractor generation burdens.
- **Schema = Structural Integrity**:
  - `schemas.py` defines output format, required keys, JSON data types, and array constraints via native API structured outputs. Eliminates syntactic drift and structural formatting instructions from system prompts.

### 1.2 Generation Pipeline Modes
- **Extraction (Vocabulary / Expressions / Grammar)**: Defaults to **One-Shot** (`enable_vocab_prose: false`, `enable_grammar_prose: false`). One-Shot provides 80% faster generation and superior verbatim grounding for text-extraction tasks.
- **Assessment / Quiz Generation Pipeline**:
  - **Standard Production Mode: One-Shot Direct JSON (`enable_prose_pipeline: false`)**:
    - **Empirical Baseline**: Thoroughly validated across 11 benchmark models (8B–27B). One-Shot achieves 100% structural completion, 2.2× faster generation (~30s vs ~78s), and eliminates secondary turn semantic drift, target-word loss, and answer-key misalignment.
    - **Architecture Strategy**: Paired with deterministic WordNet pre-computed distractors and Python-level key binding, One-Shot serves as the default production configuration.
  - **Diagnostic / Research Mode: Two-Turn Decoupled Pipeline (Prose-to-JSON)** (`enable_prose_pipeline: true`):
    - Two stateless, independent HTTP calls ($\text{Call}_1$: Prose Draft $\rightarrow$ $\text{Call}_2$: JSON Packaging).
    - **Purpose & Scope**: Exclusively retained for **offline debugging, pedagogical prompt inspection, raw response probing, and long-chain reading/translation reasoning analysis**. Not recommended for standard vocabulary assessment delivery due to high variance and packaging fragility in smaller models ($\le 14$B).
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

### 1.4 Unified CEFR Difficulty Calibration Architecture (`cefrpy` Integration)
- **Curricular Alignment Mandate**: Eliminates the systemic defect where foundational curriculum texts (e.g. CEFR A1/A2/B1 introductory units) generated assessments harder than the text itself (e.g., college-level distractors, academic TOEFL question stems, or zero grammar yield).
- **Extraction Layer (Vocabulary & Grammar)**:
  - `schemas.py`: Standardized `VOCAB_CEFR_LEVELS = Literal["A1", "A2", "B1", "B2", "C1", "C2"]`.
  - `extract_grammar.md`: Mined pattern formulas and imitation examples dynamically align with the curriculum level (`cefr_level`), banning impenetrable academic jargon on foundational texts.
- **Assessment Layer (Completed: Vocabulary & Reading)**:
  - **Vocabulary Quiz (`vocabulary_quiz`)**:
    - Pre-computed WordNet distractors pass through an offline `cefrpy` + Zipf frequency ceiling gate ($Zipf \ge 3.2$ for A1/A2, $3.0$ for B1), physically eliminating obscure or super-advanced distractors (e.g., `conceivableness` on basic words).
  - **Reading Quiz (`reading_quiz`)**:
    - **Dynamic Prompt Interpolation**: Injects `{cefr_descriptor}`, `{question_stem_guidance}`, `{option_complexity_guidance}`, `{skill_distribution_guidance}`, and `{vocab_target_guidance}`. For A1/A2, question stems are direct, options are concise (4–12 words), and skills focus on Detail/Recall & Main Idea.
    - **Level 1 Deterministic Code Gate (`audit_reading_integrity`)**: Enforces option length bounds ($\le 16$ words for A1/A2, $\le 20$ words for B1) and uses `cefrpy` to audit that non-passage option/stem vocabulary never exceeds the passage CEFR ceiling (intercepts C1/C2 words).
- **Assessment Layer (Pending: Translation, Listening, Video)**:
  - **Translation Quiz (`translation_quiz`)**: Currently retains strict CEFR B2–C1 / TEM-8 / TOEFL comparative translation appraisal standards; pending future adaptation to scale down to A1–B1 foundational curriculum sentences.
  - **Listening & Video Quizzes**: Standardized on intermediate/advanced levels, pending CEFR difficulty calibration.

### 1.5 Noun Semantic Selection & Collocational Cascade (Concrete vs. Abstract Distinction)
- **Theoretical Basis (Selectional Restrictions & Valency Theory)**:
  - **Concrete Nouns (具象实体名词，如 `telephone`, `bell`, `machine`, `car`)**:
    - Act naturally as **physical agents/actors** possessing unique exclusive actions.
    - **Cascade Priority**: `+ verb` (Verb Subject: *telephone was ringing*) $\gg$ `verb +` (Verb Object: *use telephone*) $\gg$ `adjective` (*cellular telephone*) $\gg$ `preposition` (*over the telephone*).
    - Physical action exclusivity provides maximum single-fit discrimination against distractors at zero collision.
  - **Abstract Nouns (抽象概念名词，如 `decision`, `peace`, `influence`, `difficulty`, `freedom`)**:
    - In authentic usage and systemic functional linguistics, abstract concepts primarily function as the **goal/patient** of light/support verbs or idiomatic frames.
    - **Cascade Priority**: `verb +` (Light/Support Verb Object: *make/reach a decision*, *exert influence*, *encounter difficulty*, *broker peace*) $\gg$ `preposition` (*in peace*, *under the influence of*) $\gg$ `adjective` (*firm decision*) $\gg$ `+ verb` (Verb Subject: *peace prevailed*, demoted to secondary fallback).
- **Automated WordNet Ontology Classification (`find_ocd_zero_collision_anchor`)**:
  - Automatically analyzes WordNet lexicographer files (`lexfile()`). Categorizes items into Abstract (`noun.act`, `noun.cognition`, `noun.state`, `noun.attribute`, `noun.feeling`, `noun.motive`, `noun.relation`, `noun.time`, `noun.event`, `noun.process`) vs. Concrete (`noun.artifact`, `noun.animal`, `noun.body`, `noun.food`, `noun.plant`, `noun.substance`, `noun.object`, `noun.person`), dynamically steering the OCD frame retrieval cascade to eliminate artificial and juvenile question stems.

### 1.6 Verb Collocational Selection & Valency Cascade (Transitive vs. Intransitive Distinction)
- **Theoretical Basis (Transitivity & Argument Structure)**:
  - **Transitive Verbs (及物支配动词，如 `solve`, `abandon`, `provide`, `accelerate`)**:
    - Possess direct semantic selectional restrictions on their **direct object patient/theme**.
    - **Cascade Priority**: `object` (Direct Object Noun: *solve the puzzle/crisis*, *accelerate decline*) $\gg$ `prep` (Bound Preposition: *provide sb with sth*) $\gg$ `adv_mod` (Manner Adverb: *abandon hastily*).
    - Prevents degenerative generic adverbial fallbacks (e.g. replacing generic *solve with spending cuts* with essential *solve the puzzle/mystery*).
  - **Intransitive Verbs (不及物及介词动词，如 `listen`, `arrive`, `depend`, `hesitate`)**:
    - Incapable of taking direct objects; syntactic interaction relies strictly on **bound prepositions** or **manner adverbial adjuncts**.
    - **Cascade Priority**: `prep` (Bound Preposition: *listen to*, *arrive at*, *depend on*, *hesitate about*) $\gg$ `adv_mod` (Manner Adverb: *listen attentively*, *arrive safely*) $\gg$ `verb_subject` (Subject Noun).
- **Bucketed Distractor Collision Enforcement**:
  - Distractor collocations are pre-indexed into grammatical functional buckets (`prep`, `object`, `adv_mod`, `verb_subject`).
  - Ensures a candidate preposition for target verb is audited strictly against the `prep` bucket of distractors, preventing spurious clashes while maintaining zero collision.

### 1.7 Adjective Collocational Selection & Valency Cascade (Prepositional vs. Descriptive Distinction)
- **Theoretical Basis (Adjectival Complementation & Attributive Selection)**:
  - **Prepositional Valency Adjectives (介词强配价形容词，如 `proud of`, `aware of`, `responsible for/to`, `anxious about`)**:
    - Possess strict, non-negotiable syntactico-semantic government over specific postpositional prepositions.
    - **Cascade Priority**: `prep` (Bound Preposition: *proud of*, *clear to*, *anxious about*) $\gg$ `modified_noun` (Attributive Noun: *responsible adult*) $\gg$ `adv_mod` (Degree Adverb: *deeply anxious*) $\gg$ `verb_copula` (Copula: *seem proud*).
    - Hard-welds the blank to the postpositional frame (`... is ____ of ...`), completely eliminating distractors that require different prepositions or lack prepositional valency.
  - **General / Descriptive / Relational Adjectives (一般描写与分类形容词，如 `simple`, `difficult`, `economic`, `legal`)**:
    - Function primarily as nominal modifiers or predicative complements, lacking bound prepositional valency.
    - **Cascade Priority**: `modified_noun` (Characteristic Modified Noun: *economic crisis*, *legal action*, *simple addition*) $\gg$ `adv_mod` (Collocational Degree Adverb: *devastatingly simple*, *doubly difficult*) $\gg$ `verb_copula` $\gg$ `prep`.
    - Prevents degenerative generic adverbial fallbacks (e.g. replacing essential *economic crisis* with generic *completely economic*).

### 1.8 Adverb Collocational Selection & Modification Domain Cascade (Adjective vs. Verb Modifiers)
- **Theoretical Basis (Adverbial Modification Domains & Scope)**:
  - **Degree / Focus / Stance Adverbs (程度、焦点与立场评注副词，如 `extremely`, `surprisingly`, `highly`, `deeply`, `completely`)**:
    - Function primarily as intensifiers or evaluative stance markers directly modifying scalar **adjectives** (*extremely able/difficult*, *surprisingly bright*, *deeply grateful*).
    - OCD holds explicit adjective-cluster frames for 728 adverbs (~50% of adverb headwords).
    - **Cascade Priority**: `modifies_adj` (Modified Adjective: *extremely able*, *deeply grateful*) $\gg$ `modifies_verb` (Modified Verb: *deeply absorb*).
    - Completely eliminates the structural failure where adjective-exclusive adverbs (like `extremely`, `surprisingly`) returned `None` due to verb-only scanning.
  - **Manner / Time / Frequency Adverbs (方式、时间与频度副词，如 `abruptly`, `carefully`, `politely`, `shortly`, `frequently`)**:
    - Function primarily as circumstantial adjuncts modifying core action **verbs**.
    - **Cascade Priority**: `modifies_verb` (Modified Verb: *abruptly abandon*, *answer politely*, *appear frequently*) $\gg$ `modifies_adj`.
### 1.9 Function Word Closed Paradigms & Syntactic Complement Contrast (Closed-Class Connectives)
- **Theoretical Basis (Open Class vs. Closed Class Grammatical Paradigms)**:
  - Unlike open-class content words (N, V, Adj, Adv), grammatical function words (subordinating conjunctions, prepositions, discourse connectors, complementizers) constitute a strictly bounded set (~200–300 words in English).
  - WordNet taxonomy fails on function words (e.g. misclassifying `despite` as a noun or returning empty synsets). Standard collocational lookup (OCD) is unsuited for logical connectors whose core function is relational complementation rather than lexical co-occurrence.
- **Closed Functional Paradigms (`_FUNCTION_WORD_PARADIGMS`)**:
  - Organized into definitive functional discourse families: `concession`, `cause`, `condition`, `time`, `scope`, `noun_clause`.
  - Sub-categorized by strict syntactic complement requirements:
    - `clausal`: Governs finite clauses ($\text{Subject} + \text{Finite Verb}$, e.g. *although, whereas, because, unless, while*).
    - `prepositional`: Governs noun phrases or gerunds ($\text{NP} / \text{V-ing}$, e.g. *despite, because of, during, beyond, without*).
    - `adverbial`: Independent discourse adjuncts (e.g. *however, therefore, otherwise, meanwhile*).
- **Psychometric Distractor Synthesis Models**:
  - **Syntactic Complement Contrast (Gold Standard / 单解绝杀门禁)**:
    - For logical connectors (`concession`, `cause`, `condition`, `time`), pre-selects distractors from the *opposing* syntactic complement class within the same semantic family (e.g. Target `despite` [prepositional] paired with `['although', 'though']` [clausal]).
    - All options share the identical semantic orientation (concession/contrast), but only the target satisfies the physical complement slot constraint (`____ + NP`), eliminating secondary turn semantic drift and guaranteeing mathematically absolute single-fit validity.
  - **Closed Scope & Complementizer Contrast**:
    - For spatial/metaphorical prepositions (`beyond`) and noun-clause markers (`whether`), draws distractors strictly from within their authoritative functional paradigms (`['within', 'across', 'throughout']` and `['that', 'what', 'whatever']`).
- **Micro-Task Blueprint Hard-Welding**:
  - Dynamically synthesizes structural stem constraints:
    - Prepositional target: `Syntactic Frame: The blank '____' MUST be followed directly by a noun phrase or gerund, NOT a clause with a finite verb!`
    - Clausal target: `Syntactic Frame: The blank '____' MUST introduce a complete subordinate clause with subject and finite verb!`
- **Extraction Boundary & Tri-Tier Layer Routing**:
  - **Ultra-Basic Function Words (`in, at, and, but, if, because`)**: ❌ Filtered out from extraction.
  - **High-Utility Academic Single-Word Connectors (`despite, whereas, beyond, throughout, unless, nonetheless`)**: ✅ Explicitly permitted in `extract_vocabulary.md` (AWL / CEFR B1+). Grounded by closed paradigms in `vocabulary_quiz`.
  - **Multi-Word Connective Phrases (`in spite of, due to, as long as, provided that`)**: Routed exclusively to expressions extraction (`extract_expressions.md`).

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
- [x] COBUILD slot formula deterministic generation via spaCy dependency trees (`generate_cobuild_formula`)
- [x] Distractor sanitization & position shuffling with atomic key synchronization (`_shuffle_quiz_options`)
- [x] Quote boundary magnetic snapping to indexed sentence pool (`snap_to_sentence_pool`)
- [x] Canonical headword lemmatization & in-place self-healing (`lemmatize_headword`)
- [x] Elimination of multi-turn QA retries in production mode (One-Shot default established)
- [x] Oxford Collocations Dictionary 2nd Edition integration (20,791 headwords, ~4.9MB clean JSON, `LinguisticEngine.get_rich_collocations`)
- [x] Adjective Distractor Quadruple Transformation (WordNet bipolar cluster harvesting, opposite satellite antonym tracing, modified noun collocation clash, preposition valency gate, and academic adjective families)
- [x] CEFR Distractor Difficulty Ceiling & Offline Frequency Gate (P0: eliminated super-advanced/obscure distractors on foundational texts via `cefrpy` + Zipf frequency filtering)
- [x] Grammar Extraction Adaptive Gate & Multi-Level Coverage (P1: solved clefts `nsubj/expl`, relative clauses, causative complements, stance transitions, and curriculum register calibration)
- [x] Reading Assessment CEFR Difficulty Adaptation & L1 Code Gate (P2 Step 1 & 2: dynamic prompt interpolation of difficulty matrix, option length limit, and C1/C2 difficulty ceiling gate)
- [x] Oxford Collocations Dictionary (OCD) Zero-Collision Anchor Fallback & Concrete vs. Abstract Noun Semantic Cascade (`find_ocd_zero_collision_anchor`)

### Pending
- **Extraction Pedagogical Quality (Source-to-Wiki)**:
  - **Vocabulary (Neuro-Symbolic Collocation & Academic Example Engine)**:
    - **spaCy Syntactic Extraction**: Extract authentic in-text usage (preposition binding `token.dep_ == 'prep'`, verb-object heads `dobj`, and adverbial/adjectival modifiers) directly from source sentences.
    - **Markdown Vocabulary Card Collocations Injection**: Automatically render authoritative Oxford collocations (`- **Common Collocations (Oxford)**:`) in `extractions/<Unit>_vocabulary.md` at 0 token cost during markdown serialization.
    - **Constrained Example Generation**: Feed retrieved authoritative collocations into the prompt as mandatory slot constraints (e.g., *"Construct example usage using the target collocation '[collocation]' "*), eliminating juvenile or trivial illustrative sentences.
  - **Grammar**: Strengthen sentence selection criteria for authentic pedagogical/discourse value, and enrich `explanation` with functional linguistic stance (nominalization, discourse framing, hedging).
- **Deterministic CEFR Labelling (Offline Dictionary Integration)**:
  - Integrate offline frequency/proficiency lexicons (CEFR-J, Oxford 3000/5000, AWL) to objectively label CEFR levels across words, sentences, and passages without relying on LLM estimation.
- **CEFR Difficulty Alignment for Remaining Assessment Modalities**:
  - **Translation Assessment (`translation_quiz`)**: Currently locked at B2–C1 / CET-6 / TEM-8 / TOEFL comparative appraisal standards; adapt source sentence complexity, target grammar pattern difficulty, and flaw taxonomy to scale smoothly down to foundational levels (A1–B1) when processing introductory textbooks.
  - **Listening & Video Assessments (`listening_quiz`, `video_quiz`)**: Calibrate comprehension question stems, option length limits, and distractor vocabulary ceilings based on the passage/transcript CEFR level.
- **WordNet & Pre-Computed Distractor Assembly**:
  - Implement offline synthesis of 3 collision-free, distinct-taxonomy distractors (<1ms, 0 tokens) and pair with One-Shot question stem generation to physically eliminate double keys.
- **Common Mistakes Curated Template Injection**:
  - Maintain curated pedagogical defect templates for core grammatical structures to enrich generic model explanations.
- **UI/UX & Multilingual Enhancements**:
  - Knowledge graph interactive visualization (legend, zoom, filter, preview nodes).
  - Client-side i18n support (English / Simplified Chinese).
  - Quality tier human review routing dashboard.