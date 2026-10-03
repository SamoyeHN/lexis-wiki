# Lexis Wiki Project Instructions

> **Purpose**: This document serves as the single source of truth for the Lexis Wiki project -- an AI-powered wiki generator that produces Obsidian-style educational content from source materials using a schema-first prompt architecture.

---
## 1. Architecture & Core Philosophy

### 1.1 Separation of Responsibilities (Neuro-Symbolic Quad-Tier Architecture)
- **Symbolic Grounding & Lexical Corpus (NLP & Lexicons = Semantic Truth & Constraints)**:
  - `LinguisticEngine` combines offline computational NLP (spaCy syntactic dependency trees, POS tagging, morphological inflection lemmatization) with authoritative lexical databases (WordNet taxonomy, Longman Dictionary of Contemporary English 6th Edition / LDOCE).
  - **Absolute Grounding in LDOCE 6th Edition (Semantic Truth & Pedagogical Collocation Engine)**:
    - The engine is 100% grounded in LDOCE 6th Edition (`ldoce6_essential.db`), whose collocation boxes, syntactic patterns, and sense structures are semantically prioritized by authentic pedagogical salience and frequency. Obscure, physiologically biased, or non-pedagogical mechanical collocations are eliminated.
    - Deterministically pre-computes single-fit collocational anchors (preposition binding, adjective-noun collocations, verb-object valency) and 3 zero-collision, taxonomically distinct distractors at zero token cost (<1ms).
    - Pre-binds parallel inflectional morphology (e.g. all-VBN past participles, symmetric plural forms `NNS`) and CRC32 deterministic option positions, generating itemized `🎯 Micro-Task for LLM` constraint blueprints.
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
- **Grammar Extraction: Transition to Code-Driven Extraction in Active Development (代码化提取完善中)**:
  - **Current Status**: Structural pattern matching and syntactic formulas (COBUILD formulas, clausal dependency trees, non-finite adjuncts, cleft sentences, inversion, and extraposition) have been partially migrated to deterministic spaCy rule gates.
  - **Ongoing Refinement**: Further refining fine-grained discourse-level boundaries, rhetorical functional scoping, and deep contextual explanations to finalize full code extraction without relying on subjective LLM parsing.

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

### 1.6 Noun Semantic Grounding & SLA/Psychometric Context Clues Architecture
- **Deprecation of Rigid/Rare Collocation Anchors**:
  - Forcing rare or stiff dictionary collocations (e.g. demanding *extend* for *hospitality*, or *broker* for *peace*) produces unnatural, pedantic stems that deviate from authentic classroom language and real-world communicative usage.
  - Anchor fallbacks for nouns are decoupled from mechanical verb slot constraints (`[Subject] + [verb] + ____`), relying directly on LDOCE definition features and authentic semantic context clues.
- **Natural Narrative Context Clues Grounding (Roles, Props, Actions & Definition Grounding)**:
  - Noun item stems are dynamically guided by the core definition features retrieved from authoritative lexicons (e.g. LDOCE).
  - Prompts instruct the LLM to construct rich, authentic 1-2 sentence real-life or academic scenarios embedding concrete narrative clues (such as specific **Roles**, **Physical Settings/Props**, or **Characteristic Actions**) that naturally and unambiguously pinpoint the target noun.
- **Four Classical SLA & Language Testing Psychometric Design Principles (二语教学与测试学高区分度命题法)**:
  1. **Cognitive Polarity & Contrast Constraints (认知极性与因果转折锁定)**:
     - Uses concession/contrast structures (`Although / Despite / Even though [hardship/obstacle]... [positive compensatory action: the blank]`) to physically and logically force a unique semantic fit, eliminating subjective register or formality ambiguity.
  2. **Selectional Restrictions & Feature Clash (语义选择限制与语义元特征冲突)**:
     - Embeds diagnostic semantic features (`[+Urban/Infrastructure]`, `[+Animate]`, `[+Volition]`) into the subject or predicate to create irreconcilable category clashes with near-synonym distractors (e.g., *civilization* vs *culture/society* via *canal networks and written codes*).
  3. **Triangulated Context Clues (多重语境线索交叉三角定位 - Paul Nation 范式)**:
     - Stems provide at least two independent, mutually reinforcing clue dimensions:
       * *Agent/Role Clue* (e.g. *innkeepers, researchers, villagers*)
       * *Setting/Prop Clue* (e.g. *hearth/bed, laboratory data, legal tablet*)
       * *Affective/Reaction Clue* (e.g. *relieved sigh, exhausted travelers*)
     - Triangulation renders the target mathematically closed and unchallengeable under Expert L2 Auditing.
  4. **Pragmatic Connotation & Scale Contrast (语用适切度与外延/内涵标尺)**:
     - Distinguishes subtle near-synonyms by explicit external qualifiers (e.g. separating *vulnerable* from *fragile/weak* by specifying an external threat exposure slot: *dangerously ____ to unauthorized external access*).
- **Option Morphological Symmetry & Plurale Tantum Invariant (选项形态对称与唯复数名词门禁)**:
  - Standardized testing psychometrics strictly mandate **zero morphological leakage** across options. An item must NEVER present a solitary plural target among 3 singular distractors (e.g., `[choice, goods, reason, matter]`), which allows trivial test-wiseness guessing without reading comprehension.
  - When the target noun is plural or *plurale tantum* (`NNS` / `goods, customs, clothes, belongings, surroundings, fireworks, premises`):
    * `inflection` is formally bound to `plural form (NNS)`.
    * Computational NLP automatically applies `pluralize_noun` across all 3 distractors (`choice -> choices`, `reason -> reasons`, `matter -> matters`), generating 100% symmetric options `[choices, goods, reasons, matters]`.
- **Content-Word Lesk Disambiguation with Morphosyntactic Gating (实词语义消歧与语法形态门禁)**:
  - Prunes high-frequency function words/stopwords (`are, the, of, in...`) and target headword inflections from Lesk token overlap to eliminate false lexical overlap.
  - Strictly intercepts cross-category misassignments: a sense exclusively marked as `[plural]` (e.g., *customs* as airport border control) is physically forbidden from binding to singular/base-form context sentences.


### 1.7 Verb Collocational Selection & Valency Cascade (Transitive vs. Intransitive Distinction)
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
- **Transitive Prepositional Valency Frames (`analyze_verb_valency_pattern`)**:
  - Verbs with bound prepositions that govern an intermediate direct object in the active voice (`promote somebody to something`, `provide somebody with something`, `remind somebody of something`, `deprive somebody of something`) MUST NOT degrade to bare intransitive prepositional frames (`____ + to`).
  - Active syntactic frames explicitly bind the direct patient slot: `[Subject] + ____ + [Person / Direct Object] + [prep] + [Complement]`.
- **Authentic Quote Direct Object Priority & Subject Binding**:
  - For transitive verbs with direct object patients evidenced in the passage (e.g. *promote goods/products*), the authentic direct object is prioritized over disconnected prepositional patterns.
  - For intransitive/voice verbs where the authentic passage evidences an active subject noun (e.g. *as the clock chimes*), `verb_subject` (`clock`) is preserved as the syntactic anchor with a dedicated subject-predicate frame (`[Subject] + [Modal] + ____ + [Complement]`), avoiding misidentifying the subject as a direct object.


### 1.8 Adjective Collocational Selection & Valency Cascade (Prepositional vs. Descriptive Distinction)
- **Theoretical Basis (Adjectival Complementation & Attributive Selection)**:
  - **Prepositional Valency Adjectives (介词强配价形容词，如 `proud of`, `aware of`, `responsible for/to`, `anxious about`)**:
    - Possess strict, non-negotiable syntactico-semantic government over specific postpositional prepositions.
    - **Cascade Priority**: `prep` (Bound Preposition: *proud of*, *clear to*, *anxious about*) $\gg$ `modified_noun` (Attributive Noun: *responsible adult*) $\gg$ `adv_mod` (Degree Adverb: *deeply anxious*) $\gg$ `verb_copula` (Copula: *seem proud*).
    - Hard-welds the blank to the postpositional frame (`... is ____ of ...`), completely eliminating distractors that require different prepositions or lack prepositional valency.
  - **General / Descriptive / Relational Adjectives (一般描写与分类形容词，如 `simple`, `difficult`, `economic`, `legal`)**:
    - Function primarily as nominal modifiers or predicative complements, lacking bound prepositional valency.
    - **Cascade Priority**: `modified_noun` (Characteristic Modified Noun: *economic crisis*, *legal action*, *simple addition*) $\gg$ `adv_mod` (Collocational Degree Adverb: *devastatingly simple*, *doubly difficult*) $\gg$ `verb_copula` $\gg$ `prep`.
    - Prevents degenerative generic adverbial fallbacks (e.g. replacing essential *economic crisis* with generic *completely economic*).

### 1.9 Adverb Collocational Selection & Modification Domain Cascade (Adjective vs. Verb Modifiers)
- **Theoretical Basis (Adverbial Modification Domains & Scope)**:
  - **Degree / Focus / Stance Adverbs (程度、焦点与立场评注副词，如 `extremely`, `surprisingly`, `highly`, `deeply`, `completely`)**:
    - Function primarily as intensifiers or evaluative stance markers directly modifying scalar **adjectives** (*extremely able/difficult*, *surprisingly bright*, *deeply grateful*).
    - LDOCE holds explicit adjective-cluster and verb-collocation frames prioritized by frequency.
    - **Cascade Priority**: `modifies_adj` (Modified Adjective: *extremely able*, *deeply grateful*) $\gg$ `modifies_verb` (Modified Verb: *deeply absorb*).
    - Completely eliminates the structural failure where adjective-exclusive adverbs (like `extremely`, `surprisingly`) returned `None` due to verb-only scanning.
  - **Manner / Time / Frequency Adverbs (方式、时间与频度副词，如 `abruptly`, `carefully`, `politely`, `shortly`, `frequently`)**:
    - Function primarily as circumstantial adjuncts modifying core action **verbs**.
    - **Cascade Priority**: `modifies_verb` (Modified Verb: *abruptly abandon*, *answer politely*, *appear frequently*) $\gg$ `modifies_adj`.
### 1.10 Function Word Closed Paradigms & Syntactic Complement Contrast (Closed-Class Connectives)
- **Theoretical Basis (Open Class vs. Closed Class Grammatical Paradigms)**:
  - Unlike open-class content words (N, V, Adj, Adv), grammatical function words (subordinating conjunctions, prepositions, discourse connectors, complementizers) constitute a strictly bounded set (~200–300 words in English).
  - WordNet taxonomy fails on function words (e.g. misclassifying `despite` as a noun or returning empty synsets). Standard collocational lookup is unsuited for logical connectors whose core function is relational complementation rather than lexical co-occurrence.
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

### Pending
- **Extraction Pedagogical Quality (Source-to-Wiki)**:
  - **Vocabulary (Neuro-Symbolic Collocation & Academic Example Engine)**:
    - **spaCy Syntactic Extraction**: Extract authentic in-text usage (preposition binding `token.dep_ == 'prep'`, verb-object heads `dobj`, and adverbial/adjectival modifiers) directly from source sentences.
    - **Markdown Vocabulary Card Collocations Injection**: Automatically render authoritative Longman collocations (`- **Common Collocations (Longman)**:`) in `extractions/<Unit>_vocabulary.md` at 0 token cost during markdown serialization.
    - **Constrained Example Generation**: Feed retrieved authoritative collocations into the prompt as mandatory slot constraints (e.g., *"Construct example usage using the target collocation '[collocation]' "*), eliminating juvenile or trivial illustrative sentences.
- **Common Mistakes Curated Template Injection**:
  - Maintain curated pedagogical defect templates for core grammatical structures to enrich generic model explanations.
- **UI/UX & Multilingual Enhancements**:
  - Knowledge graph interactive visualization (legend, zoom, filter, preview nodes).
  - Client-side i18n support (English / Simplified Chinese).
  - Quality tier human review routing dashboard.