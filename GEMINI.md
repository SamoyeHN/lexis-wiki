# Lexis Wiki Project Instructions

> **Purpose**: This document serves as the single source of truth for the Lexis Wiki project -- an AI-powered wiki generator that produces Obsidian-style educational content from source materials using a schema-first prompt architecture.

---
## 1. Architecture & Core Philosophy

### 1.1 Separation of Responsibilities (Neuro-Symbolic Quad-Tier Architecture)
- **Symbolic Grounding & Lexical Corpus (NLP & Lexicons = Semantic Truth & Constraints)**:
  - `LinguisticEngine` combines offline computational NLP (spaCy syntactic dependency trees, POS tagging, morphological inflection lemmatization) with authoritative lexical databases (WordNet taxonomy, Pedagogical Lexicon / PedLex).
  - **Absolute Grounding in Pedagogical Lexicon (PedLex: Semantic Truth & Pedagogical Collocation Engine)**:
    - The engine is 100% grounded in the offline Pedagogical Lexicon (`pedlex.db`), whose collocation boxes, syntactic patterns, and sense structures are semantically prioritized by authentic pedagogical salience and frequency. Obscure, physiologically biased, or non-pedagogical mechanical collocations are eliminated.
    - Deterministically pre-computes single-fit collocational anchors and 3 zero-collision distractors at zero token cost (<1ms).
    - Pre-binds parallel inflectional morphology and CRC32 deterministic option positions, generating itemized `🎯 Micro-Task for LLM` constraint blueprints. *(Details see: [docs/vocabulary-selection-and-distractor-methodology.md](docs/vocabulary-selection-and-distractor-methodology.md))*.
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
  - **Deterministic Pipeline (`LinguisticEngine` + `cefrpy` + Pedagogical Lexicon / PedLex)**:
    - **Single-Word Vocabulary**: spaCy POS-tagging, lemmatization, and syntax dependency trees extract candidate content headwords directly from the passage text. Headwords are filtered and calibrated against the CEFR lexical database and the Pedagogical Lexicon at 0 token cost, guaranteeing 100% lexical fidelity and exact syntactic alignment.
    - **Multi-Word Expressions & Phrasal Idioms**: Deterministically parsed and mined using dependency trees (verb-particle combinations, prepositional verbs, fixed idioms) and dictionary index lookups, completely eliminating arbitrary or fragmented chunking.
    - **Expression-Type Labelling**: A multi-word headword is never labelled with the part of speech of one of its tokens ('tap into' is not a 'verb'). `LinguisticEngine.classify_expression_type` reads the pedagogical lexicon filing first (a phrasal-verb block or a PHRASES entry named exactly that unit, slots and slash groups expanded), then a frame opened by a closed-class word, then verb + particle adjacency, and returns one of `phrasal verb`, `collocation`, `set phrase`, `idiom`. The evaluator, the LLM healing pass and the Markdown writer each re-type any plain part of speech that arrives on a multi-word row.
- **Grammar Extraction: Deterministic Code-Driven Extraction (方案 A：务实推荐已落地)**:
  - **Current Status**: Structural pattern matching and syntactic formulas (COBUILD formulas, clausal dependency trees, non-finite adjuncts, cleft sentences, inversion, complex transitives, and extraposition) are deterministically extracted at 0 token cost via `LinguisticEngine.extract_deterministic_grammar()`.
  - **Scheme A Implementation Policy (方案 A 务实落地)**:
    - 统一所有语法警告前缀为 `Grammar Warning (<TOPIC>): ...`，彻底清除与禁止任何带来源偏见或空洞的特定词典前缀区分与显示。
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
- *(Details see: [docs/vocabulary-selection-and-distractor-methodology.md](docs/vocabulary-selection-and-distractor-methodology.md))*.

### 1.7 Definition-Locked Word Sense Disambiguation (WSD) & Authentic Multi-Word Phrase Distractors
- **Definition-Locked WSD Engine (`LinguisticEngine._lock_sense`)**:
  - Offline multi-feature weighted disambiguation (<0.5ms): POS hard gate $\rightarrow$ conversational/spoken formulaic locution gate (`[spoken]`, `[greeting]`) $\rightarrow$ pattern & collocation syntactic intersection $\rightarrow$ contextual lemma overlap.
  - Pins the extracted headword strictly to its authoritative pedagogical definition, grammatical pattern slot, register label, and sense-specific opposites/collocations.
- **Authentic Multi-Word Phrase Distractor Generation (`LinguisticEngine.generate_phrase_distractors`)**:
  - Fully connects the 106,738 entries in the pedagogical phrase index (`pedlex_phrase_index`) to assessment distractor synthesis, eliminating arbitrary hallucinated distractors.
  - Generates strictly symmetric, authentic distractors across structural families (tail-sharing clusters, light-verb constructs, binomial coordinate matches, and 2-word phrasal verb particle/verb contrasts).
  - **Quantifier Frames (`a piece/bit/slice of`)**: Implements Scheme 2 (Head Blanking) with quantifier sibling candidate generation (`[piece, bit, slice, sheet]`) and blank frame alignment (`a ____ of [Uncountable Noun]`).
  - **Pending Special Phrase Families**: Correlative connectors, irreversible binomials, light verb-noun collocations, and prepositional frames recorded in internal linguistic roadmap.
- *(Full public distractor and valency cascade methodology see: [docs/vocabulary-selection-and-distractor-methodology.md](docs/vocabulary-selection-and-distractor-methodology.md))*.

### 1.8 Architectural Discipline & Anti-Patching Escalation Principle
- **Anti-Patching Mandate (Zero Tolerate for Band-Aid Bloat)**:
  - When existing programmatic logic, rule-based heuristics, or symbolic NLP hit a fundamental competence ceiling or cannot cleanly achieve the target pedagogical/linguistic goal, developers and AI agents must NOT endlessly accumulate ad-hoc heuristics, brittle edge-case conditions, or arbitrary exception branches.
- **Proactive Warning & First-Principles Paradigm Escalation**:
  - Whenever a feature or bug fix requires repeated, compounding patches with diminishing returns, the assistant must immediately:
    1. **Proactively Warn**: Explicitly flag the structural limitation and explain why current rule-based / linguistic heuristics are inadequate or prone to regression.
    2. **Propose New Architecture**: Present clear, first-principles alternative solutions (e.g., lightweight local ONNX semantic embedding models, semantic vector indexing, hybrid neural-symbolic classifiers, structural schema decoupling, or targeted multi-agent verification) rather than continuing to patch fragile heuristics.

### 1.9 Single Source of Truth Grounded on Source Text for Assessments (Fresh Deterministic Extraction)
- **Decoupling Downstream Markdown from Assessment Truth**:
  - `extractions/<Unit>_{vocabulary,grammar}.md` files are downstream presentation artifacts intended solely for reading in Obsidian/Web Dashboard. Teachers or students may modify, translate, or customize them at will.
  - Assessment generation (`vocabulary_quiz`, `translation_quiz`, etc.) **MUST NEVER** treat `extractions/*.md` as authoritative semantic or lexical truth. Relying on downstream markdown risks ingesting stale, incomplete, or user-altered data (e.g. definitions modified to Chinese, tampered POS, or broken examples), which fatally contaminates quiz items.
- **Mandatory Fresh Deterministic Extraction at Quiz Generation**:
  - For both `vocabulary_quiz` and `translation_quiz`, generation pipelines must ground directly on the authentic primary source text (`sources/<Unit>.md`) and its syllabus items.
  - The pipeline runs a fresh deterministic extraction pass (`LinguisticEngine` + PedLex SQLite + Pure Symbolic Pre-filtering + Local LLM Arbitration) directly against authentic source sentences, completely decoupling from external embedding models (`all-MiniLM-L6-v2` 80MB model permanently removed).
  - **Atomic Semantic Cohesion (`Sense` Primacy)**:
    - Pure symbolic syntactic constraints (POS gates, valency/preposition slots, Lesk lexical overlap, frequency) build a top-$K$ candidate pool ($K \le 4$), and a 1-token Local LLM arbitration pass dynamically locks the headword to its authoritative pedagogical `Sense` using the authentic sentence context.
    - The locked `Sense` acts as the single invariant binding `part_of_speech`, `definition`, `patterns`, `collocational anchors`, and pre-computed zero-collision distractors into an unbreakable, atomic blueprint. This permanently eliminates pos-definition desync (e.g. verb distractors with noun definitions).
  - **Translation Quiz Alignment**:
    - `translation_quiz` follows the identical rule: core grammatical patterns, key headwords, and clause structures are freshly extracted and calibrated from `sources/<Unit>.md`, ensuring target bilingual translations strictly represent authentic passage pedagogy rather than edited extraction notes.

### 1.10 Sense-First Unified Architecture & Atomic Pattern-Example Binding (义项为唯一核心与闭环绑定铁律)
- **绝对禁止“两套逻辑”与“脱节漫游”**:
  - 提取阶段（Extraction）与命题生成阶段（Quiz Blueprint）必须使用**完全统一的单一套 Sense-First 核心逻辑**，严禁提取一套规则、Quiz 阶段又脱离 Sense 去全库自由漫游。
  - **Sense（义项）是唯一的真理源头与第一公民**：
    - 进入生成前，第一步**必须且只能是执行 Sense-Locking**，锁定目标词在课文真实语境下的权威 Pedagogical Sense 对象。
    - 严禁在 Sense 锁定之前使用词条级别的全局搭配库或全库例句进行无义项归属的检索（例如为 `pay` 跨义项搜 `money` 导致引入毫无关联的 `pay for` 句子）。
- **Pattern 与 Example 的原子闭环绑定 (Atomic Binding)**:
  - 选定 Sense 后，该 Sense 原生自带的 `definition`、`patterns`、`examples` 和 `collocations` 即构成不可分割的原子集合。
  - **句式框架与例句的孪生约束**：
    - 若选择或匹配了该 Sense 下的某一个语法 Pattern（如 `pay (somebody) in dollars/euros etc`），其供给 LLM 的语料示范例句（`Corpus Example Frame`）**必须严格出自该 Pattern 原生配套的例句**（如 `Can we pay in dollars?`），或出自包含该介词/句式特征的该 Sense 原生例句。
    - **严禁左右互搏**：绝对禁止 Syntactic Frame 考某种句式（如 `pay in [currency]`），而展示的例句却来自另一个完全不相干的句式（如偷钱买烟 `pay for`）！
- **搭配锚点 (Anchor) 的向下继承**:
  - 若锁定的 Pattern 包含核心从属/介词（如 `pay in`、`serious about`、`message for`），Anchor 与 Anchor Type 必须直接从该 Pattern 的介词/结构槽位继承（如 `anchor: "in"`, `anchor_type: "prep"`），保证从 Frame、Anchor 到 Distractor 筛选的完全自洽。

### 1.11 真正神经-符号混合义项与句式重排裁决架构 (True Neuro-Symbolic Sense & Pattern Reranker)
- **核心理念 (True Neuro-Symbolic Paradigm)**:
  - 彻底终结用脆弱的硬编码启发式（if-else/正则特判/规则打补丁）去解决自然语言中多义词与复杂句法配价长尾歧义的死胡同。
  - 彻底移除体积冗余（80MB）且缺乏离散句法感知能力的密集向量模型（`all-MiniLM-L6-v2`），避免其将 "Punctuality pays!" 误判为金融付款类语义偏置。
  - **符号数据库（SQLite / PedLex / LDOCE）充当“绝对语义真理与候选池”**：
    - 负责离线、零成本检索目标词全部合法的 Sense、POS、Pattern 槽位及原版例句。
    - 结合离散句法特征（POS 硬约束、核心介词/搭配槽位重合、上下文词形交集、教学频度先验）构建精准的 Top-$K$ ($K \le 4$) 候选池。
    - LLM 永远**零自由生成权、零幻觉空间**，绝对禁止凭空臆造定义或篡改词典体系。
  - **神经网络（Local / Remote LLM）充当“极窄候选集判别器（Zero-Hallucination Reranker）”**：
    - 针对课文原句，将符号库检索出的 $N$ 个权威候选义项与 Pattern 构建为极简多选单选题（Prompt 仅 ~100 tokens）。
    - 利用大语言模型的深度自注意力机制（Self-Attention）与常识理解，毫秒级输出选项编号（`A/B/C/D`，仅需生成 1 个 token）。
- **实测验证效能 (Empirical Verification across 5 Models)**:
  - 在涵盖多词性歧义（`seal` 印章 vs 封锁 vs 海豹）、抽象引申义（`attach importance to`、`pay` 带来好处 vs 付钱）、动名词与配价框架（`take to doing sth` 习惯 vs `take sth to` 运输、`make up` 占比 vs 捏造、`charge` 费用双宾句式）的高难度 16 道全量基准集上，横向实测 **5 款跨架构主流模型全部取得 100% 满分**：
    - `llama3.1:8b`：**16/16 (100.0%)**，单题推理延迟 **0.28s** (极速)；
    - `ministral-3:8b`：**16/16 (100.0%)**，单题推理延迟 **0.35s**；
    - `qwen2.5-coder:7b`：**16/16 (100.0%)**，单题推理延迟 **0.41s**；
    - `gemma4:12b`：**16/16 (100.0%)**，单题推理延迟 **0.60s**；
    - `granite4.2:8b`：**16/16 (100.0%)**，单题推理延迟 **3.43s**。
  - **实证结论**：证明该任务（微型窄候选 Reranker）对不同家族 7B–12B 模型具有**绝对的泛化性与超高鲁棒性**，且 8B 级模型的单题推理延迟普遍低至 **0.1~0.3 秒**，完全满足在线实时编译要求。
- **全词库覆盖度与分层漏斗执行原则 (Coverage & Tiered Cascade)**:
  - **PedLex 库统计数据**：
    - **70.66%** 的词条在对应 POS 下仅有单义项（Single Sense），本地纯符号层 **0ms 瞬间锁定，消耗 0 tokens**。
    - **94.00%** 的词条在对应 POS 下候选数 $\le 4$，保证 Top-4 候选池能够涵盖绝大多数合法语义。
  - **单义词 / 单候选短语 (~71%)**：纯代码符号层 0ms 瞬间锁定，消耗 0 tokens。
  - **多义词 / 跨义项竞争短语 (~29%)**：自动激活 Local LLM 极简 1-token 裁决，以极微小开销彻底清零词义错位。

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

---

## 6. Project Documentation Index (`docs/`)

Authoritative technical specifications and architectural deep dives:
- **[docs/pedlex-database-architecture.md]**: Pedagogical Lexicon (PedLex) SQLite schema, 106k phrase inverted index, Definition-Locked WSD scoring, and phrase distractor algorithms.
- **[docs/vocabulary-selection-and-distractor-methodology.md]**: Pedagogical lexical selection, Valency cascades (verb/adj/noun/adv), single-fit collocational anchors, and distractor generation methodology.
- **[docs/audit-scores-guide.md]**: Level 1 deterministic code gate and Level 2 expert evaluator scoring rubrics and triage thresholds.
- **[docs/cli.md]**: CLI commands, compilation parameters, and dashboard orchestration workflows.

---

## 7. To-Do List

> ✅ Complete | 🔄 In Progress | ⬜ Pending

### Completed (Milestones & Architectural Core)
- [x] **Dual-Distribution Adapter & Zero-Leakage Open Sourcing (核心合规与双架构适配器已落地)**:
  - Clean separation of Public vs. Corpus-Powered modes: all `.db`, `.sqlite`, `.mdx` binaries isolated in `.gitignore`.
  - Automatic environment detection (`db_path.exists()`, config `disable_pedlex` switch, `DISABLE_PEDLEX` env var): smoothly falls back to spaCy + cefrpy + WordNet (oewn:2025+) without crashes.
- [x] **Syntactic Pattern Exclusion for Single-Fit Distractors (句型模式互斥干扰项升级已落地)**:
  - Extracted pedagogical pattern slots (`to_inf`, `gerund`, `sb_to_do`, `that_clause`, `wh_clause`).
  - Integrated `double_key_pattern` collision gate into distractor candidate filtering, guaranteeing absolute grammatical exclusion and single-fit discrimination.
- [x] **Deterministic Extraction Architecture**:
  - Full migration of Vocabulary, Expressions, and Grammar extraction from LLM to 100% deterministic Python/NLP logic (`LinguisticEngine` + `cefrpy` + Pedagogical Lexicon / WordNet + spaCy).
  - 0 token cost, 0 hallucinations, and eliminated secondary-turn drift.
- [x] **Sense-First Unified Architecture & Atomic Pattern-Example Binding**:
  - Headwords contextually locked to authoritative pedagogical senses via `all-MiniLM-L6-v2` embeddings / WSD scoring.
  - Linked native definition, patterns, collocations, and attested examples into unbreakable atomic blueprints.
- [x] **Level 1 Deterministic Code Gate & Level 2 Expert Model Semantic Audit**:
  - Two-level quality audit architecture across all 5 quiz modalities (vocabulary, reading, translation, listening, video).
  - Physical invariant validation, target coverage gating, automatic category re-mapping, and option shuffling.
- [x] **Unified One-Shot Direct JSON Structured Output**:
  - Thoroughly validated across 11 benchmark models; complete elimination of prose-to-JSON and multi-turn drift.
- [x] **CEFR Deterministic Grounding & Calibration**:
  - Statistically computed text-level ceilings via `cefrpy`; offline Zipf distractor frequency gates; dynamic prompt parameter interpolation.

### In Progress
- **Grammar Code-Driven Extraction Optimization (语法代码化提取深化与回归覆盖)**:
  - Continue refining structural pattern rules and COBUILD formulas across complex rhetorical layers and discourse transitions.
  - Strengthen discourse-level clause boundary detection and stance/hedging classification.

### Pending / Next Improvement Directions (后续演化方向)
- **1. 多词表达式代码化提取双轨制演化（Expressions & Phrasal Verbs Dual-Track Grounding）**:
  - **短语倒排索引对齐（pedlex_phrase_index）**：
    - 采用“双轨制索引 + 规范形态学归一化（Dual-Track Normalization）”消除单纯去标点造成的语义误判（如 `don't` vs `dont`，占位符 `[sb]`/`[sth]` 规范化）。
    - 建立三级渐进漏斗检索机制：一级精确字面原貌匹配（100% 置信度） ➔ 二级规范骨架插槽匹配（95% 置信度） ➔ 三级词形还原与语法容差回退。
  - **基于内存与条目结构的轻量互补提取**：依托 `_phrase_evidence_cache` 与条目原生结构，配合 spaCy 语法依存与词形还原，实现动介短语、习语的精准识别与类型推断。
- **2. 教学法与 Markdown 卡片演化（Source-to-Wiki Enrichment）**:
  - Markdown 词汇卡片自动注入权威教学级搭配列表（0 Token 消耗）。
  - 优质错题本模板与知识图谱交互可视化。 