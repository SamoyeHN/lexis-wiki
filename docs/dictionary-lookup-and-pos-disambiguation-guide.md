# Lexicographical Disambiguation & Dictionary Lookup Guidelines (POS & Sense Invariants)

> **Document Status**: Immutable Architectural Standard (Single Source of Truth)  
> **Target Components**: `LinguisticEngine` (`librarian/linguistics_core/pedlex_db.py`, `wsd_engine.py`, `cefr_and_morphology.py`, `extraction_pipeline.py`, `distractor_engine.py`), `determine_contextual_pos`, `_lock_sense`, `build_top_k_vocab_candidates`, `disambiguate_via_llm`, `get_ldoce_definition_and_example`

---

## 1. Core Philosophy: The Principle of Contextual Grounding

In educational content generation, word meaning, syntactic function, and pedagogical distractors must originate strictly from the **authentic passage sentence (Quoted Sentence)** and its surrounding discourse context (`±1` sentence window).

### 1.1 The Golden Invariant: POS Invariance is Syntactically Bound
1. **The Sentence Dictates the Part of Speech**: A word does not have an inherent, static POS in isolation; its POS is deterministically fixed by its syntactic dependency tree in the source sentence.
2. **Strict POS-First Gating**: A dictionary lookup or sense-locking pass must NEVER search across conflicting POS blocks. If a word is used as a noun in the passage (e.g., *Chinese Seal*), all verb senses in the dictionary are physically invisible to the disambiguation engine.
3. **Zero Heuristic Patching**: No syntactic heuristics based on superficial typographical cues (e.g. adjacent quotation marks, sentence length, punctuation adjacency) may ever override the dependency parse.

---

## 2. Syntactic POS Resolution Hierarchy

Before querying the Pedagogical Lexicon (`pedlex.db`), [`determine_contextual_pos`](../librarian/linguistics_core/cefr_and_morphology.py) resolves the canonical part of speech using the following strict hierarchy:

```
                  [ Authentic Quoted Sentence & Target Token ]
                                      │
                                      ▼
           ┌──────────────────────────────────────────────────────┐
           │ 1. Dependency Tree Role (spaCy POS & Dep)           │
           │    - Subject/Object/Compound Head/Prepositional Obj  │ ──► NOUN / PROPN
           │    - Main Predicate / Clausal Complement (ROOT/advcl)│ ──► VERB
           │    - Nominal Modifier / Predicate Adjective (amod)   │ ──► ADJECTIVE
           │    - Adverbial Modifier (advmod)                     │ ──► ADVERB
           └──────────────────────────────────────────────────────┘
                                      │
                                      ▼
           ┌──────────────────────────────────────────────────────┐
           │ 2. Voice, Coordinate & Predicate Verb Normalization  │
           │    - Passive Voice (auxpass, nsubjpass, VBN + acl)   │ ──► Strictly VERB
           │    - Pure Copular Complement ('be' + VBN as state)   │ ──► Attested ADJ
           │    - Coordinate Predicate Symmetry (V1, V2, and V3)  │ ──► Strictly VERB
           │    - Plural-Subject Base-Verb Rescue (NNS + base V)  │ ──► Strictly VERB
           └──────────────────────────────────────────────────────┘
                                      │
                                      ▼
           ┌──────────────────────────────────────────────────────┐
           │ 3. Dictionary POS Attestation Hard Gate              │
           │    - Does pedlex entry contain senses for this POS?  │
           │      YES ──► Lock to this POS block                  │
           │      NO  ──► Check secondary POS attested in entry  │
           └──────────────────────────────────────────────────────┘
```

### 2.1 Anti-Patterns & Prohibitions (Zero-Tolerance Rules)
- ❌ **Prohibition 1 (Quote Punctuation Guessing)**: Never promote a noun to a verb merely because it is followed by a quote mark (e.g., `'Chinese Seal, Dancing Beijing'` was mistakenly labeled verb because `'` followed *Seal* and *emblem*).
- ❌ **Prohibition 2 (Default POS Override)**: Never allow the dictionary's entry-level default POS (e.g. `seal` entry possessing both `noun` and `verb`) to flip the syntactically determined POS.
- ❌ **Prohibition 3 (Coordinate / Plural-Subject Noun Mis-Tagging)**: When spaCy mis-tags a base-form verb as `NOUN` inside a verb coordination chain (e.g., *"relationships ... bruise, collide, and comfort"*) or immediately following a plural subject (`NNS`) in a relative/subordinate clause without another finite verb, `determine_contextual_pos` and `extract_deterministic_vocabulary` rescue the token to `verb`.

### 2.2 Cross-POS Chameleon & Boundary Disambiguation Gates
[`determine_contextual_pos`](../librarian/linguistics_core/cefr_and_morphology.py) enforces five deterministic syntactic invariants for high-collision cross-POS boundary cases:
1. **Flat Adverbs, `-ly` Adjectives & Prep/Conj/Adv Chameleons**:
   - Distinguishes flat adverbs (`work hard`, `arrive early`, `run fast`) from prenominal/predicative adjectives (`a hard decision`, `early spring`, `feel well`), `-ly` adjectives (`a likely outcome`, `a costly mistake`) from `-ly` adverbs (`will likely remain`, `published weekly`), and prepositions (`since 1985`, `past the hall` + `pobj`) from subordinating conjunctions (`since the treaty was signed` + clausal `mark`/`advcl`) and intransitive particles/adverbs (`has not been seen since`, `flew past overhead`).
2. **Whole-Compound Span Resolution for Hyphenated Words**:
   - Reconstructs split hyphenated tokens (`[tok_0, '-', tok_2, ...]`) and inspects the rightmost constituent head's external dependency arc against PedLex's `comp_poses`, accurately separating nominal uses (`a brief follow-up`, `wait for a grown-up`), attributive modifiers (`a part-time job`, `a follow-up study`), and adverbial modifiers (`works part-time`, `bought it second-hand`, `met face-to-face`).
3. **Post-Possessive `'s` / `s'` Nominal Slot vs. Contracted `'s` (`is` / `has`)**:
   - Locks words governed by a possessive `'s` or plural `s'` (`the government's plan to cut taxes failed`, `the workers' protests`) to `noun` (or `adjective` when modifying a downstream head noun, e.g., `the team's winning streak`), while preserving contracted progressive/perfect predicates (`the company's planning to launch`, `she's left her keys`) as `verb`.
4. **Title Case & Sentence-Initial `PROPN` / `ADJ` / `VERB` De-Biasing**:
   - Bypasses spaCy's `PROPN` (`NNP`/`NNPS`) over-tagging on capitalized heading tokens unless the token is a pure `PERSON` entity lacking common-word entries (`Premier Zhou Enlai`), rescuing Title-Case infinitives (`How to Master Time Management` $\rightarrow$ `verb`), finite verbs (`Why Ancient Civilizations Collapsed` $\rightarrow$ `verb`), and attributive adjectives (`Key Findings`, `Corporate Action` $\rightarrow$ `adjective`).
   - Separates sentence-initial imperative verbs (`Monitor the patient's heart rate` $\rightarrow$ `verb`) from sentence-initial noun adjuncts in clauses that already possess a finite main verb (`Weather conditions deteriorated`, `Silver coins were used` $\rightarrow$ `noun`).
5. **Gerund-Clause Singular-Predicate Agreement Invariant**:
   - Resolves sentence-initial `VBG + Plural Noun (NNS)` ambiguity via subject-verb number agreement: when the main clause finite verb is 3rd-person singular `VBZ` (`Growing vegetables [NNS] in winter requires [VBZ] a greenhouse`), a plural noun cannot be the grammatical subject of `VBZ`, proving the `VBG` heads a singular gerund clause (`verb`); when the finite verb agrees with the plural noun (`Developing countries [NNS] face [VBP] shortages`), the `VBG` is a prenominal participial `adjective`.

---

## 3. Disambiguation & Sense-Locking Protocol by Part of Speech

Once the POS is physically locked, [`_lock_sense`](../librarian/linguistics_core/wsd_engine.py) scores candidate senses within that POS block (and across merged homographs `word_1`..`word_6`) using symbolic syntactic, collocational, and domain features:

```
Symbolic Sense Score = S_Transitivity + S_TypedPattern + S_GramSlot + S_Collocation + S_CueIndex + S_SecondOrderDomain + S_RegisterPenalty
```

- **Stage 1 Fast Path (~71% of items, 0ms / 0 tokens)**: Monosemous entries under the target POS or polysemous entries with an unambiguous syntactic/pattern margin lock directly in Stage 1.
- **Stage 2 1-Token Local LLM Arbitration (~29% of items)**: When Top-$K$ ($K \le 4$) symbolic candidates remain competitive, [`disambiguate_via_llm`](../librarian/linguistics_core/wsd_engine.py) arbitrates among the Top-$K$ candidates via a 1-token (`A`/`B`/`C`/`D`) local LLM call using the quoted sentence and `±1` sentence `discourse_context`. *(Note: The legacy 80MB `all-MiniLM-L6-v2` dense embedding model is permanently removed.)*

### 3.1 Verbs (VERB)
Verbs are governed by valency frames, voice transformations, and argument selectional restrictions.

1. **Voice-Aware Transitivity & Coordinate Object Ellipsis**:
   - In active voice: Direct object patient is governed via `dobj` / `obj`.
   - In passive voice (`was presented to`, `a gift given to sb`): The passive subject (`nsubjpass`) or governing head noun (`acl`) **is the deep direct object**.
   - **Contact Relative Clause Guard**: Only extracts a governing noun as a relative direct object (`rel_dobjs`) when the verb has no explicit `dobj`, is not separated by commas, and the head noun is not an adverbial manner/time/place noun (`way`, `time`, `place`, `reason`, `moment`, `day`). E.g., in *"The way we do now, ..., will collapse"*, `collapse` is never given a false relative object.
   - **Coordinate Predicate Object-Ellipsis Rescue**: In coordinated verb chains (*"bruise, collide, and comfort"*), transitive/ambitransitive senses with conditional `if ...` definitions are shielded from intransitive-only penalties when objects are ellipted across conjuncts.
2. **Bound Preposition, Particle & Typed Slot Pattern Alignment**:
   - Matches the prepositional complement or adverbial particle (`to`, `into`, `up`, `down`, `around`).
   - Slashes in pattern frames (e.g., `come in/into/out of`, `turn around/round/away`) are expanded and evaluated via CQL typed slot matchers (`compile_typed_slot_pattern` / `match_typed_slot_pattern`).
3. **Selectional Restrictions & Stance Alignment**:
   - Evaluates whether the patient/subject matches selectional restrictions (animate human vs. inanimate abstract vs. concrete physical vs. conditional `if a piece of furniture/fruit` / `if a system, idea, or organization` frames).
   - **Pejorative vs. Constructive Stance Gate**: Senses marked `used to show disapproval` (e.g. `commercialize` Sense 0: *"to care more about making money/profits..."*) are penalized in constructive enabling contexts (`help/enable/allow ... commercialize new technologies`).

### 3.2 Nouns (NOUN / PROPN)
Nouns are governed by head-modifier dependency, countability (`[uncountable]`, `[singular]`, `[plural]`), and semantic domain.

1. **Compound Head & Modifier Binding**:
   - If the noun is in a compound (`Chinese Seal`, `climate change`), the paired compound token serves as an authoritative anchor to lock the sense.
2. **Prepositional Complements (`prep` + `pobj`) & Example-Attested Frame Parity**:
   - Bound prepositions following the noun (`commitment to`, `access to`, `interest in`, `paradigm of`) boost patterns explicitly declaring those frames.
   - If a primary sense omits an explicit `patterns` array in the dictionary (e.g., `paradigm` Sense 0: *"a model or example that shows how something works..."*) while its own dictionary examples attest the exact `<headword> <prep>` frame (`a new paradigm of production`), it receives full preposition-frame parity (`+40`) so a literary/rhetorical sibling sense (`a clear example of something`) cannot steal priority.
3. **Proper Noun / Title Apposition Invariance**:
   - Nouns occurring inside capitalized titles or appositives retain their noun categorizations (e.g. *Seal* in *Chinese Seal* refers to an official stamp/mark, not the marine animal or the verb to close).

### 3.3 Adjectives (ADJ)
1. **Attributive vs. Predicative Slot Alignment (`<gram>` Gate)**:
   - Enforces `[only before noun]` (attributive `amod` / `compound` only) vs. `[not before noun]` (predicative `acomp` / `attr` / postpositive only) in both `_lock_sense` (`±35` pts) and distractor screening (`evaluate_syntactic_slot_clash`).
2. **Bound Prepositions**:
   - E.g. *serious about*, *famous for*, *similar to*.

### 3.4 Adverbs (ADV)
1. **Modification Target**:
   - Evaluates whether the adverb modifies a scalar adjective/adverb (degree adverb: *extremely*, *fairly*) or an action verb (manner/frequency: *closely*, *abruptly*, *intensively*).

---

## 4. Universal Domain Register Penalty (Eliminating Outlier Senses)

A major source of sense misallocation is rare specialized jargon stealing priority from high-frequency core educational senses.

### 4.1 Domain Register Filter Rules
Senses marked with specialized narrow domain labels in dictionary data:
- `[medical]` / `[medicine]` (e.g., *present* = show symptoms of disease)
- `[politics]` / `[parliament]` (e.g., *stand for* = run for parliament)
- `[law]` / `[legal]`
- `[nautical]` / `[military]`
- `[railroad]` / `[transport]` (e.g., *online* = on a regular train route)
- Secondary `[technical]` senses (`s_idx > 0`) when no scientific/technological context triggers exist

> [!IMPORTANT]
> **Primary Academic `[technical]` Sense Preservation**:
> Primary (`s_idx == 0`) `[technical]` senses of core academic vocabulary (e.g., `paradigm` Sense 0) are preserved without penalty, and scientific/technological triggers (`technology`, `scientific`, `research`, `theory`, `data`, `system`) satisfy `[technical]` domain checks.

**The Universal Invariant**:
> **Unless the passage context explicitly contains verified domain anchor tokens (e.g., *doctor, patient, hospital, clinic, disease, symptom* for medical), narrow domain senses carrying that label receive an immediate penalty (-35 to -80 points).**

---

## 5. Multi-Word Expressions vs. Single Words (Unified Flat Pool)

As established in the phrasal verb refactor:
1. **No Phrasal Verb Block Monopoly**: Multi-word units (`turn around`, `come in`, `make time`) combine all matching senses from both the base verb entry and the phrasal verb block into a single unified candidate pool.
2. **Single-Fit Scoring**: Senses are scored solely on pattern alignment, voice-aware transitivity, and selectional restrictions. No arbitrary overrides (`body`, `direction`, or block precedence) are permitted.

---

## 6. Audit & Acceptance Checklist

Any pull request or modification touching dictionary lookup must satisfy:
- [ ] **POS Invariance**: Target word POS in extraction matches the syntactic tree in the quoted sentence.
- [ ] **Voice Awareness**: Passive verbs correctly identify their patients without false transitivity penalties.
- [ ] **Domain Grounding**: No specialized jargon sense (`[medical]`, `[politics]`) is locked in general narrative passages.
- [ ] **Two-Stage Neuro-Symbolic WSD Compliance**: ~71% unambiguous items lock at 0 tokens (<1ms) via `_lock_sense`; ~29% polysemous items arbitrate via 1-token `disambiguate_via_llm` (zero dense embedding models).
- [ ] **Full Regression Pass**: 100% pass on all items in `tests/test_linguistics.py`.

---

## 7. Landed Architectural Milestones & Root-Cause Resolutions (已落地架构规范与根因闭环档案)

### 7.1 Database Ingestion: Same-Key Run-On Merging & Multi-POS Preservation (建库同形派生词性合并)
- **Status**: Implemented (已落地)
- **Location**: `scripts/build_ldoce_db.py` in `scan_mdx` & Pass 2 `build_ldoce_db`
- **Root Cause & Impact**:
  - 历史代码在 `scan_mdx` 中执行 `if normalize_key(word) == normalize_key(key): continue`，一刀切过滤了同形 run-on。
  - 导致全库约 1,225 个由主条目声明的次要/派生词性（Same-Key Run-On）被粗暴丢弃，包括：
    - 94 个副词（如 `half price adverb`, `online adverb`, `first-hand adverb`, `all out adverb`, `economy class adverb`）；
    - 531 个专属性形容词（如 `activist adjective [only before noun]` 带例句 *activist groups*, `adhesive adjective`, `adolescent adjective`）；
    - 及数以百计自带真实例证的动词/名词（如 `blitz verb [transitive]` 带例句 *News came that Rotterdam had been blitzed*）。
- **Remedy & Architectural Invariants**:
  1. **同形 Run-On 专门归集 (`same_key_runons`)**:
     - 在 `scan_mdx`（Pass 1）中，当 `normalize_key(word) == normalize_key(key)` 时，不再机械丢弃，而是将声明的 `(word, pos, gram, examples, declared_base_pos)` 归集到 `same_key_runons[normalize_key(key)]`。
  2. **Pass 2 原子级合并 (`merge_payload_rows`)**:
     - 在 Pass 2 处理 `article` / `entity` 主条目时，查询 `same_key_runons`。
     - 为每个同形声明调用 `build_derived_row`，并利用经充分验证的 `merge_payload_rows` 进行原子合并：
       - `pos_all` / `all_poses` 扩充容纳派生新词性（如 `['noun']` $\rightarrow$ `['noun', 'adjective']`）；
       - 派生义项（包含权威释义、语法标签 `[only before noun]` 及真实例句如 `activist groups`）原子追加至主条目的 `senses` 中；
       - 主条目 `kind = 'article'` 与核心首选义项保持不变。
  3. **例句归属互通 (`owns_form` Variant Matching)**:
     - 在 `owns_form` 中支持空格与连字符互通（`half price` $\leftrightarrow$ `half-price`），确保带连字符的派生例句（如 *Half-price tickets will be sold on the day*）能正确归属于对应义项。

### 7.2 Blind WordNet Fallback Elimination (废除 WordNet `syns[0]` 盲选)
- **Status**: Implemented (已落地)
- **Location**: [`librarian/linguistics_core/wsd_engine.py`](../librarian/linguistics_core/wsd_engine.py) (`get_wordnet_synsets`, `disambiguate_wordnet_synset`, `expression_definition_evidence`) & [`librarian/linguistics_core/distractor_engine.py`](../librarian/linguistics_core/distractor_engine.py) (`generate_zero_collision_distractors`)
- **Impact**: 彻底消除 WordNet 无现代语料库频率加权、盲选索引 `[0]` 导致过时/冷门偏门义项中选的顽疾（如 `online` 盲选 19 世纪铁路义项、`program` 盲选非计算机义项）。
- **Remedy & Architectural Invariants**:
  1. **严格 POS 规范化映射与去重 (`get_wordnet_synsets`)**:
     - 传入规范化词性参数（`noun -> n`, `verb -> v`, `adjective -> a, s`, `adverb -> r`），物理隔绝跨词性 Synset。
     - 自动折叠 EWN 与 OEWN (`oewn:2025+`) 跨词典重复条目，保留规范排序。
  2. **纯符号语境与语域消歧 (`disambiguate_wordnet_synset`)**:
     - **Lesk 实词重合与语域惩罚门禁**: 过滤通用停用词后比对课文语境词袋与候选 Synset 的定义、例句、同义词集（lemma names），对无上下文触发词的生僻/历史语域（`railroad`/`railway`/`nautical`）施加惩罚门禁，并对高阶复合核心概念给予精准加权（*注：历史 80MB `all-MiniLM-L6-v2` 稠密向量模型已永久移除*）。
  3. **命题干扰项语境感知透传 (`generate_zero_collision_distractors`)**:
     - 接收 `context_sentence` 参数，生成同级上位词/兄弟干扰项时基于消歧后的目标义项展开，彻底废除 `primary_synset = target_synsets[0]` 的历史偏置。
- **Verification Cases**:
  - `online` (adverb) $\rightarrow$ *through a computer or a computer network* (`pos='r'`)
  - `online` (adjective, *"Students can easily take lessons online."*) $\rightarrow$ *connected to a computer network or accessible by computer* (NOT 19th-century railway route)
  - `program` (noun, *"He wrote a Python program to parse files."*) $\rightarrow$ *(computer science) a sequence of instructions that a computer can interpret and execute*

### 7.3 Selectional Restriction & Argument Structure Gates (论元结构与选择性限制门禁体系)
- **Status**: Implemented (已落地)
- **Location**: [`librarian/linguistics_core/wsd_engine.py`](../librarian/linguistics_core/wsd_engine.py) in `get_wordnet_supersenses` and `_lock_sense`
- **Core Principle**: 动词与形容词的义项不是孤立的，与句中核心论元（主语、动宾、介宾、系表）存在句法-语义兼容性契约。建立按支配力排序的论元门禁阶梯：
  `动宾 (dobj) > 介宾 (prep_pobj) > 系表 (attr/acomp) > 主谓 (nsubj)`
- **Deep Argument Projection for Passive Voice (被动语态深层论元投影)**:
  - 识别 spaCy 被动依存结构（`auxpass`、`aux:pass`、`nsubjpass`、`nsubj:pass`）：
    - 被动主语 `nsubjpass` 映射为深层及物宾语 (`deep_dobj`)，直接送入及物动词配价判定与及物模式匹配，解除及物动词无表面宾语时的 $-80$ 误判惩罚。
    - 介词 `by` 的宾语（`agent` 或 `prep:by -> pobj`，包含多级介词短语嵌套）映射为深层施事主语 (`deep_nsubj`)。
- **Zero-Token Selectional Restriction via WordNet Supersenses (名词顶层语义原语分类)**:
  - 实现零 Token、离线毫秒级工具方法 `LinguisticEngine.get_wordnet_supersenses(word, pos='n')`，提取 WordNet 原生顶级语义分类（如 `noun.person`, `noun.state`, `noun.event`, `noun.group`, `noun.communication`, `noun.possession` 等）。
  - 深层动宾及施事主语自动匹配语义原语：
    - `noun.communication` / `noun.possession` (如 `medal`, `award`) 命中授予/颁奖义项（Sense 6: *give formal award/gift*）。
    - 生僻语域门禁（Rare Domain Gate）：针对 `[medical]`（如 *present illness/symptom*）、`[law]` 标签义项，在上下文无医学/法律触发词时施加 $-50$ 门禁惩罚，彻底消除被动语态误判为医学义项的缺陷。
    - 非人主语/局势与企业细分：针对双及物/作格动词（如 `turn around`），细分 `business/company` (`noun.group`) 与 `situation/crisis` (`noun.state/noun.event`)，精准分流 Sense 28 (*business turns around*) 与 Sense 29 (*situation/crisis turns around*)。

### 7.4 Phrase Constituent Stopwords Isolation (短语动词词素消歧停用词隔离)
- **Status**: Implemented (已落地)
- **Location**: [`librarian/linguistics_core/wsd_engine.py`](../librarian/linguistics_core/wsd_engine.py) in `_lock_sense`
- **Impact**: 消除“自指自嗨”（Self-Booster Bug）。查多词短语动词（如 `turn around`）时，若仅把 `turn` 加入停用词，短语动词块的释义（如 *"if a situation turns around..."*）中的 `around` 会与课文中的 `around` 发生虚假词袋重合骗取 +6 分。
- **Remedy & Architectural Invariants**:
  1. **短语小品词停用词隔离**: `stopwords` 全面扩充介词小品词（`around`, `round`, `away`, `back`, `along`, `across`, `through`, `into`, `past`, `onto`, `alongside`, `behind` 等）。
  2. **消除小品词释义自循环加分**: 阻止介词小品词在词袋重合匹配（`ex_tokens` 与 `s_def_tokens`）中骗取虚假分数。

### 7.5 Slash Pattern Normalization & Literal Primacy (斜杠模式规范化展开与字面义优先律)
- **Status**: Implemented (已落地)
- **Location**: [`librarian/linguistics_core/pedlex_db.py`](../librarian/linguistics_core/pedlex_db.py) (`clean_and_expand_pattern_slashes`, `get_sense_cue_scores`) & [`librarian/linguistics_core/wsd_engine.py`](../librarian/linguistics_core/wsd_engine.py) (`_lock_sense`)
- **Core Principle & Implementation**:
  1. **递归括号斜杠展开 (`clean_and_expand_pattern_slashes`)**:
     - 支持括号内分支的递归展开（如 `turn (your head/face) to do something` 递归展开为 `turn (your head) to do something` 与 `turn (face) to do something`，确保括号完全闭合）。
     - 支持无括号多分支斜杠展开（如 `turn around/round/away` 展开为 `['turn around', 'turn round', 'turn away']`）。
  2. **离线高频上下文线索索引接入 (`get_sense_cue_scores`)**:
     - 在 `_lock_sense` 打分循环中，根据非停用词的语境词直接查询 `pedlex_sense_cue_index`（兼容旧表名 `ldoce_sense_cue_index`）。
     - 为特定领域搭配（如 `losses`, `profits` 对应商业好转 Sense 28；`economy` 对应局势好转 Sense 43）提供第一性原则的语境增益。
  3. **身体动作选择性限制硬门禁 (Selectional Restriction & Subject Gating)**:
     - 当义项 signpost 为 `"your body"` 或释义核心限定为物理身体动作（`"move your body"`）时，要求主语必须为生物体/人类。
     - 若主语为抽象概念、机构或公司（如 `company`, `market`, `economy`），且句中无显式身体部位词（`body`, `head`, `face`, `neck` 等），施加 $-40$ 惩罚硬门禁，有效阻断比喻/企业语境对物理第一义项的误撞。

### 7.6 Two-Stage True Neuro-Symbolic Sense Reranker (二阶段真神经符号词义消歧总体架构)
- **Status**: Core Production Architecture (已全面替代历史 MiniLM 嵌入模型)
- **Location**: [`librarian/linguistics_core/wsd_engine.py`](../librarian/linguistics_core/wsd_engine.py) (`_lock_sense`, `build_top_k_vocab_candidates`, `disambiguate_via_llm`) & [`librarian/linguistics_core/extraction_pipeline.py`](../librarian/linguistics_core/extraction_pipeline.py) (`extract_deterministic_vocabulary`, `extract_deterministic_expressions`)
- **Why Dense Embeddings Were Permanently Removed (为何彻底废除 MiniLM 稠密向量模型)**:
  - 早期曾尝试引入 80MB `all-MiniLM-L6-v2` 句向量模型对候选义项打分，但稠密向量余弦相似度具有严重的**主题域偏置（Thematic/Domain Bias）**：例如会将 *"Punctuality pays!"* 因 `pays` 的词汇联想强行拉向金融付款义项，或抹平及物/不及物、`<gram>` 槽位与褒贬语用立场之间的离散边界。
  - 因此，80MB `all-MiniLM-L6-v2` 模型已被**永久移除**，升级为**纯符号 Top-$K$ 门禁筛选 + 1-Token 本地大模型精确仲裁**的二阶段真神经符号架构（基准测试准确率 100%）。
- **Two-Stage Pipeline Mechanics (二阶段协同机制)**:
  1. **Stage 1 — Symbolic Candidate Pool (`_lock_sense` + `build_top_k_vocab_candidates`, 0ms / 0 tokens)**:
     - 执行句法 POS 硬门禁（含跨同形异义词条 `word_1`..`word_6` 合并消歧）、`<gram>` 槽位约束（`[only before noun]`, `[not before noun]`, `[uncountable]`, `[singular]`, `[plural]`）、CQL 槽位句式匹配（`match_typed_slot_pattern`）、语域与褒贬立场门禁（`[spoken]`, `[medical]`, `used to show disapproval`）、二阶搭配词投影（含跨句 `±1` 句 `discourse_context`）及 `pedlex_sense_cue_index` 倒排线索加权。
     - **~71% 单义项/高置信度快速通道 (Fast Path)**: 目标词性下仅单义项或句式/搭配分差显著的词条，在 Stage 1 以 **0ms、0 Token** 直接锁定。
  2. **Stage 2 — 1-Token Local LLM Arbitration (`disambiguate_via_llm`, ~29% 多义竞争词条)**:
     - 将 Stage 1 筛出的 Top-$K$ ($K \le 4$) 合法词典义项（附带 `(pos)`、`[<gram>]`、完整 `definition`、`[Pattern: ...]` 及典型例句 `(e.g. "...")`）与 `Target Sentence`、`Surrounding Context`（前后各 1 句篇章语境）组装为 ~100 Token 的单项选择 Prompt（选项 `A`/`B`/`C`/`D`）。
     - 本地 LLM 仅受控输出 **1 个选项字母 Token**（`temperature=0.0`，耗时 ~0.28s–0.40s），**零自由生成权限**，100% 锚定于 `pedlex.db` 权威义项。
     - 内置 SHA-1 持久化内存缓存（`_wsd_llm_cache`）与单元级聚合审计日志（`wsd_log_session` $\rightarrow$ `logs/<timestamp>_wsd_arbitration_<unit>.log`）。

```text
[课文原句 (Quoted Sentence) + 前后句篇章语境 (Discourse Context) + 目标词]
          │
          ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 1: Symbolic Gate & Scoring (_lock_sense)                 │
│          (0 Token · <1ms · 离散句法与词典约束层)                │
├────────────────────────────────────────────────────────────────┤
│ 1. 词性与同形异义门禁 (POS Hard Gate & Homograph Merge)         │
│ 2. 句法槽位约束 (<gram>: [only/not before noun], [uncountable]) │
│ 3. 被动深层论元与并列谓语省略投影 (Passive & Coord Ellipsis)    │
│ 4. CQL 类型化句式槽位匹配 (match_typed_slot_pattern)            │
│ 5. 论元选择限制与条件句主语门禁 (Supersenses & 'if ...' Guards) │
│ 6. 短语动词粒子门禁与例句介词框架对等 (Particle & Prep Parity)   │
│ 7. 语域与褒贬立场门禁 ([spoken], [medical], disapproval stance) │
│ 8. 二阶搭配词与篇章窗口投影 (2nd-Order & ±1 Sent Discourse)     │
└────────────────────────────────────────────────────────────────┘
          │
          ├──► [单义项或句式唯一锁定 (~71% Fast Path)] ──► 【直接锁定义项 (0 Token)】
          │
          ▼ (~29% 多义竞争词条: Top-K Symbolic Candidate Pool, K ≤ 4)
┌────────────────────────────────────────────────────────────────┐
│ Stage 2: 1-Token Local LLM Arbitration (disambiguate_via_llm)  │
│          (~100 Prompt Tokens · 1 Output Token 'A'/'B'/'C'/'D') │
├────────────────────────────────────────────────────────────────┤
│ - 零自由编造权限：仅在 Stage 1 筛出的 Top-K 权威义项内做四选一  │
│ - 结合 Target Sentence + Surrounding Context 精确辨析微殊义项  │
│ - SHA-1 缓存复用 + 单元级日志审计 (logs/*_wsd_arbitration*.log) │
└────────────────────────────────────────────────────────────────┘
          │
          ▼
  【最终锁定义项 (Locked Sense)】
```

### 7.7 Database Phrasal Query Ingestion: Phrase Index Truncation Elimination (废除短语检索 `LIMIT 1` 物理截断)
- **Status**: Implemented (已落地)
- **Location**: [`librarian/linguistics_core/pedlex_db.py`](../librarian/linguistics_core/pedlex_db.py) (`lookup_ldoce_phrase_candidates`, `_unit_owning_sense`) & [`librarian/linguistics_core/wsd_engine.py`](../librarian/linguistics_core/wsd_engine.py) (`expression_definition_evidence`)
- **Root Cause Case**: `at hand` (课文 *the business at hand*)
  - 词典中 `at hand` 拥有多个义项：`sense_idx=13` (*likely to happen soon*), `sense_idx=14` (*close to you and available to be used*), `sense_idx=15` (*needing to be dealt with now*, *the business at hand*)。
  - 历史查询写死 `LIMIT 1`，将 `sense_idx=14, 15` 物理截断。
- **Remedy & Architectural Invariants**:
  1. **完整候选池获取 (`lookup_ldoce_phrase_candidates`)**:
     - 彻底废除 `LIMIT 1` 物理截断，覆盖 Exact literal、Canonical slot/token 及 Lemmatized headword/verb fallback 三条轨道，基于内存缓存 `_phrase_by_key_cache` / `_phrase_by_raw_cache` 返回全部合法候选 `(host_word, sense_idx, tier, raw_phrase)`。
  2. **跨义项语境感知消歧 (Context-Sensitive Sense Resolution in `_unit_owning_sense`)**:
     - 在 `_unit_owning_sense` 及 `expression_definition_evidence` 中透传 `context_sentence`，调用 `_lock_sense` 与 `disambiguate_via_llm` 进行语境消歧。
  3. **单元兄弟义项消歧验证 (`_quote_overridden_unit_sense`)**:
     - 对共享声明该短语的 `unit_sibling_senses`，在候选集闭环内按语境精准匹配，杜绝跨义项错配。

### 7.8 Anchor Evidence Sentence Preservation & Anti-Mutilation (课文证据句留存与反抹杀准则)
- **Status**: Implemented (已落地)
- **Location**: [`librarian/linguistics_core/extraction_pipeline.py`](../librarian/linguistics_core/extraction_pipeline.py) (`anchor_evidence_ok`, `tokenize_and_index_sentences`)
- **Root Cause Cases**: `pay` (*Punctuality Pays!*), `cause` (*What are the causes?*)
- **Remedy & Architectural Invariants**:
  1. **句法完整性替代长度一刀切 (Syntactic Completeness over Length)**:
     - 废除机械字数限制，以句子主谓结构的完备性为准绳：只要具备明确的有限谓词（`ROOT` 为 `VERB` / `AUX`）或具备主语与依存论元，即确认为合法证据句。
  2. **加粗标题句式解构与保留 (Markdown Heading Unwrapping)**:
     - 在文本切分（`tokenize_and_index_sentences`）、证据句有效性审计（`anchor_evidence_ok`）与词性消歧（`determine_contextual_pos`）中统一剥离加粗符号，使包含独立谓词的标题（如 `**Punctuality Pays!**`）完整保留为 S-Pool 证据句。

### 7.9 Predicate Verb Rescue against spaCy False Nominal Chunking (谓词动词语法树纠偏与抢救)
- **Status**: Implemented (已落地)
- **Location**: [`librarian/linguistics_core/cefr_and_morphology.py`](../librarian/linguistics_core/cefr_and_morphology.py) (`determine_contextual_pos`) & [`librarian/linguistics_core/extraction_pipeline.py`](../librarian/linguistics_core/extraction_pipeline.py) (`anchor_evidence_ok`, `extract_deterministic_vocabulary`)
- **Root Cause Case**: `function` (*Olympic Games function as the stage where heroes are made, records broken and medals earned.*)
- **Remedy & Architectural Invariants**:
  1. **层级小句有限谓词审计 (Clause-Level Finite Predicate Audit)**:
     - 在 `determine_contextual_pos` 中，区分主干有限动词与嵌套修饰从句动词（`relcl`、`acl`、`amod`）。当且仅当主干层面（`ROOT`、`conj`、`advcl`、`ccomp`）存在其他有限谓词时才判定已有其他谓词。
  2. **谓词动词动态抢救 (Predicate Verb Rescue)**:
     - 当目标词被 spaCy 误标为 `NOUN`/`PROPN` 并作为 `ROOT`，但词典证实其具有 `verb` 词性，且统辖复合主语（`compound`）与介词补足语（`prep: as / for / in`）或直接宾语（`dobj`），动态纠偏词性为 `verb`。

### 7.10 Syntactic Pattern Invariant Gate for Complex Complements (复杂宾语补足语模式硬门禁)
- **Status**: Implemented (已落地)
- **Location**: [`librarian/linguistics_core/wsd_engine.py`](../librarian/linguistics_core/wsd_engine.py) in `_lock_sense`
- **Root Cause Case**: `expect` (*In modern times, don’t expect anyone to feel surprised.*)
- **Remedy & Architectural Invariants**:
  1. **复杂宾补通用句法识别与硬约束**: 覆盖 `v -> dobj + xcomp` 与 `v -> ccomp (nsubj + aux:to)`，对命中 `[V sb to do sth]` 模式的合法义项奖励 $+45\sim+55$ 分，对不具备复杂宾补能力的义项施加 $-30\sim-65$ 结构惩罚。
  2. **非直接引语语域硬门禁 (`[spoken]` Domain Penalty)**: 非对话/非直接引语的书面叙述句对 `[spoken]` 交际套话义项施加 $-50$ 分门禁。
  3. **介词从属子节点停用词门禁**: 泛化时间/名词（`meta_stopwords`: `time`, `day`, `year`, `people`, `thing`）不触发虚假词袋加分。

### 7.11 Automatic Syntactic POS Binding in High-Level API (高层公开接口句法词性自动绑定与零默认违规封杀)
- **Status**: Implemented (已落地)
- **Location**: [`librarian/linguistics_core/wsd_engine.py`](../librarian/linguistics_core/wsd_engine.py) in `get_ldoce_definition_and_example`
- **Remedy & Architectural Invariants**:
  - 凡调用 `get_ldoce_definition_and_example(word, context_sentence=quote)` 未显式传入 `target_pos` 时，自动前置调用 `determine_contextual_pos(word, context_sentence)` 绑定真实句法 POS，彻底终结无 POS 裸查引发的跨词性漂移。

### 7.12 Ambitransitive Valency Arbitration, Selectional Restrictions on Prepositional Complements & Phrasal Particle Gating (两可及物动词配价仲裁、介词补足语选择限制与短语动词副词硬门禁)
- **Status**: Implemented (已落地)
- **Location**: [`librarian/linguistics_core/wsd_engine.py`](../librarian/linguistics_core/wsd_engine.py) in `_lock_sense`
- **Remedy & Architectural Invariants**:
  1. **短语动词粒子强制门禁 (`pv_sense_map`)**: 归属于短语动词块（如 `pay back`, `pay off`）的义项，其副词粒子必须在句中物理出现，否则扣除 60 分。
  2. **介词补足语选择限制语义网**: `pay for [goods/dinner]` 引导至 Sense 0（支付），`pay for [mistakes/sins]` 引导至 Sense 11（受罚），`pay in [currency]` 激活货币支付模式（$+40$）。
  3. **主语语义选择限制**: `pub/shop/business pays` 匹配盈利义项 Sense 7（$+60$）；`punctuality/honesty/effort pays` 匹配回报义项 Sense 6（$+60$）。

### 7.13 Global Generalized Collocation Stopword Gating & High-Degree Graph Pruning (全局超泛化词动态门禁与图节点剪枝)
- **Status**: Implemented (已落地)
- **Location**: [`librarian/linguistics_core/pedlex_db.py`](../librarian/linguistics_core/pedlex_db.py) (`_get_pedlex_stop_cues`, `_get_pedlex_high_freq_colloc_words`, `get_sense_cue_scores`) & [`librarian/linguistics_core/distractor_engine.py`](../librarian/linguistics_core/distractor_engine.py) (`generate_vocab_distractors`)
- **Remedy & Architectural Invariants**:
  - 对跨度超过 500 个词条的超频泛义词（`people`, `time`, `way`, `make`, `take`, `thing` 等 Super-Degree Nodes）建立动态隔离表，从线索倒排索引打分与命题锚点中物理剔除，同时保留句法树直接支配的固有介词（`rely on`, `interested in`）。

### 7.14 Sense-First Native Example Twin-Binding Protocol (Sense 优先与原生配套例句孪生绑定铁律)
- **Status**: Implemented (已落地)
- **Location**: [`librarian/linguistics_core/wsd_engine.py`](../librarian/linguistics_core/wsd_engine.py) (`get_ldoce_definition_and_example`, `_lock_sense`, `ensure_distinct_example`) & [`librarian/linguistics_core/distractor_engine.py`](../librarian/linguistics_core/distractor_engine.py) (`pick_atomic_pattern_bundle`, `build_precomputed_target_skeletons`)
- **Core Principle & Axiom**:
  - 选定 `locked_sense` 后，`definition`、`best_pattern`、`authentic_example` (`Corpus Example Frame`) 与 `context_anchor` 构成不可分割的原子绑定包（`pick_atomic_pattern_bundle`），实现提取与测评之间 `0 DEF_DRIFT`、`0 POS_DRIFT`、`0 PAT_EX_PREP_DESYNC`。

### 7.15 Coordinate Verb Rescue, Contact-Relative Guard, Conditional Subject Restrictions & Stance/Frame Parity (并列动词纠偏、接触关系从句防护、条件主语选择限制与语用/介词框架对等)
- **Status**: Implemented (已落地 — Book 4 全量语义质量审计闭环)
- **Location**: [`librarian/linguistics_core/cefr_and_morphology.py`](../librarian/linguistics_core/cefr_and_morphology.py) (`determine_contextual_pos`), [`librarian/linguistics_core/extraction_pipeline.py`](../librarian/linguistics_core/extraction_pipeline.py) (`extract_deterministic_vocabulary`), & [`librarian/linguistics_core/wsd_engine.py`](../librarian/linguistics_core/wsd_engine.py) (`_lock_sense`)
- **Root Cause Cases & Architectural Invariants**:
  1. **并列谓语对称性与复数主语原形动词抢救 (`bruise` in *"relationships ... bruise, collide, and comfort"*)**:
     - spaCy 常将并列动词链首词或复数主语（`NNS`）后的原形动词误标为 `NOUN`。`determine_contextual_pos` 与 `extract_deterministic_vocabulary` 检查并列连词链（`conj`/`cc`）中的动词对称性以及从句内复数主语是否缺乏其他有限谓词，自动纠偏为 `verb`。
  2. **接触关系从句宾语提取防护 (`collapse` in *"The way we do now, ..., will collapse"*)**:
     - `_lock_sense` 中的关系从句深层宾语提取仅在动词无显式宾语、与先行词之间无逗号隔断、且先行词不属于状语性中心名词（`way`, `time`, `place`, `reason`, `moment`, `day`）时生效，彻底阻断将 `way` 误判为 `collapse` 的及物宾语（从而避免误入 *"fold a chair/table"* 义项）。
  3. **条件定义句 `if ...` 主语选择性限制与并列省略保护 (`collapse`, `bruise`)**:
     - 对词典中以 `if a piece of furniture or equipment...` 或 `if a piece of fruit...` 开头的条件限制义项，当实际主语/宾语不属于家具器械或水果植物时施加 $-45$ 选择限制惩罚；对 `if a system, idea, or organization...` 在抽象系统/方式主语下奖励 $+35$ 分。
     - 当动词处于并列谓语省略宾语结构（`is_coord_predicate_ellipsis`）时，豁免对条件及物/两可义项的不及物惩罚。
  4. **核心学术词首选 `[technical]` 义项保护与例句介词框架对等 (`paradigm of`)**:
     - 首选义项（`s_idx == 0`）携带 `[technical]` 标签时不施加偏门语域惩罚，并扩充 `technology`/`scientific` 等触发词；同时当首选义项未显式声明 `patterns` 数组但其原生 `examples` 中已证实 `<headword> <prep>`（如 `paradigm of`）时，赋予与显式 pattern 同等的 $+40$ 框架加分。
  5. **贬义语用标签 (`used to show disapproval`) 与建设性使能语境门禁 (`commercialize`)**:
     - 当义项释义带有 `used to show disapproval` 或 `care more about making money/profits`，而课文原句处于建设性使能框架（`help/enable/allow ... design, produce, and commercialize new technologies`）且无负面线索时，对该贬义义项施加 $-40$ 惩罚，并对中性商业化义项（`make something available to buyers`）奖励 $+35$ 分。

---

## 8. Deterministic Grammar Pattern Extraction Architecture (确定性句法模式匹配与 COBUILD 画像标准)

> **Document Status**: Standard Operational Architecture  
> **Target Components**: `librarian/grammar_patterns.py`, `LinguisticEngine.extract_deterministic_grammar`, `COBUILD_GRAMMAR_PROFILES`

### 8.1 Zero-Token Syntax Candidate Generation & Title Filtering
1. **Sentence Pool Heading & Title Gate**:
   - 课文标题（如 S-1 冒号隔断标题）由于缺少主谓结构或句末标点，严禁进入句法模式匹配池。
   - `mine_grammar_skeletons` 强制过滤 `sid == "S-1"`、无标点标题及冒号隔断标题，杜绝将标题误提取为非谓语分词伴随状语（彻底解决 U5 #1 标题污染缺陷）。

### 8.2 Strict Syntactic Boundary Constraints
1. **Cleft Sentences vs. Extraposition (强调句 vs. 形式主语句)**:
   - 结构特征区分：强调句必须具备关系代词/从属连词引出从句（`that/who/whom/which/where/when`）。
   - 形式主语疑问/感叹句（如 *“Is it not a joy to have friends come from afar?”*）因引导词为不定式 `to-inf`，被强调句匹配树严格拒绝，避免假强调句提取（彻底解决 U6 #2 误判）。
2. **Antithesis Conjunction Order Validation (对照并列顺序硬约束)**:
   - 对照并列结构 `not A, but B` 强制校验标记顺序：`not_tok.i < but_tok.i`。
   - 颠倒并列句（如 *“...love..., but not too much control”*）被物理排除，避免误匹配为 `[Subject] + [VP], not + [PrepP/NP], but + [PrepP/NP]`（彻底解决 U1 #4 缺陷）。
3. **Complex Transitive Complement POS Adaptation (复合宾语补足语自适应)**:
   - 复合宾语判定根据补足语词性动态调整：当补足语为名词/名词短语（如 *“make the planet a better place”*）时，公式自适应为 `[Subject] + make + [Object] + [Noun/NP]`；为形容词时给出 `+[adj]`（彻底解决 U5 #4 缺陷）。
4. **Participial Adjunct Direction & Aux Isolation (分词伴随状语方向自适应与助动词隔离)**:
   - 过滤带有 `aux` 或 `auxpass` 的限定被动从句（排除 *was presented*），精确捕获独立非谓语分词（*saying*），并根据分词与主句的位置关系自适应公式（如句末分词输出 `[Subject] + [VP], [V-ing Phrase]`，彻底解决 U6 #1 缺陷）。
5. **Conditional Scope Tightening (条件状语从句范围收敛)**:
   - 条件从句树匹配严格限制于 `as long as` 引导的短语，全面剔除普通时间/方式裸 `as` 从句（彻底解决 U2 #3 缺陷）。

### 8.3 Pedagogical COBUILD Profile Standards (易懂仿写例句与误区重构)
1. **Register Alignment with High School Curriculum (例句语域梯次校准)**:
   - 坚决杜绝在基础语法画像中使用高阶晦涩的大学或工程级例句（如遥测数据 *telemetry data*、基准异常 *benchmark anomalies* 等）。
   - 全部 19 个 COBUILD profile 的仿写例句重构为高中生熟悉的志愿服务、环境保护、校园生活、旅行与科技场景，确保学生“一看就懂，一学就会仿写”。
2. **Clean Pitfall Explanations**:
   - 彻底清除 common mistakes 中的草稿痕迹（如 `(omit but)` 注记），确保易错点解析严谨规范。


