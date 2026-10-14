# Lexicographical Disambiguation & Dictionary Lookup Guidelines (POS & Sense Invariants)

> **Document Status**: Immutable Architectural Standard (Single Source of Truth)  
> **Target Components**: `LinguisticEngine`, `ExpressionPatternEngine`, `determine_contextual_pos`, `_lock_sense`, `get_ldoce_definition_and_example`

---

## 1. Core Philosophy: The Principle of Contextual Grounding

In educational content generation, word meaning, syntactic function, and pedagogical distractors must originate strictly from the **authentic passage sentence (Quoted Sentence)**. 

### 1.1 The Golden Invariant: POS Invariance is Syntactically Bound
1. **The Sentence Dictates the Part of Speech**: A word does not have an inherent, static POS in isolation; its POS is deterministically fixed by its syntactic dependency tree in the source sentence.
2. **Strict POS-First Gating**: A dictionary lookup or sense-locking pass must NEVER search across conflicting POS blocks. If a word is used as a noun in the passage (e.g., *Chinese Seal*), all verb senses in the dictionary are physically invisible to the disambiguation engine.
3. **Zero Heuristic Patching**: No syntactic heuristics based on superficial typographical cues (e.g. adjacent quotation marks, sentence length, punctuation adjacency) may ever override the dependency parse.

---

## 2. Syntactic POS Resolution Hierarchy

Before querying the Pedagogical Lexicon (`pedlex.db`), `determine_contextual_pos` resolves the canonical part of speech using the following strict hierarchy:

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
           │ 2. Voice & Ergative Structural Normalization        │
           │    - Passive Voice (auxpass, nsubjpass, VBN + acl)   │ ──► Strictly VERB
           │    - Pure Copular Complement ('be' + VBN as state)   │ ──► Attested ADJ
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
- ❌ **Prohibition 2 (Default Pos Override)**: Never allow the dictionary's entry-level default POS (e.g. `seal` entry possessing both `noun` and `verb`) to flip the syntactically determined POS.

---

## 3. Disambiguation & Sense-Locking Protocol by Part of Speech

Once the POS is physically locked, `_lock_sense` arbitrates among candidate senses within that POS block using four orthogonal linguistic dimensions:

```
Total Sense Score = S_Transitivity + S_Pattern + S_Collocation + S_DomainPenalty + S_Embedding
```

### 3.1 Verbs (VERB)
Verbs are governed by valency frames and voice transformations.

1. **Voice-Aware Transitivity (Passive Voice Equivalence)**:
   - In active voice: Direct object patient is governed via `dobj` / `obj`.
   - In passive voice (`was presented to`, `a gift given to sb`): The passive subject (`nsubjpass`) or governing head noun (`acl`) **is the deep direct object**.
   - **Invariant**: Transitive senses (`[transitive]`, `present sth to sb`) **MUST NOT** be penalized for lacking an active `dobj` when the construction is passive. Instead, the passive patient is evaluated as the direct object argument.
2. **Bound Preposition & Particle Alignment**:
   - Matches the prepositional complement or adverbial particle (`to`, `into`, `up`, `down`, `around`).
   - Slashes in pattern frames (e.g., `come in/into/out of`, `turn around/round/away`) must be cleanly expanded and evaluated per branch.
3. **Selectional Restrictions (Semantic Arguments)**:
   - Evaluates whether the patient/subject matches selectional restrictions (animate human vs. inanimate abstract vs. concrete physical).
   - E.g., `turn around` with human/skateboard subject aligns with physical rotation, whereas economic/corporate subjects align with metaphorical recovery.

### 3.2 Nouns (NOUN / PROPN)
Nouns are governed by head-modifier dependency, countability, and semantic domain.

1. **Compound Head & Modifier Binding**:
   - If the noun is in a compound (`Chinese Seal`, `climate change`), the paired compound token serves as an authoritative anchor to lock the sense.
2. **Prepositional Complements (`prep` + `pobj`)**:
   - Bound prepositions following the noun (`commitment to`, `access to`, `interest in`) boost patterns explicitly declaring those frames.
3. **Proper Noun / Title Apposition Invariance**:
   - Nouns occurring inside capitalized titles or appositives retain their noun categorizations (e.g. *Seal* in *Chinese Seal* refers to an official stamp/mark, not the marine animal or the verb to close).

### 3.3 Adjectives (ADJ)
1. **Attributive vs. Predicative Alignment**:
   - Attributive adjectives (`amod`, e.g. *ancient civilization*) match senses describing inherent properties of nouns.
   - Predicative adjectives (`acomp` / `attr`, e.g. *scooters are common*) match senses describing widespread availability or states.
2. **Bound Prepositions**:
   - E.g. *serious about*, *famous for*, *similar to*.

### 3.4 Adverbs (ADV)
1. **Modification Target**:
   - Evaluates whether the adverb modifies a scalar adjective/adverb (degree adverb: *extremely*, *fairly*) or an action verb (manner/frequency: *closely*, *abruptly*).

---

## 4. Universal Domain Register Penalty (Eliminating Outlier Senses)

A major source of sense misallocation is rare technical or specialized jargon stealing priority from high-frequency core educational senses.

### 4.1 Domain Register Filter Rules
Senses marked with specialized domain labels in dictionary data:
- `[medical]` / `[medicine]` (e.g., *present* = show symptoms of disease)
- `[politics]` / `[parliament]` (e.g., *stand for* = run for parliament)
- `[law]` / `[legal]`
- `[nautical]` / `[military]`
- `[railroad]` / `[transport]` (e.g., *online* = on a regular train route)

**The Universal Invariant**:
> **Unless the passage context explicitly contains verified domain anchor tokens (e.g., *doctor, patient, hospital, clinic, disease, symptom* for medical), senses carrying that domain label receive an immediate penalty (-50 to -80 points).**

This mathematically prevents medical, legal, or parliamentary jargon from hijacking standard curricular texts.

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
- [ ] **Zero Token Cost**: Entire pipeline executes deterministically offline via spaCy + SQLite in <1ms per item.
- [ ] **Full Regression Pass**: 100% pass on all items in `tests/test_linguistics.py`.

---

## 7. Pending Architectural Tasks & Identified Root Causes (待统一部署清单)

### 7.1 Database Ingestion: Same-Key Run-On Merging & Multi-POS Preservation (建库同形派生词性合并)
- **Status**: Implemented (已落地代码实现，等待整库重建触发)
- **Location**: `scripts/build_ldoce_db.py` in `scan_mdx` & Pass 2 `build_ldoce_db`
- **Root Cause & Impact**:
  - 历史代码在 `scan_mdx` 中执行 `if normalize_key(word) == normalize_key(key): continue`，一刀切过滤了同形 run-on。
  - 导致全库约 1,225 个由主条目声明的次要/派生词性（Same-Key Run-On）被粗暴丢弃，包括：
    - 94 个副词（如 `half price adverb`, `online adverb`, `first-hand adverb`, `all out adverb`, `economy class adverb`）；
    - 531 个专属性形容词（如 `activist adjective [only before noun]` 带例句 *activist groups*, `adhesive adjective`, `adolescent adjective`）；
    - 及数以百计自带真实真示例证的动词/名词（如 `blitz verb [transitive]` 带例句 *News came that Rotterdam had been blitzed*）。
- **Remedy & Architectural Invariants**:
  1. **同形 Run-On 专门归集 (`same_key_runons`)**:
     - 在 `scan_mdx`（Pass 1）中，当 `normalize_key(word) == normalize_key(key)` 时，不再机械丢弃，而是将声明的 `(word, pos, gram, examples, declared_base_pos)` 归集到 `same_key_runons[normalize_key(key)]`。
  2. **Pass 2 原子级合并 (`merge_payload_rows`)**:
     - 在 Pass 2 处理 `article` / `entity` 主条目时，查询 `same_key_runons`。
     - 为每个同形声明调用 `build_derived_row`，并利用经充分验证的 `merge_payload_rows` 进行原子合并：
       - `pos_all` / `all_poses` 扩充容纳派生新词性（如 `['noun']` $\rightarrow$ `['noun', 'adjective']`）；
       - 派生义项（包含 WordNet 权威释义、语法标签 `[only before noun]` 及真实例句如 `activist groups`）原子追加至主条目的 `senses` 中；
       - 主条目 `kind = 'article'` 与核心首选义项保持不变。
  3. **例句归属互通 (`owns_form` Variant Matching)**:
     - 在 `owns_form` 中支持空格与连字符互通（`half price` $\leftrightarrow$ `half-price`），确保带连字符的派生例句（如 *Half-price tickets will be sold on the day*）能正确归属于对应义项。
- **Verification Cases & Tests**:
  - `test_same_key_runon_merging_preserves_derived_pos_and_examples`: 验证 `activist`（主条目 `noun`）同形 run-on（`adjective`）合并后 `pos_all` 变为 `['noun', 'adjective']`，且专属例句 `activist groups` 完整保留。
  - `test_scan_and_merge_same_key_runons_from_html`: 验证从真实 LDOCE HTML 标记片段提取到合并组装的全流程。

### 7.2 Blind WordNet Fallback Elimination (废除 WordNet syns[0] 盲选)
- **Status**: Implemented (已落地)
- **Location**: `librarian/linguistics.py` in `get_wordnet_synsets`, `disambiguate_wordnet_synset`, `mine_vocabulary_skeletons`, `expression_definition_evidence`, and `generate_zero_collision_distractors`
- **Impact**: 彻底消除 WordNet 无现代语料库频率加权、盲选索引 `[0]` 导致过时/冷门偏门义项中选的顽疾（如 `online` 盲选 19 世纪铁路义项、`program` 盲选非计算机义项）。
- **Remedy & Architectural Invariants**:
  1. **严格 POS 规范化映射与去重 (`get_wordnet_synsets`)**:
     - 传入规范化词性参数（`noun -> n`, `verb -> v`, `adjective -> a, s`, `adverb -> r`），物理隔绝跨词性 Synset。
     - 自动折叠 EWN 与 OEWN 跨词典重复条目，保留规范排序。
  2. **双轨神经符号上下文消歧 (`disambiguate_wordnet_synset`)**:
     - **符号层 (Lesk-style Token Overlap)**: 过滤通用停用词后比对定义与例句词袋，无触发词时对生僻语域（`railroad`/`railway`/`nautical`）施加惩罚门禁。
     - **神经层 (MiniLM Semantic Ranking)**: 使用 `all-MiniLM-L6-v2` 嵌入模型计算课文语境与候选 Synset 纯释义的余弦相似度，联合打分选出最符合现代语料的权威释义。
  3. **命题干扰项语境感知透传 (`generate_zero_collision_distractors`)**:
     - 接收 `context_sentence` 参数，生成同级上位词/兄弟干扰项时基于消歧后的目标义项展开，彻底废除 `primary_synset = target_synsets[0]` 的历史偏置。
- **Verification Cases**:
  - `online` (adverb) $\rightarrow$ *through a computer or a computer network* (`pos='r'`)
  - `online` (adjective, *"Students can easily take lessons online."*) $\rightarrow$ *connected to a computer network or accessible by computer* (NOT 19th-century railway route)
  - `program` (noun, *"He wrote a Python program to parse files."*) $\rightarrow$ *(computer science) a sequence of instructions that a computer can interpret and execute*
### 7.3 Selectional Restriction & Argument Structure Gates (论元结构与选择性限制门禁体系)
- **Status**: Implemented (已落地)
- **Location**: `librarian/linguistics.py` in `get_wordnet_supersenses` and `_lock_sense`
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
- **Verification Cases**:
  - `"A medal was presented to the winner by the mayor."` $\rightarrow$ Sense 6 (*give / formal occasion*)
  - `"The mayor presented an award to the student."` $\rightarrow$ Sense 6
  - `"The student turned around and smiled."` $\rightarrow$ Sense 0 (*your body*)
  - `"The crisis turned around after government intervention."` $\rightarrow$ Sense 29 (*situation/crisis turns around*)
  - `"The company turned around after hiring a new CEO."` $\rightarrow$ Sense 28 (*business turns around*)

### 7.4 Phrase Constituent Stopwords Isolation (短语动词词素消歧停用词隔离)
- **Status**: Implemented (已落地)
- **Location**: `librarian/linguistics.py` in `_lock_sense`
- **Impact**: 消除“自指自嗨”（Self-Booster Bug）。查多词短语动词（如 `turn around`）时，若仅把 `turn` 加入停用词，短语动词块的释义（如 *"if a situation turns around..."*）中的 `around` 会与课文中的 `around` 发生虚假词袋重合骗取 +6 分。
- **Remedy & Architectural Invariants**:
  1. **短语小品词停用词隔离**: `stopwords` 全面扩充介词小品词（`around`, `round`, `away`, `back`, `along`, `across`, `through`, `into`, `past`, `onto`, `alongside`, `behind` 等）。
  2. **消除小品词释义自循环加分**: 阻止介词小品词在词袋重合匹配（`ex_tokens` 与 `s_def_tokens`）中骗取虚假分数。

### 7.5 Slash Pattern Normalization & Literal Primacy (斜杠模式规范化展开与字面义优先律)
- **Status**: Implemented (已落地)
- **Location**: `librarian/linguistics.py` in `clean_and_expand_pattern_slashes`, `get_sense_cue_scores`, and `_lock_sense`
- **Core Principle & Implementation**:
  1. **递归括号斜杠展开 (`clean_and_expand_pattern_slashes`)**:
     - 支持括号内分支的递归展开（如 `turn (your head/face) to do something` 递归展开为 `turn (your head) to do something` 与 `turn (face) to do something`，确保括号完全闭合）。
     - 支持无括号多分支斜杠展开（如 `turn around/round/away` 展开为 `['turn around', 'turn round', 'turn away']`）。
  2. **离线高频上下文线索索引接入 (`get_sense_cue_scores`)**:
     - 在 `_lock_sense` 打分循环中，根据非停用词的语境词直接查询 `ldoce_sense_cue_index`。
     - 为特定领域搭配（如 `losses`, `profits` 对应商业好转 Sense 28；`economy` 对应局势好转 Sense 43）提供第一性原则的语境增益。
  3. **身体动作选择性限制硬门禁 (Selectional Restriction & Subject Gating)**:
     - 当义项 signpost 为 `"your body"` 或释义核心限定为物理身体动作（`"move your body"`）时，要求主语必须为生物体/人类。
     - 若主语为抽象概念、机构或公司（如 `company`, `market`, `economy`），且句中无显式身体部位词（`body`, `head`, `face`, `neck` 等），施加 $-40$ 惩罚硬门禁，有效阻断比喻/企业语境对物理第一义项的误撞。
- **Verification Cases**:
  - `"She turned around and walked away."` $\rightarrow$ Sense 0 (`your body`)
  - `"The company turned around from losses of 1 million to profits."` $\rightarrow$ Sense 28 (商业好转)
  - `"The economy is beginning to pick up."` $\rightarrow$ Sense 43 (局势好转)
  - `"He picked up the phone."` $\rightarrow$ Sense 29 (物理拿起)

### 7.6 Dual-Tier Neuro-Symbolic WSD Strategy (神经符号双层词义消歧总体策略架构)
- **Status**: Core Architecture Strategy (先解决策略再谈战略)
- **Core Principle & Axiom (第一性原则)**:
  > **符号层负责判断：这个义项能不能出现（Legality / Invariants）**  
  > **MiniLM 负责判断：所有合法义项中哪个最自然（Naturalness / Fit）**  
  > 绝对禁止让 MiniLM 在没有符号门禁过滤的全部义项中自由竞争！
- **Historical Context & Dilemma (历史痛点复盘)**:
  - **早期纯规则/配价检索**: 逻辑严密，但在复杂或非典型句式中容易因规则覆盖不全而导致查词频繁失败（Coverage 低）。
  - **后来过早/滥用 MiniLM 语义相似度**: 官方宣传测试集 98% 准确率，但在细粒度学习词典（Learner's Dictionary）中暴露严重灾难——**过早使用 Embedding 直接在全库中计算余弦相似度，彻底抹平了词性、配价（及物/不及物）、论元结构之间的离散硬契约**。
  - **惨烈代价**：把人（生物体）当作局势、把动词当作名词、把及物当作不及物、把医学/政治偏门义项当核心义项，甚至使 `seal`、`common`、`expect`、`swell` 错得离谱。
- **Ironclad Mandate: Postpone Embedding, Enforce Symbolic Gates First (严禁过早使用 Embedding，硬门禁后置打分)**:
  - **Embedding 绝不是判决裁判，而只能是末端润色者**。
  - **必须严格执行符号门禁体系（POS、配价、论元、语态投影、语域、介词小品词），把全库几十个义项收敛为 1~3 个合法的候选池（Legal Candidate Pool）后，才允许 MiniLM 进场打分！**

```text
[课文原句 + 目标词]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Tier 1: Symbolic Gate (符号硬门禁与离散约束层)         │
│         (0 Token · <1ms · 高精度 · 高可解释性)         │
├────────────────────────────────────────────────────────┤
│ 1. 词性门禁 (POS Hard Gate): 句法 POS 一票否决         │
│ 2. 证据句存在性门禁 (Anchor Preservation Gate)         │
│ 3. 谓词动词抢救 (Predicate Verb Rescue)                │
│ 4. 被动深层投影 (Passive Projection: nsubjpass->dobj)  │
│ 5. 句式配价门禁 (Valency/Pattern: trans vs intrans)    │
│ 6. 论元选择限制 (Selectional Restrictions: Supersenses)│
│ 7. 短语动词副词粒子门禁 (Phrasal Particle Gate: pv_map)│
│ 8. 语域偏门惩罚 (Domain Register Penalties: [medical]) │
└────────────────────────────────────────────────────────┘
         │
         ▼
  [合法合规候选池 Legal Candidate Pool (1~3个候选)]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Tier 2: MiniLM Ranking (神经语义排序层)                │
│         (离线 ONNX · <3ms · 判定局部自然度)            │
├────────────────────────────────────────────────────────┤
│ - 严禁全库漫游！仅在合法候选池内部计算语义相似度       │
│ - 局部 Syntactic Subtree 与候选释义/例句向量对齐       │
└────────────────────────────────────────────────────────┘
         │
         ▼
  【最终锁定义项 (Locked Sense)】
```

### 7.7 Database Phrasal Query Ingestion: Phrase Index Truncation Elimination (废除短语检索 LIMIT 1 物理截断)
- **Status**: Implemented (已落地)
- **Location**: `librarian/linguistics.py` in `lookup_ldoce_phrase_candidates`, `_unit_owning_sense`, & `expression_definition_evidence`
- **Root Cause Case**: `at hand` (课文 *the business at hand*)
  - 词典中 `at hand` 拥有多个义项：
    - `sense_idx=13`: *likely to happen soon* (*crisis at hand*)
    - `sense_idx=14`: *close to you and available to be used* (*keep a pen at hand*)
    - `sense_idx=15`: *needing to be dealt with now* (*the business at hand*)
  - `lookup_ldoce_phrase` 在历史 SQL 查询中写死了 `LIMIT 1`，将 `sense_idx=14, 15` 从数据库物理截断，导致引擎永远只能看到即将发生的第 13 义项！
- **Remedy & Architectural Invariants**:
  1. **完整候选池获取 (`lookup_ldoce_phrase_candidates`)**:
     - 彻底废除 `LIMIT 1` 物理截断，覆盖 Exact literal、Canonical slot/token 及 Lemmatized headword/verb fallback 三条轨道，使用 `fetchall()` 返回所有合法候选 `(host_word, sense_idx, tier, raw_phrase)`。
  2. **跨义项语境感知消歧 (Context-Sensitive Sense Resolution in `_unit_owning_sense`)**:
     - 在 `_unit_owning_sense` 及 `expression_definition_evidence` 中透传 `context_sentence`。
     - 若候选池中存在多个候选义项且提供了 `context_sentence`，构建 `mock_entry` 调用 `_lock_sense` 进行语境消歧，根据真实课文语境精准锁定（如针对 *the business at hand* 锁定 Sense 15: *needing to be dealt with now*，针对 *crisis at hand* 锁定 Sense 13: *likely to happen soon*，针对 *help is at hand* 锁定 Sense 14: *close to you and available to be used*）。
  3. **单元兄弟义项消歧验证 (`_quote_overridden_unit_sense`)**:
     - 对共享声明该短语的 `unit_sibling_senses`，在候选集闭环内按语境精准匹配，杜绝跨义项错配。

### 7.8 Anchor Evidence Sentence Preservation & Anti-Mutilation (课文证据句留存与反抹杀准则)
- **Status**: Implemented (已落地)
- **Location**: `librarian/linguistics.py` in `anchor_evidence_ok` & `tokenize_and_index_sentences`
- **Root Cause Cases**: `pay` (*Punctuality Pays!*), `cause` (*What are the causes?*)
  - 现象：曾因短句字数一刀切（`len < 6`）或粗暴抹杀加粗标题，导致 2 词完整主谓句（*Punctuality Pays!*）和 4 词完整疑问句（*What are the causes?*）被机械丢弃，退化为无上下文盲查。
- **Remedy & Architectural Invariants**:
  1. **句法完整性替代长度一刀切 (Syntactic Completeness over Length)**:
     - 废除机械字数限制，以句子主谓结构的完备性为准绳：只要具备明确的有限谓词（`ROOT` 为 `VERB` / `AUX`）或具备主语与依存论元，即确认为合法证据句。
  2. **加粗标题句式解构与保留 (Markdown Heading Unwrapping)**:
     - 在文本切分（`tokenize_and_index_sentences`）、证据句有效性审计（`anchor_evidence_ok`）与词性消歧（`determine_contextual_pos`）中统一剥离加粗符号，使包含独立谓词的标题（如 `**Punctuality Pays!**`）完整保留为 S-Pool 证据句，为后续 `pay` 准确锁定及物/不及物义项（Sense 6: *brings a good result*）提供坚实语境。

### 7.9 Predicate Verb Rescue against spaCy False Nominal Chunking (谓词动词语法树纠偏与抢救)
- **Status**: Implemented (已落地)
- **Location**: `librarian/linguistics.py` in `determine_contextual_pos` & `anchor_evidence_ok`
- **Root Cause Case**: `function` (*Olympic Games function as the stage where heroes are made, records broken and medals earned.*)
  - 现象：spaCy 将大写实体 `Olympic Games` 与 `function` 连在一起，误将 `function` 标为 `NOUN` (ROOT)，`Games` 标为 `compound`。且句尾伴有从句动词（`made`、`broken`、`earned`），导致原版朴素判断“句中已有其他动词”而放弃抢救，直接误判为名词（`noun`）。
- **Remedy & Architectural Invariants**:
  1. **层级小句有限谓词审计 (Clause-Level Finite Predicate Audit)**:
     - 在 `determine_contextual_pos` 中，区分主干有限动词与嵌套修饰从句动词（`relcl`、`acl`、`amod`）。当且仅当主干层面（`ROOT`、`conj`、`advcl`、`ccomp`）存在其他有限谓词时才判定已有其他谓词。
  2. **谓词动词动态抢救 (Predicate Verb Rescue)**:
     - 当目标词被 spaCy 误标为 `NOUN`/`PROPN` 并作为 `ROOT`，但词典证实其具有 `verb` 词性，且：
       - 其自身统辖复合主语（`compound`）与介词补足语（`prep: as / for / in`）；
       - 或其自身统辖直接宾语（`dobj`）；
     - 引擎动态触发 Predicate Verb Rescue，纠偏词性为 `verb`；
  3. **介词框架搭配闭环 (Prepositional Frame Agreement)**:
     - 在 `_lock_sense` 中，`function as [stage]` 命中该动词专属性句式框架（Sense 6: *if something functions as a particular thing...*），以 40+ 分的显著优势精准胜出锁定！

### 7.10 Syntactic Pattern Invariant Gate for Complex Complements (复杂宾语补足语模式硬门禁)
- **Status**: Implemented (已落地)
- **Location**: `librarian/linguistics.py` in `_lock_sense`
- **Root Cause Case**: `expect` (*In modern times, don’t expect anyone to feel surprised.*)
  - 现象：曾错选口语释义 `[spoken] used to introduce or agree with a statement...`（*I expect you're right*）或职责要求义项 `to demand that someone does something because it is a duty`。
  - 原句具有明确的复合宾语不定式结构：`[V] + [NP] + [to-inf]`（`expect anyone to feel surprised`），且前置时间状语 `In modern times`。
- **Remedy & Architectural Invariants**:
  1. **复杂宾补通用句法识别与硬约束 (Complex Complement Structural Gating)**:
     - 在 spaCy 依存树中同时覆盖标准控制动词 `v -> dobj + xcomp` 与小句主谓不定式补足语 `v -> ccomp (nsubj + aux:to)`。
     - 凡句中存在复杂宾补，对命中 `[V sb to do sth]` 模式的合法义项奖励 $+45$ 分；对要求其他具体互斥模式（如双宾语或单纯单宾语且未包含不定式）的义项扣除 30 分，物理封杀不符义项。
  2. **非直接引语语域硬门禁 (`[spoken]` Domain Penalty)**:
     - 凡非对话/非直接引语的书面叙述句，对带有 `[spoken]` 标签的偏门口语套话/交际回应义项（如 *I expect so*, *what do you expect?*）施加 $-50$ 分强门禁，彻底阻断其污染。
  3. **介词从属子节点停用词门禁 (Prepositional Object Stopword Gate)**:
     - 当动词挂载介词依存短语（如时间状语 `In modern times`）时，其介词宾语（`time`）必须通过通用停用词门禁（`meta_stopwords`，包含 time, day, year, people, thing 等泛词）。
     - 禁止无特异度的高频时间词/泛词盲目触发例句或释义匹配加分（避免 `time` 误撞例句 `on time` 产生虚假加分），确保第一义项（`to think that something will happen`）以无可争议的优势胜出。

### 7.11 Automatic Syntactic POS Binding in High-Level API (高层公开接口句法词性自动绑定与零默认违规封杀)
- **Status**: Implemented (已落地)
- **Location**: `librarian/linguistics.py` in `get_ldoce_definition_and_example`
- **Root Cause & Architectural Hazard (历史契约裂痕)**:
  - 过去部分下游调用方或简易测试在调用 `get_ldoce_definition_and_example(word, context_sentence=quote)` 时未显式传入 `target_pos`。
  - 由于未传入 `target_pos`，底层的 `_lock_sense` 将全库词条按词典默认排列进行候选比对。若词典词条的排头义项为名词（如 `swell` 名词在先、`seal` 名词在先、`pay` 名词在先），且原句提供了明确动词语境（如 *“Her heart swelled with pride”*、*“The envelope was sealed with wax”*），未传 POS 会导致模型直接受到词典排头词性的虚假引力拉扯，违反 Section 1.1 "POS Invariance is Syntactically Bound" 的铁律！
- **Remedy & Architectural Invariants**:
  1. **零词条默认词性原则 (Zero Headword POS Default)**:
     - 在高层入口 `get_ldoce_definition_and_example` 处建立自动句法词性前置解析：
       ```python
       if not target_pos and context_sentence:
           inferred_pos = cls.determine_contextual_pos(word, context_sentence)
           if inferred_pos:
               target_pos = inferred_pos
       ```
     - 凡传入了语境证据句 `context_sentence`，一律自动通过 `determine_contextual_pos` 获取真实句法 POS（0 Token，<1ms），彻底终结无 POS 裸查引发的跨词性漂移。
  2. **端到端闭环验证 (End-to-End Verification Cases)**:
     - `swell` (*"Her heart swelled with pride."*) $\rightarrow$ 自动绑定 `target_pos="verb"` $\rightarrow$ 锁定 *to feel very proud, angry etc*，例句命中 *His heart swelled with pride...*
     - `seal` (*"The envelope was sealed with wax."*) $\rightarrow$ 自动绑定 `target_pos="verb"` $\rightarrow$ 锁定及物动词 *to close an envelope, package etc*
     - `common` (*"Electric scooters are common in many cities."*) $\rightarrow$ 自动绑定 `target_pos="adjective"` $\rightarrow$ 锁定形容词 *happening often and to many people*

### 7.12 Ambitransitive Valency Arbitration, Selectional Restrictions on Prepositional Complements & Phrasal Particle Gating (两可及物动词配价仲裁、介词补足语选择限制与短语动词副词硬门禁)
- **Status**: Implemented (已落地)
- **Location**: `librarian/linguistics.py` in `_lock_sense`
- **Root Cause & Architectural Hazard (多义及物两可动词与介词选择限制混淆)**:
  1. **短语动词副词缺失劫持基础及物/不及物义项 (Phrasal Verb Particle Hijacking)**:
     - 在原句 *“Can we pay in dollars?”* 中，由于词库中存在短语动词义项（如 `pay back` 义项 20），其释义/例句中含有金融货币泛词，而原句中并无粒子 `back`。若无粒子硬门禁，短语动词可能以虚假相似度抢占基础动词义项。
  2. **介词选择限制混淆 (Prepositional Selectional Restrictions Desync: `for` vs `in` vs `by`)**:
     - 动词 `pay` 兼具多重不及物与介词补足语用法：
       - `pay in [currency]` (Sense 0: *to give someone money for something you buy or for a service*): 以某种货币形式支付；
       - `pay for [goods/services]` (Sense 0): 为商品/服务付款；
       - `pay for [mistakes/sins]` (Sense 11: *to suffer or be punished for something you have done wrong*): 承担代价/遭受惩罚；
       - `[subject] pays` (Sense 6: *if a particular action pays, it brings a good result or advantage*): 某行为/举措带来回报或好处（如 *Punctuality pays!*、*Does hard work pay?*）；
       - `[business/shop] pays` (Sense 7: *if a shop or business pays, it makes a profit*): 商业实体盈利（如 *If the pub does not start to pay, we will have to sell it.*）。
     - 若仅基于单纯词袋或无差别介词匹配，在面对 `pay for dinner` 时极易误撞 Sense 11（*suffer or be punished*）；在面对 `pay in dollars` 时极易因无宾语而误选不及物盈利义项。
  3. **复杂句法结构优先于单纯词汇联想 (Syntactic Frame Primacy over Cue Words)**:
     - 在复杂补足语句中（如 *“We expect him to arrive tomorrow.”*），虽然嵌套从句动词 `arrive` 在词典线索库中与某个不及物义项（Sense 2: *to believe that someone or something is going to arrive*）具有局部词汇共现（cue score），但其物理句法结构为复合宾语不定式 `[V] + [NP] + [to-inf]`。若缺乏句法框架对非复杂宾补义项的结构性惩罚，单纯词汇共现将压制句法物理真实，导致义项误判。
- **Remedy & Architectural Invariants**:
  1. **短语动词粒子强制门禁 (Phrasal Verb Mandatory Particle Gate)**:
     - 建立 `pv_sense_map`，凡义项归属于词条下的短语动词块（如 `pay back`、`pay off`、`pay out`），其必需副词粒子（`back`、`off`、`out`）必须在语境句子序列中物理显现；否则强力扣除 60 分（`score -= 60`），彻底杜绝短语动词跨界侵蚀基础词条。
  2. **介词补足语选择限制语义网 (Prepositional Selectional Restrictions Semantic Grid)**:
     - 对 `pay for [NP]` 进行补足语宾语语义类型细分：
       - 当介词宾语为金融/商品/服务（`dinner`, `meal`, `ticket`, `food`, `rent`, `drink`, `car`, `coffee` 等），严厉惩罚痛苦惩罚义项（`-75`），强力引导至支付义项 Sense 0；
       - 当介词宾语为错误/罪过/怠慢（`mistake`, `error`, `crime`, `sin`, `carelessness`, `arrogance` 等）或出现修饰副词 `dearly`，强力锁定受惩罚义项 Sense 11（$+60$）；
     - 对 `pay in [currency]` 进行货币形式模式激励：
       - 当介词 `in` 统辖货币/支付方式名词（`dollar`, `euro`, `pound`, `cash`, `coin`, `instalment`），直接激活货币支付模式（$+40$），与 `pay (somebody) in dollars` 模式完全闭环。
  3. **主语语义选择限制 (Subject Valency Selectional Restrictions)**:
     - 当无有效直接宾语且无特定介词补足语时：
       - 若主语为商业实体、店铺、企业（`pub`, `shop`, `business`, `firm`, `store`, `company`, `restaurant`），强力匹配盈利义项 Sense 7（$+60$）；
       - 若主语为抽象行为、品质、动作（`punctuality`, `honesty`, `virtue`, `crime`, `work`, `effort` 等），强力匹配回报义项 Sense 6（$+60$）。
  4. **复杂宾语从句小句主语穿透与句法优先 (Syntactic Frame Primacy & Clausal Subject Penetration)**:
     - 扩展模式匹配中对于及物分离模式的选择限制判定，将控制小句主语（`ccomp`/`xcomp` 内部的 `nsubj`，如 `expect him to arrive` 中的 `him`）无缝纳入 `governing_objs` 检验。
     - 在存在复杂补足语（`has_complex_comp`）的真实语境下，对具有复杂宾补 pattern（`somebody to do`）的义项给予 $+55$ 奖励，对不具备复杂宾补能力的义项施加 $-65$ 结构门禁，确保句法真实压倒局部共现噪音。
- **端到端闭环验证 (End-to-End Verification Cases)**:
  - `Can we pay in dollars?` $\rightarrow$ 锁定 Sense 0 (*to give someone money for something you buy or for a service*)
  - `He will pay for the dinner.` $\rightarrow$ 锁定 Sense 0 (*to give someone money for something you buy or for a service*)
  - `They paid dearly for their mistakes.` $\rightarrow$ 锁定 Sense 11 (*to suffer or be punished for something you have done wrong*)
  - `Punctuality pays!` $\rightarrow$ 锁定 Sense 6 (*if a particular action pays, it brings a good result or advantage for you*)
  - `If the pub does not start to pay, we will have to sell it.` $\rightarrow$ 锁定 Sense 7 (*if a shop or business pays, it makes a profit*)
  - `We expect him to arrive tomorrow.` $\rightarrow$ 锁定 Sense 0 (*to think that something will happen because it seems likely or has been planned*)

### 7.13 Global Generalized Collocation Stopword Gating & High-Degree Graph Pruning (全局超泛化词动态门禁与图节点剪枝)
- **Status**: Implemented (已落地)
- **Location**: `librarian/linguistics.py` in `_get_pedlex_stop_cues`, `_get_pedlex_high_freq_colloc_words`, `get_sense_cue_scores`, & `generate_vocab_distractors`
- **Root Cause & Mathematical Nature (统计学本质与图论出度爆炸)**:
  - 现象：`pedlex.db` 的搭配表与倒排线索表中，诸如 `people`, `time`, `way`, `make`, `take`, `day`, `thing`, `place`, `good` 等高频词跨越了 500~1,000+ 个词条。
  - 数学本质：在语义网络与知识图谱中，此类词节点具有超大的出度（Super-Degree Nodes）。其逆文档频率（IDF）接近 0，信息增益（Information Gain）接近 0。若将其作为特征线索或搭配锚点，会发生严重的“虚假共现引力”，将不同义项、甚至毫无关联的词条强行拉扯在一起（如为 `pay` 强行拉 `people` 或 `money` 造成语义漂移）。
- **Remedy & Architectural Invariants**:
  1. **词典级全局动态高频剪枝门禁 (Dynamic IDF Threshold Gate)**:
     - 引擎在启动与加载反向索引时，对词条出现跨度超过特定阈值（> 500 词条）的泛化搭配词自动建立动态隔离表（`_get_pedlex_high_freq_colloc_words` 与 `_get_pedlex_stop_cues`）。
  2. **区分核心从属介词与弱泛化实词 (Syntactic Role Discrimination)**:
     - 严格区分固有支配介词（如 `rely on` 中的 `on`、`interested in` 中的 `in`）与泛义名词/轻动词。
     - 固有介词由句法树直接保障，而超频泛义实词从线索倒排索引打分与命题锚点中被物理剔除，从数学源头消灭伪搭配与无区分度填空。

### 7.14 Sense-First Native Example Twin-Binding Protocol (Sense 优先与原生配套例句孪生绑定铁律)
- **Status**: Implemented (已落地)
- **Location**: `librarian/linguistics.py` in `get_ldoce_definition_and_example`, `_lock_sense`, & downstream assessment generation
- **Core Principle & Axiom (第一性原则)**:
  - **义项是唯一的真理源头与第一公民**：提取与命题阶段必须使用完全统一的单一套 Sense-First 核心逻辑，严禁在未锁定义项之前去全局漫游检索搭配或例句。
  - **Pattern 与 Example 的孪生约束 (Atomic Binding)**：
    - 选定 Sense 后，该 Sense 原生自带的 `definition`、`patterns`、`examples` 和 `collocations` 构成不可分割的原子集合。
    - 若选择或考核了该 Sense 的某种句式框架（如 `pay (somebody) in dollars`），其供给 LLM 的语料示范例句必须严格出自该 Pattern 原生配套的例句（如 *Can we pay in dollars?*）或包含该句式特征的该 Sense 原生例句。
    - **严禁左右互搏**：绝对禁止考某种句式框架，而展示的例句却来自另一个完全不相干的句式，确保从 Frame、Anchor 到 Distractor 的闭环自洽。

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


