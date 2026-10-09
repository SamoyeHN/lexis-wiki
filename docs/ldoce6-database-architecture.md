# LDOCE 6th Edition Database & Linguistic Lexicon Architecture

> **Purpose**: This document archives the database schema, storage layout, parsing mechanics, and query interfaces of the Longman Dictionary of Contemporary English 6th Edition (`ldoce6_essential.db`), which serves as the deterministic semantic truth foundation for Lexis Wiki.

---

## 1. Storage Overview

The primary offline dictionary database resides at `librarian/data/ldoce6_essential.db`.
An accompanying full-text search index resides at `librarian/data/ldoce6_fts.db`.

Both databases are purely local SQLite files queried synchronously by [`LinguisticEngine`](file:///E:/teacher-wiki/librarian/linguistics.py) with zero external network requests and <1ms latency per lookup.

---

## 2. Table Schemas (`ldoce6_essential.db`)

### 2.1 Table: `ldoce`
Main lexical entries table. Contains primary headwords, derived run-ons, and alias redirects.

```sql
CREATE TABLE ldoce (
    word TEXT PRIMARY KEY,       -- Normalized lookup key (lowercase, stripped of punctuation)
    pos TEXT,                    -- Canonical primary part of speech ('noun', 'verb', 'adjective', etc.)
    pos_all TEXT,                -- Comma-separated list of all attested parts of speech across senses
    kind TEXT NOT NULL,          -- Entry category: 'article' | 'derived' | 'alias' | 'entity'
    base_word TEXT,              -- Parent lemma/headword for derived or alias rows (e.g. 'abandon' for 'abandonment')
    data_json TEXT               -- Complete JSON payload containing rich lexicographical structures
);
```

### 2.2 Table: `ldoce_phrase_index` (Active Pedagogical Phrase & Distractor Index)
Contains 106,738 authentic multi-word phrase patterns catalogued across LDOCE 6th Edition entries (covering 26,538 unique normalized phrase units in `_PHRASE_INDEX_CACHE`).

```sql
CREATE TABLE ldoce_phrase_index (
    phrase_key TEXT NOT NULL,    -- Normalized phrase lookup key
    tier TEXT NOT NULL,          -- Index tier: 'phrasal_verb' | 'phrase' | 'collocation'
    host_word TEXT NOT NULL,     -- Host headword where this phrase is catalogued in LDOCE
    sense_idx INTEGER,           -- Sense index under the host headword (if attached to a specific sense)
    raw_phrase TEXT NOT NULL,    -- Exact raw title / pattern as printed in LDOCE (with slots like 'sb/sth')
    PRIMARY KEY (phrase_key, tier, host_word, sense_idx, raw_phrase)
);
```

### 2.3 Table: `ldoce_sense_cue_index` (Sense-Collocation Reversing Lookup Index)
Contains 813,805 cue-to-sense inverted mappings across all 48,960 LDOCE headwords. Formatted as an ultra-compact `WITHOUT ROWID` index to accelerate offline WSD:

```sql
CREATE TABLE ldoce_sense_cue_index (
    host_word TEXT NOT NULL,     -- Target headword owning the sense
    cue_word TEXT NOT NULL,      -- Contextual cue token (signpost, pattern, example token, def keyword)
    sense_idx INTEGER NOT NULL,  -- 0-indexed sense position under the host headword
    weight INTEGER NOT NULL,     -- Cue salience weight (Signpost=30, Pattern=25, Example Colloc=20, Def=15)
    PRIMARY KEY (cue_word, host_word, sense_idx)
) WITHOUT ROWID;
```

> [!TIP]
> **Active Role in Sub-Millisecond WSD & Reversing Lookup**:
> Allows instant retrieval of sense candidates matching contextual tokens from the passage. Queries `SELECT host_word, sense_idx, weight FROM ldoce_sense_cue_index WHERE cue_word = ? AND host_word = ?` in sub-millisecond offline lookup without parsing full entries.

---

## 3. Data JSON Payload Structure (`ldoce.data_json`)

Every record's `data_json` contains a structured dictionary with the following core fields:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `word` | `str` | Normalized headword. |
| `pos` | `str` | Canonical part of speech (`noun`, `verb`, `adjective`, `adverb`, etc.). |
| `kind` | `str` | `article` (independent entry), `derived` (run-on form), `alias` (cross-reference), `entity`. |
| `base_word` | `str` \| `null` | Base lemma when derived from another headword. |
| `payload_headword`| `str` | Exact headword text as rendered in LDOCE source. |
| `all_poses` | `list[str]` | All attested parts of speech across all senses. |
| `homographs` | `list[dict]` | Information on distinct homograph numbers (e.g., *lead¹* vs *lead²*). |
| `senses` | `list[dict]` | Itemized sense definitions, frequency markers, gram patterns, and examples. |
| `grammar_boxes` | `list[dict]` | Pedagogical diagnostic alerts (`Don't say: ✗ ...`, common learner errors). |
| `language_activator` | `list[dict]` | Semantic concept networks, fine-grained synonym distinctions, and cluster nodes. |
| `thesaurus` | `list[dict]` | Synonym and antonym discrimination notes. |
| `collocations` | `dict[str, list]` | Collocation boxes grouped by bucket (`nouns`, `verbs`, `adjectives`, `phrases`). |
| `cross_collocations` | `list[dict]` | Collocations cross-referenced from other entries where this headword appears. |
| `word_family` | `dict[str, list]` | Morphological family groupings (`noun`, `verb`, `adjective`, `adverb`). |
| `phrasal_verbs` | `list[dict]` | Phrasal verbs catalogued under this verb entry. |
| `phrases` | `list[dict]` | Fixed idioms and phrase patterns catalogued under this entry. |
| `entry_status` | `str` | Provenance status (`active`, `derived`, `repaired`). |

### 3.1 Sense Object Format (with Subsense Unpacking & Register Metadata)
```json
{
  "definition": "to leave someone, especially someone you are responsible for",
  "pos": "verb",
  "signpost": "LEAVE SOMEBODY",
  "gram": "[transitive]",
  "sensenum": "1a)",
  "register": "formal",
  "variety": "British English",
  "examples": [
    "How could she abandon her own child?"
  ],
  "patterns": [
    "abandon somebody to something"
  ]
}
```

### 3.2 Subsense Structural Governance & Payload Deduplication
1. **Subsense Hierarchy Flattening**: Sub-senses (`subsense`, `1a`, `1b`) are unpacked into first-class citizens in `senses[]`, deterministically inheriting `signpost`, grammar traits, and semantic context from their parent sense elements.
2. **Explicit Register & Variety Fields**: Stylistic registers (`[spoken]`, `[formal]`, `[literary]`) and geographical dialect varieties (`British English`, `American English`) are stripped of styling noise and persisted as explicit queryable keys.
3. **Language Activator Multi-POS Deduplication**: Employs a strict `seen_la_concepts` uniqueness gate across multi-POS popup entries (e.g. `control`), completely preventing duplicated concept blocks (e.g. shrinking redundant concept clusters from 31 down to 17 unique blocks).
4. **Collocations Inter-Tier Deduplication**: Evaluates `(collocation, example)` tuples across both lexical entry boxes (`source: entry`) and sense-embedded examples (`source: sense`), permanently eliminating duplicate collocation rows.

---

## 4. Definition-Locked Word Sense Disambiguation (WSD) Engine

### 4.1 Multi-Feature Weighted Gating & Scoring (`LinguisticEngine._lock_sense`)
When extracting vocabulary from source passages, `_lock_sense` grounds each headword to its authentic Longman sense at 0 token cost (<0.5ms):

1. **POS Hard Gate**: Filters candidate senses to match the target part of speech from spaCy POS tagging.
2. **Formulaic Locution & Register Gating**: Detects and enforces conversational/spoken locution markers (`[spoken]`, `[greeting]`, etc.). If the target context is an isolated utterance or greeting (e.g., *"Hello, stranger!"*), formulaic senses take precedence; otherwise, formulaic senses are strictly excluded in expository/academic prose.
3. **Pattern & Syntactic Slot Intersection (+3.0 pts)**: Lemmatizes context sentences and checks for exact matches against LDOCE `patterns` (e.g. `interest in`, `refrain from`, `abandon sb to sth`).
4. **Collocation Box Intersection (+2.5 pts)**: Searches `collocations` and `cross_collocations` for lexical overlap with passage context tokens.
5. **Contextual Lemmatized Overlap (+1.0 pt per content word)**: Compares open-class passage lemmas with definition content words and authentic example sentences, applying stopword and quoted dialogue filtering.
6. **Signpost & Frequency Hierarchy (+0.5 pt)**: Respects LDOCE canonical sense ordering when context features are evenly matched.

---

## 5. Authentic Multi-Word Phrase Distractor Generation

[`LinguisticEngine.generate_phrase_distractors`](file:///E:/teacher-wiki/librarian/linguistics.py) harnesses `ldoce_phrase_index` to synthesize strictly symmetric, authentic distractors across 4 structural families:

1. **Tail-Sharing Head Noun / Preposition Cluster**:
   - Matches candidate phrases sharing the same terminal preposition or head noun.
   - Example: `peace of mind` $\rightarrow$ `['cast of mind', 'frame of mind', 'state of mind']`
   - Example: `keep in touch with` $\rightarrow$ `['be in touch with', 'get in touch with', 'stay in touch with']`
2. **Light-Verb & Construct Prefix Match**:
   - Matches light-verb constructions with identical verb heads and grammatical frame.
   - Example: `have a try` $\rightarrow$ `['give a try', 'have a ball', 'have a bash']`
3. **Binomial Coordinate Match**:
   - Matches irreversible binomials and coordinated pairs (`A and B`).
   - Example: `pros and cons` $\rightarrow$ `['bits and pieces', 'back and forth', 'give and take']`
4. **2-Word Phrasal Verbs**:
   - Generates dual contrast pools: particle contrasts (e.g. `take off` vs `take in`, `take out`) and core verb contrasts (e.g. `take off` vs `set off`, `call off`). All candidates are strictly validated against `ldoce_phrase_index` existence.

---

## 6. Derived Forms & Ingestion Invariants

1. **Run-on Derivation Tracking**:
   - Words such as `abandonment`, `punctuality`, and `abductor` are filed in LDOCE print editions as run-on derivations at the end of root entries (`abandon`, `punctual`, `abduct`).
   - `build_ldoce_db.py` tags these entries as `kind='derived'` with `base_word` pointing to the root lemma.
2. **Headword Example Alignment Gate**:
   - A derived row must **never** inherit an example sentence from the base lemma that fails to contain the derived headword (e.g. *abandonment* must not ship with *"How could she abandon her own child?"*).
   - If no headword-bearing example exists in the LDOCE entry, [`LinguisticEngine`](file:///E:/teacher-wiki/librarian/linguistics.py) drops the misaligned example and falls back to the authentic passage sentence.

---

## 7. Query Interfaces (`LinguisticEngine`)

- `LinguisticEngine.get_ldoce_entry(word)`: Fetches parsed dictionary payload from SQLite.
- `LinguisticEngine.get_ldoce_definition_and_example(word, target_pos, context_sentence)`: Retrieves sense-locked definition and headword-bearing example via `_lock_sense`.
- `LinguisticEngine.find_ldoce_collocations(word, pos)`: Returns categorized collocation buckets.
- `LinguisticEngine.find_ldoce_zero_collision_anchor(target_word, distractors, pos, quote)`: Deterministically selects an authentic collocation anchor matching only the target headword.
- `LinguisticEngine.generate_phrase_distractors(target_phrase, pos, count)`: Generates 3 structurally symmetric distractors validated against `ldoce_phrase_index`.
- `LinguisticEngine.get_ldoce_grammar_alert(word, quote)`: Returns targeted learner pitfall warnings.

---

## 8. Pending Work: Special Collocational & Multi-Word Phrase Constructions (待办清单)

针对多词短语和复杂搭配模式，量词短语（`a piece/bit/slice of`）已完成**中心词挖空与同族干扰项生成架构**（方案 2）。以下特殊短语搭配族群已列入待实施演进清单（Pending Backlog）：

### 8.1 关联与成对结构 (Correlative / Paired Connectors)
- **典型短语**: `not only... but also...`, `either... or...`, `neither... nor...`, `both... and...`, `as well as...`
- **考核特征与痛点**:
  - 属于跨越从句或并列项的成对结构，不适合整串作为一个 token 挖空。
  - 需要在单项选择中考核第二标志词（如挖空 `also` 或 `nor`）或前后平行句法平衡。
- **拟定方案 (Pending)**:
  - 建立关联连接词蓝图提取器，锁定首标志词（如题干出现 `either`），选项绑定对应关联词与其典型干扰对（如 `[or, and, nor, but]`）。

### 8.2 双词并列固定成语与不可逆短语 (Irreversible Binomials & Coordinate Idioms)
- **典型短语**: `pros and cons`, `ups and downs`, `bits and pieces`, `safe and sound`, `give and take`
- **考核特征与痛点**:
  - 核心成分不可倒置（*cons and pros* 不自然），整体作为一个不可分割的固定名/副词习语。
- **拟定方案 (Pending)**:
  - 已支持在 `ldoce_phrase_index` 中检索匹配同类双词并列作为全短语干扰项；
  - 待拓展中心词挖空模式：题干提供 `pros and ____` 或 `ups and ____`，选项提供对称名词（如 `[cons, gains, evils, losses]` 或 `[downs, falls, lows, drops]`）。

### 8.3 动宾惯用固定搭配 (Idiomatic Verb-Noun Collocations)
- **典型短语**: `pay attention to`, `take advantage of`, `make use of`, `catch sight of`, `keep track of`
- **考核特征与痛点**:
  - 包含轻动词 + 抽象核心名词 + 固定介词。
  - 挖空动词容易与其他同义动词混淆（如 *pay* vs *give/draw attention*），挖空名词容易与近义词混淆。
- **拟定方案 (Pending)**:
  - 依据 LDOCE 语法模式表，支持中心轻动词挖空（`____ attention to` $\rightarrow$ `[pay, draw, give, attract]`）或固定名词挖空（`pay ____ to` $\rightarrow$ `[attention, notice, care, focus]`），并在 Double-Key 门禁中注入动宾语义唯一性约束。

### 8.4 复合介词与介词短语框架 (Prepositional Frame Phrases)
- **典型短语**: `in terms of`, `by means of`, `in spite of`, `on behalf of`, `with regard to`, `in front of`
- **考核特征与痛点**:
  - 结构均为 `介词 + 核心名词 + 介词`（Prepositional Frame）。
  - 若整串挖空，选项过长；若只挖空首/尾介词，属于基础虚词考核。
- **拟定方案 (Pending)**:
  - 采用中心名词挖空策略：`in ____ of`，选项由同框架高频功能名词构成（如 `[terms, spite, respect, light]`），依据上下文语义唯一锁定。

### 8.5 方案 B 模式：多候选干扰项池的代码级精准初筛体系 (Multi-Candidate Distractor Screening Gate)
- **核心定位与前置条件**:
  - 方案 B 给 LLM 提供 4~6 个高质量备选词，让模型根据题干语境自主挑选 3 个并造句。
  - **决定成败的前提**：代码端（0 Token）生成的候选池**必须 100% 合格**，绝对不能夹带不合规杂质（严防错漏）。
- **待沉淀与强化的代码级硬门禁 (Hard Invariants)**:
  1. **词义锁定 (Sense Lock Invariant)**：
     - 单纯按词头查同反义词必然导致多义词漂移（如 `stamp` 漂移成印章/粘土、`interest` 漂移成利息）。必须以课文原句 (`quoted_sentence`) 和词汇表释义 (`definition`) 输入 `_lock_sense`，锁定目标义项对应的 Longman Thesaurus 与 WordNet 同义位。
  2. **词性一致性硬门禁 (POS Symmetrical Gate)**：
     - 彻底杜绝名动同形词的跨词性污染。候选词必须通过 LDOCE 主词性校验（如目标词是抽象名词 `control` 时，坚决过滤掉具有明显动词倾向的 `run`, `supervise`；目标词是副词 `closely` 时，坚决过滤掉形容词或名词）。
  3. **目标词防篡改与唯一解校验 (Target Word Usurpation Defense)**：
     - Prompt 与 Level 1 Code Gate 强力保证：目标词必须作为唯一的正确答案参与选项，严禁模型挑选候选池里的近义词直接替换目标词作为答案。
  4. **CEFR 与 Zipf 难度天花板过滤 (Ceiling & Frequency Gate)**：
     - 候选干扰词必须经过 `cefrpy` + Zipf 频率门禁（A1/A2 不得选入 C1/C2 生僻偏词），确保与目标词处于同等认知难度梯度。

