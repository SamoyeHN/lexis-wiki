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

### 2.2 Table: `ldoce_phrase_index` (Deprecated / 已废弃)
> [!NOTE]
> **Deprecation Notice (方案 A)**：经实测，该表在历史构建中仅抓取了部分义项与短语行，缺失大量搭配词框（如 `worry about`, `keep silent`, `make choices` 等核心教学短语均为 0 命中），且当前 `librarian/` 代码完全不依赖该表（引擎由 `_phrase_evidence_cache` 与条目 `data_json` 动态支持）。该表已被标记为废弃沉没资产，计划后续从数据库中剥离，避免产生误导。

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

### 3.1 Sense Object Format
```json
{
  "definition": "to leave someone, especially someone you are responsible for",
  "pos": "verb",
  "signpost": "LEAVE SOMEBODY",
  "gram": "[transitive]",
  "examples": [
    "How could she abandon her own child?"
  ],
  "patterns": [
    "abandon somebody to something"
  ]
}
```

---

## 4. Derived Forms & Ingestion Invariants

1. **Run-on Derivation Tracking**:
   - Words such as `abandonment`, `punctuality`, and `abductor` are filed in LDOCE print editions as run-on derivations at the end of root entries (`abandon`, `punctual`, `abduct`).
   - `build_ldoce_db.py` tags these entries as `kind='derived'` with `base_word` pointing to the root lemma.
2. **Headword Example Alignment Gate**:
   - A derived row must **never** inherit an example sentence from the base lemma that fails to contain the derived headword (e.g. *abandonment* must not ship with *"How could she abandon her own child?"*).
   - If no headword-bearing example exists in the LDOCE entry, [`LinguisticEngine`](file:///E:/teacher-wiki/librarian/linguistics.py) drops the misaligned example and falls back to the authentic passage sentence.

---

## 5. Query Interfaces (`LinguisticEngine`)

- `LinguisticEngine.get_ldoce_entry(word)`: Fetches parsed dictionary payload from SQLite.
- `LinguisticEngine.get_ldoce_definition_and_example(word, target_pos, context_sentence)`: Retrieves sense-locked definition and headword-bearing example.
- `LinguisticEngine.find_ldoce_collocations(word, pos)`: Returns categorized collocation buckets.
- `LinguisticEngine.get_ldoce_grammar_alert(word, quote)`: Returns targeted learner pitfall warnings.
