# Pedagogical Lexicon (PedLex) Database & Linguistic Architecture

> **Purpose**: This document archives the database schema, storage layout, parsing mechanics, and query interfaces of the Pedagogical Lexicon (`pedlex.db`), which serves as the deterministic semantic truth foundation for Lexis Wiki.

---

## 1. Storage Overview

The primary offline dictionary database resides at `librarian/data/pedlex.db`.
An accompanying full-text search index resides at `librarian/data/pedlex_fts.db`.

Both databases are purely local SQLite files queried synchronously by [`LinguisticEngine`](../librarian/linguistics.py) with zero external network requests and <1ms latency per lookup.

---

## 2. Table Schemas (`pedlex.db`)

### 2.1 Table: `ldoce` (Primary Pedagogical Lexicon Table)
Main lexical entries table (48,960 consolidated entries after homograph/variant folding). Contains primary headwords (`article`, 41,286 rows), derived run-ons (`derived`, 4,949 rows), alias redirects (`alias`, 1,932 rows), and encyclopedic/named entities (`entity`, 793 rows).

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

### 2.2 Table: `pedlex_phrase_index` (Active Pedagogical Phrase & Distractor Index)
Contains 111,012 authentic multi-word phrase patterns catalogued across Pedagogical Lexicon entries across 4 structural tiers (`phrasal_verb`, `phrase_row`, `sense_unit`, and `colloc_phrase`, with direct 0-based `sense_idx` pointers; also aliased as `ldoce_phrase_index` for backward compatibility).

```sql
CREATE TABLE pedlex_phrase_index (
    phrase_key TEXT NOT NULL,    -- Normalized phrase lookup key
    tier TEXT NOT NULL,          -- Index tier: 'phrasal_verb' | 'phrase_row' | 'sense_unit' | 'colloc_phrase'
    host_word TEXT NOT NULL,     -- Host headword where this phrase is catalogued in PedLex
    sense_idx INTEGER,           -- Direct 0-based sense index under the host headword (dereferenced from sense_num)
    raw_phrase TEXT NOT NULL,    -- Exact raw title / pattern as recorded in PedLex (with slots like 'sb/sth')
    PRIMARY KEY (phrase_key, tier, host_word, sense_idx, raw_phrase)
);
```

### 2.3 Table: `pedlex_sense_cue_index` (Sense-Collocation Reversing Lookup Index)
Contains 226,419 inverted cue-to-sense mappings across PedLex headwords (also aliased as `ldoce_sense_cue_index`). Formatted as an ultra-compact `WITHOUT ROWID` index to accelerate offline WSD:

```sql
CREATE TABLE pedlex_sense_cue_index (
    host_word TEXT NOT NULL,     -- Target headword owning the sense
    cue_word TEXT NOT NULL,      -- Contextual cue token (signpost, pattern, example token, def keyword)
    sense_idx INTEGER NOT NULL,  -- 0-indexed sense position under the host headword
    weight INTEGER NOT NULL,     -- Cue salience weight (Signpost=30, Pattern=25, Example Colloc=20, Def=15)
    PRIMARY KEY (cue_word, host_word, sense_idx)
) WITHOUT ROWID;
```

> [!TIP]
> **Active Role in Sub-Millisecond WSD & Reversing Lookup**:
> Allows instant retrieval of sense candidates matching contextual tokens from the passage. Queries `SELECT host_word, sense_idx, weight FROM pedlex_sense_cue_index WHERE cue_word = ? AND host_word = ?` in sub-millisecond offline lookup without parsing full entries.

### 2.4 Table: `oxford_thesaurus` (Sense-Grouped Synonym & Antonym Companion Table)
Stores fine-grained sense-grouped synonyms and antonyms (17,975 rows) used alongside `ldoce.thesaurus` and `ldoce.language_activator` for distractor generation and double-key collision screening:

```sql
CREATE TABLE oxford_thesaurus (
    word TEXT PRIMARY KEY,       -- Normalized headword key
    pos TEXT,                    -- Primary part of speech
    pos_all TEXT,                -- Comma-separated parts of speech
    sense_count INTEGER,         -- Number of distinct synonym/antonym sense clusters
    synonym_count INTEGER,       -- Total synonym count across senses
    antonym_count INTEGER,       -- Total antonym count across senses
    data_json TEXT               -- JSON payload containing per-sense 'definition', 'synonyms', and 'antonyms'
);
```

### 2.5 Table: `pedlex_cxn_index` (Two-Tier Construction Tagging & Reverse Subcategorization Index)
Contains 2,638 pre-compiled constructional frames and subcategorization equivalence rows across all **Eight Lexico-Syntactic Construction Families** (both entry-level `sense_idx = -1` and sense-specific `sense_idx >= 0`):

```sql
CREATE TABLE pedlex_cxn_index (
    subcat_group TEXT NOT NULL,      -- Syntactic equivalence class ('ADJ_IT_EXTRAPOS', 'V_GERUND_ONLY', 'N_SHELL_TO_DO', etc.)
    family_id TEXT NOT NULL,         -- Construction family ('FAM1A_IT_OBJ_EXTRAPOS', 'FAM3_VERB_NONFINITE_COMP', etc.)
    word TEXT NOT NULL,              -- Normalized headword or multi-word phrase key
    pos TEXT NOT NULL,               -- Canonical POS ('adjective', 'verb', 'noun', 'phrase', etc.)
    sense_idx INTEGER NOT NULL,      -- -1 for entry/phrase-level triggers; >= 0 for sense-specific triggers
    cefr_int INTEGER,                -- Pre-calibrated integer CEFR level (1=A1 .. 6=C2)
    zipf_score REAL,                 -- Word/phrase Zipf frequency score for frequency-ranked SQL retrieval
    canonical_pattern TEXT,          -- Normalized pedagogical pattern ('it is hard (for [sb]) to do [sth]')
    canonical_blank TEXT,            -- Prescribed option surface form ('make it possible', 'not only', 'hard')
    cloze_frame TEXT,                -- Mandatory stem frame ('It is ____ (for [sb]) to do [sth]')
    contrast_groups TEXT,            -- Comma-separated mutually exclusive subcat_groups for distractor lookup
    ban_groups TEXT,                 -- Comma-separated subcat_groups barred from co-occurring as distractors
    PRIMARY KEY (subcat_group, word, sense_idx, canonical_pattern)
);
CREATE INDEX idx_cxn_word ON pedlex_cxn_index(word, sense_idx);
CREATE INDEX idx_cxn_subcat ON pedlex_cxn_index(subcat_group, pos, cefr_int);
```

### 2.6 Table: `pedlex_error_traps` (Minimal-Pair Lexical & Syntactic Error-Injection Index)
Contains 16,759 pre-mined minimal-pair learner errors, valency violations, `<gram>` property traps, and `Grammar Box` `BADEXA` contrasts (`Don't say: ✗ ...`) for deterministic error injection in `translation_quiz` and `vocabulary_quiz`:

```sql
CREATE TABLE pedlex_error_traps (
    word TEXT NOT NULL,              -- Normalized headword or phrase
    pos TEXT NOT NULL,               -- Canonical part of speech
    sense_idx INTEGER DEFAULT -1,    -- Associated sense index (-1 if entry-wide)
    trap_type TEXT NOT NULL,         -- 'GRAMBOX_BADEXA' | 'LEXICAL_VALENCY_FRAME' | 'GRAM_UNCOUNTABLE' | 'GRAM_ATTRIBUTIVE_ONLY' | 'L1_CONJUNCTION_REDUNDANCY' | 'CLAUSE_PREP_CONFUSION'
    forbidden_frame TEXT,            -- Prescribed ungrammatical frame or error pattern ('*originate from/at*', '*Although ..., but ...*')
    bad_example TEXT,                -- Authentic negative example ('✗ They are able to avoid to pay taxes.')
    good_example TEXT,               -- Paired grammatical model sentence ('They are able to avoid paying taxes.')
    rule_note TEXT                   -- Pedagogical diagnostic explanation
);
CREATE INDEX idx_error_traps_word ON pedlex_error_traps(word, pos);
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
| `payload_headword`| `str` | Exact headword text as rendered in PedLex source. |
| `all_poses` | `list[str]` | All attested parts of speech across all senses. |
| `entryhead_gram` | `str` \| `dict` | Entry-level grammatical constraint (`[only before noun]`, `[uncountable]`, etc.). |
| `runon_gram` | `str` | Run-on grammatical constraint for `kind='derived'` entries. |
| `derived_forms` | `list[dict]` | Explicit morphological run-on derivatives (`word`, `pos`, `grammar`, `examples`). |
| `verb_table` | `dict` | Inflectional paradigm (`past`, `past_participle`, `present_participle`, `third_person`). |
| `pronunciations` | `dict` | BrE / AmE IPA transcriptions and audio reference metadata. |
| `cxn_specs` | `list[dict]` | Entry-level constructional frame specs (`family_id`, `subcat_group`, `cloze_frame`). |
| `homographs` | `list[dict]` | Information on distinct homograph numbers (e.g., *lead¹* vs *lead²*). |
| `senses` | `list[dict]` | Itemized sense definitions, frequency markers, `licensed_poses`, `grammar`, `register_tags`, `domain_tags`, `attitude_tags`, `syntax_codes`, `selectional_restrictions`, `synonyms`, `antonyms`, `patterns`, `pattern_bundles`, `collocations`, `cxn_specs`, `crossref_target`, `resolved_from`, and `examples`. |
| `crossrefs` | `list[dict]` | Entry-level and sense-level cross-reference pointers (`raw`, `phrase`, `target_word`, `target_homograph`, `target_sense_num`, `target_sense_idx`, `sense_idx`). |
| `grammar_boxes` | `list[dict]` | Pedagogical diagnostic alerts (`Don't say: ✗ ...`, `forbidden_frames`, `comparison_peers`, `sense_indices`, `homograph_num`, `pos`, `scope`). |
| `language_activator` | `list[dict]` | Semantic concept networks, fine-grained synonym distinctions, and cluster nodes. |
| `thesaurus` | `list[dict]` | Synonym and antonym discrimination notes with per-exponent `grammar`, `concept`, `pos`, `homograph_num`, and `sense_indices`. |
| `collocations` | `dict[str, list]` | Collocation boxes grouped by bucket (`nouns`, `verbs`, `adjectives`, `phrases`), with per-item `gloss`, `sense_idx`, `sense_indices`, `pos`, and `homograph_num`. |
| `cross_collocations` | `list[dict]` | Collocations cross-referenced from other entries (`collocation`, `example`, `gloss`, `sense_idx`, `target_word`, `target_sense_num`, `target_sense_idx`). |
| `word_family` | `dict[str, list]` | Morphological family groupings (`noun`, `verb`, `adjective`, `adverb`). |
| `phrasal_verbs` | `list[dict]` | Phrasal verbs catalogued under this verb entry (with unpacked `1a)`/`1b)` subsenses and `[SIGNPOST]` tags). |
| `phrases` | `list[dict]` | Fixed idioms and phrase patterns catalogued under this entry (`phrase`, `node_word`, `sense_idx`, `definition`, `examples`). |
| `entry_status` | `str` | Provenance status (`active`, `derived`, `repaired`). |

### 3.1 Sense Object Format (with Subsense Unpacking, Multi-POS Licensing, Direct Collocations & Construction Specs)
```json
{
  "definition": "to leave someone, especially someone you are responsible for",
  "pos": "verb",
  "licensed_poses": ["verb"],
  "signpost": "LEAVE SOMEBODY",
  "grammar": "[transitive]",
  "sensenum": "1a)",
  "register": "formal",
  "register_tags": ["formal"],
  "domain_tags": [],
  "attitude_tags": [],
  "syntax_codes": ["[T]"],
  "selectional_restrictions": {},
  "variety": "British English",
  "synonyms": ["desert", "forsake"],
  "antonyms": [],
  "examples": [
    "How could she abandon her own child?"
  ],
  "patterns": [
    "abandon somebody to something"
  ],
  "pattern_bundles": [
    {
      "pattern": "abandon somebody to something",
      "examples": ["They abandoned him to his fate."],
      "gram": "",
      "gloss": ""
    }
  ],
  "collocations": [],
  "cxn_specs": []
}
```

### 3.2 Subsense Structural Governance, Variant Consolidation & Payload Deduplication
1. **Subsense Hierarchy Flattening & Parent Inheritance (`1a`/`1b` Unpacking)**: Sub-senses (`subsense`, `1a`, `1b`, including multi-sense verb entries like `compromise`, `go on`, and `set`) are unpacked into first-class citizens in `senses[]`, deterministically inheriting `signpost`, `grammar` traits, `patterns`, and `pattern_bundles` from their parent sense elements while preserving sense-specific `patterns` and `examples`.
2. **Unnumbered Single-Sense Collocation Attachment**: For unnumbered single-sense entries (`pride`, `remorse`), sense-embedded collocation blocks are accurately attached to `senses[0].patterns` and `senses[0].pattern_bundles` without loss.
3. **British / American Variant Consolidation**: Cross-referenced regional spelling variants (`favour`/`favor`, `honour`/`honor`, `programme`/`program`) consolidate all 16+ senses and idiomatic units seamlessly across both spelling keys.
4. **Structured Pragmatic & Syntactic Metadata**: Stylistic registers (`register_tags`: `["spoken"]`, `["formal"]`, `["literary"]`), domain labels (`domain_tags`: `["medical"]`, `["law"]`), speaker attitude markers (`attitude_tags`: `["disapproving"]`, `["humorous"]`), and conditional subject restrictions (`selectional_restrictions`, e.g., `if a machine/system/plan...`) are extracted into structured queryable fields.
5. **Language Activator Multi-POS Deduplication**: Employs a strict `seen_la_concepts` uniqueness gate across multi-POS popup entries (e.g. `control`), completely preventing duplicated concept blocks (shrinking redundant concept clusters from 31 down to 17 unique blocks).
6. **Collocations Inter-Tier Deduplication**: Evaluates `(collocation, example, pos)` tuples across both lexical entry boxes (`source: entry`) and sense-embedded examples (`source: sense`), permanently eliminating duplicate collocation rows.

### 3.3 PedLex v2 New Entry & Sense Attributes and Their Runtime Applications
Built and verified via [`scripts/build_ldoce_db.py`](../scripts/build_ldoce_db.py) and [`scripts/upgrade_pedlex_db.py`](../scripts/upgrade_pedlex_db.py), PedLex v2 introduces seven structural attribute families that eliminate runtime heuristic guessing:

1. **Direct 0-Based `sense_idx` Pointers (`collocations`, `cross_collocations`, `phrases`, `pedlex_phrase_index`)**:
   - **Attribute Schema**: Every item in `collocations[bucket]`, `cross_collocations`, `phrases`, and `pedlex_phrase_index` carries an integer `sense_idx` (and `sense_indices: list[int]` when a collocation box spans multiple meanings such as `"Meanings 1 & 2"`), dereferenced offline from raw dictionary sense numbers (`(sense 3)` $\rightarrow$ `sense_idx = 2`).
   - **Runtime Application**:
     - [`PedlexDbMixin.lookup_ldoce_phrase_candidates`](../librarian/linguistics_core/pedlex_db.py) and [`ldoce_phrase_hits`](../librarian/linguistics_core/pedlex_db.py) directly resolve `senses[sense_idx]` in $O(1)$ without scanning all senses or guessing via string overlap.
     - [`CollocationsAndCardsMixin.get_vocab_card_enrichment`](../librarian/linguistics_core/collocations_and_cards.py) filters collocations by matching `item.sense_idx == locked_sense_idx` (or `locked_sense_idx in item.sense_indices`), guaranteeing sense-isolated collocation display on vocabulary cards.

2. **Inline Collocation `gloss` & `colloc_phrase` Index Tier (`peace of mind`)**:
   - **Attribute Schema**: Parenthetical glosses inside collocation boxes (`<span class="collgloss">`, e.g., `peace of mind (=a feeling of calm and not being worried)`) are extracted into `item["gloss"]` on `collocations`, `cross_collocations`, and `senses[i].collocations`, and indexed in `pedlex_phrase_index` under `tier = 'colloc_phrase'` (adding 4,274 attested collocation phrases).
   - **Runtime Application**:
     - [`WsdEngineMixin.expression_definition_evidence`](../librarian/linguistics_core/wsd_engine.py) and [`ldoce_phrase_hits`](../librarian/linguistics_core/pedlex_db.py) promote `item["gloss"]` directly as the authoritative idiom definition when a multi-word expression (such as `peace of mind`) is catalogued inside a collocation box rather than a standalone headword entry, while pairing it with the collocation's own `example` (`"Having insurance gives you peace of mind."`).

3. **Structured Cross-Entry `crossrefs` (`pros and cons`, Stub Sense Dereferencing)**:
   - **Attribute Schema**: Entry-level, tail-level, and sense-level `<span class="crossref">` elements are parsed into `entry["crossrefs"]`:
     ```json
     {
       "raw": "the pros and cons at pro1(3)",
       "phrase": "the pros and cons",
       "target_word": "pro",
       "target_homograph": 1,
       "target_sense_num": "3",
       "target_sense_idx": 2,
       "sense_idx": null
     }
     ```
     Furthermore, stub senses that consist solely of a cross-reference pointer have their `definition` and `examples` automatically populated from `target_word.senses[target_sense_idx]` (`resolved_from`, `crossref_target`).
   - **Runtime Application**:
     - When `expression_definition_evidence` or `ldoce_phrase_hits` looks up `pros and cons` (whose token `cons` lemmatizes to `con`), it follows `con.crossrefs` directly to `pro` (`sense_idx = 2`: *"the advantages and disadvantages of something"*, example *"We weighed up the pros and cons."*), resolving cross-referenced idioms with 100% accuracy.

4. **Multi-POS Entryhead `licensed_poses` (`half-price`, `all-out`, `first-hand`)**:
   - **Attribute Schema**: When an entry header declares multiple parts of speech (`<span class="pos">adjective, adverb</span>`), `homographs[h].all_poses` and every sense's `licensed_poses` record all licensed POS tags (`["adjective", "adverb"]`).
   - **Runtime Application**:
     - [`CefrAndMorphologyMixin.determine_contextual_pos`](../librarian/linguistics_core/cefr_and_morphology.py) and [`WsdEngineMixin._lock_sense`](../librarian/linguistics_core/wsd_engine.py) check `sense.get("licensed_poses")` during POS hard gating so multi-POS entries are never falsely blocked when used in their secondary declared POS.

5. **DOM Child-Span Leak Isolation (`variety` & `register` Cleanup)**:
   - **Attribute Schema**: During HTML ingestion, sense-level `variety` (`<span class="geo">`) and `register` (`<span class="registerlab">`) are extracted strictly from direct sense-header children (`recursive=False`), excluding nested `variants`, `lexvar`, `exa`, `collo`, and `grambox` spans.
   - **Runtime Application**: Prevents variant-specific labels (such as `"British English"` on the variant *trick-or-treating* inside `trick or treat`) from leaking onto the entire sense.

6. **Offline Derived-Word Sense Bridging (`gloss_source = "base_sense_bridge"`)**:
   - **Attribute Schema**: During `upgrade_pedlex_db.py`, all 4,949 `kind='derived'` rows (`-ly`, `-ness`, `-ity`, `-ment`, `-ation`) are pre-bridged with morphological definitions and headword-attested examples harvested from their `base_word` entry (`derived_forms`, `language_activator`, `thesaurus`, `collocations`, and `senses`).
   - **Runtime Application**: Works in tandem with runtime `_resolve_ldoce_kind` (Section 6) so derived words carry immediate non-empty definitions and examples in SQL queries and card enrichment.

---

## 4. Definition-Locked Word Sense Disambiguation (WSD) Engine

### 4.1 Multi-Feature Weighted Gating & Scoring (`LinguisticEngine._lock_sense`)
When extracting vocabulary from source passages, `_lock_sense` grounds each headword to its authentic pedagogical sense at 0 token cost (<0.5ms) for ~71% monosemous/unambiguous cases, escalating to 1-token Local LLM arbitration (`disambiguate_via_llm`) for the remaining ~29% polysemous cases:

1. **POS & Homograph Hard Gate**: Filters candidate senses to match the target part of speech and resolves cross-homograph entries (`word_1`..`word_6`).
2. **Syntactic Slot & `<gram>` Alignment**: Checks positional adjective constraints (`[only before noun]` vs. `[not before noun]`), noun countability (`[uncountable]`/`[singular]`), and verb transitivity (`[intransitive]` vs. `[transitive]`) against the spaCy dependency parse of the quoted sentence.
3. **Formulaic Locution & Register Gating**: Detects and enforces conversational/spoken locution markers (`[spoken]`, `[greeting]`, etc.). If the target context is an isolated utterance or greeting (e.g., *"Hello, stranger!"*), formulaic senses take precedence; otherwise, formulaic senses are strictly excluded in expository/academic prose.
4. **Pattern & Syntactic Slot Intersection (+3.0 pts)**: Lemmatizes context sentences and checks for exact matches against PedLex `patterns` (e.g. `interest in`, `refrain from`, `abandon sb to sth`).
5. **Collocation Box Intersection (+2.5 pts)**: Searches `collocations` and `cross_collocations` for lexical overlap with passage context tokens.
6. **Contextual Lemmatized Overlap (+1.0 pt per content word)**: Compares open-class passage lemmas with definition content words and authentic example sentences, applying stopword and quoted dialogue filtering.
7. **Signpost & Frequency Hierarchy (+0.5 pt)**: Respects PedLex canonical sense ordering when context features are evenly matched.

---

## 5. Authentic Multi-Word Phrase Distractor Generation

[`LinguisticEngine.generate_phrase_distractors`](../librarian/linguistics.py) harnesses `pedlex_phrase_index` to synthesize strictly symmetric, authentic distractors across 4 structural families:

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
   - Generates dual contrast pools: particle contrasts (e.g. `take off` vs `take in`, `take out`) and core verb contrasts (e.g. `take off` vs `set off`, `call off`). All candidates are strictly validated against `pedlex_phrase_index` existence.

---

## 6. Derived-Word Completion & Alias Resolution Architecture (`派生词补全架构`)

In raw learner dictionaries, **5,678 run-on derived words** (such as `-ly` adverbs, `-ness`/`-ity` state nouns, `-ization`/`-ation`/`-ment` process nominalizations, and `-ing` participial adjectives/nouns) are stored with `kind='derived'` and `base_word` pointing to their root lemma, without standalone definitions or sense hierarchies (or carrying only a circular `gloss_source='wordnet'` stub such as `"in an intensive manner"`, `"the condition of being urbanized"`, or `"the act of celebrating"`). Additionally, **2,846 rows** have `kind='alias'` (orthographic/regional variants or multi-hop redirects such as `strip mining` $\rightarrow$ `strip mine` $\rightarrow$ `strip-mine`).

[`PedlexDbMixin._resolve_ldoce_kind`](../librarian/linguistics_core/pedlex_db.py) and [`WsdEngineMixin.format_ldoce_definition`](../librarian/linguistics_core/wsd_engine.py) deterministically upgrade all `alias` and `derived` rows at runtime (<0.2ms, cached in `_ldoce_cache`) into first-class, multi-sense dictionary entries.

### 6.1 Recursive Alias Resolution (`kind == 'alias'`)
1. **Multi-Hop Cycle-Safe Traversal**:
   - Follows `base_word` pointers up to 5 hops (`seen_aliases` guard) until reaching the canonical `article` or `derived` row (or resolving via `get_ldoce_entry(current_base)` for hyphenated/spaced variants).
2. **Identity & Orthographic Preservation**:
   - Copies all lexicographical assets from the resolved target while preserving the alias's surface `word`, `alias_of`, and `entry_status = "active"`.

### 6.2 Six-Stage Derived-Word Completion Pipeline (`kind == 'derived'`)
When `get_ldoce_entry(word)` encounters `kind == 'derived'`, `_resolve_ldoce_kind` executes a 6-stage enrichment pipeline against `base_data`:

```text
[Raw 'derived' Row (word, pos, base_word, runon_gram)]
          │
          ▼
1. Base-Word Pattern Leak Stripping (剥离词根句式尾巴)
          │
          ▼
2. Language Activator & Thesaurus Exponent Harvesting (同义辨析精确义项收割)
          │
          ▼
3. Circular/Flat WordNet Gloss Upgrading (format_ldoce_definition 脱离循环释义)
          │
          ▼
4. Morphological Rule Synthesis (_synthesize_single_derived_def 构词法释义合成)
          │
          ▼
5. Polysemous Base Sense Projection & Participial-Adjective Base Guard (多义词根义项投影与过去分词基词隔离)
          │
          ▼
6. Cross-POS Asset Borrowing Guard & 5-Tier Authentic Example Harvesting (跨词性隔离与真实例句五层收割)
```

#### Stage 1: Base-Word Pattern Leak Stripping
- Raw dictionary run-ons occasionally leak a trailing syntactic pattern from the parent verb onto the derived adjective/noun definition (e.g., `agonizing` having trailing `"agonize over/about"` appended to its gloss).
- `_resolve_ldoce_kind` compares the tail of `senses[0].definition` against `base_data.senses[].patterns` and strips any leaked base-word pattern prefix/suffix cleanly.

#### Stage 2: Language Activator & Thesaurus Exponent Harvesting (`gloss_source = "ldoce_activator"`)
- Before falling back to rule-based synthesis, `_resolve_ldoce_kind` scans `base_data.language_activator` and `base_data.thesaurus` for an exponent whose headword matches `word`.
- If an exact exponent match is found with a pedagogical definition and examples, it is promoted directly to `senses[0]` with `gloss_source = "ldoce_activator"`.

#### Stage 3: Circular & Cross-POS Stub Upgrading (`format_ldoce_definition` $\rightarrow$ `gloss_source = "ldoce_upgraded"`)
- Detects circular/tautological stub definitions (`is_circular_or_flat`), including:
  - Manner adverb tautologies: `"in a [base] manner/way"` (`intensively` $\rightarrow$ `"in an intensive manner"`).
  - Nominalization tautologies: `"the act/process/state/condition/quality of being [base]"` (`urbanization` $\rightarrow$ `"the condition of being urbanized"`, `celebration` $\rightarrow$ `"the act of celebrating"`).
  - Cross-POS leaked definitions: adjective conditional definitions (`"if you are filled with..."`) leaked onto derived nouns (`wonderment`).
- [`format_ldoce_definition`](../librarian/linguistics_core/wsd_engine.py) upgrades known circular stubs via curated pedagogical definitions (`circular_upgrades`) or delegates to Stage 4 morphological synthesis.

#### Stage 4: Morphological Definition Synthesis (`_synthesize_single_derived_def`, `gloss_source = "derivational"`)
When a derived entry has no sense or its stub was cleared in Stage 3, `_synthesize_single_derived_def(base_def, base_pos)` synthesizes a natural, non-circular definition from the base sense:
- **Adverbs (`-ly`)**:
  - From verbs (`base_pos == 'verb'`): transforms `"to V ..."` into `"in a way that [verb+3sg] ..."` (e.g., `"in a way that shows or expresses..."`).
  - From adjectives: transforms into `"in a [adj] way; [base_def]"`.
- **State/Quality Nouns (`-ness`, `-ity`, `-cy`, `-ance`, `-ence`)**:
  - Transforms adjective base definitions into `"the quality or state of being [base_word]: [base_def]"`.
- **Process/Action Nominalizations (`-ization`, `-ation`, `-tion`, `-sion`, `-ment`)**:
  - Strips participial prefixes (`"made or caused to be ..."`) and converts `"to make/become X"` into `"the act or process of making/becoming X"`.
  - If the base definition starts with `"if you V something, ..."`, rewrites it into `"the act or process of [V-ing] ...; when someone or something..."`.
- **Verbal Nouns (`-ing`, `-ings`)**:
  - Synthesizes `"the act or practice of [V-ing] ..."` from the parent verb definition.

#### Stage 5: Polysemous Base Sense Projection (`_project_base_senses`) & Participial-Adjective Base Guard
- **Multi-Sense Projection for Polysemous Roots**:
  - Many derived words inherit polysemy from their root (e.g., `celebration` from `celebrate`; `intensively` from `intensive`; ` attachment` from `attach`).
  - `_project_base_senses` projects up to 4 distinct base senses into derived senses (carrying over `signpost` and base `examples` for WSD context scoring while keeping `pattern_bundles` empty unless attested for the derived form), enabling `_lock_sense` and Stage 2 LLM arbitration to disambiguate polysemous derived words accurately.
- **Participial-Adjective Base Guard (`is_participial_adj_base`)**:
  - Some process nouns (e.g., `urbanization`, `modernization`, `industrialization`) point to an `-ed`/`-ized` participial adjective (`urbanized`) as their `base_word` in raw LDOCE.
  - When `senses[0]` already holds an upgraded dynamic process definition (e.g., `urbanization` = *"the process by which towns and cities grow bigger and more and more people go to live in them"*), `is_participial_adj_base` **blocks** projecting secondary static state senses from `urbanized` (*"having a lot of houses, factories..."*), preventing static adjective senses from overwriting dynamic process nouns.

#### Stage 6: Cross-POS Asset Borrowing Guard (`same_pos_family` & `_attests_derived`) & 5-Tier Authentic Example Harvesting (`own_exs`)
1. **Cross-POS Asset Isolation across All 5,678 Derived Entries (`same_pos_family` & `_attests_derived`)**:
   - Derived words normally differ in POS from their root (`verb` $\rightarrow$ `noun`, `adjective` $\rightarrow$ `adverb`, `participial adjective` $\rightarrow$ `verb`).
   - `_resolve_ldoce_kind` enforces `same_pos_family = (d_pos == b_pos_top or d_pos[:3] == b_pos_top[:3])`: `thesaurus`, `collocations`, `cross_collocations`, `grammar_boxes`, and sense-level `patterns` / `syntactic_patterns` / `pattern_bundles` / `units` are NEVER blindly copied across different POS families.
   - A base asset is inherited only if it passes `_attests_derived(t_str)`, which masks out `base_word` before checking `text_contains_form(s, req_low)` (preventing verbs like `petrify` derived from `petrified` from inheriting participial-adjective patterns like `petrified of`). Verified with **0 cross-POS leaks across all 5,678 `derived` rows**.
   - `runon_gram` (e.g., `[uncountable]`, `[only before noun]`) is promoted to `entryhead_gram` on the derived entry and injected into `senses[0].grammar`.
2. **Multi-POS Homograph Isolation (`work`, `control`, `study`, `record`)**:
   - When an explicit `pos` is passed to `get_ldoce_preps`, `get_ldoce_grammar_alert`, `get_rich_collocations`, or `get_vocab_card_enrichment`, cross-POS collocations, `Grammar Boxes`, and `Thesaurus` peers are strictly filtered out (e.g., noun `work` never inherits verb grammar boxes *"He works in Manhattan"*, verb `control` never inherits noun collocations `control over`/`self-control`/`control group`, and noun `study` never inherits verb collocations `study law`/`study to be a doctor`).
3. **5-Tier Headword-Bearing Example Harvesting (`own_exs`)**:
   - A derived entry **never** uses a base-word example as its display example unless the sentence physically contains the inflected or base form of the derived headword (`_attests_derived`).
   - `_resolve_ldoce_kind` harvests `own_exs` across 5 tiers of `base_data`:
     1. `base_data.language_activator[].exponents`
     2. `base_data.thesaurus[].exponents`
     3. `base_data.collocations`
     4. `base_data.senses[].pattern_bundles` & `base_data.senses[].examples`
     5. `base_data.corpus_examples`
   - If the entry still lacks an example after harvesting, [`ensure_distinct_example`](../librarian/linguistics_core/wsd_engine.py) queries `search_corpus_examples(word)` across all entries and WordNet (`oewn:2025+`). If 0 authentic examples exist, the candidate is cleanly dropped at Gate 2 (`[DROP_INCOMPLETE_LEXICON]`).

---

## 7. Common Retrieval & Lookup Methods (`常用检索方法与接口速查`)

All dictionary and phrase retrieval methods live in [`PedlexDbMixin`](../librarian/linguistics_core/pedlex_db.py) and [`WsdEngineMixin`](../librarian/linguistics_core/wsd_engine.py), exposed as classmethods on [`LinguisticEngine`](../librarian/linguistics.py).

### 7.1 Primary Entry & Compound Lookup (`pedlex_db.py`)

| Method Signature | Return Type | Purpose & Mechanics |
| :--- | :--- | :--- |
| `get_pedlex_entry(word)` / `get_ldoce_entry(word)` | `dict` | **Primary entry resolver** (cached in `_ldoce_cache`). Normalizes `word`, merges numbered homographs (`word_1`..`word_6`) into unified `senses`/`phrasal_verbs`/`phrases`/`grammar_boxes`/`collocations`/`thesaurus`/`language_activator`, runs `_resolve_ldoce_kind` for `alias`/`derived` rows, resolves closed/open compounds via `_ldoce_compound_index` (`potluck` $\rightarrow$ `pot luck`), and synthesizes hyphenated compounds via `_resolve_hyphenated_compound_entry`. |
| `is_exact_pedlex_headword(word)` | `bool` | Returns `True` only if `word` (or its punctuation-normalized key) exists as an exact headword in `pedlex.db`, rather than matching via suffix-stripping fallback in `_headword_candidates`. |
| `_resolve_hyphenated_compound_entry(word)` | `dict` | Two-tier resolver for hyphenated compounds (`in-person`, `energy-saving`): **Tier 1** promotes attested open phrases (`in person`) under the right/left constituent; **Tier 2** matches productive right-hand combining forms (`-saving`, `-based`, `-free`, `-friendly`, `-oriented`, `-driven`). |
| `get_word_gram_profile(word, pos="", sense=None)` | `dict` | Aggregates entry-level (`entryhead_gram`, `runon_gram`), sense-level (`sense.grammar`), and `grammar_boxes` syntactic constraints into boolean flags (`only_before_noun`, `not_before_noun`, `uncountable`, `plural`, `singular`, `intransitive`, `transitive`, `forbidden_frames`). |
| `evaluate_syntactic_slot_clash(cand, target_gram_profile, ...)` | `Optional[str]` | Verifies whether a candidate distractor violates the grammatical slot of the target blank (`[only before noun]` in predicative slot, `[not before noun]` in attributive slot, `[uncountable]` after `a/an` or pluralized, etc.). |

### 7.2 Sense Locking, WSD & Definition/Example Retrieval (`wsd_engine.py` & `pedlex_db.py`)

| Method Signature | Return Type | Purpose & Mechanics |
| :--- | :--- | :--- |
| `_lock_sense(entry, definition="", quote="", target_pos="", return_confidence=False, return_scores=False, discourse_context="")` | `Optional[dict]` (or tuple) | **Stage 1 Symbolic Sense Scorer**. Applies POS hard gates (including coordinated subject recovery when ambiguous noun/verb tokens appear in `If NP and NP <verb>`), homograph resolution, `<gram>` slot constraints, spoken/technical/pejorative register gates, **Universal Conditional-Subject Ontology Gate** (parsing `if a <Subject-Class> <verb>s` metalanguage against passage subject WordNet supersenses), CQL typed slot pattern matching, 2nd-order collocate + `±1` sentence `discourse_context` domain scoring, and `pedlex_sense_cue_index` weights. |
| `build_top_k_vocab_candidates(word, pos, quote, discourse_context="", k=4)` | `list[dict]` | Builds the Top-$K$ ($K \le 4$) symbolic sense pool from `_lock_sense(..., return_scores=True)` with `sense_idx`, `pos`, `signpost`, `gram`, `definition`, `patterns`, `examples`, and `symbolic_score`. Includes **Cross-Lexicon Pool Fusion**: when a sole PedLex sense is pragmatically restricted (`[disapproving]`, `[spoken]`, `[informal]`, `[humorous]`, `[old-fashioned]`) or has a narrow physical condition (`if a piece of ...`), merges attested WordNet (`oewn:2025+`) synsets with examples into the pool so Stage 2 can arbitrate. |
| `disambiguate_via_llm(item_name, sentence, candidates, discourse_context="")` | `dict` | **Stage 2 1-Token Local LLM Arbitrator (Zero-Fallback Contract)**. Formats Top-$K$ candidates — explicitly surfacing `[SIGNPOST]` (e.g., `[FAIL]` vs. `[STRUCTURE]`), `<gram>` slot constraints, and attitude/register tags (`[disapproving]`, `[technical]`, `[spoken]`) — alongside `Target Sentence` and `Surrounding Context` into a multiple-choice prompt (`A`/`B`/`C`/`D`), resolved in 1 token (`temperature=0.0`, cached in `_wsd_llm_cache`, logged to `logs/<ts>_wsd_arbitration_<unit>.log`). Enforces a strict **Zero-Fallback Invariant**: if the local LLM service is unreachable or returns an unparseable token after retry, raises `WSDArbitrationError` instead of silently falling back to `candidates[0]`. |
| `get_ldoce_definition_and_example(word, target_pos="", context_sentence="", require_sense_confidence=False)` | `tuple[str, str]` | High-level helper that auto-binds `target_pos` via `determine_contextual_pos`, locks the sense via `_lock_sense`, resolves British/American cross-references (`program` $\rightarrow$ `programme`), upgrades circular derived definitions via `format_ldoce_definition`, and returns `(definition, example)`. |
| `expression_definition_evidence(phrase, expr_type="expression", context_sentence="", enable_llm_disambiguation=True)` | `dict` | **Multi-word expression WSD resolver**. Cascades across `ldoce_entry` $\rightarrow$ `phrasal_verb_block` $\rightarrow$ `sense_pattern` / `phrase_row` / `colloc_phrase` (using direct `sense_idx` pointers, collocation `gloss` definitions, `_unit_owning_sense`, & `_quote_overridden_unit_sense`) $\rightarrow$ `crossrefs` (cross-entry idiom pointers) $\rightarrow$ `wordnet`, returning `{definition, example, source, locked_sense}`. |
| `ensure_distinct_example(word, category, quote, current_example="", pos="", definition="", locked_sense=None)` | `str` | **Gate 2 Authentic Example Resolver**. Guarantees an authentic example distinct from `quote` and physically containing `word` (never stealing across senses for polysemous words). |
| `search_corpus_examples(query, category="word", limit=3)` | `list[str]` | Searches `pedlex.db` for authentic examples containing any morphological inflection of `query`: checks own entry first (`senses`, `corpus_examples`, `collocations`), then queries `pedlex_phrase_index` and `pedlex_sense_cue_index` to locate cross-entry examples in <2ms. |

### 7.3 Multi-Word Phrase Index & Typed Slot Pattern Matching (`pedlex_db.py`)

At startup (or first phrase query), `_ensure_in_memory_phrase_index()` loads all rows of `pedlex_phrase_index` into 4 in-memory O(1) lookup structures in ~170ms:
- `_phrase_by_key_cache`: `phrase_key -> list[(host_word, sense_idx, tier, raw_phrase)]`
- `_phrase_by_raw_cache`: `raw_phrase.lower() -> list[(host_word, sense_idx, tier, raw_phrase)]`
- `_PHRASE_INDEX_CACHE`: 26,538 normalized phrase records with pre-tokenized `words`, `first`, `last`, `n`, `tier`, and `pos` for fast structural distractor mining.
- `_PHRASE_SYNONYMS_CACHE`: `(host_word, sense_idx) -> set[norm_phrase]` for sense-sharing phrase synonym detection.

| Method Signature | Return Type | Purpose & Mechanics |
| :--- | :--- | :--- |
| `lookup_ldoce_phrase_candidates(phrase, headword="")` | `list[dict]` | **3-Track Multi-Candidate Phrase Lookup**: **Track 1** matches exact literal `phrase_key`/`raw_phrase` (including `colloc_phrase` rows and `get / feel / be` slash-alternation expansion); **Track 2** strips variable slots (`somebody`, `something`, `one's`) via `_unit_tokens` and checks token-subset containment; **Track 3** lemmatizes the leading verb (`took place` $\rightarrow$ `take place`) and re-queries Tracks 1–2. Returns all matching `{host_word, sense_idx, tier, raw_phrase}` rows without `LIMIT 1` truncation. |
| `lookup_ldoce_phrase(phrase, headword="")` | `Optional[dict]` | Convenience wrapper returning the top-ranked candidate from `lookup_ldoce_phrase_candidates`. |
| `ldoce_phrase_hits(phrase, headword="")` | `list[dict]` | Returns all attested dictionary sub-records (`phrasal_verb`, `phrase`, `sense_pattern`, `collocation` with `gloss` and `sense_idx`, and cross-entry `crossrefs`) matching `phrase` under `headword` or its phrase-index host entry. |
| `ldoce_phrase_evidence(phrase, headword="")` | `Optional[dict]` | Returns the highest-priority attested evidence record (`source`, `matched_unit`, `definition`, `examples`, `sense`) for `phrase`. |
| `is_attested_phrase(phrase, headword="")` | `bool` | Fast boolean check verifying whether a multi-word unit is attested in `pedlex.db`. |
| `ldoce_phrase_object_fit(phrase, obj, headword="")` | `int` | Scores (`0`, `1`, `2`) how well a direct/prepositional object `obj` fits the attested examples and collocations of `phrase`. |
| `compile_typed_slot_pattern(pattern, headword="")` | `list[tuple]` | Compiles raw dictionary patterns (`(doing) something`, `somebody to do something`, `in a ... way/fashion`) into a sequence of typed token matchers (`LIT`, `OPT_LIT`, `INF_TO`, `VBG`, `OPT_VBG`, `PRON_REFL`, `NP_SLOT`, `INFIX_MOD`, `CLAUSE_WH`). |
| `match_typed_slot_pattern(pattern, quote, doc=None, headword="")` | `int` | Matches a compiled pattern against a sentence (`doc`), enforcing clause/punctuation barriers (`_slot_Crosses_barrier`) and returning a syntactic specificity score (`45`–`95`, or `0` on mismatch). |

### 7.4 Valency, Sense Cue Index, Construction Index, Error Traps & Card Enrichment (`pedlex_db.py` & `collocations_and_cards.py`)

| Method Signature | Return Type | Purpose & Mechanics |
| :--- | :--- | :--- |
| `get_ldoce_preps(word, pos=None)` | `set[str]` | Extracts all bound prepositions attested in `patterns` and `collocations` for `word` (e.g., `{'in', 'to', 'with'}`), enforcing strict POS isolation on multi-POS homographs. |
| `get_ldoce_phrasal_verbs(word)` | `set[str]` | Returns all attested phrasal verbs (`verb + particle`) under `word`. |
| `get_sense_cue_scores(word, context_tokens)` | `dict[int, float]` | Queries `pedlex_sense_cue_index` for `context_tokens` (after filtering `_get_pedlex_stop_cues` super-degree nodes appearing in >500 entries) and returns `{sense_idx: cumulative_weight}`. |
| `get_ldoce_grammar_alert(word, quote="", sense_index=None, pos=None)` | `Optional[dict]` | Extracts sense- and POS-compatible `Grammar Boxes` (`Don't say: ✗ ...`, `comparison_peers`, `syntactic_constraints`, `sense_indices`) for vocabulary cards and translation quiz error injection. |
| `get_ldoce_thesaurus(word)` | `list[dict]` | Returns PedLex `thesaurus` and `language_activator` synonym discrimination tables for `word`. |
| `get_oxford_thesaurus_entries(word)` | `list[dict]` | Queries `oxford_thesaurus` in `pedlex.db` (cached in `_oxford_cache`), returning per-sense `definition`, `synonyms`, and `antonyms`. |
| `get_pedlex_and_oxford_synonyms_antonyms(word, pos="", definition="", quote="")` | `tuple[list[str], list[str]]` | Combines sense-locked synonyms and antonyms from both `ldoce` and `oxford_thesaurus`, including 1-hop reciprocal antonym discovery (`synonym -> antonym`). |
| `get_word_cxn_specs(word, sense_idx=None)` | `list[dict]` | Queries `pedlex_cxn_index` (and entry/sense `cxn_specs`) for constructional specs (`family_id`, `subcat_group`, `canonical_pattern`, `canonical_blank`, `cloze_frame`, `contrast_groups`, `ban_groups`). |
| `query_cxn_contrast_distractors(contrast_groups, ban_groups=None, pos=None, exclude_words=None, limit=12)` | `list[str]` | Executes a <0.2ms set-difference SQL query (`contrast_groups \ ban_groups`) on `pedlex_cxn_index` ordered by `zipf_score DESC`. |
| `get_word_error_traps(word, pos=None)` | `list[dict]` | Retrieves pre-mined minimal-pair learner errors from `pedlex_error_traps` (supplemented by `<gram>`, valency, and L1 negative-transfer rules) for `translation_quiz` and `vocabulary_quiz`. |
| `get_vocab_card_enrichment(word, pos, definition, quote="")` | `dict` | Zero-token Markdown vocabulary card enrichment returning `{pattern_note, grammar_usage, grammar_tag, collocations, thesaurus_note}` with strict `sense_idx` / `pos` isolation (`collocations` filtered by `sense_idx`, `thesaurus` boxes filtered by `sense_indices` and `pos`, `grammar_boxes` filtered by `sense_indices` and `homograph_num`). |

### 7.5 Direct CLI & Python Inspection Recipes (`命令行快速检视方法`)

When debugging dictionary entries, derived words, construction specs, or error traps from the terminal, use `LinguisticEngine` directly via `python -c`:

```powershell
# 1. Inspect a kind-resolved entry (including 'derived' or 'alias' completion)
python -c "from librarian.linguistics import LinguisticEngine as LE; import json; e = LE.get_pedlex_entry('urbanization'); print(json.dumps({'kind': e.get('kind'), 'base_word': e.get('base_word'), 'senses': e.get('senses')[:2]}, indent=2, ensure_ascii=False))"

# 2. Test Two-Stage Neuro-Symbolic WSD (Stage 1 Symbolic + Stage 2 1-Token LLM)
python -c "from librarian.linguistics import LinguisticEngine as LE; cands = LE.build_top_k_vocab_candidates('commercialize', 'verb', 'These hubs help innovators design, produce, and commercialize new technologies.'); print([(c['sense_idx'], c['symbolic_score'], c['definition']) for c in cands])"

# 3. Query construction specs and reverse subcategorization distractors in pedlex_cxn_index
python -c "from librarian.linguistics import LinguisticEngine as LE; specs = LE.get_word_cxn_specs('avoid'); print(specs[0]); print(LE.query_cxn_contrast_distractors(specs[0]['contrast_groups'], specs[0]['ban_groups'], pos='verb'))"

# 4. Query minimal-pair learner error traps in pedlex_error_traps
python -c "from librarian.linguistics import LinguisticEngine as LE; print(LE.get_word_error_traps('avoid', pos='verb')[:2])"
```

---

## 8. Implemented Architecture: Two-Tier Construction Tagging (`pedlex_cxn_index`) & Minimal-Pair Error Traps (`pedlex_error_traps`)

To power the **Deterministic Assessment Machine** across all **Eight Lexico-Syntactic Construction Families** and **Minimal-Pair Translation Quizzes** without accumulating brittle regex or hardcoded word lists in Python, `pedlex.db` provides two pre-compiled relational indexes: `pedlex_cxn_index` (2,635 rows) and `pedlex_error_traps` (16,759 rows).

### 8.1 Two-Tier Construction Tagging Granularity (`Entry` vs. `Sense -> Pattern Bundle`)

Because English lexical items range from monosemous fixed idioms to highly polysemous single words, construction triggers are attached at two distinct granularities:

1. **Tier 1 — Multi-Word Phrases & Fixed Constructions (Entry / Phrase-Key Level Trigger, `sense_idx = -1`)**:
   - Targets: `make [sth] possible`, `find [sth] hard` (Family 1A); `not only ... but also`, `no sooner ... than` (Family 2); `look forward to`, `have difficulty (in)` (Family 4); `take advantage of`, `pay attention to` (Family 6); `under no circumstances`, `let alone` (Family 7); `would rather`, `may/might as well` (Family 8).
   - **Direct Lexical Activation**: Selecting the item immediately activates its pre-compiled `cxn_spec` (`canonical_blank`, `canonical_pattern`, `cloze_frame`, `contrast_groups`, and `ban_groups`).
2. **Tier 2 — Polysemous Single-Word Headwords (`Sense` Level Trigger, `sense_idx >= 0`)**:
   - Targets: Single-word adjectives (`hard`, `easy`, `necessary`, `possible` in Family 1B), verbs (`remind`, `prevent` in Family 4; `avoid`, `manage`, `decide` in Family 3; `insist`, `suggest` in Family 8), and shell nouns (`ability`, `possibility`, `fact`, `habit` in Family 5).
   - **Quote-Driven Sense Activation**: During quiz blueprint generation (`Step 0: _lock_sense` $\rightarrow$ `Step 1: pick_atomic_pattern_bundle`), locking the passage `Quoted Sentence` activates `get_word_cxn_specs(word, sense_idx)` only when the locked sense and pattern match that constructional family.

### 8.2 Reverse Subcategorization Set-Difference for Zero-Collision Distractors (`query_cxn_contrast_distractors`)

Because `pedlex.db` records all attested syntactic patterns for every headword, the offline compiler partitions the lexicon into **mutually exclusive syntactic subcategorization groups (`subcat_group`)** via set difference ($\text{Contrast} \setminus \text{Ban}$):

| `subcat_group` | `family_id` | Offline `pedlex.db` Extraction Rule (`senses[].patterns` & `grammar`) | Examples Indexed (`0` Manual Word Lists) | Contrasting Distractor Pool (`contrast_groups`) |
| :--- | :--- | :--- | :--- | :--- |
| `V_IT_ADJ_INF` | `FAM1A_IT_OBJ_EXTRAPOS` | Preparatory-object extraposition (`make/find/keep it <adj> to do`) | `make`, `find`, `keep`, `leave`, `consider` | `V_OBJ_INF_DIRECT` (`cause`, `require`, `persuade`) |
| `ADJ_IT_EXTRAPOS` / `ADJ_TOUGH_BOTH` | `FAM1B_ADJ_COMPLEMENTATION` | Has pattern `it is <adj> (for sb) to do sth` | `hard`, `easy`, `difficult`, `possible`, `necessary`, `essential`, `vital` | `ADJ_ANIMATE_CONTROL` (`eager`, `willing`, `able`, `happy`, `proud`) |
| `V_GERUND_ONLY` | `FAM3_VERB_NONFINITE_COMP` | Has `<V> doing sth` **AND NOT** `<V> to do sth` | `avoid`, `admit`, `deny`, `enjoy`, `mind`, `postpone`, `practice`, `risk`, `finish` | `V_INF_ONLY` (excludes dual `V_BOTH_INF_GERUND`) |
| `V_INF_ONLY` | `FAM3_VERB_NONFINITE_COMP` | Has `<V> to do sth` **AND NOT** `<V> doing sth` | `manage`, `afford`, `refuse`, `fail`, `hesitate`, `agree`, `decide`, `offer`, `promise` | `V_GERUND_ONLY` |
| `V_OBJ_PREP_OF` | `FAM4_TRIVALENT_V_OBJ_PREP` | Has `<V> sb of sth` | `remind`, `inform`, `convince`, `accuse`, `suspect`, `assure`, `rob`, `warn` | `V_OBJ_PREP_WITH` / `V_OBJ_PREP_FROM` / `V_DITRANS_NP_NP` |
| `V_OBJ_PREP_FROM` | `FAM4_TRIVALENT_V_OBJ_PREP` | Has `<V> sb/sth from (doing) sth` | `prevent`, `stop`, `keep`, `discourage`, `prohibit`, `ban`, `deter`, `restrain` | `V_OBJ_PREP_OF` / `V_OBJ_PREP_WITH` / `V_INF_ONLY` |
| `N_SHELL_TO_DO` | `FAM5_SHELL_NOUN_COMP` | Has `<N> to do sth` **AND NOT** `<N> of doing sth` | `ability`, `tendency`, `attempt`, `effort`, `willingness`, `reluctance`, `decision` | `N_SHELL_OF_DOING` + `N_SHELL_THAT_ONLY` |
| `N_SHELL_OF_DOING` | `FAM5_SHELL_NOUN_COMP` | Has `<N> of doing sth` **AND NOT** `<N> to do sth` | `habit`, `intention`, `risk`, `method`, `way`, `fear`, `possibility` | `N_SHELL_TO_DO` + `N_SHELL_THAT_ONLY` |
| `N_SHELL_THAT_ONLY` | `FAM5_SHELL_NOUN_COMP` | Has `the <N> that...` **AND NOT** `<N> to do / of doing` | `fact`, `belief`, `assumption`, `notion`, `rumor`, `evidence`, `doubt` | `N_SHELL_TO_DO` + `N_SHELL_OF_DOING` |
| `PHR_FRONTED_INV` | `FAM6_FRONTED_NEG_INVERSION` | Negative fronted adverbial triggering `Aux + S + V` | `not only`, `under no circumstances`, `by no means`, `on no account`, `no sooner` | `PHR_NON_INV_PP` (`regardless of`, `owing to`, `in terms of`) |
| `V_MANDATIVE_SUBJ` | `FAM8_MANDATIVE_AND_LV_TRIAD` | Mandative verb triggering `that S (should) V_base` | `insist`, `demand`, `suggest`, `recommend`, `propose`, `require` | `V_GERUND_ONLY` (`deny`, `admit`, `fancy`) |

At runtime (<0.2ms), `LinguisticEngine.query_cxn_contrast_distractors(contrast_groups, ban_groups, pos)` executes a parameterized SQL set-difference query ordered by `zipf_score DESC`, filtered by `cefrpy` level and WordNet synset disjointness.

### 8.3 Pre-Mined Minimal-Pair Error Traps (`pedlex_error_traps` & `get_word_error_traps`)

To eliminate random unfocused errors in `translation_quiz` and provide authentic grammatical pitfalls for `vocabulary_quiz`, `pedlex_error_traps` indexes 16,759 structured error records across 5 categories:
1. **`GRAMBOX_BADEXA` (Authentic Dictionary Negative Examples)**:
   - Extracted directly from LDOCE `Grammar Boxes` (`✗ They are able to avoid to pay taxes` vs. `✓ They are able to avoid paying taxes`; `✗ He won despite he was injured` vs. `✓ He won despite being injured`).
2. **`LEXICAL_VALENCY_FRAME` (Bound Preposition & Complement Traps)**:
   - Targets obligatory prepositional/clausal complements (`originate in/from`, `decide to do`, `regulatory body/authority`).
3. **`GRAM_UNCOUNTABLE` & `GRAM_ATTRIBUTIVE_ONLY` (`<gram>` Property Traps)**:
   - Targets uncountable nouns (`*an news / *newss`, `*find a work`) and `[only before noun]` / `[not before noun]` positional adjectives.
4. **`L1_CONJUNCTION_REDUNDANCY` & `CLAUSE_PREP_CONFUSION` (Syntactic L1 Negative-Transfer Traps)**:
   - Targets paired conjunction redundancy (`*Although ..., but ...*`) and preposition-vs-conjunction complement confusion (`*despite + finite clause*`).

---

## 9. Authoritative Database Build & QA Scripts Inventory (`scripts/`)

All obsolete, one-off debugging and scratch scripts have been removed from `scripts/`. The directory retains **11 authoritative scripts** divided into the **5-Stage Offline Database Build Pipeline** and **6 Database QA & Corpus Audit Tools**:

### 9.1 Five-Stage Offline Database Build Pipeline
To rebuild `librarian/data/pedlex.db` from raw `.mdx` sources or apply incremental schema upgrades, execute the following scripts in order:

| Stage | Script | Purpose & Key Tables Populated |
| :--- | :--- | :--- |
| **Stage 1** | [`scripts/build_ldoce_db.py`](../scripts/build_ldoce_db.py) (with [`scripts/ldoce_runon_audit.py`](../scripts/ldoce_runon_audit.py)) | **Primary LDOCE6 MDX-to-SQLite Compiler**. Parses raw LDOCE6 HTML into the `ldoce` table (59,011 rows) with strict `<span class="phrvbentry">` isolation, `1a)`/`1b)` subsense unpacking, direct `sense_idx` pointers on `collocations`/`phrases`/`crossrefs`, collocation `gloss` extraction, `licensed_poses` multi-POS detection, `thesaurus` `sense_indices`/`topic` preservation, and `recursive=False` DOM label isolation. |
| **Stage 2** | [`scripts/build_oxford_thesaurus.py`](../scripts/build_oxford_thesaurus.py) | **Oxford Learner's Thesaurus Compiler**. Parses Oxford Thesaurus `.mdx` into the `oxford_thesaurus` table (31,599 rows) with sense-level synonym/antonym clusters and discriminator notes. |
| **Stage 3** | [`scripts/build_sense_cue_index.py`](../scripts/build_sense_cue_index.py) | **Contextual Sense Cue Inverted Index Builder**. Populates `pedlex_sense_cue_index` (306,166 weighted `(word, cue_token, sense_idx, weight, source)` rows) from signposts, collocations, patterns, definitions, and examples. |
| **Stage 4** | [`scripts/build_ldoce_fts.py`](../scripts/build_ldoce_fts.py) | **Multi-Word Phrase Inverted Index Builder**. Populates `pedlex_phrase_index` (~109,593 rows across `entry_head`, `phrasal_verb`, `phrase_box`, `sense_pattern`, and `colloc_phrase` tiers). |
| **Stage 5** | [`scripts/upgrade_pedlex_db.py`](../scripts/upgrade_pedlex_db.py) | **PedLex v2 Schema & Relational Index Upgrader**. Consolidates numbered homographs, bridges definitions/examples onto all 4,949 `kind='derived'` entries (`base_sense_bridge`), and builds `pedlex_cxn_index` (2,635 rows across 8 construction families) and `pedlex_error_traps` (16,759 minimal-pair learner error rows). |

### 9.2 Database QA, Regression & Inspection Tools

| Script | Purpose |
| :--- | :--- |
| [`scripts/ldoce_qa.py`](../scripts/ldoce_qa.py) | **Automated PedLex Schema & Invariant QA Suite**. Verifies 0 HTML tag residue, 0 cross-homograph/phrasal-verb leaks, subsense unpacking completeness, and derived-word resolution coverage (also invoked by `tests/test_linguistics.py`). |
| [`scripts/shadow_engine_diff.py`](../scripts/shadow_engine_diff.py) | **Shadow Regression Harness**. Compares `LinguisticEngine` outputs across database versions to verify 0 regressions in POS, WSD sense locking, and collocation retrieval. |
| [`scripts/audit_sense_extraction.py`](../scripts/audit_sense_extraction.py) | **End-to-End Passage WSD & Extraction Auditor**. Audits vocabulary and multi-word expression sense locking across textbook units. |
| [`scripts/audit_corpus_offline.py`](../scripts/audit_corpus_offline.py) | **Offline Corpus Coverage & Gate 2 Example Auditor**. Verifies Gate 1–3 pass rates and authentic example coverage across all `wiki/` units. |
| [`scripts/query_ldoce.py`](../scripts/query_ldoce.py) | **CLI Entry Inspector**. Pretty-prints any headword's JSON payload (`senses`, `collocations`, `phrasal_verbs`, `phrases`, `thesaurus`, `grammar_boxes`) from `librarian/data/pedlex.db`. |

