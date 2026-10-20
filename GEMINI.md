# Lexis Wiki Project Instructions

> **Purpose**: This document serves as the single source of truth for the Lexis Wiki project — a neuro-symbolic educational knowledge base and interactive assessment generator that transforms primary texts, audio, and video into Obsidian-style wiki nodes and psychometrically audited HTML quizzes.

---

## 1. Neuro-Symbolic Core Architecture

### 1.1 Quad-Tier Separation of Responsibilities
1. **Symbolic Grounding & Lexical Corpus (`LinguisticEngine` + `pedlex.db` + WordNet + `cefrpy`)**:
   - Serves as the **100% authoritative semantic truth and constraint engine**.
   - Combines offline computational NLP (`spaCy` syntactic dependency trees, POS tagging, morphological inflection) with the Pedagogical Lexicon (`pedlex.db` + 106,738-entry `pedlex_phrase_index`), WordNet (`oewn:2025+`), and `cefrpy`.
   - Deterministically extracts vocabulary, multi-word expressions, and syntactic grammar patterns, and pre-computes single-fit collocational anchors, parallel inflected options, and zero-collision distractors.
2. **Code Gate & Invariant Enforcement (`processor.py`, `evaluator.py`)**:
   - Enforces physical invariants at zero token cost (<5ms): verbatim quotation checks, single-blank constraints (`re.findall(r'_{2,}', stem) == 1`), option cardinality, CRC32 deterministic slot assignment, and CEFR lexical ceiling gates.
   - Performs instant programmatic self-healing and category auto-remapping without burning LLM retry tokens.
3. **Pedagogical Prompting (`librarian/prompts/*.md`)**:
   - Focuses exclusively on Second Language Acquisition (SLA) standards, contextual stem crafting, and psychometric discrimination.
   - Executes pre-computed `🎯 Micro-Task for LLM` blueprints (for quizzes) or discourse synthesis (for summaries/mind maps) without mechanical formatting or distractor-inventing burdens.
4. **Schema Enforcement (`schemas.py`)**:
   - Defines strict `snake_case` JSON structures and array bounds via Prompt-Guided JSON Mode (`format: "json"` by default, with automatic fallback from `enforce_gbnf` if needed).
   - Historical two-turn Prose-to-JSON (`enable_prose_pipeline`) is permanently removed in favor of **Unified One-Shot Direct Structured Output** (2.2× faster, 0 secondary-turn drift).

### 1.2 Architectural Discipline & Anti-Patching Mandate
- **Zero Tolerance for Band-Aid Bloat**:
  - Developers and AI agents must NEVER accumulate brittle word-specific regex hacks, ad-hoc `if-else` patches, or arbitrary exception branches when rule-based heuristics hit a competence ceiling.
- **First-Principles Escalation**:
  - When a linguistic or pedagogical edge case cannot be cleanly resolved by general syntactic/lexical invariants, proactively flag the structural limitation and escalate to a clean neuro-symbolic mechanism (such as Top-$K$ symbolic candidate filtering + 1-token LLM arbitration).

---

## 2. Sense-First Unified Extraction & WSD Architecture

### 2.1 Deterministic Code-Driven Extraction & Evidence-Completeness Gate ("宁缺勿滥")
- **Vocabulary & Multi-Word Expressions (`extract_deterministic_vocabulary`, `extract_deterministic_expressions`)**:
  - Extracted deterministically from `sources/<Unit>.md` + syllabus wordlists via `spaCy` dependency trees and `pedlex.db`, eliminating LLM extraction hallucinations and omissions.
  - **Multi-Word Expression Typing (`classify_expression_type`)**: Multi-word units are never labeled with token-level POS (`verb`/`noun`); they are strictly classified into `phrasal verb`, `collocation`, `set phrase`, or `idiom`.
  - **Multi-Slash Syllabus Preservation (`parse_raw_headwords`)**: Slash-separated alternatives (e.g., `get / feel / be lost`) are normalized prior to standalone `/IPA/` stripping so middle variants are never lost.
- **Quality-Over-Quantity Evidence-Completeness Gate ("宁缺勿滥")**:
  - Every candidate word or expression (including syllabus items such as `hybridize`, `crossbreed`, `domesticate`, or `compromise` under an unattested sense) must pass a strict **Three-Gate Selection Pipeline**:
    1. **Gate 1 — Sense Locking (Top-$K$ Symbolic + 1-Token LLM WSD)**: Locks the exact contextual `Sense` in `pedlex.db` (or WordNet) matching the passage quote's POS, syntactic frame, and semantics.
    2. **Gate 2 — Sense-Isolated Authentic Example Gate (`ensure_distinct_example`)**: The locked `Sense` must provide an authentic dictionary example distinct from the source quote (via the locked sense's own `examples`/`pattern_bundles`, or inflection-aware corpus search `search_corpus_examples` for monosemous items). **Cross-sense example stealing is strictly forbidden** for polysemous words. Any item with a definition but 0 authentic examples for its locked sense is **immediately dropped and replaced** by a better candidate (`[DROP_INCOMPLETE_LEXICON]`).
    3. **Gate 3 — Collocational Anchor & Zero Double-Key Gate**: Candidates lacking discriminative context/collocational anchors or prone to synonym double-key collisions (e.g. `crossbreed` vs `hybridize`) are dropped before quiz generation.
  - **Zero Hardcoded Example/Sense Tables**: Hardcoded Python dictionaries (`curated_examples`, `priority_sense_examples`, `supplemental_vocab_senses`) are permanently prohibited so dictionary gaps and retrieval bugs are never masked.
- **Grammar Extraction (`extract_deterministic_grammar`)**:
  - Deterministically mines structural patterns and COBUILD syntactic formulas across **Four Macro Functional Domains**:
    1. `Rhetoric & Emphasis` (inversion, clefts, antithesis/parallelism)
    2. `Cohesion & Framing` (shell nouns, appositive/complement `that`-clauses, anaphoric `, which means/suggests that`)
    3. `Information Packaging` (participial adjuncts, nominalizations, `it`-extraposition, elaborative relative clauses)
    4. `Logic & Stance` (conditionals, concessive clauses, epistemic hedging)
  - All pedagogical pitfalls use the unified prefix `Grammar Warning (<TOPIC>): ...`.

### 2.2 True Neuro-Symbolic Sense & Pattern Reranker (WSD)
- **Elimination of Dense Embedding Models**:
  - The 80MB `all-MiniLM-L6-v2` sentence-transformer is **permanently removed** (avoiding thematic/domain bias such as misclassifying *"Punctuality pays!"* into financial payment senses).
- **Two-Stage Sense Locking (`_lock_sense` + `disambiguate_via_llm`)**:
  1. **Stage 1 — Symbolic Candidate Pool (`pedlex.db` + Syntactic Gates, 0ms / 0 tokens)**:
     - Applies POS hard gates (with automatic cross-homograph resolution `word_1`..`word_6`), conversational/spoken register gates (`[spoken]`), syntactic pattern/preposition intersection, and Lesk lemma overlap to build a Top-$K$ ($K \le 4$) candidate pool.
     - **~71% Single-Sense Fast Path**: Monosemous entries under the target POS lock instantaneously in 0ms at 0 token cost.
  2. **Stage 2 — 1-Token Local LLM Arbitration (~29% Polysemous Cases)**:
     - Formats the Top-$K$ ($K \le 4$) dictionary senses/patterns alongside the quoted passage sentence into a ~100-token multiple-choice prompt, resolved via a 1-token (`A`/`B`/`C`/`D`) local LLM call (~0.28s–0.40s on 8B models, 100% benchmark accuracy).
     - LLM has **zero free-generation authority** and can only select among authentic `pedlex.db` candidates.

---

## 3. Single Source of Truth & Sense-First Quiz Blueprint Pipeline

### 3.1 Fresh Deterministic Extraction for Assessments
- **Downstream Markdown Decoupling**:
  - `extractions/<Unit>_{vocabulary,grammar}.md` are presentation artifacts for reading in Obsidian/Web Dashboard and may be edited or translated by users.
  - Assessment pipelines (`vocabulary_quiz`, `translation_quiz`) **never trust downstream Markdown as semantic truth**; they ground directly on `sources/<Unit>.md` via fresh deterministic extraction.

### 3.2 Atomic Sense-First Quiz Blueprint (`build_precomputed_target_skeletons`)
- **Zero Drift Between Extraction and Assessment (`0 DEF_DRIFT`, `0 POS_DRIFT`)**:
  - **Step 0 Sense-First Grounding**: Locks the target headword to its exact `locked_sense` and `canonical_pos` using the upstream-verified `definition`, `quote`, and `part_of_speech` before any collocation or distractor lookup occurs.
  - **Definition Conservation**: Preserves authoritative multi-word/derived definitions and register labels (`[formal]`, `[spoken]`) without allowing fallback single-word senses to overwrite them.
- **Atomic Pattern–Example–Anchor Binding (`pick_atomic_pattern_bundle`)**:
  - `definition`, `best_pattern`, `authentic_example` (`Corpus Example Frame`), and `context_anchor` are inherited as an indivisible bundle from the locked `Sense`.
  - A syntactic pattern with a bound preposition (e.g., `pay in [currency]`, `serious about`) is strictly paired with an example sentence containing that exact preposition and inherits `anchor_type = "prep"`, eliminating pattern–example desync (`0 PAT_EX_PREP_DESYNC`, `0 ANCHOR_PAT_PREP_CLASH`).
- **Valency Cascades & Post-Inflection Double-Key Screening**:
  - **POS Valency Cascades**: Nouns use narrative context clues & `plurale tantum` symmetry; transitive verbs prioritize `object` $\gg$ `prep` $\gg$ `adv_mod`; intransitive verbs prioritize `prep` $\gg$ `adv_mod`; adjectives prioritize `prep` / `modified_noun`; closed-class connectors enforce clausal vs. prepositional complement contrast.
  - **Multi-Word Phrase Distractors (`generate_phrase_distractors`)**: Queries the 106k `pedlex_phrase_index` for structurally symmetric distractors (tail-sharing clusters, light-verb constructs, phrasal verb particle/verb contrasts, and quantifier head-blanking `a ____ of`).
  - **Post-Inflection Collision Gate**: After `_reconcile_prescribed_inflection` aligns all 4 options to the target's inflectional morphology (`VBD`/`VBN`/`VBG`/`VBZ`/`NNS`), `double_key_collision` re-verifies the inflected forms (catching synset overlaps like `disguised` vs. `veiled`) and refills any colliding distractor (`0 DOUBLE_KEY_SURVIVED`).
- *(Detailed methodology: [docs/vocabulary-selection-and-distractor-methodology.md](docs/vocabulary-selection-and-distractor-methodology.md))*.

---

## 4. Deterministic CEFR Calibration & Two-Level QA Audit

### 4.1 Zero-LLM-Burden CEFR Calibration (`cefrpy`)
- All output schemas (`schemas.py`) strip out `cefr_level` / `overall_cefr_level` so LLMs never guess difficulty ratings.
- `LinguisticEngine.calculate_text_cefr` and `get_word_cefr` deterministically compute passage and headword CEFR levels (`A1`–`C2`), persisting `overall_cefr_level` in frontmatter and injecting `{cefr_level}` constraints into all 5 quiz modalities (`vocabulary`, `reading`, `translation`, `listening`, `video`).
- Offline `cefrpy` + Zipf frequency gates eliminate obscure C1/C2 words from A1–B1 distractors and stems.

### 4.2 Two-Level QA Quality Audit Architecture
1. **Level 1: Deterministic Code Gate (`evaluator.py`, <5ms, 0 tokens)**:
   - Enforces structural invariants, option length/CEFR bounds across all 5 quiz types, Fisher-Yates option shuffling with atomic answer-index remapping, and deterministic grammar category auto-remapping (-2.0 pt soft penalty instead of 30s LLM retry).
2. **Level 2: Expert Model Semantic Audit (`expert_auditor.py`, LLM-as-a-Judge)**:
   - Optional (`enable_expert_audit` in `wiki_config.json`), powered by `judge_model` (`num_predict: 16384`).
   - Performs blind-solver resolution and 3-tier triage: **Tier 3 `PASS`** (80–100), **Tier 2 `REPAIR`** (60–79, surgical distractor replacement), **Tier 1 `REWRITE`** (20–59, regeneration).
3. **Transparent Delivery Status (`qa_status`)**:
   - No `_quarantine/` folders; files deliver directly to `extractions/` or `handouts/` with frontmatter `qa_status` (`"passed"`, `"review_needed"`, or `"failed"`).

---

## 5. Stateless Storage, Configuration & Conventions

### 5.1 Self-Contained Stateless Directory Layout
```text
wiki/<UnitName>/
├── sources/              # Primary text (<UnitName>.md) and media/
├── extractions/          # <UnitName>_{vocabulary,grammar,summary}.md & <UnitName>_mindmap.json
└── handouts/             # <UnitName>_{vocabulary,reading,translation,listening,video}_quiz.html
```
- **Immutable Physical Keys**: Folder slug `<UnitName>` is the primary key (`normalize_name()`); display titles live in YAML frontmatter (`title: "..."`). No `raw/` directory, no `unit.json`, and zero background filesystem scans.

### 5.2 Live Configuration & Hardware Alignment (`wiki_config.json`)
- **Hot-Reloading**: `config.py` monitors `st_mtime` of `wiki_config.json`; changes take effect immediately without server restarts.
- **VRAM Concurrency**: $\le 12\text{GB}$ VRAM uses `max_parallel: 1` (`OLLAMA_NUM_PARALLEL=1`); $\ge 24\text{GB}$ VRAM scales to `max_parallel: 2`.
- **TTS Endpoints**: Kokoro (`http://localhost:8880/v1/audio/speech`, `af_sarah`/`am_michael`) and Edge-TTS (`http://localhost:5050/v1/audio/speech`, `en-US-AriaNeural`/`en-GB-RyanNeural`).
- **Dual-Distribution Compliance**: All `.db`/`.sqlite`/`.mdx` lexical binaries are git-ignored; when `pedlex.db` is absent or `DISABLE_PEDLEX=1`, `LinguisticEngine` smoothly falls back to `spaCy` + `cefrpy` + WordNet (`oewn:2025+`).

### 5.3 Naming & Formatting Conventions
- **JSON & Schemas**: Strict `snake_case` keys (`concepts`, `part_of_speech`, `explanation`, `target_language`).
- **Markdown**: Headwords and grammar names wrapped in `[[ ]]`; YAML frontmatter includes `source: "[[<UnitName>.md]]"`, `category`, and `overall_cefr_level`.

---

## 6. Documentation Index & Roadmap

### 6.1 Technical Documentation (`docs/`)
- [docs/pedlex-database-architecture.md](docs/pedlex-database-architecture.md): PedLex SQLite schema, 106k phrase inverted index, WSD scoring, and phrase distractor algorithms.
- [docs/vocabulary-selection-and-distractor-methodology.md](docs/vocabulary-selection-and-distractor-methodology.md): Valency cascades, single-fit collocational anchors, and distractor generation methodology.
- [docs/audit-scores-guide.md](docs/audit-scores-guide.md): Level 1 deterministic code gate and Level 2 expert evaluator rubrics.
- [docs/cli.md](docs/cli.md): CLI commands, compilation flags, and dashboard workflows.

### 6.2 Project Roadmap
- [x] **100% Deterministic Extraction Pipeline**: Vocabulary, multi-word expressions, and syntactic grammar extracted via `LinguisticEngine` + `pedlex.db` + `spaCy`.
- [x] **True Neuro-Symbolic Sense Reranker**: Pure symbolic Top-$K$ ($K \le 4$) filtering + 1-token Local LLM arbitration (80MB embedding model removed; 100% benchmark accuracy).
- [x] **Sense-First Quiz Blueprint Closure**: Zero `DEF_DRIFT`, zero `POS_DRIFT`, atomic Pattern–Example–Anchor binding, and post-inflection double-key collision screening verified across Books 2, 3, and 4.
- [x] **Unified One-Shot Structured Output & CEFR Grounding**: Direct JSON output + `cefrpy` text/word calibration + Dual-Distribution open-source adapter.
- [x] **Zero-Token Vocabulary Card Enrichment & Syntactic Slot Grammar Gating**: Enrich Markdown vocabulary cards with `get_vocab_card_enrichment` (`Pattern` / `Grammar Box`, `Collocations`, `Thesaurus Note`), enforce entryhead/sense `<gram>` constraints (`[only before noun]`, `[not before noun]`, `[uncountable]`, `[intransitive]`/`[transitive]`) across `_lock_sense` and `evaluate_syntactic_slot_clash`, and resolve definitions/examples across all 5,678 `derived` (`-ly`/`-ing`) entries.
- [x] **Orthographic / Paronym Distractor Tier (with Cognate Near-Synonym Guard)**: Introduce an orthographic look-alike (`形近词`) fallback/blending tier in `generate_vocab_distractors` for sparse synsets, strictly guarded by WordNet synset/hypernym disjointness and `double_key_collision` checks so root-sharing near-synonyms (e.g., `historic`/`historical`, `classic`/`classical`, `economic`/`economical`, `continual`/`continuous`) never create ambiguous double keys. But Paronyms like adapt / adopt / adept、complement / compliment、simulate / stimulate、thorough / through. 同根但语义域完全正交：如 specimen（标本/怪人） vs.  spectrum（光谱/范围） vs. spectacle（壮观场面/奇观） vs.
      speculation（推测/投机）
- [ ] **Pending — Symbolic Discourse Micro-Task Blueprints (`reading_quiz` & `listening_quiz`)**:
  - **Rhetorical & Span Anchoring (`spaCy` Discourse Markers)**: Pre-scan passage paragraphs and rhetorical transitions (causality, contrast/concession, stance/attribution, thesis) to deterministically assign each question an immutable `Anchor Span` (Paragraph + Sentence) and `Target Skill` (`Detail`, `Inference`, `Author's Stance/Purpose`, `Main Idea`), eliminating question clustering and full-text blind spots.
  - **Symbolic Distractor Fallacy Operators & Cross-Paragraph Lexical Bait**: Pre-assign propositional truth/fallacy operators to each option slot alongside a CRC32-locked `Prescribed Correct Answer Index`:
    - *Correct Slot (`Paraphrase Operator`)*: Enforce syntactic/lexical paraphrasing of the anchor span with a deterministic `Forbidden Verbatim Copy` list of core content words to prevent `Ctrl+F` surface matching.
    - *Distractor Slot 1 (`Cross-Paragraph Scrambling`)*: Inject authentic noun phrases extracted from non-anchor paragraphs as `Cross-Paragraph Bait Terms` to construct high-discrimination false associations.
    - *Distractor Slot 2 (`Polarity / Causal Reversal`)*: Invert the causal direction or affirmative/negative polarity of the anchor proposition.
    - *Distractor Slot 3 (`Modality / Scope Overgeneralization`)*: Escalate hedged modality (`may`, `tend to`, `some`) into absolute claims (`exclusively`, `inevitably`, `never`).
  - **Listening Turn-Taking Pragmatic State Machine**: Pre-compile a turn-by-turn pragmatic blueprint (`Turns 1–2` dilemma + target vocab $W_1$ $\rightarrow$ `Turns 3–4` stance shift/clarification + $W_2$ $\rightarrow$ `Turns 5–6` causal mechanism + $W_3$ $\rightarrow$ `Turns 7–8` resolution/next action) so dialogue generation and question anchoring are deterministically coupled.
- [ ] **Pending — Symbolic Translation Micro-Task Blueprint (`translation_quiz`)**:
  - **Step 1 — Register-Aligned Lexical & Syntactic Pairing (`Symbolic Pair Matching`)**: Deterministically pair each `Target Keyword` with a register-compatible `Target Grammar Pattern` (e.g., matching `[spoken]`/`[informal]` phrasal verbs like `wind up`/`pack up` with conversational clauses, and `[formal]` items like `originate`/`regulatory` with academic participial adjuncts, concessive clauses, or inversions) to prevent awkward "forced marriages".
  - **Step 2 — Single-Point Minimal-Pair Corruption Operator (`Targeted Error-Injection Spec`)**: Pre-assign a single deterministic flaw operator derived from `pedlex.db` (`Pattern` valency / `<gram>` property / `Grammar Box` pitfall) or `Grammar Warning (<TOPIC>)`:
    - *Lexical Valency / Property Trap*: e.g., preposition substitution (`originate in` $\rightarrow$ `*originate from/at*`) or `<gram>` violation (`[uncountable]` pluralization, `[only before noun]` predicative misuse `*is regulatory*`).
    - *Syntactic L1 Negative-Transfer Trap*: e.g., conjunction redundancy (`Although ... *but* ...`), clause-preposition confusion (`Despite + *finite clause*`), or dangling participle / comma splice.
    - Enforce a strict **Minimal Pair** contract: `idiomatic_translation` and `flawed_translation` must share $\ge 85\%$ identical non-target wording and differ **exclusively** at the prescribed trap locus (100% blind-solver discrimination, zero unfocused random typos).
  - **Step 3 — Zero-Redundancy Schema Pruning (~40% Token Reduction)**: Strip duplicate/blueprint-owned fields (`target_keyword`, `target_grammar`, `flaw_type`, `options`, `correct_answer_index`, `explanation`) from [`translation_quiz.json`](librarian/prompts/translation_quiz.json); have the LLM output only `design_audit`, `translated_sentence`, `idiomatic_translation`, `flawed_translation`, and `diagnostic_critique`, and hydrate `options` (CRC32-ordered `[idiomatic, flawed]`) and `explanation` deterministically in code.
  - **Canonical `🎯 Micro-Task` Blueprint Example**:
    ```markdown
    ### Item 2 Specification ###
    - Target Keyword: originate (verb) — "to come from a particular place or start in a particular situation"
    - Required Lexical Frame: originate in [place/situation] (Model: "Many Christmas traditions originated in Germany.")
    - Target Grammar Pattern: Concessive Clause (`Although / Even though + Subject + Verb, Main Clause`)
    - Prescribed Flaw Type: L1 Conjunction Redundancy (`Although ... but ...` double conjunction)
    - 🎯 Micro-Task for LLM:
      1. `translated_sentence`: Write a natural Chinese sentence meaning "虽然...（含 originate 词义：起源于某地/某种情况），但...".
      2. `idiomatic_translation`: Translate into natural CEFR B1 English integrating BOTH `originate in` and `Although + Clause, Main Clause` (NO 'but' in main clause).
      3. `flawed_translation`: Keep the sentence nearly identical as a Minimal Pair, ONLY introducing the prescribed flaw (adding redundant `but` before the main clause, or misusing `originate`).
      4. `diagnostic_critique`: Contrast the two versions by pointing out the exact `Although ... but ...` redundancy and confirming the `originate in` collocation.
    ```
- [ ] **Ongoing**: Expand discourse-level grammar pattern coverage and multi-word construction blanking families.