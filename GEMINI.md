# Lexis Wiki Project Instructions

> **Purpose**: Single source of truth for the Lexis Wiki project — a neuro-symbolic educational knowledge base and interactive assessment generator that transforms primary texts, audio, and video into Obsidian-style wiki nodes and psychometrically audited HTML quizzes.

---

## 1. Neuro-Symbolic Core Architecture

### 1.1 Quad-Tier Separation of Responsibilities
1. **Symbolic Grounding & Lexical Corpus (`LinguisticEngine` + `pedlex.db` + WordNet + `cefrpy`)**:
   - 100% authoritative semantic truth and constraint engine.
   - Combines offline computational NLP (`spaCy` dependency parsing, POS tagging, morphological inflection) with Pedagogical Lexicon (`pedlex.db`: 48,960 `ldoce` entries + 189,410 `ldoce_phrase_index` + 1,043,571 `ldoce_sense_cue_index` + `pedlex_cxn_index` + `pedlex_error_traps`), WordNet (`oewn:2025+`), and `cefrpy`.
   - Deterministically extracts vocabulary, multi-word expressions, and syntactic patterns; pre-computes single-fit collocational anchors, inflected options, and zero-collision distractors.
2. **Code Gate & Invariant Enforcement (`processor.py`, `evaluator.py`)**:
   - Enforces physical invariants at zero token cost (<5ms): verbatim quotation checks, single-blank constraints (`re.findall(r'_{2,}', stem) == 1`), option cardinality (exactly 4 options), CRC32 deterministic slot assignment, and CEFR lexical ceiling gates.
   - Executes instant programmatic self-healing and category auto-remapping without burning LLM retry tokens.
3. **Pedagogical Prompting (`librarian/prompts/*.md`)**:
   - Focuses strictly on Second Language Acquisition (SLA) standards, contextual stem crafting, and psychometric discrimination.
   - Executes pre-computed `🎯 Micro-Task for LLM` blueprints (for quizzes) or discourse synthesis (for summaries/mind maps) without distractor-inventing burdens.
4. **Schema Enforcement (`schemas.py`)**:
   - Defines strict `snake_case` JSON structures via Prompt-Guided JSON Mode (`format: "json"`).

### 1.2 Architectural Discipline & Theory-First Correction Mandate
- **Zero Tolerance for Band-Aid Bloat**: Never accumulate brittle word-specific regex hacks, ad-hoc `if-else` patches, or arbitrary exception branches.
- **First-Principles Escalation**: Proactively escalate linguistic edge cases to clean neuro-symbolic mechanisms (Top-$K$ symbolic candidate filtering + 1-token LLM arbitration).
- **Mandatory Correction of Flawed Methods (`违背常识与既有学科理论的方法必须主动纠偏`)**:
  - Never blindly execute requests that violate SLA, Conversation Analysis, cognitive load, or psychometric theory.
  - Proactively challenge flawed assumptions (e.g., plotline-stitching in listening) and enforce established theoretical models (Conversation Analysis Three-Part Turn Micro-Structure, Lexical Approach semi-open skeletons, Light/Heavy Turn Alternation, and Cambridge/IELTS `Bait → Real Key` listening signposts).

---

## 2. Sense-First Extraction & WSD Architecture

### 2.1 Deterministic Extraction & Evidence-Completeness Gate ("宁缺勿滥")
- **Deterministic Extraction (`extract_deterministic_vocabulary`, `extract_deterministic_expressions`, `extract_deterministic_grammar`)**:
  - Extracted deterministically from `sources/<Unit>.md` + syllabus via `spaCy` dependency trees and `pedlex.db`.
  - Multi-word expressions are strictly typed (`phrasal verb`, `collocation`, `set phrase`, `idiom`), never labeled with token-level POS. Slash alternatives (`get / feel / be lost`) are normalized prior to `/IPA/` stripping.
- **Three-Gate Quality Pipeline**:
  1. **Gate 1 — Sense Locking**: Locks the exact contextual `Sense` in `pedlex.db` (or WordNet) matching POS, syntactic frame, and semantics.
  2. **Gate 2 — Sense-Isolated Authentic Example Gate (`ensure_distinct_example`)**: Polysemous words must draw authentic examples exclusively from their locked sense's own `examples`/`pattern_bundles`. Monosemous words may use inflection-aware `search_corpus_examples`. Any item with 0 authentic examples for its locked sense is dropped (`[DROP_INCOMPLETE_LEXICON]`). Zero hardcoded Python example/sense dictionaries.
  3. **Gate 3 — Collocational Anchor & Zero Double-Key Gate**: Candidates lacking discriminative context anchors or prone to synonym collisions (e.g. `crossbreed` vs `hybridize`) are dropped before quiz generation.

### 2.2 Two-Stage Neuro-Symbolic Sense Locking (WSD)
1. **Stage 1 — Symbolic Candidate Pool (`pedlex.db` + WordNet, 0ms / 0 tokens)**:
   - Applies POS hard gates (cross-homograph resolution `word_1`..`word_6`), `<gram>` slot constraints (`[only before noun]`, `[uncountable]`), register/attitude gates (`[spoken]`, `[disapproving]`), conditional-subject ontology gates, and syntactic pattern intersection to build a Top-$K$ ($K \le 4$) candidate pool.
   - ~71% monosemous entries lock instantaneously at 0 token cost.
2. **Stage 2 — 1-Token Local LLM Arbitration (~29% Polysemous Cases)**:
   - Formats Top-$K$ ($K \le 4$) senses with lexicographic `[SIGNPOST]` cues and syntactic patterns alongside passage context into a multiple-choice prompt resolved via a 1-token (`A`/`B`/`C`/`D`) local LLM call. Zero free-generation authority.

---

## 3. Modular Sense-First Quiz Blueprint Pipeline

### 3.1 Downstream Markdown Decoupling & Read-Only Assessment Truth
- `extractions/<Unit>_{vocabulary,grammar}.md` are canonical, user-verified knowledge nodes. Assessment blueprint pipelines consume these verified extraction artifacts in read-only mode, with absolute zero disk writes to `extractions/` during quiz generation. Missing extractions are strictly intercepted by the Prerequisite Extraction Gate.

### 3.2 Modular Blueprint Mixins & Pure Five-Source LDOCE Pool
- **Modular Architecture (`librarian/linguistics_core/`)**:
  - `quiz_patterns.py`: Five-source authentic example pre-selection and atomic pattern bundles.
  - `quiz_frames.py`: First-principles verb patient ontology and example-derived cloze slot frame slicing.
  - `quiz_vocab.py`: Unified facade composing modular skeleton mixins (`quiz_vocab_cloze`, `quiz_skeleton_prep`, `quiz_skeleton_anchors`, `quiz_skeleton_distractors`, `quiz_skeleton_frames`, `quiz_skeleton_builder`).
  - `quiz_translation.py`: Translation minimal-pair micro-task blueprints and error-injection specs.
  - `quiz_listening.py`: Functional dialogue trajectories and listening comprehension blueprints.
  - `quiz_skeletons.py`: Unified Facade Mixin exposing backward-compatible public APIs.
- **Five-Source Sense-Isolated LDOCE Pool (`select_optimal_five_source_example`)**:
  - Draws candidate examples strictly across 5 LDOCE compartments: (1) `grammar_boxes`, (2) locked `Sense.pattern_bundles` (`gramexa`) & `Sense.examples`, (3) `LDOCE Thesaurus`, (4) `Language Activator`, and (5) curated `collocations` (gated by single-sense or lexical overlap). Oxford Thesaurus and unedited BNC dumps are excluded.
  - Full finite clauses strongly outrank fragments. Attributive vs predicative slot constraints and valency patterns are strictly enforced.
- **Example-Derived Cloze Slot Frame Slicing (`derive_cloze_slot_frame_from_example`)**:
  - Sliced directly from the winning example's syntactic neighborhood (`... their ____ achievement`, `... ____ [concrete physical object] into`). Zero `None` frames, zero unexpanded slash choices.
- **Distractor Architecture & Universal Post-Assembly Double-Key Guard**:
  - Valency cascades: nouns use SLA context clues; transitive verbs prioritize patient object $\gg$ prep; intransitive verbs prioritize prep $\gg$ adv; adjectives enforce attributive (`[only before noun]`) vs predicative (`[not before noun]`) complements.
  - Multi-word phrases use symmetric tail-sharing, particle contrasts, and possessive pronoun & tense synchronization (`come to one's aid` $\rightarrow$ `came to her aid`).
  - Dynamic CEFR fallback pool (`_ensure_corpus_fallback_index`, 8,102 core words) replaces hardcoded fallback lists.
  - Post-assembly screening verifies `double_key_collision` and `evaluate_syntactic_slot_clash` across ALL options against the final context anchor, auto-refilling any colliding distractor.

---

## 4. Deterministic CEFR Calibration & QA Audit

### 4.1 Zero-LLM CEFR Calibration (`cefrpy`)
- Headword and passage CEFR levels (`A1`–`C2`) are computed deterministically via `cefrpy` + Zipf frequency gates; output schemas exclude CEFR fields so LLMs never guess difficulty.

### 4.2 Two-Level QA Quality Audit
1. **Level 1: Deterministic Code Gate (`evaluator.py`, <5ms, 0 tokens)**:
   - Structural invariants, option length bounds, Fisher-Yates shuffling with atomic answer-index remapping, deterministic grammar category remapping.
2. **Level 2: Expert Model Semantic Audit (`expert_auditor.py`, LLM-as-a-Judge)**:
   - Optional (`enable_expert_audit` in `wiki_config.json`). Blind solver resolution with 3-tier triage: Tier 3 `PASS` (80–100), Tier 2 `REPAIR` (60–79), Tier 1 `REWRITE` (20–59).
3. **Delivery Status**: Direct delivery to `extractions/` or `handouts/` with frontmatter `qa_status: "passed" | "review_needed" | "failed"`.

---

## 5. Stateless Storage & Hardware Conventions

### 5.1 Self-Contained Directory Layout
```text
wiki/<UnitName>/
├── sources/              # Primary text (<UnitName>.md) and media/
├── extractions/          # <UnitName>_{vocabulary,grammar,summary}.md & <UnitName>_mindmap.json
└── handouts/             # <UnitName>_{vocabulary,cloze,reading,translation,listening,video}_quiz.html
```
- Folder slug `<UnitName>` is the immutable physical key (`normalize_name()`). Display titles live in YAML frontmatter (`title: "..."`).

### 5.2 Runtime Configuration (`wiki_config.json`)
- **Hot-Reloading**: `config.py` monitors `st_mtime` of `wiki_config.json`.
- **VRAM Concurrency**: $\le 12\text{GB}$ VRAM uses `max_parallel: 1`; $\ge 24\text{GB}$ scales to `max_parallel: 2`.
- **Dual-Distribution Compliance**: All `.db`/`.sqlite` binaries are git-ignored; when `pedlex.db` is absent or `DISABLE_PEDLEX=1`, `LinguisticEngine` falls back to `spaCy` + `cefrpy` + WordNet (`oewn:2025+`).

---

## 6. Technical Documentation & Roadmap

### 6.1 Documentation Index (`docs/`)
- [docs/vocabulary-selection-and-distractor-methodology.md](docs/vocabulary-selection-and-distractor-methodology.md): Single source of truth for candidate extraction, POS resolution, optimal example selection principles across parts of speech, valency cascades, distractor generation, and the 8 lexico-syntactic construction families.
- [docs/multi-word-units-slicing-and-distractor-architecture.md](docs/multi-word-units-slicing-and-distractor-architecture.md): Technical specification and applied lexicographic taxonomy for 9 multi-word unit (MWU) assessment families, deterministic cloze slot slicing frames, PedLex SQL mapping, Oxford Thesaurus near-synonym harvesting, and zero-collision collocational mutual exclusion gates.
- [docs/pedlex-database-architecture.md](docs/pedlex-database-architecture.md): PedLex SQLite database schemas (`ldoce`, `ldoce_phrase_index`, `ldoce_sense_cue_index`, `pedlex_cxn_index`, `pedlex_error_traps`), JSON payload specifications, and runtime query interfaces.
- [docs/listening-functional-dialogue-architecture.md](docs/listening-functional-dialogue-architecture.md): Four pragmatic interaction paradigms, 14 functional dialogue archetypes, conversation analysis turn structures, and listening blueprints.
- [docs/translation-minimal-pair-and-error-trap-architecture.md](docs/translation-minimal-pair-and-error-trap-architecture.md): Contrastive SLA assessment framework, Sense-first CEFR-gated error traps (`pedlex_error_traps`), zero-synthetic distractor invariant, dynamic 4-tier authentic stem harvesting, dialogic speech-act filtering, and graceful syntactic backfill.
- [docs/paired-listening-speaking-workbench-architecture.md](docs/paired-listening-speaking-workbench-architecture.md): Receptive-to-productive pairing architecture, AI-generated scene dossiers, Conversation Analysis (CA) turn microscopes, paired jigsaw role cards, and role-reversal automaticity drills.
- [docs/reading-discourse-architecture.md](docs/reading-discourse-architecture.md): Discourse graph extraction, 6 discourse relation types, 10 psychometric distractor fallacy operators, single-step inference entailment, and verbatim lexical leakage guard.
- [docs/cloze-discourse-architecture.md](docs/cloze-discourse-architecture.md): Authentic multi-paragraph Pure Open Cloze (`mode: "open_cloze"`) specification decoupled from MCQ distractors, Acceptable-Word Scoring (`accepted_answers`), Alderson (2000) lead-in/lead-out invariants, balanced tripartite slot typology (Discourse Cohesion / Bound Collocation / Lexico-Grammar), and zero-token L1 code gate.
- [docs/cumulative-lexical-ledger-and-cross-book-ancestry.md](docs/cumulative-lexical-ledger-and-cross-book-ancestry.md): Cumulative Lexical Ledger (CLL) architecture, dual-pathway syllabus synchronization (bottom-up extraction + top-down appendix PDF ingest), compound primary keying (`lemma#pos`), genuine homograph preservation, cross-book ancestry tracking, Second Language Acquisition (SLA) spiral progression tiers (`target_new` ~80-85% / `recycled_sense_shift` ~10-15% / `recycled_spaced_review` ~0-5%), and psychometric quiz quota governance with graceful backfilling.
- [docs/audit-scores-guide.md](docs/audit-scores-guide.md): Level 1 deterministic code gate and Level 2 expert evaluator rubrics.
- [docs/digital-human-characters.md](docs/digital-human-characters.md): Digital human dialogue cast archetypes, facial geometry standards, and multi-character rendering pipelines.
- [docs/esp-trade-logistics-roadmap-and-validation.md](docs/esp-trade-logistics-roadmap-and-validation.md): Technical roadmap, corpus collection checklist, and 6-stage model validation specification for foreign trade and logistics specialized English (ESP).
- [docs/cli.md](docs/cli.md): CLI commands, compilation flags, and dashboard workflows.

### 6.2 Project Roadmap
- [x] **100% Deterministic Extraction & Neuro-Symbolic WSD**: Pure symbolic Top-$K$ + 1-token LLM arbitration.
- [x] **Sense-First Quiz Blueprint Closure**: Five-source LDOCE pool, modular architecture (`quiz_vocab.py` etc.), zero drift, universal post-assembly double-key collision screening verified across Books 1–4 (30 units, 300 items, 100% pass rate).
- [x] **Verb Patient Ontology & Dynamic Fallback Pool**: WordNet supersense + cognitive subdomains for patient slots; 8,102-word dynamic core lexicon fallback.
- [x] **Symbolic Translation & Listening Blueprints**: Minimal-pair error-injection (`translation_quiz`); 14 dialogue archetypes with pragmatic turn structures (`listening_quiz`).
- [x] **Symbolic Reading Discourse Blueprint**: Pre-assign rhetorical span anchors (`spaCy` transitions), 10 deterministic distractor fallacy operators (cross-paragraph bait, causal reversal, modality overgeneralization, explicit restatement), single-step inference entailment, and 3-gram verbatim lexical leakage code gate.
- [x] **Syntactic Frame Clash & Countability Symmetry Gates**: Pre-assembly and post-assembly code gates for prepositional government (`insight into` vs `of`), `to do` vs `to -ing` non-finite subcat frame clash (`reluctance to do` vs `aversion to -ing`), and bidirectional countability quota symmetry (`advice` vs `suggestion`, `device` vs `equipment`).
- [x] **Negative Inversion Trigger Distractor Arena**: For high-order negative/restrictive adverbs fronted in inverted structures (`____ had the meeting begun when...`), ban non-inverting adverb giveaways and enforce distractors strictly from the `_NEGATIVE_INVERSION_ADVERBS` peer family (`scarcely`, `hardly`, `barely`, `seldom`, `rarely`, `no sooner`) with correlative (`when` vs `than`) and aspectual clash gating.
- [x] **Light-Verb Collocation Mutual Exclusion Web**: For delexical/light-verb collocations (`cast doubt on`, `shed light on`, `strike a balance`, `pay tribute to`), provide high-discrimination verb-slot peer distractors (`cast` vs `throw` vs `shed` vs `put`) with dictionary idiom validation as the sole arbiter.
- [x] **Discourse Cloze Engine (Pure Open Cloze + Acceptable-Word Scoring)**: Transform cloze into authentic multi-paragraph Pure Open Cloze (`mode: "open_cloze"`, `options: []`) completely decoupled from syllabus lists and 4-option MCQ distractors; paragraph fitness selector, balanced tripartite slot selection (P1 Sentence Linking, P2 Collocations, P3 Grammar & Usage), Alderson invariants, exact char-offset masking, per-unit target deduplication, `accepted_answers` whitelist grading, clean underline HTML + print worksheet interface (auto-hint on focus without penalty, 100-point scale, `Answer & Note` on check/`Reveal All`), L2 MCQ-auditor bypass, and standalone Dashboard `Cloze` modality. Verified across 30 units (295 blanks, `P1=29.2% / P2=42.7% / P3=28.1%`, 0 duplicate targets, 0 offset mismatches, 0/30 L1 code gate defects).
- [x] **Cumulative Lexical Ledger & Cross-Book SLA Spiral Progression**: Multi-book / multi-series Cumulative Lexical Ledger (CLL) persisted under `wiki/_glossary/<series>/Book_<N>_glossary.json`; dual-pathway synchronization (bottom-up unit extraction + top-down appendix PDF ingest with native browser stream upload), `lemma#pos` compound keying with canonical POS normalization (`adj`/`adv`) and genuine homograph separation, ascending cross-book ancestry tracking (`get_prior_books`), tripartite SLA spiral progression categorization (`target_new` ~80-85% / `recycled_sense_shift` ~10-15% / `recycled_spaced_review` ~0-5%), psychometric quiz quota governor (`allocate_quiz_lexical_quota`) with 100% graceful backfilling, and Dashboard modal workbench with CEFR/SLA pill stats, tier filters, and table micro-badges (`[Shift · B1]`, `[Review · B1]`). Audited across Book 1 (476 entries, 4 genuine homographs, 0 duplication) and Book 2 (148 entries, 100% field completeness).
- [x] **Extraction Semantic Re-Audit Suite (`re-audit --target extraction`)**: Extended audit suite to cover `extractions/` artifacts (`summary`, `mindmap`, `grammar`, `vocabulary`) via Level-1 zero-token deterministic code gates and optional Level-2 expert judge (`qwen3.8:27b`) semantic evaluation, focusing on factual entailment / anti-hallucination quote verification against source text, MECE cognitive hierarchy and structural orthogonality for mindmaps, Quirk/Biber academic grammar precision, and contextual sense alignment.
- [x] **Golden Set End-to-End Regression Suite (`lexis golden-set`)**: 5 canonical units spanning CEFR A1-C1 (`Book_1_Unit_1_Passage_A`, `Book_1_Unit_2_Passage_A`, `Book_1_Unit_1_Section_A`, `Book_3_Unit_6_Section_A`, `Book_4_Unit_1_Section_A`) benchmarked across source -> extraction -> blueprint -> handout -> audit pipeline. Verifies 25 handouts and all extractions with zero defects in <3.5s (`tests/test_golden_set_e2e.py`).
- [ ] **Paired Listening–Speaking Workbench**: Two-act receptive-to-productive pairing (`Act I: Listening` $\rightarrow$ `Act II: Speaking`) with 3-level progressive fading scaffolding (Cue Card $\rightarrow$ Acoustic Cloze $\rightarrow$ Blind Production) and cross-scenario transfer.
- [x] **Prerequisite Extraction Gate**: Gate quiz generation on verified `extractions/` artifacts to prevent topic drifts and eliminate side-effect overwrites.
- [ ] **ESP Specialized Domain Extension (Paused / Backlog)**: Multi-source trade/logistics corpus collection, Incoterms/UCP600 legal consistency gates, and domain-gated WSD (`pedlex_esp.db`). Paused to prioritize core engineering hygiene, code guardrails, and architectural debt reduction.
- [x] **Blueprint `SlotFrame` & Distractor Engine Gradual Refactoring**: Introduced structured `SlotFrame` dataclass with 100% backward-compatible string contracts; decomposed 8,825-line monolith `distractor_engine.py` into 6 focused domain mixins (`distractor_guards`, `distractor_paronyms`, `distractor_phrases`, `distractor_inflections`, `distractor_fallbacks`, `distractor_vocab`) and a 42-line clean facade mixin.
- [x] **Extraction Pipeline Monolith Decomposition**: Decomposed the 6,365-line monolith `extraction_pipeline.py` into 4 focused domain mixins (`extraction_source.py`, `extraction_grammar.py`, `extraction_expressions.py`, `extraction_vocab.py`) and a 30-line clean facade mixin (`ExtractionPipelineMixin`), with 100% backward-compatible method and attribute exposure.
- [x] **CEFR & Morphology Monolith Decomposition**: Decomposed the 4,455-line monolith `cefr_and_morphology.py` into 4 focused domain mixins (`cefr_calibration.py`, `morphology_engine.py`, `contextual_pos.py`, `multiword_syntax.py`) and a 57-line clean facade mixin (`CefrAndMorphologyMixin`), preserving all 68 methods and class-level caches with zero contract drift.
- [x] **Evaluator Dimension Scorers Monolith Decomposition**: Decomposed the 1,546-line monolith `dimension_scorers.py` (featuring a 1,210-line mega-method `_score_pedagogy`) into 6 focused modular scorers (`scorer_schema.py`, `scorer_verbatim.py`, `scorer_uniqueness.py`, `scorer_pedagogy_extraction.py`, `scorer_pedagogy_quiz.py`, `scorer_pedagogy.py`) and a 21-line clean facade with 100% backward compatibility for all 4 level-1 code gate functions.
- [x] **Collocations & Cards Monolith Decomposition**: Decomposed the 3,152-line monolith `collocations_and_cards.py` into 4 focused domain mixins (`collocation_extractor.py`, `anchor_evidence.py`, `vocab_card_enrichment.py`, `zero_collision_anchors.py`) and a 25-line clean facade mixin (`CollocationsAndCardsMixin`), preserving all 31 methods, 5 class-level constants, and zero contract drift.
- [x] **PedLex Database Monolith Decomposition**: Decomposed the 3,363-line monolith `pedlex_db.py` into 5 focused domain mixins (`pedlex_core.py`, `pedlex_patterns.py`, `pedlex_phrases.py`, `pedlex_thesaurus.py`, `pedlex_cxn_traps.py`) and a 36-line clean facade mixin (`PedlexDbMixin`), preserving all 50 methods, 14 class-level constants, and zero contract drift.
- [x] **Exporter & Portal Renderer Monolith Decomposition**: Decomposed the 1,391-line monolith `exporter.py` into 4 focused modules (`bundle_packager.py`, `markdown_renderer.py`, `portal_renderer.py`, `unit_exporter.py`) under `librarian/exporter_core/` and a 43-line clean facade (`exporter.py`), with 100% backward-compatible function and symbol exposure.
- [x] **LLM Client & JSON Healing Monolith Decomposition**: Decomposed the 2,018-line monolith `llm.py` into 5 focused domain modules/mixins (`model_profiles.py`, `model_lifecycle.py`, `json_repair.py`, `backends.py`, `chat_executor.py`, `client.py`) under `librarian/llm_core/` and a 25-line clean facade (`llm.py`), preserving all 15 methods, 8 attributes, and class-level capabilities with zero contract drift.
- [x] **Frontend Template & Static Asset Decoupling**: Decoupled inline CSS and HTML document structures from `portal_renderer.py` and `markdown_renderer.py` into dedicated authoring templates (`portal.html`, `handout.html`) and static stylesheet assets (`portal.css`, `handout.css`) under `librarian/templates/`, supported by a high-performance cached `template_loader.py` enforcing the 100% self-contained offline delivery invariant. Shrunk `portal_renderer.py` by 70% and `markdown_renderer.py` by 55%.
- [x] **CLI Monolith Decomposition**: Decomposed the 921-line monolithic CLI manager (`cli.py`, containing an 850-line `main()` function) into 9 focused subcommand modules under `librarian/cli_core/` (`cmd_init`, `cmd_lint`, `cmd_config`, `cmd_compile`, `cmd_quiz`, `cmd_dashboard`, `cmd_export`, `cmd_audit`, `cmd_pedlex`) using a clean subparser registration and handler dispatch pattern, shrinking `cli.py` into a 44-line facade with 100% backward compatibility.
- [x] **Vocab Distractor Monolith Decomposition**: Decomposed the 3,271-line monolith `distractor_vocab.py` (the largest single method/monolith in the repository) into 3 focused domain mixins (`distractor_vocab_wsd.py` [775 lines], `distractor_vocab_harvest.py` [1,342 lines], `distractor_vocab_assembly.py` [1,228 lines]) and a 149-line clean facade (`DistractorVocabMixin`), exposing proper classmethod predicates (`_pos_ok`, `_is_primarily_adj_or_verb`, `_is_primarily_noun_or_adj_for_verb`, `_is_informal_or_slang_distractor`) with 100% backward compatibility across all 59+ specialized distractor tests, golden-set benchmarks, and caller integrations.
- [x] **Quiz Frames Monolith Decomposition**: Decomposed the 2,457-line monolith `quiz_frames.py` into 2 focused domain mixins (`quiz_frames_patient.py` [734 lines: verb patient ontology classification, WordNet supersenses, and animacy resolution], `quiz_frames_slicing.py` [1,722 lines: cloze slot frame derivation, collocation half-slice, and clause isolation]) and a 21-line clean facade (`QuizFramesMixin`), preserving 100% backward-compatible method and attribute exposure across all 50 specialized frame tests and golden-set benchmarks.
- [x] **Extraction Vocab Monolith Decomposition**: Decomposed the 2,276-line monolith `extraction_vocab.py` into 3 focused domain mixins (`extraction_vocab_candidates.py` [1,093 lines: AWL mining and Top-K candidate pool generation], `extraction_vocab_examples.py` [486 lines: Gate 2 sense-isolated authentic example selection], `extraction_vocab_extractor.py` [709 lines: deterministic extraction pipeline, verbatim surface matching, and video vocabulary]) and a 27-line clean facade (`ExtractionVocabMixin`), preserving 100% backward compatibility across all extraction unit tests and golden-set benchmarks.
- [x] **WSD Scoring Monolith Decomposition**: Decomposed the 2,266-line monolith `wsd_scoring.py` (featuring a 2,215-line mega-method `_score_single_sense`) into 3 focused domain mixins (`wsd_scoring_gates.py` [330 lines: Stage 1 invariant gates, particles, morphology, register, domain, and token overlap], `wsd_scoring_patterns.py` [696 lines: Stage 2 pattern frame matches, separable phrasals, and unit alignment], `wsd_scoring_dependencies.py` [1,354 lines: Stage 3 authentic dependency trees, argument structure, locatives, selectional restrictions, and cue bonus]), a shared dataclass module (`wsd_scoring_context.py` [59 lines: `WsdContext`, `WsdSenseInfo`]), and a 31-line clean facade (`WsdScoringMixin`), preserving 100% backward compatibility across all 70 WSD tests, smoke tests, and golden-set benchmarks.
- [ ] **Agent Skill & MCP Cloud Service Packaging (Pending / Cost Feasibility)**: Evaluate packaging Lexis Wiki core capabilities (wiki node synthesis, psychometric quiz blueprinting, L1 zero-collision audit gate) into external Agent Skills (e.g. Agensi.io marketplace, Claude Code, Cursor MCP). Deferred to pending track due to cloud hosting/VPS cost constraints (`spaCy`, ~400MB `pedlex.db` memory-mapping, Python runtime); prioritize zero-cost local CLI/desktop execution first.
- [ ] **Exam Reading Reverse Deconstruction Skill & Big-Tech Model Interface (Pending / Market Validation)**: Reverse-engineer high-stakes exams (Cambridge KET/PET/FCE, IELTS) by pairing our symbolic ground truth (LDOCE paraphrase mapping, 10 psychometric distractor fallacy operators, discourse argumentation graph) with big-tech LLM APIs (DeepSeek, Qwen, Claude) or agent platforms (Coze/扣子, WeChat AI, OpenAI Actions, MCP). Decouples heavy GPU LLM costs from our lightweight CPU+SQLite core (~20-30 RMB/mo or serverless) to serve mobile-first test-taker / parent demand.
- [ ] **Engineering Hygiene & Architecture Improvement Roadmap (Active Focus - Phase 1)**: Tracked phased plan covering lint/CI guardrails, outer-layer test coverage (`exporter.py`, `dashboard.py`), broad exception triage, monolith decomposition, frontend/template modularization, and Golden Set E2E benchmarks; see [docs/project-evaluation-and-improvement-roadmap.md](docs/project-evaluation-and-improvement-roadmap.md).
