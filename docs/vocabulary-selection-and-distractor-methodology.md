# Pedagogical Vocabulary Selection, Collocation & Distractor Methodology

> **Purpose**: Detailed technical specification of the multi-tiered linguistic algorithms used by Lexis Wiki for candidate headword extraction, POS resolution, collocational valency cascading, and psychometric distractor generation.

---

## 1. Candidate Selection & "Quality-Over-Quantity" (宁缺勿滥) Three-Gate Pipeline

Candidate words and multi-word expressions extracted from authentic lesson passages (including textbook syllabus items) must satisfy a strict **Three-Gate Architecture** before entering the final vocabulary pool or assessment blueprint. **Hardcoded Python dictionaries (`curated_examples`, `priority_sense_examples`, `supplemental_vocab_senses`) are strictly forbidden** so that lexicon gaps and retrieval bugs are never silently masked.

```
[ Candidate Pool (Syllabus + Passage Mining) ]
                      │
                      ▼
┌───────────────────────────────────────────────────────────┐
│ Gate 1: Sense Locking (Top-K Symbolic + 1-Token LLM WSD)  │
│ - Locks the exact contextual Sense in pedlex.db           │
│ - Enforces POS hard gate (incl. licensed_poses),          │
│   direct sense_idx pointers, collocation gloss promotion, │
│   cross-entry crossrefs, & Zero-Fallback LLM arbitration  │
└───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────────┐
│ Gate 2: Sense-Isolated Authentic Example Gate             │
│ - Polysemous: MUST use examples/pattern_bundles from      │
│   the exact locked Sense (zero cross-sense stealing)      │
│ - Monosemous: Locked sense examples + inflection-aware    │
│   corpus search (search_corpus_examples) & WordNet        │
│ - Missing example? ──► 🗑️ DROP & REPLACE                  │
└───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────────┐
│ Gate 3: Collocational Anchor & Double-Key Screening       │
│ - Verifies single-fit anchor & 0 synonym collision        │
│ - No anchor / synset clash (e.g. crossbreed/hybridize)?   │
│   ──► 🗑️ DROP & REPLACE                                   │
└───────────────────────────────────────────────────────────┘
```

### 1.1 The Five Categories of Excluded / Re-Routed Items
1. **Example-Less Domain Verbs & Run-On Derivatives (🗑️ Drop & Replace)**:
   - Words that possess a sense definition in `pedlex.db` or WordNet (`oewn:2025+`) but **0 authentic dictionary or corpus examples** (e.g., `hybridize`, `crossbreed`, `domesticate`, `adulation`, or unattested run-on derivatives like `carpooling`, `seamlessly`, `comradely`, `conscientiously`, `weightlessness`).
   - Conversely, derived `-ly` adverbs and `-ing` participial adjectives/nouns that have authentic examples inside their base entry's `derived_forms` (`striving`, `enlightening`, `attentively`, `intensively`) or WordNet (`persevering`, `biologically`) are automatically resolved with full definitions and sense-verified examples.
   - Rather than fabricating examples in Python code—which forces downstream quizzes to test anchor-less words and invites synonym double-key collisions (`crossbreed` vs. `hybridize`)—genuinely example-less items are deterministically dropped (`[DROP_INCOMPLETE_LEXICON]`) and replaced by fully attested candidates from the passage.
2. **Transparent Free Word Combinations (🗑️ Drop & Replace)**:
   - Compositional phrases lacking a lexicalized multi-word entry in `pedlex.db` (e.g., `passenger plane`, `new generation`, `family member`, `put into the atmosphere`) are rejected at the expression gate.
3. **Inflected Multi-Word, Collocation-Box Idioms & Cross-Entry Corpus Items (✅ Recovered via Algorithmic Retrieval)**:
   - Authentic idioms and phrasal verbs whose examples in `pedlex.db` or WordNet appear in inflected surface forms (e.g., `pass away` $\rightarrow$ *"passed away"*, `stand for` $\rightarrow$ *"stood for"*, `show up` $\rightarrow$ *"showed up"*) or across other entries (`considerably`) are retrieved automatically via inflection-aware `search_corpus_examples` and multi-word WordNet synset lookup.
   - Multi-word idioms stored inside `COLLOCATIONS` boxes with inline `gloss` definitions (e.g., `peace of mind` under `peace` via `tier='colloc_phrase'` and direct `sense_idx` pointers) or via cross-entry `crossrefs` pointers (`pros and cons` under `pro_1` $\rightarrow$ `con_3`) are resolved directly to their authentic sense definitions and examples without heuristic string guessing.
4. **Polysemous Sense-Example Desynchronization (✅ Prevented via Strict Sense Isolation)**:
   - If a polysemous word's contextual sense lacks an example in `pedlex.db` (e.g., `compromise` in the transitive *"harm/weaken"* sense), `ensure_distinct_example` is strictly barred from stealing an example from another sense (*"compromise with the bosses"*). Without an authentic example for its locked sense, the item is cleanly dropped.
5. **Textual & CEFR Gates**:
   - The headword (or its inflected surface form) must physically occur in the lesson sentence pool, and its difficulty is calibrated deterministically via `cefrpy` + Zipf frequency ceilings.

---

## 2. Unified POS Resolution Protocol (`resolve_item_pos`)

To prevent discrepancies between manual teacher edits in markdown notes and the automated quiz generator, part-of-speech resolution is standardized through [`LinguisticEngine.resolve_item_pos`](../librarian/linguistics.py):

```
                       [ Input: word, quote, manual_pos ]
                                       │
                                       ▼
                     [ Multi-word or Expression Check ] ──► Return expression type
                                       │
                                       ▼
                     [ Closed-Class Function Word Check] ──► Return "function_word"
                                       │
                                       ▼
                     [ Contextual Syntactic Parse (spaCy) ]
                     - Finite verb / root predicate?
                     - Adjectival participle modifier?
                     - Nominal head or object argument?
                                       │
                                       ▼
                     [ PedLex Lexical POS Attestation ]
                     - Cross-reference with entry senses & all_poses
                                       │
                                       ▼
                     [ Unified Canonical POS Determination ]
```

- **Node 1 (Extraction Writer)**: Validates POS before writing to `extractions/*_vocabulary.md`.
- **Node 2 (Quiz Skeletons Builder)**: Dynamically resolves canonical POS at runtime, ensuring tests evaluate authentic syntactic roles even if a markdown file contains historical manual mislabels.

---

## 3. Valency Cascades by Part of Speech

### 3.1 Nouns: SLA & Psychometric Context Clues
Rather than imposing stiff, rare dictionary verbs (*extend hospitality*), noun stems employ triangulated narrative context clues:
- **Triangulation (Paul Nation paradigm)**:
  - *Agent/Role Clue* (e.g., *researchers, innkeepers, villagers*)
  - *Setting/Prop Clue* (e.g., *laboratory data, hearth, legal code*)
  - *Affective/Reaction Clue* (e.g., *relieved sigh, exhausted travelers*)
- **Cognitive Polarity**: Uses concession/contrast (`Although [hardship]... [positive target]`) to logically mandate a unique semantic fit.
- **Plurale Tantum Invariant**: If the target noun is plural or *plurale tantum* (`NNS`), computational NLP automatically inflects all 3 distractors to plural form, guaranteeing 100% morphological symmetry.

### 3.2 Transitive vs. Intransitive Verbs
- **Transitive Verbs** (*solve, abandon, provide, accelerate*):
  - Strict selectional restrictions on direct object patient/theme:
  - `object` (Direct Object Noun) $\gg$ `prep` (Bound Preposition) $\gg$ `adv_mod` (Manner Adverb).
- **Intransitive Verbs** (*listen, arrive, depend, hesitate*):
  - Cannot take direct objects; interaction relies strictly on prepositions or adverbial adjuncts:
  - `prep` (Bound Preposition) $\gg$ `adv_mod` (Manner Adverb) $\gg$ `verb_subject` (Subject Noun).
- **Prepositional Valency Frames**: Active frames governing direct objects (*provide sb with sth*) explicitly bind the direct patient slot: `[Subject] + ____ + [Person / Direct Object] + [prep] + [Complement]`.

### 3.3 Adjectives: Prepositional vs. Descriptive
- **Prepositional Valency Adjectives** (*proud of, aware of, responsible for*):
  - Bound prepositional government:
  - `prep` (Bound Preposition) $\gg$ `modified_noun` $\gg$ `adv_mod` $\gg$ `verb_copula`.
- **Descriptive / Relational Adjectives** (*economic, legal, simple*):
  - `modified_noun` (Characteristic Noun) $\gg$ `adv_mod` (Collocational Degree Adverb) $\gg$ `verb_copula` $\gg$ `prep`.

### 3.4 Adverbs: Modification Domains
- **Degree / Focus / Stance Adverbs** (*extremely, highly, deeply*):
  - Intensifiers or evaluative markers modifying scalar adjectives:
  - `modifies_adj` $\gg$ `modifies_verb`.
- **Manner / Time / Frequency Adverbs** (*abruptly, politely, frequently*):
  - Circumstantial adjuncts modifying core action verbs:
  - `modifies_verb` $\gg$ `modifies_adj`.

### 3.5 Closed-Class Function Words
Organized into discourse families (`concession`, `cause`, `condition`, `time`, `scope`, `noun_clause`).
- **Syntactic Complement Contrast**: Pairs clausal connectors (*although, whereas*) against prepositional connectors (*despite, because of*). All options share the identical semantic orientation, but only the target fits the syntactic complement slot (`____ + NP`), eliminating ambiguity.

---

## 4. Distractor Generation Architecture (`generate_vocab_distractors`)

Distractors are generated deterministically at zero token cost (<1ms) via a five-tier pipeline:

1. **PedLex Topic & Activator Lexicon**:
   - Harvests semantic neighbors and concept cluster members.
   - Applies **Cluster Throttling** (maximum 1 candidate per cluster/phrase) and word-family deduplication (`are_same_word_family`) to avoid near-identical variants (e.g. *businessman* vs *businessperson*).
2. **PedLex Thesaurus Synonyms & Antonyms**:
   - Secondary source for taxonomically related content words, filtered against synonym collision.
3. **WordNet Taxonomic Filtering**:
   - Bipolar cluster harvesting and coordinate sisters within the same POS and semantic field.
4. **Orthographic / Paronym Look-Alike Tier (`get_orthographic_distractors`)**:
   - Provides high-discrimination orthographic look-alikes (`形近词`, e.g., `adapt`/`adopt`/`adept`, `complement`/`compliment`, `simulate`/`stimulate`, `specimen`/`spectrum`/`spectacle`) when semantic clusters are sparse, strictly guarded by WordNet synset/hypernym disjointness and `is_cognate_near_synonym` so root-sharing near-synonyms (`historic`/`historical`, `classic`/`classical`, `economic`/`economical`, `continual`/`continuous`) never create ambiguous double keys.
5. **Physical Quality, Syntactic `<gram>`, & Difficulty Gates**:
   - **CEFR Difficulty Ceiling**: Intercepts words exceeding the text CEFR level using `cefrpy` and Zipf frequency ratings.
   - **Syntactic Slot & `<gram>` Compatibility (`evaluate_syntactic_slot_clash`)**: Enforces `pedlex.db` grammatical constraints (`[only before noun]` vs. predicative copula slots, `[not before noun]` vs. attributive modifier slots, `[uncountable]` vs. `a/an` determiner frames, and strict `[intransitive]` vs. `[transitive]` direct-object compatibility).
   - **Morphological Symmetry & Post-Inflection Collision Check**: Parallel inflection is applied across all 4 options (`VBD`/`VBN`/`VBG`/`VBZ`/`NNS`), followed by a post-inflection `double_key_collision` re-verification against the target's bound collocations and participial synsets (`0 DOUBLE_KEY_SURVIVED`).

---

## 5. Eight Lexico-Syntactic Construction Families (Unified Architecture)

In English, lexical items frequently project **constructional or clausal subcategorization frames** (Catenae / Construction Grammar). Because `vocabulary_quiz` enforces a strict **single-blank invariant** (`re.findall(r'_{2,}', stem) == 1`), these lexico-syntactic structures cannot be treated as flat contiguous strings. Instead, they are systematically classified into **Eight Lexico-Syntactic Construction Families** across **Extraction Routing** (`extractions/<Unit>_{vocabulary,grammar}.md`) and **Quiz Blueprint Specialization** (`build_precomputed_target_skeletons`).

### 5.1 Overview Matrix: Extraction Boundary & Single-Blank Strategy

| Family | Canonical Structure | Examples | Extraction Routing | Single-Blank (`____`) Decomposition & Frame Anchor |
| :--- | :--- | :--- | :--- | :--- |
| **Family 1A: Preparatory-Object Extraposition** | `<V> [sth] <adj/NP>` $\rightarrow$ `<V> it <adj/NP> ... to do [sth] / that...` | `make [sth] possible`, `make [sth] easy`, `find [sth] hard`, `take it for granted that` | **Vocabulary / Expressions** (`<Unit>_vocabulary.md`) | **Blank**: 3-word chunk (`make it possible`, `find it hard`).<br>**Frame**: `... ____ ... to do [sth]`. |
| **Family 1B: Preparatory-Subject Extraposition** | `It is <adj/NP/V-ed> (for/of sb) to do [sth] / doing / that...` | `It is hard/easy/difficult/necessary (for sb) to do`, `It is no use/good doing`, `It is worth doing` | **Syntactic Grammar** (`Information Packaging: it-extraposition` in `<Unit>_grammar.md`); never a multi-word vocab phrase | **Blank**: Single-word syllabus adjective (`hard`, `easy`, `difficult`, `necessary`, `possible`) when quote/model uses `It is <adj> to do`.<br>**Frame**: `It is ____ (for [sb]) to do [sth]`. |
| **Family 2: Discontinuous Correlatives & Paired Frames** | `Arm 1 ... Arm 2` | `not only ... but (also)`, `either ... or`, `neither ... nor`, `so/such ... that`, `too ... to do`, `hardly/scarcely ... when`, `no sooner ... than` | **Expressions / Grammar** | **Blank**: Opening trigger (`not only`, `neither`, `no sooner`, `hardly`).<br>**Frame**: Lock second correlative arm (`... ____ ... but also ...`, `... ____ ... than ...`). |
| **Family 3: Discontinuous `Verb + [sb/sth] + Prep / Particle`** | `<V> [sb/sth] <Prep / Particle>` | `remind [sb] of [sth]`, `provide [sb] with`, `accuse [sb] of`, `blame [sb] for`, `prevent/stop/keep [sb] from doing`, `turn [sth] ↔ down` | **Vocabulary / Expressions** (`<Unit>_vocabulary.md`) | **Blank**: Governing verb (`remind`, `accuse`, `prevent`).<br>**Frame**: Lock object + bound preposition/gerund (`... ____ [sb] of [sth]`, `... ____ [sb/sth] from [doing sth]`). |
| **Family 4: Non-Finite Verb Complementation (`to do` vs. `doing` vs. `do`)** | `<V / Phrase> + [doing sth] / to do [sth] / [sb] do [sth]` | `look forward to doing`, `be used to doing`, `devote oneself to doing`, `have difficulty (in) doing`, `spend [time] doing`, `cannot help doing`, `avoid/admit/deny doing` | **Vocabulary / Expressions** (`<Unit>_vocabulary.md`) | **Blank**: Matrix verb or multi-word prefix (`look forward to`, `avoid`).<br>**Frame**: Lock non-finite complement form (`... ____ [doing sth]`, `have difficulty ____`). |
| **Family 5: Shell Nouns & Post-Nominal Clausal Licensing** | `the <Noun> + that-clause / to do [sth] / of [doing sth]` | `the fact/assumption/belief that...`, `the ability/tendency/attempt/effort/right to do`, `the habit/risk/intention of doing` | **Vocabulary** (Single-Word Noun) & **Grammar** (`Cohesion & Framing: shell nouns`) | **Blank**: Shell noun (`ability`, `tendency`, `assumption`).<br>**Frame**: Lock post-nominal complement (`the ____ to do [sth]`, `the ____ of [doing sth]`, `the ____ that ...`). |
| **Family 6: Delexical Light-Verb + Noun + Prep Triads** | `<Light V> + (det/adj) + <Eventive N> + <Prep>` | `take advantage of`, `make use of`, `pay attention to`, `catch sight of`, `keep track of`, `make sense of`, `give rise to`, `have an impact on`, `play a role in` | **Expressions** (`collocation` in `<Unit>_vocabulary.md`) | **Blank**: Full 3-word triad (`take advantage of`) **or** internal noun with locked verb+prep (`take ____ of`, `pay [close] ____ to`). |
| **Family 7: Negative Polarity Items (NPIs) & Inversion Triggers** | Fronted Negative Chunk + `Aux + S + V` / Negative Clause + `NPI` | `under no circumstances`, `by no means`, `on no account`, `in no way`, `only by/when...`; `let alone`, `much less`, `far from`, `at all` | **Expressions / Grammar** (`Rhetoric & Emphasis: inversion`) | **Blank**: NPI or fronted trigger (`let alone`, `under no circumstances`).<br>**Frame**: Enforce negative clause or inverted auxiliary (`____ + Aux + S + V`) in `🎯 Micro-Task`. |
| **Family 8: Mandative Subjunctive & Modal Idiom Triggers** | `<Trigger> + that S (should) V_base` / `<Idiom> + S + V_past` | `insist/suggest/demand/recommend/vital/imperative that S (should) V_base`, `would rather S V_past`, `it is high time S V_past`, `may/might as well do` | **Vocabulary & Grammar** (`Logic & Stance`) | **Blank**: Matrix trigger or modal idiom.<br>**Frame**: Enforce subordinate verb morphology (`that [sb] (should) [V-base]`, `would rather [sb] [V-past]`). |

---

### 5.2 Detailed Blueprint & Distractor Specifications (`build_precomputed_target_skeletons`)

During quiz generation, [`build_precomputed_target_skeletons`](../librarian/linguistics_core/quiz_skeletons.py) inspects both the target item and its locked `Quoted Sentence` / `Corpus Example` (`Model`) to emit the construction-specific `Pattern`, `Fit into frame '...'`, and zero-collision distractors:

#### 1. Family 1A: Preparatory-Object Extraposition (`<V> it <adj> ... to do [sth]`)
- **Contiguous Target & Comparative Normalization**:
  - Discontinuous syllabus notation `make [sth] possible` / `make [sth] easy` / `find [sth] hard` is never stripped into ungrammatical 2-word `*make possible` / `*make easy`.
  - Matches both base and comparative dictionary patterns (`make(s) it possible`, `make it easier (to do something)`, `find it hard to do something`) and normalizes the target blank to the 3-word chunk `<V> it <adj>` (`make it possible`, `make it easy`, `find it hard`).
- **Pattern & Cloze Slot Frame**:
  - `Pattern`: `<V> it <adj> ... to do [sth]`
  - `Frame`: `Fit into frame '... ____ ... to do [sth]'.`
- **Dedicated Distractor Strategy (Same-Verb Subcategorization Contrast)**:
  - Distractors must share the leading `<V> it` prefix (`make it ...` or `find it ...`) so surface length and verb valency match, while their complement **rejects** `(for sb) to do sth` (taking `that`-clauses or NP objects instead):
    - For `make it <adj>`: `make it clear`, `make it known`, `make it certain`, `make it plain`, `make it good`.
    - For `find it <adj>`: `find it strange`, `find it odd`, `find it clear`, `find it funny`.
  - **Double-Key Ban**: Other infinitive-licensing adjectives (`possible`, `impossible`, `easy`, `hard`, `difficult`) and phrasal particles (`make it up`, `make it with`) are strictly excluded.

#### 2. Family 1B: Preparatory-Subject Extraposition (`It is <adj> (for/of sb) to do [sth]`)
- **Quote-First Activation**:
  - Triggered **only** when a single-word adjective target (`hard`, `easy`, `difficult`, `necessary`, `important`, `essential`, `vital`, `possible`, `rare`, `normal`, `safe`, `dangerous`, `wise`, `foolish`) has an `It is <adj> (for/of sb) to do` passage quote or locked dictionary example. If the quote tests a pre-nominal noun modifier (`the necessary skills`) or true prepositional complement (`necessary for maintaining health`), standard noun/preposition anchoring applies.
- **For-To Infinitival Guard & Pattern Completion**:
  - Blocks `spaCy` from misparsing the `for sb` subject of `for sb to do sth` as a bound prepositional collocation (`*necessary for [sth]`) or misparsing the infinitive verb as a `modified_noun` (`*difficult + understand`).
  - Expands truncated dictionary patterns (`It's difficult`, `Is it possible to`) into the full extraposition frame:
    - `Pattern`: `it is <adj> (for [sb]) to do [sth]`
    - `Frame`: `Fit into frame 'It is ____ (for [sb]) to do [sth]'.`
- **Dedicated Distractor Strategy (Chomsky `easy` vs. `eager` Subcategorization Contrast)**:
  - **Step 1 — Ban Same-Slot Extraposition Adjectives**: Strictly exclude all other adjectives that grammatically license dummy `It is ____ (for sb) to do sth` (`hard`, `difficult`, `tough`, `easy`, `simple`, `possible`, `impossible`, `necessary`, `essential`, `vital`, `important`, `common`, `normal`, `rare`).
  - **Step 2 — Prioritize Animate-Subject Control / Experiencer Adjectives**: Select distractors that take `to do` or prepositional complements **only with an animate/human subject** and are ungrammatical after dummy `It` (`*It is eager / willing / reluctant / afraid / anxious / proud / capable / allergic / fond to do sth`).
  - **Step 3 — Attributive-Only Syntactic Foils**: Blend `[only before noun]` adjectives (`solid`, `indoor`, `prior`, `excess`, `mere`, `joint`) that cannot occupy a predicative `It is ____` slot.

#### 3. Family 2: Discontinuous Correlative & Paired Constructions (`not only ... but (also)`, etc.)
- **First-Arm Blanking + Second-Arm Frame Locking**:
  - A single blank cannot simultaneously replace both halves of a discontinuous correlative (`not only ... but also`, `either ... or`, `neither ... nor`, `whether ... or`, `so ... that`, `such ... that`, `too ... to do`, `hardly/scarcely ... when`, `no sooner ... than`).
  - The blueprint decomposes the item into:
    - **Target Word (Blank `____`)**: The opening trigger (`not only`, `neither`, `either`, `no sooner`, `hardly`).
    - **Locked Frame Anchor**: The correlative partner embedded verbatim in the Cloze Slot Frame (`... ____ ... but also ...`, `... ____ ... nor ...`, `... ____ ... than ...`).
- **Dedicated Distractor Strategy (Cross-Correlative Pairing Mismatch)**:
  - Distractors are drawn from competing correlative/discourse triggers that are ungrammatical with the locked partner arm in the stem (e.g., when the stem locks `... but also ...`, options pair `not only` against `neither`, `either`, `as well as` / `rather than`).

#### 4. Family 3: Discontinuous `Verb + [sb/sth] + Prep / Particle` Frames (`remind [sb] of`, `prevent [sb] from doing`)
- **Verb Blanking + Object-Preposition Frame Locking**:
  - Extending the `discontinuous_prep` logic in `quiz_skeletons.py`, transitive verbs with obligatory object-assigned prepositions (`remind/inform/convince/accuse [sb] of`, `provide/supply/present [sb] with`, `blame/thank [sb] for`, `prevent/stop/keep/discourage [sb] from [doing sth]`) blank the **governing verb** (`____`) while locking `[sb/sth] + <Prep> (+ [doing sth])` in the stem frame.
- **Dedicated Distractor Strategy (Prepositional Valency Clash)**:
  - Distractors are chosen from semantically adjacent verbs that govern incompatible prepositions or complement structures (e.g., for `____ [sb] of [sth]`: `explain` [`to`], `mention` [`to`], `suggest` [`to`]; for `____ [sb] from doing`: `forbid` [`to do`], `avoid` [`doing` without `from`], `refuse` [`to do`]).

#### 5. Family 4: Non-Finite Verb Complementation (`to do` vs. `doing` vs. bare `do`)
- **Complement-Form Frame Preservation**:
  - Covers preposition-`to` + gerund phrases (`look forward to doing`, `be used/accustomed to doing`, `devote oneself to doing`, `object to doing`), implicit gerund frames (`have difficulty/trouble (in) doing`, `spend/waste [time] doing`, `cannot help doing`, `be busy doing`), and single-word verbs with strict gerund (`avoid`, `admit`, `deny`, `postpone`, `risk`) or infinitive (`manage`, `afford`, `refuse`, `fail`, `hesitate`) subcategorization.
  - The blueprint preserves `[doing sth]` or `to do [sth]` inside the `Cloze Slot Frame` (`... ____ [doing sth]`).
- **Dedicated Distractor Strategy (Non-Finite Subcategorization Inversion)**:
  - When the frame locks `[doing sth]`, distractors prioritize verbs/phrases that strictly require `to do [sth]` (`manage`, `afford`, `refuse`, `agree`, `decide`, `fail`).
  - Conversely, when the frame locks `to do [sth]`, distractors prioritize gerund-only verbs (`avoid`, `enjoy`, `mind`, `admit`, `postpone`, `deny`) while banning aspectual verbs that accept both (`begin`, `start`, `continue`).

#### 6. Family 5: Shell Nouns & Post-Nominal Clausal Licensing (`the ability to do`, `the fact that`)
- **Post-Nominal Complement Frame Locking**:
  - Abstract head nouns ("shell nouns") license specific post-nominal complements:
    - Infinitive-licensing: `ability`, `tendency`, `attempt`, `effort`, `willingness`, `reluctance`, `decision`, `right`, `opportunity`, `failure` + `to do [sth]`.
    - Prepositional-gerund licensing: `habit`, `intention`, `risk`, `method`, `way`, `fear`, `possibility` + `of [doing sth]`.
    - Appositive `that`-clause licensing: `fact`, `belief`, `assumption`, `notion`, `evidence`, `doubt`, `rumor` + `that ...`.
- **Dedicated Distractor Strategy (Complement Clash & Synset Disjointness)**:
  - Strictly filter out co-licensing synonyms (`attempt` vs. `effort` before `to do`) and select abstract nouns whose post-nominal subcategorization clashes with the locked complement (e.g., testing `ability` in `the ____ to adapt` against `possibility` [`*the possibility to adapt` $\rightarrow$ `of adapting`], `habit` [`of adapting`], and `fact` [`that...`]).

#### 7. Family 6: Delexical Light-Verb + Noun + Preposition Triads (`take advantage of`, `pay attention to`)
- **Dual Blanking Modes**:
  1. **Full 3-Word Triad Blanking (`take advantage of`, `make use of`, `keep track of`, `catch sight of`, `give rise to`)**:
     - Filters synonymous triads via `_PHRASE_SYNONYMS_CACHE` (`take advantage of` vs. `make use of`; `have an impact on` vs. `have an effect on`) so both never appear in the same option set.
  2. **Internal Noun / Light-Verb Blanking (`pay [close] ____ to`, `____ advantage of`)**:
     - Activated when the passage quote inserts an internal adjective modifier (`pay close attention to`, `have a profound impact on`, `play a vital role in`), locking `<Light V> + <Adj> + ____ + <Prep>` in the stem frame.

#### 8. Families 7 & 8: Negative Polarity Items (NPIs), Fronted Inversion Triggers & Mandative Subjunctives
- **Syntactic Environment Enforcement in `🎯 Micro-Task`**:
  - **Family 7 (NPIs & Fronted Negative Inversion)**:
    - When `pedlex.db` marks an expression with `[usually in negatives]` (`let alone`, `much less`, `at all`, `any longer`) or fronted negative inversion (`under no circumstances`, `by no means`, `on no account`, `in no way`, `little did...`), the `Micro-Task` explicitly mandates a negative matrix clause (for NPIs) or inverted `Aux + S + V` word order (for fronted triggers).
    - Fronted inversion triggers are paired with non-inverting prepositional phrase distractors (`in addition`, `by accident`, `on purpose`, `in particular`), allowing pure syntactic elimination when `Aux + S + V` follows the blank.
  - **Family 8 (Mandative Subjunctive & Modal Idioms)**:
    - For mandative triggers (`insist`, `suggest`, `demand`, `recommend`, `propose`, `vital`, `essential`, `imperative` + `that S (should) V_base`) and modal idioms (`would rather S V_past`, `it is high time S V_past`, `may/might as well do`), `Micro-Task` locks the subordinate verb form and feeds these structures directly into `translation_quiz` as high-yield minimal-pair grammar traps.

#### 9. Family 9: Internal Possessive Idioms (`come to one's aid`, `a roof over one's head`, `come into one's mind`) — Whole-Idiom Preservation + Concrete Possessive Pronoun & Tense Synchronization
- **Core Linguistic Problem**:
  - Multi-word idioms and collocations containing internal `one's` / `somebody's` / `someone's` slots (`come to one's aid`, `a roof over one's head`, `come into one's mind`, `turn one's attention to`, `do one's utmost`, `pick one's brains`, `catch one's eye`) possess **non-compositional idiomatic semantics**.
  - Splitting such an idiom to blank only a single word (e.g., `come into her ____` or `____ her mind`) destroys the holistic idiomatic unit (`come into one's mind` = *suddenly occur to someone*) or opens massive single-word collocation double-keys. Conversely, leaving raw dictionary citation forms (`"come to one's aid"`, `"a roof over one's head"`) inside `options` causes severe pronoun and tense clashes in concrete third-person narrative stems (`*A kind stranger immediately come to one's aid`).
- **Four-Step Neuro-Symbolic Solution**:
  1. **Whole-Idiom Slot Preservation**:
     - The entire multi-word idiom is preserved inside the blank `____` (never fractured into a single-word noun or verb slot).
  2. **Strategy A2 Structural Symmetry & Truncation Guards (`generate_phrase_distractors`)**:
     - Queries `pedlex_phrase_index` (`sense_unit` and `phrase_row` complete idiom tiers prioritized over bare patterns) for structurally parallel `one's` idioms matching the target's syntactic skeleton (`<Verb> + <Prep> + one's + <Noun>`, `<Verb> + one's + <Noun> + <Prep>`, `<Det> + <Noun> + <Prep> + one's + <Noun>`).
     - Enforces three hard structural filters:
       - **Trailing Preposition vs. Adverbial Particle Symmetry**: When the target ends with a relational preposition requiring a complement (`turn one's attention to [sth]`), candidates ending in intransitive adverbial particles (`turn one's nose up`) are rejected unless they share the exact trailing preposition (`set one's mind on`, `get one's teeth into`, `turn one's back on`).
       - **Truncated Phrasal-Verb Prefix Guard**: 3-word `one's` prefixes that only exist in `pedlex_phrase_index` as 4-word particle constructions (`keep one's hands [off]`, `let one's guard [down]`, `keep one's head [down]`) are filtered out.
       - **Near-Synonym & Register Guard**: Co-occurring synonymous idioms (`turn one's mind to` vs. `turn one's attention to`, `do one's best` vs. `do one's utmost`, `come to one's rescue` vs. `come to one's aid`) and undignified colloquialisms (`pick one's nose`) are blocked.
  3. **Concrete Possessive Determiner & Singular Head-Noun Agreement (`_reconcile_prescribed_inflection`)**:
     - Extracts the concrete 3rd-person possessive determiner (`her`, `his`, `their`, `its`) attested in the passage `quote` (primary) or `authentic_example` (secondary; e.g., converting 1st/2nd-person `"came to my aid"` $\rightarrow$ `"her"` / `"his"`).
     - **Singular Countable Head-Noun Agreement Guard**: If the extracted possessive is plural `"their"` (e.g., from a plural subject in `"put a roof over their heads"`), but the dictionary idiom's head noun after `one's` is singular countable (`head`, `mind`, `eye`, `cap`, `hand`, `back`, `heart`, `way`), the synchronizer normalizes the determiner to singular `"his"` (or `"her"`) so options never produce ungrammatical distributive number clashes like `*a roof over their head` or `*a feather in their cap`.
  4. **Parallel Leading-Verb Tense Inflection & Cross-Possessive Frame Slicing**:
     - Detects the target verb's inflectional tag (`VBD`, `VBZ`, `VBG`, `VBN`, `VB`) from the passage quote or winning dictionary example and inflects the leading verb across all 4 idiom options in lockstep (e.g., `["came to her aid", "got to her feet", "had to her credit", "looked to her laurels"]`).
     - `derive_cloze_slot_frame_from_example` matches across any possessive pronoun variant (`my`/`your`/`his`/`her`/`our`/`their`/`one's`) and uses `spaCy` `nsubj` recovery for finite verbal idioms so `cloze_slot_frame` anchors on the subject (`... a man ____`) rather than a stray preposition (`*... to ____`), while `🎯 Micro-Task` explicitly instructs the LLM to introduce an unambiguous matching antecedent (`a woman/girl` for `her`, `a man/boy` for `his`) and keep the possessive + noun inside `____`.

---

### 5.3 Data-Driven Execution Engine: Two-Tier `pedlex.db` Construction Tagging & Reverse Subcategorization Index (`pedlex_cxn_index`)

To execute all Eight Construction Families with **zero hardcoded word lists or brittle regex trees in runtime Python**, the assessment machine delegates both **Forward Pattern Triggering** and **Reverse Mutually-Exclusive Distractor Mining** to an offline-compiled `pedlex.db` table: [`pedlex_cxn_index`](./pedlex-database-architecture.md#8-implemented-architecture-two-tier-construction-tagging-pedlex_cxn_index--minimal-pair-error-traps-pedlex_error_traps).

1. **Two-Tier Trigger Attachment (Resolving Polysemy & Multi-Frame Words)**:
   - **Tier 1 (Phrase / Entry Level — `sense_idx = -1`)**: Monosemous multi-word constructions (`make [sth] possible`, `not only ... but also`, `no sooner ... than`, `look forward to`, `take advantage of`, `under no circumstances`, `let alone`, `would rather`) carry their `cxn_spec` directly at the entry/phrase level. Selecting the phrase in a quiz 100% triggers its fixed `canonical_blank`, `canonical_pattern`, and `cloze_frame`.
   - **Tier 2 (`Sense -> pattern_bundle` Level — `sense_idx >= 0`)**: Polysemous single words (`hard`, `necessary`, `remind`, `prevent`, `avoid`, `manage`, `ability`, `possibility`, `insist`) attach `cxn_spec` to the specific **`pattern_bundle`** inside `senses[]` rather than the global headword.
     - After `Step 0 (_lock_sense)` and `Step 1 (pick_atomic_pattern_bundle)` match the passage `Quoted Sentence` to its exact dictionary pattern (e.g., matching *"It is hard for children to understand"* to `"it is hard (for somebody) to do something"`), the bundle's `cxn_spec` automatically activates **Family 1B** (`It is ____ (for [sb]) to do [sth]`).
     - Conversely, when the passage quote is *"Brick is a hard material"*, `pick_atomic_pattern_bundle` selects an attributive noun-modifier bundle (`cxn_spec = null`), cleanly routing to standard `modified_noun` collocation anchoring.
2. **Reverse Subcategorization Set-Difference for Zero-Collision Distractors (`subcat_group`)**:
   - During offline compilation ([`scripts/upgrade_pedlex_db.py`](../scripts/upgrade_pedlex_db.py)), `pedlex.db` scans `senses[].patterns` and `<gram>` across all 48,960 consolidated entries to populate 2,635 rows in `pedlex_cxn_index`, partitioning words into mutually exclusive syntactic equivalence classes (`ADJ_IT_EXTRAPOS` vs. `ADJ_ANIMATE_CONTROL`; `V_GERUND_ONLY` vs. `V_INF_ONLY`; `N_SHELL_TO_DO` vs. `N_SHELL_OF_DOING` vs. `N_SHELL_THAT_ONLY`; `PHR_FRONTED_INV` vs. `PHR_NON_INV_PP`).
   - At runtime (<0.2ms), `LinguisticEngine.get_word_cxn_specs` and `LinguisticEngine.query_cxn_contrast_distractors` query `pedlex_cxn_index` for words belonging to the target's `contrast_groups` while excluding `ban_groups` ($\text{Contrast} \setminus \text{Ban}$), filtered by `cefrpy` level and WordNet synset disjointness—guaranteeing 100% syntactic elimination of distractors and 0 double-key collisions.



---

## 6. Six-Source Optimal Example Pre-Selection & Example-Derived Cloze Slot Frame Architecture

### 6.1 Core Architectural Insight ("Select Optimal Example First, Then Slice the Frame")
Empirical evaluation across Books 1–4 (`Book_1_Unit_1`, `Book_1_Unit_2`, `Book_2_Unit_1`, `Book_2_Unit_2`, `Book_3_Unit_6`, `Book_4_Unit_1`, `Book_4_Unit_5`) revealed four structural root causes whenever `vocabulary_quiz` items degraded to bare `... ____ ...` frames or suffered pattern–example desynchronization:
1. **Cross-Sense Collocation Pollution on Polysemous Entries (`plot`)**:
   - In `pedlex.db`, entry-level `collocations` are stored at the homograph level rather than per `sense_idx`. For polysemous words like `plot` (Sense 0: *conspiracy* vs. Sense 1: *storyline of a book/film*), blindly harvesting entry-level verb collocations assigned `discover a plot` (*conspiracy*) to a *storyline* item.
   - Conversely, `Sense.pattern_bundles` (`the plot unfolds`), `Language Activator` (`[the events in a story]` $\rightarrow$ *"The plot was so complicated..."*), and `Oxford Thesaurus` (*"the plot of her next novel"*) are strictly concept-isolated.
2. **Idiom Over-Binding vs. Syntactic Valency (`weird`, `routine`)**:
   - `Sense.pattern_bundles` mixes `source: "gramexa"` (core syntactic subcategorization such as `pledge to do something`, `pose for`, `be absorbed in`, `benchmark for`, `originate in`) with `source: "colloinexa"` (incidental idioms such as `weird and wonderful` or `matter of routine`). Treating `colloinexa` idioms as mandatory patterns artificially locks general adjectives (`weird`) into fixed idioms (`... ____ and wonderful`) even when the passage tests simple predicative or `adj + noun` usage (*"a weird dream"*).
3. **Fragment / Postpositive Sense Examples When Rich Full Sentences Exist Elsewhere (`unparalleled`, `earthy`, `impersonal`, `tedious`, `loan`)**:
   - Relying solely on `sense['examples'][0]` often selects a verbless fragment (`"a loan of £60,000"`, `"earthy language"`, `"a formal and impersonal style of management"`) or an atypical postpositive participial clause (`"an achievement unparalleled in sporting history"`) when `LDOCE Thesaurus`, `Language Activator`, or `Oxford Thesaurus` provides a complete, idiomatic sentence with a canonical slot (`"We took out a loan to buy a new car."`, `"She was shocked by Emma's earthy language."`, `"an unparalleled opportunity to change society"`, `"Jake began the tedious task of sorting through his papers."`).
4. **Ambiguous Slash Placeholders (`... ____ someone/something`)**:
   - Generalizing `[sb]` or `[sth]` into `____ someone/something` forces the LLM to choose between human and non-human objects (or copy the literal string `someone/something` into `design_audit`). Instead, object animacy/form must be resolved deterministically from the locked example (`appeal to the government` $\rightarrow$ `... ____ someone`; `on top of everything else` $\rightarrow$ `... ____ something`).

### 6.2 Six-Source Candidate Example Pool (Strict Sense Isolation)
Before constructing any quiz blueprint in [`build_precomputed_target_skeletons`](../librarian/linguistics_core/quiz_skeletons.py), the engine gathers candidate `(example, pattern_hint, source_tier)` tuples across all six dictionary compartments in `pedlex.db`, subject to **Strict Sense Isolation**:
1. **`grammar_boxes` (Tier 1 — Pedagogical Error-Trap Goldmine)**:
   - Matched via `sense_indices` (or monosemous headword). Provides high-discrimination positive models paired with `Don't say` rules (`ability to listen` vs. `*of listening`; `take into account` vs. `*in account`; `admit making` vs. `*to make`; `give some good advice` vs. `*an advice`).
2. **`Sense.pattern_bundles` & `Sense.examples` (Tier 2 — Core Sense Valency)**:
   - Directly inside `locked_sense`. `source == "gramexa"` bundles receive top syntactic priority (`pledge to do`, `pose for`, `be absorbed in`, `benchmark for`, `originate in`, `escalate into`).
3. **`LDOCE Thesaurus` (`thesaurus` self-entries — Tier 3 — Concept-Aligned Full Sentences & `<gram>` Tags)**:
   - Matched by headword + `pos` + concept/definition overlap with `locked_sense`. Supplies explicit countability/position tags (`loan [countable]` vs. `interest [uncountable]`, `urban [only before noun]`, `intrigued [not before noun]`) and curated full-sentence examples (`"We took out a loan to buy a new car."`, `"Jake began the tedious task of sorting through his papers."`).
4. **`Language Activator` (`language_activator` — Tier 4 — Concept-Indexed Phrase & Subcategorization Frames)**:
   - Matched by concept gloss + headword/subphrase (`[the events in a story / plot]`, `[to make something more modern / modernize]`, `[to increase / escalate to]`).
5. **`Oxford Thesaurus` (`oxford_thesaurus` — Tier 5 — Canonical `adj + noun` / `verb + noun` Supplement)**:
   - Matched by `pos` + sense synonym/example overlap. Rescues senses whose LDOCE entry only contains fragments or postpositive constructions (`"an unparalleled opportunity to change society"`, `"she was shocked by Emma's earthy language"`, `"the plot of her next novel"`).
6. **`collocations` & `cross_collocations` (Tier 6 — Sense-Verified Entry Collocations)**:
   - Enabled unconditionally for monosemous words, or for polysemous words **only when** the collocation's `gloss`/`example` shares lexical cues with `locked_sense` or the passage `quote` (preventing cross-sense drift like `discover a plot` for storyline `plot`).

### 6.3 Deterministic Scoring Rubric for Selecting the Optimal Quiz Example
Every candidate example in the sense-isolated pool is scored deterministically so the **single best pedagogical example** wins before slot-frame extraction:
- **Rule 1 (Complete Finite Sentence > Fragment)**: Full sentences containing a finite verb (`ROOT` / `aux`) outrank verbless noun-phrase fragments (`"We took out a loan to buy a new car."` $\gg$ `"a loan of £60,000"`; `"She was shocked by Emma's earthy language."` $\gg$ `"earthy language"`).
- **Rule 2 (Syntactic Slot & Inflection Alignment with Target/Distractors)**:
  - When an adjective target is paired against `[not before noun]` distractors (e.g., `unparalleled` vs. `alone`) or appears attributively in the passage (`unparalleled levels`, `a modernized nation`, `a tedious oddity`), examples where the target occupies a **pre-nominal attributive slot (`det/poss + ____ + noun`)** receive a decisive bonus (`"an unparalleled opportunity..."`, `"a tastefully modernized old farmhouse"`, `"the tedious task"`).
  - When a countable singular noun target is paired against `[uncountable]` distractors (e.g., `loan` vs. `interest`/`credit`), examples containing an **explicit singular indefinite article + verb (`take out a loan`, `became a benchmark for`)** receive a decisive bonus.
- **Rule 3 (`gramexa` Valency > General Collocation > Idiomatic `colloinexa`)**:
  - Core syntactic subcategorization (`gramexa` / `grammar_boxes`, e.g., `pledge to do`, `pose for`, `absorbed in`, `benchmark for`) outranks general `verb + noun` / `adj + noun` collocations, which in turn outrank narrow `colloinexa` coordinate idioms (`weird and wonderful`) unless the passage quote itself uses that idiom.
- **Rule 4 (Pedagogical Clarity & Length)**:
  - Sentences of 6–22 words with clean everyday vocabulary outrank overly long corpus extracts containing obscure proper nouns or technical jargon.

### 6.4 Example-Derived Concrete Cloze Slot Frame (`0 Bare Frames`, `0 Slash Choices`)
Once the winning `authentic_example` is locked, `cloze_slot_frame` is **sliced directly from that example's syntactic dependency neighborhood around the target word** (combining `gramexa` pattern structure with concrete lexical/POS neighbors from the winning sentence):
1. **Guaranteed Left or Right Context Neighbor**:
   - **Adjective (`adj`)**: `(det/poss) + ____ + head_noun` or `____ + prep` (e.g., `'... an ____ opportunity'`, `'... her ____ language'`, `'... the ____ task'`, `'... a (very) ____ career'`, `'... ____ words'`, `'... ____ to'`).
   - **Verb / Phrasal Verb (`verb` / `phrase`)**: `____ + prep`, `____ + to [verb]`, `____ + (det) + dobj_noun`, or `subj_noun + ____` (e.g., `'... ____ to [verb]'` for `pledge`, `'... ____ for'` for `pose`, `'... ____ in'` for `absorbed`, `'... ____ a decision'` for `await`, `'... ____ a programme'` for `put out`, `'... ____ the system'` for `modernize`).
   - **Noun (`noun`)**: `verb + (a/the/poss) + ____ (+ prep)` or `(det/adj) + ____ + prep/verb` (e.g., `'... take out a ____'` for `loan`, `'... a ____ for'` for `benchmark`, `'... the ____ unfolds'` or `'... the ____ of'` for `plot`, `'... a personal ____'` for `triumph`).
   - **Adverb (`adv`)**: `(aux) + ____ + verb/adj` or `verb + ____` (e.g., `'... is ____ working'` or `'... ____ employs'` for `currently`).
2. **Zero Slash Placeholders (`someone/something` Ban)**:
   - Any `[sb/sth]` or `someone/something` placeholder is resolved against the actual argument in the winning `authentic_example`: human/animate/organization objects (`the government`, `her`, `farmers`) resolve to `someone`, and inanimate/abstract objects (`everything else`, `the French one`) resolve to `something`.
