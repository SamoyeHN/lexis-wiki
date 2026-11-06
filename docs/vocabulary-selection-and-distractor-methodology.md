# Pedagogical Vocabulary Selection, Collocation & Distractor Methodology

> **Purpose**: Single authoritative technical specification of the multi-tiered linguistic algorithms in Lexis Wiki — covering candidate extraction, contextual POS resolution, definition-locked WSD, optimal authentic example pre-selection across parts of speech, cloze slot frame slicing, valency cascading, and psychometric distractor generation.

---

## 📋 Table of Contents
1. [Candidate Selection & Three-Gate Pipeline ("宁缺勿滥")](#1-candidate-selection--three-gate-pipeline-宁缺勿滥)
2. [Contextual Grounding & Syntactic POS Resolution Hierarchy](#2-contextual-grounding--syntactic-pos-resolution-hierarchy)
3. [Two-Stage Neuro-Symbolic WSD & Domain Register Penalty](#3-two-stage-neuro-symbolic-wsd--domain-register-penalty)
4. [Five-Source Optimal Example Pre-Selection Architecture](#4-five-source-optimal-example-pre-selection-architecture)
5. [Linguistic Principles of Optimal Example Selection across Parts of Speech](#5-linguistic-principles-of-optimal-example-selection-across-parts-of-speech)
   - [5.1 Nouns (名词精选例句的语言学原理)](#51-nouns-名词精选例句的语言学原理)
   - [5.2 Verbs (动词精选例句的语言学原理)](#52-verbs-动词精选例句的语言学原理)
   - [5.3 Adjectives (形容词精选例句的语言学原理)](#53-adjectives-形容词精选例句的语言学原理)
   - [5.4 Adverbs (副词精选例句的语言学原理)](#54-adverbs-副词精选例句的语言学原理)
   - [5.5 Multi-Word Expressions & Idioms (多词短语与习语精选例句的语言学原理)](#55-multi-word-expressions--idioms-多词短语与习语精选例句的语言学原理)
6. [Valency Cascades & Single-Fit Collocational Anchors](#6-valency-cascades--single-fit-collocational-anchors)
7. [Distractor Generation Architecture & Universal Double-Key Guard](#7-distractor-generation-architecture--universal-double-key-guard)
8. [Eight Lexico-Syntactic Construction Families](#8-eight-lexico-syntactic-construction-families)

---

## 1. Candidate Selection & Three-Gate Pipeline ("宁缺勿滥")

Candidate words and multi-word expressions extracted from authentic lesson passages (and syllabus wordlists) must pass a strict **Three-Gate Quality-Over-Quantity Pipeline** before entering the final vocabulary pool or assessment blueprint. **Hardcoded Python dictionaries (`curated_examples`, `priority_sense_examples`, `supplemental_vocab_senses`) are permanently prohibited** so dictionary gaps and retrieval bugs are never silently masked.

```
[ Candidate Pool (Syllabus + Passage Mining) ]
                      │
                      ▼
┌───────────────────────────────────────────────────────────┐
│ Gate 1: Sense Locking (Top-K Symbolic + 1-Token LLM WSD)  │
│ - Locks exact contextual Sense in pedlex.db (or WordNet) │
│ - Enforces POS hard gate, syntactic subcat, & domain cues │
└───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────────┐
│ Gate 2: Sense-Isolated Authentic Example Gate             │
│ - Polysemous: MUST draw examples from locked Sense itself │
│ - Monosemous: Locked sense examples + inflection search   │
│ - Missing example? ──► 🗑️ DROP & REPLACE                  │
└───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────────┐
│ Gate 3: Collocational Anchor & Zero Double-Key Screening  │
│ - Verifies single-fit anchor & 0 synonym collision        │
│ - No anchor / synset clash (e.g. crossbreed/hybridize)?   │
│   ──► 🗑️ DROP & REPLACE                                   │
└───────────────────────────────────────────────────────────┘
```

### 1.1 The Five Categories of Excluded / Re-Routed Items
1. **Example-Less Domain Verbs & Run-On Derivatives (🗑️ Drop & Replace)**:
   - Words with a definition in `pedlex.db` or WordNet but **0 authentic dictionary or corpus examples** (e.g., `hybridize`, `crossbreed`, `domesticate`, `adulation`, or unattested run-on derivatives like `carpooling`, `seamlessly`, `comradely`, `weightlessness`).
   - Genuinely example-less items are deterministically dropped (`[DROP_INCOMPLETE_LEXICON]`) and replaced by fully attested candidates from the passage, preventing anchor-less quiz questions and ambiguous synonym collisions.
2. **Transparent Free Word Combinations (🗑️ Drop & Replace)**:
   - Compositional phrases lacking a lexicalized multi-word entry in `pedlex.db` (e.g., `passenger plane`, `new generation`, `family member`, `put into the atmosphere`) are rejected at the expression gate.
3. **Inflected Multi-Word & Collocation-Box Idioms (✅ Recovered via Algorithmic Retrieval)**:
   - Authentic idioms and phrasal verbs whose examples appear in inflected surface forms (`passed away`, `stood for`, `showed up`) or across other entries are retrieved automatically via inflection-aware `search_corpus_examples` and WordNet multi-word lookup.
   - Idioms stored inside `COLLOCATIONS` boxes with inline `gloss` definitions (e.g., `peace of mind` under `peace`) or cross-entry pointers (`pros and cons` under `pro_1` $\rightarrow$ `con_3`) resolve directly to authentic sense definitions without string guessing.
4. **Polysemous Sense-Example Desynchronization (✅ Prevented via Sense Isolation)**:
   - If a polysemous word's contextual sense lacks an example in `pedlex.db` (e.g., `compromise` in the transitive *"harm/weaken"* sense), cross-sense example stealing is strictly forbidden (*"compromise with the bosses"*). The item is dropped rather than paired with a conflicting example.
5. **Textual & CEFR Gates**:
   - The headword must occur in the lesson sentence pool, and difficulty is calibrated deterministically via `cefrpy` + Zipf frequency ceilings.

---

## 2. Contextual Grounding & Syntactic POS Resolution Hierarchy

In educational assessment, word meaning, syntactic function, and pedagogical distractors must originate strictly from the **authentic passage sentence (Quoted Sentence)** and its surrounding discourse context.

### 2.1 The Golden Invariant: POS is Syntactically Bound
1. **The Sentence Dictates the Part of Speech**: A word does not possess an inherent POS in isolation; its POS is deterministically fixed by its syntactic dependency tree in the source sentence.
2. **Strict POS-First Gating**: A dictionary lookup or sense-locking pass must NEVER search across conflicting POS blocks. If a word is used as a noun in the passage (*Chinese Seal*), all verb senses in the dictionary are physically invisible to the disambiguation engine.
3. **Zero Heuristic Patching**: No superficial typographical cues (punctuation adjacency, sentence length) may override dependency parsing.

### 2.2 Syntactic POS Resolution Hierarchy (`resolve_item_pos`)
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

---

## 3. Two-Stage Neuro-Symbolic WSD & Domain Register Penalty

### 3.1 Two-Stage Sense Disambiguation Architecture
- **Stage 1 — Pure Symbolic Candidate Gating (0ms / 0 tokens, ~71% Fast Path)**:
  - Filters `pedlex.db` and WordNet (`oewn:2025+`) using POS hard gates (cross-homograph resolution `word_1`..`word_6`), `<gram>` slot constraints (`[only before noun]`, `[uncountable]`), register/attitude tags (`[spoken]`, `[disapproving]`), conditional-subject ontology gates, and syntactic pattern intersection.
  - Monosemous entries and items with decisive syntactic margins lock instantaneously.
- **Stage 2 — 1-Token Local LLM Arbitration (~29% Polysemous Cases)**:
  - Formats Top-$K$ ($K \le 4$) candidate senses — explicitly surfacing lexicographic `[SIGNPOST]` cues, `<gram>` tags, and syntactic patterns — alongside passage context into a 100-token prompt.
  - Resolved via a 1-token (`A`/`B`/`C`/`D`) local LLM call (~0.3s, 100% benchmark accuracy). Zero free-generation authority.

### 3.2 Universal Domain Register Penalty
Unless the passage explicitly contains verified domain anchor tokens (e.g., *doctor, patient, hospital, disease* for medical), narrow specialized senses receive immediate penalties (-35 to -80 points):
- `[medical]` / `[medicine]` (e.g., *present* = show symptoms)
- `[politics]` / `[parliament]` (e.g., *stand for* = run for parliament)
- `[law]` / `[legal]`
- `[nautical]` / `[military]`
- `[railroad]` / `[transport]` (e.g., *online* = on a regular train route)
- Secondary `[technical]` senses (`s_idx > 0`) lacking scientific/technological triggers.

---

## 4. Five-Source Optimal Example Pre-Selection Architecture

### 4.1 "Select Optimal Example First, Then Slice the Frame"
To eliminate bare frames (`... ____ ...`) and pattern–example desynchronization, the engine pre-selects the **single best pedagogical example** across five clean LDOCE compartments before slicing the slot frame:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Five-Source Pure LDOCE Pool (Strict Sense Isolation)       │
├─────────────────────────────────────────────────────────────────────────┤
│ Tier 1: grammar_boxes         (High-discrimination pedagogical traps)   │
│ Tier 2: Sense.pattern_bundles (gramexa core subcategorization & ex)     │
│ Tier 3: LDOCE Thesaurus       (Concept-aligned full clauses + <gram>)   │
│ Tier 4: Language Activator    (Concept-indexed functional frames)       │
│ Tier 5: Curated Collocations  (Sense-verified collocations; no corpus)  │
└─────────────────────────────────────────────────────────────────────────┘
```
*(Oxford Thesaurus and unedited BNC corpus dumps are permanently excluded for hygiene).*

### 4.2 Scoring Rubric for Winning Example
1. **Finite Clause Completeness**: Full sentences containing finite verbs strongly outrank fragments (`"We took out a loan to buy a new car."` $\gg$ `"a loan of £60,000"`).
2. **Syntactic Slot Alignment**:
   - Attributive adjectives paired against `[not before noun]` distractors receive an attributive pre-noun example (`det/poss + ____ + noun`).
   - Countable singular nouns paired against `[uncountable]` distractors receive an explicit singular indefinite article example (`verb + a/an + noun`).
3. **Pattern Hierarchy**: `gramexa` core valency $\gg$ general collocations $\gg$ narrow coordinate idioms (`colloinexa`).
4. **Length & Hygiene**: Sentences of 6–22 everyday words outrank technical extracts.

---

## 5. Linguistic Principles of Optimal Example Selection across Parts of Speech

To ensure assessment items achieve maximum psychometric validity, cognitive focus, and SLA naturalness, candidate examples are screened against rigorous linguistic principles tailored to each word class.

### 5.1 Nouns (名词精选例句的语言学原理)
1. **Ontological Tiering (1st-Order Entity vs. 2nd/3rd-Order Relational & Shell Nouns)**:
   - **1st-Order Concrete Entities** (*vehicle, telescope, timber*): Examples must feature canonical physical predicates and perceptual settings, avoiding abstract metaphoric uses.
   - **2nd-Order Eventive / Process Nouns** (*attempt, collision, reform*): Examples must contain authentic **Support Verbs (Light Verbs)** (`make an attempt`, `undergo reform`, `survive a collision`) that define the noun's operational syntactic frame.
   - **3rd-Order Relational & Shell Nouns** (*ability, tendency, hypothesis, dilemma*): Examples must explicitly attest their **Post-Nominal Complements** (`ability to adapt`, `tendency to overestimate`, `hypothesis that ...`).
2. **Countability Licensing & Determiner Symmetry**:
   - For countable nouns (*loan, benchmark, precedent*), examples must explicitly attest singular indefinite articles (`take out a loan`, `set a benchmark`) or plural inflection (`set precedents`).
   - **Rejection of Pre-Nominal Compound Modifiers**: Sentences where the target noun is demoted to a pre-noun modifier (*loan application*, *timber industry*) are rejected because they strip the noun of its determiner valency and confuse learners regarding countability.
3. **Finite Sentence Over Fragmentary Headings**:
   - Dictionary noun entries frequently supply verbless title fragments (*"a loan of £60,000"*, *"earthy language"*). The selector enforces finite clausal structure (`ROOT` verb present) to supply subject/predicate context.

### 5.2 Verbs (动词精选例句的语言学原理)
1. **Four-Tier Valency Cascades**:
   - **Class I: Monotransitive Verbs** (*solve, abandon, accelerate*): Examples must provide an explicit, prototypical nominal direct object.
   - **Class II: Ditransitive & Complex Transitive Verbs** (*provide [sb] with [sth]*, *remind [sb] of [sth]*, *elect [sb] [as NP]*): Examples must complete both the patient argument and the secondary prepositional/predicative argument.
   - **Class III: Intransitive with Obligatory Preposition** (*depend on, originate in, adhere to*): Examples must attest the bound prepositional complement.
   - **Class IV: Pure Intransitive & Ergative Verbs** (*hesitate, collapse, vanish*): Examples must provide an authentic agent/experiencer subject and manner/adverbial adjunct.
2. **First-Principles Verb Patient Ontology**:
   - For transitive action and event verbs, the example's direct object (or passive subject) is mapped to one of 6 cognitive subdomains via WordNet Supersenses:
     - `situation/crisis` (*escalate, defuse*): `[situation/crisis, e.g. the war / the conflict]`
     - `problem/issue` (*resolve, tackle*): `[problem/issue, e.g. the crisis / the issue]`
     - `process/activity` (*facilitate, accelerate*): `[process/activity, e.g. the development / the training]`
     - `feeling/state` (*intensify, trigger*): `[feeling/state, e.g. the anger / the tension]`
     - `funds/resources` (*allocate*): `[funds/resources, e.g. the funds / the budget]`
     - `opinion/concern` (*express, voice*): `[opinion/concern, e.g. the concern / the view]`
     - `person/recipient` (*comfort, reassure*): `[person/group, e.g. the passengers / the parents]`
     - `concrete_physical_object` (*fasten, peel, carve*): `[concrete physical object, e.g. the cloth / the lid]`
3. **Explicit Post-Verbal Object Priority (Banning Interrogative Fronting)**:
   - Examples exhibiting interrogative or relative pronoun fronting (*"Who should we invite to the party?"*, *"the house which they abandoned"*) displace the object before the verb, causing the cloze frame to degrade into a meaningless `... we ____`. Examples must place the direct object in canonical post-verbal position (*"They abandoned the project..."*).
4. **Passive Transformation & Ergative Subject Projection**:
   - When a winning example is in passive voice (*"The war was escalated by external powers"*), the frame generator projects it cleanly either to an active frame (`... ____ [situation/crisis, e.g. the war]`) or an authentic passive frame with tense auxiliary (`... [situation/crisis] was/were ____`).
5. **Protection of Ellipsis Collocations & Measure Objects**:
   - Preserves discontinuous collocations with internal ellipsis (*"exceeded ... expectations"*) and quantitative measure objects (*"walked 42 hours"*, *"tons of salt"*).

### 5.3 Adjectives (形容词精选例句的语言学原理)
1. **Three Syntactic Distribution Classes**:
   - **Class I: Attributive-Only Adjectives (`[only before noun]`, e.g., `mere`, `sheer`, `received`, `urban`, `indoor`)**: Examples must position the target as a pre-nominal modifier (`det/poss + ____ + noun`). Predicative copular sentences are strictly barred.
   - **Class II: Predicative-Only Adjectives (`[not before noun]` / Copular Complements, e.g., `accustomed`, `prone`, `aware`, `full`)**: Examples must feature copular matrix clauses with their bound complements (`be accustomed to`, `be aware of`).
   - **Class III: Alternating Adjectives**: Examples are aligned with the passage quote's distribution (attributive or predicative).
2. **For-To Infinitival Guard**:
   - In extraposition frames (*"It is important for children to read"*), the preposition `for` introduces the infinitival subject (`for children to read`), NOT a lexical preposition governed by `important`. The selector strictly forbids extracting `for` as a bound preposition collocation for adjectives.
3. **Cross-POS Homograph & Participle Interception**:
   - Intercepts verbal past participles used as active verbs (*"He exhausted his energy"* when testing adjective `exhausted`) and homograph nouns (*veteran*, *doctoral*).

### 5.4 Adverbs (副词精选例句的语言学原理)
1. **Three Functional Modification Domains**:
   - **Discourse & Stance Connectors** (*similarly, hence, nevertheless, conversely*): Examples must position the adverb in sentence-initial or clause-linking stance (`"Similarly, other studies found..."`).
   - **Degree & Viewpoint Modifiers** (*vaguely, deeply, highly, thoroughly*): Examples must pair the adverb with a scalar adjective or participle (`"looked vaguely familiar"`, `"deeply concerned"`).
   - **Manner & Frequency Modifiers** (*broadly, abruptly, frequently*): Examples must modify a finite lexical action verb (`"smiled broadly"`, `"stopped abruptly"`).
2. **Exact Surface Form Matching**:
   - Example search must match the exact inflected surface form, preventing base adjectives or stems from producing false matches.
3. **Filtering Dangling Participial Fragments**:
   - Eliminates verbless advertising or directory fragments (*"Conveniently located near the airport"*).

### 5.5 Multi-Word Expressions & Idioms (多词短语与习语精选例句的语言学原理)
1. **Co-variant Example Guard**:
   - Prevents search engines from matching unrelated particles or pronoun fragments (e.g. treating `"s"` from `one's` as a matching token, or matching `talk ... out of` when querying `talk of`).
2. **Whole-Idiom Preservation in Cloze Testing**:
   - Idioms containing internal possessive slots (*come to one's aid*, *turn one's attention to*, *a roof over one's head*) are preserved in their entirety inside the blank `____`, rather than fractured into single words.
3. **Possessive Determiner & Tense Synchronization**:
   - The winning example's possessive determiner is normalized to concrete 3rd-person (`her`, `his`, `their`), and the leading verb is inflected to match passage narrative tense across all 4 options simultaneously (`came to her aid`, `got to her feet`, `had to her credit`, `looked to her laurels`).

---

## 6. Valency Cascades & Single-Fit Collocational Anchors

Every cloze question derives its discriminative power from an immutable **Context Anchor** embedded in `cloze_slot_frame`:

| Word Class | Primary Valency Anchor | Secondary Anchor | Discriminative Mechanism |
| :--- | :--- | :--- | :--- |
| **Transitive Verb** | Direct Object Patient (`object`) | Bound Preposition (`prep`) | Semantic selectional restriction / collocations |
| **Intransitive Verb** | Bound Preposition (`prep`) | Manner Adverb (`adv_mod`) | Prepositional subcategorization |
| **Prepositional Adj** | Bound Preposition (`prep`) | Head Copula (`cop`) | Lexical preposition selection |
| **Attributive Adj** | Modified Head Noun (`modified_noun`) | Degree Modifier (`adv_mod`) | Collocational semantic fit |
| **Eventive Noun** | Support / Light Verb (`verb`) | Prepositional Complement (`prep`)| Light-verb collocation frame |
| **Shell Noun** | Clausal Complement (`to_inf` / `that`) | Post-nominal prep (`of`) | Subcategorization complement clash |
| **Function Word** | Clausal vs. Prepositional NP | Discourse polarity | Structural complement contrast |

---

## 7. Distractor Generation Architecture & Universal Double-Key Guard

### 7.1 Five-Tier Zero-Token Distractor Generation
1. **PedLex Topic & Activator Lexicon**: Concept cluster harvesting with cluster throttling.
2. **PedLex Thesaurus Synonyms & Antonyms**: Concept-aligned sister words filtered against synonym collision.
3. **WordNet Coordinate Sisters**: Strict POS and semantic field alignment.
4. **Orthographic / Paronym Tier (`get_orthographic_distractors`)**:
   - High-discrimination look-alikes (`adapt`/`adopt`/`adept`, `complement`/`compliment`, `specimen`/`spectrum`/`spectacle`), guarded by WordNet synset disjointness to ban root-sharing double keys (`historic`/`historical`).
5. **CEFR & Dynamic Corpus Fallback Pool (`_ensure_corpus_fallback_index`)**:
   - Dynamically loads 8,102 core headwords from `pedlex.db` partitioned by CEFR (Tier 1: A1-B1, Tier 2: B2-C1), sampling via CRC32 + golden ratio stride (`0.618`). Zero hardcoded fallback lists.

### 7.2 Universal Post-Assembly Double-Key Guard
Once the final options are assembled and inflected, `has_double_key_collision` checks all distractors against the target in the context of the **final context anchor, anchor type, and sentence quote**:
- Detects bound preposition sharing (`transform` vs. `work` under `into`).
- Detects light-verb collocation sharing (`take [sth] into practice`).
- Detects predicate phrase collisions via `pedlex_phrase_index` (`be through with`).
- Auto-refills colliding distractors from the dynamic fallback pool, guaranteeing **0 double keys across all items**.

---

## 8. Eight Lexico-Syntactic Construction Families

Lexical items projecting structural catenae are classified into **Eight Construction Families** with specialized single-blank strategies:

| Family | Canonical Frame | Blank `____` Strategy | Distractor Contrast Mechanism |
| :--- | :--- | :--- | :--- |
| **Family 1A: Preparatory-Object** | `<V> it <adj> ... to do / that` | 3-word chunk (`make it possible`) | Same-verb prefix, non-infinitival complement (`make it clear`) |
| **Family 1B: Preparatory-Subject** | `It is <adj> (for sb) to do` | Adjective (`hard`, `necessary`) | Animate-subject control adjectives (`reluctant`, `eager`) |
| **Family 2: Correlatives** | `Arm 1 ... Arm 2` | Opening trigger (`not only`, `neither`) | Cross-correlative pairing mismatch (`either`, `rather than`) |
| **Family 3: Discontinuous `V+NP+Prep`**| `<V> [sb] <Prep>` | Governing verb (`remind`, `prevent`) | Prepositional valency clash (`explain to`, `forbid to`) |
| **Family 4: Non-Finite Verb Complements**| `<V> + [doing / to do]` | Matrix verb (`look forward to`) | Non-finite subcategorization inversion (`manage` vs. `avoid`) |
| **Family 5: Shell Nouns** | `the <Noun> + [to do / that]`| Shell noun (`ability`, `fact`) | Complement clash (`habit of`, `possibility of`) |
| **Family 6: Light-Verb Triads** | `<V> + <N> + <Prep>` | Triad (`take advantage of`) | Synonymous triad exclusion (`make use of`) |
| **Family 7: NPIs & Inversion Triggers** | `Fronted Chunk + Aux + S + V` | Trigger chunk (`under no circumstances`)| Non-inverting prepositional phrases (`in addition`) |
| **Family 8: Mandative Subjunctives** | `<Trigger> + that S (should) V`| Matrix trigger (`suggest`, `vital`) | Subordinate bare infinitive / minimal-pair grammar traps |
| **Family 9: Internal Possessive Idioms**| `<V> + <Prep> + one's + <N>` | Whole idiom (`came to her aid`) | Symmetric `one's` idioms + possessive & tense sync |
