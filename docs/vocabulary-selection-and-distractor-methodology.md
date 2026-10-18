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
│ - Enforces POS hard gate, homograph resolution, & syntax  │
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
3. **Inflected Multi-Word & Cross-Entry Corpus Items (✅ Recovered via Algorithmic Retrieval)**:
   - Authentic idioms and phrasal verbs whose examples in `pedlex.db` or WordNet appear in inflected surface forms (e.g., `pass away` $\rightarrow$ *"passed away"*, `stand for` $\rightarrow$ *"stood for"*, `show up` $\rightarrow$ *"showed up"*) or across other entries (`considerably`) are retrieved automatically via inflection-aware `search_corpus_examples` and multi-word WordNet synset lookup.
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
