# Pedagogical Vocabulary Selection, Collocation & Distractor Methodology

> **Purpose**: Detailed technical specification of the multi-tiered linguistic algorithms used by Lexis Wiki for candidate headword extraction, POS resolution, collocational valency cascading, and psychometric distractor generation.

---

## 1. Candidate Selection & Filtering Gates

Candidate words extracted from authentic lesson passages must satisfy a dual-gate architecture before entering the syllabus vocabulary pool:

1. **Lexicon Gate (LDOCE 6th Edition Grounding)**:
   - The headword must have an independent, fully-formed entry in `ldoce6_essential.db` with valid definitions and senses.
   - Suffix-derived words or nominalizations that lack standalone definitions in the dictionary are excluded rather than force-converted.
2. **Textual Gate (Authentic Textual Grounding)**:
   - The headword (or its exact inflected surface form) must physically occur in the lesson sentence pool.
   - Abstract headwords deduced by morphological clustering (e.g. *entrepreneur* when the passage only contains *entrepreneurship*) are barred unless physically attested in the source text.
3. **CEFR & Zipf Ceilings**:
   - High-utility headwords are indexed against Oxford 3000/5000 and CEFR standards (`cefrpy`).

---

## 2. Unified POS Resolution Protocol (`resolve_item_pos`)

To prevent discrepancies between manual teacher edits in markdown notes and the automated quiz generator, part-of-speech resolution is standardized through [`LinguisticEngine.resolve_item_pos`](file:///E:/teacher-wiki/librarian/linguistics.py):

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
                     [ LDOCE Lexical POS Attestation ]
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

Distractors are generated deterministically at zero token cost (<1ms) via a four-tier pipeline:

1. **LDOCE Language Activator & Topic Lexicon**:
   - Harvests semantic neighbors and concept cluster members.
   - Applies **Cluster Throttling** (maximum 1 candidate per cluster/phrase) and word-family deduplication (`are_same_word_family`) to avoid near-identical variants (e.g. *businessman* vs *businessperson*).
2. **LDOCE Thesaurus Synonyms & Antonyms**:
   - Secondary source for taxonomically related content words.
3. **WordNet Taxonomic Filtering**:
   - Bipolar cluster harvesting and coordinate sisters within the same POS and semantic field.
4. **Physical Quality & Difficulty Gates**:
   - **CEFR Difficulty Ceiling**: Intercepts words exceeding the text CEFR level using Zipf frequency ratings.
   - **Morphological Symmetry**: Parallel inflection applied across all candidates (e.g., all past participles `VBN`, all plurals `NNS`).
   - **Bucketed Collision Check**: Distractors are tested against the target's bound collocations to ensure zero unintended valid answers.
