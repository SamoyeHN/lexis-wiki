# Multi-Word Units (MWU) Slicing & Distractor Architecture

> **Purpose**: Definitive technical specification and applied lexicographic taxonomy for assessing Multi-Word Units (MWUs) in Lexis Wiki. Establishes deterministic cloze slot slicing frames, inflection-preserving invariants, and psychometrically audited zero-collision distractor pools grounded in `pedlex.db` (`ldoce`, `ldoce_phrase_index`, and `oxford_thesaurus`).

---

## 📋 Table of Contents
1. [Theoretical Foundations & The Anti-Oscillation Mandate](#1-theoretical-foundations--the-anti-oscillation-mandate)
2. [The Nine Canonical MWU Assessment Families](#2-the-nine-canonical-mwu-assessment-families)
   - [Family 1: Phrasal Verbs (动词短语)](#family-1-phrasal-verbs-动词短语)
   - [Family 2: Compound Nouns (复合名词)](#family-2-compound-nouns-复合名词)
   - [Family 3: Nominal Genitive Idioms (名人格成语 N1 + of + N2)](#family-3-nominal-genitive-idioms-名人格成语-n1--of--n2)
   - [Family 4: Irreversible Binomials (并列二元习语 X and Y)](#family-4-irreversible-binomials-并列二元习语-x-and-y)
   - [Family 5: Exclamatory & Conversational Formulas (感叹与交际惯用语)](#family-5-exclamatory--conversational-formulas-感叹与交际惯用语)
   - [Family 6: Composite Verbal-Prepositional Frames (复合动词-介词成语框)](#family-6-composite-verbal-prepositional-frames-复合动词-介词成语框)
   - [Family 7: Complex Prepositions & Conjunctions (三段式复合介词)](#family-7-complex-prepositions--conjunctions-三段式复合介词)
   - [Family 8: Preposition-Stranding Bare Verbal Idioms (零冠动名短语)](#family-8-preposition-stranding-bare-verbal-idioms-零冠动名短语)
   - [Family 9: Prepositional Zero-Article Adverbials (零冠介宾副词短语)](#family-9-prepositional-zero-article-adverbials-零冠介宾副词短语)
3. [Neuro-Symbolic Distractor Synthesis: PedLex & Oxford Integration](#3-neuro-symbolic-distractor-synthesis-pedlex--oxford-integration)
   - [3.1 Structural Slot Extraction Pipeline](#31-structural-slot-extraction-pipeline)
   - [3.2 PedLex Database Schema Mapping](#32-pedlex-database-schema-mapping)
   - [3.3 Oxford Thesaurus Semantic Near-Synonym Harvesting](#33-oxford-thesaurus-semantic-near-synonym-harvesting)
   - [3.4 Collocational Mutual Exclusion Screening (Double-Key Filter)](#34-collocational-mutual-exclusion-screening-double-key-filter)
4. [Master Slicing & Distractor Decision Matrix](#4-master-slicing--distractor-decision-matrix)
5. [Anti-Pattern Gallery & Prohibited Slicing Hacks](#5-anti-pattern-gallery--prohibited-slicing-hacks)

---

## 1. Theoretical Foundations & The Anti-Oscillation Mandate

In Second Language Acquisition (SLA), assessing multi-word expressions requires balancing two psychometric constraints:
1. **Valid Germane Cognitive Load**: Learners must process the contextual collocational anchor (e.g. `____ in touch with`, `in ____ of`, `____ doubt on`) rather than matching whole visual chunks through rote memory.
2. **Grammatical & Lexicographical Naturalness**: Sliced items must never present ungrammatical fragments (e.g. `arrival plane`, `station down`) or trivial preposition giveaway drills (`turn ____` $\rightarrow$ `[off, for, by, with]`).

### The Anti-Oscillation Mandate
Historical assessment generators suffered from violent oscillation between two flawed extremes:
- **Extreme A (Blind Whole-Chunk Masking)**: Masking `pros and cons` as an entire unit prompted systems to mask *all* phrases whole. This led to length disparity giveaways, syntactic mismatch with separable pronouns (e.g. `shutting it down` vs bare verb distractors `bring down`), and bizarre, semantically disconnected distractors (e.g. `peace of mind` paired with `right of abode` or `acres of room`).
- **Extreme B (Naive Fallback Slicing)**: Mechanically masking word 0 of any two-word phrase (e.g. masking `passenger` in `passenger plane`, yielding `[arrival, comer, astronaut, passenger]` for `____ plane`; or masking `get` in `get ahead`, yielding `[end, leave, give, get]` for `____ ahead`).

```
[ Multi-Word Unit Input (MWU) ]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│   Neuro-Symbolic Lexico-Grammatical Classifier Gate    │
│   (spaCy POS tags + dependency trees + pedlex.db)       │
└────────────────────────────────────────────────────────┘
                 │
  ┌──────────────┼──────────────┬──────────────┐
  ▼              ▼              ▼              ▼
[Phrasal Verb] [Compound Noun] [N of N Idiom] [Complex Prep / Light Verb]
  │              │              │              │
  ▼              ▼              ▼              ▼
Whole Chunk    Attributive    Medial Noun     Medial Noun or Light Verb
Symmetric      Classifier     Slot Cloze      Deterministic Slot Cloze
Particles      Domain Cloze   (Abstract N)    (PedLex Frame Intersection)
```

---

## 2. The Nine Canonical MWU Assessment Families

### Family 1: Phrasal Verbs (动词短语)
- **Canonical Forms**: `V + Particle` (`get ahead`, `shut down`, `turn off`, `bring about`) or `V + Particle + Prep` (`look forward to`, `catch up with`).
- **Slicing Invariant**: **NEVER slice into single-word fragments.** Masking the verb creates hundreds of irrelevant verb combinations (`end ahead`, `station down`); masking the particle degrades the test into trivial function-word trivia (`shut for / by / with`).
- **Standard Assessment Mode**: Whole phrasal verb with **Symmetric Particle or Verb Contrasts**:
  - Target: `get ahead` (to be successful in career/life)
    - ❌ *Prohibited distractors*: `[get behind, get burned, get burnt]` (mixed voice, asymmetrical participles).
    - ✅ *Standard distractor pool*: `[get ahead, get along, get by, get away]` (Symmetric core verb `get` + authentic A2/B1/B2 phrasal particles).
  - Target: `shut down` (to stop operating/close down)
    - ✅ *Standard distractor pool*: `[shut down, shut off, shut out, shut up]` or `[shut down, break down, slow down, turn down]`.
- **Pronominal Invariant**: If the stem contains a separated pronoun object (`shutting it down`), all four options **MUST** be strictly grammatically mirrored:
  `[shutting it down, turning it off, cutting it off, putting it away]`.

---

### Family 2: Compound Nouns (复合名词)
- **Canonical Forms**: `Noun + Noun` (`passenger plane`, `traffic jam`, `sports car`, `alarm clock`) or `Adj + Noun` (`credit card`, `solar energy`).
- **Slicing Invariant**: Compound nouns function as unified conceptual units. Plain nouns cannot freely serve as noun adjuncts (e.g. `arrival` cannot modify `plane`).
- **Standard Assessment Mode**:
  - **Mode A (Attributive Classifier Slot Cloze)**:
    - Stem: `Large ____ planes fly over our heads, carrying hundreds of travelers.`
    - Blank Frame: `... large ____ planes`
    - Target: `passenger`
    - Distractors: Derived from the **same functional vehicle domain**: `[passenger, cargo, fighter, charter]`.
    - Pedagogical Value: All four form valid English compound nouns (`passenger plane`, `cargo plane`, `fighter plane`, `charter plane`); context clue `travelers` establishes single-fit validity.
  - **Mode B (Whole-Entity Semantic Field Contrast)**:
    - Stem: `Large ____ carry travelers across oceans within a single day.`
    - Options: `[passenger planes, freight trains, cargo ships, public buses]`. Single-fit is enforced by verbal anchor `fly over / across oceans`.

---

### Family 3: Nominal Genitive Idioms (名人格成语 N1 + of + N2)
- **Canonical Forms**: `[Abstract Noun 1] + of + [Noun 2]` (`peace of mind`, `point of view`, `state of the art`, `presence of mind`, `frame of mind`).
- **Slicing Invariant**: Anchor on the trailing genitive phrase (`of mind`, `of view`), slice the **head abstract noun (N1)**.
- **Standard Assessment Mode**:
  - Stem: `Winning the legal battle finally brought her ____ of mind after months of acute stress.`
  - Blank Frame: `... ____ of mind`
  - Target: `peace`
  - Distractors: **Oxford Thesaurus Near-Synonyms with Zero-Collocation Proof**:
    `[peace, ease, calm, rest]`.
  - Pedagogical Value: While `ease`, `calm`, and `rest` are core synonyms of `peace`, English idiomatic usage strictly rejects `*calm of mind` and `*rest of mind`. Demonstrates peak psychometric discrimination without bizarre distractors like `acres of room` or `right of abode`.
  - **Alternative Mode (Whole Idiom Tail-Sharing)**:
    `[peace of mind, presence of mind, frame of mind, state of mind]`.

---

### Family 4: Irreversible Binomials (并列二元习语 X and Y)
- **Canonical Forms**: `pros and cons`, `ups and downs`, `bits and pieces`, `back and forth`, `give and take`, `ins and outs`.
- **Slicing Invariant**: Fixed order cannot be reversed (`*cons and pros`).
- **Standard Assessment Mode**:
  - **Mode A (Target Slot 1 Cloze)**:
    - Stem: `Before making the investment, the board carefully weighed the ____ and cons of the expansion.`
    - Blank Frame: `... the ____ and cons`
    - Target: `pros`
    - Distractors: Oxford synonyms of advantage/merit: `[pros, merits, gains, perks]`.
  - **Mode B (Whole Binomial Coordinate Peer Contrast)**:
    - Options: `[pros and cons, ins and outs, ups and downs, dos and don'ts]`.

---

### Family 5: Exclamatory & Conversational Formulas (感叹与交际惯用语)
- **Canonical Forms**: `hats off to`, `good riddance to`, `thumbs up to`, `credit where it's due`.
- **Standard Assessment Mode**:
  - **Mode A (Head Metaphor Noun Slicing)**:
    - Stem: `I take my ____ off to the medical staff for their heroic dedication.`
    - Blank Frame: `... take my ____ off to`
    - Target: `hats` (or `hat`)
    - Distractors: Garment/clothing peers: `[hats, caps, coats, gloves]`.
  - **Mode B (Whole Conversational Pragmatic Peer Match)**:
    - Options: `[hats off to, good luck to, thumbs up to, no thanks to]`.

---

### Family 6: Composite Verbal-Prepositional Frames (复合动词-介词成语框)
- **Canonical Forms**: `[V] + in touch + with`, `[V] + advantage + of`, `[V] + sense + of`, `[V] + sight + of`.
- **Slicing Invariant**: Three-part catena: Matrix Verb + Prepositional Idiom + Bound Preposition.
- **Standard Assessment Mode**:
  - **Mode A (Medial Idiom Noun Slot — Recommended)**:
    - Stem: `Even after moving to different countries, they made a conscious effort to stay in ____ with each other.`
    - Blank Frame: `... stay in ____ with`
    - Target: `touch`
    - Distractors: Prepositional frame peers fitting `in ____ with`:
      `[touch, line, step, tune]`.
      - `in line with` (conforming to)
      - `in step with` (keeping pace with)
      - `in tune with` (attuned to)
      - `in touch with` (maintaining communication)
    - Single-fit guarantee: In the context of interpersonal communication, only `touch` is semantically valid. Double-key hazard `contact` (`in contact with`) is strictly banned.
  - **Mode B (Aspectual Light-Verb Slicing)**:
    - Stem: `Before parting ways, we promised to ____ in touch with each other.`
    - Blank Frame: `... ____ in touch with`
    - Target: `keep`
    - Distractors: High-frequency L1 negative-transfer light verbs: `[keep, hold, take, catch]`.

---

### Family 7: Complex Prepositions & Conjunctions (三段式复合介词)
- **Canonical Forms**: `in terms of`, `on behalf of`, `in light of`, `at the expense of`, `in accordance with`, `in spite of`, `by means of`.
- **Slicing Invariant**: Slicing prepositions (`in terms ____` $\rightarrow$ `of`) reduces difficulty to trivial grammar. Slicing the entire phrase creates clunky options.
- **Standard Assessment Mode (Medial Noun Slot Cloze)**:
  - Stem: `The candidate was evaluated strictly in ____ of her professional qualifications.`
  - Blank Frame: `... in ____ of`
  - Target: `terms`
  - Distractors: Authentic complex preposition nouns sharing `in ____ of`:
    `[terms, light, view, spite]`.
    - `in terms of` (with regard to)
    - `in light of` (taking into consideration)
    - `in view of` (because of / considering)
    - `in spite of` (despite)
  - Target: `on behalf of` $\rightarrow$ Blank Frame: `... on ____ of` $\rightarrow$ Distractors: `[behalf, account, basis, charge]`.

---

### Family 8: Preposition-Stranding Bare Verbal Idioms (零冠动名短语)
- **Canonical Forms**: `give rise to`, `catch sight of`, `shed light on`, `cast doubt on`, `set fire to`, `lose track of`, `pay attention to`.
- **Slicing Invariant**: The nominal element is strictly bare (zero article). Slicing the noun isolates the core metaphorical collocator.
- **Standard Assessment Mode**:
  - **Mode A (Bare Noun Slot Slicing)**:
    - Stem: `Poor hygiene and overcrowding frequently give ____ to outbreaks of infectious diseases.`
    - Blank Frame: `... give ____ to`
    - Target: `rise`
    - Distractors: Authentic `give ____ to` idiom nouns:
      `[rise, birth, way, place]`.
      - `give rise to` (cause / trigger)
      - `give birth to` (produce / originate)
      - `give way to` (be replaced by)
      - `give place to` (yield to)
  - **Mode B (Governing Verb Slicing)**:
    - Stem: `The recent archaeological discovery has ____ new light on early human migration.`
    - Blank Frame: `... ____ new light on`
    - Target: `shed`
    - Distractors: Light-verb contrast candidates: `[shed, cast, throw, beam]`. Tests dictionary-governed idiom restriction (`shed light on` vs `cast doubt on`).

---

### Family 9: Prepositional Zero-Article Adverbials (零冠介宾副词短语)
- **Canonical Forms**: `by accident`, `on purpose`, `at stake`, `in vain`, `under arrest`, `at length`, `by design`.
- **Slicing Invariant**: Slicing the noun tests whether the learner knows which abstract noun pairs with which specific governing preposition without an article.
- **Standard Assessment Mode**:
  - Stem: `I did not intend to delete the document; it was done purely by ____.`
  - Blank Frame: `... by ____`
  - Target: `accident`
  - Distractors: Oxford semantic near-synonyms rigorously screened to have **ZERO collocational validity with `by`**:
    `[accident, incident, event, happening]`.
    - `by accident` is an established adverbial idiom.
    - `*by incident`, `*by event`, and `*by happening` are unidiomatic in English.
    - Eliminates synonym double-key traps (e.g. banning `mistake` because `by mistake` would create an invalid double key).

---

## 3. Neuro-Symbolic Distractor Synthesis: PedLex & Oxford Integration

The synthesis of cloze slots and distractors is grounded deterministically in `librarian/data/pedlex.db` across three core tables:
- `ldoce` (48,960 entries with pattern bundles and collocations)
- `ldoce_phrase_index` (189,410 phrase records across tiers: `phrasal_verb`, `phrase_row`, `sense_unit`, `crossref_phrase`)
- `oxford_thesaurus` (16,234 sense-disambiguated synonym and antonym records)

```
[ Target Multi-Word Expression (MWU) ]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ 1. Structural Pattern Slicer (RegEx + spaCy Subcat)    │
│    Extracts: Target Slot, Blank Frame, Headword Anchor │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ 2. Symbolic Candidate Mining in pedlex.db               │
│    - Query ldoce_phrase_index for matching frame peers │
│    - Query oxford_thesaurus for semantic near-synonyms │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ 3. Collocational Double-Key & Validity Gate            │
│    - Frame Collision Check: SELECT phrase_key          │
│    - POS & Morphological Inflection Lock               │
│    - Rank by CEFR Proximity & Zipf Frequency           │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
[ Final High-Discrimination Quad-Option Array ]
```

### 3.1 Structural Slot Extraction Pipeline
For any given multi-word string and context sentence, the extraction pipeline executes pattern classification:
```python
def classify_mwu_structure(phrase: str, doc: spacy.tokens.Doc) -> str:
    # 1. Complex Preposition: Prep + Noun + Prep
    if re.match(r"^(in|on|at|by|with|under)\s+[a-z]+\s+(of|with|to|for)$", phrase):
        return "complex_preposition"
    # 2. Phrasal Verb: Verb + Particle
    if phrase_index.get(phrase) == "phrasal_verb":
        return "phrasal_verb"
    # 3. Nominal Genitive: N1 + of + N2
    if re.match(r"^[a-z]+\s+of\s+[a-z]+$", phrase):
        return "nominal_genitive_idiom"
    # 4. Irreversible Binomial: X and Y
    if re.match(r"^[a-z]+\s+and\s+[a-z]+$", phrase):
        return "binomial"
    # 5. Composite Frame: V + Prep + N + Prep
    if re.match(r"^[a-z]+\s+(in|out of|into)\s+[a-z]+\s+(with|of|to)$", phrase):
        return "composite_verbal_frame"
    # 6. Compound Noun: N + N
    if is_compound_noun(phrase):
        return "compound_noun"
    return "generic_collocation"
```

### 3.2 PedLex Database Schema Mapping
- **Complex Prepositions (Family 7)**:
  ```sql
  SELECT phrase_key FROM ldoce_phrase_index 
  WHERE phrase_key LIKE 'in % of' AND tier IN ('phrase_row', 'sense_unit');
  -- Returns: 'in terms of', 'in light of', 'in view of', 'in spite of', 'in place of', 'in honor of'
  ```
- **Light-Verb Bare Nouns (Family 8)**:
  ```sql
  SELECT phrase_key FROM ldoce_phrase_index 
  WHERE host_word = 'give' AND phrase_key LIKE 'give % to';
  -- Returns: 'give rise to', 'give birth to', 'give way to', 'give place to'
  ```
- **Phrasal Verbs (Family 1)**:
  ```sql
  SELECT phrase_key FROM ldoce_phrase_index 
  WHERE host_word = 'get' AND tier = 'phrasal_verb';
  -- Returns: 'get ahead', 'get along', 'get by', 'get away', 'get over', 'get through'
  ```

### 3.3 Oxford Thesaurus Semantic Near-Synonym Harvesting
When testing Family 3 (`peace of mind`) or Family 9 (`by accident`), candidate distractors are harvested from `oxford_thesaurus`:
```python
def harvest_oxford_synonyms(word: str, target_pos: str) -> List[str]:
    # Query oxford_thesaurus for core and extended synonyms
    cursor.execute("SELECT data_json FROM oxford_thesaurus WHERE word = ?", (word,))
    row = cursor.fetchone()
    if not row:
        return []
    entries = json.loads(row[0])
    candidates = []
    for sense in entries:
        if sense.get("pos") == target_pos:
            candidates.extend(sense.get("core_synonyms", []))
            candidates.extend(sense.get("synonyms", []))
    return list(dict.fromkeys(candidates))
```

### 3.4 Collocational Mutual Exclusion Screening (Double-Key Filter)
Before accepting an Oxford synonym as a distractor, it is cross-checked against `ldoce_phrase_index`:
```python
def is_valid_negative_collocator(candidate_word: str, stem_frame: str) -> bool:
    """
    Verifies that the candidate noun does NOT form an authentic idiom with the frame.
    Example: For frame 'by ____', candidate 'mistake' -> 'by mistake' EXISTS -> REJECT.
             Candidate 'incident' -> 'by incident' DOES NOT EXIST -> ACCEPT.
    """
    hypothetical_phrase = stem_frame.replace("____", candidate_word).strip()
    cursor.execute(
        "SELECT 1 FROM ldoce_phrase_index WHERE phrase_key = ?", 
        (hypothetical_phrase,)
    )
    return cursor.fetchone() is None
```

---

## 4. Master Slicing & Distractor Decision Matrix

| Family | Canonical Frame | Tested Slot | Distractor Source | Validation Invariant |
| :--- | :--- | :--- | :--- | :--- |
| **1. Phrasal Verbs** | `V + Particle` (`get ahead`, `shut down`) | **Whole Chunk** | `ldoce_phrase_index` (same base verb, contrasting particles) | Pronoun & inflection synchronization (`shutting it down`) |
| **2. Compound Nouns** | `Noun + Noun` (`passenger plane`) | **Classifier Noun** (`passenger`) | Domain-specific vehicle/tool modifiers (`cargo, fighter, charter`) | Must be authentic noun adjuncts in English |
| **3. Nominal Genitive Idioms** | `N1 + of + N2` (`peace of mind`) | **Head Noun** (`peace`) | `oxford_thesaurus` abstract synonyms | Block candidate if `N + of + N2` exists in PedLex |
| **4. Irreversible Binomials** | `X and Y` (`pros and cons`) | **First Arm** (`pros`) or **Whole** | Oxford synonyms of X, or coordinate binomial pool | Preserve fixed coordinate order |
| **5. Exclamatory Formulas** | Formula (`hats off to`) | **Metaphor Noun** (`hats`) or **Whole** | Concrete apparel nouns, or pragmatic formulas | Symmetric register & speech-act function |
| **6. Composite Verbal Frames** | `V + in N + with` (`keep in touch with`) | **Medial Noun** (`touch`) | Peer prepositional idioms (`line, step, tune`) | Strictly ban synonymous double keys (`contact`) |
| **7. Complex Prepositions** | `Prep + N + Prep` (`in terms of`) | **Medial Noun** (`terms`) | `ldoce_phrase_index` sharing `Prep ... Prep` frame | Valid prepositional nouns only |
| **8. Bare Verbal Idioms** | `V + Bare N + Prep` (`give rise to`) | **Bare Noun** (`rise`) | `ldoce_phrase_index` sharing `V ... Prep` frame | Zero article invariant (`*give a rise to`) |
| **9. Prepositional Adverbials** | `Prep + Bare N` (`by accident`) | **Adverbial Noun** (`accident`) | `oxford_thesaurus` near-synonyms | Verify zero collocation with governing preposition |

---

## 5. Anti-Pattern Gallery & Prohibited Slicing Hacks

> [!CAUTION]
> The following slicing patterns violate applied SLA and psychometric testing standards and are permanently forbidden in codebase generation:

1. **The Phrasal Particle Stripping Fallacy**:
   - ❌ Slicing `shut down` $\rightarrow$ `Please remember to do the ____ turning` with options `[for, by, off, with]`.
   - *Reason*: Tests non-existent prepositions and fragments authentic lexical chunks.
2. **The Naive Word-0 Fallback Slicing Fallacy**:
   - ❌ Slicing `passenger plane` $\rightarrow$ `... the ____ plane` with options `[arrival, comer, astronaut, passenger]`.
   - *Reason*: `arrival` and `comer` cannot grammatically function as noun adjuncts.
3. **The Mixed Pronoun/Bare Verb Asymmetry**:
   - ❌ Options: `[shutting it down, bringing down, standing down, putting away]`.
   - *Reason*: Pronominal insertion gives away the target option without testing comprehension.
4. **The Bizarre Structurally-Blind Retrieval Fallacy**:
   - ❌ Target: `peace of mind` $\rightarrow$ Distractors: `[right of abode, acres of room, act of god]`.
   - *Reason*: Blind structural matching (`_ of _`) creates absurd, semantically orthogonal distractors.
5. **The Complex Preposition Function-Word Fallacy**:
   - ❌ Slicing `in terms of` $\rightarrow$ `... in terms ____` with options `[of, for, to, with]`.
   - *Reason*: Degrades higher-order vocabulary assessment to trivial functional preposition guessing.
