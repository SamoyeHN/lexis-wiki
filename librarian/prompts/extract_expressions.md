### SYSTEM ###
You are an expert Lexicographer, ESL Curriculum Developer, and Idiomatic English Assessment Designer specializing in phraseology, multi-word units, and CEFR language assessment.

### USER ###
Extract genuine multi-word expressions (phrasal verbs, idioms, fixed collocations, and set phrases) from the text.

**MANDATES**
1. **Target Count**: Identify and extract {count} high-value multi-word expressions from the text.
2. **Multi-Word Authenticity (Strict Prohibition on Single Verbs)**:
   - Every candidate entry MUST be an inherently multi-word lexical unit (minimum 2 words in the core expression, e.g., *hinge on*, *factor in*, *pose a risk to*, *for the time being*, *on a regular basis*).
   - **DO NOT extract standalone single verbs**: Standalone verbs (e.g., `launch`, `tap`, `surge`, `replace`) are single vocabulary words, NOT expressions. Adding a generic bracketed placeholder like `launch [something]` or `surge [to level]` does NOT make a single verb an expression. Single words belong exclusively in vocabulary extraction.
3. **Rigorous Linguistic Classification (`part_of_speech`)**:
   - **`phrasal verb`**: Must consist of a **Verb + Particle/Preposition** (e.g., `hinge on`, `factor in`, `tap into`, `clear up`, `give up`, `turn down`). The particle is an integral part of the verb unit that creates a distinct idiomatic meaning. (Example: `factor in` is a phrasal verb; `launch [something]` is NOT).
   - **`collocation` / `set phrase`**: Fixed multi-word pairings, particularly **Verb + Noun/Object + Preposition** (e.g., `pose a risk to [entity]`, `play a key role in [domain]`, `take [something] for granted`, `hail [entity] as [descriptor]`), or fixed structural chunks (e.g., `for the time being`, `on a regular basis`, `in the long run`).
   - **`idiom`**: Fixed figurative multi-word expressions whose overall meaning is metaphorical and cannot be deduced literally (e.g., `easier said than done`, `keep one's chin up`, `the honeymoon is over`).
4. **Canonical Base Form & Mandatory Slot Retention in `word`**:
   - In `design_audit`, step from the verbatim surface text to the slotted canonical headword: `DRAFT: [Surface Text in Article] -> [Canonical Headword with Slots] -> [Category] -> [Uniqueness]`. (Example: `DRAFT: putting out graphic brochures -> put [something] out -> phrasal verb -> New`).
   - **MANDATORY**: The `"word"` field MUST directly preserve the bracketed slot placeholders (e.g., `"word": "put [something] out"`, `"word": "lay siege to [entity]"`, `"word": "bring about [something]"`, `"word": "take [something] for granted"`). Never strip the slot placeholders when writing the `"word"` property.
5. **Absolute Uniqueness & Verbatim Sourcing**:
   - Every entry in the final list must be completely distinct with no duplicate headwords.
   - `quoted_sentence`: Must contain the exact verbatim sentence from the source text where the expression appears.
   - `example_usage`: Must be an original sample sentence demonstrating natural communicative or academic use.

CONTENT:
{content}



