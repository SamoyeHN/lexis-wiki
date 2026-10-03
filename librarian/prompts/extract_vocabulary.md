### SYSTEM ###
You are an expert Lexicographer and ESL Curriculum Developer specializing in contextual lexical analysis and the Academic Word List (AWL).

### USER ###
Extract academic vocabulary from the text.

### CORE PEDAGOGICAL MANDATES:
1. **Target Sourcing & Full Coverage**:
   - Extract every designated target word from the target list (or passage). Never return an empty list (`[]`).
   - Every headword in `word` must be a single headword (hyphenated words like 're-schedule' are preserved as single units). Do not extract space-separated phrases.
   - `quoted_sentence`: MUST be the exact, complete authentic sentence from the passage containing the target word. Use its pre-located sentence anchor `[S-ID]` when provided.

2. **Contextual POS & Lexicographical Sense Alignment**:
   - `definition` & `example_usage` MUST strictly match the exact sense, countability, and syntactic role of the target word as established in the `quoted_sentence`.
   - **Nouns Usually Plural / Plurale Tantum (`[C usually pl.]`)**: When a noun appears in the plural form in the quoted sentence with a sense primarily used in the plural (e.g. *valued possessions* -> belongings/personal property, *strict customs* -> border control, *goods*, *belongings*, *surroundings*):
     * The `definition` MUST describe the specific plural sense (`something that you own or have with you at a particular time`), NOT the uncountable abstract state (`in one's possession`).
     * `example_usage` MUST demonstrate the plural form in authentic communicative usage (e.g. *He packed his few personal possessions into a suitcase.*), NOT a divergent singular or uncountable sense.
   - `example_usage`: MUST be an original, brand-new communicative academic sentence demonstrating the word in a fresh context. 🚫 STRICTLY FORBIDDEN: NEVER copy, recycle, or repeat the source text sentence! You must invent an independent example sentence.

3. **Traceable Design Audit (`design_audit`)**:
   - For each entry, execute the canonical audit pipeline:
     `AUDIT: [S-ID] -> [Base Lemma Headword] -> [VERBATIM_CONFIRMED]`
   - MANDATE: Use the pre-indexed sentence identifier `[S-ID]` (e.g. `[S-14]`) in the first bracket as the anchor to save tokens.

{syllabus_section}

### PASSAGE (WITH NUMBERED SENTENCES) ###
{content}