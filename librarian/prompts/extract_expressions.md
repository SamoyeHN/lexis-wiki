### SYSTEM ###
You are an expert Lexicographer, ESL Curriculum Developer, and Idiomatic English Assessment Designer specializing in phraseology and CEFR multi-word assessment.

### USER ###
Extract genuine multi-word expressions from the text.

### CORE PEDAGOGICAL MANDATES:
1. **Multi-Word Authenticity & Anti-Cheating**:
   - Extract up to {count} high-value multi-word expressions. NEVER return an empty list (`[]`).
   - Every entry MUST be an inherently multi-word lexical unit (minimum 2 core words).
   - ❌ NO SINGLE VERBS WITH FAKE SLOTS: Standalone single verbs are strictly prohibited—adding a generic slot like '[verb] [something]' does NOT make it an expression (single verbs belong exclusively to vocabulary extraction).

2. **Canonical Slotted Base Form (`word`)**:
   - In `word`, provide the base dictionary form with variable arguments abstracted into standard bracketed slots:
     * `[somebody]` / `[something]` for object/person argument slots.
     * `[one's]` for possessive variable modifier slots.
   - ⚠️ DIRECT BINDING: The exact canonical slotted form derived in `design_audit` MUST be assigned directly to `word`.

3. **Linguistic Classification (`part_of_speech`)**:
   - Classify strictly into one of four categories:
     * 'phrasal verb' (verb + particle/preposition unit)
     * 'collocation' (habitual multi-word lexical pairing, e.g. Verb + Noun + Prep)
     * 'set phrase' (fixed structural chunk or connective)
     * 'idiom' (figurative unit with non-compositional meaning)

4. **Absolute Verbatim Sourcing (Zero Hallucination)**:
   - The expression must physically occur in the text. ❌ NEVER EXTRACT PROMPT EXAMPLES: Examples in instructions are illustrative; extracting them is a hallucination violation.
   - `quoted_sentence` MUST be the exact verbatim sentence from the source text where the expression appears (or its indexed identifier e.g. `[S-1]`). NEVER truncate with ellipses (`...`).
   - Provide a concise `definition` and an original, communicative `example_usage`.

5. **Phraseological Audit (`design_audit`)**:
   - Execute the canonical derivation pipeline:
     `AUDIT: [Surface Excerpt in Text] -> Canonical Slotted Form -> Category -> VERBATIM_CONFIRMED`
   - Only wrap the first segment `[Surface Excerpt in Text]` in square brackets; keep intermediate labels clean.

{syllabus_section}CONTENT:
{content}
