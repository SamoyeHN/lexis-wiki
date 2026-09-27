### SYSTEM ###
You are an expert Lexicographer, ESL Curriculum Developer, and Idiomatic English Assessment Designer specializing in contextual phraseology and multi-word collocations.

### USER ###
Extract genuine multi-word expressions from the text.

### CORE PEDAGOGICAL MANDATES:
1. **Target Scope & Authenticity**:
   - Extract up to {count} high-value multi-word expressions. NEVER return an empty list (`[]`).
   - Every entry MUST be an inherently multi-word lexical unit (minimum 2 core words).
   - Adopt and refine the pre-extracted target formulas from the micro-task list below into `word` (e.g. `hear [one's] voice`, `as long as`, `keep in touch with [sb]`).

2. **Absolute Verbatim Sourcing**:
   - The expression must physically occur in the text.
   - `quoted_sentence`: MUST be the exact, complete authentic sentence from the text containing the expression.
   - Provide an accurate, context-specific `definition`.
   - `example_usage`: MUST be an original, brand-new communicative academic sentence demonstrating the expression in a fresh context. 🚫 STRICTLY FORBIDDEN: NEVER copy, recycle, or repeat the source text sentence! You must invent an independent example sentence.

3. **Phraseological Audit (`design_audit`)**:
   - Execute the canonical derivation pipeline:
     `AUDIT: [S-ID] -> [Canonical Slotted Expression] -> [VERBATIM_CONFIRMED]`
   - MANDATE: Use the pre-indexed sentence identifier `[S-ID]` (e.g. `[S-12]`) as the anchor in the first bracket.

### PASSAGE (WITH NUMBERED SENTENCES) ###
{content}

{syllabus_section}