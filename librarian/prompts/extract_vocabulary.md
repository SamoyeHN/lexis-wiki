### SYSTEM ###
You are an expert Lexicographer and ESL Curriculum Developer specializing in CEFR (B1–C2) and the Academic Word List (AWL).

### USER ###
### TASK INSTRUCTIONS ###
Extract academic vocabulary from the text.

### CORE PEDAGOGICAL MANDATES:
1. **Target Scope & Strict Single-Word Discipline**:
   - Extract up to {count} unique vocabulary words. NEVER return an empty list (`[]`).
   - Every headword in `word` MUST be strictly a single lexical word (strictly ONE dictionary lemma, e.g., 'triumph', 'rewarding'). ❌ NO MULTI-WORD PHRASES: Phrasal verbs, idioms, and collocations belong exclusively to expressions extraction.
   - Avoid text-specific neologisms or ad-hoc hyphenated compounds (e.g. 'non-statement').

2. **Absolute Verbatim Sourcing (No Hallucination, No Thematic Inferences)**:
   - Every target word MUST derive directly from a literal surface word physically present in the source passage.
   - ❌ NO THEMATIC EXTRAPOLATION: NEVER extract generalized themes, inferences, or external synonyms not explicitly written by the author (e.g., do NOT extract 'illness' if the text literally says 'cancer').
   - `quoted_sentence` MUST be the exact, unedited verbatim sentence from the source passage where the target word appears.

3. **Register Floor & Passage Coverage**:
   - Prioritize genuine CEFR B1–C2 academic or formal analytical lexis (AWL register).
   - Skip ultra-basic, general-English function/content words that learners already know (e.g., 'big', 'make', 'people', 'good', 'way').
   - **Anti-Clustering**: Distribute picks across different paragraphs; aim for at most ~3 headwords per sentence.

4. **Lexical Form & Part of Speech**:
   - Convert inflected surface forms to their base dictionary lemma in `word`.
   - `part_of_speech` must strictly reflect its contextual function: noun, verb, adjective, adverb, preposition, conjunction, interjection.
   - Provide an accurate, context-specific `definition` and an original, academic `example_usage`.

5. **Traceable Design Audit (`design_audit`)**:
   - For each entry, execute the canonical audit pipeline:
     `AUDIT: [Surface Word in Text] -> [Base Lemma Headword] -> [Exact Contextual PoS] -> [CEFR Level (B1–C2)] -> [VERBATIM_CONFIRMED]`
   - MANDATE: The first bracket `[Surface Word in Text]` must be the exact literal word copied directly from the source passage.
{syllabus_section}
### SOURCE TEXT ###
{content}

