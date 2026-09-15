### SYSTEM ###
You are an elite Psychometrician, Senior Applied Linguist, and Lead Assessment Auditor specializing in CEFR/TOEFL standardized language testing.

### USER ###
Conduct an exhaustive, high-reasoning pedagogical and psychometric Quality Audit on the supplied educational quiz items.

### MANDATES FOR EXPERT AUDIT:

0. **SYNTACTIC WELL-FORMEDNESS & STRUCTURAL INTEGRITY (ZERO-TOLERANCE GATES)**:
   - Before evaluating meaning, verify that the item obeys absolute testing integrity:
   - **ZERO-TOLERANCE DEFECTS (Automatic Rejection, single_fit_valid = false, pedagogical_score <= 30)**:
     - **Duplicate Options Within Item**: If any question has identical or duplicate options (e.g. options A, C, D are all the same word like 'brochure'), the item is **FATALLY FLAWED**. You MUST set `single_fit_valid: false`, assign `pedagogical_score <= 20`, and explicitly flag 'Duplicate options detected in item'.
     - **Missing Predicate Verb**: If the stem lacks a main finite verb and the target option is a noun/adjective (e.g. *"perseverance [triumph] as a testament"*), the item is **FATALLY FLAWED**. You MUST set `single_fit_valid: false`, assign `pedagogical_score <= 30`, and flag 'Missing predicate verb / ungrammatical sentence'.
     - **In-List Distractor Recycling**: If distractors contain other unexercised vocabulary items from the supplied unit list instead of independent contextual distractors, the item exhibits cross-item cueing/test pollution. You MUST set `single_fit_valid: false`, assign `pedagogical_score <= 30`, and flag 'In-list word recycling detected'.
     - **Severe Lexical/Collocation Tautology**: Phrasings that are unnatural or grammatically redundant (e.g. *"pledge a commitment"* instead of *"make a commitment"* or *"pledge to do"*) must be penalized severely.
   - If ANY option causes a sentence fragment, duplicate option, or grammatical breakdown, it CANNOT be considered a valid answer key.

1. **BLIND TEST-SOLVER SIMULATION (`blind_solved_index` & `confidence`)**:
   - `item_index` MUST strictly match the Item number displayed in the prompt (e.g., Item #1 -> item_index: 1, Item #2 -> item_index: 2, etc.).
   - Independently solve each question stem with its 4 options (Option A = 0, B = 1, C = 2, D = 3).
   - Determine the objectively correct answer based on textual evidence (if reading/video passage provided) or strict sentence-level syntax, dependent prepositions, and logical polarity anchors (for standalone items).
   - Set `confidence`:
     - `Definite`: Textual evidence or stem syntax/collocation are explicit, direct, and leave zero doubt.
     - `Hesitant`: The stem requires inference with slight interpretive friction.
     - `Ambiguous`: Multiple options could be argued as plausible OR the stem is structurally flawed/ungrammatical.

2. **ABSOLUTE SINGLE-FIT VALIDITY & SENTENCE-LEVEL ISOLATION (`single_fit_valid`)**:
   - Verify that there is EXACTLY ONE uniquely correct, grammatically sound, and contextually defensible answer.
   - ⚠️ **ISOLATED STEM TEST**: The key must be uniquely defensible based on the linguistic constraints of the **sentence stem itself**. If a distractor also works naturally in the sentence, the item is an **INVALID DOUBLE-KEY**, even if the original reading text happened to use the declared key.
   - If two options can both be justified by the context OR if the declared key creates an ungrammatical sentence, you MUST flag `single_fit_valid: false`.
   - 🌟 **RECOGNIZE AUTHENTIC METAPHOR, PERSONIFICATION & RHETORICAL EXTENSION**:
     * Do NOT be an overly literal pedant. In advanced English (CEFR B2-C2), institutional agency, metaphorical extension, hyperbole, and personification are standard, highly authentic linguistic devices.
     * Examples: Treating an institution/nation/museum as an *"inheritor"*, *"custodian"*, or *"guardian"* of heritage, or describing an economy as *"thirsty"* or *"reaping benefits"*, is 100% legitimate figurative English.
     * NEVER disqualify a correct answer or declare an item 'keyless' merely because the noun is figurative or personified rather than a strictly literal biological entity. If context clues clearly anchor the intended figurative meaning, confirm the key and rate it high quality (85-95).

3. **COGNITIVE DISTRACTOR TRAP ANALYSIS (`distractors`)**:
   - Evaluate all 4 options (including the correct key and the 3 distractors).
   - Identify the authentic educational trap type:
     - `None (Correct Answer)`: The uniquely valid key.
     - `L1 Negative Transfer / False Friend`: Leverages common ESL/Chinglish structural or lexical transfer errors.
     - `Scope Shift / Over-generalization`: Distorts text by turning a specific fact into an absolute/extreme universal claim.
     - `Speaker / Entity Misattribution`: Attributes a quote, thought, or deed to the wrong character/speaker.
     - `Chronological / Causal Inversion`: Reverses sequence of events or confuses cause with effect.
     - `Plausible Real-World Distractor`: Conceptually plausible in the real world, but contradicted or unsupported by the text.
     - `Flawed / Trivial Giveaway`: Grammatically flawed, absurd, childish/elementary filler, or trivially easy to eliminate without reading the passage.
   - Rate `plausibility_rating` (`High`, `Medium`, or `Low (Flawed)`). Any distractor with `Low (Flawed)` must be reported in `diagnostic_feedback`.
   - Provide a concise `elimination_rationale` explaining why a test-taker must definitively reject this choice.

4. **PROHIBITED ELIMINATION RATIONALES (MANDATORY GATE)**:
   - The following rationales are LOGICALLY INVALID and MUST NOT be used to eliminate a distractor or justify `single_fit_valid: true`:
     ✗ 'Not used in the passage / source material'
     ✗ 'The author / source uses [key] instead'
     ✗ 'Could be plausible but [key] is the target word'
     ✗ 'Fails the strict lexical match to the vocabulary list'
   - A valid elimination rationale MUST reference at least one objective linguistic constraint:
     ✓ Part-of-speech / inflectional / grammatical constraint
     ✓ Fixed collocation pattern or collocational frequency gap (≥3 orders of magnitude)
     ✓ Semantic contradiction or logical polarity clash with explicit stem clues
     ✓ Register / domain mismatch
     ✓ Preposition / complement / syntactic frame mismatch (e.g. requires different dependent preposition)
   - If you CANNOT provide an objective linguistic rationale for eliminating a distractor, you MUST set `single_fit_valid: false`.

5. **KEY VALIDITY REVERSE TEST (MANDATORY)**:
   - For EACH question, before confirming the key:
     * Read the stem with EACH of the 4 options inserted.
     * Ask: *'Does any NON-key option produce a MORE natural, more idiomatic, or more contextually precise sentence?'*
     * If YES → the key is SUBOPTIMAL or WRONG. Set `single_fit_valid: false` and explicitly flag in `diagnostic_feedback`: 'Key suboptimal: [option X] is a stronger/more natural collocation in this context.'
     * Only confirm the key if it is the UNIQUELY BEST fit, not merely 'acceptable' or 'present in the vocabulary list'.

6. **SCORE DIFFERENTIATION & VERDICT (`overall_quality_score` & `pass_audit`)**:
   - You MUST assign meaningfully differentiated `pedagogical_score` values across questions based on objective elimination rigor:
     - 90–100: All 3 distractors eliminated by HARD criteria (syntax, dependent prepositions, strict semantic contradiction).
     - 75–89: 1–2 distractors eliminated by hard criteria, 1 eliminated by softer criteria (register, collocational frequency gap).
     - 60–74: A distractor is defensible (potential double-key); item is usable but not ideal.
     - < 60: Clear double-key, triple-key, wrong/suboptimal key, or ungrammatical stem.
7. **IN-PLACE SURGICAL CURE MANDATE (`triage_action` & `cured_question`)**:
   - You are NOT a passive critic; you are an active, authoritative surgeon. You MUST assign `triage_action` for EVERY item:
     * `PASS`: The item is grammatically sound, contextually defensible, and has a unique valid key (`single_fit_valid == true`, score >= 80). `cured_question` MUST be null.
     * `REPAIR`: The question stem/context and core focus are strong, but has surface defects (e.g. minor article/syntax error like 'a ingredient' -> 'an ingredient', weak distractor, or mislabeled answer index). You MUST provide `cured_question` with the surgically corrected question matching the original question schema.
     * `REWRITE`: The question stem is fatally flawed, illogical, or exhibits severe double-keys/hallucination that cannot be salvaged by micro-edits. Discard the flawed item and provide a completely NEW, pristine question in `cured_question` testing the exact same target keyword/grammar point grounded in the reference material.
   - When `triage_action` is `REPAIR` or `REWRITE`, `cured_question` is MANDATORY and must be a complete, schema-compliant replacement item ready for immediate classroom deployment.

CONTENT:
{content}
