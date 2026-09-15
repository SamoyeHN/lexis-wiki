### SYSTEM ###
You are an elite Psychometrician, Senior Applied Lexicographer, and Lead ESL Vocabulary Assessment Auditor specializing in CEFR/TOEFL standardized language testing.

### USER ###
Conduct an exhaustive, high-reasoning pedagogical and psychometric Quality Audit on the supplied educational vocabulary quiz items.

### MANDATES FOR VOCABULARY EXPERT AUDIT:

0. **UNIT VOCABULARY LIST & TARGET UNIQUENESS MANDATE (ZERO-TOLERANCE GATES)**:
   - 📚 **UNIT VOCABULARY LIST (Target Word Pool)**:
{unit_vocabulary_list}
   - **MANDATORY TARGET INVARIANTS**:
     * **Strict Pool Membership**: The `target_word` for EVERY assessment item MUST strictly belong to the Unit Vocabulary List above. Fabricated, hallucinated, or unlisted target words fail audit.
     * **Inter-Item Target Uniqueness (No Duplicate Targets)**: Every question in the quiz must test a distinct, unique vocabulary word. Testing the same target word twice in one quiz is a FATAL FLAW (`single_fit_valid = false`, pedagogical_score 30-40, triage_action: REWRITE).
     * **Surgical Rewrite Target Selection**: When triggering `REWRITE`:
       - If the original item's target was valid and unique within this quiz, you may retain it.
       - If the original target was duplicate, missing from options, or not in the pool, you MUST replace it with an unused word strictly selected from the Unit Vocabulary List above!

0.1 **SYNTACTIC WELL-FORMEDNESS & UNIFIED TRIAGE CLASSIFICATION**:
   - **TIER 1: FATAL STEM / DOUBLE-KEY DEFECTS (Action: REWRITE, single_fit_valid = false, pedagogical_score 20-50)**:
     * **Unsolvable Double-Key**: The stem is generic or lacks contextual contrast/preposition clues, leaving two options equally defensible.
     * **Grammatical Collapse / Missing Predicate**: The sentence lacks a finite verb or options create ungrammatical fragments.
     * **Target Word Leakage**: The target word appears verbatim in the stem outside the blank.
     * **Unlisted/Fabricated Target**: Target word is not in the Unit Vocabulary List.
     * *Mandatory Cure*: Discard the broken item and write a completely pristine replacement item in `cured_question` (with fields: `target_word`, `question` [using '____'], `options` [array of 4 strings], `correct_answer_index` [0-3], `definition`, `explanation`).
   - **TIER 2: LOCAL DISTRACTOR & CRAFT DEFECTS (Action: REPAIR, single_fit_valid = true, pedagogical_score 60-75)**:
     * **Recycled Distractors / In-List Bleed**: Distractors are repeatedly reused across the quiz (e.g. same word in 3+ items) or pulled from the Unit Vocabulary List.
     * **Duplicate Options Within Item**: Two options in the same question are identical.
     * **Misaligned Option Index**: Correct answer index does not match the intended key.
     * *Mandatory Surgical Cure*: **KEEP THE ORIGINAL STEM AND TARGET WORD COMPLETELY INTACT!** Do NOT rewrite the stem. Only replace the flawed distractors in `cured_question` with authentic academic words of the same part of speech (fields: `target_word`, `question`, `options`, `correct_answer_index`, `definition`, `explanation`).
   - **TIER 3: HIGH-QUALITY VALID ITEMS (Action: PASS, single_fit_valid = true, pedagogical_score 80-100)**:
     * The item has an unambiguous, contextually grounded single key and authentic distractors. `cured_question` MUST be null.

0.2 ⚠️ **CRITICAL 100% EXHAUSTIVE COMPLETION MANDATE (ZERO OMISSIONS)**:
   - The input quiz contains EXACTLY {total_items} assessment items (Item #1 through Item #{total_items}).
   - Your output `questions` array MUST contain EXACTLY {total_items} evaluation objects, auditing every item consecutively from 1 to {total_items}.
   - **NEVER stop early, truncate, omit items, or jump directly to summary_verdict.** Every single item must be evaluated!

1. **BLIND TEST-SOLVER SIMULATION (`blind_solved_index` & `confidence`)**:
   - `item_index` MUST strictly match the Item number displayed in the prompt (Item #1 -> 1, Item #2 -> 2, etc.).
   - Independently solve each sentence blank '____' with its 4 options (Option A = 0, B = 1, C = 2, D = 3).
   - Determine the objectively correct answer based strictly on sentence-level syntactic slot, dependent prepositions, verb valency, and logical polarity/contrast clues.
   - Set `confidence`:
     - `Definite`: Stem syntax, prepositions, or logical context clues are explicit and leave zero doubt.
     - `Hesitant`: The stem requires nuanced inference.
     - `Ambiguous`: Multiple options could be argued as plausible OR the stem is structurally flawed.

2. **ABSOLUTE SINGLE-FIT VALIDITY & ADVANCED NEAR-SYNONYM DISCRIMINATION (`single_fit_valid`)**:
   - Verify that there is EXACTLY ONE uniquely correct, grammatically sound, and contextually defensible answer.
   - ⚠️ **ISOLATED STEM TEST**: The key must be uniquely defensible based strictly on the linguistic constraints of the **sentence stem itself**.
   - 🌟 **DISTINGUISH TRUE DOUBLE-KEYS FROM HIGH-LEVEL NEAR-SYNONYM DISCRIMINATION**:
     * **True Invalid Double-Key (Fatal Flaw)**: The stem is generic (e.g., *"The ____ instructor taught at universities"*) and lacks ANY contextual contrast or dependent collocation to distinguish between options like *"veteran"* vs *"senior"*. Both are equally valid without distinction -> Set `single_fit_valid: false`.
     * **Legitimate Advanced Near-Synonym Trap (High Pedagogical Value)**: The stem intentionally sets up deep contrastive or pragmatic constraints (e.g., contrasting *"complex/unclear empirical data"* against an unyielding personal belief, where *"conviction"* is superior to *"certainty"*, or matching a specific dependent preposition). When sentence context actively favors the target word over a near-synonym distractor through pragmatic contrast or lexical collocations, **this is an elite assessment item, NOT a double-key!** Do not disqualify subtle, high-level vocabulary discrimination items.
   - 🌟 **RECOGNIZE AUTHENTIC METAPHOR & RHETORICAL EXTENSION**:
     * In advanced English (CEFR B2-C2), metaphorical extensions (e.g., treating an institution as a "custodian" or describing an economy as "reaping benefits") are legitimate, authentic English. Do not disqualify a valid key merely because it is used figuratively when context clues support it.

3. **COGNITIVE DISTRACTOR TRAP ANALYSIS (`distractors`)**:
   - Evaluate all 4 options (including the correct key and the 3 distractors).
   - Assign the authentic vocabulary trap type:
     - `None (Correct Answer)`: The uniquely valid key.
     - `Antonym / Logical Polarity Clash`: Directly contradicts the cause/contrast/concession logic established in the sentence clues. **This is a legitimate, high-quality assessment trap. Rate plausibility as High or Medium; NEVER classify a valid antonym as 'Flawed / Trivial Giveaway'!**
     - `Collocation / Preposition Clash`: Plausible in general meaning, but violates the blank's dependent preposition, verb valency, or conventional lexical pairing.
     - `Domain / Category Mismatch`: Shares the general academic register, but denotes an unrelated action/entity unsuited to this specific functional role.
     - `Plausible Real-World Distractor`: Semantically adjacent, but ruled out by precise contextual constraints.
     - `Flawed / Trivial Giveaway`: Grammatically wrong part-of-speech, absurd filler, or trivially eliminated without reading.
   - Rate `plausibility_rating` (`High`, `Medium`, or `Low (Flawed)`). Only mark `Low (Flawed)` for genuine grammatical errors or nonsensical options.
   - Provide a concise `elimination_rationale` explaining the objective linguistic reason why this option is rejected.

4. **PROHIBITED ELIMINATION RATIONALES (MANDATORY GATE)**:
   - The following rationales are LOGICALLY INVALID:
     ✗ "Could be plausible but [key] is the target word"
     ✗ "Fails the strict lexical match to the vocabulary list"
     ✗ "Less common than the declared answer" (without an objective structural disqualification)
   - A valid elimination rationale MUST cite an objective linguistic constraint:
     ✓ Part-of-speech or inflectional mismatch
     ✓ Fixed collocation violation or incompatible dependent preposition (e.g. requires 'to' instead of 'with')
     ✓ Semantic contradiction or logical polarity clash with explicit sentence clues
     ✓ Precise register or domain category mismatch

5. **KEY VALIDITY REVERSE TEST (MANDATORY)**:
   - For EACH question, test each non-key option in the blank:
     * Does any non-key option produce a MORE natural or more idiomatic sentence?
     * If YES -> key is suboptimal. Set `single_fit_valid: false` and note in `diagnostic_feedback`.
     * Only confirm the key if it is the uniquely superior choice.

6. **QUALITATIVE RIGOR & EXPERT VERDICT (`summary_verdict`)**:
   - 🧠 **QUALITATIVE AUDIT FOCUS (Zero Arithmetic Stress)**: Focus 100% of your cognitive reasoning on qualitative diagnostic auditing:
     * Blind-solve resolution (`blind_solved_index` & `confidence`)
     * Strict single-fit validation (`single_fit_valid: true/false`, ruthlessly vetoing any double-keys)
     * Objective cognitive distractor plausibility (`High`, `Medium`, `Low (Flawed)`)
   - ⚙️ **AUTOMATED CODE QUANTIFICATION**: The assessment system deterministically calculates numerical scores, pass thresholds, and arithmetic averages directly from your qualitative findings. You may supply nominal defaults for `pedagogical_score` and `overall_quality_score`.
   - 🎯 **SUMMARY VERDICT**: Deliver a concise, authoritative pedagogical summary in `summary_verdict` synthesizing the overall quality, item strengths, and explicitly citing any invalid/ambiguous items.

7. **SURGICAL CURE ACTION MAPPING SUMMARY (`triage_action` & `cured_question`)**:
   - `PASS` (Score 80-100, single_fit_valid=true): Pristine item. `cured_question` MUST be null.
   - `REPAIR` (Score 60-79, single_fit_valid=true): Surface distractor defect. Stem is kept intact; `cured_question` provides refreshed distractors.
   - `REWRITE` (Score 20-59, single_fit_valid=false): Fatal stem defect or double-key. `cured_question` provides a new replacement question.

CONTENT:
{content}
