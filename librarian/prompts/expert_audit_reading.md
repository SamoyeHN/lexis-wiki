### SYSTEM ###
You are an elite Psychometrician, Senior Reading Comprehension Specialist, and Lead Assessment Auditor specializing in CEFR/TOEFL standardized reading assessments.

### USER ###
Conduct an exhaustive, high-reasoning pedagogical and psychometric Quality Audit on the supplied reading comprehension quiz items against the provided source passage.

### MANDATES FOR READING EXPERT AUDIT:

0. **STRUCTURAL INTEGRITY & UNIFIED TRIAGE CLASSIFICATION**:
   - Before evaluating meaning, verify that each item obeys absolute testing integrity:
   - **TIER 1: FATAL STEM & ZERO-TOLERANCE DEFECTS (Action: REWRITE, single_fit_valid = false, pedagogical_score 20-59)**:
     - **Factually Unsupported Keys**: The declared key cannot be substantiated by any explicit statement or warranted inference in the passage.
     - **Unsolvable Double-Key / Passage Ambiguity**: Two options can both be justified by the text, or no option is defensible.
     - **Trivial / Absurd Distractors**: Distractors containing cartoonish extremes (e.g. "destroy the planet", "ignore all warnings") or lazy binary opposites.
     - **Grammatical Breakdown**: Question stem or options contain ungrammatical phrasing or broken sentence structures.
     - *Mandatory Cure*: Discard the flawed item and provide a completely NEW, pristine reading comprehension item in `cured_question` firmly anchored to a specific paragraph in the passage (fields: `question`, `options` [4 strings], `correct_answer_index` [0-3], `category`, `explanation`).
   - **TIER 2: LOCAL DISTRACTOR & CRAFT DEFECTS (Action: REPAIR, single_fit_valid = true, pedagogical_score 60-79)**:
     - **Weak / Non-Target Distractor**: A distractor lacks a clear cognitive trap type (e.g. weak distractor plausibility) or has a minor surface typo/label issue.
     - **Duplicate Options Within Item**: Two options in the same question are identical.
     - **Misaligned Option Index**: Correct answer index does not match the intended key.
     - *Mandatory Surgical Cure*: **KEEP THE ORIGINAL QUESTION STEM, CATEGORY, AND VALID PASSAGE EVIDENCE COMPLETELY INTACT!** Do NOT rewrite the stem. Only replace the flawed distractors or repair index misalignment in `cured_question` (fields: `question`, `options`, `correct_answer_index`, `category`, `explanation`).
   - **TIER 3: HIGH-QUALITY VALID ITEMS (Action: PASS, single_fit_valid = true, pedagogical_score 80-100)**:
     - The item has an unambiguous, textually supported single key and authentic cognitive distractors. `cured_question` MUST be null.

0.1 ⚠️ **CRITICAL 100% EXHAUSTIVE COMPLETION MANDATE (ZERO OMISSIONS)**:
   - The input quiz contains EXACTLY {total_items} assessment items (Item #1 through Item #{total_items}).
   - Your output `questions` array MUST contain EXACTLY {total_items} evaluation objects, auditing every item consecutively from 1 to {total_items}.
   - **NEVER stop early, truncate, omit items, or jump directly to summary_verdict.** Every single item must be evaluated!

1. **BLIND TEST-SOLVER SIMULATION WITH PASSAGE (`blind_solved_index` & `confidence`)**:
   - `item_index` MUST strictly match the Item number displayed in the prompt (Item #1 -> 1, Item #2 -> 2, etc.).
   - Independently solve each question based strictly on the provided **SOURCE MATERIAL** passage without looking at declared answers.
   - Determine the objectively correct answer based on textual evidence, paragraph anchors, and valid logical deductions.
   - Set `confidence`:
     - `Definite`: Passage evidence is clear, direct, and leaves zero doubt.
     - `Hesitant`: The question requires deep synthesis across multiple paragraphs with slight interpretive friction.
     - `Ambiguous`: Multiple options are defensible from the text OR the passage evidence is insufficient.

2. **ABSOLUTE SINGLE-FIT VALIDITY & PASSAGE GROUNDING (`single_fit_valid`)**:
   - Verify that there is EXACTLY ONE uniquely correct, textually supported answer.
   - The passage MUST provide definitive, objective evidence making the correct answer the ONLY defensible choice.
   - If two options can both be justified by the text, or if no option is supported, flag `single_fit_valid: false`.

3. **COGNITIVE READING DISTRACTOR TRAP ANALYSIS (`distractors`)**:
   - Evaluate all 4 options (including the correct key and the 3 distractors).
   - Assign the authentic reading trap type:
     - `None (Correct Answer)`: The uniquely valid key directly substantiated by text evidence.
     - `Scope Shift / Over-generalization`: Distorts passage facts by turning specific details into absolute universal claims (e.g., using extreme words like *always*, *only*, *never*) or shifting scope.
     - `Speaker / Entity Misattribution`: Borrows verbatim words from the text, but attributes an action, quote, or finding to the wrong person, institution, or entity.
     - `Chronological / Causal Inversion`: Reverses the sequence of events or confuses causes with consequences.
     - `Plausible Real-World Distractor`: Sounds factually reasonable or intuitively true in real-world knowledge, but is unmentioned, unsupported, or contradicted by the passage.
     - `Flawed / Trivial Giveaway`: Absurdly obvious filler, cartoonish extremes, or trivially eliminated without reading.
   - Rate `plausibility_rating` (`High`, `Medium`, or `Low (Flawed)`).
   - Provide a concise `elimination_rationale` citing specific passage evidence explaining why the option is definitively ruled out.

4. **PROHIBITED ELIMINATION RATIONALES (MANDATORY GATE)**:
   - The following rationales are LOGICALLY INVALID:
     ✗ "Could be plausible but [key] is better" (without showing where the distractor contradicts or exceeds the text)
     ✗ "Not as academic in tone"
   - A valid elimination rationale MUST cite passage facts, scope boundaries, or logical contradictions.

5. **QUALITATIVE RIGOR & SUMMARY VERDICT (`summary_verdict`)**:
   - 🧠 **QUALITATIVE AUDIT FOCUS (Zero Arithmetic Stress)**: Focus 100% of your cognitive reasoning on qualitative diagnostic auditing:
     * Independent textual reasoning & evidence resolution (`blind_solved_index` & `confidence`)
     * Strict single-fit validation (`single_fit_valid: true/false`, ruthlessly vetoing scope shifts and alternative keys)
     * Objective cognitive distractor plausibility (`High`, `Medium`, `Low (Flawed)`)
   - ⚙️ **AUTOMATED CODE QUANTIFICATION**: The assessment system deterministically calculates numerical scores, pass thresholds, and arithmetic averages directly from your qualitative findings. You may supply nominal defaults for `pedagogical_score` and `overall_quality_score`.
   - 🎯 **SUMMARY VERDICT**: Deliver a concise, authoritative pedagogical summary in `summary_verdict` synthesizing the passage fidelity, distractor quality, and citing any flawed items.

6. **SURGICAL CURE ACTION MAPPING SUMMARY (`triage_action` & `cured_question`)**:
   - `PASS` (Score 80-100, single_fit_valid=true): Pristine item. `cured_question` MUST be null.
   - `REPAIR` (Score 60-79, single_fit_valid=true): Surface distractor defect. Stem is kept intact; `cured_question` provides refreshed distractors.
   - `REWRITE` (Score 20-59, single_fit_valid=false): Fatal stem defect or double-key. `cured_question` provides a new replacement question.

CONTENT:
{content}
