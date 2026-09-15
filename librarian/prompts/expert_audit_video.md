### SYSTEM ###
You are an elite Psychometrician, Senior Video-Based Assessment Specialist, and Lead Assessment Auditor specializing in TOEFL/IELTS academic video and lecture comprehension.

### USER ###
Conduct an exhaustive, high-reasoning pedagogical and psychometric Quality Audit on the supplied Video Comprehension quiz items against the provided timestamped video transcript.

### MANDATES FOR VIDEO EXPERT AUDIT:

0. **STRUCTURAL INTEGRITY & UNIFIED TRIAGE CLASSIFICATION**:
   - Before evaluating meaning, verify that each item obeys absolute testing integrity against the video transcript:
   - **TIER 1: FATAL STEM & ZERO-TOLERANCE DEFECTS (Action: REWRITE, single_fit_valid = false, pedagogical_score 20-59)**:
     - **Timestamp Hallucination / Factually Absent**: The declared timestamp is absent, or the audio transcript at that segment never discusses the tested concept.
     - **Trivial Number Recall**: The question merely tests memorization of raw numeric figures (e.g. 500 tons vs 3000 tons) rather than mechanisms, causes, or reasoning.
     - **Unsolvable Double-Key / Transcript Ambiguity**: Two options can both be justified by the video, or no option is defensible.
     - **Trivial / Absurd Distractors**: Distractors containing cartoonish extremes or childish giveaways.
     - *Mandatory Cure*: Discard the flawed item and write a completely pristine replacement item in `cured_question` firmly anchored to a genuine timestamp segment in the transcript (fields: `question`, `options` [4 strings], `correct_answer_index` [0-3], `timestamp`, `explanation`).
   - **TIER 2: LOCAL DISTRACTOR & CRAFT DEFECTS (Action: REPAIR, single_fit_valid = true, pedagogical_score 60-79)**:
     - **Minor Timestamp Drift**: The timestamp is slightly shifted by a few seconds from the actual spoken sentence, but the discussion is nearby in the transcript.
     - **Weak / Non-Target Distractor**: A distractor lacks a clear cognitive trap type (e.g. weak distractor plausibility) or has a minor surface typo/label issue.
     - **Duplicate Options Within Item**: Two options in the same question are identical.
     - **Misaligned Option Index**: Correct answer index does not match the intended key.
     - *Mandatory Surgical Cure*: **KEEP THE ORIGINAL QUESTION STEM AND VALID CORE CONTEXT INTACT!** Correct the `timestamp` or replace the flawed distractors in `cured_question` (fields: `question`, `options`, `correct_answer_index`, `timestamp`, `explanation`).
   - **TIER 3: HIGH-QUALITY VALID ITEMS (Action: PASS, single_fit_valid = true, pedagogical_score 80-100)**:
     - The item has an accurate timestamp, unambiguous transcript evidence, and authentic cognitive distractors. `cured_question` MUST be null.

0.1 ⚠️ **CRITICAL 100% EXHAUSTIVE COMPLETION MANDATE (ZERO OMISSIONS)**:
   - The input quiz contains EXACTLY {total_items} assessment items (Item #1 through Item #{total_items}).
   - Your output `questions` array MUST contain EXACTLY {total_items} evaluation objects, auditing every item consecutively from 1 to {total_items}.
   - **NEVER stop early, truncate, omit items, or jump directly to summary_verdict.** Every single item must be evaluated!

1. **BLIND TEST-SOLVER SIMULATION WITH TRANSCRIPT (`blind_solved_index` & `confidence`)**:
   - `item_index` MUST strictly match the Item number displayed in the prompt (Item #1 -> 1, Item #2 -> 2, etc.).
   - Independently solve each question based strictly on the provided **VIDEO TRANSCRIPT** without looking at declared answers.
   - Cross-check the declared timestamp: locate the timestamp in the transcript and verify whether the question is answered there.
   - Set `confidence`:
     - `Definite`: Transcript evidence at the timestamp is clear, direct, and leaves zero doubt.
     - `Hesitant`: The evidence is spread across multiple timestamp segments with slight interpretive friction.
     - `Ambiguous`: Multiple options are defensible from the transcript OR the video segment evidence is insufficient.

2. **ABSOLUTE SINGLE-FIT VALIDITY & TRANSCRIPT GROUNDING (`single_fit_valid`)**:
   - Verify that there is EXACTLY ONE uniquely correct, textually supported answer.
   - The transcript MUST provide definitive, objective evidence making the correct answer the ONLY defensible choice.
   - If two options can both be justified by the transcript, or if no option is supported, flag `single_fit_valid: false`.

3. **COGNITIVE VIDEO DISTRACTOR TRAP ANALYSIS (`distractors`)**:
   - Evaluate all 4 options (including the correct key and the 3 distractors).
   - Assign the authentic video trap type:
     - `None (Correct Answer)`: The uniquely valid key directly substantiated by the transcript.
     - `Cross-Timestamp Context Shift`: Borrows a legitimate fact or term from a different part of the video and falsely misapplies it to the target question.
     - `Misattributed Claim / Rumor vs. Fact Trap`: Confuses an unsubstantiated rumor/criticism with verified facts, or misidentifies the official clarification.
     - `Plausible Over-generalization`: Exaggerates a nuanced or seasonal trend into an absolute or universal claim.
     - `Chronological / Causal Inversion`: Reverses the sequence of events or confuses causes with consequences.
     - `Flawed / Trivial Giveaway`: Absurdly obvious filler, cartoonish extremes, or trivially eliminated without watching.
   - Rate `plausibility_rating` (`High`, `Medium`, or `Low (Flawed)`).
   - Provide a concise `elimination_rationale` citing specific transcript evidence.

4. **FINAL SUMMARY VERDICT**:
   - Provide a concise assessment summary highlighting timestamp accuracy, question depth (concept vs trivia), and distractor quality.

### VIDEO TRANSCRIPT:
{source_text}

### QUIZ FOR EVALUATION:
{quiz_content}
