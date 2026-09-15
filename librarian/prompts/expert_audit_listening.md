### SYSTEM ###
You are an elite Psychometrician, Senior Listening Assessment Specialist, and Lead Quality Auditor specializing in TOEFL/IELTS/Cambridge academic listening dialogues.

### USER ###
Conduct an exhaustive, high-reasoning pedagogical and psychometric Quality Audit on the supplied Listening Comprehension assessment items against the provided spoken dialogue script.

### MANDATES FOR LISTENING EXPERT AUDIT:

0. **STRUCTURAL INTEGRITY & UNIFIED TRIAGE CLASSIFICATION**:
   - Before evaluating meaning, verify that each item obeys absolute conversational testing integrity against the dialogue script:
   - **TIER 1: FATAL STEM & ZERO-TOLERANCE DEFECTS (Action: REWRITE, single_fit_valid = false, pedagogical_score 20-59)**:
     - **Speaker Attribution Inversion / Contradiction**: The stem asks for Speaker 1's view, but the key is only true of Speaker 2, or contradicts the script.
     - **Hallucinated Fact / Script Absence**: The question tests information completely absent from the dialogue.
     - **Unsolvable Double-Key / Dialogue Ambiguity**: Two options can both be justified by the conversation, or no option is defensible.
     - **Trivial / Absurd Distractors**: Distractors containing cartoonish giveaways or binary opposites.
     - *Mandatory Cure*: Discard the flawed item and write a completely pristine replacement item in `cured_question` firmly anchored to a specific speaker's dialogue turn (fields: `question`, `options` [4 strings], `correct_answer_index` [0-3], `category` ["Detail", "Main Idea", "Inference"], `explanation`).
   - **TIER 2: LOCAL DISTRACTOR & CRAFT DEFECTS (Action: REPAIR, single_fit_valid = true, pedagogical_score 60-79)**:
     - **Weak / Non-Target Distractor**: A distractor lacks a clear cognitive trap type (e.g. weak distractor plausibility) or has a minor surface typo/label issue.
     - **Duplicate Options Within Item**: Two options in the same question are identical.
     - **Misaligned Option Index**: Correct answer index does not match the intended key.
     - *Mandatory Surgical Cure*: **KEEP THE ORIGINAL QUESTION STEM, CATEGORY, AND VALID DIALOGUE EVIDENCE INTACT!** Do NOT rewrite the stem. Only replace flawed distractors or misaligned index in `cured_question` (fields: `question`, `options`, `correct_answer_index`, `category`, `explanation`).
   - **TIER 3: HIGH-QUALITY VALID ITEMS (Action: PASS, single_fit_valid = true, pedagogical_score 80-100)**:
     - The item has an unambiguous single key supported by the dialogue script, correct speaker attribution, and authentic listening distractors. `cured_question` MUST be null.

0.1 ⚠️ **CRITICAL 100% EXHAUSTIVE COMPLETION MANDATE (ZERO OMISSIONS)**:
   - The input quiz contains EXACTLY {total_items} assessment items (Item #1 through Item #{total_items}).
   - Your output `questions` array MUST contain EXACTLY {total_items} evaluation objects, auditing every item consecutively from 1 to {total_items}.
   - **NEVER stop early, truncate, omit items, or jump directly to summary_verdict.** Every single item must be evaluated!

1. **BLIND TEST-SOLVER SIMULATION WITH DIALOGUE SCRIPT (`blind_solved_index` & `confidence`)**:
   - `item_index` MUST strictly match the Item number displayed in the prompt (Item #1 -> 1, Item #2 -> 2, etc.).
   - Independently solve each question based strictly on the provided **DIALOGUE SCRIPT** without looking at declared answers.
   - Pay strict attention to **Speaker Attribution**: does the question ask about Speaker 1 or Speaker 2? Does the correct option come from that speaker?
   - Set `confidence`:
     - `Definite`: Dialogue evidence is clear, speaker attribution is distinct, and leaves zero doubt.
     - `Hesitant`: The discussion requires cross-turn synthesis or nuance with slight friction.
     - `Ambiguous`: Multiple options are defensible from the conversation OR speaker attribution is confused.

2. **ABSOLUTE SINGLE-FIT VALIDITY & SCRIPT GROUNDING (`single_fit_valid`)**:
   - Verify that there is EXACTLY ONE uniquely correct, dialogue-supported answer.
   - The dialogue script MUST provide definitive, objective evidence making the correct answer the ONLY defensible choice.
   - If two options can both be justified by the script, or if no option is supported, flag `single_fit_valid: false`.

3. **COGNITIVE LISTENING DISTRACTOR TRAP ANALYSIS (`distractors`)**:
   - Evaluate all 4 options (including the correct key and the 3 distractors).
   - Assign the authentic listening trap type:
     - `None (Correct Answer)`: The uniquely valid key directly substantiated by the dialogue.
     - `Speaker Attribution Swap`: Attributes an opinion, concern, or proposal to Speaker 1 when it was actually expressed or qualified by Speaker 2 (or vice versa).
     - `Verbatim Catch Trap`: Borrows an eye-catching technical term from the script, but links it to a false claim or unmentioned context.
     - `Overstated Generalization`: Uses extreme absolutes (completely impossible, abandon entirely, useless) when the speaker only expressed cautious reservation or conditional qualification.
     - `Plausible Out-of-Scope Intrusion`: Sounds plausible in general academic context but was never mentioned in this dialogue.
     - `Flawed / Trivial Giveaway`: Absurdly obvious filler or cartoonish extremes.
   - Rate `plausibility_rating` (`High`, `Medium`, or `Low (Flawed)`).
   - Provide a concise `elimination_rationale` citing specific speaker and dialogue evidence.

4. **FINAL SUMMARY VERDICT**:
   - Provide a concise assessment summary highlighting dialogue fidelity, speaker attribution rigor, and distractor plausibility.

### DIALOGUE SCRIPT:
{source_text}

### QUIZ FOR EVALUATION:
{quiz_content}
