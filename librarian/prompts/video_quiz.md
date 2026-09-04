### SYSTEM ###
You are an expert Video-Based ESL Assessment Designer (TOEFL / IELTS / Academic Documentary standards).
### USER ###
Create a timestamp-aware video comprehension quiz based on the provided video transcript.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Exact Question Count & Chronological Timestamp Coverage**:
   - Generate EXACTLY {count} timestamp-aware video questions in the 'questions' array.
   - Distribute questions evenly across the chronological timeline of the video (e.g. Early context/mechanisms, Middle engineering challenges/environmental impacts, Later controversies/future upgrades).
   - Every question must map to a specific timestamp present verbatim in the transcript (e.g. [01:25.10] or [07:46.50]) where the evidence is clearly discussed.

2. **Higher-Order Video Comprehension (NO Trivial Number Recall)**:
   - ❌ **BAN TRIVIAL NUMBER GUESSING**: Do NOT write pure numeric recall questions (e.g., guessing between 500 tons vs 3000 tons, or 10 GW vs 22.5 GW).
   - Focus on meaningful conceptual, causal, and analytical understanding:
     * `Technical Mechanism & Cause-Effect`: Why a specific engineering solution was implemented, or how a natural condition affects operations.
     * `Controversy & Argumentation`: Contrasting external criticisms or rumors with official explanations or engineering realities.
     * `Comparative Analysis & Future Outlook`: Evaluating how the project compares with international counterparts or what future innovations (e.g., AI, railways) are proposed.

3. **Video Distractor Taxonomy (STRICT BAN on Absurd Options)**:
   - All 4 options must be plausible, grammatically parallel, and written in formal academic English.
   - ❌ **STRICTLY PROHIBIT**: Childish, absurd, or comical answers (e.g., 'Komodo dragons', 'the dam is made of steel', 'it is too small to hold water'), trivial common-sense giveaways, and pure polar opposites.
   - Engineer distractors using authentic video assessment cognitive traps:
     * *Cross-Timestamp Context Shift*: Borrows a legitimate fact or term from a different part of the video and falsely misapplies it to the target question.
     * *Misattributed Claim / Rumor vs. Fact Trap*: Confuses an unsubstantiated rumor/criticism with verified facts, or misidentifies the official clarification.
     * *Plausible Over-generalization*: Exaggerates a nuanced or seasonal trend into an absolute or universal claim.

4. **Design Audit & Explanation**:
   - `design_audit`: `AUDIT: [Timestamp Segment] -> [Core Focus: Mechanism / Controversy / Comparison] -> [Traps: Cross-timestamp shift, Rumor vs fact, Over-generalization] -> [Why Distractors Fail]`
   - `explanation`: State what the video explicitly clarifies at the given timestamp, and contrastively explain why each distractor trap is invalid.
   - `correct_answer_index`: MUST be an integer 0, 1, 2, or 3 matching the exact position of the true answer in 'options'.
   - Do NOT wrap question or option text in quotes or labels (A, B).

VIDEO TRANSCRIPT:
{transcript_content}
