### SYSTEM ###
You are an expert ESL Audio Script Writer and Listening Assessment Designer (TOEFL / IELTS / Cambridge English standards).
### USER ###
Generate a realistic academic dialogue and a rigorous comprehension assessment based on the provided vocabulary items.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Authentic Academic Dialogue Script**:
   - Create a natural, engaging academic discussion between Speaker 1 and Speaker 2 consisting of 6 to 8 conversational turns.
   - Natural spoken register: realistic conversational flow with natural discourse markers (e.g., 'Well, look at it this way...', 'That's a valid point, but...', 'You mean...?'), gentle counter-arguments, and mutual clarification.
   - Seamlessly embed at least 5 target academic vocabulary items into natural spoken contexts without sounding like textbook recitations.

2. **Question & Skill Diversity**:
   - Generate EXACTLY {count} comprehension questions in the 'questions' array.
   - Cover a balanced mix of listening skills across questions:
     * `Detail`: Specific fact, limitation, or rationale stated by a speaker.
     * `Inference`: Drawing logical conclusions not explicitly phrased in the script.
     * `Main Idea`: Overall core purpose or takeaway of the conversation.

3. **Listening Distractor Taxonomy (NO Cartoonish Choices)**:
   - All 4 options must be plausible, concise, grammatically parallel, and closely tied to the discussion.
   - ❌ **STRICTLY PROHIBIT**: Childish or absurd choices (e.g., 'machines are too heavy to move', 'destroy all electronics'), trivial common-sense giveaways, and pure polar opposites.
   - Engineer distractors using authentic listening test cognitive traps:
     * *Speaker Attribution Trap*: Attributes an opinion, concern, or proposal to Speaker 1 when it was actually expressed or qualified by Speaker 2 (or vice versa).
     * *Verbatim Catch Trap*: Borrows an eye-catching technical term from the script (e.g., 'nuclear fusion', 'semiconductors'), but links it to a false claim or unmentioned context.
     * *Overstated Generalization Trap*: Uses extreme absolutes (*completely impossible*, *abandon entirely*, *useless*) when the speaker only expressed cautious reservation or conditional qualification.

4. **Design Audit & Explanation**:
   - `design_audit`: `AUDIT: [Skill (Detail/Inference/Main Idea)] -> [Speaker Anchor (e.g. Mark, Turn 4)] -> [Traps: Speaker Confusion / Verbatim Catch / Overstatement] -> [Why Distractors Fail]`
   - `explanation`: State the exact dialogue turn supporting the correct answer, and contrastively explain why each distractor trap is invalid.
   - `correct_answer_index`: MUST be an integer 0, 1, 2, or 3 matching the exact position of the true answer in 'options'. Do NOT use alternative key names.
   - Do NOT wrap question or option text in quotes or labels (A, B).

VOCABULARY:
{vocabulary_content}
