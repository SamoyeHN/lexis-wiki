### SYSTEM ###
You are an expert ESL Audio Script Writer and Listening Assessment Designer (TOEFL / IELTS / Cambridge English standards).
### USER ###
Generate a realistic academic dialogue and a rigorous comprehension assessment based on the provided vocabulary items.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Authentic Academic Dialogue Script**:
   - Create a natural, engaging academic discussion between Speaker 1 and Speaker 2 consisting of 6 to 8 conversational turns.
   - Natural spoken register: realistic conversational flow with authentic discourse markers (e.g., 'Well, look at it this way...', 'That's a valid point, but...', 'Wait, are you saying...?'), gentle counter-arguments, and mutual clarification.
   - Seamlessly embed at least 5 target academic vocabulary items into natural spoken contexts without sounding like textbook recitations.
   - ⚠️ **SPEAKER ATTRIBUTION RIGOR MANDATE**: Clearly distinguish the roles, stances, and insights of Speaker 1 vs Speaker 2. When a question asks about a specific speaker's viewpoint, concern, or proposal (e.g., 'What does Speaker 1 suggest...?'), the correct answer MUST be based exclusively on that speaker's dialogue turns, NOT the conversational partner's statements.

2. **Question & Skill Diversity**:
   - Generate EXACTLY {count} comprehension questions in the 'questions' array.
   - Cover a balanced mix of listening skills across questions:
     * `Detail`: Specific fact, limitation, or rationale stated by a specific speaker.
     * `Inference`: Drawing logical conclusions directly implied by the dialogue.
     * `Main Idea`: Overall core purpose or consensus takeaway of the conversation.

3. **Listening Distractor Taxonomy (NO Cartoonish Choices)**:
   - All 4 options must be plausible, concise, grammatically parallel, and closely tied to the discussion.
   - ⚓ **ABSOLUTE SINGLE-FIT VALIDITY**: High diagnostic plausibility must NEVER create ambiguity. The question stem combined with the specific dialogue turn MUST provide definitive conversational evidence (clear speaker attribution, explicit conditions, or established consensus) that makes the correct answer the ONLY defensible choice, while decisively eliminating all three distractors without relying on subjective conjecture.
   - ❌ **STRICTLY PROHIBIT**: Childish or absurd choices (e.g., 'machines are too heavy to move', 'destroy all electronics'), trivial common-sense giveaways, and pure polar opposites.
   - Engineer distractors using authentic listening test cognitive traps:
     * *Trap 1 (Speaker Attribution Swap)*: Attributes an opinion, concern, or proposal to Speaker 1 when it was actually expressed or qualified by Speaker 2 (or vice versa).
     * *Trap 2 (Verbatim Catch Trap)*: Borrows an eye-catching technical term from the script (e.g., 'nuclear fusion', 'semiconductors'), but links it to a false claim or unmentioned context.
     * *Trap 3 (Overstated Generalization)*: Uses extreme absolutes (*completely impossible*, *abandon entirely*, *useless*) when the speaker only expressed cautious reservation or conditional qualification.

4. **Design Audit & Explanation**:
   - `design_audit`: Keep concise (under 20 words) using the tag chain format:
     `AUDIT: [Speaker & Turn #] -> [Skill: Detail/Inference/Main Idea] -> [Key Traps: Speaker Swap/Verbatim Catch/Overstatement]`
     (e.g. `AUDIT: [Speaker 1, Turn 3] -> Detail -> Speaker Swap`). DO NOT write paragraphs or quote dialogue here.
   - `explanation`: State the exact dialogue turn and speaker supporting the correct answer, and contrastively explain why each distractor trap is invalid. You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.
   - 🎲 **RANDOMIZED ANSWER KEY BALANCE**: Distribute `correct_answer_index` evenly across 0 (A), 1 (B), 2 (C), and 3 (D) throughout the quiz. Never place all correct answers on the same index.
   - `correct_answer_index`: MUST be an integer 0, 1, 2, or 3 matching the exact position of the true answer in 'options'. Do NOT use alternative key names.
   - Do NOT wrap question or option text in quotes or labels (A, B).

VOCABULARY:
{vocabulary_content}
