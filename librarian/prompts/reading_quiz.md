### SYSTEM ###
You are an expert Reading Comprehension Assessment Designer.
### USER ###
Create a reading comprehension assessment based on the provided passage.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Count & Vocabulary**:
   - Generate EXACTLY {count} comprehension questions.
   - Extract 5 to 8 challenging academic vocabulary items with verbatim context sentences, parts of speech (noun, verb, adjective, adverb, preposition, conjunction, interjection), concise definitions, and authentic example sentences.

2. **Question & Skill Diversity**:
   - Cover a balanced mix of skills across questions: `Main Idea`, `Detail/Recall`, `Inference`, and `Author's Tone/Purpose`.
   - Questions must require genuine comprehension of the text rather than superficial string-matching. Use clear phrasing without outer quotation marks.
   - ⚠️ **VERBATIM TEXT ANCHOR MANDATE**: Every Detail/Recall and Inference question MUST be anchored to specific, verifiable statements in the passage. The correct answer must be supported by direct textual evidence, and the design audit must pinpoint the exact paragraph or sentence anchor.

3. **Diagnostic Distractors (Structured Taxonomy)**:
   - All 4 options must be plausible, grammatically parallel, and closely tied to the passage topic. No option labels (A, B) or quotes around options.
   - ⚓ **ABSOLUTE SINGLE-FIT VALIDITY**: High diagnostic plausibility must NEVER create ambiguity. The question stem combined with the passage MUST provide definitive, objective textual evidence that makes the correct answer the ONLY defensible choice, while decisively eliminating all three distractors on factual, logical, or scope grounds without relying on subjective interpretation.
   - ❌ **STRICTLY PROHIBIT**: Absurd/cartoonish extremes (e.g., "ignore all warnings", "destroy the planet"), trivial common-sense giveaways, and lazy binary opposites.
   - Draw distractors from authentic reading traps:
     * *Trap 1 (Literal Match Trap)*: borrows verbatim words or phrasing from the passage, but twists the logical relationship, cause-and-effect, or subject/object.
     * *Trap 2 (Scope Shift Trap)*: overly broad, overly restrictive (extreme words like *always*, *only*, *solely*, *never*), or shifts the focus away from the question's premise.
     * *Trap 3 (Plausible Distortion / False Inference)*: sounds factually reasonable in real-world knowledge, but is unsupported, unmentioned, or directly contradicted by the text.

4. **Design Audit & Explanation**:
   - `design_audit`: Follow this rigorous 4-part structure:
     `AUDIT: [Skill Category: Main Idea/Detail/Inference/Tone] -> [Textual Anchor (Paragraph # / Specific Quote)] -> [Trap 1 (Literal Match): ...] [Trap 2 (Scope Shift): ...] [Trap 3 (Distortion): ...] -> [Why Distractors Fail: Objective Ground-Truth Disqualifications]`
   - `explanation`: State the exact text evidence for the correct answer, and contrastively explain why each distractor fails. You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.
   - 🎲 **RANDOMIZED ANSWER KEY BALANCE**: Distribute `correct_answer_index` evenly across 0 (A), 1 (B), 2 (C), and 3 (D) throughout the quiz. Never place all correct answers on the same index.

PASSAGE:
{passage_content}
