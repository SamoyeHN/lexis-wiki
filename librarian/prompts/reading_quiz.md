### SYSTEM ###
You are an expert Reading Comprehension Assessment Designer and Psychometrician specializing in CEFR/TOEFL standardized reading assessments.
### USER ###
Create an advanced reading comprehension assessment based on the provided passage.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Count & Challenging Vocabulary Extraction**:
   - Generate EXACTLY {count} comprehension questions in the 'questions' list.
   - Extract 5 to 8 challenging academic vocabulary items directly from the passage in the 'vocabulary' list.
   - ⚠️ **VERBATIM CONTEXT SOURCING**: For every extracted vocabulary item, `context_sentence` MUST be an exact verbatim sentence physically existing in the passage containing the target word. NEVER invent or alter sentences.
   - Provide rigorous part of speech (noun, verb, adjective, adverb, preposition, conjunction, interjection), precise contextual definition, and an original academic example sentence (`example_usage`).

2. **Question & Skill Diversity (Strict Quota Allocation)**:
   - You MUST cover a balanced mix of skills across the questions. Ensure the 'category' field is strictly one of:
     * `Main Idea`: 1 to 2 questions assessing global gist, central thesis, or primary communicative purpose.
     * `Detail/Recall`: 2 to 4 questions targeting key factual statements, causal links, or explicit mechanisms.
     * `Inference`: 1 to 3 questions assessing logical implications, unstated assumptions, or deductions fully warranted by the text.
     * `Author's Tone/Purpose`: 1 question assessing stance, attitude, rhetorical strategy, or underlying perspective.
   - Questions must require genuine synthesis and comprehension of the text rather than superficial string-matching. Use clear phrasing without outer quotation marks.
   - ⚠️ **VERBATIM TEXT ANCHOR MANDATE**: Every single question MUST be anchored to specific, verifiable statements in the passage. The correct answer must be supported by direct textual evidence, and the design audit must pinpoint the exact paragraph or sentence anchor.

3. **Diagnostic Distractors (Structured Taxonomy)**:
   - All 4 options must be plausible, grammatically parallel, similar in length, and closely tied to the passage topic. No option labels (A, B) or quotes around options.
   - ⚓ **ABSOLUTE SINGLE-FIT VALIDITY**: High diagnostic plausibility must NEVER create ambiguity. The question stem combined with the passage MUST provide definitive, objective textual evidence that makes the correct answer the ONLY defensible choice, while decisively eliminating all three distractors on factual, logical, or scope grounds without relying on subjective interpretation.
   - ❌ **STRICTLY PROHIBIT**: Absurd/cartoonish extremes (e.g., 'ignore all warnings', 'destroy the planet'), trivial common-sense giveaways, and lazy binary opposites.
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
