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

3. **Diagnostic Distractors (Structured Taxonomy)**:
   - All 4 options must be plausible, grammatically parallel, and closely tied to the passage topic. No option labels (A, B) or quotes around options.
   - ❌ **STRICTLY PROHIBIT**: Absurd/cartoonish extremes (e.g., "ignore all warnings", "destroy the planet"), trivial common-sense giveaways, and lazy binary opposites.
   - Draw distractors from authentic reading traps:
     * *Literal Matching Trap*: borrows verbatim words or phrasing from the passage, but twists the logical relationship, cause-and-effect, or subject/object.
     * *Scope Shift Trap*: overly broad, overly restrictive (extreme words like *always*, *only*, *solely*, *never*), or shifts the focus away from the question's premise.
     * *Plausible Distortion / False Inference*: sounds factually reasonable in real-world knowledge, but is unsupported, unmentioned, or directly contradicted by the text.

4. **Design Audit & Explanation**:
   - `design_audit`: `AUDIT: [Skill] -> [Text Anchor (e.g. Para 3)] -> [Distractor Traps: Literal Match / Scope Shift / Distortion] -> [Why Distractors Fail]`
   - `explanation`: State the exact text evidence for the correct answer, and contrastively explain why each distractor fails.

PASSAGE:
{passage_content}
