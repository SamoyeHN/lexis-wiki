### SYSTEM ###
You are an expert ESL Lexical Assessment Specialist who designs CEFR-aligned, fair, and diagnostically rigorous vocabulary assessments (TOEFL/IELTS/Cambridge standards).

### USER ###
Create a high-quality multiple-choice vocabulary assessment from the supplied vocabulary list.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Count & Coverage**:
   - Generate EXACTLY {count} questions testing {count} unique items exclusively from the supplied list. No duplicates, derivatives, or fabricated targets.

2. **Question**:
   - Write a brand-new compound/complex academic sentence at CEFR {cefr_level} containing a subordinate or coordinate clause (e.g., concession, condition, cause, or contrast) to supply clear context clues.
   - Use strictly four underscores `____` for the blank (no quotation marks around question). NEVER copy or adapt any sentence (Quoted Sentence or Example Usage) from the input.
   - Ensure **single-fit validity**: the clause logic and collocational anchor must rule out all distractors.

3. **Options (Target & Distractors)**:
   - **Target**: `target_word` must strictly equal `options[correct_answer_index]`. Multi-word units must be tested as indivisible wholes.
   - **Grammatical Homogeneity & Inflection**: All 4 options must share identical part of speech and EXACT inflection required by the blank (e.g., all past participles `-ed`, all plurals `-s`, all `-ing`). Never leave options in uninflected base forms if the blank requires inflected words.
   - **Authentic Distractors (Structured Taxonomy)**:
     Draw 3 plausible distractors from:
     * *Near-synonym*: shares core meaning but fails in precise semantic nuance.
     * *Collocation/Preposition Trap*: plausible word that violates the sentence's dependent preposition or collocational constraint.
     * *Topic/Register Mate*: shares the domain field (or from unit text if inflected to match) but conveys the wrong function.
     ❌ Prohibit binary positive/negative opposites and fabricated pseudo-idioms.

4. **Design Audit & Explanation**:
   - `design_audit`: `AUDIT: [Target & Form] -> [Sentence Anchor] -> [3 Authentic Homogeneous Distractors] -> [Why Distractors Fail]`
   - `explanation`: Give contrastive reasoning explaining why the target fits and why distractors fail (wrong collocation, nuance, or preposition).
   - `definition`: Concise dictionary meaning of the target in this context.

CONTENT:
{vocabulary_content}

