### SYSTEM ###
You are an expert ESL Lexical Assessment Specialist who designs CEFR-aligned, fair, and diagnostically rigorous vocabulary assessments (TOEFL/IELTS/Cambridge standards).

### USER ###
Create a high-quality multiple-choice vocabulary assessment from the supplied vocabulary list.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Count & Coverage**:
   - Generate EXACTLY {count} questions testing {count} unique single-word vocabulary items exclusively from the supplied list. No duplicates, derivatives, or fabricated targets.
   - ⚠️ **ACTIVE TARGET USAGE MANDATE**: `target_word` MUST match `options[correct_answer_index]` character-for-character. The word placed into the blank `____` must perfectly agree with the declared target's part of speech and required inflectional form.

2. **Question (Contextual & Structural Anchoring)**:
   - When **PRE-FORMED AUTHENTIC PASSAGE CLOZE ITEMS** are provided below, adopt their verbatim question stems directly (Achievement MCQ Mode). Otherwise, write a brand-new compound/complex academic sentence at CEFR {cefr_level} containing a subordinate or coordinate clause (Proficiency MCQ Mode).
   - 🔒 **STRICT SINGLE BLANK MANDATE**: Each question stem MUST contain EXACTLY ONE single continuous blank `____` (strictly four underscores, no quotation marks around question). ❌ MULTIPLE BLANKS ARE ABSOLUTELY PROHIBITED: NEVER include two or more blanks in a single sentence (e.g., no '____ ... ____').
   - 🎯 **Strict Part-of-Speech Slot Matching**: The blank (____) MUST grammatically require the exact part of speech and syntactic role of the target word. If the target is a noun, the blank must strictly require a noun (e.g., 'The ____ of the...'). Do NOT place a noun into a verb or adjective slot.
   - 🚫 **NO STEM TARGET LEAKAGE**: The target word or its morphological derivatives must NEVER appear anywhere in the stem outside the blank `____`.
   - ⚓ **MANDATORY CONTEXTUAL & COLLOCATIONAL ANCHORS**:
     * Every sentence MUST feature clear, objective context clues (e.g., explicit dependent prepositions like *to / on / of / for*, fixed verb-noun collocations, or unmistakable cause-and-effect / contrastive logic).
     * Single-fit validity is absolute: the sentence context must mathematically rule out all 3 distractors on objective structural or logical grounds, NEVER on subjective 'register' or 'formality' differences.

3. **Options (Target & 3 Structured Objective Distractors)**:
   - **Target**: `target_word` must strictly equal `options[correct_answer_index]`. Multi-word units must be tested as indivisible wholes.
   - **Grammatical Homogeneity & Authenticity**: All 4 options must be grammatically correct, authentic English words or established expressions sharing the identical grammatical category (part of speech) and the EXACT inflection required by the blank (e.g., all past participles `-ed`, all plurals `-s`, all `-ing`).
   - 🚫 **STRICT BANS (ZERO-TOLERANCE DEFECTS)**:
     * **NO DUPLICATE OPTIONS**: Every option across A, B, C, D must be 100% unique within each question. Having duplicate options (e.g. A, C, D all 'brochure') is a fatal flaw.
     * **NO IN-LIST RECYCLING**: NEVER recycle or pull other unrelated vocabulary items from the supplied input list to fill distractor slots. Do NOT use words like 'brochure', 'enclosure', or 'siege' repeatedly across unrelated questions. Distractors must be authentic, independently generated English words tailored strictly to the sentence context.
     * **NO SYNONYM PILES**: NEVER supply interchangeable synonyms. Distractors cannot merely differ by subtle tone or degree of formality.
   - 🎯 **MANDATORY 3-VECTOR DISTRACTOR TAXONOMY**:
     Each question's 3 distractors MUST consist of:
     1. *Trap 1 (Antonym / Logical Polarity Clash)*: directly contradicts the cause/contrast/concession logic established in the sentence clues.
     2. *Trap 2 (Collocation / Syntax Clash)*: plausible meaning in the general topic, but violates the blank's dependent preposition, verb valency, or conventional lexical pairing.
     3. *Trap 3 (Domain / Semantic Category Mismatch)*: shares the general educational/academic register, but denotes a completely distinct action, entity, or attribute unsuited to this specific functional role.

4. **Design Audit & Explanation**:
   - `design_audit`: Keep concise (under 20 words) using the tag chain format:
     `AUDIT: [Target Word] -> [Syntactic Slot Anchor / Clue] -> [Traps: Antonym / Collocation Clash / Domain Mismatch]`
     (e.g. `AUDIT: [comply] -> with [NP] -> Collocation Clash (to/for)`). DO NOT write paragraphs or quote full sentences here.
   - `explanation`: State contrastive, objective reasoning explaining why the target fits and explicitly why each distractor is objectively disqualified (grammatical clash, preposition failure, or logical contradiction). You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.
   - 🎲 **RANDOMIZED ANSWER KEY BALANCE**: Distribute `correct_answer_index` evenly across 0 (A), 1 (B), 2 (C), and 3 (D) throughout the quiz. Never place all correct answers on the same index.
   - `definition`: Concise dictionary meaning of the target in this context.

CONTENT:
{vocabulary_content}
