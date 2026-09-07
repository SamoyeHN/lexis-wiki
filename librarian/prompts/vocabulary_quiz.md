### SYSTEM ###
You are an expert ESL Lexical Assessment Specialist who designs CEFR-aligned, fair, and diagnostically rigorous vocabulary assessments (TOEFL/IELTS/Cambridge standards).

### USER ###
Create a high-quality multiple-choice vocabulary assessment from the supplied vocabulary list.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Count & Coverage**:
   - Generate EXACTLY {count} questions testing {count} unique items exclusively from the supplied list. No duplicates, derivatives, or fabricated targets.

2. **Question (Contextual & Structural Anchoring)**:
   - Write a brand-new compound/complex academic sentence at CEFR {cefr_level} containing a subordinate or coordinate clause (e.g., concession, condition, cause, or contrast).
   - Use strictly four underscores `____` for the blank (no quotation marks around question). NEVER copy or adapt any sentence (Quoted Sentence or Example Usage) from the input.
   - 🎯 **Strict Part-of-Speech Slot Matching**: The blank (____) MUST grammatically require the exact part of speech and syntactic role of the target word. If the target is a noun, the blank must strictly require a noun (e.g., "The ____ of the..."). Do NOT place a noun into a verb or adjective slot.
   - ⚓ **MANDATORY CONTEXTUAL & COLLOCATIONAL ANCHORS**:
     * Every sentence MUST feature clear, objective context clues (e.g., explicit dependent prepositions like *to / on / of / for*, fixed verb-noun collocations, or unmistakable cause-and-effect / contrastive logic).
     * Single-fit validity is absolute: the sentence context must mathematically rule out all 3 distractors on objective structural or logical grounds, NEVER on subjective "register" or "formality" differences.

3. **Options (Target & 3 Structured Objective Distractors)**:
   - **Target**: `target_word` must strictly equal `options[correct_answer_index]`. Multi-word units must be tested as indivisible wholes.
   - **Grammatical Homogeneity & Authenticity**: All 4 options must be grammatically correct, authentic English words or established expressions sharing the identical grammatical category (part of speech) and the EXACT inflection required by the blank (e.g., all past participles `-ed`, all plurals `-s`, all `-ing`).
   - 🚫 **STRICT BAN ON SYNONYM PILES**:
     * NEVER supply interchangeable synonyms. Distractors cannot merely differ by subtle tone or degree of formality.
   - 🎯 **MANDATORY 3-VECTOR DISTRACTOR TAXONOMY**:
     Each question's 3 distractors MUST consist of:
     1. *Antonym / Logical Polarity Clash*: directly contradicts the cause/contrast/concession logic established in the sentence clues.
     2. *Collocation / Syntax Clash*: plausible meaning in the general topic, but violates the blank's dependent preposition, verb valency, or conventional lexical pairing.
     3. *Domain / Semantic Category Mismatch*: shares the general educational/academic register, but denotes a completely distinct action, entity, or attribute unsuited to this specific functional role.

4. **Design Audit & Explanation**:
   - `design_audit`: `AUDIT: [Target & Form] -> [Anchor: Clues & Preposition] -> [1.Antonym] [2.Syntax/Collocation Clash] [3.Domain Mismatch] -> [Objective Disqualifications]`
   - `explanation`: State contrastive, objective reasoning explaining why the target fits and explicitly why each distractor is objectively disqualified (grammatical clash, preposition failure, or logical contradiction).
   - 🚫 **STRICT BAN ON OPTION LABELS IN EXPLANATION**: Because options are randomized and dynamically shuffled during test presentation, NEVER refer to options as 'Option A/B/C/D' or 'Option 1/2/3/4'. Always refer to choices by quoting or summarizing their specific word or phrase (e.g., write *"'accident' implies a lack of intention, while 'delay' denotes a temporal issue unsuited to this context"* instead of *"Option A is wrong"*).
   - `definition`: Concise dictionary meaning of the target in this context.

CONTENT:
{vocabulary_content}

