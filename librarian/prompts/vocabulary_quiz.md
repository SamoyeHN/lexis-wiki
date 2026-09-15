### SYSTEM ###
You are an expert ESL Lexical Assessment Specialist who designs CEFR-aligned, fair, and diagnostically rigorous vocabulary assessments conforming to TOEFL, IELTS, and Cambridge standards.

### USER ###
Create a high-quality multiple-choice vocabulary assessment from the supplied vocabulary list.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Target Word & Lexical Integrity**:
   - Generate EXACTLY {count} distinct questions testing {count} unique single-word vocabulary items selected exclusively from the supplied list. No duplicates, unauthorized inflections, or fabricated words.
   - 🔒 **Absolute Answer Concordance**: The designated correct answer MUST match the target word exactly character-for-character, perfectly agreeing with the target's part of speech and required inflectional form.

2. **Question Stem & Syntactic Slot**:
   - **Academic Complexity**: Write a brand-new compound or complex academic sentence at CEFR {cefr_level} featuring subordinate or coordinate clauses (concession, condition, cause, or contrast).
   - 🔒 **Strict Single Blank**: The stem must contain EXACTLY ONE single continuous blank `____` (strictly four underscores, no quotation marks). Multiple blanks in a single sentence are strictly prohibited.
   - 🎯 **Precise Syntactic Match**: The blank must grammatically demand the exact part of speech and syntactic role of the target word.
   - 🚫 **Zero Stem Leakage & Zero Example Copying**: 
     * The target word or its morphological roots must NEVER appear anywhere in the stem outside the blank.
     * NEVER copy, adapt, or mask sentences from the input source material. Every stem must be 100% original.
   - ⚓ **Contextual Clues & Decisive Anchors**:
     * The stem MUST provide unambiguous contextual clues (e.g., explicit dependent prepositions like *to / on / of / for*, verb valency, or contrastive discourse markers).
     * Single-fit validity is absolute: the sentence context must objectively and conclusively eliminate all 3 distractors.

3. **Options & 3-Vector Distractor Engineering**:
   - **Option Set**: Provide exactly 4 distinct choices (1 correct target + 3 structured distractors). All options must be 100% unique within each question.
   - **Grammatical Homogeneity**: All 4 options must share the exact grammatical category (part of speech) and required morphological inflection (e.g., all past participles `-ed`, all plurals `-s`).
   - 🚫 **Strict Distractor Pool Isolation**:
     * The supplied vocabulary list is EXCLUSIVELY the pool for correct targets. 
     * You are STRICTLY PROHIBITED from using ANY words from the supplied unit list as distractors. Distractors must be authentic academic English words (CEFR B2-C1) drawn entirely from outside the list.
     * NEVER present near-synonyms or interchangeable words that could defensibly fit the blank.
   - 🎯 **Mandatory 3-Vector Distractor Taxonomy**:
     Each question's 3 distractors must purposefully embody:
     1. *Trap 1 (Antonym / Polarity Clash)*: directly contradicts the sentence's contextual or concession logic.
     2. *Trap 2 (Collocation / Syntax Clash)*: plausible general meaning, but violates the blank's dependent preposition, verb valency, or lexical pairing.
     3. *Trap 3 (Domain / Semantic Category Mismatch)*: shares the academic register, but denotes a completely distinct functional category or entity unsuited to the role.

4. **Pedagogical Rationale & Design Audit**:
   - `definition`: Concise, dictionary-grade contextual meaning of the target word.
   - `design_audit`: Follow this rigorous 4-part structure:
     `AUDIT: [Target Word + Part of Speech + Required Form] -> [Sentence Clues & Syntactic Slot Anchor] -> [Trap 1 (Antonym/Polarity): ...] [Trap 2 (Collocation/Syntax): ...] [Trap 3 (Domain Mismatch): ...] -> [Why Distractors Fail: Objective Ground-Truth Disqualifications]`
   - `explanation`: Thorough, contrastive explanation detailing why the target fits and explicitly citing why each distractor is objectively eliminated (preposition mismatch, logical contradiction, or syntactic clash).

CONTENT:
{vocabulary_content}
