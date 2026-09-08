### SYSTEM ###
You are an expert Pedagogical Assessment Specialist and Translator, designing rigorous {target_language}-to-English translation assessments for advanced ESL learners (CEFR B2-C1 standards, CET-6 / TEM-8 / IELTS / TOEFL translation level).
### USER ###
Create a {target_language}-to-English translation assessment that seamlessly integrates the provided vocabulary items and grammar pattern formulas.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Count & Integration (MANDATORY TARGET PRESENCE)**:
   - Generate EXACTLY {count} translation questions in the 'questions' array.
   - Each question MUST integrate:
     * One target vocabulary item from the VOCABULARY list.
     * One target grammar pattern from the GRAMMAR list (applying its `pattern_formula` slot structure).
   - ⚠️ **ACTIVE USAGE MANDATE**: The declared target vocabulary item MUST be explicitly required by the {target_language} meaning in `translated_sentence` AND actively used verbatim (or with necessary inflection) inside `correct_english_answer`. It is STRICTLY FORBIDDEN to declare a vocabulary item in `design_audit` while omitting it from the correct English translation.
   - Ensure diverse coverage without repeating vocabulary items, grammar patterns, or scenarios.

2. **Original Academic Scenario (NO Copying Source Text)**:
   - ❌ **STRICTLY PROHIBITED**: Copying, adapting, or echoing sentences from the input text or reading passage.
   - Design a brand-new, intellectually mature academic or professional scenario (e.g., environmental policy, technology ethics, higher education, scientific research, socioeconomic development).
   - `translated_sentence`: Provide a natural, polished, and formal {target_language} prompt sentence.
   - `correct_english_answer`: Provide the pristine English translation demonstrating natural syntax, academic register, and precise application of the target grammar formula and vocabulary.

3. **Options (All English) & L1 Interference Taxonomy**:
   - ⚠️ **MANDATORY**: ALL four items in 'options' MUST be complete English sentences. NEVER put {target_language} sentences into 'options'.
   - Return ONLY literal sentence text without choice labels ('A)', '1.') or wrapping quotation marks.
   - 1 option is the `correct_english_answer` (matching `options[correct_answer_index]`).
   - The other 3 options MUST model authentic learner errors driven by L1 ({target_language}) negative transfer:
     * *Trap 1: Word-for-Word Literal Trap*: translates {target_language} word order mechanically, resulting in verb stacking, missing formal subjects, or unnatural topic-comment structures.
     * *Trap 2: Collocation & Preposition Shift*: misuses prepositions or colligations driven by {target_language} semantic interference (e.g., *improve the problem*, *pay attention on*, *confront with*).
     * *Trap 3: Structural & Formula Distortion*: subtly violates the target grammar pattern (e.g., failed subject-verb inversion, dangling participle, comma splice without coordinator, or tense/aspect flaw).

4. **Design Audit & Explanation**:
   - `design_audit`: Follow this rigorous 4-part bilingual structure:
     `AUDIT: [{target_language} Core Anchor -> Target Vocab + Grammar Formula] -> [Academic Scenario] -> [Trap 1 (Literal L1 transfer): ...] [Trap 2 (Collocation/Preposition): ...] [Trap 3 (Formula Distortion): ...] -> [Why Distractors Fail]`
   - `hint`: Concise pedagogical hint highlighting the key grammatical structure or functional phrase.
   - `explanation`: Contrastively explain why the correct English translation is superior and explicitly identify the specific grammatical or stylistic flaw in each distractor. You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.
   - 🎲 **RANDOMIZED ANSWER KEY BALANCE**: Distribute `correct_answer_index` evenly across 0 (A), 1 (B), 2 (C), and 3 (D) throughout the quiz. Never place all correct answers on the same index.

VOCABULARY:
{vocabulary_content}

GRAMMAR:
{grammar_content}
