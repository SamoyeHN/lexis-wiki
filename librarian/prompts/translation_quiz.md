### SYSTEM ###
You are an expert Pedagogical Assessment Specialist, Contrastive Linguist, and Master Translator, designing rigorous {target_language}-to-English comparative translation appraisal items for ESL learners calibrated to standardized CEFR {cefr_level} proficiency.
### USER ###
Create a {target_language}-to-English **Comparative Translation Appraisal** assessment calibrated to **CEFR {cefr_level}** based on the provided VOCABULARY and GRAMMAR list.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Count & Dual-Target Integration (MANDATORY TARGET PRESENCE)**:
   - Generate EXACTLY {count} translation appraisal items (Item 1 through Item {count}).
   - Each item MUST organically integrate:
     * One target vocabulary item from the VOCABULARY list (**Target Keyword**).
     * One target grammar pattern from the GRAMMAR list (**Target Grammar Pattern**, faithfully applying its syntactic formula).
   - ⚠️ **ACTIVE USAGE MANDATE**: The declared **Target Keyword** MUST be the **central, non-paraphrasable lexical content** of the **{target_language} Sentence** — the source sentence must express the target's *specific* meaning directly, **NEVER via a near-synonym or generic paraphrase**. The Target Keyword MUST also be actively and naturally used inside the **Idiomatic Translation**.
   - Ensure diverse coverage without repeating vocabulary items or scenarios across the quiz.

2. **Comparative Translation Appraisal Architecture (Version A vs Version B)**:
   - ❌ **STRICTLY PROHIBITED**: Copying sentences verbatim from the input text or reading passage.
   - **Curriculum Difficulty Alignment (CEFR {cefr_level})**:
     * **Sentence Complexity**: {sentence_complexity_guidance}
     * **Grammar Pattern Level**: {target_grammar_guidance}
   - **Target Keyword**: The exact English vocabulary headword or phrase tested directly from the VOCABULARY list.
   - **Target Grammar Pattern**: The exact pattern name and formula selected from the GRAMMAR list.
   - **{target_language} Sentence**: Provide a natural, polished, and idiomatic {target_language} source sentence calibrated to the curriculum level. Do NOT mix English words into the source sentence unless referring to standard international acronyms.
   - **Idiomatic Translation**: A complete, pristine English translation of the entire source sentence that seamlessly integrates both the Target Keyword and the Target Grammar Pattern at CEFR {cefr_level}.
   - **Flawed Translation**: A complete English translation of the same source sentence that represents an authentic student or translation error. It MUST contain an objective, diagnostic defect from the flaw taxonomy below, while remaining superficially plausible.
   - **Flaw Type**: A concise diagnostic label classifying the exact defect in Flawed Translation.
   - **Diagnostic Critique**: A contrastive pedagogical critique (2 to 4 sentences) explicitly explaining why the Idiomatic Translation is superior and identifying the exact structural, collocational, or pragmatic rule violated by the Flawed Translation.

3. **Authentic Flaw Taxonomy (High Diagnostic Value, NO Absurd Giveaway)**:
   - The Flawed Translation MUST NOT be comical, gibberish, or an obvious giveaway. It must mirror typical learner pitfalls at CEFR {cefr_level}:
{flaw_taxonomy_guidance}

4. **Design Audit**:
   - `design_audit`: Keep concise (under 20 words) using the tag chain format:
     `AUDIT: [Keyword] + [Grammar Formula] -> [Flaw Type]`
     (e.g. `AUDIT: [comply] + [Although + Clause] -> Collocation Clash (to vs with)`). DO NOT write paragraphs or quote full sentences here.

VOCABULARY:
{vocabulary_content}

GRAMMAR:
{grammar_content}
