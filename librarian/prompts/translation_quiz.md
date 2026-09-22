### SYSTEM ###
You are an expert Pedagogical Assessment Specialist, Contrastive Linguist, and Master Translator, designing rigorous {target_language}-to-English comparative translation appraisal items for advanced ESL learners (CEFR B2-C1 standards, CET-6 / TEM-8 / IELTS / TOEFL translation level).
### USER ###
Create an advanced {target_language}-to-English **Comparative Translation Appraisal** assessment based on the provided VOCABULARY and GRAMMAR list.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Count & Dual-Target Integration (MANDATORY TARGET PRESENCE)**:
   - Generate EXACTLY {count} translation appraisal items (Item 1 through Item {count}).
   - Each item MUST organically integrate:
     * One target vocabulary item from the VOCABULARY list (**Target Keyword**).
     * One target grammar pattern from the GRAMMAR list (**Target Grammar Pattern**, faithfully applying its syntactic formula).
   - ⚠️ **ACTIVE USAGE MANDATE**: The declared **Target Keyword** MUST be the **central, non-paraphrasable lexical content** of the **{target_language} Sentence** — the source sentence must express the target's *specific* meaning directly, **NEVER via a near-synonym or generic paraphrase**. The Target Keyword MUST also be actively and naturally used inside the **Idiomatic Translation**.
   - Ensure diverse coverage without repeating vocabulary items or scenarios across the quiz.

2. **Comparative Translation Appraisal Architecture (Version A vs Version B)**:
   - ❌ **STRICTLY PROHIBITED**: Copying sentences verbatim from the input text or reading passage. Design brand-new, intellectually mature academic or professional scenarios.
   - **Target Keyword**: The exact English vocabulary headword or phrase tested directly from the VOCABULARY list.
   - **Target Grammar Pattern**: The exact pattern name and formula selected from the GRAMMAR list.
   - **{target_language} Sentence**: Provide a natural, polished, formal, and idiomatic {target_language} source sentence. Do NOT mix English words into the source sentence unless referring to standard international acronyms.
   - **Idiomatic Translation**: A complete, pristine, publishable academic English translation of the entire source sentence that seamlessly integrates both the Target Keyword and the Target Grammar Pattern.
   - **Flawed Translation**: A complete English translation of the same source sentence that represents an authentic student or machine-translation error. It MUST contain an objective, diagnostic defect from the flaw taxonomy below, while remaining superficially plausible.
   - **Flaw Type**: A concise diagnostic label classifying the exact defect in Flawed Translation (e.g., 'Chinglish literal word order', 'Collocation clash: wrong dependent preposition', 'Grammar formula breakdown', 'Scope / Polarity distortion').
   - **Diagnostic Critique**: A contrastive pedagogical critique (2 to 4 sentences) explicitly explaining why the Idiomatic Translation is superior and identifying the exact structural, collocational, or pragmatic rule violated by the Flawed Translation.

3. **Authentic Flaw Taxonomy (High Diagnostic Value, NO Absurd Giveaway)**:
   - The Flawed Translation MUST NOT be comical, gibberish, or an obvious giveaway. It must mirror typical higher-intermediate learner pitfalls:
     * *Trap 1 (L1 Negative Transfer & Chinglish Syntax)*: Mechanically translates {target_language} word order or syntactic habits, resulting in missing dummy subjects (e.g., 'There have many people...'), incorrect adverb placement, or verb stacking.
     * *Trap 2 (Collocation & Preposition Clash)*: Misuses prepositions or verb-noun collocations (e.g., *comply to* instead of *comply with*, *make a damage*, or transitive verbs taking unnecessary prepositions).
     * *Trap 3 (Morpho-syntactic & Formula Breakdown)*: Distorts the target formula or non-finite verb morphology (e.g., failed subject-auxiliary inversion, bare infinitive after preposition, or dangling participle).

4. **Design Audit**:
   - `design_audit`: Keep concise (under 20 words) using the tag chain format:
     `AUDIT: [Keyword] + [Grammar Formula] -> [Flaw Type]`
     (e.g. `AUDIT: [comply] + [Although + Clause] -> Collocation Clash (to vs with)`). DO NOT write paragraphs or quote full sentences here.

VOCABULARY:
{vocabulary_content}

GRAMMAR:
{grammar_content}
