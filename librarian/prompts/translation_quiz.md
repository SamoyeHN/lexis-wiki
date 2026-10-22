### SYSTEM ###
You are an expert Pedagogical Assessment Specialist, Contrastive Linguist, and Master Translator, designing rigorous {target_language}-to-English comparative translation appraisal items for ESL learners calibrated to standardized CEFR {cefr_level} proficiency.
### USER ###
Create a {target_language}-to-English **Comparative Translation Appraisal** assessment calibrated to **CEFR {cefr_level}** by executing each item's pre-computed **🎯 Micro-Task for LLM** specification.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Count & Blueprint Execution (MANDATORY TARGET PRESENCE)**:
   - Generate EXACTLY {count} translation appraisal items (Item 1 through Item {count}), following each `### Item N Specification ###` in order.
   - Each item MUST organically integrate:
     * The assigned **Target Keyword** and its **Required Lexical Frame** from the blueprint (or VOCABULARY list).
     * The assigned **Target Grammar Pattern** from the blueprint (or GRAMMAR list), faithfully applying its syntactic formula.
   - ⚠️ **ACTIVE USAGE MANDATE**: The declared **Target Keyword** MUST be the **central, non-paraphrasable lexical content** of the **{target_language} Sentence** — the source sentence must express the target's *specific* meaning directly, **NEVER via a near-synonym or generic paraphrase**. The Target Keyword MUST also be actively and naturally used inside the **Idiomatic Translation**.
   - Ensure diverse coverage without repeating vocabulary items or scenarios across the quiz.

2. **Minimal-Pair Comparative Translation Appraisal Architecture (`idiomatic_translation` vs `flawed_translation`)**:
   - ❌ **STRICTLY PROHIBITED**: Copying sentences verbatim from the input text, reading passage, or dictionary model example.
   - **Curriculum Difficulty Alignment (CEFR {cefr_level})**:
     * **Sentence Complexity**: {sentence_complexity_guidance}
     * **Grammar Pattern Level**: {target_grammar_guidance}
   - **`translated_sentence`**: Provide a natural, polished, and idiomatic {target_language} source sentence calibrated to CEFR {cefr_level} that requires both the Target Keyword's meaning and the Target Grammar Pattern. Do NOT mix English words into the source sentence unless referring to standard international acronyms.
   - **`idiomatic_translation`**: A complete, pristine English translation of the entire source sentence that seamlessly integrates both the Target Keyword (in its Required Lexical Frame) and the Target Grammar Pattern at CEFR {cefr_level}.
   - **`flawed_translation` (STRICT MINIMAL-PAIR CONTRACT — ≥85% Word Overlap)**:
     * `idiomatic_translation` and `flawed_translation` MUST share **≥85% identical non-target wording** and differ **EXCLUSIVELY at the prescribed trap locus** (`Prescribed Flaw Type`).
     * Do NOT rewrite the rest of the sentence or introduce random unfocused typos. Keep all surrounding clauses and vocabulary identical so the pair forms a surgical diagnostic Minimal Pair.
   - **`diagnostic_critique`**: A contrastive pedagogical critique (2 to 3 sentences) explicitly pointing out the exact minimal-pair error locus in the flawed version and confirming the correct lexical frame and syntactic structure.

3. **Authentic Single-Point Flaw Taxonomy (High Diagnostic Value, NO Absurd Giveaway)**:
   - The `flawed_translation` MUST NOT be comical, gibberish, or an obvious giveaway. Inject ONLY the item's `Prescribed Flaw Type` (or, if none is specified, one authentic trap below):
{flaw_taxonomy_guidance}

4. **Design Audit**:
   - `design_audit`: Keep concise (under 20 words) using the tag chain format:
     `AUDIT: [Keyword] + [Grammar Formula] -> [Flaw Type]`
     (e.g. `AUDIT: [originate] + [Although + Clause] -> L1 Conjunction Redundancy (Although... but)`). DO NOT write paragraphs or quote full sentences here.

VOCABULARY & ITEM BLUEPRINTS:
{vocabulary_content}

GRAMMAR REFERENCE:
{grammar_content}
