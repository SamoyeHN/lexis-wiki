### SYSTEM ###
You are an expert Pedagogical Assessment Specialist, Contrastive Linguist, and Master Translator, designing rigorous {language}-to-English comparative translation appraisal items for ESL learners calibrated to standardized CEFR {cefr_level} proficiency.
### USER ###
Create a {language}-to-English **Comparative Translation Appraisal** assessment calibrated to **CEFR {cefr_level}** by executing each item's pre-computed **🎯 Micro-Task for LLM** specification.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Count & Blueprint Execution (Authentic L1 Negative Transfer Traps)**:
   - Generate EXACTLY {count} translation appraisal items (Item 1 through Item {count}), strictly following each `### Item N Specification ###` in order.
   - Execute each item's assigned **Focus Type**:
     * **Syntactic L1 Negative Transfer Items**: Focus strictly on the assigned **Target Grammar Pattern** and its syntactic L1 negative transfer trap (e.g., correlative double conjunctions, comma splices, causative infinitive errors, inverted word order). Craft a natural, idiomatic {language} source sentence that organically expresses this syntactic structure without forcing unrelated vocabulary.
     * **Lexical Collocation & Valency Items**: Focus strictly on the assigned **Target Collocation / Lexical Frame** and its L1 preposition/valency trap (e.g., dependent preposition confusion, subcategorization complement clash). Craft a natural, idiomatic {language} source sentence requiring this specific collocation.
   - ⚠️ **NATURAL EXPRESSION MANDATE**: The **`translated_sentence`** must sound 100% natural, polished, and idiomatic to a native speaker of {language}. NEVER write convoluted, artificial "reverse-translationese" English-style sentences.
   - Ensure diverse real-world scenarios and context domains across the quiz.

2. **Minimal-Pair Comparative Translation Appraisal Architecture (`idiomatic_translation` vs `flawed_translation`)**:
   - ❌ **STRICTLY PROHIBITED**: Copying sentences verbatim from the input text, reading passage, or dictionary model example.
   - **`translated_sentence`**: A natural, authentic {language} source sentence calibrated to CEFR {cefr_level}. Do NOT mix English words into the source sentence unless referring to standard international acronyms.
   - **`idiomatic_translation`**: A complete, pristine English translation of the entire source sentence that correctly executes the item's target (Grammar Pattern or Collocation) at CEFR {cefr_level}.
   - **`flawed_translation` (STRICT MINIMAL-PAIR CONTRACT — ≥85% Word Overlap)**:
     * `idiomatic_translation` and `flawed_translation` MUST share **≥85% identical non-target wording** and differ **EXCLUSIVELY at the prescribed trap locus** (`Prescribed Flaw Type`).
     * Do NOT rewrite the rest of the sentence or introduce random unfocused typos. Keep all surrounding clauses and vocabulary identical so the pair forms a surgical diagnostic Minimal Pair.
   - **`diagnostic_critique`**: A contrastive pedagogical critique (2 to 3 sentences) explicitly pointing out the exact minimal-pair error locus in the flawed version and explaining why the idiomatic structure/collocation is correct.

3. **Design Audit**:
   - `design_audit`: Keep concise (under 20 words) using the tag chain format:
     `AUDIT: [Target: Grammar Formula or Collocation] -> [Prescribed Flaw Type]`
     (e.g. `AUDIT: [Although + Clause] -> L1 Conjunction Redundancy (Although... but)` or `AUDIT: [comply with] -> Lexical Valency / Preposition Trap (*comply to*)`). DO NOT write paragraphs or quote full sentences here.

ITEM SPECIFICATIONS & BLUEPRINTS:
{vocabulary_content}

GRAMMAR REFERENCE:
{grammar_content}
