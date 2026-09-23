### SYSTEM ###
You are an expert Academic English Stylist and Lexicographer. You compose publishable, CEFR-aligned academic sentences and precise pedagogical lexical explanations.

### USER ###
Compose an academic vocabulary assessment strictly using the pre-computed item specifications supplied below.

**CORE PEDAGOGICAL TASKS**

1. **Sentence Composition (Contextualizing Target & Anchor)**:
   - For EACH pre-computed specification, compose an original, intellectually mature academic sentence at CEFR {cefr_level} containing a subordinate or coordinate clause.
   - **Collocational Grounding**: Naturally contextualize the `Target Word` with its specified `Collocational Anchor` (e.g. if Target is *lay* and Anchor is *foundation*, weave *lay the foundation* into the sentence).
   - **Single Blank Target Positioning**:
     * Construct the sentence such that **the target word or its grammatically inflected form** occupies EXACTLY ONE blank `____` (strictly four underscores).
     * The blank must strictly require the target's syntactic function and part of speech.
     * 💡 **Inflectional Coherence**: You may use either the prescribed base form or an inflected variant (e.g. past tense `-ed`, plural `-s`, participle `-ing`) to suit the sentence frame. Whichever form fills the blank, ensure all 4 options share the exact same inflectional form so that `options[correct_answer_index]` matches the word required by the blank character-for-character.
   - **Clue Engineering**: Provide clear syntactic or semantic clues (e.g. dependent prepositions, contrastive logic) so that the target fits impeccably.
   - 🚫 **NO STEM TARGET LEAKAGE**: The target word or its derivatives must NEVER appear anywhere in the stem outside the blank `____`.

2. **Options Binding & Randomization**:
   - For EACH item, populate the `options` array with the 4 words from `Prescribed Options` (applying parallel inflection if required by the sentence frame).
   - Shuffle or randomize the placement of the target word so that `correct_answer_index` (0, 1, 2, or 3) is naturally balanced across the quiz.
   - ⚠️ `target_word` MUST match `options[correct_answer_index]` character-for-character.

3. **Explanatory Discrimination**:
   - `design_audit`: Concise tag chain (under 15 words):
     `AUDIT: [Target] -> [Anchor/Syntactic Clue] -> [Trap Mechanism]`
     (e.g. `AUDIT: [abolish] -> ceiling -> Collocation Clash (cancel vs abolish)`).
   - `explanation`: Write targeted pedagogical rationale explaining:
     (1) Why the target word and anchor collocation correctly fulfill the sentence logic;
     (2) Explicitly why each of the other 3 prescribed options fails in this specific sentence context (e.g., preposition clash, semantic ill-fit, or unnatural collocation).
   - `definition`: Concise dictionary meaning of the target in this context.

CONTENT:
{vocabulary_content}
