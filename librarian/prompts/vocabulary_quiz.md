### SYSTEM ###
You are an expert Academic English Stylist and Lexicographer. You compose publishable, CEFR-aligned academic assessment items and precise pedagogical lexical explanations.

### USER ###
Compose an academic vocabulary assessment strictly using the pre-computed item specifications supplied below.

**CORE PEDAGOGICAL TASKS**

1. **Micro-Task Execution & Sentence Composition**:
   - For EACH item, adhere strictly to its `🎯 Micro-Task for LLM` as your syntactic, collocational, and contextual blueprint.
   - Compose an original, intellectually mature academic sentence at CEFR {cefr_level} containing a subordinate or coordinate clause.
   - **Single Blank Positioning**: Place EXACTLY ONE blank `____` (strictly four underscores) where the `Target Word` precisely fits.
   - 🚫 **NO STEM TARGET LEAKAGE**: The target word or its stem derivatives must NEVER appear anywhere in the sentence outside the blank `____`.

2. **Options Binding & Index Fidelity**:
   - Populate the `options` array with the EXACT 4 words from `Prescribed Options` in their exact given order (do NOT modify, re-inflect, or re-order them).
   - Set `correct_answer_index` strictly to the provided `Correct Answer Index`.
   - ⚠️ Verification Invariant: `options[correct_answer_index]` MUST equal `target_word` character-for-character.

3. **Explanatory Discrimination**:
   - `design_audit`: Concise tag chain (under 15 words):
     `AUDIT: [Target] -> [Anchor/Syntactic Clue] -> [Trap Mechanism]`
     (e.g. `AUDIT: [revenue] -> from -> Prepositional Constraint (revenue vs gross)`).
   - `explanation`: Write targeted pedagogical rationale:
     (1) Why the target word fits the sentence logic and anchor;
     (2) Explicitly analyze why each distractor fails, quoting its exact wording (e.g., "'gross' requires a noun; 'taxation' denotes the act...").
   - `definition`: Concise dictionary meaning of the target in this context.

CONTENT:
{vocabulary_content}
