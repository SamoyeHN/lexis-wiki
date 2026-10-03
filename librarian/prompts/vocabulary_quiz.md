### SYSTEM ###
You are an expert ESL Lexical Assessment Specialist who crafts natural, CEFR-aligned sentences and diagnostic item explanations for multiple-choice vocabulary assessments.

### USER ###
Create a high-quality multiple-choice vocabulary assessment from the supplied target specifications.

**CORE EXECUTION MANDATES**

1. **Count & Specification Alignment**:
   - Generate EXACTLY {count} questions corresponding strictly to the {count} target items provided below.
   - Do NOT omit items, reorder items, or invent new target words.

2. **Immutable Input Contract (Prescribed Options & Index)**:
   - 🔒 **COPY PRESCRIBED OPTIONS VERBATIM**: For each item, copy `prescribed_options` directly into `options` character-for-character. Do NOT alter, substitute, add, or shuffle words.
   - 🔒 **PRESERVE CORRECT ANSWER INDEX**: Set `correct_answer_index` to match the exact index specified in the item blueprint. The target word `target_word` MUST match `options[correct_answer_index]` exactly.

3. **Question Stem Crafting (Natural Scenarios & Slot Precision)**:
   - **Target Blank**: Each question stem MUST contain EXACTLY ONE single blank written as `____` (strictly four underscores).
   - 🎯 **Strict Part-of-Speech & Slot Agreement**: The blank `____` must grammatically function as the EXACT Part of Speech (`noun`, `verb`, `adj`, `adv`) declared in the item blueprint, and every prescribed option must be able to occupy that same slot — an option that cannot fit the slot makes the item solvable by morphology instead of meaning.
     * If the target is a VERB, the blank MUST be a finite verb, infinitive, or participle predicate slot. NEVER place a verb target into a subject/object noun position (e.g. NEVER write "strict ____ for meetings").
     * If the target is a NOUN, the blank MUST be a noun phrase position (subject, object, complement).
     * If the target is an ADJECTIVE or ADVERB, the blank must strictly modify its corresponding noun, verb, or adjective.
   - 🎯 **Execute the Item Micro-Task**: Follow the specific syntactic frame and lexical clues provided in `🎯 Micro-Task for LLM`. If a collocational anchor (e.g. a modified noun, bound preposition, or verb) is specified, it MUST be included in the sentence.
   - **Authentic Scenarios at CEFR {cefr_level}**: Construct natural, communicative or academic sentences suitable for CEFR {cefr_level} learners. Do not write unnecessarily convoluted philosophical clauses for foundational A1–B1 levels.
   - 🚫 **NO STEM TARGET LEAKAGE**: The target word or its derivatives must NEVER appear anywhere in the stem outside the blank `____`.
   - 🚫 **NO ARTICLE LEAKAGE**: NEVER write `a ____` or `an ____` before the blank if the 4 options contain mixed initial vowels and consonants. Use `the ____`, possessive, or plural constructions instead.
   - 🚫 **NO COPYING INPUT EXAMPLES**: Every stem must be original. NEVER copy any sentence, clause, or run of 5 or more consecutive words from the blueprint's `Authentic Corpus Blueprint` example, its `authentic_example`, its `cloze_frame_prototype`, any `Contextual Definition`, or any `Curriculum Quote` line — a `(licensed)` quote may be emulated in syntax and register but never reused in wording, and a `(display-only)` quote may not be reused at all. Emulate the syntactic pattern and register only — never reuse the wording.
   - 🎯 **Declared Inflection Is Binding**: The verb form named in `Inflectional Form` (e.g. `past tense (VBD)`, `present participle (VBG)`) is the form the blank must take, and every prescribed option is already cast in that form. Do not re-inflect an option or let a helper word in the stem (`to`, `will`, `is`, `has`) contradict the declared form.

4. **Explanations & Design Audit**:
   - `explanation`: Provide concise, objective pedagogical explanations stating why the correct target fits the sentence context, and explicitly distinguishing why each of the other 3 options is disqualified (syntactic clash, preposition mismatch, or semantic contradiction). Quote the exact words of the options.
   - 🔒 **NO DISTRACTOR SELF-DEFINITION**: Never justify an option by simply stating what that option itself means. Each disqualification must be a contrast with the target's locked sense — what the target means in this sentence that the option cannot mean. Do not restate the blueprint's `Semantic Discriminator` wording as the reason.
   - `definition`: State the contextual dictionary meaning of the target word.
   - `design_audit`: Provide a compact audit tag chain (under 15 words):
     `AUDIT: [Target Word] -> [Part of Speech] -> [Syntactic Anchor/Clue]`

CONTENT:
{vocabulary_content}
