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
   - 🎯 **Strict Part-of-Speech & Slot Agreement**: The blank `____` must grammatically function as the EXACT Part of Speech (`noun`, `verb`, `adj`, `adv`, `phrase`) and inflectional form declared in the item blueprint, and every prescribed option must fit that exact slot:
     * **Verbs & Verbal Phrases**: Match the exact declared inflection (`base form`, `past tense`, `-s` form, `past participle`, or `-ing` form). Never place a verb target into a subject/object noun position (e.g. NEVER write `"strict ____ for meetings"` or `"the unauthorized ____ of funds"`).
     * **Nouns & Determiners**: Place `____` in a noun phrase position matching the target's countability and number. Never place indefinite `'a'`, `'an'`, `'a(n)'`, or `'a/an'` directly before `____` when `prescribed_options` mix vowel and consonant initials — use `'the'`, `'each'`, or a possessive instead.
     * **Adjectives & Adverbs**: For predicate adjectives (`... are ____ to`, `... fell ____`), include the linking verb so the clause is complete; for attributive-only adjectives (`[only before noun]`), place `____` directly before the modified noun.
   - 🎯 **Execute `Required Stem Frame` & Instantiate Bracketed Slots**:
     * Preserve the surrounding collocational anchors, prepositions, and auxiliaries in `Required Stem Frame` (e.g. `... suspected of ____`, `... repay a ____`, `... ____ in`) explicitly next to `____` — NEVER swallow the governing verb, preposition, auxiliary, or article into `____`.
     * When `Required Stem Frame` contains a bracketed slot — such as `[concrete physical object, e.g. the lid]`, `[concrete physical object/animal, e.g. the elephants]`, `[verb]`, or `[noun]` — replace the entire bracketed placeholder with a natural word or concrete noun phrase (using one of the `e.g.` exemplars or a comparable physical noun), and choose the matching singular or plural auxiliary (`was` vs. `were`, `is` vs. `are`, `has` vs. `have`) to agree with your chosen subject.
   - **Authentic Scenarios at CEFR {cefr_level}**: Construct natural, communicative or academic sentences suitable for CEFR {cefr_level} learners. Do not write unnecessarily convoluted philosophical clauses for foundational A1–B1 levels.
   - 🚫 **NO STEM TARGET LEAKAGE**: The target word or its root cognates must NEVER appear anywhere in the stem outside the blank `____`.
   - 🚫 **NO COPYING INPUT EXAMPLES OR MODEL SENTENCES**: Every stem must be original. NEVER copy any sentence, clause, or run of 4 or more consecutive words from the blueprint's `Model: "..."` example, its `Authentic Corpus Blueprint` example, its `authentic_example`, its `cloze_frame_prototype`, any `Contextual Definition`, or any `Curriculum Quote` line. The `Model:` example illustrates syntax only — NEVER copy its sentence structure directly or let its singular/plural morphology alter the declared target inflection.
   - 🚫 **NO DEFINITION PARAPHRASING IN STEMS**: Write a situational context sentence rather than copying the `Contextual Definition` wording verbatim into the stem. You ARE encouraged to preserve the exact collocations, auxiliaries, and physical noun exemplars specified in `Required Stem Frame` and `🎯 Micro-Task for LLM`.
   - 🚫 **NO SCAFFOLDING OR RAW BRACKET LEAKAGE**: NEVER output meta-instructions (`"Fit into frame..."`, `"Plural noun slot required..."`) or raw brackets (`[concrete physical object...]`, `[sb]`, `[sth]`, `[verb]`) into the question stem.
   - 🎯 **Declared Inflection Is Binding**: For verb targets, the prescribed options are already cast in a specific inflectional form (e.g. past tense, `-s` form, past participle, or base form), and the stem syntax must grammatically require that exact form.

4. **Explanations & Design Audit**:
   - `explanation`: Provide concise, objective pedagogical explanations stating why the correct target fits the sentence context, and explicitly distinguishing why each of the other 3 options is disqualified (syntactic clash, preposition mismatch, or semantic contradiction). Quote the exact words of the options.
   - 🚫 **NO SELF-TALK OR DRAFT REVISIONS IN EXPLANATIONS**: Write the final `explanation` directly in an authoritative, third-person instructional voice. NEVER include internal reasoning, self-corrections, or draft notes (such as `"Wait, ..."`, `"The stem I wrote..."`, `"Let's adjust..."`, or `"Revised Stem:"`) inside `explanation`.
   - 🔒 **NO DISTRACTOR SELF-DEFINITION**: Never justify an option by simply stating what that option itself means. Each disqualification must be a contrast with the target's locked sense — what the target means in this sentence that the option cannot mean. Do not restate the blueprint's `Semantic Discriminator` wording as the reason.
   - `definition`: State the contextual dictionary meaning of the target word.
   - `design_audit`: Provide a compact audit tag chain (under 15 words) showing the exact blanked frame used in your question stem:
     `AUDIT: [Target Word] -> [Part of Speech] -> [Exact Blanked Frame in Stem, e.g. 'repay a ____' or '____ in']`

CONTENT:
{vocabulary_content}
