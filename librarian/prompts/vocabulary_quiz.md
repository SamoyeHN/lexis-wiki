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
     * If the target is a NOUN, the blank MUST be a noun phrase position (subject, object, complement). If the declared target is plural, the stem MUST naturally require a plural noun.
     * If the target is an ADJECTIVE or ADVERB, the blank must strictly modify its corresponding noun, verb, or adjective. If the target functions as a predicate adjective (e.g. following 'fell', 'became', 'remained', 'is/was'), the stem MUST include the linking/copular verb so the predicate clause is grammatically complete (NEVER write a headless fragment like 'the classroom ____ .').
   - 🎯 **Execute the Item Micro-Task & Preserve Required Stem Frame**: Follow the specific syntactic frame and lexical clues in `Required Stem Frame` and `🎯 Micro-Task for LLM`. When a collocational frame is specified (e.g. `... ____ in`, `... ____ authority`, or `... repay a ____`), `____` replaces ONLY the target word — you MUST write the surrounding anchor words (`in`, `authority`, `repay a` / `repaid a`) explicitly in the sentence next to `____`. NEVER swallow the collocational verb or article into `____`.
   - **Authentic Scenarios at CEFR {cefr_level}**: Construct natural, communicative or academic sentences suitable for CEFR {cefr_level} learners. Do not write unnecessarily convoluted philosophical clauses for foundational A1–B1 levels.
   - 🚫 **NO STEM TARGET LEAKAGE**: The target word or its derivatives must NEVER appear anywhere in the stem outside the blank `____`.
   - 🚫 **NO COPYING INPUT EXAMPLES OR MODEL SENTENCES**: Every stem must be original. NEVER copy any sentence, clause, or run of 4 or more consecutive words from the blueprint's `Model: "..."` example, its `Authentic Corpus Blueprint` example, its `authentic_example`, its `cloze_frame_prototype`, any `Contextual Definition`, or any `Curriculum Quote` line. The `Model:` example illustrates syntax only — NEVER copy its sentence structure directly or let its singular/plural morphology alter the declared target inflection.
   - 🚫 **NO SCAFFOLDING OR META-INSTRUCTION LEAKAGE**: NEVER output instruction strings (such as "Fit into frame...", "Plural noun slot required...", "Complete predicate required...", or "Write a natural sentence...") or bracketed placeholders (like `[sb]` or `[sth]`) into the question stem. The stem MUST be a pure, natural context sentence containing exactly one `____`.
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
