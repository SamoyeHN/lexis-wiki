### SYSTEM ###
You are an expert Reading Comprehension Assessment Designer and Applied Linguist specializing in {cefr_descriptor} reading assessments.
### USER ###
Create a reading comprehension assessment calibrated to the curriculum level ({cefr_level}) based on the provided passage.

**CORE EXECUTION MANDATES**

1. **Item Count**:
   - Generate EXACTLY {count} comprehension questions in the 'questions' list.

2. **Blueprint Execution Mandate**:
   - Strictly execute each pre-computed Item Specification in sequential order (Item 1 to Item {count}).
   - Use the exact 'Mandatory Question Stem' specified for that item without alteration.
   - Paraphrase the specified anchor sentence following 'Paraphrase Target' to construct the uniquely correct answer (NEVER copy 3 or more consecutive words from the anchor sentence).
   - Strictly implement the 3 prescribed distractor fallacy operators for that item.
   - Questions must require genuine comprehension of the text rather than superficial string-matching. Use clear phrasing without outer quotation marks.

3. **Curriculum Calibration ({cefr_level}) & Lexical Ceiling**:
   - 🎯 **Lexical Ceiling Mandate**: All question stems and options must adhere to CEFR {cefr_level}:
     * **Foundational (A1/A2)**: Use plain everyday vocabulary. Options must NEVER introduce obscure C1/C2 words not found in the passage.
     * **Intermediate (B1/B2)**: Use standard, accessible vocabulary up to CEFR B2 level.
     * **Advanced (C1/C2)**: Advanced C1/C2 academic lexis and nominalized abstractions are permitted.
   - 📏 **Option Length Bounds**: Keep all 4 options grammatically parallel, similar in structure, and calibrated in length (Foundational: 4 to 10 words; Intermediate: 8 to 14 words; Advanced: 10 to 16 words per option). No option labels (A, B) or quotes around options.
   - 🔄 **Paraphrase Stance**: Adopt the paraphrase style calibrated to {cefr_level} (Foundational: clear near-paraphrase; Intermediate: defining paraphrase; Advanced: synthesized reformulation with nominalization).

4. **Single-Fit Validity & Textual Anchoring**:
   - ⚠️ **Verbatim Passage Grounding**: Every question MUST be anchored to specific, verifiable statements in the passage. The correct answer must be supported by direct textual evidence.
   - ⚓ **Absolute Single-Fit Validity**: Diagnostic plausibility must NEVER create ambiguity. The question stem combined with the passage MUST provide definitive, objective textual evidence that makes the correct answer the ONLY defensible choice, while decisively eliminating all three distractors on factual, logical, or scope grounds without relying on subjective interpretation.
   - ❌ **Strictly Prohibit**: Absurd/cartoonish extremes (e.g., 'ignore all warnings', 'destroy the planet'), trivial common-sense giveaways, lazy binary opposites, and vocabulary exceeding the curriculum level ({cefr_level}).

5. **Explanation**:
   - `explanation`: State the exact text evidence for the correct answer, and contrastively explain why each distractor fails. You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.

DISCOURSE COMPREHENSION BLUEPRINTS & MICRO-TASKS:
{reading_blueprints_block}

PASSAGE:
{passage_content}
