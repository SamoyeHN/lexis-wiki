### SYSTEM ###
You are an expert Reading Comprehension Assessment Designer.

### USER ###
Create a reading comprehension assessment based on the provided passage.

**MANDATES**
1. **EXACT QUESTION COUNT**: Generate EXACTLY {count} comprehension questions in the 'questions' array. Do NOT stop early.
2. **VOCABULARY GLOSSARY & NON-OVERLAP MANDATE**: 
   - Identify 5 to 8 challenging academic vocabulary words from the passage and populate the 'vocabulary' array with context sentences, parts of speech, definitions, and new example sentences.
   - **NON-OVERLAP RULE**: Glossary terms must NOT give away answers to or directly duplicate the target words/phrases tested in the comprehension questions.
3. **DESIGN AUDIT MANDATE**: For every question, 'design_audit' MUST follow this exact 3-step planning format: 'DRAFT: [Tested Skill (e.g. Inference)] -> [Text Evidence Location (e.g. Paragraph 3, Line 2)] -> [3 Planned Distractor Traps (Literal Matching / Scope Shift / Logic Error)]'
4. **DIAGNOSTIC DISTRACTORS**: Distractors must use intentional cognitive traps (e.g. literal phrase matching in false context, overgeneralization/scope shift, or reversed logic).
5. **QUESTION FIELD MANDATE**: Write clear comprehension questions for 'question'. Do NOT wrap the 'question' text in outer quotation marks.
6. **OPTIONS FIELD MANDATE**: Return ONLY the literal answer/phrase for each item in 'options'. DO NOT include option labels (e.g., 'A)', 'a.', '1.'). DO NOT wrap option items in single quotes ('), double quotes ("), or curly smart quotes (“ ”).

PASSAGE:
{passage_content}
