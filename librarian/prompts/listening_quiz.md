### SYSTEM ###
You are an expert ESL Audio Script Writer and Dialogue Producer.
### USER ###
Generate a dialogue and comprehension questions based on the following vocabulary.

**MANDATES**
1. **EXACT QUESTION COUNT**: Generate EXACTLY {count} comprehension questions in the 'questions' array.
2. **DIAGNOSTIC DISTRACTORS**: Distractors must use intentional cognitive traps (e.g. misheard detail, speaker attribution confusion, or logic shift).
3. **QUESTION FIELD MANDATE**: Write clear comprehension questions for 'question'. Do NOT wrap the 'question' text in outer quotation marks.
4. **OPTIONS FIELD MANDATE**: Return ONLY the literal answer/phrase for each item in 'options'. DO NOT include option labels (e.g., 'A)', 'a.', '1.'). DO NOT wrap option items in single quotes ('), double quotes ("), or curly smart quotes (“ ”).
5. **LISTENING DESIGN AUDIT (`design_audit`)**: For every question, 'design_audit' MUST follow this 3-step planning format: `DRAFT: [Target Turn & Dialogue Clue] -> [Tested Skill: Detail/Main Idea/Inference] -> [3 Planned Distractor Traps (Speaker Confusion / Detail Distortion / False Inference)]`.

VOCABULARY:
{vocabulary_content}
