### SYSTEM ###
You are an expert ESL Lexical Assessment Specialist (CEFR/TOEFL standard).
### USER ###
Create a high-quality multiple-choice vocabulary assessment based on the provided vocabulary list.

**PEDAGOGICAL ASSESSMENT MANDATES**
1. **EXACT QUESTION COUNT**: Generate EXACTLY {count} questions in the 'questions' array.
2. **STRICT PART-OF-SPEECH & INFLECTION MATCHING**: 
   - The blank (____) in every assessment sentence MUST grammatically require the EXACT part of speech and grammatical form of the target word.
   - All 4 options must share the EXACT same part of speech, grammatical category, and inflectional form (e.g., all past tense verbs, all plural nouns, or all base adjectives) to eliminate giveaway grammatical clues.
3. **CONTEXTUAL & COLLOCATIONAL CONSTRAINT**: 
   - Write completely NEW, rich academic context sentences at CEFR {cefr_level}. DO NOT copy or re-use any 'Quoted Sentence' or 'Example Usage' from the input vocabulary list.
   - The sentence MUST provide explicit contextual, syntactic, or collocational constraints (e.g., dependent prepositions, specific semantic collocations, or contrastive clauses) that make the target word the SINGLE, UNAMBIGUOUSLY correct choice.
4. **PLAUSIBLE DISTRACTORS**: Distractors must be plausible near-synonyms or register matches at the same CEFR tier, rendered strictly incorrect by the specific preposition, collocation, or semantic context in the sentence.
5. **CONTRASTIVE EXPLANATIONS**: In `explanation`, provide contrastive reasoning: clearly state why `target_word` is the precise fit in this context AND specifically why key distractors are incorrect (e.g., wrong dependent preposition, semantic mismatch, or improper register).
6. **ALIGNMENT WITH DESIGN AUDIT**: The assessment sentence in 'question' must be the populated version of the advanced academic sentence planned in 'design_audit'.
7. **QUESTION FIELD MANDATE**: Use exactly four underscores (____) for the blank in 'question'. Do NOT wrap the 'question' sentence in outer quotation marks.
8. **OPTIONS FIELD MANDATE**: Return ONLY the literal word/phrase for each item in 'options'. DO NOT include option labels (e.g., 'A)', 'a.', '1.'). DO NOT wrap option items or target words in single quotes ('), double quotes ("), or curly smart quotes (“ ”).

VOCABULARY:
{vocabulary_content}
