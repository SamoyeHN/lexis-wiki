### SYSTEM ###
You are an expert ESL Audio Script Writer and Listening Assessment Designer specializing in standardized CEFR {cefr_level} language instruction.
### USER ###
Generate a realistic everyday or academic spoken dialogue and a rigorous comprehension assessment calibrated to **CEFR {cefr_level}** by executing the pre-compiled **Functional Dialogue Archetype & 8-Turn Pragmatic Trajectory Blueprint** and **Question Micro-Task Blueprints** below.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Functional Dialogue Archetype, Cohesive Storyline & Condensed Spoken Patterns**:
   - Execute the **8-Turn Pragmatic Progression** (`Turns 1–8`) between `{speaker_1_name}` and `{speaker_2_name}` specified in the blueprint below.
   - **Cohesive Storyline & Natural Turn Uptake**:
     * Keep ONE consistent topic/situation from Turn 1 to Turn 8. Every turn (`Turns 2–8`) MUST briefly react to what the other speaker just said before adding the next question or point.
   - **Flexible Spoken Pattern Toolkit (Affirmative / Negative / Conditional Openness)**:
     * Treat the `Flexible Spoken Patterns` in each turn (e.g. `be quite light / tight on [day]`) as condensed syntactic building blocks rather than pre-written scripts.
     * Freely choose affirmative, negative, or conditional directions, adapt bracketed slots to fit your storyline, or coin your own natural spoken phrasing—NEVER stitch disconnected textbook sentences together, and do NOT repeat the same spoken opener or pattern across different turns.
   - **Curriculum Alignment (CEFR {cefr_level})**:
     * **Dialogue Register & Complexity**: {dialogue_style_guidance}
   - **Speaker Persona, Gender & Accent Alignment**:
     * Set `speaker_1` to `"{speaker_1_name}"` (`speaker_1_gender`: `"{speaker_1_gender}"`, `speaker_1_accent`: `"{speaker_1_accent}"`) and assign a realistic situational role in `speaker_1_role`.
     * Set `speaker_2` to `"{speaker_2_name}"` (`speaker_2_gender`: `"{speaker_2_gender}"`, `speaker_2_accent`: `"{speaker_2_accent}"`) and assign a realistic situational role in `speaker_2_role`.
     * In `script`, every turn's `speaker` MUST be either `"{speaker_1_name}"` or `"{speaker_2_name}"`.
     * In `question`, `options`, and `explanation`, ALWAYS refer to the speakers by their actual names (`{speaker_1_name}` and `{speaker_2_name}`)—NEVER mention `'Turn X'` or write generic labels like `"Speaker 1"` or `"Speaker 2"`.
   - ⚠️ **SPEAKER ATTRIBUTION RIGOR MANDATE**: Clearly distinguish the roles, stances, and proposals of `{speaker_1_name}` vs `{speaker_2_name}`.
   - ⚠️ **SCENARIO & DOMAIN BINDING MANDATE**:
     * The conversation MUST genuinely execute the specific situation described in **Everyday Scenario Setting** and its **Communicative Goal** (e.g. if the scenario is asking for directions or comparing routes to a campus facility, the dialogue MUST directly involve asking directions, landmark guidance, and route comparison).
     * The **Macro Thematic Backdrop** provides the topical context (such as the specific department, facility, or course subject).
     * NEVER drift or default into generic, unrelated corporate tropes (such as 'remote work pilot', 'work-from-home policy', or 'office scheduling') unless the unit theme is specifically about workplace telecommuting.

2. **Dialogue-Driven Question Blueprints (Adapt Naturally to Your Dialogue)**:
   - Generate **EXACTLY {count}** comprehension questions in the `questions` array (`Question 1 Blueprint` through `Question {count} Blueprint` in order).
   - **Write Dialogue First, Adapt Questions Second**:
     * Base every question stem and option on **what actually happens in the dialogue you just generated** (including whichever affirmative, negative, or conditional direction you chose), while strictly obeying each item's **Required Category (`category`)** and **Prescribed Correct Answer Index (`correct_answer_index`)**:
{skill_distribution_guidance}
   - **Question Stem Guidance**: {question_stem_guidance}
   - **Option Guidelines & Single-Fit Validity**:
     * Provide 4 concise, grammatically parallel options grounded in your generated dialogue ({option_complexity_guidance}).
     * **Correct Answer (`options[correct_answer_index]`)**: Place the true answer at the exact `Prescribed Correct Answer Index`, paraphrasing the target exchange naturally—NEVER copy 5+ consecutive words verbatim from a dialogue turn.
     * **3 Plausible Distractors**: Fill the other 3 option slots with realistic false choices grounded in your dialogue using the item's suggested `Distractor Traps` (e.g., swapping speaker roles, flipping a condition/polarity, or mixing in a detail from another turn). Ensure no distractor is accidentally true.

3. **Design Audit & Explanation**:
   - `design_audit`: Keep concise (under 20 words) using the tag chain format:
     `AUDIT: [Speaker & Turn #] -> [Category] -> [Key Traps]`
     (e.g. `AUDIT: [{speaker_1_name}, Turn 3] -> [Detail] -> [Polarity Flip + Role Swap]`). DO NOT write paragraphs or quote dialogue here.
   - `explanation`: Explain why the correct option (`Option A`/`B`/`C`/`D` matching `Prescribed Correct Answer Index`) is supported by what `{speaker_1_name}` or `{speaker_2_name}` said, and contrastively explain why each of the other 3 options (referred to as `Option A`, `Option B`, `Option C`, `Option D` — NEVER `"Option 0"` or `"Option 1"`) is false.
   - `correct_answer_index`: MUST equal the exact integer (`0`, `1`, `2`, or `3`) specified by `Prescribed Correct Answer Index` in that question's blueprint.
   - Do NOT wrap question or option text in quotes or labels (`A.`, `B)`).

FUNCTIONAL DIALOGUE & QUESTION BLUEPRINTS:
{vocabulary_content}
