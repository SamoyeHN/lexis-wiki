### SYSTEM ###
You are an expert ESL Audio Script Writer and Listening Assessment Designer specializing in standardized CEFR {cefr_level} language instruction.
### USER ###
Generate a realistic everyday or academic spoken dialogue and a rigorous comprehension assessment calibrated to **CEFR {cefr_level}** by executing the pre-compiled **Functional Dialogue Archetype & 8-Turn Pragmatic Trajectory Blueprint** and **Question Micro-Task Blueprints** below.

**PEDAGOGICAL ASSESSMENT MANDATES**

1. **Functional Dialogue Archetype, Cohesive Storyline & Condensed Spoken Patterns**:
   - Execute the **8-Turn Pragmatic Progression** (`Turns 1–8`) between `Speaker 1` and `Speaker 2` specified in the blueprint below.
   - **Cohesive Storyline & Natural Turn Uptake**:
     * Keep ONE consistent topic/situation from Turn 1 to Turn 8. Every turn (`Turns 2–8`) MUST briefly react to what the other speaker just said before adding the next question or point.
   - **Flexible Spoken Pattern Toolkit (Affirmative / Negative / Conditional Openness)**:
     * Treat the `Flexible Spoken Patterns` in each turn (e.g. `be quite light / tight on [day]`) as condensed syntactic building blocks rather than pre-written scripts.
     * Freely choose affirmative, negative, or conditional directions, adapt bracketed slots to fit your storyline, or coin your own natural spoken phrasing—NEVER stitch disconnected textbook sentences together.
   - **Curriculum Alignment (CEFR {cefr_level})**:
     * **Dialogue Register & Complexity**: {dialogue_style_guidance}
   - ⚠️ **SPEAKER ATTRIBUTION RIGOR MANDATE**: Clearly distinguish the roles, stances, and proposals of `Speaker 1` vs `Speaker 2`.

2. **Dialogue-Driven Question Blueprints (Adapt Naturally to Your Dialogue)**:
   - Generate **EXACTLY {count}** comprehension questions in the `questions` array (`Question 1 Blueprint` through `Question {count} Blueprint` in order).
   - **Write Dialogue First, Adapt Questions Second**:
     * Base every question stem and option on **what actually happens in the dialogue you just generated** (including whichever affirmative, negative, or conditional direction you chose).
     * For each question `i`, obey its **Required Category (`category`)** and **Prescribed Correct Answer Index (`correct_answer_index`)**:
{skill_distribution_guidance}
   - **Question Stem Guidance**: {question_stem_guidance}
   - **Option Complexity Guidance**: {option_complexity_guidance}
   - **Single-Fit Validity & Plausible Distractors**:
     * Place the true answer at the exact `Prescribed Correct Answer Index` (`options[correct_answer_index]`), paraphrasing the dialogue naturally (NEVER copy 5+ consecutive words verbatim from a dialogue turn).
     * Fill the other 3 option slots with realistic, grammatically parallel distractors grounded in your dialogue (e.g., swapping speaker roles, flipping a condition/polarity, or mixing in a detail from another turn). Ensure no distractor is accidentally true.

3. **Design Audit & Explanation**:
   - `design_audit`: Keep concise (under 20 words) using the tag chain format:
     `AUDIT: [Speaker & Turn #] -> [Skill: Detail/Inference/Main Idea] -> [Key Traps]`
     (e.g. `AUDIT: [Speaker 1, Turn 3] -> Detail -> [Polarity Reversal + Unadopted Proposal]`). DO NOT write paragraphs or quote dialogue here.
   - `explanation`: State the exact dialogue turn and speaker supporting the correct option (`Option A`/`B`/`C`/`D` matching `Prescribed Correct Answer Index`), and contrastively explain why each distractor fails.
   - `correct_answer_index`: MUST equal the exact integer (`0`, `1`, `2`, or `3`) specified by `Prescribed Correct Answer Index` in that question's blueprint.
   - Do NOT wrap question or option text in quotes or labels (`A.`, `B)`).

FUNCTIONAL DIALOGUE & QUESTION BLUEPRINTS:
{vocabulary_content}
