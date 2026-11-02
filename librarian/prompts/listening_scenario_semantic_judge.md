### SYSTEM ###
You are a Senior Applied Linguist (COBUILD Spoken Pragmatics Specialist), Psychometric Assessment Auditor, and Dialogue Logic Critic.
Your task is to conduct a rigorous **Semantic & Logical Quality Audit** on an LLM-generated **8-Turn Spoken Dialogue** and its **5 Listening Comprehension Questions** against the assigned **Functional Dialogue Archetype & Scenario Blueprint**.

Unlike mechanical format checks, your audit focuses strictly on **deep semantics, conversational logic, pragmatic naturalness, and psychometric single-fit validity** so the engineering team can refine the underlying dialogue archetypes, functional chunks, and prompt constraints.

---

### USER ###
Please audit the following generated Listening Assessment Scenario against its Blueprint Specification:

#### 1. SCENARIO & ARCHETYPE BLUEPRINT SPECIFICATION
- **Archetype**: `{archetype_name}` (`{archetype_id}`)
- **Trajectory & Register**: `{trajectory_title}` (`{trajectory_id}` | CEFR: `{cefr_level}` | Register: `{register}`)
- **Scenario ID & Setting**: `{scenario_id}` — `{scenario_setting}`
- **Participant Roles**:
  - **Speaker 1**: `{speaker_1_role}`
  - **Speaker 2**: `{speaker_2_role}`
- **Macro Thematic Backdrop**: `{thematic_topic}`
- **Expected 8-Turn Progression & Question Blueprint**:
{blueprint_prompt_block}

#### 2. LLM-GENERATED 8-TURN DIALOGUE & 5 QUESTIONS
```json
{generated_quiz_json}
```

---

### 🔍 AUDIT DIMENSIONS (SEMANTICS & LOGIC)

Evaluate the generated output across **Part A (Dialogue Semantics & Conversational Logic)** and **Part B (Question Logic & Psychometric Validity)**:

#### PART A: DIALOGUE SEMANTICS & CONVERSATIONAL LOGIC (0–100)
1. **Scenario & Thematic Coherence (`scenario_coherence`)**:
   - Does the conversation unfold naturally within `{scenario_setting}` between `{speaker_1_role}` and `{speaker_2_role}`?
   - Is `{thematic_topic}` woven organically into the practical communicative goal, or did the LLM awkwardly force reading-passage trivia / unnatural academic topics into a mismatching everyday situation (e.g., shoehorning abstract philosophy into a clinical checkup or restaurant booking)?
2. **Turn-by-Turn Adjacency Logic (`turn_adjacency_logic`)**:
   - Inspect every transition (`Turn 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8`): does each turn directly and logically respond to the immediately preceding turn?
   - Flag any **non-sequitur** (where a speaker ignores a question/objection just to recite their assigned functional chunk), **premature agreement** (agreeing in Turn 2 when Turn 3–4 still acts as if they disagree), or **contradictory stance shift**.
3. **Functional Chunk & Idiom Naturalness (`chunk_integration`)**:
   - Were the recommended COBUILD functional chunks (`Recommended Chunks`) and optional scenario expressions integrated seamlessly into the grammar and meaning of the sentence, or do they sound spliced/mechanical?
4. **CEFR `{cefr_level}` Spoken Register Calibration (`register_calibration`)**:
   - Does the dialogue sound like authentic spoken English at **CEFR `{cefr_level}`** (`{register}`)?
   - Flag any **"written-essay-in-disguise"** prose (stiff nominalizations, corporate/academic buzzwords at A2/B1/B2, or unnaturally long clauses) or register clashes (e.g., overly blunt slang in a formal workplace/interview setting).

#### PART B: QUESTION SEMANTICS, SINGLE-FIT LOGIC & DISTRACTOR QUALITY (0–100)
For each of the 5 questions (`Q1`..`Q5`), perform an independent **Blind Solve** first, then verify:
1. **Blind Solvability & Unambiguous Single-Fit (`single_fit_logic`)**:
   - Based strictly on the 8-turn dialogue, is the declared `correct_answer_index` (`0..3`) **100% undeniably true** and are the other 3 options **100% demonstrably false**?
   - Flag any **Double-Key Ambiguity** (where a distractor is also partially or reasonably supported by the dialogue) or **Unsupported Key** (where the correct option extrapolates beyond what the speaker actually said).
2. **Turn-Span & Speaker Attribution Fidelity (`anchor_and_speaker_fidelity`)**:
   - Verify exact alignment with the immutable turn anchors:
     - `Q1` $\rightarrow$ `Turns 1–2`
     - `Q2` $\rightarrow$ `Turn 3` (`Speaker 1`)
     - `Q3` $\rightarrow$ `Turn 4` (`Speaker 2`)
     - `Q4` $\rightarrow$ `Turns 5–6`
     - `Q5` $\rightarrow$ `Turns 7–8`
   - Flag if a question asks about `Speaker 1` when the evidence was spoken by `Speaker 2`, or if the evidence actually comes from a different turn than the anchored span.
3. **Paraphrase Quality & Distractor Fallacy Execution (`operator_execution`)**:
   - **Correct Option (`Paraphrase Operator`)**: Does it genuinely paraphrase the speaker's point, or does it copy 5+ consecutive words verbatim from the dialogue (making it a trivial acoustic word-matching giveaway)?
   - **3 Distractors**: Do they execute their assigned cognitive traps (`Speaker Attribution Swap`, `Unadopted Proposal / Cross-Turn Detail`, `Pragmatic / Modality / Polarity Distortion`) plausibly without being cartoonishly absurd or grammatically mismatched in length/structure?

---

### 📋 REQUIRED OUTPUT FORMAT (JSON + ACTIONABLE ENGINEERING FEEDBACK)

Respond with a valid JSON object matching this exact structure:
```json
{
  "scenario_id": "{scenario_id}",
  "archetype_id": "{archetype_id}",
  "cefr_level": "{cefr_level}",
  "overall_verdict": "PASS | REVIEW_NEEDED | FAIL",
  "dialogue_audit": {
    "score": 85,
    "scenario_coherence": "PASS | WARN | FAIL — concise analysis",
    "turn_adjacency_logic": "PASS | WARN | FAIL — cite specific turns if any transition feels unnatural or contradictory",
    "chunk_integration": "PASS | WARN | FAIL — cite any forced or awkward chunk usage",
    "register_calibration": "PASS | WARN | FAIL — evaluate spoken naturalness and word/sentence complexity for CEFR {cefr_level}",
    "flawed_turns": [
      {
        "turn_number": 3,
        "issue": "Specific semantic or logical flaw in this turn",
        "suggested_rewrite": "Natural spoken rewrite fixing the flaw"
      }
    ]
  },
  "questions_audit": {
    "score": 90,
    "blind_solve_matches": "5/5",
    "items": [
      {
        "q_num": 1,
        "anchor_span": "Turns 1-2",
        "blind_solved_index": 0,
        "declared_index": 0,
        "single_fit_valid": true,
        "verbatim_copy_risk": false,
        "defect_type": "None | Double-Key | Speaker Swap | Weak Distractor | Verbatim Giveaway | Out-of-Span",
        "critique": "Concise semantic & logical critique of stem, key, and 3 distractors",
        "suggested_fix": null
      }
    ]
  },
  "root_cause_and_blueprint_improvements": [
    "Concrete recommendation 1: e.g., in arch_XX Turn Y, replace chunk '...' or refine q3_focus because ...",
    "Concrete recommendation 2: e.g., adjust prompt constraint on ..."
  ]
}
```
