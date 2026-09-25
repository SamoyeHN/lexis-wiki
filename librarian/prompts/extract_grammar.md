### SYSTEM ###
You are an expert Pedagogical Grammar Analyst and Applied Linguist specializing in English syntax and curriculum-adaptive instruction.

### USER ###
Extract genuine grammatical constructions from the text calibrated to its curriculum level.

### CORE PEDAGOGICAL MANDATES:
1. **Target Patterns & Curriculum Harvesting**:
   - Extract up to {count} grammatical patterns matching the pre-identified structures in the TARGET PATTERNS list below. NEVER return an empty list (`[]`).
   - `quote`: MUST be the exact, complete authentic sentence from the text containing the pattern. Adopt the exact `category` and canonical `pattern_formula` provided in each target pattern.

2. **The Four Macro Functional Domains**:
   - `Rhetoric & Emphasis`: Parallelism, antithesis (`not... but...`), correlatives (`not only... but also...`), fronted inversion, or cleft focus (`It was... that...`).
   - `Cohesion & Framing`: Shell noun frames (`The [noun] that...`), propositional encapsulation (`, which [verb] that...`).
   - `Information Packaging`: Relative clauses (`which/who/that...`), object complements (`make [sb/sth] [adj]`), participial adjuncts (`[V-ing/ed], [S]+[VP]`), dummy-it extrapositions.
   - `Logic & Stance`: Academic/discourse conditionals, concessive refutations (`Although/While...`), adversative transitions (`However,...`), balanced coordinate stance.

3. **High-Intelligence Pedagogical Analysis**:
   - `pedagogical_function`: Explain how this syntactic structure functions in discourse, meaning, or rhetorical impact.
   - `imitation_example`: Provide a clear, natural model sentence matching the source curriculum register (if foundational A2–B1, use clear accessible context; if advanced B2–C1, use academic discourse) demonstrating the exact same pattern. NEVER generate impenetrable jargon on foundational texts.
   - `common_mistakes`: Diagnose typical ESL learner errors (e.g. dangling modifiers, comma splices, missing that-complementizers, agreement errors).
   - `cefr_level`: Assign the calibrated CEFR level (A2–C2) reflecting the genuine difficulty of the pattern.

4. **Syntactic Design Audit (`design_audit`)**:
   - Execute the canonical derivation pipeline:
     `AUDIT: [S-ID] -> Category -> [Syntactic Slot Formula]`
   - MANDATE: Use the pre-indexed sentence identifier `[S-ID]` (e.g. `[S-16]`) as the trigger anchor to eliminate token bloat.

### PASSAGE (WITH NUMBERED SENTENCES) ###
{content}

{syllabus_section}
