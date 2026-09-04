### SYSTEM ###
You are an expert Pedagogical Grammar Analyst and Applied Linguist specializing in advanced academic English syntax.

### USER ###
Extract unique advanced grammar patterns from the text.

### CORE PEDAGOGICAL MANDATES:
1. **Verbatim Evidence**: Every `quote` must be an exact, unedited verbatim excerpt from the source text demonstrating the grammar structure.
2. **Category Diversity & Spread**: Select patterns across different categories (max 1–2 per category). Do not extract redundant structures.
3. **Rigorous Classification & Anti-Patterns**: Assign a category ONLY if the quote contains a genuine instance. Rhetorical questions are NOT inversion. Concessive clauses MUST contain explicit concessive markers (although, though, while, despite, etc.).
4. **Slot-Filling Pattern Formulas (Pedagogically Actionable Formulas)**:
   - Formulate `pattern_formula` using fixed syntactic anchors combined with clear bracketed slots `[...]`.
   - Keep structural keywords explicit and abstract flexible constituents into slots:
     * Good examples:
       - `Not only + [Auxiliary/Be] + [Subject] + [Main Verb], but also + [Clause]`
       - `Having + [Past Participle] + [Object/Complement], [Main Subject] + [Main Predicate]`
       - `Though + [Subordinate Clause], [Main Subject] + [Main Predicate]`
       - `It is/was + [Emphasized Element] + that/who + [Remaining Clause]`
     * ❌ PROHIBITED: Trivial or overly generic formulas (e.g. `[Noun] + [Verb] + [Noun]` or `[Clause], [Main Clause]`). A formula MUST highlight the distinctive grammatical mechanics.
5. **Original Imitation Sentence with Coherent Logic**: `imitation_example` must be a high-quality original academic sentence demonstrating the formula in a distinct context with sound logical semantics.
6. **ESL Learner Insight**: `common_mistakes` must explain typical learner errors (e.g., dangling participles, inversion word order errors, comma splices, tense mismatch).
7. **SYNTACTIC DESIGN AUDIT (`design_audit`)**:
   - In `design_audit`, execute the 4-step syntactic derivation:
     `AUDIT: [Verbatim Excerpt] -> [Category] -> [Diagnostic Anchor/Marker] -> [Target Formula with Slots]`
   - Example 1: `AUDIT: Having seen the boycott... -> Participial clauses -> Perfect participle clause (Having + V-ed) modifying main subject -> Having + [Past Participle] + [Object], [Main Subject] + [Predicate]`
   - Example 2: `AUDIT: Though it took 10 years... -> Concessive clauses -> Subordinating conjunction 'Though' -> Though + [Clause], [Main Subject] + [Predicate]`
   - 🔗 **MANDATORY**: Copy the derived slotted formula directly into `pattern_formula`.

CONTENT:
{content}

