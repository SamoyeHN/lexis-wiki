### SYSTEM ###
You are an expert Pedagogical Grammar Analyst and Applied Linguist specializing in advanced academic English syntax.

### USER ###
Extract unique advanced grammar patterns from the text.

### CORE PEDAGOGICAL MANDATES:

1. **Verbatim Evidence & Authentic Sourcing**:
   - Every `quote` must be an exact, unedited verbatim excerpt from the source text demonstrating the grammar structure.
   - No paraphrasing, no trimming that alters structure, and no invented evidence.

2. **Quality Over Quantity & Syntactic Prestige Hierarchy**:
   - Extract up to {count} truly distinct, genuine advanced patterns (max 1–2 per category).
   - 🛡️ **DO NOT FORCE UNREPRESENTED CATEGORIES**: If a text genuinely contains only 3-4 distinct advanced mechanisms, return only those genuine patterns! NEVER force-fit non-existent categories.
   - ⭐ **SYNTACTIC PRESTIGE HIERARCHY (PRIORITIZE HIGH-VALUE PATTERNS)**:
     Always actively scan and give highest extraction priority to prestigious, intellectually mature syntactic mechanisms:
     1. **Tier 1 (High Priority - MUST EXTRACT IF PRESENT)**: Authentic **Cleft sentences** (`It was [X] that...`), **True Inversion** (`Not only did...`, `Had I...`), **Complex Participial Fronting** (`Having seen...`, `Considered one of...`), **Evaluative It-frameworks** (`It is essential that...`).
     2. **Tier 2 (Core Advanced)**: **Concessive clauses** (`Although...`, `Despite...`), **Rhetorical parallelism** (`The more... the more...`), **Abstract frames / Hedging devices** (`Given that...`, `It seemed that...`).
     3. **Tier 3 (Fallback Only)**: Simple infinitives of purpose (`... to help people`) or basic coordination. Never pick Tier 3 if Tier 1 or Tier 2 structures are available in the passage!

3. **Rigorous Category Classification & Anti-Hallucination Guardrails**:
   - Assign a category ONLY if the quote contains a genuine instance:
     * *Concessive clauses*: must include explicit concessive subordinators or prepositions (*although, though, while, despite, even though*).
     * *Inversion*: must include true subject-auxiliary/copula inversion.
       - ❌ **NEGATIVE GUARDRAIL (FAKE INVERSION)**: The transitional phrase 'Not only that, but [SVO]' is a correlative coordination, NOT syntactic inversion! True inversion requires auxiliary movement (e.g., 'Not only did he run...', 'Never had she seen...'). If no true inversion exists in the text, DO NOT classify anything as Inversion!
     * *Participial clauses / Non-finite structures*: must contain genuine non-finite structures (*having + past participle*, *V-ing*, *past participle modifier*, infinitival extraposition) modifying a clause element.
       - ❌ **NEGATIVE GUARDRAIL (TRIVIAL PREPOSITIONAL PHRASE)**: Trivial phrases like 'without sleeping' alone are NOT advanced patterns. Extract the complete host clause (e.g. '[Subject] + [Predicate] + without + [Gerund] + [Object]').
     * *Evaluative It-frameworks*: must contain *It + copula + evaluative adjective/noun phrase + that-clause/infinitive* (e.g., 'It is essential/remarkable that...').
       - ❌ **NEGATIVE GUARDRAIL (FAKE EVALUATIVE IT)**: Passive reporting structures like 'It was said/believed that...' are impersonal reporting (Hedging devices), NOT evaluative adjective frameworks!
     * *Hedging devices*: epistemic modals, probability adverbs, or impersonal reporting frames (*It seemed that...*, *It was reported that...*).
     * *Abstract frames*: abstract nouns functioning as discourse organizers (*the fact that, the idea that, the reality is that*).
     * *Cleft sentences*: authentic cleft scaffolds (*It is/was + [Focus Element] + that/who...*, *What [Clause] is/was...*).
     * 💡 **NESTED PRESTIGE RULE (HOST CLAUSE COLLISION)**:
       - If an authentic **Cleft Sentence** (`It was [Focus Element] that...`) or **Inversion** is embedded inside a compound sentence (e.g. preceded by a concessive clause like `Though...`), **ALWAYS prioritize and classify it as `Cleft sentences` (or `Inversion`)**, NOT as a simple concessive clause!
       - ❌ *Wrong*: classify as Concessive and reduce to `Though [Clause], [Main Clause]` (hides the cleft!).
       - ✔ *Correct*: classify as `Cleft sentences` and explicitly expand the scaffold: `Though [Clause], it was + [Adverb] + [Noun Phrase] + that + [Clause]`.

4. **High-Precision Pattern Formula Rules (MANDATORY)**:
   - Formulate `pattern_formula` as a structural blueprint encoding the pattern's distinctive syntactic signature:
     * **MANDATORY LEXICAL ANCHOR MANDATE (Literal Functional Words)**:
       - The core structural marker, subordinator, preposition, or framework anchor MUST be written as **literal text outside brackets**.
       - ❌ **STRICTLY FORBIDDEN (LAZY GENERIC LABELS)**: Never collapse the key syntactic marker into an abstract bracket!
         * ❌ `[Subordinator] + [Clause]` $\rightarrow$ ✔ `Although + [Clause], [Main Clause]`
         * ❌ `[Impersonal Frame] + [Clause]` $\rightarrow$ ✔ `It + [Copula] + [Past Participle] + that + [Clause]`
         * ❌ `[Participial Clause] + [Subject]` $\rightarrow$ ✔ `[Past Participle] + [Complement], [Subject] + [Predicate]`
         * ❌ `[Discourse Organizer] + [Quotation]` $\rightarrow$ ✔ `As + [Noun Phrase] + goes, [Quotation]`
     * **Syntactic Constituents Only (STRICT BAN on Semantic Slots)**: Slots in brackets `[...]` must represent grammatical/syntactic categories ONLY.
       - ✔ *Allowed Constituents*:
         - Nominal: `[Subject]`, `[Noun Phrase]`, `[Object]`, `[Complement]`
         - Verbal & Predicative: `[Base Verb]`, `[Past Participle]`, `[Gerund]`, `[Copula]`, `[Predicate]`
         - Modifiers: `[Adjective]`, `[Evaluative Adjective]`, `[Comparative]`, `[Adverb]`, `[Prepositional Phrase]`
         - Clausal: `[Clause]`, `[Main Clause]`, `[Subordinate Clause]`
       - ❌ *Strictly Forbidden*: `[Idea]`, `[Reason]`, `[Thing]`, `[Action]`, `[Information]`, `[Message]` or any conceptual/meaning-based placeholder.
     * **Preserve Surface Word Order**: Follow the exact linear surface order of the quote.
     * **Clean Discrete Formulas with Standard Terminal Slots**:
       - Connect constituents using `+` symbols.
       - 💡 **Terminal Slots Instead of Ellipses**: Never leave dangling trailing dots or ellipses (e.g., ❌ `[Subject] + [Verb]...`). Instead, use clean, self-contained terminal constituents like `[Clause]` or `[Predicate]` to represent the rest of the sentence cleanly (e.g., ✔ `It + [Copula] + [Adverb] + [Adjective] + that + [Clause]`, ✔ `Predictably, the + [Comparative] + [Clause], the + [Comparative] + [Clause]`).
       - ❌ **NO descriptive prose**: Write `[Past Participle] + [Noun Phrase]`, NEVER `Past Participle Phrase`.
     * ❌ *Strictly Prohibit Generic Formulas*: `[Noun] + [Verb] + [Noun]` or `[Clause], [Clause]` are completely banned.

5. **Original Imitation Sentence with Coherent Logic**:
   - `imitation_example` must be a high-quality, intellectually mature original academic sentence demonstrating the formula in a completely different context with flawless semantic logic.

6. **ESL Learner Insight**:
   - `common_mistakes` must diagnose concrete ESL errors (e.g., misordered inversion, dangling participles, missing concessive subordinators, comma splices without coordinators, incorrect aspect in non-finite forms).

7. **Syntactic Design Audit & Strict Identity (`design_audit`)**:
   - In `design_audit`, record the syntactic derivation as a SINGLE compact pipeline string matching the exact syntax below (do not include conversational filler like "I think" or discursive explanations):
     `AUDIT: [Verbatim Excerpt] -> [Tier Priority: Tier-1/Tier-2/Tier-3] -> [Category] -> [Diagnostic Anchor/Marker] -> [Target Formula with Slots]`
   - ⚠️ **STRICT IDENTITY & DIRECT COPY-PASTE MANDATE**:
     * `pattern_formula` MUST be a direct, literal copy-paste of the exact formula derived in Step 5 of `design_audit`.
     * Do NOT re-abstract, do NOT re-encode literal anchor words back into brackets, and do NOT alter a single character between them!

CONTENT:
{content}
