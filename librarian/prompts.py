import os
import re
import json
import dataclasses
from pathlib import Path
from typing import List, Literal, Any, Dict, Type, Union, Tuple
from .config import config
from .schemas import (
    get_json_schema,
    VocabularyExtraction,
    ExpressionsExtraction,
    GrammarExtraction,
    SummaryExtraction,
    VocabularyQuiz,
    ReadingQuiz,
    TranslationQuiz,
    ListeningQuiz,
    VideoQuiz,
    MindMapExtraction,
    QuizQualityAuditReport,
)

class Prompts:
    # Internal hard-coded schemas (Fallback)
    INTERNAL_SCHEMAS = {
        "extract_vocabulary": VocabularyExtraction,
        "extract_expressions": ExpressionsExtraction,
        "extract_grammar": GrammarExtraction,
        "extract_summary": SummaryExtraction,
        "extract_mindmap": MindMapExtraction,
        "vocabulary_quiz": VocabularyQuiz,
        "reading_quiz": ReadingQuiz,
        "translation_quiz": TranslationQuiz,
        "listening_quiz": ListeningQuiz,
        "video_quiz": VideoQuiz,
        "expert_audit": QuizQualityAuditReport,
    }

    @classmethod
    def get(cls, name: str) -> Tuple[str, Union[Type, Dict]]:
        """
        Dynamically loads a prompt and schema directly from librarian/prompts/.
        1. Loads .md prompt from librarian/prompts/<name>.md.
        2. Loads .json schema from librarian/prompts/<name>.json if present.
        3. Falls back to internal Python dataclass if .json is missing.
        """
        md_path = Path(__file__).parent / "prompts" / f"{name}.md"
        json_path = Path(__file__).parent / "prompts" / f"{name}.json"

        # 1. Load Prompt Text (.md)
        prompt_text = ""
        if md_path.exists():
            with open(md_path, "r", encoding="utf-8") as f:
                prompt_text = f.read()
            # Strip YAML if present (for backward compatibility)
            if prompt_text.startswith("---"):
                prompt_text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", prompt_text, flags=re.DOTALL)
        
        # 2. Load Schema (.json)
        schema_out = cls.INTERNAL_SCHEMAS.get(name) # Default to internal dataclass
        
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    schema_out = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load JSON schema from {json_path}: {e}")

        return prompt_text.strip(), schema_out

    @classmethod
    def sync_factory_from_code(cls) -> Tuple[bool, str]:
        """
        Regenerates the internal factory .json and .md files based on the 
        dataclasses defined in schemas.py. Writes files only if content changes
        to preserve modification timestamps.
        """
        factory_dir = Path(__file__).parent / "prompts"
        factory_dir.mkdir(parents=True, exist_ok=True)
        
        files_updated = 0
        for name, dataclass_cls in cls.INTERNAL_SCHEMAS.items():
            # 1. Generate JSON Schema
            schema_dict = get_json_schema(dataclass_cls, include_descriptions=False)
            schema_str = json.dumps(schema_dict, indent=2, ensure_ascii=False)
            json_path = factory_dir / f"{name}.json"
            
            json_changed = True
            if json_path.exists():
                try:
                    existing_json = json_path.read_text(encoding="utf-8")
                    existing_dict = json.loads(existing_json)
                    if existing_dict == schema_dict:
                        json_changed = False
                except Exception:
                    pass
            
            if json_changed:
                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(schema_str)
                files_updated += 1
            
            # 2. Generate Clean Markdown Template
            md_path = factory_dir / f"{name}.md"
            template_text = cls.get_default_template(name)
            
            md_changed = True
            if md_path.exists():
                try:
                    existing_md = md_path.read_text(encoding="utf-8")
                    if existing_md.strip() == template_text.strip():
                        md_changed = False
                except Exception:
                    pass
            
            if md_changed:
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(template_text)
                files_updated += 1
            
        return True, f"Successfully synchronized {files_updated} factory files from Python code in {factory_dir}"

    @staticmethod
    def get_default_template(name):
        """Returns the absolute minimum baseline template. Logic is entirely in schemas."""
        templates = {
            "extract_vocabulary": (
                "### SYSTEM ###\n"
                "You are an expert Lexicographer and ESL Curriculum Developer specializing in the Common European Framework of Reference for Languages (CEFR) and the Academic Word List (AWL).\n\n"
                "### USER ###\n"
                "Extract academic vocabulary from the text.\n\n"
                "### CORE PEDAGOGICAL MANDATES:\n"
                "1. **Target Count & Quality over Quota**: Extract up to {count} vocabulary words if the text allows. Quality > Quota: Extract genuine words that physically exist in the text. NEVER pad the list with hallucinated words or duplicates to reach {count}.\n"
                "2. **Absolute Verbatim Sourcing (STRICT ANTI-HALLUCINATION MANDATE)**: ❌ ZERO HALLUCINATION: Every single target word/lemma MUST be derived directly from a surface word physically present in the source text. NEVER invent, infer, or import external words that do not appear in the text. `quoted_sentence` MUST be an exact, unedited verbatim sentence from the source text where the surface form appears. The target word (or its direct inflection) MUST be explicitly present in `quoted_sentence`.\n"
                "3. **Academic Word List (AWL) & High-Register Priority**: Prioritize words that belong to the Academic Word List (AWL) or represent high-utility CEFR B1–C2 vocabulary THAT ACTUALLY APPEAR IN THE TEXT. If the source text is an essay, narrative, or non-technical piece, strictly identify the formal, analytical, or thematic academic register words actually used by the author—do NOT import external AWL words.\n"
                "4. **Lemmatization & Exact Part of Speech (PoS)**: Convert inflected surface forms to base lemma form. Headword PoS must match context (noun, verb, adjective, adverb, preposition, conjunction, interjection).\n"
                "5. **Absolute Uniqueness & Distinct Definitions**: Every entry must be completely distinct. Every word must have an accurate, unique definition contextualized to the passage. NEVER copy-paste identical definitions across different headwords.\n"
                "6. **Contextual Accuracy & Original Usage**: `example_usage` must be an original, high-quality sample sentence demonstrating academic usage.\n"
                "7. **LEMMA & CEFR DESIGN AUDIT (`design_audit`)**: Pipeline: `AUDIT: [Surface Word in Text] -> [Base Lemma Headword] -> [Exact Contextual PoS] -> [CEFR Level (B1–C2)] -> [VERBATIM_CONFIRMED]`.\n\n"
                "CONTENT:\n{content}\n"
            ),
            "extract_expressions": (
                "### SYSTEM ###\n"
                "You are an expert Lexicographer, ESL Curriculum Developer, and Idiomatic English Assessment Designer specializing in phraseology, multi-word units, and CEFR language assessment.\n\n"
                "### USER ###\n"
                "Extract genuine multi-word expressions (phrasal verbs, idioms, fixed collocations, and set phrases) from the text.\n\n"
                "### CORE PEDAGOGICAL MANDATES:\n"
                "1. **Target Count & Quality over Quota**:\n"
                "   - Identify and extract up to {count} high-value multi-word expressions from the text. NEVER return an empty expressions list (`[]`).\n"
                "   - Extract only genuine expressions found; never fabricate expressions to meet a quota.\n\n"
                "2. **Multi-Word Authenticity**:\n"
                "   - Every entry MUST be an inherently multi-word lexical unit (minimum 2 core words).\n"
                "   - Standalone single verbs are strictly prohibited—adding a generic slot like '[verb] [something]' does NOT make it an expression (single verbs belong exclusively to vocabulary extraction).\n\n"
                "3. **Rigorous Linguistic Classification (`part_of_speech`)**:\n"
                "   - 'phrasal verb': Verb + particle/preposition unit.\n"
                "   - 'collocation': Fixed multi-word pairing, especially Verb + Noun/Object + Preposition.\n"
                "   - 'set phrase': Fixed structural chunk or idiomatic connective.\n"
                "   - 'idiom': Fixed figurative expression with non-compositional meaning.\n\n"
                "4. **CANONICAL BASE FORM & MANDATORY ASSIGNMENT TO `word`**:\n"
                "   - In this schema, treat `word` as the dictionary headword of the multi-word expression.\n"
                "   - Abstract variable arguments into canonical bracketed slots:\n"
                "     * `[something]` / `[somebody]` for direct/indirect argument slots.\n"
                "     * `[entity]`, `[domain]`, or `[factor]` for formal/academic collocation slots.\n"
                "     * `[one's]` for possessive variable modifier slots.\n"
                "   - ⚠️ **DIRECT FIELD ASSIGNMENT MANDATE**:\n"
                "     * The `word` field MUST receive the exact canonical base form WITH ALL BRACKETED SLOTS INCLUDED.\n"
                "     * ❌ NEVER output inflected surface forms (convert inflected verbs to their base dictionary lemma).\n"
                "     * ❌ NEVER strip necessary variable slots from `word`.\n\n"
                "5. **PHRASEOLOGICAL AUDIT & COPY PIPELINE (`design_audit`)**:\n"
                "   - In `design_audit`, execute the canonical derivation pipeline:\n"
                "     `AUDIT: [Surface Excerpt in Text] -> Canonical Slotted Form -> Category -> VERBATIM_CONFIRMED`\n"
                "   - Rules:\n"
                "     1. Only wrap the first segment `[Surface Excerpt in Text]` in square brackets to quote the exact text from the article.\n"
                "     2. Keep intermediate category and confirmation labels clean without outer brackets.\n"
                "     3. 🔗 **PIPELINE BINDING MANDATE**: The exact Canonical Slotted Form derived in `design_audit` MUST BE COPIED VERBATIM into the `word` field!\n\n"
                "6. **Absolute Verbatim Sourcing in `quoted_sentence` (ZERO HALLUCINATION & NO PROMPT COPYING)**:\n"
                "   - ❌ **NEVER EXTRACT OR COPY PROMPT EXAMPLES**: Any examples in these prompt instructions are purely illustrative. If an expression does not physically exist in the source text under CONTENT, extracting it is an instant hallucination violation!\n"
                "   - The core invariant lexical elements of the expression MUST physically appear in `quoted_sentence`.\n"
                "   - `quoted_sentence`: Must contain the exact verbatim sentence from the source text where the expression appears.\n"
                "   - `example_usage`: Must be an original, natural sample sentence demonstrating communicative usage in a novel context.\n\n"
                "CONTENT:\n{content}\n"
            ),
            "extract_grammar": (
                "### SYSTEM ###\n"
                "You are an expert Pedagogical Grammar Analyst and Applied Linguist specializing in advanced academic English syntax.\n\n"
                "### USER ###\n"
                "Extract unique advanced grammar patterns from the text.\n\n"
                "### CORE PEDAGOGICAL MANDATES:\n\n"
                "1. **Verbatim Evidence & Authentic Sourcing**:\n"
                "   - Every `quote` must be an exact, unedited verbatim excerpt from the source text demonstrating the grammar structure.\n"
                "   - No paraphrasing, no trimming that alters structure, and no invented evidence.\n\n"
                "2. **Quality Over Quantity & Syntactic Prestige Hierarchy**:\n"
                "   - Extract between 3 and {count} truly distinct, genuine advanced patterns (max 1–2 per category).\n"
                "   - 🛡️ **DO NOT FORCE UNREPRESENTED CATEGORIES**: If a text genuinely contains only 3-4 distinct advanced mechanisms, return only those genuine patterns! NEVER force-fit non-existent categories.\n"
                "   - ⭐ **SYNTACTIC PRESTIGE HIERARCHY (PRIORITIZE HIGH-VALUE PATTERNS)**:\n"
                "     Always actively scan and give highest extraction priority to prestigious, intellectually mature syntactic mechanisms:\n"
                "     1. **Tier 1 (High Priority - MUST EXTRACT IF PRESENT)**: Authentic **Cleft sentences** (`It was [X] that...`), **True Inversion** (`Not only did...`, `Had I...`), **Complex Participial Fronting** (`Having seen...`, `Considered one of...`), **Evaluative It-frameworks** (`It is essential that...`).\n"
                "     2. **Tier 2 (Core Advanced)**: **Concessive clauses** (`Although...`, `Despite...`), **Rhetorical parallelism** (`The more... the more...`), **Abstract frames / Hedging devices** (`Given that...`, `It seemed that...`).\n"
                "     3. **Tier 3 (Fallback Only)**: Simple infinitives of purpose (`... to help people`) or basic coordination. Never pick Tier 3 if Tier 1 or Tier 2 structures are available in the passage!\n\n"
                "3. **Rigorous Category Classification & Anti-Hallucination Guardrails**:\n"
                "   - Assign a category ONLY if the quote contains a genuine instance:\n"
                "     * *Concessive clauses*: must include explicit concessive subordinators or prepositions (*although, though, while, despite, even though*).\n"
                "     * *Inversion*: must include true subject-auxiliary/copula inversion.\n"
                "       - ❌ **NEGATIVE GUARDRAIL (FAKE INVERSION)**: The transitional phrase 'Not only that, but [SVO]' is a correlative coordination, NOT syntactic inversion! True inversion requires auxiliary movement (e.g., 'Not only did he run...', 'Never had she seen...'). If no true inversion exists in the text, DO NOT classify anything as Inversion!\n"
                "     * *Participial clauses / Non-finite structures*: must contain genuine non-finite structures (*having + past participle*, *V-ing*, *past participle modifier*, infinitival extraposition) modifying a clause element.\n"
                "       - ❌ **NEGATIVE GUARDRAIL (TRIVIAL PREPOSITIONAL PHRASE)**: Trivial phrases like 'without sleeping' alone are NOT advanced patterns. Extract the complete host clause (e.g. '[Subject] + [Predicate] + without + [Gerund] + [Object]').\n"
                "     * *Evaluative It-frameworks*: must contain *It + copula + evaluative adjective/noun phrase + that-clause/infinitive* (e.g., 'It is essential/remarkable that...').\n"
                "       - ❌ **NEGATIVE GUARDRAIL (FAKE EVALUATIVE IT)**: Passive reporting structures like 'It was said/believed that...' are impersonal reporting (Hedging devices), NOT evaluative adjective frameworks!\n"
                "     * *Hedging devices*: epistemic modals, probability adverbs, or impersonal reporting frames (*It seemed that...*, *It was reported that...*).\n"
                "     * *Abstract frames*: abstract nouns functioning as discourse organizers (*the fact that, the idea that, the reality is that*).\n"
                "     * *Cleft sentences*: authentic cleft scaffolds (*It is/was + [Focus Element] + that/who...*, *What [Clause] is/was...*).\n"
                "     * 💡 **NESTED PRESTIGE RULE (HOST CLAUSE COLLISION)**:\n"
                "       - If an authentic **Cleft Sentence** (`It was [Focus Element] that...`) or **Inversion** is embedded inside a compound sentence (e.g. preceded by a concessive clause like `Though...`), **ALWAYS prioritize and classify it as `Cleft sentences` (or `Inversion`)**, NOT as a simple concessive clause!\n"
                "       - ❌ *Wrong*: classify as Concessive and reduce to `Though [Clause], [Main Clause]` (hides the cleft!).\n"
                "       - ✔ *Correct*: classify as `Cleft sentences` and explicitly expand the scaffold: `Though [Clause], it was + [Adverb] + [Noun Phrase] + that + [Clause]`.\n\n"
                "4. **High-Precision Pattern Formula Rules (MANDATORY)**:\n"
                "   - Formulate `pattern_formula` as a structural blueprint encoding the pattern's distinctive syntactic signature:\n"
                "     * **MANDATORY LEXICAL ANCHOR MANDATE (Literal Functional Words)**:\n"
                "       - The core structural marker, subordinator, preposition, or framework anchor MUST be written as **literal text outside brackets**.\n"
                "       - ❌ **STRICTLY FORBIDDEN (LAZY GENERIC LABELS)**: Never collapse the key syntactic marker into an abstract bracket!\n"
                "         * ❌ `[Subordinator] + [Clause]` $\\rightarrow$ ✔ `Although + [Clause], [Main Clause]`\n"
                "         * ❌ `[Impersonal Frame] + [Clause]` $\\rightarrow$ ✔ `It + [Copula] + [Past Participle] + that + [Clause]`\n"
                "         * ❌ `[Participial Clause] + [Subject]` $\\rightarrow$ ✔ `[Past Participle] + [Complement], [Subject] + [Predicate]`\n"
                "         * ❌ `[Discourse Organizer] + [Quotation]` $\\rightarrow$ ✔ `As + [Noun Phrase] + goes, [Quotation]`\n"
                "     * **Syntactic Constituents Only (STRICT BAN on Semantic Slots)**: Slots in brackets `[...]` must represent grammatical/syntactic categories ONLY.\n"
                "       - ✔ *Allowed Constituents*:\n"
                "         - Nominal: `[Subject]`, `[Noun Phrase]`, `[Object]`, `[Complement]`\n"
                "         - Verbal & Predicative: `[Base Verb]`, `[Past Participle]`, `[Gerund]`, `[Copula]`, `[Predicate]`\n"
                "         - Modifiers: `[Adjective]`, `[Evaluative Adjective]`, `[Comparative]`, `[Adverb]`, `[Prepositional Phrase]`\n"
                "         - Clausal: `[Clause]`, `[Main Clause]`, `[Subordinate Clause]`\n"
                "       - ❌ *Strictly Forbidden*: `[Idea]`, `[Reason]`, `[Thing]`, `[Action]`, `[Information]`, `[Message]` or any conceptual/meaning-based placeholder.\n"
                "     * **Preserve Surface Word Order**: Follow the exact linear surface order of the quote.\n"
                "     * **Clean Discrete Formulas with Standard Terminal Slots**:\n"
                "       - Connect constituents using `+` symbols.\n"
                "       - 💡 **Terminal Slots Instead of Ellipses**: Never leave dangling trailing dots or ellipses (e.g., ❌ `[Subject] + [Verb]...`). Instead, use clean, self-contained terminal constituents like `[Clause]` or `[Predicate]` to represent the rest of the sentence cleanly (e.g., ✔ `It + [Copula] + [Adverb] + [Adjective] + that + [Clause]`, ✔ `Predictably, the + [Comparative] + [Clause], the + [Comparative] + [Clause]`).\n"
                "       - ❌ **NO descriptive prose**: Write `[Past Participle] + [Noun Phrase]`, NEVER `Past Participle Phrase`.\n"
                "     * ❌ *Strictly Prohibit Generic Formulas*: `[Noun] + [Verb] + [Noun]` or `[Clause], [Clause]` are completely banned.\n\n"
                "5. **Original Imitation Sentence with Coherent Logic**:\n"
                "   - `imitation_example` must be a high-quality, intellectually mature original academic sentence demonstrating the formula in a completely different context with flawless semantic logic.\n\n"
                "6. **ESL Learner Insight**:\n"
                "   - `common_mistakes` must diagnose concrete ESL errors (e.g., misordered inversion, dangling participles, missing concessive subordinators, comma splices without coordinators, incorrect aspect in non-finite forms).\n\n"
                "7. **Syntactic Design Audit & Strict Identity (`design_audit`)**:\n"
                "   - In `design_audit`, record the syntactic derivation as a SINGLE compact pipeline string matching the exact syntax below (do not include conversational filler like \"I think\" or discursive explanations):\n"
                "     `AUDIT: [Verbatim Excerpt] -> [Tier Priority: Tier-1/Tier-2/Tier-3] -> [Category] -> [Diagnostic Anchor/Marker] -> [Target Formula with Slots]`\n"
                "   - ⚠️ **STRICT IDENTITY & DIRECT COPY-PASTE MANDATE**:\n"
                "     * `pattern_formula` MUST be a direct, literal copy-paste of the exact formula derived in Step 5 of `design_audit`.\n"
                "     * Do NOT re-abstract, do NOT re-encode literal anchor words back into brackets, and do NOT alter a single character between them!\n\n"
                "CONTENT:\n{content}\n"
            ),
            "extract_summary": (
                "### SYSTEM ###\n"
                "You are an expert Reading Specialist and Educational Content Developer.\n"
                "### USER ###\n"
                "Perform a comprehensive thematic and structural analysis of the text for educational use.\n\n"
                "MANDATE:\n"
                "1. Provide a cohesive summary of the text's narrative plot, storyline, or main arguments.\n"
                "2. Assess and assign `overall_cefr_level` (A1 to C2).\n"
                "3. Identify up to {count} distinct core concepts or topics from the text.\n"
                "4. For each concept, extract its specific details and list 1-3 related concept titles.\n"
                "5. FOR 'related_connections': Provide concise, 1-4 word short topic titles (e.g., 'Personal Growth', 'Media Literacy', 'Intentional Living'). DO NOT write full sentences, explanations, or 'Connect to:' prefixes.\n\n"
                "CONTENT:\n{content}\n"
            ),
            "vocabulary_quiz": (
                "### SYSTEM ###\n"
                "You are an expert ESL Lexical Assessment Specialist who designs CEFR-aligned, fair, and diagnostically rigorous vocabulary assessments (TOEFL/IELTS/Cambridge standards).\n\n"
                "### USER ###\n"
                "Create a high-quality multiple-choice vocabulary assessment from the supplied vocabulary list.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Count & Coverage**:\n"
                "   - Generate EXACTLY {count} questions testing {count} unique items exclusively from the supplied list. No duplicates, derivatives, or fabricated targets.\n"
                "   - ⚠️ **ACTIVE TARGET USAGE MANDATE**: `target_word` MUST match `options[correct_answer_index]` character-for-character. The word placed into the blank `____` must perfectly agree with the declared target's part of speech and required inflectional form.\n\n"
                "2. **Question (Contextual & Structural Anchoring)**:\n"
                "   - Write a brand-new compound/complex academic sentence at CEFR {cefr_level} containing a subordinate or coordinate clause (e.g., concession, condition, cause, or contrast).\n"
                "   - Use strictly four underscores `____` for the blank (no quotation marks around question). NEVER copy or adapt any sentence (Quoted Sentence or Example Usage) from the input.\n"
                "   - 🎯 **Strict Part-of-Speech Slot Matching**: The blank (____) MUST grammatically require the exact part of speech and syntactic role of the target word. If the target is a noun, the blank must strictly require a noun (e.g., 'The ____ of the...'). Do NOT place a noun into a verb or adjective slot.\n"
                "   - ⚓ **MANDATORY CONTEXTUAL & COLLOCATIONAL ANCHORS**:\n"
                "     * Every sentence MUST feature clear, objective context clues (e.g., explicit dependent prepositions like *to / on / of / for*, fixed verb-noun collocations, or unmistakable cause-and-effect / contrastive logic).\n"
                "     * Single-fit validity is absolute: the sentence context must mathematically rule out all 3 distractors on objective structural or logical grounds, NEVER on subjective 'register' or 'formality' differences.\n\n"
                "3. **Options (Target & 3 Structured Objective Distractors)**:\n"
                "   - **Target**: `target_word` must strictly equal `options[correct_answer_index]`. Multi-word units must be tested as indivisible wholes.\n"
                "   - **Grammatical Homogeneity & Authenticity**: All 4 options must be grammatically correct, authentic English words or established expressions sharing the identical grammatical category (part of speech) and the EXACT inflection required by the blank (e.g., all past participles `-ed`, all plurals `-s`, all `-ing`).\n"
                "   - 🚫 **STRICT BAN ON SYNONYM PILES**:\n"
                "     * NEVER supply interchangeable synonyms. Distractors cannot merely differ by subtle tone or degree of formality.\n"
                "   - 🎯 **MANDATORY 3-VECTOR DISTRACTOR TAXONOMY**:\n"
                "     Each question's 3 distractors MUST consist of:\n"
                "     1. *Trap 1 (Antonym / Logical Polarity Clash)*: directly contradicts the cause/contrast/concession logic established in the sentence clues.\n"
                "     2. *Trap 2 (Collocation / Syntax Clash)*: plausible meaning in the general topic, but violates the blank's dependent preposition, verb valency, or conventional lexical pairing.\n"
                "     3. *Trap 3 (Domain / Semantic Category Mismatch)*: shares the general educational/academic register, but denotes a completely distinct action, entity, or attribute unsuited to this specific functional role.\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: Follow this rigorous 4-part structure:\n"
                "     `AUDIT: [Target Word + Part of Speech + Required Form] -> [Sentence Clues & Syntactic Slot Anchor] -> [Trap 1 (Antonym/Polarity): ...] [Trap 2 (Collocation/Syntax Clash): ...] [Trap 3 (Domain Mismatch): ...] -> [Why Distractors Fail: Objective Ground-Truth Disqualifications]`\n"
                "   - `explanation`: State contrastive, objective reasoning explaining why the target fits and explicitly why each distractor is objectively disqualified (grammatical clash, preposition failure, or logical contradiction). You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.\n"
                "   - 🎲 **RANDOMIZED ANSWER KEY BALANCE**: Distribute `correct_answer_index` evenly across 0 (A), 1 (B), 2 (C), and 3 (D) throughout the quiz. Never place all correct answers on the same index.\n"
                "   - `definition`: Concise dictionary meaning of the target in this context.\n\n"
                "CONTENT:\n{vocabulary_content}\n"
            ),
            "reading_quiz": (
                "### SYSTEM ###\n"
                "You are an expert Reading Comprehension Assessment Designer.\n"
                "### USER ###\n"
                "Create a reading comprehension assessment based on the provided passage.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Count & Vocabulary**:\n"
                "   - Generate EXACTLY {count} comprehension questions.\n"
                "   - Extract 5 to 8 challenging academic vocabulary items with verbatim context sentences, parts of speech (noun, verb, adjective, adverb, preposition, conjunction, interjection), concise definitions, and authentic example sentences.\n\n"
                "2. **Question & Skill Diversity**:\n"
                "   - Cover a balanced mix of skills across questions: `Main Idea`, `Detail/Recall`, `Inference`, and `Author's Tone/Purpose`.\n"
                "   - Questions must require genuine comprehension of the text rather than superficial string-matching. Use clear phrasing without outer quotation marks.\n"
                "   - ⚠️ **VERBATIM TEXT ANCHOR MANDATE**: Every Detail/Recall and Inference question MUST be anchored to specific, verifiable statements in the passage. The correct answer must be supported by direct textual evidence, and the design audit must pinpoint the exact paragraph or sentence anchor.\n\n"
                "3. **Diagnostic Distractors (Structured Taxonomy)**:\n"
                "   - All 4 options must be plausible, grammatically parallel, and closely tied to the passage topic. No option labels (A, B) or quotes around options.\n"
                "   - ⚓ **ABSOLUTE SINGLE-FIT VALIDITY**: High diagnostic plausibility must NEVER create ambiguity. The question stem combined with the passage MUST provide definitive, objective textual evidence that makes the correct answer the ONLY defensible choice, while decisively eliminating all three distractors on factual, logical, or scope grounds without relying on subjective interpretation.\n"
                "   - ❌ **STRICTLY PROHIBIT**: Absurd/cartoonish extremes (e.g., 'ignore all warnings', 'destroy the planet'), trivial common-sense giveaways, and lazy binary opposites.\n"
                "   - Draw distractors from authentic reading traps:\n"
                "     * *Trap 1 (Literal Match Trap)*: borrows verbatim words or phrasing from the passage, but twists the logical relationship, cause-and-effect, or subject/object.\n"
                "     * *Trap 2 (Scope Shift Trap)*: overly broad, overly restrictive (extreme words like *always*, *only*, *solely*, *never*), or shifts the focus away from the question's premise.\n"
                "     * *Trap 3 (Plausible Distortion / False Inference)*: sounds factually reasonable in real-world knowledge, but is unsupported, unmentioned, or directly contradicted by the text.\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: Follow this rigorous 4-part structure:\n"
                "     `AUDIT: [Skill Category: Main Idea/Detail/Inference/Tone] -> [Textual Anchor (Paragraph # / Specific Quote)] -> [Trap 1 (Literal Match): ...] [Trap 2 (Scope Shift): ...] [Trap 3 (Distortion): ...] -> [Why Distractors Fail: Objective Ground-Truth Disqualifications]`\n"
                "   - `explanation`: State the exact text evidence for the correct answer, and contrastively explain why each distractor fails. You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.\n"
                "   - 🎲 **RANDOMIZED ANSWER KEY BALANCE**: Distribute `correct_answer_index` evenly across 0 (A), 1 (B), 2 (C), and 3 (D) throughout the quiz. Never place all correct answers on the same index.\n\n"
                "PASSAGE:\n"
                "{passage_content}\n"
            ),
            "translation_quiz": (
                "### SYSTEM ###\n"
                "You are an expert Pedagogical Assessment Specialist and Translator, designing rigorous {target_language}-to-English translation assessments for advanced ESL learners (CEFR B2-C1 standards, CET-6 / TEM-8 / IELTS / TOEFL translation level).\n"
                "### USER ###\n"
                "Create a {target_language}-to-English translation assessment that seamlessly integrates the provided vocabulary items and grammar pattern formulas.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Count & Integration (MANDATORY TARGET PRESENCE)**:\n"
                "   - Generate EXACTLY {count} translation questions in the 'questions' array.\n"
                "   - Each question MUST integrate:\n"
                "     * One target vocabulary item from the VOCABULARY list.\n"
                "     * One target grammar pattern from the GRAMMAR list (applying its `pattern_formula` slot structure).\n"
                "   - ⚠️ **ACTIVE USAGE MANDATE**: The declared target vocabulary item MUST be explicitly required by the {target_language} meaning in `translated_sentence` AND actively used verbatim (or with necessary inflection) inside `correct_english_answer`. It is STRICTLY FORBIDDEN to declare a vocabulary item in `design_audit` while omitting it from the correct English translation.\n"
                "   - Ensure diverse coverage without repeating vocabulary items, grammar patterns, or scenarios.\n\n"
                "2. **Original Academic Scenario (NO Copying Source Text)**:\n"
                "   - ❌ **STRICTLY PROHIBITED**: Copying, adapting, or echoing sentences from the input text or reading passage.\n"
                "   - Design a brand-new, intellectually mature academic or professional scenario (e.g., environmental policy, technology ethics, higher education, scientific research, socioeconomic development).\n"
                "   - `translated_sentence`: Provide a natural, polished, and formal {target_language} prompt sentence.\n"
                "   - `correct_english_answer`: Provide the pristine English translation demonstrating natural syntax, academic register, and precise application of the target grammar formula and vocabulary.\n\n"
                "3. **Options (All English) & L1 Interference Taxonomy**:\n"
                "   - ⚠️ **MANDATORY**: ALL four items in 'options' MUST be complete English sentences. NEVER put {target_language} sentences into 'options'.\n"
                "   - Return ONLY literal sentence text without choice labels ('A)', '1.') or wrapping quotation marks.\n"
                "   - 1 option is the `correct_english_answer` (matching `options[correct_answer_index]`).\n"
                "   - The other 3 options MUST model authentic learner errors driven by L1 ({target_language}) negative transfer:\n"
                "     * *Trap 1: Word-for-Word Literal Trap*: translates {target_language} word order mechanically, resulting in verb stacking, missing formal subjects, or unnatural topic-comment structures.\n"
                "     * *Trap 2: Collocation & Preposition Shift*: misuses prepositions or colligations driven by {target_language} semantic interference (e.g., *improve the problem*, *pay attention on*, *confront with*).\n"
                "     * *Trap 3: Structural & Formula Distortion*: subtly violates the target grammar pattern (e.g., failed subject-verb inversion, dangling participle, comma splice without coordinator, or tense/aspect flaw).\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: Follow this rigorous 4-part bilingual structure:\n"
                "     `AUDIT: [{target_language} Core Anchor -> Target Vocab + Grammar Formula] -> [Academic Scenario] -> [Trap 1 (Literal L1 transfer): ...] [Trap 2 (Collocation/Preposition): ...] [Trap 3 (Formula Distortion): ...] -> [Why Distractors Fail]`\n"
                "   - `hint`: Concise pedagogical hint highlighting the key grammatical structure or functional phrase.\n"
                "   - `explanation`: Contrastively explain why the correct English translation is superior and explicitly identify the specific grammatical or stylistic flaw in each distractor. You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.\n"
                "   - 🎲 **RANDOMIZED ANSWER KEY BALANCE**: Distribute `correct_answer_index` evenly across 0 (A), 1 (B), 2 (C), and 3 (D) throughout the quiz. Never place all correct answers on the same index.\n\n"
                "VOCABULARY:\n{vocabulary_content}\n\n"
                "GRAMMAR:\n{grammar_content}\n"
            ),
            "listening_quiz": (
                "### SYSTEM ###\n"
                "You are an expert ESL Audio Script Writer and Listening Assessment Designer (TOEFL / IELTS / Cambridge English standards).\n"
                "### USER ###\n"
                "Generate a realistic academic dialogue and a rigorous comprehension assessment based on the provided vocabulary items.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Authentic Academic Dialogue Script**:\n"
                "   - Create a natural, engaging academic discussion between Speaker 1 and Speaker 2 consisting of 6 to 8 conversational turns.\n"
                "   - Natural spoken register: realistic conversational flow with authentic discourse markers (e.g., 'Well, look at it this way...', 'That's a valid point, but...', 'Wait, are you saying...?'), gentle counter-arguments, and mutual clarification.\n"
                "   - Seamlessly embed at least 5 target academic vocabulary items into natural spoken contexts without sounding like textbook recitations.\n"
                "   - ⚠️ **SPEAKER ATTRIBUTION RIGOR MANDATE**: Clearly distinguish the roles, stances, and insights of Speaker 1 vs Speaker 2. When a question asks about a specific speaker's viewpoint, concern, or proposal (e.g., 'What does Speaker 1 suggest...?'), the correct answer MUST be based exclusively on that speaker's dialogue turns, NOT the conversational partner's statements.\n\n"
                "2. **Question & Skill Diversity**:\n"
                "   - Generate EXACTLY {count} comprehension questions in the 'questions' array.\n"
                "   - Cover a balanced mix of listening skills across questions:\n"
                "     * `Detail`: Specific fact, limitation, or rationale stated by a specific speaker.\n"
                "     * `Inference`: Drawing logical conclusions directly implied by the dialogue.\n"
                "     * `Main Idea`: Overall core purpose or consensus takeaway of the conversation.\n\n"
                "3. **Listening Distractor Taxonomy (NO Cartoonish Choices)**:\n"
                "   - All 4 options must be plausible, concise, grammatically parallel, and closely tied to the discussion.\n"
                "   - ⚓ **ABSOLUTE SINGLE-FIT VALIDITY**: High diagnostic plausibility must NEVER create ambiguity. The question stem combined with the specific dialogue turn MUST provide definitive conversational evidence (clear speaker attribution, explicit conditions, or established consensus) that makes the correct answer the ONLY defensible choice, while decisively eliminating all three distractors without relying on subjective conjecture.\n"
                "   - ❌ **STRICTLY PROHIBIT**: Childish or absurd choices (e.g., 'machines are too heavy to move', 'destroy all electronics'), trivial common-sense giveaways, and pure polar opposites.\n"
                "   - Engineer distractors using authentic listening test cognitive traps:\n"
                "     * *Trap 1 (Speaker Attribution Swap)*: Attributes an opinion, concern, or proposal to Speaker 1 when it was actually expressed or qualified by Speaker 2 (or vice versa).\n"
                "     * *Trap 2 (Verbatim Catch Trap)*: Borrows an eye-catching technical term from the script (e.g., 'nuclear fusion', 'semiconductors'), but links it to a false claim or unmentioned context.\n"
                "     * *Trap 3 (Overstated Generalization)*: Uses extreme absolutes (*completely impossible*, *abandon entirely*, *useless*) when the speaker only expressed cautious reservation or conditional qualification.\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: Follow this rigorous 4-part structure:\n"
                "     `AUDIT: [Skill Category: Detail/Inference/Main Idea] -> [Dialogue Anchor: Speaker Name, Turn # (Verbatim Clue)] -> [Trap 1 (Speaker Swap): ...] [Trap 2 (Verbatim Catch): ...] [Trap 3 (Overstatement): ...] -> [Why Distractors Fail: Objective Dialogue Disqualifications]`\n"
                "   - `explanation`: State the exact dialogue turn and speaker supporting the correct answer, and contrastively explain why each distractor trap is invalid. You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.\n"
                "   - 🎲 **RANDOMIZED ANSWER KEY BALANCE**: Distribute `correct_answer_index` evenly across 0 (A), 1 (B), 2 (C), and 3 (D) throughout the quiz. Never place all correct answers on the same index.\n"
                "   - `correct_answer_index`: MUST be an integer 0, 1, 2, or 3 matching the exact position of the true answer in 'options'. Do NOT use alternative key names.\n"
                "   - Do NOT wrap question or option text in quotes or labels (A, B).\n\n"
                "VOCABULARY:\n{vocabulary_content}\n"
            ),
            "video_quiz": (
                "### SYSTEM ###\n"
                "You are an expert Video-Based ESL Assessment Designer (TOEFL / IELTS / Academic Documentary standards).\n"
                "### USER ###\n"
                "Create a timestamp-aware video comprehension quiz based on the provided video transcript.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Exact Question Count & Chronological Timestamp Coverage**:\n"
                "   - Generate EXACTLY {count} timestamp-aware video questions in the 'questions' array.\n"
                "   - Distribute questions evenly across the chronological timeline of the video (e.g. Early context/mechanisms, Middle engineering challenges/environmental impacts, Later controversies/future upgrades).\n"
                "   - Every question must map to a specific timestamp present verbatim in the transcript (e.g. [01:25.10] or [07:46.50]) where the evidence is clearly discussed.\n\n"
                "2. **Higher-Order Video Comprehension (NO Trivial Number Recall)**:\n"
                "   - ❌ **BAN TRIVIAL NUMBER GUESSING**: Do NOT write pure numeric recall questions (e.g., guessing between 500 tons vs 3000 tons, or 10 GW vs 22.5 GW).\n"
                "   - Focus on meaningful conceptual, causal, and analytical understanding:\n"
                "     * `Technical Mechanism & Cause-Effect`: Why a specific engineering solution was implemented, or how a natural condition affects operations.\n"
                "     * `Controversy & Argumentation`: Contrasting external criticisms or rumors with official explanations or engineering realities.\n"
                "     * `Comparative Analysis & Future Outlook`: Evaluating how the project compares with international counterparts or what future innovations (e.g., AI, railways) are proposed.\n\n"
                "3. **Video Distractor Taxonomy (STRICT BAN on Absurd Options)**:\n"
                "   - All 4 options must be plausible, grammatically parallel, and written in formal academic English.\n"
                "   - ⚓ **ABSOLUTE SINGLE-FIT VALIDITY**: High diagnostic plausibility must NEVER create ambiguity. The question stem combined with the specific timestamp context MUST provide definitive transcript evidence that makes the correct answer the ONLY defensible choice, while decisively eliminating all three distractors on factual, chronological, or scope grounds.\n"
                "   - ❌ **STRICTLY PROHIBIT**: Childish, absurd, or comical answers (e.g., 'Komodo dragons', 'the dam is made of steel', 'it is too small to hold water'), trivial common-sense giveaways, and pure polar opposites.\n"
                "   - Engineer distractors using authentic video assessment cognitive traps:\n"
                "     * *Cross-Timestamp Context Shift*: Borrows a legitimate fact or term from a different part of the video and falsely misapplies it to the target question.\n"
                "     * *Misattributed Claim / Rumor vs. Fact Trap*: Confuses an unsubstantiated rumor/criticism with verified facts, or misidentifies the official clarification.\n"
                "     * *Plausible Over-generalization*: Exaggerates a nuanced or seasonal trend into an absolute or universal claim.\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: `AUDIT: [Timestamp Segment] -> [Core Focus: Mechanism / Controversy / Comparison] -> [Traps: Cross-timestamp shift, Rumor vs fact, Over-generalization] -> [Why Distractors Fail]`\n"
                "   - `explanation`: State what the video explicitly clarifies at the given timestamp, and contrastively explain why each distractor trap is invalid. You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.\n"
                "   - 🎲 **RANDOMIZED ANSWER KEY BALANCE**: Distribute `correct_answer_index` evenly across 0 (A), 1 (B), 2 (C), and 3 (D) throughout the quiz. Never place all correct answers on the same index.\n"
                "   - `correct_answer_index`: MUST be an integer 0, 1, 2, or 3 matching the exact position of the true answer in 'options'.\n"
                "   - Do NOT wrap question or option text in quotes or labels (A, B).\n\n"
                "VIDEO TRANSCRIPT:\n{transcript_content}\n"
            ),
            "extract_mindmap": (
                "### SYSTEM ###\n"
                "You are an expert Educational Content Designer and Mind Mapping Specialist.\n"
                "### USER ###\n"
                "Construct a highly detailed hierarchical mind map representing the structural ideas, themes, and supporting details of the text.\n\n"
                "MANDATES:\n"
                "1. Establish a single central root theme summarizing the unit and output it in `root_name`.\n"
                "2. Extract exactly 3 to 5 distinct primary branches representing the major sub-themes or narrative stages. Assign each a unique, harmonious color theme from the allowed values.\n"
                "3. Consistent Hierarchical Structure: Every branch MUST contain one or more sub-branch objects. Each sub-branch has a clear `sub_branch_name` and a `leaves` array. If a branch covers only a single general topic, simply name its sub-branch 'Overview' or 'Key Points'.\n"
                "4. Concise, High-Impact Leaves: Leaf nodes should be concise bullet-point details, phrases, or short examples (aim for 3-10 words per leaf). DO NOT copy long, multi-line paragraphs as leaf nodes.\n\n"
                "CONTENT:\n{content}\n"
            ),
            "expert_audit": (
                "### SYSTEM ###\n"
                "You are an elite Psychometrician, Senior Applied Linguist, and Lead Assessment Auditor specializing in CEFR/TOEFL standardized language testing.\n\n"
                "### USER ###\n"
                "Conduct an exhaustive, high-reasoning pedagogical and psychometric Quality Audit on the supplied educational quiz items.\n\n"
                "### MANDATES FOR EXPERT AUDIT:\n\n"
                "0. **SYNTACTIC WELL-FORMEDNESS & GRAMMATICAL LEGITIMACY (MANDATORY GATE)**:\n"
                "   - Before evaluating meaning, verify that inserting the candidate answer produces an **impeccably grammatical and complete English sentence**.\n"
                "   - **ZERO-TOLERANCE DEFECTS**:\n"
                "     - **Missing Predicate Verb**: If the stem lacks a main finite verb and the target option is a noun/adjective (e.g. 'perseverance [triumph] as a testament'), the item is **FATALLY FLAWED**. You MUST set `single_fit_valid: false`, assign `pedagogical_score <= 40`, and explicitly flag 'Missing predicate verb / ungrammatical sentence' in `diagnostic_feedback`.\n"
                "     - **Severe Lexical/Collocation Tautology**: Phrasings that are unnatural or grammatically redundant (e.g. 'pledge a commitment' instead of 'make a commitment' or 'pledge to do') must be penalized severely.\n"
                "   - If ANY option causes a sentence fragment or grammatical breakdown, it CANNOT be considered a valid answer key.\n\n"
                "1. **BLIND TEST-SOLVER SIMULATION (`blind_solved_index` & `confidence`)**:\n"
                "   - Independently read the source text and each question stem with its 4 options (Option A = 0, B = 1, C = 2, D = 3).\n"
                "   - Determine the objectively correct answer based SOLELY on direct textual evidence from the passage.\n"
                "   - Set `confidence`: 'Definite', 'Hesitant', or 'Ambiguous'.\n\n"
                "2. **ABSOLUTE SINGLE-FIT VALIDITY (`single_fit_valid`)**:\n"
                "   - Verify that there is EXACTLY ONE uniquely correct, grammatically sound, and textually defensible answer.\n"
                "   - If two options can both be justified by the text (Double Key / Key Leak) OR if the declared key creates an ungrammatical sentence, flag `single_fit_valid: false`.\n\n"
                "3. **COGNITIVE DISTRACTOR TRAP ANALYSIS (`distractors`)**:\n"
                "   - Evaluate all 4 options, identifying authentic educational trap types and plausibility ratings.\n"
                "   - Provide concise elimination rationales explaining why test-takers must definitively reject distractors. Trivial giveaway filler (e.g. childish words 'study', 'experiment') must be flagged as 'Flawed / Trivial Giveaway'.\n\n"
                "4. **SCORING AND VERDICT (`overall_quality_score` & `pass_audit`)**:\n"
                "   - Assign `pedagogical_score` (0–100) per question: flawless = 90-100; minor weakness = 75-89; critical defects/missing verb/trivial distractors < 75.\n"
                "   - **PASS REQUIREMENT**: Set `pass_audit: true` ONLY IF `overall_quality_score >= 80` AND every question has `single_fit_valid == true` AND no question has grammatical/syntactic collapse. If even ONE question has a missing verb or invalid single fit, `pass_audit` MUST be `false`.\n\n"
                "CONTENT:\n{content}\n"
            ),
        }
        return templates.get(name, "Analyze the content.\n\nCONTENT:\n{content}\n")

    # Property redirects for compatibility (renamed to snake_case)
    @property
    def extract_vocabulary(self): return self.get("extract_vocabulary")[0]
    @property
    def extract_grammar(self): return self.get("extract_grammar")[0]
    @property
    def extract_summary(self): return self.get("extract_summary")[0]
