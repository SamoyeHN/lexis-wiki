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
                "   - Identify and extract up to {count} high-value multi-word expressions from the text.\n"
                "   - Extract only genuine expressions found; never fabricate items to meet a quota.\n\n"
                "2. **Multi-Word Authenticity**:\n"
                "   - Every entry MUST be an inherently multi-word lexical unit (minimum 2 core words, e.g., 'hinge on', 'pose a risk to').\n"
                "   - Standalone single verbs are strictly prohibited—adding a generic slot like 'launch [something]' does NOT make it an expression (single verbs belong exclusively to vocabulary extraction).\n\n"
                "3. **Rigorous Linguistic Classification (`part_of_speech`)**:\n"
                "   - 'phrasal verb': Verb + particle/preposition unit (e.g., 'hinge on').\n"
                "   - 'collocation': Fixed multi-word pairing, especially Verb + Noun/Object + Prep (e.g., 'pose a risk to [entity]').\n"
                "   - 'set phrase': Fixed structural chunk (e.g., 'for the time being').\n"
                "   - 'idiom': Fixed figurative expression with metaphorical meaning (e.g., 'keep one's chin up').\n\n"
                "4. **CANONICAL BASE FORM & MANDATORY ASSIGNMENT TO `\"word\"`**:\n"
                "   - In this schema, treat `\"word\"` as the dictionary headword of the multi-word expression.\n"
                "   - Variable arguments MUST be abstracted into standard bracketed slots:\n"
                "     * `[something]` / `[somebody]` for direct/indirect arguments (e.g., `take [something] for granted`, `put [something] out`).\n"
                "     * `[entity]`, `[domain]`, or `[factor]` for formal/academic collocations (e.g., `pose a risk to [entity]`).\n"
                "     * `one's` for possessive variable modifiers (e.g., `make up one's mind`).\n"
                "   - ⚠️ **DIRECT FIELD ASSIGNMENT MANDATE**:\n"
                "     * The `\"word\"` field MUST receive the exact canonical base form WITH ALL BRACKETED SLOTS INCLUDED.\n"
                "     * ❌ NEVER output the inflected surface text (e.g., write `\"word\": \"turn up at [location]\"`, NOT `\"turned up at\"`).\n"
                "     * ❌ NEVER strip slots from `\"word\"` (e.g., write `\"word\": \"take [something] for granted\"`, NOT `\"take for granted\"`).\n\n"
                "5. **PHRASEOLOGICAL AUDIT & COPY PIPELINE (`design_audit`)**:\n"
                "   - In `design_audit`, execute the 4-step canonical derivation:\n"
                "     `AUDIT: [Surface Text in Article] -> [Variable Arguments Abstracted into Slots] -> [Canonical Slotted Headword] -> [Category] -> [VERBATIM_CONFIRMED]`\n"
                "   - 🔗 **PIPELINE BINDING MANDATE**:\n"
                "     The [Canonical Slotted Headword] derived in step 3 of `design_audit` MUST BE COPIED VERBATIM into the `\"word\"` field!\n"
                "   - Paired Example 1:\n"
                "     `\"design_audit\": \"AUDIT: turned up at -> turn up at [location] -> phrasal verb -> VERBATIM_CONFIRMED\"`\n"
                "     `\"word\": \"turn up at [location]\"`\n"
                "   - Paired Example 2:\n"
                "     `\"design_audit\": \"AUDIT: putting out the blaze -> put [something] out -> phrasal verb -> VERBATIM_CONFIRMED\"`\n"
                "     `\"word\": \"put [something] out\"`\n"
                "   - Paired Example 3:\n"
                "     `\"design_audit\": \"AUDIT: took his kindness for granted -> take [something] for granted -> idiom -> VERBATIM_CONFIRMED\"`\n"
                "     `\"word\": \"take [something] for granted\"`\n\n"
                "6. **Absolute Verbatim Sourcing in `quoted_sentence` (ZERO HALLUCINATION)**:\n"
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
                "### CORE PEDAGOGICAL MANDATES:\n"
                "1. **Verbatim Evidence**: Every `quote` must be an exact, unedited verbatim excerpt from the source text demonstrating the grammar structure.\n"
                "2. **Category Diversity & Spread**: Select patterns across different categories (max 1–2 per category). Do not extract redundant structures.\n"
                "3. **Rigorous Classification & Anti-Patterns**: Assign a category ONLY if the quote contains a genuine instance. Rhetorical questions are NOT inversion. Concessive clauses MUST contain explicit concessive markers (although, though, while, despite, etc.).\n"
                "4. **Slot-Filling Pattern Formulas (Pedagogically Actionable Formulas)**:\n"
                "   - Formulate `pattern_formula` using fixed syntactic anchors combined with clear bracketed slots `[...]`.\n"
                "   - Keep structural keywords explicit and abstract flexible constituents into slots:\n"
                "     * Good examples:\n"
                "       - `Not only + [Auxiliary/Be] + [Subject] + [Main Verb], but also + [Clause]`\n"
                "       - `Having + [Past Participle] + [Object/Complement], [Main Subject] + [Main Predicate]`\n"
                "       - `Though + [Subordinate Clause], [Main Subject] + [Main Predicate]`\n"
                "       - `It is/was + [Emphasized Element] + that/who + [Remaining Clause]`\n"
                "     * ❌ PROHIBITED: Trivial or overly generic formulas (e.g. `[Noun] + [Verb] + [Noun]` or `[Clause], [Main Clause]`). A formula MUST highlight the distinctive grammatical mechanics.\n"
                "5. **Original Imitation Sentence with Coherent Logic**: `imitation_example` must be a high-quality original academic sentence demonstrating the formula in a distinct context with sound logical semantics.\n"
                "6. **ESL Learner Insight**: `common_mistakes` must explain typical learner errors (e.g., dangling participles, inversion word order errors, comma splices, tense mismatch).\n"
                "7. **SYNTACTIC DESIGN AUDIT (`design_audit`)**:\n"
                "   - In `design_audit`, execute the 4-step syntactic derivation:\n"
                "     `AUDIT: [Verbatim Excerpt] -> [Category] -> [Diagnostic Anchor/Marker] -> [Target Formula with Slots]`\n"
                "   - Example 1: `AUDIT: Having seen the boycott... -> Participial clauses -> Perfect participle clause (Having + V-ed) modifying main subject -> Having + [Past Participle] + [Object], [Main Subject] + [Predicate]`\n"
                "   - Example 2: `AUDIT: Though it took 10 years... -> Concessive clauses -> Subordinating conjunction 'Though' -> Though + [Clause], [Main Subject] + [Predicate]`\n"
                "   - 🔗 **MANDATORY**: Copy the derived slotted formula directly into `pattern_formula`.\n\n"
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
                "   - Generate EXACTLY {count} questions testing {count} unique items exclusively from the supplied list. No duplicates, derivatives, or fabricated targets.\n\n"
                "2. **Question**:\n"
                "   - Write a brand-new compound/complex academic sentence at CEFR {cefr_level} containing a subordinate or coordinate clause (e.g., concession, condition, cause, or contrast) to supply clear context clues.\n"
                "   - Use strictly four underscores `____` for the blank (no quotation marks around question). NEVER copy or adapt any sentence (Quoted Sentence or Example Usage) from the input.\n"
                "   - Ensure **single-fit validity**: the clause logic and collocational anchor must rule out all distractors.\n\n"
                "3. **Options (Target & Distractors)**:\n"
                "   - **Target**: `target_word` must strictly equal `options[correct_answer_index]`. Multi-word units must be tested as indivisible wholes.\n"
                "   - **Grammatical Homogeneity & Inflection**: All 4 options must share identical part of speech and EXACT inflection required by the blank (e.g., all past participles `-ed`, all plurals `-s`, all `-ing`). Never leave options in uninflected base forms if the blank requires inflected words.\n"
                "   - **Authentic Distractors (Structured Taxonomy)**:\n"
                "     Draw 3 plausible distractors from:\n"
                "     * *Near-synonym*: shares core meaning but fails in precise semantic nuance.\n"
                "     * *Collocation/Preposition Trap*: plausible word that violates the sentence's dependent preposition or collocational constraint.\n"
                "     * *Topic/Register Mate*: shares the domain field (or from unit text if inflected to match) but conveys the wrong function.\n"
                "     ❌ Prohibit binary positive/negative opposites and fabricated pseudo-idioms.\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: `AUDIT: [Target & Form] -> [Sentence Anchor] -> [3 Authentic Homogeneous Distractors] -> [Why Distractors Fail]`\n"
                "   - `explanation`: Give contrastive reasoning explaining why the target fits and why distractors fail (wrong collocation, nuance, or preposition).\n"
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
                "   - Questions must require genuine comprehension of the text rather than superficial string-matching. Use clear phrasing without outer quotation marks.\n\n"
                "3. **Diagnostic Distractors (Structured Taxonomy)**:\n"
                "   - All 4 options must be plausible, grammatically parallel, and closely tied to the passage topic. No option labels (A, B) or quotes around options.\n"
                "   - ❌ **STRICTLY PROHIBIT**: Absurd/cartoonish extremes (e.g., 'ignore all warnings', 'destroy the planet'), trivial common-sense giveaways, and lazy binary opposites.\n"
                "   - Draw distractors from authentic reading traps:\n"
                "     * *Literal Matching Trap*: borrows verbatim words or phrasing from the passage, but twists the logical relationship, cause-and-effect, or subject/object.\n"
                "     * *Scope Shift Trap*: overly broad, overly restrictive (extreme words like *always*, *only*, *solely*, *never*), or shifts the focus away from the question's premise.\n"
                "     * *Plausible Distortion / False Inference*: sounds factually reasonable in real-world knowledge, but is unsupported, unmentioned, or directly contradicted by the text.\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: `AUDIT: [Skill] -> [Text Anchor (e.g. Para 3)] -> [Distractor Traps: Literal Match / Scope Shift / Distortion] -> [Why Distractors Fail]`\n"
                "   - `explanation`: State the exact text evidence for the correct answer, and contrastively explain why each distractor fails.\n\n"
                "PASSAGE:\n"
                "{passage_content}\n"
            ),
            "translation_quiz": (
                "### SYSTEM ###\n"
                "You are an expert Pedagogical Assessment Specialist and Translator, designing rigorous Chinese-to-English translation assessments for advanced Chinese ESL learners (CEFR B2-C1 standards, CET-6 / TEM-8 / IELTS / TOEFL translation level).\n"
                "### USER ###\n"
                "Create a Chinese-to-English translation assessment that seamlessly integrates the provided vocabulary items and grammar pattern formulas.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Count & Integration**:\n"
                "   - Generate EXACTLY {count} translation questions in the 'questions' array.\n"
                "   - Each question MUST integrate:\n"
                "     * One target vocabulary item from the VOCABULARY list.\n"
                "     * One target grammar pattern from the GRAMMAR list (applying its `pattern_formula` slot structure).\n"
                "   - Ensure diverse coverage without repeating vocabulary items, grammar patterns, or scenarios.\n\n"
                "2. **Original Academic Scenario (NO Copying Source Text)**:\n"
                "   - ❌ **STRICTLY PROHIBITED**: Copying, adapting, or echoing sentences from the input text or reading passage.\n"
                "   - Design a brand-new, intellectually mature academic or professional scenario (e.g., environmental policy, technology ethics, higher education, scientific research, socioeconomic development).\n"
                "   - `translated_sentence`: Provide a natural, polished, and formal Chinese prompt sentence.\n"
                "   - `correct_english_answer`: Provide the pristine English translation demonstrating natural syntax, academic register, and precise application of the target grammar formula and vocabulary.\n\n"
                "3. **Options (All English) & L1 Interference Taxonomy**:\n"
                "   - ⚠️ **MANDATORY**: ALL four items in 'options' MUST be complete English sentences. NEVER put Chinese sentences into 'options'.\n"
                "   - Return ONLY literal sentence text without choice labels ('A)', '1.') or wrapping quotation marks.\n"
                "   - 1 option is the `correct_english_answer` (matching `options[correct_answer_index]`).\n"
                "   - The other 3 options MUST model authentic Chinese learner errors (L1 negative transfer):\n"
                "     * *Trap 1: Word-for-Word Literal Trap (Chinglish)*: translates Chinese word order mechanically, resulting in verb stacking, missing formal subjects, or unnatural topic-comment structures.\n"
                "     * *Trap 2: Collocation & Preposition Shift*: misuses prepositions or colligations driven by Chinese semantic interference (e.g., *improve the problem*, *pay attention on*, *confront with*).\n"
                "     * *Trap 3: Structural & Formula Distortion*: subtly violates the target grammar pattern (e.g., failed subject-verb inversion, dangling participle, comma splice without coordinator, or tense/aspect flaw).\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: `AUDIT: [Target Vocab + Grammar Formula] -> [Academic Scenario] -> [Trap 1 (Literal Chinglish), Trap 2 (Collocation Shift), Trap 3 (Formula Flaw)] -> [Why Distractors Fail]`\n"
                "   - `hint`: Concise pedagogical hint highlighting the key grammatical structure or functional phrase.\n"
                "   - `explanation`: Contrastively explain why the correct English translation is superior and explicitly identify the specific grammatical or stylistic flaw in each distractor.\n\n"
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
                "   - Natural spoken register: realistic conversational flow with natural discourse markers (e.g., 'Well, look at it this way...', 'That's a valid point, but...', 'You mean...?'), gentle counter-arguments, and mutual clarification.\n"
                "   - Seamlessly embed at least 5 target academic vocabulary items into natural spoken contexts without sounding like textbook recitations.\n\n"
                "2. **Question & Skill Diversity**:\n"
                "   - Generate EXACTLY {count} comprehension questions in the 'questions' array.\n"
                "   - Cover a balanced mix of listening skills across questions:\n"
                "     * `Detail`: Specific fact, limitation, or rationale stated by a speaker.\n"
                "     * `Inference`: Drawing logical conclusions not explicitly phrased in the script.\n"
                "     * `Main Idea`: Overall core purpose or takeaway of the conversation.\n\n"
                "3. **Listening Distractor Taxonomy (NO Cartoonish Choices)**:\n"
                "   - All 4 options must be plausible, concise, grammatically parallel, and closely tied to the discussion.\n"
                "   - ❌ **STRICTLY PROHIBIT**: Childish or absurd choices (e.g., 'machines are too heavy to move', 'destroy all electronics'), trivial common-sense giveaways, and pure polar opposites.\n"
                "   - Engineer distractors using authentic listening test cognitive traps:\n"
                "     * *Speaker Attribution Trap*: Attributes an opinion, concern, or proposal to Speaker 1 when it was actually expressed or qualified by Speaker 2 (or vice versa).\n"
                "     * *Verbatim Catch Trap*: Borrows an eye-catching technical term from the script (e.g., 'nuclear fusion', 'semiconductors'), but links it to a false claim or unmentioned context.\n"
                "     * *Overstated Generalization Trap*: Uses extreme absolutes (*completely impossible*, *abandon entirely*, *useless*) when the speaker only expressed cautious reservation or conditional qualification.\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: `AUDIT: [Skill (Detail/Inference/Main Idea)] -> [Speaker Anchor (e.g. Mark, Turn 4)] -> [Traps: Speaker Confusion / Verbatim Catch / Overstatement] -> [Why Distractors Fail]`\n"
                "   - `explanation`: State the exact dialogue turn supporting the correct answer, and contrastively explain why each distractor trap is invalid.\n"
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
                "   - ❌ **STRICTLY PROHIBIT**: Childish, absurd, or comical answers (e.g., 'Komodo dragons', 'the dam is made of steel', 'it is too small to hold water'), trivial common-sense giveaways, and pure polar opposites.\n"
                "   - Engineer distractors using authentic video assessment cognitive traps:\n"
                "     * *Cross-Timestamp Context Shift*: Borrows a legitimate fact or term from a different part of the video and falsely misapplies it to the target question.\n"
                "     * *Misattributed Claim / Rumor vs. Fact Trap*: Confuses an unsubstantiated rumor/criticism with verified facts, or misidentifies the official clarification.\n"
                "     * *Plausible Over-generalization*: Exaggerates a nuanced or seasonal trend into an absolute or universal claim.\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: `AUDIT: [Timestamp Segment] -> [Core Focus: Mechanism / Controversy / Comparison] -> [Traps: Cross-timestamp shift, Rumor vs fact, Over-generalization] -> [Why Distractors Fail]`\n"
                "   - `explanation`: State what the video explicitly clarifies at the given timestamp, and contrastively explain why each distractor trap is invalid.\n"
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
        }
        return templates.get(name, "Analyze the content.\n\nCONTENT:\n{content}\n")

    # Property redirects for compatibility (renamed to snake_case)
    @property
    def extract_vocabulary(self): return self.get("extract_vocabulary")[0]
    @property
    def extract_grammar(self): return self.get("extract_grammar")[0]
    @property
    def extract_summary(self): return self.get("extract_summary")[0]
