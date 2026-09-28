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
        "expert_audit_vocabulary": QuizQualityAuditReport,
        "expert_audit_reading": QuizQualityAuditReport,
        "expert_audit_translation": QuizQualityAuditReport,
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

    # Hand-managed .md prompts that have NO entry in the templates dict of
    # get_default_template (a lookup falls back to the generic stub).
    # sync_factory_from_code must SKIP regenerating their .md, otherwise it
    # would clobber the full hand-written audit prompts (incl. the 1-based
    # item_index pin) with that stub. Their .json schema is still synced.
    _HAND_MANAGED_TEMPLATES = {
        "expert_audit_translation",
        "expert_audit_vocabulary",
        "expert_audit_reading",
    }

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
            
            if name in cls._HAND_MANAGED_TEMPLATES:
                # Hand-managed audit prompt (no inline definition): keep the
                # existing .md untouched so we never clobber it with the stub.
                continue
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
                "You are an expert Lexicographer and ESL Curriculum Developer specializing in contextual lexical analysis and the Academic Word List (AWL).\n\n"
                "### USER ###\n"
                "Extract academic vocabulary from the text.\n\n"
                "### CORE PEDAGOGICAL MANDATES:\n"
                "1. **Target Sourcing & Full Coverage**:\n"
                "   - Extract every designated target word from the target list (or passage). Never return an empty list (`[]`).\n"
                "   - Every headword in `word` must be a single headword (hyphenated words like 're-schedule' are preserved as single units). Do not extract space-separated phrases.\n"
                "   - `quoted_sentence`: MUST be the exact, complete authentic sentence from the passage containing the target word. Use its pre-located sentence anchor `[S-ID]` when provided.\n\n"
                "2. **Contextual POS Alignment & Pedagogical Depth**:\n"
                "   - `definition` & `example_usage` MUST strictly match the exact Part of Speech (POS) and syntactic role of the target word as established in the `quoted_sentence`.\n"
                "   - `definition`: Provide a precise, academic, context-specific definition.\n"
                "   - `example_usage`: MUST be an original, brand-new communicative academic sentence demonstrating the word in a fresh context. 🚫 STRICTLY FORBIDDEN: NEVER copy, recycle, or repeat the source text sentence! You must invent an independent example sentence.\n\n"
                "3. **Traceable Design Audit (`design_audit`)**:\n"
                "   - For each entry, execute the canonical audit pipeline:\n"
                "     `AUDIT: [S-ID] -> [Base Lemma Headword] -> [VERBATIM_CONFIRMED]`\n"
                "   - MANDATE: Use the pre-indexed sentence identifier `[S-ID]` (e.g. `[S-14]`) in the first bracket as the anchor to save tokens.\n\n"
                "{syllabus_section}\n\n"
                "### PASSAGE (WITH NUMBERED SENTENCES) ###\n{content}"
            ),
            "extract_expressions": (
                "### SYSTEM ###\n"
                "You are an expert Lexicographer, ESL Curriculum Developer, and Idiomatic English Assessment Designer specializing in contextual phraseology and multi-word collocations.\n\n"
                "### USER ###\n"
                "Extract genuine multi-word expressions from the text.\n\n"
                "### CORE PEDAGOGICAL MANDATES:\n"
                "1. **Target Sourcing & Full Coverage**:\n"
                "   - Extract every designated expression target from the list below. Never return an empty list (`[]`).\n"
                "   - For `word`, copy the exact pre-extracted canonical formula (e.g. `keep in touch with [sb]`) without altering the slot brackets.\n"
                "   - `quoted_sentence`: MUST be the exact, complete authentic sentence from the passage containing the expression. Use its pre-located sentence anchor `[S-ID]`.\n\n"
                "2. **Contextual Meaning & Original Usage**:\n"
                "   - `definition`: Provide a precise, context-specific pedagogical definition.\n"
                "   - `example_usage`: MUST be an original, brand-new communicative academic sentence demonstrating the expression in a fresh context. 🚫 STRICTLY FORBIDDEN: NEVER copy, recycle, or repeat the source text sentence! You must invent an independent example sentence.\n\n"
                "3. **Phraseological Design Audit (`design_audit`)**:\n"
                "   - For each entry, execute the canonical audit pipeline:\n"
                "     `AUDIT: [S-ID] -> [Canonical Slotted Expression] -> [VERBATIM_CONFIRMED]`\n"
                "   - MANDATE: Use the pre-indexed sentence identifier `[S-ID]` (e.g. `[S-12]`) as the anchor in the first bracket.\n\n"
                "{syllabus_section}\n\n"
                "### PASSAGE (WITH NUMBERED SENTENCES) ###\n{content}"
            ),
            "extract_grammar": (
                "### SYSTEM ###\n"
                "You are an expert Pedagogical Grammar Analyst and Applied Linguist specializing in advanced academic English syntax.\n\n"
                "### USER ###\n"
                "### TASK INSTRUCTIONS ###\n"
                "Analyze the provided text and extract genuinely present advanced grammatical constructions (up to {count} patterns).\n\n"
                "### CORE PEDAGOGICAL MANDATES:\n"
                "1. **Target Sourcing & Full Coverage**:\n"
                "   - Extract every designated structural pattern from the list below. Never return an empty list (`[]`).\n"
                "   - For `pattern_formula`, copy the exact pre-extracted canonical formula without modification.\n"
                "   - `quote`: MUST be the exact, complete authentic sentence from the passage containing the pattern. Use its pre-located sentence anchor `[S-ID]`.\n\n"
                "2. **Pedagogical Analysis & Authentic Error Diagnosis**:\n"
                "   - `pedagogical_function`: Explain concisely how this structure enhances academic nuance, formality, cohesion, or rhetorical impact.\n"
                "   - `imitation_example`: Provide a high-quality original academic model sentence illustrating this pattern in a fresh context.\n"
                "   - `common_mistakes`: Diagnose typical ESL learner errors with this pattern. Ensure mistakes are strictly syntactically or semantically flawed (e.g. tense mismatch, missing auxiliary, dangling modifier, illicit word order); NEVER invent an example that is actually standard English.\n\n"
                "3. **Syntactic Design Audit (`design_audit`)**:\n"
                "   - For each entry, execute the canonical audit pipeline:\n"
                "     `AUDIT: [S-ID] -> [Formula] -> [VERBATIM_CONFIRMED]`\n"
                "   - MANDATE: Use the pre-indexed sentence identifier `[S-ID]` (e.g. `[S-8]`) as the anchor in the first bracket.\n\n"
                "{syllabus_section}\n\n"
                "### PASSAGE (WITH NUMBERED SENTENCES) ###\n{content}"
            ),

            "extract_summary": (
                "### SYSTEM ###\n"
                "You are an expert Reading Specialist and Educational Content Developer.\n"
                "### USER ###\n"
                "Perform a comprehensive thematic and structural analysis of the text for educational use.\n\n"
                "MANDATE:\n"
                "1. Provide a cohesive summary of the text's narrative plot, storyline, or main arguments.\n"
                "2. Identify up to {count} distinct core concepts or topics from the text.\n"
                "3. For each concept, extract its specific details and list 1-3 related concept titles.\n"
                "4. FOR 'related_connections': Provide concise, 1-4 word short topic titles (e.g., 'Personal Growth', 'Media Literacy', 'Intentional Living'). DO NOT write full sentences, explanations, or 'Connect to:' prefixes.\n\n"
                "CONTENT:\n{content}\n"
            ),
            "vocabulary_quiz": (
                "### SYSTEM ###\n"
                "You are an expert ESL Lexical Assessment Specialist who designs standardized, fair, and diagnostically rigorous CEFR-aligned vocabulary assessments.\n\n"
                "### USER ###\n"
                "Create a high-quality multiple-choice vocabulary assessment from the supplied vocabulary list.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Count & Coverage**:\n"
                "   - Generate EXACTLY {count} questions testing {count} unique single-word vocabulary items exclusively from the supplied list. No duplicates, derivatives, or fabricated targets.\n"
                "   - ⚠️ **ACTIVE TARGET USAGE MANDATE**: `target_word` MUST match `options[correct_answer_index]` character-for-character. The word placed into the blank `____` must perfectly agree with the declared target's part of speech and required inflectional form.\n\n"
                "2. **Question (Contextual & Structural Anchoring)**:\n"
                "   - Write a brand-new compound/complex academic sentence at CEFR {cefr_level} containing a subordinate or coordinate clause (e.g., concession, condition, cause, or contrast).\n"
                "   - 🔒 **STRICT SINGLE BLANK MANDATE**: Each question stem MUST contain EXACTLY ONE single continuous blank `____` (strictly four underscores, no quotation marks around question). ❌ MULTIPLE BLANKS ARE ABSOLUTELY PROHIBITED: NEVER include two or more blanks in a single sentence (e.g., no '____ ... ____').\n"
                "   - 🚫 **NO COPYING INPUT EXAMPLES**: NEVER copy, adapt, or fill-in-the-blank mask any sentence from the input (neither 'Quoted Sentence' nor 'Example Usage'). Copying examples from the list is strictly prohibited!\n"
                "   - 🎯 **Strict Part-of-Speech Slot Matching**: The blank (____) MUST grammatically require the exact part of speech and syntactic role of the target word. If the target is a noun, the blank must strictly require a noun (e.g., 'The ____ of the...'). Do NOT place a noun into a verb or adjective slot.\n"
                "   - 🚫 **NO STEM TARGET LEAKAGE**: The target word or its morphological derivatives must NEVER appear anywhere in the stem outside the blank `____`.\n"
                "   - ⚓ **MANDATORY CONTEXTUAL & COLLOCATIONAL ANCHORS**:\n"
                "     * Every sentence MUST feature clear, objective context clues (e.g., explicit dependent prepositions like *to / on / of / for*, fixed verb-noun collocations, or unmistakable cause-and-effect / contrastive logic).\n"
                "     * Single-fit validity is absolute: the sentence context must mathematically rule out all 3 distractors on objective structural or logical grounds, NEVER on subjective 'register' or 'formality' differences.\n\n"
                "3. **Options (Target & 3 Structured Objective Distractors)**:\n"
                "   - **Target**: `target_word` must strictly equal `options[correct_answer_index]`. Multi-word units must be tested as indivisible wholes.\n"
                "   - **Grammatical Homogeneity & Authenticity**: All 4 options must be grammatically correct, authentic English words or established expressions sharing the identical grammatical category (part of speech) and the EXACT inflection required by the blank (e.g., all past participles `-ed`, all plurals `-s`, all `-ing`).\n"
                "   - 🚫 **STRICT BANS (ZERO-TOLERANCE DEFECTS)**:\n"
                "     * **NO DUPLICATE OPTIONS**: Every option across A, B, C, D must be 100% unique within each question. Having duplicate options (e.g. A, C, D all 'brochure') is a fatal flaw.\n"
                "     * **NO IN-LIST RECYCLING**: NEVER recycle or pull other unrelated vocabulary items from the supplied input list to fill distractor slots. Do NOT use words like 'brochure', 'enclosure', or 'siege' repeatedly across unrelated questions. Distractors must be authentic, independently generated English words tailored strictly to the sentence context.\n"
                "     * **NO SYNONYM PILES**: NEVER supply interchangeable synonyms. Distractors cannot merely differ by subtle tone or degree of formality.\n"
                "   - 🎯 **MANDATORY 3-VECTOR DISTRACTOR TAXONOMY**:\n"
                "     Each question's 3 distractors MUST consist of:\n"
                "     1. *Trap 1 (Antonym / Logical Polarity Clash)*: directly contradicts the cause/contrast/concession logic established in the sentence clues.\n"
                "     2. *Trap 2 (Collocation / Syntax Clash)*: plausible meaning in the general topic, but violates the blank's dependent preposition, verb valency, or conventional lexical pairing.\n"
                "     3. *Trap 3 (Domain / Semantic Category Mismatch)*: shares the general educational/academic register, but denotes a completely distinct action, entity, or attribute unsuited to this specific functional role.\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: Keep concise (under 20 words) using the tag chain format:\n"
                "     `AUDIT: [Target Word] -> [Syntactic Slot Anchor / Clue] -> [Traps: Antonym / Collocation Clash / Domain Mismatch]`\n"
                "     (e.g. `AUDIT: [comply] -> with [NP] -> Collocation Clash (to/for)`). DO NOT write paragraphs or quote full sentences here.\n"
                "   - `explanation`: State contrastive, objective reasoning explaining why the target fits and explicitly why each distractor is objectively disqualified (grammatical clash, preposition failure, or logical contradiction). You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.\n"
                "   - 🎲 **RANDOMIZED ANSWER KEY BALANCE**: Distribute `correct_answer_index` evenly across 0 (A), 1 (B), 2 (C), and 3 (D) throughout the quiz. Never place all correct answers on the same index.\n"
                "   - `definition`: Concise dictionary meaning of the target in this context.\n\n"
                "CONTENT:\n{vocabulary_content}\n"
            ),
            "reading_quiz": (
                "### SYSTEM ###\n"
                "You are an expert Reading Comprehension Assessment Designer and Applied Linguist specializing in {cefr_descriptor} reading assessments.\n"
                "### USER ###\n"
                "Create a reading comprehension assessment calibrated to the curriculum level ({cefr_level}) based on the provided passage.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Count & Target Vocabulary Extraction**:\n"
                "   - Generate EXACTLY {count} comprehension questions in the 'questions' list.\n"
                "   - Extract {vocab_target_guidance} in the 'vocabulary' list.\n"
                "   - ⚠️ **VERBATIM CONTEXT SOURCING**: For every extracted vocabulary item, `context_sentence` MUST be an exact verbatim sentence physically existing in the passage containing the target word. NEVER invent or alter sentences.\n"
                "   - Provide rigorous part of speech (noun, verb, adjective, adverb, preposition, conjunction, interjection), precise contextual definition, and an illustrative example sentence (`example_usage`) matching the target proficiency level ({cefr_level}).\n\n"
                "2. **Question & Skill Diversity (Calibrated for {cefr_level})**:\n"
                "   - Skill distribution guidelines:\n"
                "{skill_distribution_guidance}\n"
                "   - **Question Stem Crafting**:\n"
                "     {question_stem_guidance}\n"
                "   - **Option Complexity & Length**:\n"
                "     {option_complexity_guidance}\n"
                "   - Questions must require genuine comprehension of the text rather than superficial string-matching. Use clear phrasing without outer quotation marks.\n"
                "   - ⚠️ **VERBATIM TEXT ANCHOR MANDATE**: Every single question MUST be anchored to specific, verifiable statements in the passage. The correct answer must be supported by direct textual evidence, and the design audit must pinpoint the exact paragraph or sentence anchor.\n\n"
                "3. **Diagnostic Distractors (Structured Taxonomy)**:\n"
                "   - All 4 options must be plausible, grammatically parallel, similar in length, and closely tied to the passage topic. No option labels (A, B) or quotes around options.\n"
                "   - ⚓ **ABSOLUTE SINGLE-FIT VALIDITY**: High diagnostic plausibility must NEVER create ambiguity. The question stem combined with the passage MUST provide definitive, objective textual evidence that makes the correct answer the ONLY defensible choice, while decisively eliminating all three distractors on factual, logical, or scope grounds without relying on subjective interpretation.\n"
                "   - ❌ **STRICTLY PROHIBIT**: Absurd/cartoonish extremes (e.g., 'ignore all warnings', 'destroy the planet'), trivial common-sense giveaways, lazy binary opposites, and vocabulary exceeding the curriculum level ({cefr_level}).\n"
                "   - Draw distractors from authentic reading traps:\n"
                "     * *Trap 1 (Literal Match Trap)*: borrows verbatim words or phrasing from the passage, but twists the logical relationship, cause-and-effect, or subject/object.\n"
                "     * *Trap 2 (Scope Shift Trap)*: overly broad, overly restrictive (extreme words like *always*, *only*, *solely*, *never*), or shifts the focus away from the question's premise.\n"
                "     * *Trap 3 (Plausible Distortion / False Inference)*: sounds factually reasonable in real-world knowledge, but is unsupported, unmentioned, or directly contradicted by the text.\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: Keep concise (under 20 words) using the tag chain format:\n"
                "     `AUDIT: [S-ID] -> [Skill: Main Idea/Detail/Inference/Tone] -> [Key Traps: Literal Match/Scope Shift/Distortion]`\n"
                "     (e.g. `AUDIT: [S-14] -> Detail -> Literal Match + Scope Shift`). DO NOT write paragraphs or quote full sentences here.\n"
                "   - `explanation`: State the exact text evidence for the correct answer, and contrastively explain why each distractor fails. You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.\n"
                "   - 🎲 **RANDOMIZED ANSWER KEY BALANCE**: Distribute `correct_answer_index` evenly across 0 (A), 1 (B), 2 (C), and 3 (D) throughout the quiz. Never place all correct answers on the same index.\n\n"
                "PASSAGE:\n"
                "{passage_content}\n"
            ),
            "translation_quiz": (
                "### SYSTEM ###\n"
                "You are an expert Pedagogical Assessment Specialist, Contrastive Linguist, and Master Translator, designing rigorous {target_language}-to-English comparative translation appraisal items for ESL learners calibrated to standardized CEFR {cefr_level} proficiency.\n"
                "### USER ###\n"
                "Create a {target_language}-to-English **Comparative Translation Appraisal** assessment calibrated to **CEFR {cefr_level}** based on the provided VOCABULARY and GRAMMAR list.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Count & Dual-Target Integration (MANDATORY TARGET PRESENCE)**:\n"
                "   - Generate EXACTLY {count} translation appraisal items (Item 1 through Item {count}).\n"
                "   - Each item MUST organically integrate:\n"
                "     * One target vocabulary item from the VOCABULARY list (**Target Keyword**).\n"
                "     * One target grammar pattern from the GRAMMAR list (**Target Grammar Pattern**, faithfully applying its syntactic formula).\n"
                "   - ⚠️ **ACTIVE USAGE MANDATE**: The declared **Target Keyword** MUST be the **central, non-paraphrasable lexical content** of the **{target_language} Sentence** — the source sentence must express the target's *specific* meaning directly, **NEVER via a near-synonym or generic paraphrase**. The Target Keyword MUST also be actively and naturally used inside the **Idiomatic Translation**.\n"
                "   - Ensure diverse coverage without repeating vocabulary items or scenarios across the quiz.\n\n"
                "2. **Comparative Translation Appraisal Architecture (Version A vs Version B)**:\n"
                "   - ❌ **STRICTLY PROHIBITED**: Copying sentences verbatim from the input text or reading passage.\n"
                "   - **Curriculum Difficulty Alignment (CEFR {cefr_level})**:\n"
                "     * **Sentence Complexity**: {sentence_complexity_guidance}\n"
                "     * **Grammar Pattern Level**: {target_grammar_guidance}\n"
                "   - **Target Keyword**: The exact English vocabulary headword or phrase tested directly from the VOCABULARY list.\n"
                "   - **Target Grammar Pattern**: The exact pattern name and formula selected from the GRAMMAR list.\n"
                "   - **{target_language} Sentence**: Provide a natural, polished, and idiomatic {target_language} source sentence calibrated to the curriculum level. Do NOT mix English words into the source sentence unless referring to standard international acronyms.\n"
                "   - **Idiomatic Translation**: A complete, pristine English translation of the entire source sentence that seamlessly integrates both the Target Keyword and the Target Grammar Pattern at CEFR {cefr_level}.\n"
                "   - **Flawed Translation**: A complete English translation of the same source sentence that represents an authentic student or translation error. It MUST contain an objective, diagnostic defect from the flaw taxonomy below, while remaining superficially plausible.\n"
                "   - **Flaw Type**: A concise diagnostic label classifying the exact defect in Flawed Translation.\n"
                "   - **Diagnostic Critique**: A contrastive pedagogical critique (2 to 4 sentences) explicitly explaining why the Idiomatic Translation is superior and identifying the exact structural, collocational, or pragmatic rule violated by the Flawed Translation.\n\n"
                "3. **Authentic Flaw Taxonomy (High Diagnostic Value, NO Absurd Giveaway)**:\n"
                "   - The Flawed Translation MUST NOT be comical, gibberish, or an obvious giveaway. It must mirror typical learner pitfalls at CEFR {cefr_level}:\n"
                "{flaw_taxonomy_guidance}\n\n"
                "4. **Design Audit**:\n"
                "   - `design_audit`: Keep concise (under 20 words) using the tag chain format:\n"
                "     `AUDIT: [Keyword] + [Grammar Formula] -> [Flaw Type]`\n"
                "     (e.g. `AUDIT: [comply] + [Although + Clause] -> Collocation Clash (to vs with)`). DO NOT write paragraphs or quote full sentences here.\n\n"
                "VOCABULARY:\n{vocabulary_content}\n\n"
                "GRAMMAR:\n{grammar_content}\n"
            ),
            "listening_quiz": (
                "### SYSTEM ###\n"
                "You are an expert ESL Audio Script Writer and Listening Assessment Designer specializing in standardized CEFR {cefr_level} language instruction.\n"
                "### USER ###\n"
                "Generate a realistic academic dialogue and a rigorous comprehension assessment calibrated to **CEFR {cefr_level}** based on the provided vocabulary items.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Authentic Dialogue Script**:\n"
                "   - Create a natural, engaging discussion between Speaker 1 and Speaker 2 consisting of 6 to 8 conversational turns.\n"
                "   - **Curriculum Alignment (CEFR {cefr_level})**:\n"
                "     * **Dialogue Register & Complexity**: {dialogue_style_guidance}\n"
                "   - Natural spoken register: realistic conversational flow with authentic discourse markers (e.g., 'Well, look at it this way...', 'That's a valid point, but...', 'Wait, are you saying...?'), gentle counter-arguments, and mutual clarification.\n"
                "   - Seamlessly embed at least 5 target academic vocabulary items into natural spoken contexts without sounding like textbook recitations.\n"
                "   - ⚠️ **SPEAKER ATTRIBUTION RIGOR MANDATE**: Clearly distinguish the roles, stances, and insights of Speaker 1 vs Speaker 2. When a question asks about a specific speaker's viewpoint, concern, or proposal (e.g., 'What does Speaker 1 suggest...?'), the correct answer MUST be based exclusively on that speaker's dialogue turns, NOT the conversational partner's statements.\n\n"
                "2. **Question & Skill Diversity**:\n"
                "   - Generate EXACTLY {count} comprehension questions in the 'questions' array.\n"
                "   - **Question Stem Guidance**: {question_stem_guidance}\n"
                "   - **Option Complexity Guidance**: {option_complexity_guidance}\n"
                "   - Cover a balanced mix of listening skills across questions:\n"
                "{skill_distribution_guidance}\n\n"
                "3. **Listening Distractor Taxonomy (NO Cartoonish Choices)**:\n"
                "   - All 4 options must be plausible, concise, grammatically parallel, and closely tied to the discussion.\n"
                "   - ⚓ **ABSOLUTE SINGLE-FIT VALIDITY**: High diagnostic plausibility must NEVER create ambiguity. The question stem combined with the specific dialogue turn MUST provide definitive conversational evidence (clear speaker attribution, explicit conditions, or established consensus) that makes the correct answer the ONLY defensible choice, while decisively eliminating all three distractors without relying on subjective conjecture.\n"
                "   - ❌ **STRICTLY PROHIBIT**: Childish or absurd choices (e.g., 'machines are too heavy to move', 'destroy all electronics'), trivial common-sense giveaways, and pure polar opposites.\n"
                "   - Engineer distractors using authentic listening test cognitive traps:\n"
                "     * *Trap 1 (Speaker Attribution Swap)*: Attributes an opinion, concern, or proposal to Speaker 1 when it was actually expressed or qualified by Speaker 2 (or vice versa).\n"
                "     * *Trap 2 (Verbatim Catch Trap)*: Borrows an eye-catching technical term from the script, but links it to a false claim or unmentioned context.\n"
                "     * *Trap 3 (Overstated Generalization)*: Uses extreme absolutes (*completely impossible*, *abandon entirely*, *useless*) when the speaker only expressed cautious reservation or conditional qualification.\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: Keep concise (under 20 words) using the tag chain format:\n"
                "     `AUDIT: [Speaker & Turn #] -> [Skill: Detail/Inference/Main Idea] -> [Key Traps: Speaker Swap/Verbatim Catch/Overstatement]`\n"
                "     (e.g. `AUDIT: [Speaker 1, Turn 3] -> Detail -> Speaker Swap`). DO NOT write paragraphs or quote dialogue here.\n"
                "   - `explanation`: State the exact dialogue turn and speaker supporting the correct answer, and contrastively explain why each distractor trap is invalid. You may refer to choices using standard option labels ('Option A', 'Option B', 'Option C', 'Option D') and/or by quoting their specific wording.\n"
                "   - 🎲 **RANDOMIZED ANSWER KEY BALANCE**: Distribute `correct_answer_index` evenly across 0 (A), 1 (B), 2 (C), and 3 (D) throughout the quiz. Never place all correct answers on the same index.\n"
                "   - `correct_answer_index`: MUST be an integer 0, 1, 2, or 3 matching the exact position of the true answer in 'options'. Do NOT use alternative key names.\n"
                "   - Do NOT wrap question or option text in quotes or labels (A, B).\n\n"
                "VOCABULARY:\n{vocabulary_content}\n"
            ),
            "video_quiz": (
                "### SYSTEM ###\n"
                "You are an expert Video-Based ESL Assessment Designer specializing in standardized CEFR {cefr_level} curriculum video instruction.\n"
                "### USER ###\n"
                "Create a timestamp-aware video comprehension quiz calibrated to **CEFR {cefr_level}** based on the provided video transcript.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Exact Question Count & Chronological Timestamp Coverage**:\n"
                "   - Generate EXACTLY {count} timestamp-aware video questions in the 'questions' array.\n"
                "   - Distribute questions evenly across the chronological timeline of the video (e.g. Early context/mechanisms, Middle engineering challenges/environmental impacts, Later controversies/future upgrades).\n"
                "   - Every question must map to a specific timestamp present verbatim in the transcript (e.g. [01:25.10] or [07:46.50]) where the evidence is clearly discussed.\n\n"
                "2. **Video Comprehension & Cognitive Depth**:\n"
                "   - ❌ **BAN TRIVIAL NUMBER GUESSING**: Do NOT write pure numeric recall questions (e.g., guessing between 500 tons vs 3000 tons, or 10 GW vs 22.5 GW).\n"
                "   - **Curriculum Alignment (CEFR {cefr_level})**:\n"
                "     * **Question Depth Guidance**: {question_depth_guidance}\n"
                "     * **Option Complexity Guidance**: {option_complexity_guidance}\n"
                "   - Focus on meaningful conceptual, causal, and analytical understanding appropriate for CEFR {cefr_level}.\n\n"
                "3. **Video Distractor Taxonomy (STRICT BAN on Absurd Options)**:\n"
                "   - All 4 options must be plausible, grammatically parallel, and written in clear English appropriate for CEFR {cefr_level}.\n"
                "   - ⚓ **ABSOLUTE SINGLE-FIT VALIDITY**: High diagnostic plausibility must NEVER create ambiguity. The question stem combined with the specific timestamp context MUST provide definitive transcript evidence that makes the correct answer the ONLY defensible choice, while decisively eliminating all three distractors on factual, chronological, or scope grounds.\n"
                "   - ❌ **STRICTLY PROHIBIT**: Childish, absurd, or comical answers (e.g., 'Komodo dragons', 'the dam is made of steel', 'it is too small to hold water'), trivial common-sense giveaways, and pure polar opposites.\n"
                "   - Engineer distractors using authentic video assessment cognitive traps:\n"
                "     * *Cross-Timestamp Context Shift*: Borrows a legitimate fact or term from a different part of the video and falsely misapplies it to the target question.\n"
                "     * *Misattributed Claim / Rumor vs. Fact Trap*: Confuses an unsubstantiated rumor/criticism with verified facts, or misidentifies the official clarification.\n"
                "     * *Plausible Over-generalization*: Exaggerates a nuanced or seasonal trend into an absolute or universal claim.\n\n"
                "4. **Design Audit & Explanation**:\n"
                "   - `design_audit`: Keep concise (under 20 words) using the tag chain format:\n"
                "     `AUDIT: [Timestamp] -> [Core Focus] -> [Key Traps: Cross-timestamp/Rumor vs fact/Over-generalization]`\n"
                "     (e.g. `AUDIT: [03:15] -> Mechanism -> Cross-timestamp shift`). DO NOT write paragraphs or quote full transcript lines here.\n"
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
                "You are an elite Psychometrician, Senior Applied Linguist, and Lead Assessment Auditor specializing in standardized CEFR language testing.\n\n"
                "### USER ###\n"
                "Conduct an exhaustive, high-reasoning pedagogical and psychometric Quality Audit on the supplied educational quiz items.\n\n"
                "### MANDATES FOR EXPERT AUDIT:\n\n"
                "0. **SYNTACTIC WELL-FORMEDNESS & STRUCTURAL INTEGRITY (ZERO-TOLERANCE GATES)**:\n"
                "   - Before evaluating meaning, verify that the item obeys absolute testing integrity:\n"
                "   - **ZERO-TOLERANCE DEFECTS (Automatic Rejection, single_fit_valid = false, pedagogical_score <= 30)**:\n"
                "     - **Duplicate Options Within Item**: If any question has identical or duplicate options (e.g. options A, C, D are all the same word like 'brochure'), the item is **FATALLY FLAWED**. You MUST set `single_fit_valid: false`, assign `pedagogical_score <= 20`, and explicitly flag 'Duplicate options detected in item'.\n"
                "     - **Missing Predicate Verb**: If the stem lacks a main finite verb and the target option is a noun/adjective (e.g. *\"perseverance [triumph] as a testament\"*), the item is **FATALLY FLAWED**. You MUST set `single_fit_valid: false`, assign `pedagogical_score <= 30`, and flag 'Missing predicate verb / ungrammatical sentence'.\n"
                "     - **In-List Distractor Recycling**: If distractors contain other unexercised vocabulary items from the supplied unit list instead of independent contextual distractors, the item exhibits cross-item cueing/test pollution. You MUST set `single_fit_valid: false`, assign `pedagogical_score <= 30`, and flag 'In-list word recycling detected'.\n"
                "     - **Severe Lexical/Collocation Tautology**: Phrasings that are unnatural or grammatically redundant (e.g. *\"pledge a commitment\"* instead of *\"make a commitment\"* or *\"pledge to do\"*) must be penalized severely.\n"
                "   - If ANY option causes a sentence fragment, duplicate option, or grammatical breakdown, it CANNOT be considered a valid answer key.\n\n"
                "1. **BLIND TEST-SOLVER SIMULATION (`blind_solved_index` & `confidence`)**:\n"
                "   - `item_index` MUST strictly match the Item number displayed in the prompt (e.g., Item #1 -> item_index: 1, Item #2 -> item_index: 2, etc.).\n"
                "   - Independently solve each question stem with its 4 options (Option A = 0, B = 1, C = 2, D = 3).\n"
                "   - Determine the objectively correct answer based on textual evidence (if reading/video passage provided) or strict sentence-level syntax, dependent prepositions, and logical polarity anchors (for standalone items).\n"
                "   - Set `confidence`:\n"
                "     - `Definite`: Textual evidence or stem syntax/collocation are explicit, direct, and leave zero doubt.\n"
                "     - `Hesitant`: The stem requires inference with slight interpretive friction.\n"
                "     - `Ambiguous`: Multiple options could be argued as plausible OR the stem is structurally flawed/ungrammatical.\n\n"
                "2. **ABSOLUTE SINGLE-FIT VALIDITY & SENTENCE-LEVEL ISOLATION (`single_fit_valid`)**:\n"
                "   - Verify that there is EXACTLY ONE uniquely correct, grammatically sound, and contextually defensible answer.\n"
                "   - ⚠️ **ISOLATED STEM TEST**: The key must be uniquely defensible based on the linguistic constraints of the **sentence stem itself**. If a distractor also works naturally in the sentence, the item is an **INVALID DOUBLE-KEY**, even if the original reading text happened to use the declared key.\n"
                "   - If two options can both be justified by the context OR if the declared key creates an ungrammatical sentence, you MUST flag `single_fit_valid: false`.\n"
                "   - 🌟 **RECOGNIZE AUTHENTIC METAPHOR, PERSONIFICATION & RHETORICAL EXTENSION**:\n"
                "     * Do NOT be an overly literal pedant. In advanced English (CEFR B2-C2), institutional agency, metaphorical extension, hyperbole, and personification are standard, highly authentic linguistic devices.\n"
                "     * Examples: Treating an institution/nation/museum as an *\"inheritor\"*, *\"custodian\"*, or *\"guardian\"* of heritage, or describing an economy as *\"thirsty\"* or *\"reaping benefits\"*, is 100% legitimate figurative English.\n"
                "     * NEVER disqualify a correct answer or declare an item 'keyless' merely because the noun is figurative or personified rather than a strictly literal biological entity. If context clues clearly anchor the intended figurative meaning, confirm the key and rate it high quality (85-95).\n\n"
                "3. **COGNITIVE DISTRACTOR TRAP ANALYSIS (`distractors`)**:\n"
                "   - Evaluate all 4 options (including the correct key and the 3 distractors).\n"
                "   - Identify the authentic educational trap type:\n"
                "     - `None (Correct Answer)`: The uniquely valid key.\n"
                "     - `L1 Negative Transfer / False Friend`: Leverages common ESL/Chinglish structural or lexical transfer errors.\n"
                "     - `Scope Shift / Over-generalization`: Distorts text by turning a specific fact into an absolute/extreme universal claim.\n"
                "     - `Speaker / Entity Misattribution`: Attributes a quote, thought, or deed to the wrong character/speaker.\n"
                "     - `Chronological / Causal Inversion`: Reverses sequence of events or confuses cause with effect.\n"
                "     - `Plausible Real-World Distractor`: Conceptually plausible in the real world, but contradicted or unsupported by the text.\n"
                "     - `Flawed / Trivial Giveaway`: Grammatically flawed, absurd, childish/elementary filler, or trivially easy to eliminate without reading the passage.\n"
                "   - Rate `plausibility_rating` (`High`, `Medium`, or `Low (Flawed)`). Any distractor with `Low (Flawed)` must be reported in `diagnostic_feedback`.\n"
                "   - Provide a concise `elimination_rationale` explaining why a test-taker must definitively reject this choice.\n\n"
                "4. **PROHIBITED ELIMINATION RATIONALES (MANDATORY GATE)**:\n"
                "   - The following rationales are LOGICALLY INVALID and MUST NOT be used to eliminate a distractor or justify `single_fit_valid: true`:\n"
                "     ✗ 'Not used in the passage / source material'\n"
                "     ✗ 'The author / source uses [key] instead'\n"
                "     ✗ 'Could be plausible but [key] is the target word'\n"
                "     ✗ 'Fails the strict lexical match to the vocabulary list'\n"
                "   - A valid elimination rationale MUST reference at least one objective linguistic constraint:\n"
                "     ✓ Part-of-speech / inflectional / grammatical constraint\n"
                "     ✓ Fixed collocation pattern or collocational frequency gap (≥3 orders of magnitude)\n"
                "     ✓ Semantic contradiction or logical polarity clash with explicit stem clues\n"
                "     ✓ Register / domain mismatch\n"
                "     ✓ Preposition / complement / syntactic frame mismatch (e.g. requires different dependent preposition)\n"
                "   - If you CANNOT provide an objective linguistic rationale for eliminating a distractor, you MUST set `single_fit_valid: false`.\n\n"
                "5. **KEY VALIDITY REVERSE TEST (MANDATORY)**:\n"
                "   - For EACH question, before confirming the key:\n"
                "     * Read the stem with EACH of the 4 options inserted.\n"
                "     * Ask: *'Does any NON-key option produce a MORE natural, more idiomatic, or more contextually precise sentence?'*\n"
                "     * If YES → the key is SUBOPTIMAL or WRONG. Set `single_fit_valid: false` and explicitly flag in `diagnostic_feedback`: 'Key suboptimal: [option X] is a stronger/more natural collocation in this context.'\n"
                "     * Only confirm the key if it is the UNIQUELY BEST fit, not merely 'acceptable' or 'present in the vocabulary list'.\n\n"
                "6. **SCORE DIFFERENTIATION & VERDICT (`overall_quality_score` & `pass_audit`)**:\n"
                "   - You MUST assign meaningfully differentiated `pedagogical_score` values across questions based on objective elimination rigor:\n"
                "     - 90–100: All 3 distractors eliminated by HARD criteria (syntax, dependent prepositions, strict semantic contradiction).\n"
                "     - 75–89: 1–2 distractors eliminated by hard criteria, 1 eliminated by softer criteria (register, collocational frequency gap).\n"
                "     - 60–74: A distractor is defensible (potential double-key); item is usable but not ideal.\n"
                "     - < 60: Clear double-key, triple-key, wrong/suboptimal key, or ungrammatical stem.\n"
                "   - **PASS REQUIREMENT**: Set `pass_audit: true` ONLY IF `overall_quality_score >= 80` AND every question has `single_fit_valid == true` AND no question has grammatical/syntactic collapse or double-keys. If even ONE question has an invalid single fit or wrong key, `pass_audit` MUST be `false`.\n\n"
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
