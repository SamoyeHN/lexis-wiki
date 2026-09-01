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
            schema_dict = get_json_schema(dataclass_cls, include_descriptions=True)
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
                "You are an expert Lexicographer and ESL Curriculum Developer specializing in the Common European Framework of Reference for Languages (CEFR).\n\n"
                "### USER ###\n"
                "Extract academic vocabulary from the text.\n\n"
                "### CORE PEDAGOGICAL MANDATES:\n"
                "1. **Target Count**: Extract up to {count} academic vocabulary words if the text allows. Aim for about {count} suitable words, but never pad the list with duplicates or non-academic words.\n"
                "2. **Lemmatization & Part of Speech (PoS)**: Convert inflected surface forms (e.g., \"running\", \"stabilized\") to their dictionary headword form (e.g., \"run\", \"stabilize\"). The headword MUST preserve the exact Part of Speech used in the text (e.g., if \"process\" is used as a verb, extract it as a verb, not a noun).\n"
                "3. **Absolute Uniqueness**: Every entry in the final list must be completely distinct. No duplicate headwords are allowed under any circumstances.\n"
                "4. **Academic Focus**: Prioritize words that belong to the Academic Word List (AWL) or represent mid-to-high level CEFR vocabulary (B1–C2) crucial for academic literacy.\n"
                "5. **Contextual Accuracy & Original Usage**:\n"
                "   - `quoted_sentence`: Must contain the exact verbatim sentence from the source text where the word appears.\n"
                "   - `example_usage`: Must be an original, high-quality sample sentence demonstrating how to use the word in a typical academic or professional context.\n\n"
                "CONTENT:\n{content}\n"
            ),
            "extract_expressions": (
                "### SYSTEM ###\n"
                "You are an expert Lexicographer, ESL Curriculum Developer, and Idiomatic English Assessment Designer specializing in phraseology, multi-word units, and CEFR language assessment.\n\n"
                "### USER ###\n"
                "Extract genuine multi-word expressions (phrasal verbs, idioms, fixed collocations, and set phrases) from the text.\n\n"
                "### CORE PEDAGOGICAL MANDATES:\n"
                "1. **Target Count**: Identify and extract {count} high-value multi-word expressions from the text.\n"
                "2. **Multi-Word Authenticity (Strict Prohibition on Single Verbs)**:\n"
                "   - Every candidate entry MUST be an inherently multi-word lexical unit (minimum 2 words in the core expression, e.g., 'hinge on', 'factor in', 'pose a risk to', 'for the time being', 'on a regular basis').\n"
                "   - DO NOT extract standalone single verbs (e.g. 'launch', 'tap', 'surge'). Adding generic placeholders like 'launch [something]' does NOT make a single verb an expression.\n"
                "3. **Rigorous Linguistic Classification (`part_of_speech`)**:\n"
                "   - 'phrasal verb': Verb + Particle/Preposition (e.g. 'hinge on', 'factor in', 'clear up', 'fill up'). A single verb taking an object is NEVER a phrasal verb.\n"
                "   - 'collocation' / 'set phrase': Fixed multi-word lexical pairings (e.g. 'pose a risk to [entity]', 'play a role in', 'for the time being', 'on a regular basis').\n"
                "   - 'idiom': Fixed figurative expressions (e.g. 'easier said than done', 'the honeymoon is over').\n"
                "4. **CANONICAL BASE FORM & SLOT-FILLING**:\n"
                "   Convert every extracted expression into a standardized dictionary headword using generic slot placeholders (e.g., 'factor [something] into [something]', 'pose a risk to [entity]').\n"
                "5. **Absolute Uniqueness & Verbatim Sourcing**:\n"
                "   - Every entry in the final list must be completely distinct with no duplicate headwords.\n"
                "   - `quoted_sentence`: Must contain the exact verbatim sentence from the source text where the expression appears.\n"
                "   - `example_usage`: Must be an original, natural sample sentence demonstrating communicative usage.\n\n"
                "CONTENT:\n{content}\n"
            ),
            "extract_grammar": (
                "### SYSTEM ###\n"
                "You are an expert Pedagogical Grammar Analyst and Applied Linguist specializing in advanced academic English syntax.\n\n"
                "### USER ###\n"
                "Extract unique advanced grammar patterns from the text.\n\n"
                "### CORE PEDAGOGICAL MANDATES:\n"
                "1. **Verbatim Evidence**: Every `quote` must be an exact, unedited verbatim excerpt from the source text demonstrating the grammar structure.\n"
                "2. **Category Diversity & Spread**: Select patterns across **different** grammar categories (maximum 1–2 patterns per category). Do not repeat the same category when other advanced structures exist in the text.\n"
                "3. **Rigorous Classification**: Assign a category ONLY if the quote contains a genuine instance of that specific syntactic pattern (e.g., do NOT label relative clauses as \"Inversion\" unless subject-verb inversion is explicitly present).\n"
                "4. **Slot-Filling Pattern Formulas**: Formulate `pattern_formula` using clear bracketed slots (e.g., `Not only + [Auxiliary Verb] + [Subject] + [Main Verb], but also + [Clause]`).\n"
                "5. **Original Imitation Sentence**: `imitation_example` must be a high-quality original academic sentence demonstrating the formula in a new, distinct context.\n"
                "6. **ESL Learner Insight**: `common_mistakes` must explain typical learner errors (e.g., word order, missing auxiliaries, tense mismatch) associated with this specific pattern.\n\n"
                "CONTENT:\n{content}\n"
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
                "4. FOR 'related_connections': Provide ONLY 1-3 word short topic titles (e.g., 'Personal Growth', 'Responsibility'). DO NOT write full sentences, explanations, or 'Connect to:' prefixes.\n\n"
                "CONTENT:\n{content}\n"
            ),
            "vocabulary_quiz": (
                "### SYSTEM ###\n"
                "You are an expert ESL Lexical Assessment Specialist (CEFR/TOEFL standard).\n"
                "### USER ###\n"
                "Create a high-quality multiple-choice vocabulary assessment based on the provided vocabulary list.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n"
                "1. **EXACT QUESTION COUNT**: Generate EXACTLY {count} questions in the 'questions' array.\n"
                "2. **STRICT PART-OF-SPEECH & INFLECTION MATCHING**: \n"
                "   - The blank (____) in every assessment sentence MUST grammatically require the EXACT part of speech and grammatical form of the target word.\n"
                "   - All 4 options must share the EXACT same part of speech, grammatical category, and inflectional form (e.g., all past tense verbs, all plural nouns, or all base adjectives) to eliminate giveaway grammatical clues.\n"
                "3. **CONTEXTUAL & COLLOCATIONAL CONSTRAINT**: \n"
                "   - Write completely NEW, rich academic context sentences at CEFR {cefr_level}. DO NOT copy or re-use any 'Quoted Sentence' or 'Example Usage' from the input vocabulary list.\n"
                "   - The sentence MUST provide explicit contextual, syntactic, or collocational constraints (e.g., dependent prepositions, specific semantic collocations, or contrastive clauses) that make the target word the SINGLE, UNAMBIGUOUSLY correct choice.\n"
                "4. **PLAUSIBLE DISTRACTORS**: Distractors must be plausible near-synonyms or register matches at the same CEFR tier, rendered strictly incorrect by the specific preposition, collocation, or semantic context in the sentence.\n"
                "5. **CONTRASTIVE EXPLANATIONS**: In `explanation`, provide contrastive reasoning: clearly state why `target_word` is the precise fit in this context AND specifically why key distractors are incorrect (e.g., wrong dependent preposition, semantic mismatch, or improper register).\n"
                "6. **ALIGNMENT WITH DESIGN AUDIT**: The assessment sentence in 'question' must be the populated version of the advanced academic sentence planned in 'design_audit'.\n"
                "7. **QUESTION FIELD MANDATE**: Use exactly four underscores (____) for the blank in 'question'. Do NOT wrap the 'question' sentence in outer quotation marks.\n"
                "8. **OPTIONS FIELD MANDATE**: Return ONLY the literal word/phrase for each item in 'options'. DO NOT include option labels (e.g., 'A)', 'a.', '1.'). DO NOT wrap option items or target words in single quotes ('), double quotes (\"), or curly smart quotes (“ ”).\n\n"
                "VOCABULARY:\n{vocabulary_content}\n"
            ),
            "reading_quiz": (
                "### SYSTEM ###\n"
                "You are an expert Reading Comprehension Assessment Designer.\n"
                "### USER ###\n"
                "Create a reading comprehension assessment based on the provided passage.\n\n"
                "**MANDATES**\n"
                "1. **EXACT QUESTION COUNT**: Generate EXACTLY {count} comprehension questions in the 'questions' array. Do NOT stop early.\n"
                "2. **VOCABULARY GLOSSARY MANDATE**: Identify 5 to 8 challenging academic vocabulary words from the passage and populate the 'vocabulary' array with context sentences, parts of speech, definitions, and new example sentences.\n"
                "3. **DESIGN AUDIT MANDATE**: For every question, 'design_audit' MUST follow this exact 3-step planning format: 'DRAFT: [Tested Skill (e.g. Inference)] -> [Text Evidence Location (e.g. Paragraph 3, Line 2)] -> [3 Planned Distractor Traps (Literal Matching / Scope Shift / Logic Error)]'\n"
                "4. **DIAGNOSTIC DISTRACTORS**: Distractors must use intentional cognitive traps (e.g. literal phrase matching in false context, overgeneralization/scope shift, or reversed logic).\n"
                "5. **QUESTION FIELD MANDATE**: Write clear comprehension questions for 'question'. Do NOT wrap the 'question' text in outer quotation marks.\n"
                "6. **OPTIONS FIELD MANDATE**: Return ONLY the literal answer/phrase for each item in 'options'. DO NOT include option labels (e.g., 'A)', 'a.', '1.'). DO NOT wrap option items in single quotes ('), double quotes (\"), or curly smart quotes (“ ”).\n\n"
                "PASSAGE:\n"
                "{passage_content}\n"
            ),
            "translation_quiz": (
                "### SYSTEM ###\n"
                "You are an expert Bilingual Pedagogical Translator (English → {target_language}), specializing in CEFR‑aligned translation assessment design. You strictly follow schemas, avoid hallucinated categories, and produce fully original academic content.\n"
                "### USER ###\n"
                "Create a **translation assessment** that integrates the provided vocabulary list and grammar pattern list.\n\n"
                "**MANDATES**\n"
                "1. **EXACT QUESTION COUNT**: Generate EXACTLY {count} unique translation items in the 'questions' array.\n"
                "2. **One vocabulary item + one grammar pattern per question.**  \n"
                "3. **No repetition** of vocabulary, grammar, or scenario themes.  \n"
                "4. **All English content must match CEFR {cefr_level}.**  \n"
                "5. **All {target_language} sentences must be natural, academic, and region‑appropriate.**  \n"
                "6. **L1 INTERFERENCE DISTRACTORS**: Distractor options must model common L1 interference errors (e.g., literal word-for-word translation errors, misplaced modifiers, incorrect preposition collocations, or verb tense mismatches).\n"
                "7. **QUESTION FIELD MANDATE**: Do NOT wrap the 'question' or translated sentence text in outer quotation marks.\n"
                "8. **OPTIONS FIELD MANDATE**: Each item in 'options' must be a full English sentence. Return ONLY literal text without choice labels (e.g., 'A)', 'a.', '1.'). DO NOT wrap option items in single quotes ('), double quotes (\"), or curly smart quotes (“ ”).\n"
                "9. **All items must follow the JSON schema exactly. No additional fields.**\n\n"
                "VOCABULARY:\n{vocabulary_content}\n\n"
                "GRAMMAR:\n{grammar_content}\n"
            ),
            "listening_quiz": (
                "### SYSTEM ###\n"
                "You are an expert ESL Audio Script Writer and Dialogue Producer.\n"
                "### USER ###\n"
                "Generate a dialogue and comprehension questions based on the following vocabulary.\n\n"
                "**MANDATES**\n"
                "1. **EXACT QUESTION COUNT**: Generate EXACTLY {count} comprehension questions in the 'questions' array.\n"
                "2. **DIAGNOSTIC DISTRACTORS**: Distractors must use intentional cognitive traps (e.g. misheard detail, speaker attribution confusion, or logic shift).\n"
                "3. **QUESTION FIELD MANDATE**: Write clear comprehension questions for 'question'. Do NOT wrap the 'question' text in outer quotation marks.\n"
                "4. **OPTIONS FIELD MANDATE**: Return ONLY the literal answer/phrase for each item in 'options'. DO NOT include option labels (e.g., 'A)', 'a.', '1.'). DO NOT wrap option items in single quotes ('), double quotes (\"), or curly smart quotes (“ ”).\n\n"
                "VOCABULARY:\n{vocabulary_content}\n"
            ),
            "video_quiz": (
                "### SYSTEM ###\n"
                "You are an expert Video-Based ESL Assessment Designer.\n"
                "### USER ###\n"
                "Create a timestamp-aware video comprehension quiz based on the provided video transcript.\n\n"
                "**MANDATES**\n"
                "1. **EXACT QUESTION COUNT**: Generate EXACTLY {count} timestamp-aware video questions in the 'questions' array.\n"
                "2. Every question must map to a specific timestamp present verbatim in the transcript (e.g. [01:25] or [00:15:20]) where the answer is clearly discussed.\n"
                "3. **DIAGNOSTIC DISTRACTORS**: Distractors must use intentional video comprehension traps (e.g. plausible false claims, facts from wrong timestamp segments, or misinterpreted context).\n"
                "4. **QUESTION FIELD MANDATE**: Write timestamp-aware comprehension questions for 'question'. Do NOT wrap the 'question' text in outer quotation marks.\n"
                "5. **OPTIONS FIELD MANDATE**: Return ONLY the literal answer/phrase for each item in 'options'. DO NOT include option labels (e.g., 'A)', 'a.', '1.'). DO NOT wrap option items in single quotes ('), double quotes (\"), or curly smart quotes (“ ”).\n\n"
                "VIDEO TRANSCRIPT:\n{transcript_content}\n"
            ),
            "extract_mindmap": (
                "### SYSTEM ###\n"
                "You are an expert Educational Content Designer and Mind Mapping Specialist.\n"
                "### USER ###\n"
                "Construct a highly detailed hierarchical mind map representing the structural ideas, themes, and supporting details of the text.\n\n"
                "MANDATES:\n"
                "1. Establish a single central root theme summarizing the unit.\n"
                "2. Extract exactly 3 to 5 distinct primary branches representing the major sub-themes or narrative stages. Assign each a unique, harmonious color theme from the allowed values.\n"
                "3. Drill down into nested sub-branches and key details/leaves for each sub-theme. If a sub-theme does not need sub-categorization, place its details directly into the leaves of a sub-branch with an empty name (sub_branch_name: '').\n"
                "4. Maintain a perfect hierarchy where leaf nodes represent concrete examples or specific details verbatim/highly faithful to the text.\n\n"
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
