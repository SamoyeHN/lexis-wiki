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
                "You are an expert Lexicographer and ESL Curriculum Developer specializing in CEFR (B1–C2) and the Academic Word List (AWL).\n\n"
                "### USER ###\n"
                "Extract academic vocabulary from the text.\n\n"
                "### CORE PEDAGOGICAL MANDATES:\n"
                "1. **Target Scope & Strict Single-Word Discipline**:\n"
                "   - Extract up to {count} unique vocabulary words. NEVER return an empty list (`[]`).\n"
                "   - Every headword in `word` MUST be strictly a single lexical word (strictly ONE dictionary lemma, e.g., 'triumph', 'rewarding'). ❌ NO MULTI-WORD PHRASES: Phrasal verbs, idioms, and collocations belong exclusively to expressions extraction.\n"
                "   - Avoid text-specific neologisms or ad-hoc hyphenated compounds (e.g. 'non-statement').\n\n"
                "2. **Absolute Verbatim Sourcing (No Hallucination, No Thematic Inferences)**:\n"
                "   - Every target word MUST derive directly from a literal surface word physically present in the text.\n"
                "   - ❌ NO THEMATIC EXTRAPOLATION: NEVER extract generalized themes, inferences, or external synonyms not explicitly written by the author.\n"
                "   - `quoted_sentence`: MUST be the exact, complete authentic sentence from the text containing the word.\n\n"
                "3. **Register Floor & Passage Coverage**:\n"
                "   - Prioritize genuine CEFR B1–C2 academic or formal analytical lexis (AWL register).\n"
                "   - Skip ultra-basic, general-English function/content words that learners already know (e.g., 'big', 'make', 'people', 'good', 'way').\n"
                "   - **Anti-Clustering**: Distribute picks across different paragraphs; aim for at most ~3 headwords per sentence.\n\n"
                "4. **Lexical Form & Part of Speech**:\n"
                "   - Convert inflected surface forms to their base dictionary lemma in `word`.\n"
                "   - `part_of_speech` must strictly reflect its contextual function: noun, verb, adjective, adverb, preposition, conjunction, interjection.\n"
                "   - Provide an accurate, context-specific `definition`.\n"
                "   - `example_usage`: MUST be an original, brand-new communicative academic sentence demonstrating the word in a fresh context. 🚫 STRICTLY FORBIDDEN: NEVER copy, recycle, or repeat the source text sentence! You must invent an independent example sentence.\n\n"
                "5. **Traceable Design Audit (`design_audit`)**:\n"
                "   - For each entry, execute the canonical audit pipeline:\n"
                "     `AUDIT: [S-ID] -> [Base Lemma Headword] -> [Exact Contextual PoS] -> [CEFR Level (B1–C2)] -> [VERBATIM_CONFIRMED]`\n"
                "   - MANDATE: Use the pre-indexed sentence identifier `[S-ID]` (e.g. `[S-14]`) in the first bracket as the anchor to save tokens.\n\n"
                "### PASSAGE (WITH NUMBERED SENTENCES) ###\n{content}\n\n"
                "{syllabus_section}"
            ),
            "extract_expressions": (
                "### SYSTEM ###\n"
                "You are an expert Lexicographer, ESL Curriculum Developer, and Idiomatic English Assessment Designer specializing in phraseology and CEFR multi-word assessment.\n\n"
                "### USER ###\n"
                "Extract genuine multi-word expressions from the text.\n\n"
                "### CORE PEDAGOGICAL MANDATES:\n"
                "1. **Multi-Word Authenticity & Anti-Cheating**:\n"
                "   - Extract up to {count} high-value multi-word expressions. NEVER return an empty list (`[]`).\n"
                "   - Every entry MUST be an inherently multi-word lexical unit (minimum 2 core words).\n"
                "   - ❌ NO SINGLE VERBS WITH FAKE SLOTS: Standalone single verbs are strictly prohibited—adding a generic slot like '[verb] [something]' does NOT make it an expression (single verbs belong exclusively to vocabulary extraction).\n\n"
                "2. **Canonical Slotted Base Form (`word`)**:\n"
                "   - In `word`, provide the base dictionary form with variable arguments abstracted into standard bracketed slots:\n"
                "     * `[sb]` (somebody) / `[sth]` (something) for person/object argument slots.\n"
                "     * `[one's]` for possessive variable modifier slots (e.g. `attain [one's] best`, `make up [one's] mind`).\n"
                "   - ⚠️ DIRECT BINDING: The exact canonical slotted form derived in `design_audit` MUST be assigned directly to `word`.\n\n"
                "3. **Linguistic Classification (`part_of_speech`)**:\n"
                "   - Classify strictly into one of four categories:\n"
                "     * 'phrasal verb' (verb + particle/preposition unit)\n"
                "     * 'collocation' (habitual multi-word lexical pairing, e.g. Verb + Noun + Prep)\n"
                "     * 'set phrase' (fixed structural chunk or connective)\n"
                "     * 'idiom' (figurative unit with non-compositional meaning)\n\n"
                "4. **Absolute Verbatim Sourcing (Zero Hallucination)**:\n"
                "   - The expression must physically occur in the text. ❌ NEVER EXTRACT PROMPT EXAMPLES: Examples in instructions are illustrative; extracting them is a hallucination violation.\n"
                "   - `quoted_sentence`: MUST be the exact, complete authentic sentence from the text containing the expression.\n"
                "   - Provide a concise `definition`.\n"
                "   - `example_usage`: MUST be an original, brand-new communicative academic sentence demonstrating the expression in a fresh context. 🚫 STRICTLY FORBIDDEN: NEVER copy, recycle, or repeat the source text sentence! You must invent an independent example sentence.\n\n"
                "5. **Phraseological Audit (`design_audit`)**:\n"
                "   - Execute the canonical derivation pipeline:\n"
                "     `AUDIT: [S-ID] -> Canonical Slotted Form -> Category -> VERBATIM_CONFIRMED`\n"
                "   - MANDATE: Use the pre-indexed sentence identifier `[S-ID]` (e.g. `[S-12]`) as the anchor in the first bracket.\n\n"
                "### PASSAGE (WITH NUMBERED SENTENCES) ###\n{content}\n\n"
                "{syllabus_section}"
            ),
            "extract_grammar": (
                "### SYSTEM ###\n"
                "You are an expert Pedagogical Grammar Analyst and Applied Linguist specializing in advanced academic English syntax.\n\n"
                "### USER ###\n"
                "### TASK INSTRUCTIONS ###\n"
                "Analyze the provided text and extract ONLY genuinely present advanced grammatical constructions (up to {count} patterns).\n\n"
                "🛡️ **CORE PRINCIPLES**:\n"
                "- **Quality Over Quota**: Extract ONLY authentic structures genuinely present in the text. If the text only contains 2 or 3 genuine structures, return ONLY those 2 or 3. NEVER force-fit, stretch, or fabricate weak sentences to meet a numerical target.\n"
                "- **Sentence Pool Grounding**: For `quote`, provide the exact, complete authentic sentence from the text containing the pattern. The code automatically resolves and verifies complete verbatim sentences.\n"
                "- 🚫 **Universal Ban on Elementary Sentences**: Short conversational sentences, basic SVO clauses (<12 words), simple declarative definitions (`[Subject] + [be] + [Noun]`), or standalone imperative prompts are 100% DISQUALIFIED!\n\n"
                "### THE FOUR MACRO FUNCTIONAL DOMAINS & CANONICAL FORMULAS:\n"
                "Every extracted sentence must be classified strictly into ONE of the following four functional domains. Align with the canonical formulas below, or construct an isomorphic formula following the **Slot Abstraction Mandate**:\n\n"
                "1. **Rhetoric & Emphasis**:\n"
                "   - Pedagogical Function: Constructs rhythmic symmetry, rhetorical balance, or thematic focus shift via inverted or cleaved word order.\n"
                "   - Canonical Formulas:\n"
                "     * Parallelism: `[Subject] + not only + [VP], but also + [VP]`\n"
                "     * Antithesis / Corrective: `[Subject] + [VP], not + [PrepP/NP], but + [PrepP/NP]`\n"
                "     * Correlative: `Either + [Clause], or + [Clause]`\n"
                "     * Inversion: `[Negative/Restrictive Adv] + [aux/be] + [Subject] + [VP]`\n"
                "     * Cleft Focus: `It + [be] + [Focal Element] + that/who + [Clause]` | `What + [Subject] + [VP] + [be] + [Focus]`\n"
                "   - STRICTLY FORBIDDEN: Ordinary coordination without structural balance, or simple copular statements lacking genuine cleft relative linkers.\n\n"
                "2. **Cohesion & Framing**:\n"
                "   - Pedagogical Function: Establishes discourse cohesion across clauses, organizing complex propositions under abstract shell nouns or encapsulating preceding discourse ideas into an explicit interpreted proposition.\n"
                "   - Canonical Formulas:\n"
                "     * Shell Noun Frame: `The + [Shell Noun] + that + [Proposition Clause]`\n"
                "     * Propositional Encapsulation: `[Preceding Discourse], which + [Interpretive Verb] + that + [Proposition Clause]`\n"
                "     * Summary Noun Transition: `This + [Summary Noun] + [VP]`\n"
                "     * Relational Framework: `The extent / degree to which + [Subject] + [VP]`\n"
                "   - STRICTLY FORBIDDEN: Relative clauses that merely attach descriptive or resultative actions to an immediately preceding noun, lacking an interpretive verb governing a complement proposition.\n\n"
                "3. **Information Packaging**:\n"
                "   - Pedagogical Function: Condenses multiple predications into a dense, compact academic clause via non-finite verb phrases, nominalizations, or evaluative extrapositions.\n"
                "   - Canonical Formulas:\n"
                "     * Participial Adjunct: `[V-ing / V-ed Phrase], [Subject] + [VP]` | `[Subject] + [VP], [V-ing Phrase]`\n"
                "     * Evaluative Extraposition: `It + [be] + [Evaluative Adj/NP] + to + [VP]` | `It + [be] + [Evaluative Adj] + that + [Proposition Clause]`\n"
                "     * Dense Prepositional Frame: `Instead of + [V-ing/NP], [Subject] + [VP]` | `Thanks to / Due to + [NP], [Subject] + [VP]`\n"
                "     * Elaborative Clause: `[Subject] + [VP], which + [VP]`\n"
                "   - STRICTLY FORBIDDEN: Simple coordinate independent clauses connected merely by coordinators without hierarchical compression.\n\n"
                "4. **Logic & Stance**:\n"
                "   - Pedagogical Function: Formulates deductive hypotheses, academic concessive refutations, or modulates epistemic stance/hedging to express calibrated scientific certainty.\n"
                "   - Canonical Formulas:\n"
                "     * Condition: `If + [Subject] + [VP], (then) + [Subject] + [modal] + [VP]` | `Unless + [Subject] + [VP], [Subject] + [VP]`\n"
                "     * Concession: `Although / Even though / While + [Clause], [Subject] + [VP]` | `Despite / In spite of + [NP/V-ing], [Subject] + [VP]`\n"
                "     * Epistemic Stance/Hedging: `[Subject] + [hedging verb: appears to / seems to / tends to] + [VP]` | `It + [hedging verb: suggests / indicates] + that + [Proposition Clause]`\n"
                "   - STRICTLY FORBIDDEN: Simple temporal clauses, causal coordinators, or uncalibrated absolute assertions.\n\n"
                "### ⚖️ DISAMBIGUATION & STRUCTURAL CONTRAST (STRUCTURAL EXCLUSION GATES):\n"
                "1. **Relative Clauses (`which`)**:\n"
                "   - `, which + [Interpretive Verb: means/meant, suggests/suggested, indicates/indicated, showed, proved] + that + [Proposition]` ➔ **`Cohesion & Framing`**\n"
                "   - `[Subject] + [VP], which + [Action / Predicate VP]` ➔ **`Information Packaging`**\n"
                "2. **Complement Clauses (`that`)**:\n"
                "   - `The + [Shell Noun] + that + [Proposition Clause]` ➔ **`Cohesion & Framing`**\n"
                "   - Concrete noun modifiers (`The [Concrete Noun] that + [VP]`) are ordinary grammar and DISQUALIFIED.\n"
                "3. **`It`-Constructions**:\n"
                "   - `It + [be] + [Focal Element] + that/who + [Clause]` ➔ **`Rhetoric & Emphasis`** (Cleft focus)\n"
                "   - `It + [be] + [Evaluative Adj/NP] + [to-VP / that-Clause]` ➔ **`Information Packaging`** (Dummy-it extraposition)\n"
                "4. **Concessive vs Temporal Clauses (`while / whereas`)**:\n"
                "   - `[Sentence A] + while / whereas + [Sentence B]` ➔ **`Logic & Stance`** (Contrastive stance)\n"
                "   - Pure temporal simultaneous clauses (`[Action] while [Action]`) are DISQUALIFIED from `Logic & Stance`.\n\n"
                "### PEDAGOGY & SYNTACTIC DESIGN AUDIT:\n"
                "1. `quote`: MUST be the exact, complete authentic sentence from the text containing the pattern.\n"
                "2. `pattern_formula`: Provide the structural formula (e.g. `Although + [Clause], [Subject] + [VP]`).\n"
                "3. `pedagogical_function`: Explain how this structure enhances academic nuance, formality, or rhetoric.\n"
                "4. `design_audit`: Execute the cognitive derivation pipeline: `AUDIT: [S-ID] -> [Category] -> [Syntactic Slot Formula]` (e.g. `AUDIT: [S-16] -> Information Packaging -> [Subject] + [VP], [V-ing Phrase]`).\n"
                "5. `category`: Must strictly be one of the 4 macro domains: `Rhetoric & Emphasis`, `Cohesion & Framing`, `Information Packaging`, `Logic & Stance`.\n"
                "6. `imitation_example`: Provide a high-quality academic model sentence illustrating this pattern in a different domain.\n"
                "7. `common_mistakes`: Diagnose typical ESL learner errors with this pattern.\n"
                "8. `cefr_level`: Calibrate the proficiency level (B1-C2).\n\n"
                "### PASSAGE (WITH NUMBERED SENTENCES) ###\n{content}\n\n"
                "{syllabus_section}"
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
                "You are an expert Reading Comprehension Assessment Designer and Psychometrician specializing in CEFR/TOEFL standardized reading assessments.\n"
                "### USER ###\n"
                "Create an advanced reading comprehension assessment based on the provided passage.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Count & Challenging Vocabulary Extraction**:\n"
                "   - Generate EXACTLY {count} comprehension questions in the 'questions' list.\n"
                "   - Extract 5 to 8 challenging academic vocabulary items directly from the passage in the 'vocabulary' list.\n"
                "   - ⚠️ **VERBATIM CONTEXT SOURCING**: For every extracted vocabulary item, `context_sentence` MUST be an exact verbatim sentence physically existing in the passage containing the target word. NEVER invent or alter sentences.\n"
                "   - Provide rigorous part of speech (noun, verb, adjective, adverb, preposition, conjunction, interjection), precise contextual definition, and an original academic example sentence (`example_usage`).\n\n"
                "2. **Question & Skill Diversity (Strict Quota Allocation)**:\n"
                "   - You MUST cover a balanced mix of skills across the questions. Ensure the 'category' field is strictly one of:\n"
                "     * `Main Idea`: 1 to 2 questions assessing global gist, central thesis, or primary communicative purpose.\n"
                "     * `Detail/Recall`: 2 to 4 questions targeting key factual statements, causal links, or explicit mechanisms.\n"
                "     * `Inference`: 1 to 3 questions assessing logical implications, unstated assumptions, or deductions fully warranted by the text.\n"
                "     * `Author's Tone/Purpose`: 1 question assessing stance, attitude, rhetorical strategy, or underlying perspective.\n"
                "   - Questions must require genuine synthesis and comprehension of the text rather than superficial string-matching. Use clear phrasing without outer quotation marks.\n"
                "   - ⚠️ **VERBATIM TEXT ANCHOR MANDATE**: Every single question MUST be anchored to specific, verifiable statements in the passage. The correct answer must be supported by direct textual evidence, and the design audit must pinpoint the exact paragraph or sentence anchor.\n\n"
                "3. **Diagnostic Distractors (Structured Taxonomy)**:\n"
                "   - All 4 options must be plausible, grammatically parallel, similar in length, and closely tied to the passage topic. No option labels (A, B) or quotes around options.\n"
                "   - ⚓ **ABSOLUTE SINGLE-FIT VALIDITY**: High diagnostic plausibility must NEVER create ambiguity. The question stem combined with the passage MUST provide definitive, objective textual evidence that makes the correct answer the ONLY defensible choice, while decisively eliminating all three distractors on factual, logical, or scope grounds without relying on subjective interpretation.\n"
                "   - ❌ **STRICTLY PROHIBIT**: Absurd/cartoonish extremes (e.g., 'ignore all warnings', 'destroy the planet'), trivial common-sense giveaways, and lazy binary opposites.\n"
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
                "You are an expert Pedagogical Assessment Specialist, Contrastive Linguist, and Master Translator, designing rigorous {target_language}-to-English comparative translation appraisal items for advanced ESL learners (CEFR B2-C1 standards, CET-6 / TEM-8 / IELTS / TOEFL translation level).\n"
                "### USER ###\n"
                "Create an advanced {target_language}-to-English **Comparative Translation Appraisal** assessment based on the provided VOCABULARY and GRAMMAR list.\n\n"
                "**PEDAGOGICAL ASSESSMENT MANDATES**\n\n"
                "1. **Count & Dual-Target Integration (MANDATORY TARGET PRESENCE)**:\n"
                "   - Generate EXACTLY {count} translation appraisal items (Item 1 through Item {count}).\n"
                "   - Each item MUST organically integrate:\n"
                "     * One target vocabulary item from the VOCABULARY list (**Target Keyword**).\n"
                "     * One target grammar pattern from the GRAMMAR list (**Target Grammar Pattern**, faithfully applying its syntactic formula).\n"
                "   - ⚠️ **ACTIVE USAGE MANDATE**: The declared **Target Keyword** MUST be the **central, non-paraphrasable lexical content** of the **{target_language} Sentence** — the source sentence must express the target's *specific* meaning directly, **NEVER via a near-synonym or generic paraphrase**. The Target Keyword MUST also be actively and naturally used inside the **Idiomatic Translation**.\n"
                "   - Ensure diverse coverage without repeating vocabulary items or scenarios across the quiz.\n\n"
                "2. **Comparative Translation Appraisal Architecture (Version A vs Version B)**:\n"
                "   - ❌ **STRICTLY PROHIBITED**: Copying sentences verbatim from the input text or reading passage. Design brand-new, intellectually mature academic or professional scenarios.\n"
                "   - **Target Keyword**: The exact English vocabulary headword or phrase tested directly from the VOCABULARY list.\n"
                "   - **Target Grammar Pattern**: The exact pattern name and formula selected from the GRAMMAR list.\n"
                "   - **{target_language} Sentence**: Provide a natural, polished, formal, and idiomatic {target_language} source sentence. Do NOT mix English words into the source sentence unless referring to standard international acronyms.\n"
                "   - **Idiomatic Translation**: A complete, pristine, publishable academic English translation of the entire source sentence that seamlessly integrates both the Target Keyword and the Target Grammar Pattern.\n"
                "   - **Flawed Translation**: A complete English translation of the same source sentence that represents an authentic student or machine-translation error. It MUST contain an objective, diagnostic defect from the flaw taxonomy below, while remaining superficially plausible.\n"
                "   - **Flaw Type**: A concise diagnostic label classifying the exact defect in Flawed Translation (e.g., 'Chinglish literal word order', 'Collocation clash: wrong dependent preposition', 'Grammar formula breakdown', 'Scope / Polarity distortion').\n"
                "   - **Diagnostic Critique**: A contrastive pedagogical critique (2 to 4 sentences) explicitly explaining why the Idiomatic Translation is superior and identifying the exact structural, collocational, or pragmatic rule violated by the Flawed Translation.\n\n"
                "3. **Authentic Flaw Taxonomy (High Diagnostic Value, NO Absurd Giveaway)**:\n"
                "   - The Flawed Translation MUST NOT be comical, gibberish, or an obvious giveaway. It must mirror typical higher-intermediate learner pitfalls:\n"
                "     * *Trap 1 (L1 Negative Transfer & Chinglish Syntax)*: Mechanically translates {target_language} word order or syntactic habits, resulting in missing dummy subjects (e.g., 'There have many people...'), incorrect adverb placement, or verb stacking.\n"
                "     * *Trap 2 (Collocation & Preposition Clash)*: Misuses prepositions or verb-noun collocations (e.g., *comply to* instead of *comply with*, *make a damage*, or transitive verbs taking unnecessary prepositions).\n"
                "     * *Trap 3 (Morpho-syntactic & Formula Breakdown)*: Distorts the target formula or non-finite verb morphology (e.g., failed subject-auxiliary inversion, bare infinitive after preposition, or dangling participle).\n\n"
                "4. **Design Audit**:\n"
                "   - `design_audit`: Keep concise (under 20 words) using the tag chain format:\n"
                "     `AUDIT: [Keyword] + [Grammar Formula] -> [Flaw Type]`\n"
                "     (e.g. `AUDIT: [comply] + [Although + Clause] -> Collocation Clash (to vs with)`). DO NOT write paragraphs or quote full sentences here.\n\n"
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
                "You are an elite Psychometrician, Senior Applied Linguist, and Lead Assessment Auditor specializing in CEFR/TOEFL standardized language testing.\n\n"
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
