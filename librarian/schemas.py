import json
import dataclasses
from typing import List, Dict, Any, Type, get_origin, get_args, Union, Literal

def get_json_schema(cls: Type, include_descriptions: bool = False) -> Dict[str, Any]:
    """
    Converts a dataclass into a JSON Schema dictionary.
    If cls is already a dict, returns it as-is.
    """
    if isinstance(cls, dict):
        return cls
        
    if not dataclasses.is_dataclass(cls):
        raise ValueError(f"{cls} must be a dataclass")

    properties = {}
    required = []

    for field in dataclasses.fields(cls):
        if field.metadata.get("exclude_from_schema"):
            continue
        schema_part = _type_to_schema(field.type, field_name=field.name, metadata=field.metadata, include_descriptions=include_descriptions)
        properties[field.name] = schema_part
        required.append(field.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False
    }

def _type_to_schema(t: Any, field_name: str = None, metadata: Dict[str, Any] = None, include_descriptions: bool = False) -> Dict[str, Any]:
    origin = get_origin(t)
    args = get_args(t)
    metadata = metadata or {}

    schema = {}

    if t is str:
        schema = {"type": "string"}
    elif t is int:
        schema = {"type": "integer"}
    elif t is float:
        schema = {"type": "number"}
    elif t is bool:
        schema = {"type": "boolean"}
    elif origin is Literal:
        # Use strict enum for better model compliance as requested
        schema = {
            "type": "string",
            "enum": [str(a) for a in args]
        }
    elif origin is list or origin is List:
        # Pass item-specific metadata down to the items schema
        item_metadata = {k[5:]: v for k, v in metadata.items() if k.startswith("item_")}
        
        # Build schema with specific order: type, then constraints, then items
        schema = {"type": "array"}
        
        if "minItems" in metadata:
            v = metadata["minItems"]
            if isinstance(v, int): schema["minItems"] = v
            elif isinstance(v, str):
                schema["minItems"] = int(v) if v.isdigit() else v
        if "maxItems" in metadata:
            v = metadata["maxItems"]
            if isinstance(v, int): schema["maxItems"] = v
            elif isinstance(v, str):
                schema["maxItems"] = int(v) if v.isdigit() else v
        
        # Handle count constraints if passed in metadata or specific field names
        if field_name == "options" and "minItems" not in metadata:
            schema["minItems"] = 4
            schema["maxItems"] = 4

        # Add items last so the model sees the constraints first
        schema["items"] = _type_to_schema(args[0], metadata=item_metadata, include_descriptions=include_descriptions)
    elif origin is dict or origin is Dict:
        schema = {"type": "object"}
    elif dataclasses.is_dataclass(t):
        schema = get_json_schema(t, include_descriptions=include_descriptions)
    elif origin is Union:
        # Handle Optional[T] which is Union[T, NoneType]
        actual_types = [a for a in args if a is not type(None)]
        if len(actual_types) == 1:
            schema = _type_to_schema(actual_types[0], include_descriptions=include_descriptions)
        else:
            schema = {"type": "string"} # Fallback
    else:
        schema = {"type": "string"} # Fallback

    # Apply all standard JSON schema keywords from metadata with GBNF sanitization
    standard_keywords = [
        "description", "enum", "minLength", "maxLength", "pattern",
        "minItems", "maxItems", "uniqueItems", "minimum", "maximum", "multipleOf"
    ]
    for key in standard_keywords:
        if key in metadata:
            val = metadata[key]
            if key == "description" and not include_descriptions:
                continue
            # GBNF Sanitization 1: Ignore pattern if it contains non-ASCII (e.g. Chinese characters) to prevent llama.cpp C++ regex compiler crash
            if key == "pattern" and isinstance(val, str):
                if any(ord(c) > 127 for c in val):
                    continue
            # GBNF Sanitization 2: Handle string templates like "{count}" cleanly
            if key in ["minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum"]:
                if isinstance(val, str):
                    if val.isdigit():
                        val = int(val)
                    # If string contains {placeholder}, keep as string template so _interpolate_schema converts it to int later
            schema[key] = val

    return schema

def get_json_example(cls: Type) -> str:
    """
    Generates a sample JSON string with dummy values based on the dataclass structure.
    """
    if not dataclasses.is_dataclass(cls):
        return "{}"
    
    example = _generate_dummy_value(cls)
    return json.dumps(example, indent=2, ensure_ascii=False)

def _generate_dummy_value(t: Any) -> Any:
    origin = get_origin(t)
    args = get_args(t)

    if t is str:
        return "string"
    if t is int:
        return 0
    if t is float:
        return 0.0
    if t is bool:
        return True
    
    if origin is Literal:
        return args[0]
    
    if origin is list or origin is List:
        return [_generate_dummy_value(args[0])]
    
    if dataclasses.is_dataclass(t):
        return {f.name: _generate_dummy_value(f.type) for f in dataclasses.fields(t)}
    
    if origin is Union:
        actual_types = [a for a in args if a is not type(None)]
        if actual_types:
            return _generate_dummy_value(actual_types[0])
            
    return None

def validate_and_map(cls: Type, data: Dict[str, Any]) -> Any:
    """
    Instantiates a dataclass from a dictionary, with recursive type mapping.
    """
    if not dataclasses.is_dataclass(cls):
        return data

    if not isinstance(data, dict):
        return data

    kwargs = {}
    for field in dataclasses.fields(cls):
        val = data.get(field.name)
        if val is None:
            if field.default is not dataclasses.MISSING:
                kwargs[field.name] = field.default
                continue
            elif field.default_factory is not dataclasses.MISSING:
                kwargs[field.name] = field.default_factory()
                continue
            else:
                # Provide type-safe fallbacks for missing required fields to avoid constructor/method crashes
                if field.type is str:
                    kwargs[field.name] = ""
                elif field.type is int:
                    kwargs[field.name] = 0
                elif field.type is float:
                    kwargs[field.name] = 0.0
                elif field.type is bool:
                    kwargs[field.name] = False
                elif field.type is list or get_origin(field.type) in (list, List):
                    kwargs[field.name] = []
                else:
                    kwargs[field.name] = None
        else:
            kwargs[field.name] = _map_value(field.type, val)
    
    return cls(**kwargs)

def _map_value(t: Any, val: Any) -> Any:
    origin = get_origin(t)
    args = get_args(t)

    if dataclasses.is_dataclass(t) and isinstance(val, dict):
        return validate_and_map(t, val)
    
    if (origin is list or origin is List) and isinstance(val, list):
        return [_map_value(args[0], item) for item in val]
    
    return val

# --- Schema Definitions ---

CEFR_LEVELS = Literal["A1", "A2", "B1", "B2", "C1", "C2"]
PARTS_OF_SPEECH = Literal["noun", "verb", "adjective", "adverb", "preposition", "conjunction"]

@dataclasses.dataclass
class VocabularyItem:
    design_audit: str = dataclasses.field(default="", metadata={"description": "Thinking Step. Verify that this candidate word is present in the text, lemmatized to its headword form (preserving the exact Part of Speech as used in the text), and has not been used in previous list items. Format: 'DRAFT: [Surface Form] -> [Headword] -> [Uniqueness Check (New/Repeat)]'.", "minLength": 1})
    word: str = dataclasses.field(default="", metadata={"description": "The dictionary headword entry form (e.g., 'analyze' instead of 'analyzing'). It must match the exact Part of Speech used in the context.", "minLength": 1 })
    part_of_speech: PARTS_OF_SPEECH = dataclasses.field(default="noun", metadata={"description": "The grammatical category of the word as it is used within the quoted sentence."})
    definition: str = dataclasses.field(default="", metadata={"description": "A concise, clear academic definition suitable for ESL learners at this CEFR level.", "minLength": 1})
    word_cefr_level: CEFR_LEVELS = dataclasses.field(default="B2", metadata={"description": "The specific CEFR difficulty level of this individual word."})
    quoted_sentence: str = dataclasses.field(default="", metadata={"description": "The exact verbatim sentence from the source text where this word appears.", "minLength": 1})
    example_usage: str = dataclasses.field(default="", metadata={"description": "An original, high-quality sample sentence demonstrating how to use the word in a typical academic or professional context.", "minLength": 1})

@dataclasses.dataclass
class VocabularyExtraction:
    title: str = dataclasses.field(metadata={"description": "Title for the extraction (e.g. 'Book 3 Unit 4').", "minLength": 1})
    overall_cefr_level: CEFR_LEVELS = dataclasses.field(metadata={"description": "The holistic CEFR difficulty rating for the entire text."})
    vocabulary: List[VocabularyItem] = dataclasses.field(metadata={"description": "List of up to {count} academic vocabulary words found in the text.", "maxItems": "{count}"})

@dataclasses.dataclass
class ExpressionItem:
    design_audit: str = dataclasses.field(default="", metadata={"description": "Thinking Step. Verify that this candidate phrasal verb or idiom is present in the text, identified in its base dictionary form, and has not been used in previous list items. Format: 'DRAFT: [Surface Form] -> [Base Phrase Form] -> [Uniqueness Check (New/Repeat)]'.", "minLength": 1})
    word: str = dataclasses.field(default="", metadata={"description": "The standardized dictionary headword entry form using slot-filling placeholders (e.g. 'keep one's chin up', 'fill [someone] with [emotion]', 'pose a threat to [entity]').", "minLength": 1})
    part_of_speech: Literal["phrasal verb", "idiom", "collocation"] = dataclasses.field(default="phrasal verb", metadata={"description": "The classification of the multi-word expression."})
    definition: str = dataclasses.field(default="", metadata={"description": "A concise, clear definition of the phrasal verb or idiom.", "minLength": 1})
    word_cefr_level: CEFR_LEVELS = dataclasses.field(default="B2", metadata={"description": "The specific CEFR difficulty level of this phrasal verb or idiom."})
    quoted_sentence: str = dataclasses.field(default="", metadata={"description": "The exact sentence where the phrasal verb or idiom appears in the text.", "minLength": 1})
    example_usage: str = dataclasses.field(default="", metadata={"description": "A new, original example sentence demonstrating correct usage of the phrasal verb or idiom.", "minLength": 1})

@dataclasses.dataclass
class ExpressionsExtraction:
    title: str = dataclasses.field(metadata={"description": "Title for the extraction (e.g. 'Book 3 Unit 4 Expressions').", "minLength": 1})
    overall_cefr_level: CEFR_LEVELS = dataclasses.field(metadata={"description": "The holistic CEFR difficulty rating for the entire set of expressions."})
    expressions: List[ExpressionItem] = dataclasses.field(metadata={"description": "List of up to {count} unique phrasal verbs, idioms, or collocations found in the text.", "maxItems": "{count}"})

@dataclasses.dataclass
class GrammarItem:
    design_audit: str = dataclasses.field(default="", metadata={"description": "Thinking Step. Verify that the verbatim quote contains a genuine instance of the selected grammar category. Format: 'DRAFT: [Verbatim Quote] -> [Grammar Category] -> [Structural Verification]'.", "minLength": 1})
    category: Literal[
        "Concessive clauses",
        "Conditional clauses",
        "Participial clauses",
        "Inversion",
        "Cleft sentences",
        "Nominalization",
        "Abstract frames",
        "Rhetorical parallelism",
        "Non-finite structures",
        "Hedging devices",
        "Anaphoric and cataphoric nouns",
        "Evaluative It-frameworks",
    ] = "Concessive clauses"
    quote: str = dataclasses.field(default="", metadata={"description": "Extract short verbatim quote or core segment demonstrating the pattern.", "minLength": 1})
    pattern_formula: str = dataclasses.field(default="", metadata={"description": "Generalize pattern following Slot-Filling Template: 'He + [Past Tense Verb] + [Object] + .'", "minLength": 1})
    pedagogical_function: str = dataclasses.field(default="", metadata={"description": "Rhetorical purpose: why writers use this structure and what effect it creates.", "minLength": 1})
    imitation_example: str = dataclasses.field(default="", metadata={"description": "Create new sentence demonstrating the pattern. not from the original text but similar in style and complexity.", "minLength": 1})
    common_mistakes: str = dataclasses.field(default="", metadata={"description": "common ESL learner mistakes with this pattern.", "minLength": 1})
    cefr_level: CEFR_LEVELS = dataclasses.field(default="B2", metadata={"description": "CEFR difficulty level of this pattern."})

@dataclasses.dataclass
class GrammarExtraction:
    title: str = dataclasses.field(metadata={"description": "Title for the extraction.", "minLength": 1})
    overall_cefr_level: CEFR_LEVELS = dataclasses.field(metadata={"description": "The holistic CEFR difficulty rating for the entire text."})
    grammar_patterns: List[GrammarItem] = dataclasses.field(metadata={"description": "List of {count} unique, diverse advanced grammar patterns found in the text across different categories.", "maxItems": "{count}"})

@dataclasses.dataclass
class ConceptItem:
    concept_name: str = dataclasses.field(default="", metadata={"description": "The name of the main topic, theme, or concept (e.g., 'Urbanization', 'Smart Technology').", "minLength": 1})
    educational_significance: str = dataclasses.field(default="", metadata={"description": "The pedagogical takeaway: why this concept is important to learn and what core lesson it teaches.", "minLength": 1})
    key_details: List[str] = dataclasses.field(default_factory=list, metadata={"description": "Bullet points highlighting specific details, subtopics, or examples of this concept from the text.", "minItems": 1})
    related_connections: List[str] = dataclasses.field(default_factory=list, metadata={
        "description": "List of 1-3 short related topic titles (1-3 words maximum each, e.g. 'Personal Growth', 'Responsibility') suitable for Obsidian [[Wikilinks]]. DO NOT write full sentences or 'Connect to:' prefixes.",
        "maxItems": 3,
        "item_minLength": 1,
        "item_maxLength": 30
    })

@dataclasses.dataclass
class SummaryExtraction:
    title: str = dataclasses.field(default="", metadata={"description": "Title for the summary extraction (e.g., 'Book 4 Unit 1 Themes').", "minLength": 1})
    text_summary_or_plot: str = dataclasses.field(default="", metadata={"description": "A concise summary (exactly 1-2 sentences) of the text's narrative plot, main argument, or storyline for quick teacher reference.", "minLength": 1})
    estimated_reading_time: str = dataclasses.field(default="", metadata={"description": "Estimated word count and reading time for student pacing reference (e.g., '350 words, approx. 4 minutes reading time').", "minLength": 1})
    essential_questions: List[str] = dataclasses.field(default_factory=list, metadata={"description": "A list of 2-3 engaging, open-ended warm-up or discussion questions to start the lesson.", "minItems": 2, "maxItems": 3})
    lesson_hook: str = dataclasses.field(default="", metadata={"description": "A brief creative activity idea, scenario, or question to hook students' interest at the beginning of the lesson.", "minLength": 1})
    concepts: List[ConceptItem] = dataclasses.field(default_factory=list, metadata={"description": "List of up to {count} unique main topics or concepts extracted from the text.", "maxItems": "{count}"})

@dataclasses.dataclass
class QuizQuestion:
    design_audit: str = dataclasses.field(metadata={"description": "Concise 1-line thinking step (max 25 words). Format: 'DRAFT: [target word] -> [Sentence Plan] -> [3 Distractor Traps]'.", "minLength": 1})

    target_word: str = dataclasses.field(metadata={"description": "The specific vocabulary word selected from the list for this question.", "minLength": 1})
    question: str = dataclasses.field(metadata={"description": "The assessment sentence. Use exactly four underscores (____) for the blank. Do NOT include options or labels (A, B, C, D) inside this field.", "minLength": 1})
    options: List[str] = dataclasses.field(metadata={"description": "Exactly 4 options containing the target_word and the 3 planned distractors. Return ONLY the literal word/phrase without labels like 'A)' or surrounding quotes (\" \").", "minItems": 4, "maxItems": 4, "item_minLength": 1})
    correct_answer_index: int = dataclasses.field(metadata={"description": "Index of the target_word inside the options array (0-3).", "enum": [0, 1, 2, 3]})
    definition: str = dataclasses.field(metadata={"description": "A concise academic definition of the target word.", "minLength": 1})
    explanation: str = dataclasses.field(metadata={"description": "Contrastive pedagogical explanation explaining why the target word is the most accurate/natural choice in this specific context AND why key distractors are invalid (e.g., preposition mismatch, register, semantic nuance).", "minLength": 1})

@dataclasses.dataclass
class VocabularyQuiz:
    title: str = dataclasses.field(metadata={"description": "Title for the quiz."})
    questions: List[QuizQuestion] = dataclasses.field(metadata={"description": "List of {count} distinct multiple-choice questions.", "maxItems": "{count}"})

@dataclasses.dataclass
class ReadingQuizQuestion:
    design_audit: str = dataclasses.field(metadata={"description": "Thinking Step. Format: 'DRAFT: [Tested Skill] -> [Text Evidence Location] -> [3 Distractor Traps (Literal Matching / Scope / Logic Shift)]'. Must match CEFR {cefr_level}.", "minLength": 25})
    question: str = dataclasses.field(metadata={"description": "Comprehension question (match {cefr_level}) addressing the planned skill.", "minLength": 1})
    options: List[str] = dataclasses.field(metadata={"description": "Exactly 4 options containing the correct answer and 3 planned distractors. Return ONLY the literal phrase without labels like 'A)' or surrounding quotes (\" \").", "minItems": 4, "maxItems": 4, "item_minLength": 1})
    correct_answer_index: int = dataclasses.field(metadata={"description": "Index of the correct answer (0-3) inside the options array.", "enum": [0, 1, 2, 3]})
    category: Literal["Main Idea", "Detail/Recall", "Inference", "Author's Tone/Purpose"] = dataclasses.field(metadata={"description": "The specific skill tested by this question."})
    explanation: str = dataclasses.field(metadata={"description": "Detailed reasoning supported by specific text evidence, explaining why the correct answer is valid and why distractors are incorrect.", "minLength": 1})

@dataclasses.dataclass
class ReadingVocabItem:
    word: str = dataclasses.field(metadata={"minLength": 1})
    context_sentence: str = dataclasses.field(metadata={"description": "The exact sentence from the passage where this word first appears.", "minLength": 1})
    part_of_speech: str = dataclasses.field(metadata={"minLength": 1})
    definition: str = dataclasses.field(metadata={"minLength": 1})
    example_usage: str = dataclasses.field(metadata={"description": "A new, original example sentence.", "minLength": 1})

@dataclasses.dataclass
class ReadingQuiz:
    title: str = dataclasses.field(metadata={"description": "Title for the quiz."})
    vocabulary: List[ReadingVocabItem] = dataclasses.field(metadata={"description": "List of 5-8 glossary items identified organically from the text to assist the reader.", "minItems": 5, "maxItems": 8})
    questions: List[ReadingQuizQuestion] = dataclasses.field(metadata={"description": "List of {count} generated comprehension questions.", "maxItems": "{count}"})

@dataclasses.dataclass
class TranslationQuestion:
    design_audit: str = dataclasses.field(metadata={"description": "MANDATED THINKING STEP. Format: 'DRAFT: [Vocab + Grammar Pair] -> [Scenario] -> [English Gold Standard (CEFR {cefr_level})] -> [3 L1 Interference Traps (Word Order / False Cognate / Collocation)]'. Must show explicit pairing and scenario planning. Must be 100% original and must NOT reuse or resemble any source examples.", "minLength": 1})
    translated_sentence: str = dataclasses.field(metadata={"description": "A natural, academic sentence in {target_language} (e.g. Simplified Chinese) to be translated into English, integrating the selected vocabulary item and grammar pattern.", "minLength": 1})
    correct_english_answer: str = dataclasses.field(metadata={"description": "The gold-standard correct English translation of the translated_sentence, written at CEFR {cefr_level} level.", "minLength": 1})
    hint: str = dataclasses.field(metadata={"description": "A pedagogical hint referencing meaning, grammar, or context—not the answer directly.", "minLength": 1})
    options: List[str] = dataclasses.field(metadata={"description": "Exactly 4 unique English options (1 correct_english_answer and 3 distinct plausible distractors modeling L1 interference). Each option must be a full sentence. Return literal text without labels or quotation marks.", "minItems": 4, "maxItems": 4, "item_minLength": 1})
    correct_answer_index: int = dataclasses.field(metadata={"description": "Zero-based index (0-3) of the correct_english_answer inside the options array (options[correct_answer_index] must equal correct_english_answer).", "enum": [0, 1, 2, 3]})
    explanation: str = dataclasses.field(metadata={"description": "A clear explanation of why the correct answer matches the {target_language} sentence, referencing vocabulary meaning and grammar pattern usage, and why key distractors are incorrect.", "minLength": 1})


@dataclasses.dataclass
class TranslationQuiz:
    title: str = dataclasses.field(metadata={"description": "Concise assessment title."})
    questions: List[TranslationQuestion] = dataclasses.field(metadata={"description": "Exactly {count} unique translation items. Each item must use a different vocabulary-grammar pair and a distinct scenario.", "maxItems": "{count}"})

@dataclasses.dataclass
class ListeningQuizTurn:
    speaker: str = dataclasses.field(metadata={"minLength": 1})
    text: str = dataclasses.field(metadata={"minLength": 1})

@dataclasses.dataclass
class ListeningQuizQuestion:
    design_audit: str = dataclasses.field(metadata={"description": "Thinking Step. Format: 'DRAFT: [Audio Evidence Turn] -> [3 Distractor Traps (Misheard Detail / Speaker Confusion / Logic Trap)]'. Must match CEFR {cefr_level}.", "minLength": 1})
    question: str = dataclasses.field(metadata={"description": "Comprehension question (match {cefr_level}) based on the script.", "minLength": 1})
    options: List[str] = dataclasses.field(metadata={"description": "Exactly 4 options. Return ONLY the literal word/phrase without labels or surrounding quotes (\" \").", "minItems": 4, "maxItems": 4, "item_minLength": 1})
    correct_answer_index: int = dataclasses.field(metadata={"description": "Index of the correct answer (0-3).", "enum": [0, 1, 2, 3]})
    category: Literal["Detail", "Main Idea", "Inference"] = dataclasses.field(metadata={"description": "The skill tested."})
    explanation: str = dataclasses.field(metadata={"description": "Reasoning based on the script, explaining why the correct answer is supported and why distractors are incorrect.", "minLength": 1})

@dataclasses.dataclass
class ListeningQuiz:
    title: str = dataclasses.field(metadata={"description": "A short, engaging title for the quiz (max 5 words).", "minLength": 1})
    topic: str = dataclasses.field(metadata={"description": "A one-sentence outline or topic summarizing the dialogue.", "minLength": 1})
    cefr_level: CEFR_LEVELS = dataclasses.field(metadata={"description": "Target CEFR level for the dialogue and questions."})
    speaker_1: str = dataclasses.field(metadata={"description": "Name of the first character (MANDATE: must be a {speaker_1_gender} name).", "minLength": 1})
    speaker_2: str = dataclasses.field(metadata={"description": "Name of the second character (MANDATE: must be a {speaker_2_gender} name).", "minLength": 1})
    script: List[ListeningQuizTurn] = dataclasses.field(metadata={"description": "A natural academic dialogue (150-250 words) incorporating 5-8 items from the vocabulary list. MANDATE: Speaker 1 ({speaker_1_gender}) and Speaker 2 ({speaker_2_gender}) must alternate speaking, starting with Speaker 1."})
    questions: List[ListeningQuizQuestion] = dataclasses.field(metadata={"description": "List of {count} multiple-choice questions based on the script.", "maxItems": "{count}"})
    speaker_1_gender: str = dataclasses.field(default="female", metadata={"description": "MANDATE: must be exactly '{speaker_1_gender}'.", "enum": ["{speaker_1_gender}"]})
    speaker_2_gender: str = dataclasses.field(default="male", metadata={"description": "MANDATE: must be exactly '{speaker_2_gender}'.", "enum": ["{speaker_2_gender}"]})
    speaker_1_role: str = dataclasses.field(default="Creative Director", metadata={"description": "Role of speaker 1."})
    speaker_2_role: str = dataclasses.field(default="Technical Lead", metadata={"description": "Role of speaker 2."})
    speaker_1_accent: str = dataclasses.field(default="Accent: British", metadata={"description": "MANDATE: must be exactly '{speaker_1_accent}'.", "enum": ["{speaker_1_accent}"]})
    speaker_2_accent: str = dataclasses.field(default="Accent: American", metadata={"description": "MANDATE: must be exactly '{speaker_2_accent}'.", "enum": ["{speaker_2_accent}"]})

@dataclasses.dataclass
class RoutingResult:
    selected_files: List[str]

@dataclasses.dataclass
class VideoQuizQuestion:
    design_audit: str = dataclasses.field(metadata={"description": "Thinking Step. Format: 'DRAFT: [Timestamp Range] -> [Verbatim Clue] -> [3 Distractor Traps (False Claim / Wrong Segment / Misinterpreted Context)]'. Must match CEFR {cefr_level}.", "minLength": 1})
    question: str = dataclasses.field(metadata={"description": "The comprehension question (match {cefr_level}) based on what is spoken in the video segment.", "minLength": 1})
    options: List[str] = dataclasses.field(metadata={"description": "Exactly 4 options containing the correct answer and 3 planned distractors. Return ONLY the literal phrase without labels like 'A)' or surrounding quotes (\" \").", "minItems": 4, "maxItems": 4, "item_minLength": 1})
    correct_answer_index: int = dataclasses.field(metadata={"description": "Index of the correct answer (0-3).", "enum": [0, 1, 2, 3]})
    timestamp: str = dataclasses.field(metadata={"description": "The starting timestamp of the video segment where the clue/answer is discussed (e.g. '01:25' or '10:45'). Must exist verbatim in the transcript.", "minLength": 1})
    explanation: str = dataclasses.field(metadata={"description": "Detailed explanation of the correct answer referencing the video dialogue and why distractors do not match.", "minLength": 1})

@dataclasses.dataclass
class VideoQuiz:
    title: str = dataclasses.field(metadata={"description": "Title for the video quiz.", "minLength": 1})
    video_url: str = dataclasses.field(metadata={"description": "The original video URL or identifier.", "minLength": 1})
    video_type: Literal["youtube", "local"] = dataclasses.field(metadata={"description": "Type of the video."})
    questions: List[VideoQuizQuestion] = dataclasses.field(metadata={"description": "List of {count} timestamp-aware video comprehension questions.", "maxItems": "{count}"})


@dataclasses.dataclass
class MindMapSubBranch:
    sub_branch_name: str = dataclasses.field(default="", metadata={"description": "The name of the sub-branch. Leave empty ('') if this branch does not have nested sub-branches and the details/leaves should connect directly to the main branch."})
    leaves: List[str] = dataclasses.field(default_factory=list, metadata={"description": "A list of detailed points, key facts, or specific examples for this branch/sub-branch.", "minItems": 1})

@dataclasses.dataclass
class MindMapBranch:
    branch_name: str = dataclasses.field(default="", metadata={"description": "The name of the primary branch representing a major sub-theme or narrative stage.", "minLength": 1})
    color_theme: Literal["pink", "orange", "blue", "green", "purple"] = dataclasses.field(default="blue", metadata={"description": "The color theme for this branch to visually group related ideas."})
    sub_branches: List[MindMapSubBranch] = dataclasses.field(default_factory=list, metadata={"description": "List of sub-branches. If a branch has no sub-categories, provide a single sub-branch with sub_branch_name set to '' and list the items in leaves.", "minItems": 1})

@dataclasses.dataclass
class MindMapExtraction:
    title: str = dataclasses.field(default="", metadata={"description": "Title for the mind map.", "minLength": 1})
    overall_cefr_level: CEFR_LEVELS = dataclasses.field(default="B2", metadata={"description": "The overall CEFR level of the text."})
    root_name: str = dataclasses.field(default="", metadata={"description": "The central root theme or core question of the unit.", "minLength": 1})
    branches: List[MindMapBranch] = dataclasses.field(default_factory=list, metadata={"description": "List of exactly 3 to 5 primary branches forming the first level of the mind map.", "minItems": 3, "maxItems": 5})

