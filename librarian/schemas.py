import json
import dataclasses
import re
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

def _normalize_enum(val: Any, allowed_args: tuple) -> Any:
    """Fuzzy and case-insensitive normalization for Literal enum constraints."""
    if not isinstance(val, str) or not allowed_args:
        return val
    
    # 1. Exact match
    if val in allowed_args:
        return val
    
    clean_val = val.strip().lower().rstrip(".,;:")
    
    # 2. Case-insensitive exact match
    for allowed in allowed_args:
        if str(allowed).lower() == clean_val:
            return allowed
            
    # 3. Cleaned character/punctuation match (e.g. "phrasal_verb" -> "phrasal verb", "set-phrase" -> "set phrase")
    normalized_val = re.sub(r'[\-_]+', ' ', clean_val)
    for allowed in allowed_args:
        norm_allowed = re.sub(r'[\-_]+', ' ', str(allowed).lower())
        if norm_allowed == normalized_val:
            return allowed

    # 4. Longest-first prefix / token / substring matching (so 'phrasal verb' matches before 'verb')
    sorted_args = sorted(allowed_args, key=lambda x: len(str(x)), reverse=True)
    
    # 4.1. Stem/Plural check (e.g. 'phrasal verbs' -> 'phrasal verb', 'collocations' -> 'collocation')
    val_singular = normalized_val.rstrip('s')
    for allowed in sorted_args:
        norm_allowed = re.sub(r'[\-_]+', ' ', str(allowed).lower()).rstrip('s')
        if norm_allowed == val_singular:
            return allowed

    # 4.2. Prefix match
    for allowed in sorted_args:
        norm_allowed = re.sub(r'[\-_]+', ' ', str(allowed).lower())
        if normalized_val.startswith(norm_allowed) or norm_allowed.startswith(normalized_val):
            return allowed

    # 4.3. Substring match
    for allowed in sorted_args:
        norm_allowed = re.sub(r'[\-_]+', ' ', str(allowed).lower())
        if norm_allowed in normalized_val or normalized_val in norm_allowed:
            return allowed

    # 5. Safe fallback to default canonical enum
    return allowed_args[0]

def validate_and_map(cls: Type, data: Dict[str, Any]) -> Any:
    """
    Instantiates a dataclass from a dictionary, with recursive type mapping,
    enum normalization, and array length bounds enforcement.
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
            mapped_val = _map_value(field.type, val)
            # Array length bounds protection
            if isinstance(mapped_val, list):
                if field.metadata and "maxItems" in field.metadata:
                    raw_max = field.metadata["maxItems"]
                    if isinstance(raw_max, int):
                        mapped_val = mapped_val[:raw_max]
                    elif isinstance(raw_max, str) and raw_max.isdigit():
                        mapped_val = mapped_val[:int(raw_max)]
                if field.name == "options" and len(mapped_val) > 4:
                    mapped_val = mapped_val[:4]
            kwargs[field.name] = mapped_val
    
    return cls(**kwargs)

def _map_value(t: Any, val: Any) -> Any:
    origin = get_origin(t)
    args = get_args(t)

    # 1. Enum / Literal Constraint Validation & Normalization
    if origin is Literal:
        return _normalize_enum(val, args)

    # 2. Nested Dataclass Mapping
    if dataclasses.is_dataclass(t) and isinstance(val, dict):
        return validate_and_map(t, val)
    
    # 3. List Mapping
    if (origin is list or origin is List) and isinstance(val, list):
        return [_map_value(args[0], item) for item in val]
    
    # 4. Optional / Union Mapping
    if origin is Union:
        actual_types = [a for a in args if a is not type(None)]
        if actual_types:
            return _map_value(actual_types[0], val)

    return val

# --- Schema Definitions ---

CEFR_LEVELS = Literal["A1", "A2", "B1", "B2", "C1", "C2"]
VOCAB_CEFR_LEVELS = Literal["B1", "B2", "C1", "C2"]
PARTS_OF_SPEECH = Literal["noun", "verb", "adjective", "adverb", "preposition", "conjunction", "interjection"]
EXPRESSION_TYPES = Literal["phrasal verb", "idiom", "collocation", "set phrase"]

@dataclasses.dataclass
class VocabularyItem:
    design_audit: str = dataclasses.field(default="", metadata={"minLength": 1})
    word: str = dataclasses.field(default="", metadata={"minLength": 1})
    part_of_speech: PARTS_OF_SPEECH = dataclasses.field(default="noun")
    definition: str = dataclasses.field(default="", metadata={"minLength": 1})
    word_cefr_level: VOCAB_CEFR_LEVELS = dataclasses.field(default="B2")
    quoted_sentence: str = dataclasses.field(default="", metadata={"minLength": 1})
    example_usage: str = dataclasses.field(default="", metadata={"minLength": 1})

@dataclasses.dataclass
class VocabularyExtraction:
    title: str = dataclasses.field(default="", metadata={"minLength": 1})
    overall_cefr_level: CEFR_LEVELS = dataclasses.field(default="B2")
    vocabulary: List[VocabularyItem] = dataclasses.field(default_factory=list, metadata={"maxItems": "{count}"})

@dataclasses.dataclass
class ExpressionItem:
    design_audit: str = dataclasses.field(default="", metadata={"minLength": 1})
    word: str = dataclasses.field(default="", metadata={"minLength": 1})
    part_of_speech: EXPRESSION_TYPES = dataclasses.field(default="phrasal verb")
    definition: str = dataclasses.field(default="", metadata={"minLength": 1})
    word_cefr_level: CEFR_LEVELS = dataclasses.field(default="B2")
    quoted_sentence: str = dataclasses.field(default="", metadata={"minLength": 1})
    example_usage: str = dataclasses.field(default="", metadata={"minLength": 1})

@dataclasses.dataclass
class ExpressionsExtraction:
    title: str = dataclasses.field(default="", metadata={"minLength": 1})
    overall_cefr_level: CEFR_LEVELS = dataclasses.field(default="B2")
    expressions: List[ExpressionItem] = dataclasses.field(default_factory=list, metadata={"maxItems": "{count}"})

GRAMMAR_CATEGORIES = Literal[
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
]

@dataclasses.dataclass
class GrammarItem:
    design_audit: str = dataclasses.field(default="", metadata={"minLength": 1})
    category: GRAMMAR_CATEGORIES = "Concessive clauses"
    quote: str = dataclasses.field(default="", metadata={"minLength": 1})
    pattern_formula: str = dataclasses.field(default="", metadata={"minLength": 1})
    pedagogical_function: str = dataclasses.field(default="", metadata={"minLength": 1})
    imitation_example: str = dataclasses.field(default="", metadata={"minLength": 1})
    common_mistakes: str = dataclasses.field(default="", metadata={"minLength": 1})
    cefr_level: CEFR_LEVELS = dataclasses.field(default="B2")

@dataclasses.dataclass
class GrammarExtraction:
    title: str = dataclasses.field(default="", metadata={"minLength": 1})
    overall_cefr_level: CEFR_LEVELS = dataclasses.field(default="B2")
    grammar_patterns: List[GrammarItem] = dataclasses.field(default_factory=list, metadata={"maxItems": "{count}"})

@dataclasses.dataclass
class ConceptItem:
    concept_name: str = dataclasses.field(default="", metadata={"minLength": 1})
    educational_significance: str = dataclasses.field(default="", metadata={"minLength": 1})
    key_details: List[str] = dataclasses.field(default_factory=list, metadata={"minItems": 1})
    related_connections: List[str] = dataclasses.field(default_factory=list, metadata={
        "maxItems": 3,
        "item_minLength": 1,
        "item_maxLength": 50
    })

@dataclasses.dataclass
class SummaryExtraction:
    title: str = dataclasses.field(default="", metadata={"minLength": 1})
    overall_cefr_level: CEFR_LEVELS = dataclasses.field(default="B2")
    text_summary_or_plot: str = dataclasses.field(default="", metadata={"minLength": 1})
    estimated_reading_time: str = dataclasses.field(default="", metadata={"minLength": 1})
    essential_questions: List[str] = dataclasses.field(default_factory=list, metadata={"minItems": 2, "maxItems": 3})
    lesson_hook: str = dataclasses.field(default="", metadata={"minLength": 1})
    concepts: List[ConceptItem] = dataclasses.field(default_factory=list, metadata={"maxItems": "{count}"})

@dataclasses.dataclass
class QuizQuestion:
    design_audit: str = dataclasses.field(default="", metadata={"minLength": 1})
    target_word: str = dataclasses.field(default="", metadata={"minLength": 1})
    question: str = dataclasses.field(default="", metadata={"minLength": 1})
    options: List[str] = dataclasses.field(default_factory=list, metadata={"minItems": 4, "maxItems": 4, "item_minLength": 1})
    correct_answer_index: int = dataclasses.field(default=0, metadata={"enum": [0, 1, 2, 3]})
    definition: str = dataclasses.field(default="", metadata={"minLength": 1})
    explanation: str = dataclasses.field(default="", metadata={"minLength": 1})

@dataclasses.dataclass
class VocabularyQuiz:
    title: str = dataclasses.field(default="", metadata={"minLength": 1})
    questions: List[QuizQuestion] = dataclasses.field(default_factory=list, metadata={"maxItems": "{count}"})

@dataclasses.dataclass
class ReadingQuizQuestion:
    design_audit: str = dataclasses.field(default="", metadata={"minLength": 1})
    question: str = dataclasses.field(default="", metadata={"minLength": 1})
    options: List[str] = dataclasses.field(default_factory=list, metadata={"minItems": 4, "maxItems": 4, "item_minLength": 1})
    correct_answer_index: int = dataclasses.field(default=0, metadata={"enum": [0, 1, 2, 3]})
    category: Literal["Main Idea", "Detail/Recall", "Inference", "Author's Tone/Purpose"] = dataclasses.field(default="Main Idea")
    explanation: str = dataclasses.field(default="", metadata={"minLength": 1})

@dataclasses.dataclass
class ReadingVocabItem:
    word: str = dataclasses.field(default="", metadata={"minLength": 1})
    context_sentence: str = dataclasses.field(default="", metadata={"minLength": 1})
    part_of_speech: PARTS_OF_SPEECH = dataclasses.field(default="noun")
    definition: str = dataclasses.field(default="", metadata={"minLength": 1})
    example_usage: str = dataclasses.field(default="", metadata={"minLength": 1})

@dataclasses.dataclass
class ReadingQuiz:
    title: str = dataclasses.field(default="", metadata={"minLength": 1})
    vocabulary: List[ReadingVocabItem] = dataclasses.field(default_factory=list, metadata={"minItems": 5, "maxItems": 8})
    questions: List[ReadingQuizQuestion] = dataclasses.field(default_factory=list, metadata={"maxItems": "{count}"})

@dataclasses.dataclass
class TranslationQuestion:
    design_audit: str = dataclasses.field(default="", metadata={"minLength": 1})
    translated_sentence: str = dataclasses.field(default="", metadata={"minLength": 1})
    correct_english_answer: str = dataclasses.field(default="", metadata={"minLength": 1})
    hint: str = dataclasses.field(default="", metadata={"minLength": 1})
    options: List[str] = dataclasses.field(default_factory=list, metadata={"minItems": 4, "maxItems": 4, "item_minLength": 1})
    correct_answer_index: int = dataclasses.field(default=0, metadata={"enum": [0, 1, 2, 3]})
    explanation: str = dataclasses.field(default="", metadata={"minLength": 1})

@dataclasses.dataclass
class TranslationQuiz:
    title: str = dataclasses.field(default="", metadata={"minLength": 1})
    questions: List[TranslationQuestion] = dataclasses.field(default_factory=list, metadata={"maxItems": "{count}"})

@dataclasses.dataclass
class ListeningQuizTurn:
    speaker: str = dataclasses.field(default="", metadata={"minLength": 1})
    text: str = dataclasses.field(default="", metadata={"minLength": 1})

@dataclasses.dataclass
class ListeningQuizQuestion:
    design_audit: str = dataclasses.field(default="", metadata={"minLength": 1})
    question: str = dataclasses.field(default="", metadata={"minLength": 1})
    options: List[str] = dataclasses.field(default_factory=list, metadata={"minItems": 4, "maxItems": 4, "item_minLength": 1})
    correct_answer_index: int = dataclasses.field(default=0, metadata={"enum": [0, 1, 2, 3]})
    category: Literal["Detail", "Main Idea", "Inference"] = dataclasses.field(default="Detail")
    explanation: str = dataclasses.field(default="", metadata={"minLength": 1})

@dataclasses.dataclass
class ListeningQuiz:
    title: str = dataclasses.field(default="", metadata={"minLength": 1})
    topic: str = dataclasses.field(default="", metadata={"minLength": 1})
    cefr_level: CEFR_LEVELS = dataclasses.field(default="B2")
    speaker_1: str = dataclasses.field(default="", metadata={"minLength": 1})
    speaker_2: str = dataclasses.field(default="", metadata={"minLength": 1})
    script: List[ListeningQuizTurn] = dataclasses.field(default_factory=list)
    questions: List[ListeningQuizQuestion] = dataclasses.field(default_factory=list, metadata={"maxItems": "{count}"})
    speaker_1_gender: str = dataclasses.field(default="female", metadata={"enum": ["{speaker_1_gender}"]})
    speaker_2_gender: str = dataclasses.field(default="male", metadata={"enum": ["{speaker_2_gender}"]})
    speaker_1_role: str = dataclasses.field(default="Creative Director")
    speaker_2_role: str = dataclasses.field(default="Technical Lead")
    speaker_1_accent: str = dataclasses.field(default="Accent: British", metadata={"enum": ["{speaker_1_accent}"]})
    speaker_2_accent: str = dataclasses.field(default="Accent: American", metadata={"enum": ["{speaker_2_accent}"]})

@dataclasses.dataclass
class RoutingResult:
    selected_files: List[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class VideoQuizQuestion:
    design_audit: str = dataclasses.field(default="", metadata={"minLength": 1})
    question: str = dataclasses.field(default="", metadata={"minLength": 1})
    options: List[str] = dataclasses.field(default_factory=list, metadata={"minItems": 4, "maxItems": 4, "item_minLength": 1})
    correct_answer_index: int = dataclasses.field(default=0, metadata={"enum": [0, 1, 2, 3]})
    timestamp: str = dataclasses.field(default="", metadata={"minLength": 1})
    explanation: str = dataclasses.field(default="", metadata={"minLength": 1})

@dataclasses.dataclass
class VideoQuiz:
    title: str = dataclasses.field(default="", metadata={"minLength": 1})
    video_url: str = dataclasses.field(default="", metadata={"minLength": 1})
    video_type: Literal["youtube", "local"] = dataclasses.field(default="local")
    questions: List[VideoQuizQuestion] = dataclasses.field(default_factory=list, metadata={"maxItems": "{count}"})

@dataclasses.dataclass
class MindMapSubBranch:
    sub_branch_name: str = dataclasses.field(default="")
    leaves: List[str] = dataclasses.field(default_factory=list, metadata={"minItems": 0})

@dataclasses.dataclass
class MindMapBranch:
    branch_name: str = dataclasses.field(default="", metadata={"minLength": 1})
    color_theme: Literal["pink", "orange", "blue", "green", "purple"] = dataclasses.field(default="blue")
    sub_branches: List[MindMapSubBranch] = dataclasses.field(default_factory=list, metadata={"minItems": 0})

@dataclasses.dataclass
class MindMapExtraction:
    title: str = dataclasses.field(default="", metadata={"minLength": 1})
    overall_cefr_level: CEFR_LEVELS = dataclasses.field(default="B2")
    root_name: str = dataclasses.field(default="", metadata={"minLength": 1})
    branches: List[MindMapBranch] = dataclasses.field(default_factory=list, metadata={"minItems": 3, "maxItems": 5})

