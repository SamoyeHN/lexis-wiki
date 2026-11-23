"""
Data models, enums, dataclasses, and connective regexes for discourse reading assessment.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("librarian.linguistics")



class DiscourseRelationType(str, Enum):
    MACRO_GIST = "macro_gist"
    CAUSAL_CHAIN = "causal_chain"
    RHETORICAL_CONTRAST = "rhetorical_contrast"
    FACTUAL_MECHANISM = "factual_mechanism"
    MODALITY_HEDGING = "modality_hedging"
    PRAGMATIC_INFERENCE = "pragmatic_inference"


class CEFRDifficultyTier(str, Enum):
    FOUNDATIONAL = "foundational"  # A1, A2 (e.g. Book 1)
    INTERMEDIATE = "intermediate"  # B1, B2 (e.g. Books 2-3)
    ADVANCED = "advanced"          # C1, C2 (e.g. Book 4)

    @classmethod
    def from_cefr(cls, cefr: str) -> "CEFRDifficultyTier":
        c = (cefr or "").strip().upper()
        if c in ("A1", "A2", "A2+"):
            return cls.FOUNDATIONAL
        elif c in ("B1", "B1+", "B2", "B2+"):
            return cls.INTERMEDIATE
        else:
            return cls.ADVANCED


class DistractorFallacyOperator(str, Enum):
    TRUE_BUT_IRRELEVANT = "true_but_irrelevant"
    CAUSAL_REVERSAL = "causal_reversal"
    POLARITY_INVERSION = "polarity_inversion"
    MODALITY_ESCALATION = "modality_escalation"
    OPPOSITE_FACT = "opposite_fact"
    EXTREME_WORD = "extreme_word"
    SAME_PARA_MISMATCH = "same_para_mismatch"
    OVER_EXTENSION = "over_extension"
    EXPLICIT_RESTATEMENT = "explicit_restatement"
    REVERSE_ENTAILMENT = "reverse_entailment"


@dataclass
class ReadingDistractorSpec:
    operator: DistractorFallacyOperator
    blueprint_instruction: str
    source_reference: Optional[str] = None
    target_misconception: str = ""
    source_text: str = ""


# Backward-compatible alias
PrescribedDistractor = ReadingDistractorSpec


@dataclass
class ReadingItemBlueprint:
    item_index: int
    skill_type: DiscourseRelationType
    target_paragraph_index: int
    anchor_sentence_id: str
    anchor_sentence_text: str
    question_stem_prompt: str
    paraphrase_target_instruction: str
    distractors: List[ReadingDistractorSpec] = field(default_factory=list)
    tier: CEFRDifficultyTier = CEFRDifficultyTier.ADVANCED
    passage_words: Optional[Set[str]] = None
    section_title: str = ""

    def to_micro_task(self) -> Dict[str, Any]:
        """Formats the blueprint into an actionable micro-task for LLM prompts."""
        if self.tier == CEFRDifficultyTier.FOUNDATIONAL:
            opt_len = "5 to 10 words per option"
            lex_ceiling = "Use simple A1-B1 vocabulary. Do NOT introduce unfamiliar B2+ words (e.g. mandatory, prompt, encapsulate) unless they appear in the passage."
            para_style = "Near-paraphrase: keep the key subject/noun from the anchor, change simple verbs/adjectives, light syntax shift."
        elif self.tier == CEFRDifficultyTier.INTERMEDIATE:
            opt_len = "8 to 14 words per option"
            lex_ceiling = "Standard vocabulary up to B2 level."
            para_style = "Defining paraphrase: explain the concept clearly in standard English."
        else:
            opt_len = "10 to 16 words per option"
            lex_ceiling = "Advanced C1/C2 academic lexis permitted."
            para_style = "Synthesized reformulation: use nominalization and high-level academic abstraction."

        return {
            "item_index": self.item_index,
            "skill_type": self.skill_type.value,
            "tier": self.tier.value,
            "paragraph": self.target_paragraph_index,
            "section_title": self.section_title,
            "anchor_id": self.anchor_sentence_id,
            "anchor_text": self.anchor_sentence_text,
            "question_stem": self.question_stem_prompt,
            "paraphrase_target": self.paraphrase_target_instruction,
            "paraphrase_style": para_style,
            "option_length_guidance": opt_len,
            "lexical_ceiling_instruction": lex_ceiling,
            "prescribed_distractor_operators": [
                {
                    "operator": d.operator.value,
                    "instruction": d.blueprint_instruction,
                    "trap_target": d.target_misconception,
                    "source_reference": d.source_reference or "n/a",
                    "source_text": d.source_text or "n/a",
                }
                for d in self.distractors
            ],
        }


# Backward-compatible alias
DiscourseItemBlueprint = ReadingItemBlueprint


@dataclass
class DiscourseSection:
    section_index: int
    title: str
    body_text: str
    sentence_ids: List[str] = field(default_factory=list)


@dataclass
class DiscourseGraph:
    paragraphs: List[str]
    sentence_pool: List[Tuple[int, str, str]]  # (para_idx, sentence_id, text)
    causal_chains: List[Tuple[int, str, str]]  # (para_idx, sentence_id, text)
    contrast_spans: List[Tuple[int, str, str]]
    modality_spans: List[Tuple[int, str, str]]
    inference_spans: List[Tuple[int, str, str]] = field(default_factory=list)
    introduction_unit: Optional[Tuple[int, str, str]] = None
    conclusion_unit: Optional[Tuple[int, str, str]] = None
    sections: List[DiscourseSection] = field(default_factory=list)
    top_heading_title: str = ""


@dataclass
class ReadingEvaluationResult:
    is_valid: bool
    score: float
    flags: List[str]
    lexical_overlap_ratio: float
    length_std_dev_ratio: float


CAUSAL_CONNECTIVES_REGEX = re.compile(
    r"\b(because|therefore|thus|consequently|as a result|so that|leads to|due to|hence|causes|leads|since)\b",
    re.IGNORECASE,
)

CONTRAST_CONNECTIVES_REGEX = re.compile(
    r"\b(however|although|though|nevertheless|in contrast|whereas|on the other hand|yet|despite|instead|while)\b",
    re.IGNORECASE,
)

MODALITY_HEDGES_REGEX = re.compile(
    r"\b(might|may|could|suggests|likely|potentially|indicates|appears to|seems|arguably|largely|typically)\b",
    re.IGNORECASE,
)

INFERENCE_TRIGGER_REGEX = re.compile(
    r"\b(this year|instead of|no longer|used to|no more|inexpensive|budget|prevent.*from|without|despite|unable to)\b",
    re.IGNORECASE,
)


__all__ = [
    "CAUSAL_CONNECTIVES_REGEX",
    "CEFRDifficultyTier",
    "CONTRAST_CONNECTIVES_REGEX",
    "DiscourseGraph",
    "DiscourseItemBlueprint",
    "DiscourseRelationType",
    "DiscourseSection",
    "DistractorFallacyOperator",
    "INFERENCE_TRIGGER_REGEX",
    "MODALITY_HEDGES_REGEX",
    "PrescribedDistractor",
    "ReadingDistractorSpec",
    "ReadingEvaluationResult",
    "ReadingItemBlueprint",
]
