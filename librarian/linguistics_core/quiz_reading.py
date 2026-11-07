"""
Discourse Reading Assessment Blueprints and Psychometric Code Gates.
====================================================================
Implements discourse-level reading comprehension question generation based on:
1. Questions Under Discussion (QUD) and Rhetorical Structure Theory (RST).
2. Ten canonical psychometric distractor fallacy operators across 3 CEFR tiers:
   - TRUE_BUT_IRRELEVANT: Factually true statement drawn from a distant, irrelevant paragraph.
   - CAUSAL_REVERSAL / POLARITY_INVERSION: Inversion of cause-and-effect or claim polarity.
   - MODALITY_ESCALATION: Overgeneralization of cautious hedges into absolute extremes.
   - EXTREME_WORD: Scope shift trap utilizing absolute quantifiers (only, always, never).
   - OPPOSITE_FACT: Direct contradiction of explicit textual statements.
   - SAME_PARA_MISMATCH: Attributing action to incorrect actor within the same paragraph.
   - EXPLICIT_RESTATEMENT: Surface restatement trap for pragmatic inference items.
   - OVER_EXTENSION: Speculative multi-step overgeneralization beyond textual scope.
   - REVERSE_ENTAILMENT: Logical inversion of conditional or causal necessity.
3. Level 1 Deterministic Code Gate & Invariant Enforcement:
   - Lexical Leakage Gate (<35% 3-gram overlap with anchor sentence).
   - Option Length Parity Gate (relative standard deviation <= 25%).
   - Cardinality & Option Distinctness Gate (exactly 4 distinct choices).
   - Lexical Ceiling Gate (A1/A2 foundational texts reject out-of-passage B2+ words).
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple


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
                }
                for d in self.distractors
            ],
        }


# Backward-compatible alias
DiscourseItemBlueprint = ReadingItemBlueprint


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


class QuizReadingMixin:
    """
    Mixin providing neuro-symbolic reading comprehension blueprints and evaluation gates.
    """

    @classmethod
    def extract_discourse_graph(cls, text: str) -> DiscourseGraph:
        """
        Parses text into paragraphs and deterministic rhetorical spans.
        Uses spaCy sentence segmentation for sentence ID anchoring.
        """
        # Clean markdown frontmatter and syllabus if present
        clean_text = re.sub(r"^---[\s\S]*?---\s*", "", text).strip()
        clean_text = re.sub(r"<syllabus>[\s\S]*?</syllabus>", "", clean_text).strip()

        raw_paras = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
        paragraphs: List[str] = []
        for p in raw_paras:
            # Strip markdown headers if standalone
            cleaned = re.sub(r"^#+\s*.*?\n", "", p).strip()
            if cleaned:
                paragraphs.append(cleaned)

        sentence_pool: List[Tuple[int, str, str]] = []
        causal_chains: List[Tuple[int, str, str]] = []
        contrast_spans: List[Tuple[int, str, str]] = []
        modality_spans: List[Tuple[int, str, str]] = []
        inference_spans: List[Tuple[int, str, str]] = []

        nlp = cls.get_spacy() if hasattr(cls, "get_spacy") else None
        if nlp is None:
            import spacy
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")

        s_counter = 1
        for p_idx, para in enumerate(paragraphs, 1):
            doc = nlp(para)
            for sent in doc.sents:
                s_text = sent.text.strip()
                if len(s_text) < 15:
                    continue
                s_id = f"S-{s_counter}"
                s_counter += 1

                entry = (p_idx, s_id, s_text)
                sentence_pool.append(entry)

                if CAUSAL_CONNECTIVES_REGEX.search(s_text):
                    causal_chains.append(entry)
                if CONTRAST_CONNECTIVES_REGEX.search(s_text):
                    contrast_spans.append(entry)
                if MODALITY_HEDGES_REGEX.search(s_text):
                    modality_spans.append(entry)
                if INFERENCE_TRIGGER_REGEX.search(s_text):
                    inference_spans.append(entry)

        intro_unit = sentence_pool[0] if sentence_pool else None
        conclusion_unit = sentence_pool[-1] if sentence_pool else None

        return DiscourseGraph(
            paragraphs=paragraphs,
            sentence_pool=sentence_pool,
            causal_chains=causal_chains,
            contrast_spans=contrast_spans,
            modality_spans=modality_spans,
            inference_spans=inference_spans,
            introduction_unit=intro_unit,
            conclusion_unit=conclusion_unit,
        )

    @classmethod
    def build_discourse_blueprints(
        cls,
        text: str,
        count: int = 5,
        cefr_level: Optional[str] = None,
    ) -> List[ReadingItemBlueprint]:
        """
        Builds psychometrically audited reading micro-task blueprints with 3-tier difficulty calibration:
        - Foundational (A1, A2): Simple stems, near-paraphrase keys, accessible distractors.
        - Intermediate (B1, B2): Standard stems, defining paraphrase keys, balanced distractors.
        - Advanced (C1, C2): Academic stems, abstract synthesis keys, subtle fallacy operators.
        Supports up to 6 items:
        1. MACRO_GIST
        2. CAUSAL_CHAIN
        3. RHETORICAL_CONTRAST
        4. FACTUAL_MECHANISM
        5. MODALITY_HEDGING
        6. PRAGMATIC_INFERENCE (when count >= 6)
        """
        graph = cls.extract_discourse_graph(text)
        if not graph.sentence_pool:
            raise ValueError("Input text has no valid sentences for discourse analysis.")

        if not cefr_level:
            try:
                from librarian.linguistics import LinguisticEngine
                cefr_level = LinguisticEngine.calculate_text_cefr(text) or "B2"
            except Exception:
                cefr_level = "B2"

        tier = CEFRDifficultyTier.from_cefr(cefr_level)
        is_foundational = (tier == CEFRDifficultyTier.FOUNDATIONAL)
        is_intermediate = (tier == CEFRDifficultyTier.INTERMEDIATE)

        passage_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
        blueprints: List[ReadingItemBlueprint] = []

        # Helper: Extract anchor topic noun
        def get_topic_noun(s_text: str) -> str:
            matches = re.findall(r"\b[A-Z][a-z]+\b|\b[a-z]{4,}\b", s_text)
            stopwords = {"this", "that", "these", "those", "their", "there", "which", "where", "about", "after"}
            candidates = [m for m in matches if m.lower() not in stopwords]
            return candidates[0] if candidates else "the topic"

        # -------------------------------------------------------------
        # ITEM 1: MACRO_GIST (Global Thesis)
        # -------------------------------------------------------------
        gist_candidates = [s for s in graph.sentence_pool if s[0] == 1]
        gist_anchor = gist_candidates[0] if gist_candidates else graph.sentence_pool[0]
        if len(gist_candidates) > 1 and len(gist_candidates[1][2]) > len(gist_anchor[2]):
            gist_anchor = gist_candidates[1]

        distant_candidates = [s for s in graph.sentence_pool if s[0] > 1]
        distant_ref_1 = distant_candidates[len(distant_candidates) // 2][1] if distant_candidates else "S-last"

        if is_foundational:
            gist_stem = "What is the main idea of this passage?"
            gist_para_target = (
                f"State the main point of anchor [{gist_anchor[1]}]: '{gist_anchor[2]}' "
                "in simple, clear words (5 to 10 words). Keep key topic words; do NOT copy 3 consecutive words."
            )
            gist_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.EXTREME_WORD,
                    blueprint_instruction="Craft an overstatement using 'only' or 'always' that makes the idea too extreme.",
                    target_misconception="Falling for an overstatement with extreme words like 'only' or 'always'.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.OPPOSITE_FACT,
                    blueprint_instruction="State the opposite of the passage's main point using simple vocabulary.",
                    target_misconception="Reversing the main message of the passage.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Mention a true small detail from sentence [{distant_ref_1}] instead of the main idea.",
                    source_reference=distant_ref_1,
                    target_misconception="Mistaking an accurate supporting detail for the overarching thesis.",
                ),
            ]
        elif is_intermediate:
            gist_stem = "Which of the following best describes the main idea of this passage?"
            gist_para_target = (
                f"Paraphrase the overarching thesis from [{gist_anchor[1]}]: '{gist_anchor[2]}' "
                "using standard vocabulary (8 to 14 words, zero 3-gram verbatim copying)."
            )
            gist_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.EXTREME_WORD,
                    blueprint_instruction="Overgeneralize the thesis using restrictive or absolute qualifiers (e.g. 'exclusively', 'must always').",
                    target_misconception="Selecting an overgeneralized statement with absolute scope.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.OPPOSITE_FACT,
                    blueprint_instruction="Invert the author's primary conclusion or stance.",
                    target_misconception="Confusing the author's central stance with its direct negation.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Draw an authentic factual claim from sentence [{distant_ref_1}] that is accurate but merely a local detail.",
                    source_reference=distant_ref_1,
                    target_misconception="Mistaking an accurate supporting detail for the overarching thesis.",
                ),
            ]
        else:
            gist_stem = "Which statement best encapsulates the author's primary thesis in this passage?"
            gist_para_target = (
                f"Reformulate the macro-level thesis established in [{gist_anchor[1]}]: '{gist_anchor[2]}' "
                "using academic abstraction (10 to 16 words, zero 3-gram verbatim copying)."
            )
            gist_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Extract a verified factual assertion from sentence [{distant_ref_1}] that functions as a localized supporting argument rather than the macro-thesis.",
                    source_reference=distant_ref_1,
                    target_misconception="Mistaking an accurate supporting detail for the overarching thesis.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.MODALITY_ESCALATION,
                    blueprint_instruction="Convert the author's nuanced macro-argument into an unconditional, absolute claim using categorical assertions (e.g., 'inevitably requires', 'solely determines').",
                    target_misconception="Conflating conditional or hedged claims with categorical certainty.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.POLARITY_INVERSION,
                    blueprint_instruction="Craft an option that asserts the diametric opposite of the author's core thesis while preserving authentic topic terminology.",
                    target_misconception="Reversing the communicative intention of the passage.",
                ),
            ]

        blueprints.append(
            ReadingItemBlueprint(
                item_index=1,
                skill_type=DiscourseRelationType.MACRO_GIST,
                target_paragraph_index=gist_anchor[0],
                anchor_sentence_id=gist_anchor[1],
                anchor_sentence_text=gist_anchor[2],
                question_stem_prompt=gist_stem,
                paraphrase_target_instruction=gist_para_target,
                distractors=gist_distractors,
                tier=tier,
                passage_words=passage_words,
            )
        )

        # -------------------------------------------------------------
        # ITEM 2: CAUSAL_CHAIN (Cause & Effect)
        # -------------------------------------------------------------
        if graph.causal_chains:
            causal_anchor = graph.causal_chains[0]
        else:
            causal_candidates = [s for s in graph.sentence_pool if s[0] > 1]
            causal_anchor = causal_candidates[0] if causal_candidates else graph.sentence_pool[-1]

        distant_ref_2 = distant_candidates[-1][1] if distant_candidates else "S-1"

        if is_foundational:
            causal_stem = f"According to Paragraph {causal_anchor[0]}, what was the main reason or result mentioned?"
            causal_para_target = (
                f"Simply explain the reason or result stated in anchor [{causal_anchor[1]}]: '{causal_anchor[2]}' "
                "(5 to 10 words, zero 3-gram verbatim copying)."
            )
            causal_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.OPPOSITE_FACT,
                    blueprint_instruction="Reverse the cause and effect using simple words (say it caused the opposite).",
                    target_misconception="Confusing a positive cause or outcome with its direct opposite.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.EXTREME_WORD,
                    blueprint_instruction="Use 'never' or 'completely' to exaggerate the outcome.",
                    target_misconception="Overgeneralizing a specific reason into an absolute rule.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Include a true fact from sentence [{distant_ref_2}] that does not explain this cause.",
                    source_reference=distant_ref_2,
                    target_misconception="Selecting a true statement that does not address the question asked in the stem.",
                ),
            ]
        elif is_intermediate:
            causal_stem = f"Based on Paragraph {causal_anchor[0]}, what is the primary consequence or cause discussed?"
            causal_para_target = (
                f"Accurately paraphrase the cause-and-effect relationship in [{causal_anchor[1]}]: '{causal_anchor[2]}' "
                "(8 to 14 words, zero 3-gram verbatim copying)."
            )
            causal_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.CAUSAL_REVERSAL,
                    blueprint_instruction="Invert the causal direction: make the result appear as the cause or precondition.",
                    target_misconception="Reversing the causal relationship (treating the outcome as the root cause).",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.EXTREME_WORD,
                    blueprint_instruction="Claim that this cause is the 'only' factor responsible for the outcome.",
                    target_misconception="Attributing an effect exclusively to a single factor when others exist.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Introduce a true statement from sentence [{distant_ref_2}] that is irrelevant to the causal chain.",
                    source_reference=distant_ref_2,
                    target_misconception="Selecting a true statement that does not address the question asked in the stem.",
                ),
            ]
        else:
            causal_stem = f"According to Paragraph {causal_anchor[0]}, what causal mechanism is directly responsible for the observed development?"
            causal_para_target = (
                f"Reformulate the precise cause-and-effect chain in [{causal_anchor[1]}]: '{causal_anchor[2]}' "
                "(10 to 16 words, zero 3-gram verbatim copying)."
            )
            causal_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.CAUSAL_REVERSAL,
                    blueprint_instruction="Reverse the directionality of the causal link: assert that the outcome/effect serves as the generative antecedent.",
                    target_misconception="Reversing the causal relationship (treating the outcome as the root cause).",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.MODALITY_ESCALATION,
                    blueprint_instruction="Assert that this causal mechanism inevitably triggers an absolute and irreversible collapse.",
                    target_misconception="Sensationalizing an empirical correlation into an unavoidable certainty.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Incorporate a verified factual statement from sentence [{distant_ref_2}] that answers a different question.",
                    source_reference=distant_ref_2,
                    target_misconception="Selecting a true statement that does not address the question asked in the stem.",
                ),
            ]

        blueprints.append(
            ReadingItemBlueprint(
                item_index=2,
                skill_type=DiscourseRelationType.CAUSAL_CHAIN,
                target_paragraph_index=causal_anchor[0],
                anchor_sentence_id=causal_anchor[1],
                anchor_sentence_text=causal_anchor[2],
                question_stem_prompt=causal_stem,
                paraphrase_target_instruction=causal_para_target,
                distractors=causal_distractors,
                tier=tier,
                passage_words=passage_words,
            )
        )

        # -------------------------------------------------------------
        # ITEM 3: RHETORICAL_CONTRAST (Contrast / Concession)
        # -------------------------------------------------------------
        if graph.contrast_spans:
            contrast_anchor = graph.contrast_spans[0]
        else:
            middle_paras = [s for s in graph.sentence_pool if s[0] > 1 and s[1] != causal_anchor[1]]
            contrast_anchor = middle_paras[len(middle_paras) // 2] if middle_paras else graph.sentence_pool[0]

        distant_ref_3 = graph.sentence_pool[0][1]

        if is_foundational:
            contrast_stem = f"In Paragraph {contrast_anchor[0]}, what difference or contrast is pointed out?"
            contrast_para_target = (
                f"Express the difference or comparison in anchor [{contrast_anchor[1]}]: '{contrast_anchor[2]}' "
                "in simple words (5 to 10 words, zero 3-gram verbatim copying)."
            )
            contrast_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.OPPOSITE_FACT,
                    blueprint_instruction="Reverse the contrast: say the two things are identical or emphasize the wrong side.",
                    target_misconception="Ignoring the contrastive qualification and reversing authorial emphasis.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.EXTREME_WORD,
                    blueprint_instruction="Use 'every single' or 'all' to turn the comparison into an absolute claim.",
                    target_misconception="Confusing a mild contrast with an absolute extreme.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Borrow familiar wording from sentence [{distant_ref_3}] from Paragraph 1.",
                    source_reference=distant_ref_3,
                    target_misconception="Recognizing familiar vocabulary from another part of the text.",
                ),
            ]
        elif is_intermediate:
            contrast_stem = f"In Paragraph {contrast_anchor[0]}, what contrast does the author draw?"
            contrast_para_target = (
                f"Paraphrase the contrastive or concessive relationship established in [{contrast_anchor[1]}]: '{contrast_anchor[2]}' "
                "(8 to 14 words, zero 3-gram verbatim copying)."
            )
            contrast_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.POLARITY_INVERSION,
                    blueprint_instruction="Misidentify which element the author favors or emphasizes across the contrastive marker.",
                    target_misconception="Reversing the communicative weight across a concessive or contrastive junction.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.EXTREME_WORD,
                    blueprint_instruction="Exaggerate the distinction into an irreconcilable or complete opposition.",
                    target_misconception="Amplifying a nuanced comparative difference into total incompatibility.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Introduce an authentic fact from sentence [{distant_ref_3}] that plays no part in this comparison.",
                    source_reference=distant_ref_3,
                    target_misconception="Recognizing familiar vocabulary from another part of the text.",
                ),
            ]
        else:
            contrast_stem = f"In Paragraph {contrast_anchor[0]}, how does the author qualify or contrast the preceding assertion?"
            contrast_para_target = (
                f"Synthesize the concession/contrast in [{contrast_anchor[1]}]: '{contrast_anchor[2]}' "
                "(10 to 16 words, zero 3-gram verbatim copying)."
            )
            contrast_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.POLARITY_INVERSION,
                    blueprint_instruction="Assert that the author dismisses or invalidates the contrasted premise rather than qualifying it.",
                    target_misconception="Misinterpreting a concessive hedge as an outright refutation.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.MODALITY_ESCALATION,
                    blueprint_instruction="Escalate the contrast into an absolute incompatibility (e.g. 'fundamentally precludes', 'strictly incompatible').",
                    target_misconception="Amplifying a nuanced comparative difference into total incompatibility.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Incorporate a verified factual claim from sentence [{distant_ref_3}] outside this rhetorical boundary.",
                    source_reference=distant_ref_3,
                    target_misconception="Recognizing familiar vocabulary from another part of the text.",
                ),
            ]

        blueprints.append(
            ReadingItemBlueprint(
                item_index=3,
                skill_type=DiscourseRelationType.RHETORICAL_CONTRAST,
                target_paragraph_index=contrast_anchor[0],
                anchor_sentence_id=contrast_anchor[1],
                anchor_sentence_text=contrast_anchor[2],
                question_stem_prompt=contrast_stem,
                paraphrase_target_instruction=contrast_para_target,
                distractors=contrast_distractors,
                tier=tier,
                passage_words=passage_words,
            )
        )

        # -------------------------------------------------------------
        # ITEM 4: FACTUAL_MECHANISM (Paraphrased Detail)
        # -------------------------------------------------------------
        used_anchors = {gist_anchor[1], causal_anchor[1], contrast_anchor[1]}
        factual_pool = [s for s in graph.sentence_pool if s[1] not in used_anchors and len(s[2]) >= 25]
        factual_anchor = factual_pool[len(factual_pool) // 3] if factual_pool else graph.sentence_pool[1]

        distant_ref_4 = graph.sentence_pool[-2][1] if len(graph.sentence_pool) >= 2 else graph.sentence_pool[0][1]

        if is_foundational:
            factual_stem = f"According to Paragraph {factual_anchor[0]}, what fact or action is mentioned?"
            factual_para_target = (
                f"State the fact or action in anchor [{factual_anchor[1]}]: '{factual_anchor[2]}' "
                "using simple words (5 to 10 words, zero 3-gram verbatim copying)."
            )
            factual_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.OPPOSITE_FACT,
                    blueprint_instruction="State the opposite of the fact (e.g. say they refused or did not do it).",
                    target_misconception="Contradicting explicit information mentioned in the paragraph.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.EXTREME_WORD,
                    blueprint_instruction="Add an extreme word like 'only' or 'strictly' to restrict the fact.",
                    target_misconception="Exaggerating manageable facts into an urgent or absolute extreme.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"State a true detail from sentence [{distant_ref_4}] found in another paragraph.",
                    source_reference=distant_ref_4,
                    target_misconception="Selecting a true factoid that pertains to a different paragraph.",
                ),
            ]
        elif is_intermediate:
            factual_stem = f"According to Paragraph {factual_anchor[0]}, which of the following is true regarding {get_topic_noun(factual_anchor[2])}?"
            factual_para_target = (
                f"Accurately paraphrase the concrete fact in [{factual_anchor[1]}]: '{factual_anchor[2]}' "
                "(8 to 14 words, zero 3-gram verbatim copying)."
            )
            factual_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.POLARITY_INVERSION,
                    blueprint_instruction="Swap a key attribute or predicate with its direct opposite.",
                    target_misconception="Inverting a key factual attribute stated in the text.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.EXTREME_WORD,
                    blueprint_instruction="Frame the factual detail as an absolute or universal phenomenon.",
                    target_misconception="Unwarranted generalization from a specific case.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Introduce a verifiable fact from sentence [{distant_ref_4}] that answers a different question.",
                    source_reference=distant_ref_4,
                    target_misconception="Selecting a true factoid that pertains to a different paragraph.",
                ),
            ]
        else:
            factual_stem = f"According to Paragraph {factual_anchor[0]}, what specific operational condition applies to {get_topic_noun(factual_anchor[2])}?"
            factual_para_target = (
                f"Provide a rigorous academic paraphrase of [{factual_anchor[1]}]: '{factual_anchor[2]}' "
                "(10 to 16 words, zero 3-gram verbatim copying)."
            )
            factual_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.POLARITY_INVERSION,
                    blueprint_instruction="Invert the specific technical condition or constraint described in the text.",
                    target_misconception="Inverting the operational polarity of the factual constraint.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.MODALITY_ESCALATION,
                    blueprint_instruction="Assert that this operational condition is an unavoidable, catastrophic failure mode.",
                    target_misconception="Sensationalizing a routine operational constraint.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Extract an accurate detail from sentence [{distant_ref_4}] located in an entirely different paragraph.",
                    source_reference=distant_ref_4,
                    target_misconception="Selecting a true factoid that pertains to a different paragraph.",
                ),
            ]

        blueprints.append(
            ReadingItemBlueprint(
                item_index=4,
                skill_type=DiscourseRelationType.FACTUAL_MECHANISM,
                target_paragraph_index=factual_anchor[0],
                anchor_sentence_id=factual_anchor[1],
                anchor_sentence_text=factual_anchor[2],
                question_stem_prompt=factual_stem,
                paraphrase_target_instruction=factual_para_target,
                distractors=factual_distractors,
                tier=tier,
                passage_words=passage_words,
            )
        )

        # -------------------------------------------------------------
        # ITEM 5: MODALITY_HEDGING (Stance / Recommendation)
        # -------------------------------------------------------------
        if graph.modality_spans:
            modality_anchor = graph.modality_spans[-1]
        else:
            final_paras = [s for s in graph.sentence_pool if s[0] == len(graph.paragraphs)]
            modality_anchor = final_paras[-1] if final_paras else graph.sentence_pool[-1]

        distant_ref_5 = graph.sentence_pool[0][1]

        if is_foundational:
            modality_stem = f"In Paragraph {modality_anchor[0]}, what does the author suggest or recommend?"
            modality_para_target = (
                f"State the author's suggestion or feeling in anchor [{modality_anchor[1]}]: '{modality_anchor[2]}' "
                "using simple words (5 to 10 words)."
            )
            modality_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.EXTREME_WORD,
                    blueprint_instruction="Turn the gentle suggestion into a strict command using 'must' or 'demands'.",
                    target_misconception="Turning a gentle recommendation into a mandatory rule.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.OPPOSITE_FACT,
                    blueprint_instruction="Suggest the opposite: say the author dislikes or advises against this.",
                    target_misconception="Promoting an attitude directly opposite to the author's perspective.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Include a factual detail from sentence [{distant_ref_5}] from the start of the text.",
                    source_reference=distant_ref_5,
                    target_misconception="Conflating factual context with the author's prescriptive stance.",
                ),
            ]
        elif is_intermediate:
            modality_stem = f"What is the author's perspective or outlook in Paragraph {modality_anchor[0]}?"
            modality_para_target = (
                f"Paraphrase the hedged or prospective stance expressed in [{modality_anchor[1]}]: '{modality_anchor[2]}' "
                "(8 to 14 words, zero 3-gram verbatim copying)."
            )
            modality_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.MODALITY_ESCALATION,
                    blueprint_instruction="Convert the cautious prospect into an inevitable certainty using 'will guarantee'.",
                    target_misconception="Conflating tentative prospects with definitive guarantees.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.POLARITY_INVERSION,
                    blueprint_instruction="Assert that the author expresses skepticism or disapproval regarding this prospect.",
                    target_misconception="Misjudging authorial sentiment or alignment.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Incorporate a factual statement from sentence [{distant_ref_5}] from Paragraph 1.",
                    source_reference=distant_ref_5,
                    target_misconception="Conflating factual context with the author's prescriptive stance.",
                ),
            ]
        else:
            modality_stem = f"Which statement accurately characterizes the author's epistemic stance in Paragraph {modality_anchor[0]}?"
            modality_para_target = (
                f"Synthesize the author's hedged assessment in [{modality_anchor[1]}]: '{modality_anchor[2]}' "
                "(10 to 16 words, zero 3-gram verbatim copying)."
            )
            modality_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.MODALITY_ESCALATION,
                    blueprint_instruction="Upgrade the cautious hypothetical into an unassailable empirical fact (e.g. 'indisputably proves').",
                    target_misconception="Treating speculative hypotheses as empirical verities.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.POLARITY_INVERSION,
                    blueprint_instruction="Claim that the author rejects or refutes the proposal under discussion.",
                    target_misconception="Inverting the author's evaluative stance.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Extract a verified premise from sentence [{distant_ref_5}] that is descriptive rather than normative.",
                    source_reference=distant_ref_5,
                    target_misconception="Conflating descriptive data with normative stance.",
                ),
            ]

        blueprints.append(
            ReadingItemBlueprint(
                item_index=5,
                skill_type=DiscourseRelationType.MODALITY_HEDGING,
                target_paragraph_index=modality_anchor[0],
                anchor_sentence_id=modality_anchor[1],
                anchor_sentence_text=modality_anchor[2],
                question_stem_prompt=modality_stem,
                paraphrase_target_instruction=modality_para_target,
                distractors=modality_distractors,
                tier=tier,
                passage_words=passage_words,
            )
        )

        # -------------------------------------------------------------
        # ITEM 6: PRAGMATIC_INFERENCE (Single-Step Logical Entailment)
        # -------------------------------------------------------------
        if count >= 6:
            used_so_far = {b.anchor_sentence_id for b in blueprints}
            inf_candidates = [s for s in graph.inference_spans if s[1] not in used_so_far]
            if not inf_candidates:
                inf_candidates = [s for s in graph.sentence_pool if s[1] not in used_so_far and len(s[2]) >= 20]
            inf_anchor = inf_candidates[0] if inf_candidates else graph.sentence_pool[-1]
            topic_subject = get_topic_noun(inf_anchor[2])

            if is_foundational:
                inf_stem = f"In Paragraph {inf_anchor[0]}, what is suggested about {topic_subject}?"
                inf_para_target = (
                    f"State the single-step logical inference implied by anchor [{inf_anchor[1]}]: '{inf_anchor[2]}' "
                    "(5 to 10 words). Do NOT merely repeat explicit facts; state what logically follows."
                )
                inf_distractors = [
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.EXPLICIT_RESTATEMENT,
                        blueprint_instruction="State an explicitly mentioned literal fact from the paragraph (a true fact, but an explicit restatement rather than an inference).",
                        target_misconception="Picking an explicitly stated fact instead of an implied conclusion.",
                    ),
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.OVER_EXTENSION,
                        blueprint_instruction="Make a wild leap or speculate about an extreme real-world consequence not mentioned in the text.",
                        target_misconception="Making an unsubstantiated real-world over-inference.",
                    ),
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.OPPOSITE_FACT,
                        blueprint_instruction="Draw an inference that is the opposite of what the text suggests.",
                        target_misconception="Drawing an inference directly opposite to the textual evidence.",
                    ),
                ]
            elif is_intermediate:
                inf_stem = f"What can be inferred from Paragraph {inf_anchor[0]} regarding {topic_subject}?"
                inf_para_target = (
                    f"Articulate the unstated logical deduction or prerequisite implied by [{inf_anchor[1]}]: '{inf_anchor[2]}' "
                    "(8 to 14 words). Focus on what must necessarily be true based strictly on this premise."
                )
                inf_distractors = [
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.EXPLICIT_RESTATEMENT,
                        blueprint_instruction="Provide an option that merely repeats an explicit factual detail from the text rather than deriving an unstated deduction.",
                        target_misconception="Confusing an explicit textual restatement with an implicit logical inference.",
                    ),
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.OVER_EXTENSION,
                        blueprint_instruction="Extrapolate an unsupported multi-step speculation or real-world assumption far beyond the anchor's logical scope.",
                        target_misconception="Extrapolating multi-step real-world speculation beyond the textual premise.",
                    ),
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.REVERSE_ENTAILMENT,
                        blueprint_instruction="Reverse the direction of logical necessity (e.g. asserting that the consequence is a mandatory prerequisite).",
                        target_misconception="Reversing the direction of logical entailment or necessity.",
                    ),
                ]
            else:
                inf_stem = f"What does Paragraph {inf_anchor[0]} implicitly presuppose or suggest regarding {topic_subject}?"
                inf_para_target = (
                    f"Synthesize the single-step logical entailment (P -> Q) warranted by [{inf_anchor[1]}]: '{inf_anchor[2]}' "
                    "(10 to 16 words). Formulate the latent systemic condition that must hold for the statement to be valid."
                )
                inf_distractors = [
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.EXPLICIT_RESTATEMENT,
                        blueprint_instruction="Present an explicit textual proposition from the paragraph as if it were an unstated deductive presupposition.",
                        target_misconception="Confusing an explicit textual restatement with an implicit logical inference.",
                    ),
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.OVER_EXTENSION,
                        blueprint_instruction="Project a speculative, normative, or remote multi-step geopolitical/technological consequence unsupported by textual logic.",
                        target_misconception="Extrapolating multi-step real-world speculation beyond the textual premise.",
                    ),
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.REVERSE_ENTAILMENT,
                        blueprint_instruction="Commit the fallacy of denying the antecedent or reversing logical sufficiency into absolute necessity.",
                        target_misconception="Reversing the direction of logical entailment or necessity.",
                    ),
                ]

            blueprints.append(
                ReadingItemBlueprint(
                    item_index=6,
                    skill_type=DiscourseRelationType.PRAGMATIC_INFERENCE,
                    target_paragraph_index=inf_anchor[0],
                    anchor_sentence_id=inf_anchor[1],
                    anchor_sentence_text=inf_anchor[2],
                    question_stem_prompt=inf_stem,
                    paraphrase_target_instruction=inf_para_target,
                    distractors=inf_distractors,
                    tier=tier,
                    passage_words=passage_words,
                )
            )

        return blueprints[:count]

    @classmethod
    def format_reading_micro_task_blueprints(
        cls,
        blueprints: List[ReadingItemBlueprint],
    ) -> str:
        """
        Formats a list of ReadingItemBlueprint objects into structured markdown
        ready for injection into the reading_quiz LLM prompt.
        """
        blocks = []
        for bp in blueprints:
            task = bp.to_micro_task()
            d_lines = []
            for i, d in enumerate(task["prescribed_distractor_operators"], 1):
                src = f" (Reference: [{d['source_reference']}])" if d.get("source_reference") and d["source_reference"] != "n/a" else ""
                d_lines.append(f"     * Choice Trap {i} (`{d['operator']}`){src}: {d['instruction']}")
            distractor_block = "\n".join(d_lines)

            block = (
                f"### Item {task['item_index']} Specification ###\n"
                f"- **Discourse Skill**: `{task['skill_type']}` ({task['tier']} tier)\n"
                f"- **Target Paragraph**: Paragraph {task['paragraph']}\n"
                f"- **Anchor Sentence [{task['anchor_id']}]**: \"{task['anchor_text']}\"\n"
                f"- **Mandatory Question Stem**: \"{task['question_stem']}\"\n"
                f"- **Paraphrase Target**: {task['paraphrase_target']}\n"
                f"- **Paraphrase Style**: {task['paraphrase_style']}\n"
                f"- **Option Length Bounds**: {task['option_length_guidance']}\n"
                f"- **Lexical Ceiling Mandate**: {task['lexical_ceiling_instruction']}\n"
                f"- **Prescribed Distractor Operators**:\n"
                f"{distractor_block}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    @classmethod
    def evaluate_reading_item(
        cls,
        item: Dict[str, Any],
        blueprint: ReadingItemBlueprint,
    ) -> ReadingEvaluationResult:
        """
        Level 1 Deterministic Code Gate for Reading Items:
        1. Cardinality: Exactly 4 options.
        2. No Empty Options or Duplicates.
        3. Lexical Leakage Gate: Checks verbatim 3-gram overlap between correct option and anchor text (< 35%).
        4. Option Length Parity Gate: Relative Std Dev <= 25%.
        5. Lexical Ceiling Gate: In Foundational tier (A1/A2), options must not introduce out-of-passage B2+ words.
        """
        options = item.get("options", [])
        correct_idx = item.get("correct_answer_index", 0)
        flags: List[str] = []
        score = 100.0

        # 1. Cardinality
        if len(options) != 4:
            flags.append(f"Cardinality Gate Violation: Expected 4 options, found {len(options)}")
            score -= 40.0

        # 2. Distinctness
        clean_opts = [str(o).strip() for o in options]
        if len(set(clean_opts)) != len(clean_opts):
            flags.append("Option Distinctness Gate Violation: Duplicate choices detected.")
            score -= 30.0

        # 3. Lexical Leakage (Verbatim 3-gram check)
        leakage_ratio = 0.0
        if 0 <= correct_idx < len(clean_opts):
            key_text = clean_opts[correct_idx]
            anchor_text = blueprint.anchor_sentence_text

            def get_3grams(s: str) -> Set[Tuple[str, str, str]]:
                words = re.findall(r"\b[a-zA-Z]{2,}\b", s.lower())
                return set(zip(words, words[1:], words[2:]))

            key_3grams = get_3grams(key_text)
            anchor_3grams = get_3grams(anchor_text)

            if key_3grams and anchor_3grams:
                shared = key_3grams.intersection(anchor_3grams)
                leakage_ratio = len(shared) / len(key_3grams)

            if leakage_ratio > 0.35:
                flags.append(
                    f"Lexical Leakage Gate Violation ({leakage_ratio:.1%} shared 3-grams > 35%): "
                    f"correct answer copied verbatim phrasing from anchor sentence [{blueprint.anchor_sentence_id}]."
                )
                score -= 25.0

        # 4. Option Length Parity Gate
        lengths = [len(o.split()) for o in clean_opts] if clean_opts else [0]
        mean_len = sum(lengths) / len(lengths) if lengths else 1.0
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths) if lengths else 0.0
        std_dev = math.sqrt(variance)
        rel_std_dev = std_dev / mean_len if mean_len > 0 else 0.0

        if rel_std_dev > 0.25:
            flags.append(
                f"Length Parity Gate Warning: Relative StdDev is {rel_std_dev:.1%} (max allowed: 25.0%). Options: {lengths}"
            )
            score -= 10.0

        # 5. Lexical Ceiling Gate (Foundational tier)
        if blueprint.tier == CEFRDifficultyTier.FOUNDATIONAL and blueprint.passage_words:
            ADVANCED_CEFR_TIERS = {"B2", "C1", "C2"}

            def get_word_cefr_level(w: str) -> str:
                try:
                    from librarian.linguistics import LinguisticEngine
                    return str(LinguisticEngine.get_word_cefr(w) or "A1").upper()
                except Exception:
                    pass
                try:
                    from cefrpy import CEFRAnalyzer
                    analyzer = CEFRAnalyzer()
                    res = analyzer.get_word_cefr(w)
                    if res and isinstance(res, tuple) and len(res) > 0:
                        return str(res[0]).upper()
                except Exception:
                    pass
                return "A1"

            b2_violations = []
            for opt in clean_opts:
                opt_tokens = re.findall(r"\b[a-zA-Z]{4,}\b", opt.lower())
                for tok in opt_tokens:
                    if tok not in blueprint.passage_words:
                        lvl = get_word_cefr_level(tok)
                        if lvl in ADVANCED_CEFR_TIERS:
                            b2_violations.append(f"'{tok}' ({lvl})")

            if b2_violations:
                flags.append(
                    f"Lexical Ceiling Warning (Foundational Tier): Options introduced advanced out-of-passage words: {', '.join(set(b2_violations[:4]))}"
                )
                score -= 10.0

        is_valid = (score >= 80.0)
        return ReadingEvaluationResult(
            is_valid=is_valid,
            score=max(0.0, score),
            flags=flags,
            lexical_overlap_ratio=leakage_ratio,
            length_std_dev_ratio=rel_std_dev,
        )

    @classmethod
    def evaluate_reading_quiz(
        cls,
        questions: List[Dict[str, Any]],
        blueprints: List[ReadingItemBlueprint],
    ) -> Dict[str, Any]:
        """
        Evaluates an entire generated reading quiz against its blueprints.
        """
        results = []
        all_valid = True
        total_score = 0.0

        for idx, q in enumerate(questions):
            bp = blueprints[idx] if idx < len(blueprints) else blueprints[-1]
            res = cls.evaluate_reading_item(q, bp)
            if not res.is_valid:
                all_valid = False
            total_score += res.score
            results.append({
                "item_index": idx + 1,
                "skill": bp.skill_type.value,
                "is_valid": res.is_valid,
                "score": res.score,
                "flags": res.flags,
                "lexical_overlap": res.lexical_overlap_ratio,
                "length_std_dev": res.length_std_dev_ratio,
            })

        avg_score = total_score / len(questions) if questions else 0.0
        return {
            "all_valid": all_valid,
            "average_score": avg_score,
            "item_results": results,
        }


# Standalone class alias for direct imports
class DiscourseReadingEngine(QuizReadingMixin):
    pass


__all__ = [
    "QuizReadingMixin",
    "DiscourseReadingEngine",
    "DiscourseRelationType",
    "CEFRDifficultyTier",
    "DistractorFallacyOperator",
    "ReadingDistractorSpec",
    "PrescribedDistractor",
    "ReadingItemBlueprint",
    "DiscourseItemBlueprint",
    "DiscourseGraph",
    "ReadingEvaluationResult",
]
