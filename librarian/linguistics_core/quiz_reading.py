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

    _spacy_nlp = None

    @classmethod
    def get_spacy(cls):
        """
        Lazily loads and returns the spaCy en_core_web_sm pipeline.
        """
        if getattr(cls, "_spacy_nlp", None) is None:
            try:
                import spacy
                cls._spacy_nlp = spacy.load("en_core_web_sm", disable=["ner"])
            except Exception:
                cls._spacy_nlp = False
        return cls._spacy_nlp if cls._spacy_nlp is not False else None

    @classmethod
    def extract_discourse_graph(cls, text: str) -> DiscourseGraph:
        """
        Parses text into paragraphs, sections, and deterministic rhetorical spans.
        Uses spaCy sentence segmentation for sentence ID anchoring.
        """
        # Clean markdown frontmatter and syllabus if present
        clean_text = re.sub(r"^---[\s\S]*?---\s*", "", text).strip()
        clean_text = re.sub(r"<syllabus>[\s\S]*?</syllabus>", "", clean_text).strip()
        clean_text = re.sub(r"##\s*Syllabus[\s\S]*", "", clean_text, flags=re.IGNORECASE).strip()

        raw_blocks = [b.strip() for b in clean_text.split("\n\n") if b.strip()]
        blocks: List[str] = []
        for b in raw_blocks:
            # Strip passage/text level titles like '## Passage A' or '## Text A'
            if re.match(r"^#+\s*(?:Passage|Text)\s+[A-Z]", b, re.I):
                continue
            # Strip top-level bold passage title if at the very beginning
            if re.match(r"^\*\*.*?\*\*$", b) and len(blocks) == 0:
                continue
            blocks.append(b)

        sections: List[DiscourseSection] = []
        i = 0
        sec_idx = 1
        while i < len(blocks):
            b = blocks[i]
            is_heading = False
            title = ""
            if b.startswith("#"):
                is_heading = True
                title = b.lstrip("#").strip()
            elif "\n" not in b and len(b) < 70 and not b.endswith((".", "?", "!", ":", ";", '"', "'")):
                is_heading = True
                title = b.strip("*_ ")

            if is_heading and i + 1 < len(blocks):
                body = blocks[i + 1]
                sections.append(DiscourseSection(section_index=sec_idx, title=title, body_text=body))
                sec_idx += 1
                i += 2
            else:
                title = f"Paragraph {sec_idx}"
                sections.append(DiscourseSection(section_index=sec_idx, title=title, body_text=b))
                sec_idx += 1
                i += 1

        paragraphs = [sec.body_text for sec in sections]
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
        for sec in sections:
            doc = nlp(sec.body_text)
            for sent in doc.sents:
                s_text = sent.text.strip()
                if len(s_text) < 15:
                    continue
                s_id = f"S-{s_counter}"
                s_counter += 1
                sec.sentence_ids.append(s_id)

                entry = (sec.section_index, s_id, s_text)
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
            sections=sections,
        )

    @classmethod
    def format_paragraph_numbered_passage(cls, text: str) -> str:
        """
        Formats text with explicit [Paragraph N] labels corresponding to
        the DiscourseGraph sections, ensuring unambiguous paragraph references
        for LLM generation prompts and L2 Expert Auditor.
        """
        if not text or not text.strip():
            return text
        graph = cls.extract_discourse_graph(text)
        if not graph.sections:
            return text
        formatted_blocks = []
        for sec in graph.sections:
            if sec.title and not re.match(r"^Paragraph\s+\d+$", sec.title, re.IGNORECASE):
                formatted_blocks.append(f"[Paragraph {sec.section_index}: {sec.title}]\n{sec.body_text}")
            else:
                formatted_blocks.append(f"[Paragraph {sec.section_index}]\n{sec.body_text}")
        return "\n\n".join(formatted_blocks)

    @classmethod
    def extract_reading_vocabulary(
        cls,
        text: str,
        target_count: int = 8,
        enable_llm_disambiguation: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Deterministically extracts academic vocabulary for reading comprehension handouts
        from the passage text (0 tokens if disambiguation disabled or 1-token WSD).
        """
        if not text or not text.strip():
            return []
        try:
            raw_items = cls.extract_deterministic_vocabulary(
                text,
                target_count=target_count,
                enable_llm_disambiguation=enable_llm_disambiguation,
            )
        except Exception as e:
            logger.warning(f"Failed deterministic reading vocabulary extraction: {e}")
            raw_items = []
        vocab_items = []
        for item in raw_items[:target_count]:
            vocab_items.append({
                "word": item["word"],
                "context_sentence": item.get("quoted_sentence", ""),
                "part_of_speech": item.get("part_of_speech", "noun"),
                "definition": item.get("definition", ""),
                "example_usage": item.get("example_usage", ""),
            })
        return vocab_items

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
            stopwords = {
                "this", "that", "these", "those", "their", "there", "which", "where", "about", "after",
                "here", "very", "what", "when", "some", "more", "most", "also", "just", "such", "than",
                "then", "from", "with", "into", "over", "under", "both", "each", "other", "have", "were",
                "been", "would", "could", "should", "will", "shall", "does", "doing", "done", "make", "made",
            }
            light_nouns = {
                "thing", "things", "way", "ways", "fact", "facts", "case", "cases",
                "point", "points", "part", "parts", "aspect", "aspects", "degree",
                "degrees", "level", "levels", "extent", "extents", "type", "types",
                "kind", "kinds", "sort", "sorts", "item", "items", "matter", "matters",
                "idea", "ideas", "sense", "senses", "view", "views", "topic", "topics",
                "reason", "reasons", "result", "results", "term", "terms", "manner",
                "piece", "pieces", "lot", "lots", "number", "numbers", "amount", "amounts",
                "person", "people", "one", "ones", "something", "anything", "nothing",
            }
            nlp = cls.get_spacy() if hasattr(cls, "get_spacy") else None
            if nlp:
                try:
                    doc = nlp(s_text)
                    nouns = [
                        t.text for t in doc
                        if t.pos_ in ("NOUN", "PROPN")
                        and t.text.lower() not in stopwords
                        and t.text.lower() not in light_nouns
                        and len(t.text) > 2
                    ]
                    if nouns:
                        return nouns[0]
                except Exception:
                    pass
            matches = re.findall(r"\b[A-Z][a-z]+\b|\b[a-z]{4,}\b", s_text)
            candidates = [m for m in matches if m.lower() not in stopwords and m.lower() not in light_nouns]
            return candidates[0] if candidates else ""

        used_sections: Set[int] = set()
        used_anchors: Set[str] = set()
        used_distant_refs: Set[str] = set()

        def get_sec_title(sec_idx: int) -> str:
            for s in graph.sections:
                if s.section_index == sec_idx:
                    return s.title
            return f"Paragraph {sec_idx}"

        def pick_anchor(
            candidates: List[Tuple[int, str, str]],
            fallback_pool: Optional[List[Tuple[int, str, str]]] = None,
        ) -> Tuple[int, str, str]:
            # 1. Prefer candidate in an unused section
            for c in candidates:
                if c[0] not in used_sections and c[1] not in used_anchors:
                    return c
            # 2. Prefer unused section from fallback pool or sentence pool
            unused_sec_indices = [s.section_index for s in graph.sections if s.section_index not in used_sections]
            if unused_sec_indices:
                target_sec_idx = unused_sec_indices[0]
                sec_sentences = [
                    s for s in (fallback_pool or graph.sentence_pool)
                    if s[0] == target_sec_idx and s[1] not in used_anchors
                ]
                if sec_sentences:
                    return sec_sentences[0]
            # 3. Candidate not in used anchors
            for c in candidates:
                if c[1] not in used_anchors:
                    return c
            # 4. Any sentence not in used anchors
            for s in (fallback_pool or graph.sentence_pool):
                if s[1] not in used_anchors:
                    return s
            return graph.sentence_pool[0]

        def get_distant_ref(target_sec_idx: int) -> Tuple[str, str]:
            # 1. Prefer sentences in a different section, not used as anchor, not yet used as distant ref, with reasonable length (>= 6 words)
            cands = [
                s for s in graph.sentence_pool
                if s[0] != target_sec_idx and s[1] not in used_anchors and s[1] not in used_distant_refs and len(s[2].split()) >= 6
            ]
            if not cands:
                cands = [
                    s for s in graph.sentence_pool
                    if s[0] != target_sec_idx and s[1] not in used_anchors and s[1] not in used_distant_refs
                ]
            if not cands:
                cands = [
                    s for s in graph.sentence_pool
                    if s[0] != target_sec_idx and s[1] not in used_distant_refs
                ]
            if not cands:
                cands = [s for s in graph.sentence_pool if s[0] != target_sec_idx]

            chosen = cands[0] if cands else (graph.sentence_pool[0] if graph.sentence_pool else (1, "S-1", ""))
            used_distant_refs.add(chosen[1])
            return chosen[1], chosen[2]

        # -------------------------------------------------------------
        # Extract title from frontmatter (prefer text_title, title, unit_title, or top heading)
        # -------------------------------------------------------------
        passage_title = ""
        for key in ("text_title", "title", "unit_title"):
            m = re.search(rf"^---\s*[\s\S]*?^{key}:\s*[\"']?([^\"'\n]+)[\"']?", text, re.M)
            if m:
                passage_title = m.group(1).strip()
                break
        if not passage_title:
            b_match = re.search(r"^\*\*([^*\n]+)\*\*", text, re.M)
            passage_title = b_match.group(1).strip() if b_match else "the passage"

        # -------------------------------------------------------------
        # ITEM 1: MACRO_GIST (Global Thesis with Dual-Pole RST Grounding)
        # -------------------------------------------------------------
        # Pole 1: Introductory Thesis Nucleus (Paragraph 1)
        gist_candidates = [s for s in graph.sentence_pool if s[0] == 1]
        gist_anchor = None
        thesis_marker_pattern = r"\b(yet|however|nevertheless|nonetheless|but|instead|true|essential|crucial|real|underlying|meaning)\b"
        for s in gist_candidates:
            if re.search(thesis_marker_pattern, s[2], re.IGNORECASE):
                gist_anchor = s
                break
        if not gist_anchor:
            # Standard expository thesis convention: closing sentence of introductory paragraph
            gist_anchor = gist_candidates[-1] if gist_candidates else (graph.sentence_pool[0] if graph.sentence_pool else (1, "S-1", "Passage thesis"))

        # Pole 2: Concluding Resolution Nucleus (Final Section)
        concl_anchor = None
        last_sec_idx = graph.sections[-1].section_index if graph.sections else 1
        if last_sec_idx > 1:
            concl_cands = [
                s for s in graph.sentence_pool
                if s[0] == last_sec_idx
                and not re.match(r"^\(?(?:Posted|Written|By|Date)\b", s[2], re.I)
                and len(s[2].split()) >= 4
            ]
            if not concl_cands and len(graph.sections) > 1:
                penult_idx = graph.sections[-2].section_index
                concl_cands = [s for s in graph.sentence_pool if s[0] == penult_idx and len(s[2].split()) >= 4]

            res_marker_pattern = r"\b(ultimately|finally|decision|choice|conclude|conclusion|future|growth|therefore|thus|resolution|build|learn|live|world|change|hope|forward)\b"
            for s in concl_cands:
                if re.search(res_marker_pattern, s[2], re.IGNORECASE):
                    concl_anchor = s
                    break
            if not concl_anchor and concl_cands:
                concl_anchor = concl_cands[-1]

        used_sections.add(gist_anchor[0])
        used_anchors.add(gist_anchor[1])
        if concl_anchor:
            used_anchors.add(concl_anchor[1])
        ref_id_1, ref_text_1 = get_distant_ref(gist_anchor[0])
        sec_title_1 = get_sec_title(gist_anchor[0])

        if is_foundational:
            gist_stem = "What is the main idea of this passage?"
            if concl_anchor and concl_anchor[1] != gist_anchor[1]:
                gist_para_target = (
                    f"State the overarching main idea of the entire passage, connecting the opening point in [{gist_anchor[1]}]: '{gist_anchor[2]}' "
                    f"with the concluding resolution in [{concl_anchor[1]}]: '{concl_anchor[2]}'. "
                    f"The correct option MUST align with the overarching topic of the passage title '{passage_title}' "
                    "in simple, clear words (5 to 10 words). Keep key topic words; do NOT copy 3 consecutive words. Must capture the overall thesis, not an isolated opening hook."
                )
            else:
                gist_para_target = (
                    f"State the overarching main idea of the entire passage, guided by [{gist_anchor[1]}]: '{gist_anchor[2]}'. "
                    f"The correct option MUST align with the overarching topic of the passage title '{passage_title}' "
                    "in simple, clear words (5 to 10 words). Keep key topic words; do NOT copy 3 consecutive words. Must capture the overall thesis, not an isolated opening hook."
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
                    blueprint_instruction=f"Mention a true small detail from sentence [{ref_id_1}] instead of the main idea.",
                    source_reference=ref_id_1,
                    source_text=ref_text_1,
                    target_misconception="Mistaking an accurate supporting detail for the overarching thesis.",
                ),
            ]
        elif is_intermediate:
            gist_stem = "Which of the following best describes the main idea of this passage?"
            if concl_anchor and concl_anchor[1] != gist_anchor[1]:
                gist_para_target = (
                    f"Synthesize the overarching thesis of the ENTIRE passage by linking the introductory proposition in [{gist_anchor[1]}]: '{gist_anchor[2]}' "
                    f"with the concluding resolution in [{concl_anchor[1]}]: '{concl_anchor[2]}'. "
                    f"The correct key MUST directly govern the thematic domain defined by the passage title '{passage_title}' "
                    "using standard vocabulary (8 to 14 words, zero 3-gram verbatim copying). Must capture the comprehensive central takeaway across the whole article, strictly avoiding opening anecdotes or local paragraph details."
                )
            else:
                gist_para_target = (
                    f"Synthesize the overarching thesis of the ENTIRE passage (centering on [{gist_anchor[1]}]: '{gist_anchor[2]}'). "
                    f"The correct key MUST directly govern the thematic domain defined by the passage title '{passage_title}' "
                    "using standard vocabulary (8 to 14 words, zero 3-gram verbatim copying). Must capture the comprehensive central takeaway across the whole article, strictly avoiding opening anecdotes or local paragraph details."
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
                    blueprint_instruction=f"Draw an authentic factual claim from sentence [{ref_id_1}] that is accurate but merely a local detail.",
                    source_reference=ref_id_1,
                    source_text=ref_text_1,
                    target_misconception="Mistaking an accurate supporting detail for the overarching thesis.",
                ),
            ]
        else:
            gist_stem = "Which statement best encapsulates the author's primary thesis in this passage?"
            if concl_anchor and concl_anchor[1] != gist_anchor[1]:
                gist_para_target = (
                    f"Synthesize the macro-level thesis of the ENTIRE passage by bridging the initial problem/proposition in [{gist_anchor[1]}]: '{gist_anchor[2]}' "
                    f"with the final resolution in [{concl_anchor[1]}]: '{concl_anchor[2]}'. "
                    f"The correct key MUST govern the holistic thematic domain defined by the passage title '{passage_title}' "
                    "using academic abstraction (10 to 16 words, zero 3-gram verbatim copying). Formulate the core systemic argument of the whole article rather than an isolated premise."
                )
            else:
                gist_para_target = (
                    f"Synthesize the macro-level thesis of the ENTIRE passage (anchored in [{gist_anchor[1]}]: '{gist_anchor[2]}'). "
                    f"The correct key MUST govern the holistic thematic domain defined by the passage title '{passage_title}' "
                    "using academic abstraction (10 to 16 words, zero 3-gram verbatim copying). Formulate the core systemic argument of the whole article rather than an isolated premise."
                )
            gist_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                    blueprint_instruction=f"Extract a verified factual assertion from sentence [{ref_id_1}] that functions as a localized supporting argument rather than the macro-thesis.",
                    source_reference=ref_id_1,
                    source_text=ref_text_1,
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
                section_title=sec_title_1,
            )
        )

        # -------------------------------------------------------------
        # ITEM 2: CAUSAL_CHAIN (Cause & Effect)
        # -------------------------------------------------------------
        causal_anchor = pick_anchor(graph.causal_chains, fallback_pool=[s for s in graph.sentence_pool if s[0] > 1])
        used_sections.add(causal_anchor[0])
        used_anchors.add(causal_anchor[1])
        ref_id_2, ref_text_2 = get_distant_ref(causal_anchor[0])
        c_title = get_sec_title(causal_anchor[0])
        has_custom_c = not bool(re.match(r"^Paragraph\s+\d+$", c_title, re.IGNORECASE))

        if is_foundational:
            causal_stem = (
                f'In the "{c_title}" section, what was the main reason or result mentioned?'
                if has_custom_c
                else f"According to Paragraph {causal_anchor[0]}, what was the main reason or result mentioned?"
            )
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
                    blueprint_instruction=f"Include a true fact from sentence [{ref_id_2}] that does not explain this cause.",
                    source_reference=ref_id_2,
                    source_text=ref_text_2,
                    target_misconception="Selecting a true statement that does not address the question asked in the stem.",
                ),
            ]
        elif is_intermediate:
            causal_stem = (
                f'Based on the "{c_title}" section, what is the primary consequence or cause discussed?'
                if has_custom_c
                else f"Based on Paragraph {causal_anchor[0]}, what is the primary consequence or cause discussed?"
            )
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
                    blueprint_instruction=f"Introduce a true statement from sentence [{ref_id_2}] that is irrelevant to the causal chain.",
                    source_reference=ref_id_2,
                    source_text=ref_text_2,
                    target_misconception="Selecting a true statement that does not address the question asked in the stem.",
                ),
            ]
        else:
            causal_stem = (
                f'In the "{c_title}" section, what causal mechanism is directly responsible for the observed development?'
                if has_custom_c
                else f"According to Paragraph {causal_anchor[0]}, what causal mechanism is directly responsible for the observed development?"
            )
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
                    blueprint_instruction=f"Incorporate a verified factual statement from sentence [{ref_id_2}] that answers a different question.",
                    source_reference=ref_id_2,
                    source_text=ref_text_2,
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
                section_title=c_title,
            )
        )

        # -------------------------------------------------------------
        # ITEM 3: RHETORICAL_CONTRAST (Contrast / Concession) or FACTUAL_MECHANISM Fallback
        # -------------------------------------------------------------
        contrast_anchor = pick_anchor(graph.contrast_spans, fallback_pool=[s for s in graph.sentence_pool if s[1] not in used_anchors])
        used_sections.add(contrast_anchor[0])
        used_anchors.add(contrast_anchor[1])
        ref_id_3, ref_text_3 = get_distant_ref(contrast_anchor[0])
        cnt_title = get_sec_title(contrast_anchor[0])
        has_custom_cnt = not bool(re.match(r"^Paragraph\s+\d+$", cnt_title, re.IGNORECASE))
        is_genuine_contrast = any(contrast_anchor[1] == s[1] for s in graph.contrast_spans)

        if not is_genuine_contrast:
            item3_skill = DiscourseRelationType.FACTUAL_MECHANISM
            if is_foundational:
                contrast_stem = (
                    f'In the "{cnt_title}" section, what action or detail is described?'
                    if has_custom_cnt
                    else f"In Paragraph {contrast_anchor[0]}, what action or detail is described?"
                )
                contrast_para_target = (
                    f"Express the action or detail in anchor [{contrast_anchor[1]}]: '{contrast_anchor[2]}' "
                    "in simple words (5 to 10 words, zero 3-gram verbatim copying)."
                )
                contrast_distractors = [
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.OPPOSITE_FACT,
                        blueprint_instruction="State the opposite of what the author describes using simple words.",
                        target_misconception="Reversing the factual detail or action described in the text.",
                    ),
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.EXTREME_WORD,
                        blueprint_instruction="Use 'every single' or 'all' to turn the detail into an absolute rule.",
                        target_misconception="Overgeneralizing a specific factual detail into an absolute requirement.",
                    ),
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                        blueprint_instruction=f"Borrow familiar wording from sentence [{ref_id_3}] from another paragraph.",
                        source_reference=ref_id_3,
                        source_text=ref_text_3,
                        target_misconception="Recognizing familiar vocabulary from another part of the text.",
                    ),
                ]
            elif is_intermediate:
                contrast_stem = (
                    f'In the "{cnt_title}" section, what specific practice or detail is highlighted?'
                    if has_custom_cnt
                    else f"In Paragraph {contrast_anchor[0]}, what specific practice or detail is highlighted?"
                )
                contrast_para_target = (
                    f"Paraphrase the practice or detail described in [{contrast_anchor[1]}]: '{contrast_anchor[2]}' "
                    "(8 to 14 words, zero 3-gram verbatim copying)."
                )
                contrast_distractors = [
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.POLARITY_INVERSION,
                        blueprint_instruction="Invert the specific detail or claim stated in the text.",
                        target_misconception="Inverting a key factual attribute stated in the text.",
                    ),
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.EXTREME_WORD,
                        blueprint_instruction="Exaggerate the detail into a universal obligation or absolute condition.",
                        target_misconception="Unwarranted generalization from a specific case.",
                    ),
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                        blueprint_instruction=f"Introduce an authentic fact from sentence [{ref_id_3}] that plays no part in this section.",
                        source_reference=ref_id_3,
                        source_text=ref_text_3,
                        target_misconception="Recognizing familiar vocabulary from another part of the text.",
                    ),
                ]
            else:
                contrast_stem = (
                    f'In the "{cnt_title}" section, what procedure or mechanism is detailed?'
                    if has_custom_cnt
                    else f"According to Paragraph {contrast_anchor[0]}, what procedure or mechanism is detailed?"
                )
                contrast_para_target = (
                    f"Reformulate the procedure or mechanism in [{contrast_anchor[1]}]: '{contrast_anchor[2]}' "
                    "(10 to 16 words, zero 3-gram verbatim copying)."
                )
                contrast_distractors = [
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.POLARITY_INVERSION,
                        blueprint_instruction="Invert the specific technical condition or constraint described in the text.",
                        target_misconception="Inverting the operational polarity of the factual constraint.",
                    ),
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.EXTREME_WORD,
                        blueprint_instruction="Assert that this operational detail is an unavoidable absolute.",
                        target_misconception="Sensationalizing a routine operational detail.",
                    ),
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.TRUE_BUT_IRRELEVANT,
                        blueprint_instruction=f"Extract an accurate detail from sentence [{ref_id_3}] located in an entirely different paragraph.",
                        source_reference=ref_id_3,
                        source_text=ref_text_3,
                        target_misconception="Selecting a true factoid that pertains to a different paragraph.",
                    ),
                ]
        else:
            item3_skill = DiscourseRelationType.RHETORICAL_CONTRAST
            if is_foundational:
                contrast_stem = (
                    f'In the "{cnt_title}" section, what difference or contrast is pointed out?'
                    if has_custom_cnt
                    else f"In Paragraph {contrast_anchor[0]}, what difference or contrast is pointed out?"
                )
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
                        blueprint_instruction=f"Borrow familiar wording from sentence [{ref_id_3}] from Paragraph 1.",
                        source_reference=ref_id_3,
                        source_text=ref_text_3,
                        target_misconception="Recognizing familiar vocabulary from another part of the text.",
                    ),
                ]
            elif is_intermediate:
                contrast_stem = (
                    f'In the "{cnt_title}" section, what contrast does the author draw?'
                    if has_custom_cnt
                    else f"In Paragraph {contrast_anchor[0]}, what contrast does the author draw?"
                )
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
                        blueprint_instruction=f"Introduce an authentic fact from sentence [{ref_id_3}] that plays no part in this comparison.",
                        source_reference=ref_id_3,
                        source_text=ref_text_3,
                        target_misconception="Recognizing familiar vocabulary from another part of the text.",
                    ),
                ]
            else:
                contrast_stem = (
                    f'In the "{cnt_title}" section, how does the author qualify or contrast the preceding assertion?'
                    if has_custom_cnt
                    else f"In Paragraph {contrast_anchor[0]}, how does the author qualify or contrast the preceding assertion?"
                )
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
                        blueprint_instruction=f"Incorporate a verified factual claim from sentence [{ref_id_3}] outside this rhetorical boundary.",
                        source_reference=ref_id_3,
                        source_text=ref_text_3,
                        target_misconception="Recognizing familiar vocabulary from another part of the text.",
                    ),
                ]

        blueprints.append(
            ReadingItemBlueprint(
                item_index=3,
                skill_type=item3_skill,
                target_paragraph_index=contrast_anchor[0],
                anchor_sentence_id=contrast_anchor[1],
                anchor_sentence_text=contrast_anchor[2],
                question_stem_prompt=contrast_stem,
                paraphrase_target_instruction=contrast_para_target,
                distractors=contrast_distractors,
                tier=tier,
                passage_words=passage_words,
                section_title=cnt_title,
            )
        )

        # -------------------------------------------------------------
        # ITEM 4: FACTUAL_MECHANISM (Paraphrased Detail)
        # -------------------------------------------------------------
        factual_pool = [s for s in graph.sentence_pool if s[1] not in used_anchors and len(s[2]) >= 20]
        factual_anchor = pick_anchor([], fallback_pool=factual_pool)
        used_sections.add(factual_anchor[0])
        used_anchors.add(factual_anchor[1])
        ref_id_4, ref_text_4 = get_distant_ref(factual_anchor[0])
        f_title = get_sec_title(factual_anchor[0])
        has_custom_f = not bool(re.match(r"^Paragraph\s+\d+$", f_title, re.IGNORECASE))
        f_topic = get_topic_noun(factual_anchor[2])

        if is_foundational:
            factual_stem = (
                f'According to the "{f_title}" section, what fact or action is mentioned?'
                if has_custom_f
                else f"According to Paragraph {factual_anchor[0]}, what fact or action is mentioned?"
            )
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
                    blueprint_instruction=f"State a true detail from sentence [{ref_id_4}] found in another paragraph.",
                    source_reference=ref_id_4,
                    source_text=ref_text_4,
                    target_misconception="Selecting a true factoid that pertains to a different paragraph.",
                ),
            ]
        elif is_intermediate:
            if f_topic:
                factual_stem = (
                    f'According to the "{f_title}" section, which of the following is true regarding {f_topic}?'
                    if has_custom_f
                    else f"According to Paragraph {factual_anchor[0]}, which of the following is true regarding {f_topic}?"
                )
            else:
                factual_stem = (
                    f'According to the "{f_title}" section, which of the following statements is true?'
                    if has_custom_f
                    else f"According to Paragraph {factual_anchor[0]}, which of the following statements is true?"
                )
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
                    blueprint_instruction=f"Introduce a verifiable fact from sentence [{ref_id_4}] that answers a different question.",
                    source_reference=ref_id_4,
                    source_text=ref_text_4,
                    target_misconception="Selecting a true factoid that pertains to a different paragraph.",
                ),
            ]
        else:
            if f_topic:
                factual_stem = (
                    f'According to the "{f_title}" section, what specific operational condition applies to {f_topic}?'
                    if has_custom_f
                    else f"According to Paragraph {factual_anchor[0]}, what specific operational condition applies to {f_topic}?"
                )
            else:
                factual_stem = (
                    f'According to the "{f_title}" section, which of the following statements is true?'
                    if has_custom_f
                    else f"According to Paragraph {factual_anchor[0]}, which of the following statements is true?"
                )
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
                    blueprint_instruction=f"Extract an accurate detail from sentence [{ref_id_4}] located in an entirely different paragraph.",
                    source_reference=ref_id_4,
                    source_text=ref_text_4,
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
                section_title=f_title,
            )
        )

        # -------------------------------------------------------------
        # ITEM 5: MODALITY_HEDGING (Stance / Recommendation)
        # -------------------------------------------------------------
        modality_anchor = pick_anchor(graph.modality_spans, fallback_pool=[s for s in graph.sentence_pool if s[1] not in used_anchors])
        used_sections.add(modality_anchor[0])
        used_anchors.add(modality_anchor[1])
        ref_id_5, ref_text_5 = get_distant_ref(modality_anchor[0])
        m_title = get_sec_title(modality_anchor[0])
        has_custom_m = not bool(re.match(r"^Paragraph\s+\d+$", m_title, re.IGNORECASE))

        if is_foundational:
            modality_stem = (
                f'In the "{m_title}" section, what does the author suggest or recommend?'
                if has_custom_m
                else f"In Paragraph {modality_anchor[0]}, what does the author suggest or recommend?"
            )
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
                    blueprint_instruction=f"Include a factual detail from sentence [{ref_id_5}] from the start of the text.",
                    source_reference=ref_id_5,
                    source_text=ref_text_5,
                    target_misconception="Conflating factual context with the author's prescriptive stance.",
                ),
            ]
        elif is_intermediate:
            modality_stem = (
                f'What is the author\'s perspective or outlook in the "{m_title}" section?'
                if has_custom_m
                else f"What is the author's perspective or outlook in Paragraph {modality_anchor[0]}?"
            )
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
                    blueprint_instruction=f"Incorporate a factual statement from sentence [{ref_id_5}] from Paragraph 1.",
                    source_reference=ref_id_5,
                    source_text=ref_text_5,
                    target_misconception="Conflating factual context with the author's prescriptive stance.",
                ),
            ]
        else:
            modality_stem = (
                f'Which statement accurately characterizes the author\'s epistemic stance in the "{m_title}" section?'
                if has_custom_m
                else f"Which statement accurately characterizes the author's epistemic stance in Paragraph {modality_anchor[0]}?"
            )
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
                    blueprint_instruction=f"Extract a verified premise from sentence [{ref_id_5}] that is descriptive rather than normative.",
                    source_reference=ref_id_5,
                    source_text=ref_text_5,
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
                section_title=m_title,
            )
        )

        # -------------------------------------------------------------
        # ITEM 6: PRAGMATIC_INFERENCE (Single-Step Logical Entailment)
        # -------------------------------------------------------------
        if count >= 6:
            inf_anchor = pick_anchor(graph.inference_spans, fallback_pool=[s for s in graph.sentence_pool if s[1] not in used_anchors and len(s[2]) >= 20])
            used_sections.add(inf_anchor[0])
            used_anchors.add(inf_anchor[1])
            ref_id_6, ref_text_6 = get_distant_ref(inf_anchor[0])
            i_title = get_sec_title(inf_anchor[0])
            has_custom_i = not bool(re.match(r"^Paragraph\s+\d+$", i_title, re.IGNORECASE))
            raw_topic = get_topic_noun(inf_anchor[2])
            if raw_topic:
                if not raw_topic.lower().startswith(("the ", "a ", "an ", "this ", "these ")):
                    topic_subject = f"the {raw_topic}" if not raw_topic.endswith("s") else raw_topic
                else:
                    topic_subject = raw_topic
            else:
                topic_subject = "this topic"

            if is_foundational:
                inf_stem = (
                    f'In the "{i_title}" section, what is suggested about {topic_subject}?'
                    if has_custom_i
                    else f"In Paragraph {inf_anchor[0]}, what is suggested about {topic_subject}?"
                )
                inf_para_target = (
                    f"State what is implied or suggested about {topic_subject} in anchor [{inf_anchor[1]}]: '{inf_anchor[2]}' "
                    "(5 to 10 words, zero 3-gram verbatim copying)."
                )
                inf_distractors = [
                    ReadingDistractorSpec(
                        operator=DistractorFallacyOperator.EXPLICIT_RESTATEMENT,
                        blueprint_instruction="State an explicitly mentioned literal procedural step or minor detail from the text (a literal fact, but not the main inferred conclusion).",
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
                inf_stem = (
                    f'What can be inferred from the "{i_title}" section regarding {topic_subject}?'
                    if has_custom_i
                    else f"What can be inferred from Paragraph {inf_anchor[0]} regarding {topic_subject}?"
                )
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
                inf_stem = (
                    f'What does the "{i_title}" section implicitly presuppose or suggest regarding {topic_subject}?'
                    if has_custom_i
                    else f"What does Paragraph {inf_anchor[0]} implicitly presuppose or suggest regarding {topic_subject}?"
                )
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
                    section_title=i_title,
                )
            )

        selected = blueprints[:count]
        if not selected:
            return selected

        # Sort items: Keep MACRO_GIST at Item 1, sort paragraph-specific items chronologically by target_paragraph_index
        macro_items = [b for b in selected if b.skill_type == DiscourseRelationType.MACRO_GIST]
        local_items = [b for b in selected if b.skill_type != DiscourseRelationType.MACRO_GIST]
        local_items.sort(key=lambda b: b.target_paragraph_index)

        ordered = macro_items + local_items
        for idx, item in enumerate(ordered, 1):
            item.item_index = idx

        return ordered

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
                src_parts = []
                if d.get("source_reference") and d["source_reference"] != "n/a":
                    src_parts.append(f"[{d['source_reference']}]")
                if d.get("source_text") and d["source_text"] != "n/a":
                    st = d["source_text"].strip()
                    if len(st) > 75:
                        st = st[:72] + "..."
                    src_parts.append(f'"{st}"')
                src = f" (Reference: {' '.join(src_parts)})" if src_parts else ""
                d_lines.append(f"     * Choice Trap {i} (`{d['operator']}`){src}: {d['instruction']}")
            distractor_block = "\n".join(d_lines)

            sec_title = task.get("section_title", "")
            has_custom_title = bool(sec_title and not re.match(r"^paragraph\s+\d+$", sec_title, re.IGNORECASE))
            sec_header = f' (Section: "{sec_title}")' if has_custom_title else ""

            block = (
                f"### Item {task['item_index']} Specification ###\n"
                f"- **Discourse Skill**: `{task['skill_type']}` ({task['tier']} tier)\n"
                f"- **Target Paragraph**: Paragraph {task['paragraph']}{sec_header}\n"
                f"- **Anchor Sentence [{task['anchor_id']}]**: \"{task['anchor_text']}\"\n"
                f"- **Mandatory Question Stem**: \"{task['question_stem']}\"\n"
                f"- **Paraphrase Target**: {task['paraphrase_target']}\n"
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
