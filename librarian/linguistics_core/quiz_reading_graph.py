"""
Discourse graph extraction and paragraph analysis for reading comprehension.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .quiz_reading_models import (
    CAUSAL_CONNECTIVES_REGEX,
    CONTRAST_CONNECTIVES_REGEX,
    DiscourseGraph,
    DiscourseRelationType,
    DiscourseSection,
    INFERENCE_TRIGGER_REGEX,
    MODALITY_HEDGES_REGEX,
)

logger = logging.getLogger("librarian.linguistics")


class QuizReadingGraphMixin:
    """
    Mixin providing discourse graph extraction and passage analysis.
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
                cls._spacy_nlp = spacy.load("en_core_web_sm")
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
        top_heading_title = ""
        while i < len(blocks):
            b = blocks[i]
            is_heading = False
            title = ""
            lines = [l.strip() for l in b.split("\n") if l.strip()]
            if b.startswith("#"):
                is_heading = True
                title = b.lstrip("#").strip()
            elif len(lines) <= 2 and all(len(l) < 80 for l in lines) and len(b) < 140:
                has_sub = len(lines) == 2 and lines[1].startswith(("-", "—", ":"))
                no_period = not lines[-1].endswith((".", ";", '"', "'"))
                if has_sub or (no_period and (i == 0 or not lines[0].endswith("."))):
                    is_heading = True
                    title = " - ".join([l.lstrip("-— :").strip() for l in lines]) if has_sub else " ".join(lines)
            elif "\n" not in b and len(b) < 70 and not b.endswith((".", "?", "!", ":", ";", '"', "'")):
                is_heading = True
                title = b.strip("*_ ")

            if is_heading and i + 1 < len(blocks):
                body = blocks[i + 1]
                if i == 0:
                    top_heading_title = title
                    sec_title = f"Paragraph {sec_idx}"
                else:
                    sec_title = title
                sections.append(DiscourseSection(section_index=sec_idx, title=sec_title, body_text=body))
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
            # Recombine sentences that were split inside unclosed quotation marks
            merged_sents: List[str] = []
            pending = ""
            for sent in doc.sents:
                t = sent.text.strip()
                if not t:
                    continue
                if pending:
                    pending += " " + t
                    if pending.count('"') % 2 == 0 and pending.count('“') == pending.count('”'):
                        merged_sents.append(pending)
                        pending = ""
                else:
                    if (t.count('"') % 2 != 0) or (t.count('“') != t.count('”')):
                        pending = t
                    else:
                        merged_sents.append(t)
            if pending:
                merged_sents.append(pending)

            for s_text in merged_sents:
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
            top_heading_title=top_heading_title,
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


__all__ = [
    "QuizReadingGraphMixin",
]
