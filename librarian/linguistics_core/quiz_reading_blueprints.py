"""
Reading comprehension blueprint generation and 10 psychometric distractor fallacy operators.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .quiz_reading_models import (
    CEFRDifficultyTier,
    DiscourseGraph,
    DiscourseItemBlueprint,
    DiscourseRelationType,
    DistractorFallacyOperator,
    PrescribedDistractor,
    ReadingDistractorSpec,
    ReadingItemBlueprint,
)

logger = logging.getLogger("librarian.linguistics")


class QuizReadingBlueprintsMixin:
    """
    Mixin providing psychometrically audited reading micro-task blueprints.
    """

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
        def clean_topic_text(c_text: str) -> str:
            t = re.sub(r"^[\s•\-*#>\d\.\(\)]+", "", c_text)
            t = re.sub(r"\s*-\s*", "-", t)
            t = re.sub(r"\s+'s\b", "'s", t)
            t = re.sub(r"\s+", " ", t).strip()
            t = re.sub(r"^(?:a|an|the|my|your|his|her|its|our|their|this|that|these|those)\s+", "", t, flags=re.I)
            return t.strip()

        def get_topic_noun(s_text: str) -> str:
            stopwords = {
                "this", "that", "these", "those", "their", "there", "which", "where", "about", "after",
                "here", "very", "what", "when", "some", "more", "most", "also", "just", "such", "than",
                "then", "from", "with", "into", "over", "under", "both", "each", "other", "have", "were",
                "been", "would", "could", "should", "will", "shall", "does", "doing", "done", "make", "made",
                "many", "much", "few", "several", "every", "all", "not", "no", "never", "neither", "nor",
                "none", "nobody", "nothing", "and", "but", "or", "so", "as", "well", "if", "while", "even",
                "only", "too", "quite", "really", "almost", "nearly", "up", "down", "out", "off", "on", "in",
                "at", "by", "for", "to", "of", "i", "me", "my", "myself", "we", "us", "our", "ours", "you",
                "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
                "their", "theirs", "is", "am", "are", "was", "be", "has", "had", "having", "did", "can",
                "may", "might", "must", "let", "today", "now", "yesterday", "tomorrow",
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
                "end", "ends", "start", "starts", "beginning", "beginnings", "time", "times",
                "year", "years", "day", "days", "week", "weeks", "month", "months",
                "story", "stories", "example", "examples", "instance", "instances",
                "passage", "paragraph", "author", "article", "text", "section", "detail",
                "details", "question", "questions", "answer", "answers", "word", "words",
                "life", "lives", "rest", "course", "bit", "bits", "side", "sides",
            }
            nlp = cls.get_spacy() if hasattr(cls, "get_spacy") else None
            if nlp:
                try:
                    doc = nlp(s_text)
                    # 1. Prioritize salient named entities (ORG, PERSON, PRODUCT, GPE, etc.)
                    for ent in doc.ents:
                        if ent.label_ in ("ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "NORP", "PERSON"):
                            ent_text = clean_topic_text(ent.text)
                            if (
                                len(ent_text) > 2
                                and ent_text.lower() not in stopwords
                                and ent_text.lower() not in light_nouns
                                and not any(w in stopwords for w in ent_text.lower().split() if len(ent_text.split()) == 1)
                            ):
                                return ent_text

                    # 2. Prioritize key noun chunks (multi-word concepts or specific nouns)
                    for chunk in doc.noun_chunks:
                        if chunk.root.pos_ not in ("NOUN", "PROPN"):
                            continue
                        root_lemma = chunk.root.lemma_.lower()
                        root_text = chunk.root.text.lower()
                        if (
                            root_lemma in light_nouns
                            or root_text in light_nouns
                            or root_lemma in stopwords
                            or root_text in stopwords
                        ):
                            continue

                        c_text = clean_topic_text(chunk.text)
                        if not c_text or len(c_text) < 3:
                            continue

                        first_w = c_text.split()[0].lower()
                        if first_w in ("not", "no", "never", "neither", "nor", "what", "which", "how", "why", "only", "even"):
                            continue

                        if c_text.lower() in light_nouns or c_text.lower() in stopwords:
                            continue

                        return c_text
                except Exception:
                    pass
            return ""

        def format_topic_subject(raw_topic: str) -> str:
            if not raw_topic:
                return ""
            t = raw_topic.strip()
            if t.lower().startswith(("the ", "a ", "an ", "this ", "these ")):
                return t
            if t[0].isupper():
                needs_the = t.startswith(("United ", "People's ")) or t.endswith((
                    "States", "Kingdom", "Union", "Republic", "Group", "Force",
                    "Station", "Program", "Administration", "Organization",
                    "Association", "Department", "Ministry",
                ))
                return f"the {t}" if needs_the else t
            return f"the {t}" if not t.endswith("s") else t

        def score_sentence(s_text: str) -> int:
            score = 10
            words = s_text.split()
            w_len = len(words)
            if w_len < 6:
                score -= 10
            elif w_len < 10:
                score -= 4
            elif 12 <= w_len <= 35:
                score += 4

            # Penalize questions / rhetorical dialogic queries (assessments must test assertions, not questions)
            if s_text.endswith("?") or "?" in s_text:
                score -= 12

            # Penalize meta-announcements and framing
            if re.match(r"^(here\s+(?:is|are|'s)|let(?:\'s|\s+us|\s+me)|of course|in this section|for example|as (?:we|you) know)\b", s_text, re.I):
                score -= 8

            # Penalize dialogic interlocutor framing (e.g. 'you may say', 'you might think', 'warn him')
            if re.search(r"\b(?:you\s+may\s+say|you\s+might\s+say|you\s+may\s+ask|you\s+might\s+think|warn\s+(?:him|her|them))\b", s_text, re.I):
                score -= 10

            # Penalize quotes / direct speech fragments
            if re.match(r'^[“\"\'‘]', s_text) or re.search(r'[”\"\'’]$', s_text) or '"' in s_text or '“' in s_text:
                score -= 5

            # Penalize list bullets if any
            if s_text.startswith(('•', '-', '*')):
                score -= 2

            return score

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
            scored_candidates = sorted(candidates, key=lambda c: score_sentence(c[2]), reverse=True)

            # 1. Prefer candidate in an unused section with positive score
            for c in scored_candidates:
                if c[0] not in used_sections and c[1] not in used_anchors and score_sentence(c[2]) >= 0:
                    return c

            # 2. Prefer unused section from fallback pool or sentence pool with highest sentence score
            unused_sec_indices = [s.section_index for s in graph.sections if s.section_index not in used_sections]
            for target_sec_idx in unused_sec_indices:
                sec_sentences = [
                    s for s in (fallback_pool or graph.sentence_pool)
                    if s[0] == target_sec_idx and s[1] not in used_anchors
                ]
                if sec_sentences:
                    sec_sentences.sort(key=lambda s: score_sentence(s[2]), reverse=True)
                    if score_sentence(sec_sentences[0][2]) >= 0:
                        return sec_sentences[0]

            # 3. Candidate not in used anchors with positive score
            for c in scored_candidates:
                if c[1] not in used_anchors and score_sentence(c[2]) >= 0:
                    return c

            # 4. Any sentence not in used anchors with highest sentence score
            pool = [s for s in (fallback_pool or graph.sentence_pool) if s[1] not in used_anchors]
            if pool:
                pool.sort(key=lambda s: score_sentence(s[2]), reverse=True)
                return pool[0]

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
            if graph.top_heading_title:
                passage_title = graph.top_heading_title
            else:
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
                    blueprint_instruction="Overgeneralize the main idea using restrictive qualifiers (e.g. 'only', 'always'), keeping the phrasing plausible rather than cartoonish.",
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

        c_topic = format_topic_subject(get_topic_noun(causal_anchor[2]))

        if is_foundational:
            if c_topic:
                causal_stem = (
                    f'In the "{c_title}" section, what was the main reason or result mentioned regarding {c_topic}?'
                    if has_custom_c
                    else f"According to Paragraph {causal_anchor[0]}, what was the main reason or result mentioned regarding {c_topic}?"
                )
            else:
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
            if c_topic:
                causal_stem = (
                    f'Based on the "{c_title}" section, what is the primary consequence or cause discussed regarding {c_topic}?'
                    if has_custom_c
                    else f"Based on Paragraph {causal_anchor[0]}, what is the primary consequence or cause discussed regarding {c_topic}?"
                )
            else:
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
            if c_topic:
                causal_stem = (
                    f'In the "{c_title}" section, what key cause-and-effect relationship is described regarding {c_topic}?'
                    if has_custom_c
                    else f"According to Paragraph {causal_anchor[0]}, what key cause-and-effect relationship is described regarding {c_topic}?"
                )
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
                    blueprint_instruction="Over-generalize this causal link into an absolute, universal rule or claim it is the sole factor governing all outcomes (e.g., 'solely determines', 'exclusively guarantees').",
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
        cnt_topic = format_topic_subject(get_topic_noun(contrast_anchor[2]))

        if not is_genuine_contrast:
            item3_skill = DiscourseRelationType.FACTUAL_MECHANISM
            if is_foundational:
                if cnt_topic:
                    contrast_stem = (
                        f'In the "{cnt_title}" section, what action or detail is described regarding {cnt_topic}?'
                        if has_custom_cnt
                        else f"In Paragraph {contrast_anchor[0]}, what action or detail is described regarding {cnt_topic}?"
                    )
                else:
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
                        blueprint_instruction="Overgeneralize the detail into an absolute requirement or universal condition (e.g. 'all people always...', 'requires everyone to...'), keeping the tone plausible rather than cartoonish.",
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
                if cnt_topic:
                    contrast_stem = (
                        f'In the "{cnt_title}" section, what specific practice or detail is highlighted regarding {cnt_topic}?'
                        if has_custom_cnt
                        else f"In Paragraph {contrast_anchor[0]}, what specific practice or detail is highlighted regarding {cnt_topic}?"
                    )
                else:
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
                if cnt_topic:
                    contrast_stem = (
                        f'In the "{cnt_title}" section, what procedure or mechanism is detailed regarding {cnt_topic}?'
                        if has_custom_cnt
                        else f"According to Paragraph {contrast_anchor[0]}, what procedure or mechanism is detailed regarding {cnt_topic}?"
                    )
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
                if cnt_topic:
                    contrast_stem = (
                        f'In the "{cnt_title}" section, what difference or contrast is pointed out concerning {cnt_topic}?'
                        if has_custom_cnt
                        else f"In Paragraph {contrast_anchor[0]}, what difference or contrast is pointed out concerning {cnt_topic}?"
                    )
                else:
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
                if cnt_topic:
                    contrast_stem = (
                        f'In the "{cnt_title}" section, what contrast does the author draw concerning {cnt_topic}?'
                        if has_custom_cnt
                        else f"In Paragraph {contrast_anchor[0]}, what contrast does the author draw concerning {cnt_topic}?"
                    )
                else:
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
                if cnt_topic:
                    contrast_stem = (
                        f'In the "{cnt_title}" section, how does the author qualify or contrast the situation regarding {cnt_topic}?'
                        if has_custom_cnt
                        else f"In Paragraph {contrast_anchor[0]}, how does the author qualify or contrast the situation regarding {cnt_topic}?"
                    )
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
        f_topic = format_topic_subject(get_topic_noun(factual_anchor[2]))

        if is_foundational:
            if f_topic:
                factual_stem = (
                    f'According to the "{f_title}" section, what fact or action is mentioned regarding {f_topic}?'
                    if has_custom_f
                    else f"According to Paragraph {factual_anchor[0]}, what fact or action is mentioned regarding {f_topic}?"
                )
            else:
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
                    blueprint_instruction="State the opposite of the key fact or action using simple words.",
                    target_misconception="Contradicting explicit information mentioned in the paragraph.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.EXTREME_WORD,
                    blueprint_instruction="Add an extreme qualifier like 'only' or 'strictly' to restrict the fact in a plausible way.",
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
                    f'According to the "{f_title}" section, what specific development or requirement is noted regarding {f_topic}?'
                    if has_custom_f
                    else f"According to Paragraph {factual_anchor[0]}, what specific development or requirement is noted regarding {f_topic}?"
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
                    blueprint_instruction="Escalate this specific detail into an unconditional, all-encompassing mandate or claim it applies universally without exception (e.g., 'exclusively requires', 'guarantees in every case').",
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
        m_topic = format_topic_subject(get_topic_noun(modality_anchor[2]))

        if is_foundational:
            if m_topic:
                modality_stem = (
                    f'In the "{m_title}" section, what is the author\'s view or perspective regarding {m_topic}?'
                    if has_custom_m
                    else f"In Paragraph {modality_anchor[0]}, what is the author\'s view or perspective regarding {m_topic}?"
                )
            else:
                modality_stem = (
                    f'In the "{m_title}" section, what is the author\'s view or main point?'
                    if has_custom_m
                    else f"In Paragraph {modality_anchor[0]}, what is the author\'s view or main point?"
                )
            modality_para_target = (
                f"State the author's suggestion or feeling in anchor [{modality_anchor[1]}]: '{modality_anchor[2]}' "
                "using simple words (5 to 10 words, zero 3-gram verbatim copying)."
            )
            modality_distractors = [
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.EXTREME_WORD,
                    blueprint_instruction="Turn the tentative recommendation or possibility into an absolute requirement or certainty (e.g. asserting that riders always suffer injuries or must obey strict mandates), avoiding cartoonish phrasing.",
                    target_misconception="Turning a tentative recommendation into a mandatory rule or certainty.",
                ),
                ReadingDistractorSpec(
                    operator=DistractorFallacyOperator.OPPOSITE_FACT,
                    blueprint_instruction="Suggest the opposite: claim the author strongly disapproves or advises against this, keeping the statement plausible.",
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
            if m_topic:
                modality_stem = (
                    f'What is the author\'s perspective or assessment regarding {m_topic} in the "{m_title}" section?'
                    if has_custom_m
                    else f"What is the author\'s perspective or assessment regarding {m_topic} in Paragraph {modality_anchor[0]}?"
                )
            else:
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
            if m_topic:
                modality_stem = (
                    f'Which statement accurately characterizes the author\'s evaluative stance regarding {m_topic} in the "{m_title}" section?'
                    if has_custom_m
                    else f"Which statement accurately characterizes the author\'s evaluative stance regarding {m_topic} in Paragraph {modality_anchor[0]}?"
                )
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
            topic_subject = format_topic_subject(raw_topic) or "this topic"

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

        # Sort items: Keep MACRO_GIST at Item 1, PRAGMATIC_INFERENCE at the end, sort middle paragraph-specific items chronologically by target_paragraph_index
        macro_items = [b for b in selected if b.skill_type == DiscourseRelationType.MACRO_GIST]
        inf_items = [b for b in selected if b.skill_type == DiscourseRelationType.PRAGMATIC_INFERENCE]
        local_items = [
            b for b in selected
            if b.skill_type not in (DiscourseRelationType.MACRO_GIST, DiscourseRelationType.PRAGMATIC_INFERENCE)
        ]
        local_items.sort(key=lambda b: b.target_paragraph_index)

        ordered = macro_items + local_items + inf_items
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


__all__ = [
    "QuizReadingBlueprintsMixin",
]
