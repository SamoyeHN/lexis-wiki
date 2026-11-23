"""Multi-modal quiz generation orchestrator (generate_quiz).

Composes:
- QuizShufflerMixin: Fisher-Yates option shuffling and explanation label remapping.
- QuizHydratorMixin: Blueprint field hydration and surface frame rendering.
- QuizSanitizerMixin: Syllabus and extraction sanitization.
- QuizRendererMixin: HTML template rendering with asset inlining.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import config
from ..linguistics import LinguisticEngine
from ..llm import llm
from ..prompts import Prompts
from ..schemas import (
    GrammarExtraction,
    GrammarItem,
    ListeningQuiz,
    MindMapExtraction,
    ReadingQuiz,
    RoutingResult,
    SummaryExtraction,
    TranslationQuiz,
    VocabularyExtraction,
    VocabularyItem,
    VocabularyQuiz,
    validate_and_map,
)
from .quiz_hydrators import QuizHydratorMixin
from .quiz_renderers import QuizRendererMixin
from .quiz_sanitizers import QuizSanitizerMixin
from .quiz_shuffler import QuizShufflerMixin

logger = logging.getLogger("librarian.processor")


class QuizGeneratorMixin(
    QuizShufflerMixin,
    QuizHydratorMixin,
    QuizSanitizerMixin,
    QuizRendererMixin,
):
    """Unified quiz generator orchestrator composing shuffling, hydration, sanitization, and rendering."""

    def generate_quiz(self, unit_name, count=10, template_name="vocabulary", audit_callback=None, listening_archetype=None, listening_scenario=None):
        """
        Generates a quiz handout based on extracted wiki data.
        Optional audit_callback(stage, progress, message, extra=None) for live dashboard progress.
        """
        if isinstance(count, str) and count.strip().lower() in ("vocabulary", "cloze", "reading", "translation", "listening", "video"):
            template_name = count.strip().lower()
            count = 10 if template_name in ("vocabulary", "cloze") else 5
        # 1. Normalize unit_name to extract core unit folder name robustly
        parts = Path(unit_name).parts
        core_name = None
        if "wiki" in parts:
            idx = parts.index("wiki")
            if idx + 1 < len(parts):
                core_name = parts[idx + 1]
        elif "raw" in parts:
            idx = parts.index("raw")
            if idx + 1 < len(parts):
                core_name = parts[idx + 1]
        else:
            core_name = parts[0] if parts else unit_name
            
        if core_name.endswith(".md"): core_name = core_name[:-3]
        elif core_name.endswith(".txt"): core_name = core_name[:-4]
            
        for suffix in ["_vocabulary", "_grammar", "_concepts"]:
            if core_name.endswith(suffix):
                core_name = core_name[:-len(suffix)]

        # Pre-Flight LLM Availability Gate (Skip for 100% pure symbolic cloze):
        # Fail fast before running fresh extraction or quiz blueprint generation if LLM is offline.
        if template_name != "cloze":
            try:
                llm.ensure_available(
                    task_name=f"{core_name}_quiz_{template_name}",
                    prompt_summary=f"Quiz generation ({template_name}_quiz) pre-flight check for unit '{core_name}'.",
                )
            except Exception as pre_err:
                logger.error(f"Pre-flight LLM check failed for {core_name}_quiz_{template_name}: {pre_err}")
                return f"Error: {pre_err}"

        # 2. Resolve Data
        try:
            data = self._load_wiki_data(core_name, template_name)
        except Exception as load_err:
            logger.error(f"Failed loading/extracting wiki data for {core_name} ({template_name}): {load_err}", exc_info=True)
            return f"Error: Failed to prepare source data for {core_name} ({template_name}): {load_err}"
        if not data:
            return f"Error: Missing prerequisite extraction for unit '{core_name}'. Please compile and extract knowledge nodes (extractions/{core_name}_vocabulary.md) before generating quizzes."

        # 3. Build Prompt
        prompt_template, schema_cls = None, None
        if template_name != "cloze":
            prompt_template, schema_cls = Prompts.get(f"{template_name}_quiz")
        
        kwargs = {"count": count}
        unit_headwords: List[str] = []
        unit_grammar_patterns: List[str] = []
        banned_quiz_sentences: List[str] = []
        vocab_sense_map: Dict[str, str] = {}
        has_prescribed_options: bool = False
        blueprint_specs: List[Dict[str, Any]] = []

        if template_name == "vocabulary":
            raw_vocab = data.get("content", "")
            # Extract authorized unit headwords and banned authentic sentences for L1/L2 code gates
            _, unit_headwords, banned_quiz_sentences = self._sanitize_vocab_for_quiz(raw_vocab)

            # Pre-Computed Generative Skeletons (Proficiency MCQ Mode - 100% Single Headwords)
            skeletons = LinguisticEngine.build_precomputed_target_skeletons(raw_vocab, target_count=count, multi_word_ratio=0.0)
            effective_count = min(count, len(skeletons)) if skeletons else 0
            kwargs["count"] = effective_count
            has_prescribed_options = True
            logger.info(f"🎯 Built {len(skeletons)} pre-computed target skeletons (Proficiency MCQ Mode, count capped at {effective_count}).")
            skeleton_bullets = []
            for idx, s in enumerate(skeletons[:effective_count], 1):
                tw_key = str(s.get("target_word") or "").strip().lower()
                if tw_key and s.get("definition"):
                    vocab_sense_map[tw_key] = f"({s.get('part_of_speech', '')}) {s['definition']}".strip()
                prescribed_opts = s.get("prescribed_options", [])
                raw_frame_str = s.get("cloze_slot_frame") or s.get("best_pattern") or "... ____ ..."
                frame_str, surface_notes = self._render_llm_surface_frame(raw_frame_str, s)
                blueprint_specs.append({
                    "target_word": s.get("target_word"),
                    "base_headword": s.get("base_headword"),
                    "part_of_speech": s.get("part_of_speech", ""),
                    "cloze_slot_frame": frame_str,
                    "definition": s.get("definition"),
                    "options": list(prescribed_opts) if prescribed_opts else None,
                    "correct_answer_index": s.get("correct_answer_index", 0) if prescribed_opts else None,
                    "micro_task": s.get("micro_task", ""),
                })
                opts_str = ", ".join(prescribed_opts)
                micro_task_str = s.get("micro_task", "")
                pattern_line = f"- Required Stem Frame: {frame_str}\n"
                extra_note = re.sub(
                    r"^Fit into (?:frame\s+'[^']*'|the Required Stem Frame above)\.\s*",
                    "",
                    micro_task_str,
                ).strip()
                if extra_note.startswith("(") and extra_note.endswith(")."):
                    # Verify that depth only returns to 0 at the very end of the string
                    depth = 0
                    is_single_outer = True
                    for ch in extra_note[:-2]:
                        if ch == "(":
                            depth += 1
                        elif ch == ")":
                            depth -= 1
                            if depth == 0:
                                is_single_outer = False
                                break
                    if is_single_outer and depth == 1:
                        extra_note = extra_note[1:-2].strip() + "."
                if surface_notes:
                    extra_note = " ".join(surface_notes + ([extra_note] if extra_note else [])).strip()
                note_line = f"\n- 🎯 Slot / Grammar Note: {extra_note}" if extra_note else ""
                skeleton_bullets.append(
                    f"### Item {idx} ###\n"
                    f"- Target Word: {s['target_word']}\n"
                    f"- Part of Speech: {s['part_of_speech']}\n"
                    f"- Contextual Definition: {s['definition']}\n"
                    + pattern_line
                    + f"- Prescribed Options: [{opts_str}]\n"
                    f"- Correct Answer Index: {s.get('correct_answer_index', 0)}"
                    + note_line
                )

            kwargs["vocabulary_content"] = (
                f"### TARGET SPECIFICATIONS ({len(skeleton_bullets)} ITEMS) ###\n\n"
                + "\n\n".join(skeleton_bullets)
            )

            if blueprint_specs:
                selected_targets: List[str] = []
                seen_targets_low: set = set()
                for s_item in blueprint_specs:
                    for k_name in ("target_word", "base_headword"):
                        val = str(s_item.get(k_name) or "").strip()
                        if val and val.lower() not in seen_targets_low:
                            seen_targets_low.add(val.lower())
                            selected_targets.append(val)
                if selected_targets:
                    unit_headwords = selected_targets

            kwargs["cefr_level"] = data.get("cefr_level", "B2")
        elif template_name == "reading":
            kwargs["passage_content"] = LinguisticEngine.format_paragraph_numbered_passage(data["passage"])
            cefr = str(data.get("cefr_level", "B2")).upper()
            kwargs["cefr_level"] = cefr
            is_foundational = cefr in ("A1", "A2", "B1")
            kwargs["is_foundational"] = is_foundational
            if cefr in ("A1", "A2"):
                kwargs["cefr_descriptor"] = f"Foundational English (CEFR {cefr})"
            elif cefr == "B1":
                kwargs["cefr_descriptor"] = "Intermediate English (CEFR B1)"
            else:
                kwargs["cefr_descriptor"] = f"Advanced Academic English (CEFR {cefr})"

            # Neuro-symbolic discourse reading blueprints
            try:
                reading_blueprints = LinguisticEngine.build_discourse_blueprints(
                    data["passage"],
                    count=count,
                    cefr_level=cefr,
                )
                kwargs["reading_blueprints_block"] = LinguisticEngine.format_reading_micro_task_blueprints(reading_blueprints)
                blueprint_specs = [bp.to_micro_task() for bp in reading_blueprints]
                logger.info(
                    f"🎯 Built {len(blueprint_specs)} symbolic reading discourse micro-task blueprints "
                    f"({cefr=}, count={count})"
                )
            except Exception as e:
                logger.warning(f"Failed to build reading discourse blueprints: {e}")
                kwargs["reading_blueprints_block"] = ""
        elif template_name == "translation":
            raw_v = data.get("vocab_list", "")
            sanitized_v, unit_headwords, banned_quiz_sentences = self._sanitize_vocab_for_quiz(raw_v)
            raw_g = data.get("grammar_list", "")
            sanitized_g, unit_grammar_patterns = self._sanitize_grammar_for_quiz(raw_g)
            target_lang = self.config.get("target_language") or "Chinese"
            cefr = (data.get("cefr_level") or "B2").upper()
            kwargs["target_language"] = target_lang
            kwargs["language"] = target_lang
            kwargs["cefr_level"] = cefr

            # Build Symbolic Translation Micro-Task Blueprints (Decoupled L1 Negative Transfer Traps)
            translation_blueprints = LinguisticEngine.build_translation_micro_task_blueprints(
                raw_v,
                grammar_content=raw_g,
                target_count=count,
                target_language=target_lang,
                cefr_level=cefr,
            )
            if translation_blueprints:
                effective_count = min(count, len(translation_blueprints))
                kwargs["count"] = effective_count
                has_prescribed_options = True
                blueprint_specs = list(translation_blueprints[:effective_count])
                kwargs["vocabulary_content"] = "\n\n".join(
                    bp["micro_task_block"] for bp in blueprint_specs if bp.get("micro_task_block")
                )
                kwargs["grammar_content"] = sanitized_g or "Included in each Item Specification above."
                logger.info(
                    f"🎯 Built {len(blueprint_specs)} symbolic translation micro-task blueprints "
                    f"(Decoupled Syntactic & Lexical L1 Negative Transfer Operators)."
                )
            else:
                kwargs["vocabulary_content"] = sanitized_v
                kwargs["grammar_content"] = sanitized_g
                if unit_headwords and unit_grammar_patterns:
                    kwargs["count"] = min(count, len(unit_headwords), len(unit_grammar_patterns))
                elif unit_headwords:
                    kwargs["count"] = min(count, len(unit_headwords))

        elif template_name == "listening":
            raw_v = data.get("vocab_list", "")
            sanitized_v, unit_headwords, banned_quiz_sentences = self._sanitize_vocab_for_quiz(raw_v)
            kwargs["vocabulary_content"] = sanitized_v
            cefr = (data.get("cefr_level") or "B2").upper()
            kwargs["cefr_level"] = cefr

            # Retrieve TTS configurations to bind speaker names, genders & accents for Schema-First Live Injection
            from ..tts import tts_service
            tts_service._refresh()

            s1_name = tts_service.get_speaker_name(tts_service.voice_a)
            s2_name = tts_service.get_speaker_name(tts_service.voice_b)
            g1 = tts_service.get_gender(tts_service.voice_a)
            g2 = tts_service.get_gender(tts_service.voice_b)
            a1 = tts_service.get_accent(tts_service.voice_a)
            a2 = tts_service.get_accent(tts_service.voice_b)

            kwargs["speaker_1_name"] = s1_name
            kwargs["speaker_2_name"] = s2_name
            kwargs["speaker_1_gender"] = g1
            kwargs["speaker_2_gender"] = g2
            kwargs["speaker_1_accent"] = a1
            kwargs["speaker_2_accent"] = a2

            if cefr in ("A1", "A2", "A2+"):
                kwargs["dialogue_style_guidance"] = (
                    "Conversational, everyday social or campus service interactions (e.g. discussing schedules, course plans, making arrangements, or solving daily problems). "
                    "Keep speaker turns short and natural (8 to 15 words per turn), using accessible S1/S2 spoken phrasing and concrete dialogue markers."
                )
                kwargs["question_stem_guidance"] = (
                    f"Use direct, unambiguous question stems referring to {s1_name} and {s2_name} by name "
                    f"(e.g. 'What does {s1_name} want to do?', 'Why is {s2_name} concerned?'). "
                    "Avoid complex speculative or hypothetical framing."
                )
                kwargs["option_complexity_guidance"] = (
                    "strictly 12 words or fewer (ideally 3 to 8 words), using only basic everyday A1-A2 vocabulary "
                    "such as 'work hours', 'schedule', 'sign in', 'next', 'cold', 'rest', 'water', 'doctor'"
                )
                kwargs["skill_distribution_guidance"] = (
                    "     * `Detail`: concrete facts, explicitly stated intentions, or direct causes.\n"
                    "     * `Main Idea`: overall topic or mutual agreement.\n"
                    "     * `Inference`: a straightforward, obvious conclusion."
                )
            elif cefr in ("B1", "B1+"):
                kwargs["dialogue_style_guidance"] = (
                    "Everyday study-group, campus, or practical colleague conversation using natural spoken English and contractions (`I'm`, `we've`, `that's`). "
                    "Keep turns conversational and concise (10 to 16 words per turn), expressing clear opinions and practical suggestions without stiff formal jargon."
                )
                kwargs["question_stem_guidance"] = (
                    "Use clear question stems assessing explicit facts, speaker reasoning, and immediate deductions."
                )
                kwargs["option_complexity_guidance"] = (
                    "ideally 5 to 10 words, with natural everyday vocabulary"
                )
                kwargs["skill_distribution_guidance"] = (
                    "     * `Detail`: specific rationale or facts stated by a speaker.\n"
                    "     * `Main Idea`: core theme or opening purpose.\n"
                    "     * `Inference`: speaker attitude or logical takeaway."
                )
            elif cefr == "B2":
                kwargs["dialogue_style_guidance"] = (
                    "Natural, idiomatic spoken discussion between peers, student and advisor, or colleagues (e.g. IELTS Section 3 / university project review). "
                    "Use high-frequency spoken English (S1/S2 vocabulary and natural contractions like `we've`, `that's`, `I'd`) at 15 to 22 words per turn (NEVER exceeding 28 words). "
                    "Anchor the conversation to a tangible project artifact (e.g. presentation slide, survey sample size, lab booking, or report deadline). "
                    "Difficulty at B2 comes from tracking pragmatic stance shifts, polite hedges, and conversational updates—NEVER from dense academic jargon. "
                    "DO NOT use essay connectors like 'furthermore', 'moreover', or 'in conclusion'; use spoken signposts like 'to be fair', 'the thing is', or 'on the flip side'."
                )
                kwargs["question_stem_guidance"] = (
                    "Use clear, natural question stems testing speaker intention, stance shifts, and agreed next steps."
                )
                kwargs["option_complexity_guidance"] = (
                    "ideally 6 to 12 words, paraphrasing spoken meaning without overly formal or bookish vocabulary"
                )
                kwargs["skill_distribution_guidance"] = (
                    "     * `Detail`: specific rationale or constraints stated by a speaker.\n"
                    "     * `Main Idea`: opening purpose or overall topic.\n"
                    "     * `Inference`: speaker attitude, hedged disagreement, or final outcome."
                )
            else:
                kwargs["dialogue_style_guidance"] = (
                    "University seminar, tutorial consultation, or project review (e.g. IELTS Listening Section 3 / TOEFL academic discussion where two students, colleagues, or a tutor and student evaluate project trade-offs or methodology). "
                    "Turns MUST be genuinely spoken at 18 to 25 words per turn (NEVER exceeding 30 words), anchored to a concrete project artifact (e.g. slide diagram, questionnaire numbers, or deadline constraint). "
                    "When theoretical terms appear, unpack them with an everyday spoken analogy. "
                    "NEVER use written essay connectors like 'furthermore', 'moreover', or 'in conclusion'; use natural spoken signposts like 'the bottom line is', 'to be fair', or 'having said that'."
                )
                kwargs["question_stem_guidance"] = (
                    "Use rigorous assessment stems testing speaker stance, implicit qualifications, and analytical deductions."
                )
                kwargs["option_complexity_guidance"] = (
                    "8 to 16 words, with nuanced and precise paraphrasing"
                )
                kwargs["skill_distribution_guidance"] = (
                    "     * `Detail`: key nuances, qualifying conditions, or exact rationales.\n"
                    "     * `Main Idea`: overarching academic purpose, consensus, or disagreement.\n"
                    "     * `Inference`: deductions warranted by tone, stance, or logic."
                )

            # Resolve broad thematic topic from source frontmatter (unit_title / text_title) or Summary Core Concepts
            summary_txt = data.get("summary_content", "")
            raw_src_txt = data.get("raw_source_content") or data.get("source_content") or ""
            core_unit = data.get("unit_core_name", core_name)

            thematic_concepts = []
            if raw_src_txt:
                m_u_title = re.search(r'^unit_title:\s*["\']?([^"\'\r\n]+)["\']?', raw_src_txt, re.MULTILINE)
                m_t_title = re.search(r'^text_title:\s*["\']?([^"\'\r\n]+)["\']?', raw_src_txt, re.MULTILINE)
                m_doc_title = re.search(r'^title:\s*["\']?([^"\'\r\n]+)["\']?', raw_src_txt, re.MULTILINE)
                u_t = m_u_title.group(1).strip() if m_u_title else ""
                t_t = m_t_title.group(1).strip() if m_t_title else (m_doc_title.group(1).strip() if m_doc_title else "")
                if u_t and t_t and u_t.lower() != t_t.lower():
                    thematic_concepts.append(f"1. {u_t} ({t_t})")
                elif u_t or t_t:
                    thematic_concepts.append(f"1. {u_t or t_t}")

            if summary_txt:
                # Extract first sentence of Narrative Overview for concrete situational domain grounding
                m_narrative = re.search(r'##\s*Narrative Overview\s*\n+([^#\n]+(?:\.|\?|!))', summary_txt)
                if m_narrative:
                    narrative_snippet = re.sub(r'\[\[(.*?)\]\]', r'\1', m_narrative.group(1).strip())
                    if narrative_snippet and len(narrative_snippet) <= 220:
                        thematic_concepts.append(f"Domain Context: {narrative_snippet}")

                concept_matches = re.findall(r'###\s*\[\[(.*?)\]\]', summary_txt)
                if not concept_matches:
                    concept_matches = re.findall(r'###\s*([^\n]+)', summary_txt)
                if concept_matches:
                    clean_concepts = [
                        c.strip() for c in concept_matches
                        if c.strip() and not c.strip().lower().startswith("narrative")
                    ]
                    start_idx = len(thematic_concepts) + 1
                    for idx, c in enumerate(clean_concepts[:2], start_idx):
                        thematic_concepts.append(f"{idx}. {c}")

            if not thematic_concepts:
                thematic_concepts.append(f"1. {core_unit.replace('_', ' ')}")

            thematic_topic_str = "\n".join(thematic_concepts)
            kwargs["thematic_topic"] = thematic_topic_str

            forced_arch = ""
            if listening_archetype:
                forced_arch = str(listening_archetype).strip()
            elif data.get("listening_archetype"):
                forced_arch = str(data["listening_archetype"]).strip()
            elif raw_src_txt:
                m_arch = re.search(r'^listening_archetype:\s*["\']?([^"\'\r\n]+)["\']?', raw_src_txt, re.MULTILINE)
                if m_arch:
                    forced_arch = m_arch.group(1).strip()

            forced_scenario = ""
            if listening_scenario:
                forced_scenario = str(listening_scenario).strip()
            elif data.get("listening_scenario"):
                forced_scenario = str(data["listening_scenario"]).strip()
            elif raw_src_txt:
                m_sc = re.search(r'^listening_scenario:\s*["\']?([^"\'\r\n]+)["\']?', raw_src_txt, re.MULTILINE)
                if m_sc:
                    forced_scenario = m_sc.group(1).strip()

            try:
                listening_bp = LinguisticEngine.build_listening_micro_task_blueprints(
                    raw_v,
                    thematic_topic=thematic_topic_str,
                    target_count=count,
                    cefr_level=cefr,
                    speaker_1_name=s1_name,
                    speaker_2_name=s2_name,
                    forced_archetype_id=forced_arch,
                    forced_scenario_id=forced_scenario,
                    unit_title=u_t,
                )
            except Exception as e:
                logger.warning(f"Failed to build listening micro-task blueprints: {e}")
                listening_bp = None

            if isinstance(listening_bp, dict) and listening_bp.get("combined_prompt_block"):
                kwargs["vocabulary_content"] = listening_bp["combined_prompt_block"]
                blueprint_specs = listening_bp.get("questions") or []
                has_prescribed_options = True

        elif template_name == "video":
            kwargs["transcript_content"] = data["transcript"]
            kwargs["video_url"] = data["video_url"]
            kwargs["video_type"] = data["video_type"]
            cefr = (data.get("cefr_level") or "B2").upper()
            kwargs["cefr_level"] = cefr

            if cefr in ("A1", "A2"):
                kwargs["question_depth_guidance"] = (
                    "Focus on direct, concrete observations and simple facts explicitly described in the video at the timestamp "
                    "(e.g. 'What is shown at [MM:SS]?', 'What problem does the speaker describe?'). Avoid complex synthesis across long segments."
                )
                kwargs["option_complexity_guidance"] = (
                    "Keep options short and straightforward (ideally 4 to 10 words). Avoid dense academic terminology in options."
                )
            elif cefr == "B1":
                kwargs["question_depth_guidance"] = (
                    "Focus on clear causal relationships, operational mechanisms, and primary challenges discussed at the timestamp "
                    "(e.g. 'Why was this method chosen?', 'How does the process function?')."
                )
                kwargs["option_complexity_guidance"] = (
                    "Keep options clear and standard (ideally 6 to 14 words), using accessible technical vocabulary."
                )
            else:
                kwargs["question_depth_guidance"] = (
                    "Focus on technical mechanisms, comparative evaluations, underlying controversies, or future projections discussed at the timestamp."
                )
                kwargs["option_complexity_guidance"] = (
                    "Craft intellectually mature options (8 to 18 words) with nuanced distractors and academic phrasing."
                )

        # 4. Assemble or Generate Quiz Object (100% Pure Symbolic for Cloze, or LLM Direct Structured Output)
        try:
            if template_name == "cloze":
                passage = data.get("passage", "")
                cefr = data.get("cefr_level", "B2")
                if not passage:
                    return f"Error: No passage text found in sources for unit '{core_name}'."
                
                cloze_res = LinguisticEngine.build_discourse_cloze_assessment(
                    passage, target_count=count, cefr_level=cefr
                )
                display_title = (
                    data.get("display_title")
                    or data.get("title")
                    or core_name.replace("_", " ").title()
                )
                quiz_obj = {
                    "title": f"{display_title} Discourse Cloze Assessment",
                    "mode": cloze_res.get("mode", "open_cloze"),
                    "passage": cloze_res["passage"],
                    "masked_passage": cloze_res["masked_passage"],
                    "paragraphs": cloze_res["paragraphs"],
                    "blanks": cloze_res["blanks"],
                    "questions": cloze_res["questions"],
                    "token_count": cloze_res["token_count"],
                    "cefr_level": cefr,
                }
                logger.info(
                    f"⚡ [0-TOKEN CODE GENERATION] Successfully synthesized {len(cloze_res['blanks'])} "
                    f"authentic discourse cloze blanks via 100% pure symbolic pipeline."
                )
            else:
                prompt = prompt_template.format(**kwargs)
                interpolated_schema = self._interpolate_schema(schema_cls, kwargs)
                quiz_obj = llm.chat(
                    [{"role": "user", "content": prompt}],
                    schema=interpolated_schema,
                    task_name=f"{core_name}_quiz_{template_name}"
                )

                if not quiz_obj:
                    return "Error: LLM returned empty quiz data."

            # 4.4. Deterministic Blueprint Hydration: bind blueprint-owned fields
            # (target_word/target_keyword, definition, target_grammar, flaw_type, options, correct_answer_index) directly from code
            if blueprint_specs:
                if template_name == "translation":
                    quiz_obj = self.hydrate_translation_blueprint_items(quiz_obj, blueprint_specs)
                elif template_name == "listening":
                    quiz_obj = self.hydrate_listening_blueprint_items(
                        quiz_obj,
                        blueprint_specs,
                        speaker_1_name=kwargs.get("speaker_1_name", ""),
                        speaker_2_name=kwargs.get("speaker_2_name", ""),
                    )
                elif template_name == "reading":
                    quiz_obj = self.hydrate_reading_blueprint_items(quiz_obj, blueprint_specs)
                elif template_name == "vocabulary":
                    quiz_obj = self.hydrate_vocabulary_blueprint_items(quiz_obj, blueprint_specs)
                else:
                    quiz_obj = self.hydrate_vocabulary_blueprint_items(quiz_obj, blueprint_specs)
            elif template_name == "video":
                quiz_obj = self.hydrate_video_blueprint_items(quiz_obj)

            # 4.5. Randomize options (skip when blueprint prescribes CRC32-balanced options)
            quiz_obj = self._shuffle_quiz_options(quiz_obj, preserve_blueprint_order=has_prescribed_options)

            # Deterministically attach physical CEFR level to quiz_obj
            cefr_val = str(data.get("cefr_level") or kwargs.get("cefr_level") or "B2").upper()
            if isinstance(quiz_obj, dict):
                quiz_obj["cefr_level"] = cefr_val
            else:
                try:
                    setattr(quiz_obj, "cefr_level", cefr_val)
                except AttributeError:
                    pass

            if template_name == "reading":
                reading_vocab = []
                try:
                    reading_vocab = LinguisticEngine.extract_reading_vocabulary(
                        data.get("passage", ""),
                        target_count=8,
                        enable_llm_disambiguation=False,
                    )
                except Exception as e:
                    logger.warning(f"Reading vocabulary extraction skipped: {e}")
                if isinstance(quiz_obj, dict):
                    quiz_obj["passage"] = data["passage"]
                    quiz_obj["vocabulary"] = reading_vocab
                else:
                    quiz_obj.passage = data["passage"]
                    quiz_obj.vocabulary = reading_vocab

            if template_name == "video":
                quiz_obj = self.hydrate_video_blueprint_items(quiz_obj)
                video_vocab = []
                try:
                    video_vocab = LinguisticEngine.extract_video_vocabulary(
                        data.get("transcript", ""),
                        target_count=12,
                        enable_llm_disambiguation=True,
                    )
                except Exception as e:
                    logger.warning(f"Video vocabulary extraction skipped: {e}")
                if isinstance(quiz_obj, dict):
                    quiz_obj["video_url"] = data["video_url"]
                    quiz_obj["video_type"] = data["video_type"]
                    quiz_obj["transcript"] = data["transcript"]
                    quiz_obj["vocabulary"] = video_vocab
                else:
                    quiz_obj.video_url = data["video_url"]
                    quiz_obj.video_type = data["video_type"]
                    quiz_obj.transcript = data["transcript"]
                    quiz_obj.vocabulary = video_vocab

            # 4.6. Two-Level Auditing: Pre-Audit L1 Code Gate & L2 Expert In-Place Surgical Cure
            is_open_cloze = (
                template_name == "cloze"
                and (
                    (isinstance(quiz_obj, dict) and quiz_obj.get("mode") == "open_cloze")
                    or not any(
                        isinstance(q, dict) and q.get("options")
                        for q in (quiz_obj.get("questions", []) if isinstance(quiz_obj, dict) else getattr(quiz_obj, "questions", []))
                    )
                )
            )
            if is_open_cloze:
                l1_defective, l1_defects = self.audit_cloze_integrity(
                    quiz_obj if isinstance(quiz_obj, dict) else dataclasses.asdict(quiz_obj),
                    passage_text=data.get("passage", ""),
                    cefr_level=kwargs.get("cefr_level", "B2"),
                )
                if l1_defects:
                    logger.warning(
                        f"Pure Open Cloze L1 Code Gate flagged {len(l1_defects)} issue(s): {l1_defects}"
                    )
            elif self.config.get("enable_expert_audit", False):
                try:
                    from ..expert_auditor import ExpertAuditor
                    source_context = ""
                    if template_name == "reading":
                        source_context = LinguisticEngine.format_paragraph_numbered_passage(data.get("passage", ""))
                    elif template_name == "video":
                        source_context = data.get("transcript", "")
                    elif template_name == "translation":
                        # For translation, syllabus pools (vocabulary list & grammar patterns with common mistakes)
                        # are passed directly into the Expert Auditor's structured gates.
                        # We do NOT dump redundant raw dictionary definitions into source_context.
                        source_context = ""
                    elif template_name == "listening":
                        script_turns = []
                        raw_script = quiz_obj.get("script") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "script", [])
                        if isinstance(raw_script, list):
                            for turn in raw_script:
                                if isinstance(turn, dict):
                                    spk = turn.get("speaker", "Speaker")
                                    txt = turn.get("text", "")
                                    script_turns.append(f"{spk}: {txt}")
                                elif hasattr(turn, "speaker") and hasattr(turn, "text"):
                                    script_turns.append(f"{turn.speaker}: {turn.text}")
                        source_context = "\n".join(script_turns)
                    elif template_name == "vocabulary":
                        # For vocabulary quizzes, each item must be a standalone sentence test.
                        # We intentionally DO NOT supply the vocabulary dictionary definitions/quoted sentences
                        # to the blind solver, ensuring the judge evaluates pure sentence-level single-fit validity without key-leaks.
                        source_context = ""
                    elif template_name == "cloze":
                        source_context = data.get("passage", "")

                    quiz_dict_eval = dataclasses.asdict(quiz_obj) if dataclasses.is_dataclass(quiz_obj) else quiz_obj
                    if template_name == "vocabulary" and vocab_sense_map and isinstance(quiz_dict_eval, dict):
                        for q_item in quiz_dict_eval.get("questions", []):
                            if isinstance(q_item, dict):
                                tw_q = str(q_item.get("target_word") or "").strip().lower()
                                if tw_q in vocab_sense_map and not q_item.get("contextual_sense"):
                                    q_item["contextual_sense"] = vocab_sense_map[tw_q]

                    # ---------------------------------------------------------
                    # Step 1: Pre-Audit Level 1 Code Gate (Sanitize & Surface Flaws)
                    # ---------------------------------------------------------
                    curr_tgt_lang = kwargs.get("target_language") or self.config.get("target_language") or "Chinese"
                    if template_name == "reading":
                        l1_defective, l1_defects = self.audit_reading_integrity(
                            quiz_dict_eval,
                            passage_text=source_context,
                            cefr_level=kwargs.get("cefr_level", "B2"),
                            blueprint_specs=blueprint_specs,
                        )
                    elif template_name == "cloze":
                        l1_defective, l1_defects = self.audit_cloze_integrity(
                            quiz_dict_eval,
                            passage_text=source_context,
                            cefr_level=kwargs.get("cefr_level", "B2"),
                        )
                    elif template_name == "video":
                        l1_defective, l1_defects = self.audit_video_integrity(
                            quiz_dict_eval,
                            transcript_text=source_context,
                            cefr_level=kwargs.get("cefr_level", "B2")
                        )
                    elif template_name == "listening":
                        l1_defective, l1_defects = self.audit_listening_integrity(
                            quiz_dict_eval,
                            script_text=source_context,
                            cefr_level=kwargs.get("cefr_level", "B2"),
                            unit_headwords=unit_headwords
                        )
                    elif template_name == "translation":
                        l1_defective, l1_defects = self.audit_translation_integrity(
                            quiz_dict_eval,
                            unit_headwords=unit_headwords,
                            target_language=curr_tgt_lang,
                            cefr_level=kwargs.get("cefr_level", "B2"),
                            enforce_minimal_pair=bool(blueprint_specs),
                            passage_text=source_context,
                        )
                    else:
                        l1_defective, l1_defects = self.audit_quiz_integrity(
                            quiz_dict_eval,
                            banned_sentences=banned_quiz_sentences,
                            unit_headwords=unit_headwords,
                            allow_authentic_cloze=False,
                            strict_distractor_recycling=not has_prescribed_options,
                        )

                    if l1_defects:
                        logger.warning(
                            f"Pre-Audit Level 1 Code Gate flagged {len(l1_defects)} issue(s) before Level 2: {l1_defects}"
                        )

                    # ---------------------------------------------------------
                    # Step 2: Level 2 Expert Model Quality Audit & Triage
                    # ---------------------------------------------------------
                    if audit_callback:
                        try:
                            audit_callback(
                                stage="auditing",
                                audit_progress=30,
                                message="🔍 Level 2 Expert Audit: Running psychometric blind evaluation & surgical triage...",
                                extra={"attempt": 1, "max_retries": 1}
                            )
                        except Exception:
                            pass

                    judge_model_to_use = ExpertAuditor.get_judge_model()
                    active_gen_model = llm.model

                    # If generator model and judge model are distinct, unload generator from Ollama to protect VRAM
                    if judge_model_to_use and active_gen_model and judge_model_to_use != active_gen_model:
                        llm.unload_model(active_gen_model)

                    audit_report = ExpertAuditor.audit_quiz(
                        source_context,
                        quiz_dict_eval,
                        judge_model=judge_model_to_use,
                        quiz_type=template_name,
                        target_language=curr_tgt_lang,
                        unit_headwords=unit_headwords,
                        unit_grammar_patterns=unit_grammar_patterns,
                        cefr_level=kwargs.get("cefr_level", "B2"),
                        unit_slug=core_name
                    )

                    # If distinct models, unload judge model to free memory
                    if judge_model_to_use and active_gen_model and judge_model_to_use != active_gen_model:
                        llm.unload_model(judge_model_to_use)

                    if audit_report:
                        # Record raw generation score before cure
                        raw_gen_score = audit_report.get("overall_quality_score", 100)
                        audit_report["raw_generation_score"] = raw_gen_score

                        orig_questions = quiz_dict_eval.get("questions", [])
                        audit_items = audit_report.get("questions", [])
                        cured_questions = list(orig_questions)

                        cure_stats = {"pass": 0, "repair": 0, "rewrite": 0, "discard": 0}

                        # ---------------------------------------------------------
                        # Step 3: In-Place Surgical Cure & Post-Cure L1 Verification
                        # ---------------------------------------------------------
                        for a_idx, qa in enumerate(audit_items):
                            item_slot = qa.get("item_index", a_idx + 1) - 1
                            if not (0 <= item_slot < len(cured_questions)):
                                item_slot = a_idx
                            if not (0 <= item_slot < len(cured_questions)):
                                continue

                            triage = str(qa.get("triage_action", "PASS")).upper()
                            candidate_cure = qa.get("cured_question")

                            if triage in ("REPAIR", "REWRITE") and isinstance(candidate_cure, dict) and candidate_cure:
                                # Modality-specific and defensive pre-normalization for candidate cure
                                for q_alias in ("stem", "revised_question", "new_question", "cured_stem", "revised_stem"):
                                    if q_alias in candidate_cure and not candidate_cure.get("question"):
                                        candidate_cure["question"] = candidate_cure.pop(q_alias)
                                for opt_alias in ("revised_options", "new_options", "cured_options"):
                                    if opt_alias in candidate_cure and not candidate_cure.get("options"):
                                        candidate_cure["options"] = candidate_cure.pop(opt_alias)
                                for idx_alias in ("revised_correct_answer_index", "new_correct_answer_index"):
                                    if idx_alias in candidate_cure and candidate_cure.get("correct_answer_index") is None:
                                        candidate_cure["correct_answer_index"] = candidate_cure.pop(idx_alias)
                                for exp_alias in ("revised_explanation", "new_explanation"):
                                    if exp_alias in candidate_cure and not candidate_cure.get("explanation"):
                                        candidate_cure["explanation"] = candidate_cure.pop(exp_alias)

                                # Normalize options if LLM outputted list of dicts e.g. [{"letter": "A", "text": "..."}]
                                raw_cand_opts = candidate_cure.get("options")
                                if isinstance(raw_cand_opts, list):
                                    normalized_opts = []
                                    cand_corr_idx = candidate_cure.get("correct_answer_index")
                                    for o_i, o_val in enumerate(raw_cand_opts):
                                        if isinstance(o_val, dict):
                                            txt = o_val.get("text") or o_val.get("option_text") or o_val.get("sentence") or ""
                                            if o_val.get("is_correct") is True and cand_corr_idx is None:
                                                cand_corr_idx = o_i
                                        else:
                                            txt = str(o_val)
                                        if template_name in ("vocabulary", "cloze"):
                                            txt = re.sub(r'\s*\([^)]*\)', '', str(txt)).strip()
                                        else:
                                            txt = re.sub(
                                                r'\s*\((?:synonym|antonym|distractor|correct|key|target|meaning|i\.e\.|e\.g\.)\b[^)]*\)',
                                                '',
                                                str(txt),
                                                flags=re.IGNORECASE
                                            ).strip()
                                        normalized_opts.append(txt)
                                    candidate_cure["options"] = normalized_opts
                                    if cand_corr_idx is not None:
                                        try:
                                            candidate_cure["correct_answer_index"] = int(cand_corr_idx)
                                        except (ValueError, TypeError):
                                            pass

                                if isinstance(candidate_cure.get("explanation"), str):
                                    candidate_cure["explanation"] = re.sub(
                                        r'\s*(?:This\s+(?:rewrite|revision|item|stem|change)\s+(?:ensures|eliminates|prevents|fixes|resolves)[^.\n]*\.?)+\s*$',
                                        '',
                                        candidate_cure["explanation"],
                                        flags=re.IGNORECASE
                                    ).strip()

                                orig_item = orig_questions[item_slot] if 0 <= item_slot < len(orig_questions) else {}
                                orig_dict = orig_item if isinstance(orig_item, dict) else (dataclasses.asdict(orig_item) if dataclasses.is_dataclass(orig_item) else {})

                                # Vocabulary fallback: inherit question / target_word / definition / explanation if omitted in cure
                                if template_name in ("vocabulary", "cloze"):
                                    if not candidate_cure.get("question") and orig_dict.get("question") and triage == "REPAIR":
                                        candidate_cure["question"] = orig_dict["question"]
                                    if not candidate_cure.get("explanation") and orig_dict.get("explanation"):
                                        candidate_cure["explanation"] = orig_dict["explanation"]
                                    # For REPAIR, target word was valid, so inheritance is safe.
                                    # For REWRITE, only inherit if target_word was omitted AND original target belongs to authorized unit headwords.
                                    if not candidate_cure.get("target_word"):
                                        orig_tw = orig_dict.get("target_word")
                                        if orig_tw:
                                            if triage == "REPAIR":
                                                candidate_cure["target_word"] = orig_tw
                                            elif unit_headwords:
                                                hw_set = {h.strip().lower() for h in unit_headwords if h.strip()}
                                                if orig_tw.strip().lower() in hw_set:
                                                    candidate_cure["target_word"] = orig_tw
                                    if not candidate_cure.get("definition") and orig_dict.get("definition"):
                                        candidate_cure["definition"] = orig_dict["definition"]

                                # Reading fallback: inherit category if omitted in cure
                                if template_name == "reading":
                                    if not candidate_cure.get("category") and orig_dict.get("category"):
                                        candidate_cure["category"] = orig_dict["category"]

                                # Video fallback: inherit timestamp if omitted in cure
                                if template_name == "video":
                                    if not candidate_cure.get("timestamp") and orig_dict.get("timestamp"):
                                        candidate_cure["timestamp"] = orig_dict["timestamp"]

                                # Listening fallback: inherit category if omitted in cure
                                if template_name == "listening":
                                    if not candidate_cure.get("category") and orig_dict.get("category"):
                                        candidate_cure["category"] = orig_dict["category"]

                                if template_name == "translation":
                                    cand_opts = candidate_cure.get("options")
                                    corr_idx = candidate_cure.get("correct_answer_index", 0)
                                    extracted_opts = []
                                    if isinstance(cand_opts, list) and len(cand_opts) == 2:
                                        extracted_opts = [str(o) for o in cand_opts]
                                    elif candidate_cure.get("idiomatic_translation") and candidate_cure.get("flawed_translation"):
                                        extracted_opts = [
                                            candidate_cure["idiomatic_translation"],
                                            candidate_cure["flawed_translation"]
                                        ]
                                        corr_idx = 0

                                    if len(extracted_opts) == 2:
                                        # Clean option prefixes e.g. "A) " or "B) "
                                        label_strip = r'^(?:[A-Da-d\d][\.\)\:\-]\s*)'
                                        clean_cand_opts = [re.sub(label_strip, '', str(o)).strip() for o in extracted_opts]
                                        candidate_cure["options"] = clean_cand_opts
                                        if not isinstance(corr_idx, int) or corr_idx not in (0, 1):
                                            corr_idx = 0
                                        candidate_cure["correct_answer_index"] = corr_idx
                                        # Always synchronize idiomatic_translation, flawed_translation, and correct_english_answer with clean options
                                        candidate_cure["idiomatic_translation"] = clean_cand_opts[corr_idx]
                                        candidate_cure["flawed_translation"] = clean_cand_opts[1 - corr_idx]
                                        candidate_cure["correct_english_answer"] = clean_cand_opts[corr_idx]

                                    # Preserve original target language stem if cured item accidentally provided English translation as stem
                                    orig_stem = orig_dict.get("translated_sentence", "")
                                    curr_stem = candidate_cure.get("translated_sentence", "")
                                    # If target language is Chinese and curr_stem lacks Chinese chars, restore original stem
                                    if orig_stem:
                                        target_lang_str = str(curr_tgt_lang or "Chinese").lower()
                                        if any(w in target_lang_str for w in ["chinese", "mandarin", "cjk", "中文", "汉语", "漢語"]):
                                            if not re.search(r'[\u4e00-\u9fff]', curr_stem) and re.search(r'[\u4e00-\u9fff]', orig_stem):
                                                candidate_cure["translated_sentence"] = orig_stem

                                    for bp_field in ("target_keyword", "target_grammar", "flaw_type"):
                                        if not candidate_cure.get(bp_field) and orig_dict.get(bp_field):
                                            candidate_cure[bp_field] = orig_dict[bp_field]
                                    if not candidate_cure.get("explanation") and candidate_cure.get("diagnostic_critique"):
                                        candidate_cure["explanation"] = candidate_cure["diagnostic_critique"]

                                # Validate candidate cure through Level 1 Code Gate
                                temp_wrap = {"questions": [candidate_cure]}
                                temp_wrap = self._shuffle_quiz_options(temp_wrap, preserve_blueprint_order=has_prescribed_options)
                                healed_item = temp_wrap["questions"][0]

                                # Modality-specific Level 1 gate on candidate cure
                                cure_l1_defects = []
                                cure_cefr = kwargs.get("cefr_level", "B2")
                                if template_name == "reading":
                                    cure_bp_spec = [blueprint_specs[item_slot]] if (blueprint_specs and item_slot < len(blueprint_specs)) else None
                                    cure_l1_flagged, cure_l1_defects = self.audit_reading_integrity(
                                        {"questions": [healed_item]},
                                        passage_text=source_context,
                                        cefr_level=cure_cefr,
                                        blueprint_specs=cure_bp_spec
                                    )
                                    cure_item_failed = bool(cure_l1_flagged)
                                elif template_name == "video":
                                    cure_l1_flagged, cure_l1_defects = self.audit_video_integrity(
                                        {"questions": [healed_item]},
                                        transcript_text=source_context,
                                        cefr_level=cure_cefr
                                    )
                                    cure_item_failed = bool(cure_l1_flagged)
                                elif template_name == "listening":
                                    cure_l1_flagged, cure_l1_defects = self.audit_listening_integrity(
                                        {"questions": [healed_item]},
                                        script_text=source_context,
                                        cefr_level=cure_cefr,
                                        unit_headwords=unit_headwords
                                    )
                                    cure_item_failed = bool(cure_l1_flagged)
                                elif template_name == "translation":
                                    cure_l1_flagged, cure_l1_defects = self.audit_translation_integrity(
                                        {"questions": [healed_item]},
                                        unit_headwords=unit_headwords,
                                        target_language=curr_tgt_lang,
                                        cefr_level=cure_cefr,
                                        enforce_minimal_pair=bool(blueprint_specs),
                                        passage_text=source_context,
                                    )
                                    cure_item_failed = bool(cure_l1_flagged)
                                elif template_name == "cloze":
                                    cure_l1_flagged, cure_l1_defects = self.audit_cloze_integrity(
                                        {"questions": [healed_item]},
                                        passage_text=source_context,
                                        cefr_level=cure_cefr
                                    )
                                    cure_item_failed = bool(cure_l1_flagged)
                                else:
                                    candidate_batch = list(cured_questions)
                                    candidate_batch[item_slot] = healed_item
                                    cure_l1_flagged, cure_l1_defects = self.audit_quiz_integrity(
                                        {"questions": candidate_batch},
                                        banned_sentences=banned_quiz_sentences,
                                        unit_headwords=unit_headwords,
                                        strict_distractor_recycling=True,
                                        allow_authentic_cloze=False
                                    )
                                    cure_item_failed = item_slot in cure_l1_flagged

                                if not cure_item_failed:
                                    # Candidate passed L1 gate: splice in-place and assign deterministic score
                                    cured_questions[item_slot] = healed_item
                                    qa["single_fit_valid"] = True
                                    if triage == "REPAIR":
                                        qa["pedagogical_score"] = 90
                                        cure_stats["repair"] += 1
                                        logger.info(f"Surgically REPAIRED Item #{item_slot + 1} with in-place expert cure (Score: 90).")
                                    else:
                                        qa["pedagogical_score"] = 95
                                        cure_stats["rewrite"] += 1
                                        logger.info(f"Surgically REWROTE Item #{item_slot + 1} with in-place expert cure (Score: 95).")
                                else:
                                    # Candidate failed L1 gate: reject cure
                                    logger.warning(
                                        f"Candidate expert cure for Item #{item_slot + 1} failed Level 1 gate ({cure_l1_defects}). Keeping original item penalized."
                                    )
                                    cure_stats["discard"] += 1
                            else:
                                if qa.get("single_fit_valid", True):
                                    cure_stats["pass"] += 1
                                else:
                                    cure_stats["discard"] += 1

                        # Prune uncured defective items to protect pedagogical output
                        final_clean_questions = []
                        for idx, q_candidate in enumerate(cured_questions):
                            corresponding_audit = audit_items[idx] if idx < len(audit_items) and isinstance(audit_items[idx], dict) else {}
                            # Keep item if it is single_fit_valid
                            if corresponding_audit.get("single_fit_valid", True):
                                final_clean_questions.append(q_candidate)
                            else:
                                logger.warning(
                                    f"Pruned uncurable defective Item #{idx + 1} from final handout (single_fit_valid=False)."
                                )

                        # Apply cured & pruned questions to quiz_obj
                        if isinstance(quiz_obj, dict):
                            quiz_obj["questions"] = final_clean_questions
                        else:
                            try:
                                quiz_obj.questions = final_clean_questions
                            except AttributeError:
                                pass
                        quiz_obj = self._shuffle_quiz_options(quiz_obj, preserve_blueprint_order=has_prescribed_options)

                        # Re-bind modality-specific properties and deterministic CEFR level
                        if isinstance(quiz_obj, dict):
                            quiz_obj["cefr_level"] = cefr_val
                        else:
                            try:
                                setattr(quiz_obj, "cefr_level", cefr_val)
                            except AttributeError:
                                pass

                        if template_name == "reading":
                            if isinstance(quiz_obj, dict):
                                quiz_obj["passage"] = data["passage"]
                                quiz_obj["vocabulary"] = reading_vocab
                            else:
                                quiz_obj.passage = data["passage"]
                                quiz_obj.vocabulary = reading_vocab
                        elif template_name == "video":
                            if isinstance(quiz_obj, dict):
                                quiz_obj["video_url"] = data["video_url"]
                                quiz_obj["video_type"] = data["video_type"]
                                quiz_obj["transcript"] = data["transcript"]
                            else:
                                quiz_obj.video_url = data["video_url"]
                                quiz_obj.video_type = data["video_type"]
                                quiz_obj.transcript = data["transcript"]

                        # ---------------------------------------------------------
                        # Step 4: Deterministic Post-Cure Scoring
                        # ---------------------------------------------------------
                        total_items_count = len(orig_questions) or len(audit_items) or 1
                        missing_audit_items = max(0, len(orig_questions) - len(audit_items)) if orig_questions else 0
                        audit_report["missing_audit_items"] = missing_audit_items

                        item_scores = [qa.get("pedagogical_score", 90) for qa in audit_items if isinstance(qa, dict)]
                        score_denom = max(len(item_scores), total_items_count)
                        final_avg = round(sum(item_scores) / score_denom) if item_scores else 100
                        audit_report["base_quality_score"] = final_avg

                        # Preserve judge's cap and fail verdict if uncured defects or truncated items remain
                        has_fatal_residual = any(
                            qa.get("single_fit_valid") is False 
                            for qa in audit_items if isinstance(qa, dict)
                        ) or (cure_stats["discard"] > 0) or (missing_audit_items > 0)

                        if has_fatal_residual:
                            audit_report["pass_audit"] = False
                            cap_limit = 59 if missing_audit_items > 0 else 70
                            audit_report["overall_quality_score"] = min(final_avg, cap_limit)
                        else:
                            audit_report["pass_audit"] = (final_avg >= 80)
                            audit_report["overall_quality_score"] = final_avg

                        final_overall_score = audit_report["overall_quality_score"]
                        audit_report["cure_stats"] = cure_stats

                        # Post-cure accuracy updates: successfully cured items now have a unique, verified key
                        valid_items_count = cure_stats["pass"] + cure_stats["repair"] + cure_stats["rewrite"]
                        audit_report["post_cure_accuracy"] = round(valid_items_count / total_items_count, 2)
                        audit_report["blind_solve_accuracy"] = audit_report["post_cure_accuracy"]

                        cured_count = cure_stats["repair"] + cure_stats["rewrite"]
                        # Update summary verdict to disclose in-place surgical cure when actual cures occurred
                        if cured_count > 0:
                            cure_summary = (
                                f"[SURGICAL CURE APPLIED: Pre-Cure {raw_gen_score}% -> Post-Cure {final_overall_score}%. "
                                f"Pass: {cure_stats['pass']}, Repaired: {cure_stats['repair']}, "
                                f"Rewritten: {cure_stats['rewrite']}, Discard/Flawed: {cure_stats['discard']}]."
                            )
                            orig_verdict = str(audit_report.get("summary_verdict", ""))
                            audit_report["summary_verdict"] = f"{cure_summary} {orig_verdict}".strip()

                        # Synchronize Pre-Cure vs Post-Cure scores to the expert audit log file
                        try:
                            logs_dir = Path(self.config.get("project_root", ".")).resolve() / "logs"
                            audit_task = f"{core_name}_audit_{template_name}"
                            matching_logs = sorted(logs_dir.glob(f"*{audit_task}*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
                            if not matching_logs:
                                task_prefix = f"expert_audit_{quiz_dict_eval.get('title', 'quiz')[:20]}"
                                safe_prefix = re.sub(r'[\\/:*?"<>|\r\n]+', '_', task_prefix).strip('_')
                                matching_logs = sorted(logs_dir.glob(f"*_{safe_prefix}*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
                            if matching_logs:
                                latest_log = matching_logs[0]
                                log_txt = latest_log.read_text(encoding="utf-8")
                                cured_tag = " (CURED)" if cured_count > 0 else ""
                                score_block = (
                                    f"=== L2_QUALITY_SCORE: {final_overall_score}.0%{cured_tag} ===\n"
                                    f"=== COMPOSITE_SCORE: {final_overall_score}.0%{cured_tag} ===\n"
                                    f"=== PRE_CURE_SCORE: {raw_gen_score}.0% ===\n"
                                    f"=== POST_CURE_SCORE: {final_overall_score}.0% ===\n"
                                    f"=== SURGICAL_CURE: Repaired={cure_stats['repair']}, Rewritten={cure_stats['rewrite']}, Passed={cure_stats['pass']}, Flawed={cure_stats['discard']} ==="
                                )
                                if "=== COMPOSITE_SCORE:" in log_txt:
                                    log_txt = re.sub(
                                        r"(?:=== L2_QUALITY_SCORE:[^\n]+\n)?=== COMPOSITE_SCORE:\s*[\d\.]+%\s*===(?:\n=== L2_PENALTY:[^\n]+===)?",
                                        score_block,
                                        log_txt
                                    )
                                    latest_log.write_text(log_txt, encoding="utf-8")
                        except Exception as log_err:
                            logger.debug(f"Could not update audit log with post-cure score: {log_err}")

                        # Attach final audit report to quiz_obj
                        if isinstance(quiz_obj, dict):
                            quiz_obj["_expert_audit"] = audit_report
                        else:
                            try:
                                setattr(quiz_obj, "_expert_audit", audit_report)
                            except AttributeError:
                                pass

                        final_accuracy = audit_report.get("blind_solve_accuracy", 1.0)
                        final_passed = audit_report.get("pass_audit", True)

                        logger.info(
                            f"Level 2 In-Place Surgical Cure Complete for {core_name} ({template_name}): "
                            f"Raw Gen: {raw_gen_score}% -> Final Post-Cure: {final_avg}%, "
                            f"Pass: {final_passed}, Blind Acc: {final_accuracy * 100:.1f}%, Stats: {cure_stats}"
                        )

                        if audit_callback:
                            try:
                                if final_passed:
                                    audit_callback(
                                        stage="passed",
                                        audit_progress=100,
                                        message=f"✅ Level 2 Quality Audit PASSED (Final: {final_avg}%, Blind Acc: {final_accuracy*100:.0f}%)",
                                        extra={"score": final_avg, "accuracy": final_accuracy, "passed": True, "cure_stats": cure_stats}
                                    )
                                else:
                                    audit_callback(
                                        stage="completed_with_warnings",
                                        audit_progress=100,
                                        message=f"⚠️ Audit finished with warnings (Score: {final_avg}%). Proceeding with best attempt.",
                                        extra={"score": final_avg, "accuracy": final_accuracy, "passed": False, "cure_stats": cure_stats}
                                    )
                            except Exception:
                                pass

                except Exception as audit_err:
                    logger.warning(f"Level 2 Expert Quality Audit encountered non-fatal error: {audit_err}", exc_info=True)



            # 5. TTS for Listening Quiz (Base64 Embedding)
            audio_url = None
            if template_name == "listening":
                # Detect dialogue data (dataclass or dict)
                is_dialogue = isinstance(quiz_obj, ListeningQuiz) or (isinstance(quiz_obj, dict) and "script" in quiz_obj)

                if is_dialogue:
                    from ..tts import tts_service
                    tts_service._refresh()

                    # Synchronize speaker names, genders, roles, accents, and TTS voices
                    _COMMON_FEMALE_NAMES = {
                        "alice", "anna", "aria", "ava", "bella", "chloe", "claire", "clara", "DIANA", "diana",
                        "elena", "ella", "emily", "emma", "eva", "grace", "hannah", "helen", "iris", "isabella",
                        "jade", "jenny", "jessica", "julia", "karen", "kate", "laura", "leah", "libby", "lily",
                        "linda", "lisa", "lucy", "maria", "maya", "mei", "mia", "michelle", "naomi", "natasha",
                        "nicole", "nina", "Nora", "nora", "olivia", "rachel", "rose", "ruby", "samantha",
                        "sara", "sarah", "sofia", "sonia", "sophia", "stella", "tina", "victoria", "wendy", "zoe",
                    }
                    _COMMON_MALE_NAMES = {
                        "adam", "alex", "andrew", "aris", "ben", "brian", "charles", "chris", "christopher",
                        "dan", "daniel", "david", "edward", "eric", "ethan", "evan", "frank", "george", "guy",
                        "harry", "henry", "ian", "isaac", "jack", "jacob", "james", "jason", "john", "jordan",
                        "josh", "kevin", "leo", "lewis", "liam", "lucas", "luke", "marcus", "mark", "matt",
                        "matthew", "max", "michael", "mike", "nathan", "nick", "noah", "oliver", "oscar",
                        "owen", "paul", "peter", "ray", "richard", "robert", "roger", "ryan", "sam", "samuel",
                        "scott", "sean", "simon", "steffan", "stephen", "steve", "steven", "thomas", "tim",
                        "tom", "victor", "william",
                    }

                    def _infer_name_gender(full_name: str) -> str | None:
                        tokens = [
                            re.sub(r"[^a-zA-Z]", "", t).lower()
                            for t in str(full_name or "").split()
                        ]
                        tokens = [t for t in tokens if t and t not in ("dr", "prof", "mr", "ms", "mrs", "miss", "speaker")]
                        if not tokens:
                            return None
                        first = tokens[0]
                        if first in _COMMON_FEMALE_NAMES:
                            return "female"
                        if first in _COMMON_MALE_NAMES:
                            return "male"
                        return None

                    def _resolve_speaker_metadata(s1_name, s2_name, raw_g1, raw_g2, raw_r1, raw_r2):
                        g1_norm = str(raw_g1 or kwargs.get("speaker_1_gender") or "female").strip().lower()
                        g2_norm = str(raw_g2 or kwargs.get("speaker_2_gender") or "male").strip().lower()
                        if g1_norm not in ("female", "male"):
                            g1_norm = "female"
                        if g2_norm not in ("female", "male"):
                            g2_norm = "male" if g1_norm == "female" else "female"

                        inferred_1 = _infer_name_gender(s1_name)
                        inferred_2 = _infer_name_gender(s2_name)
                        if inferred_1 and inferred_2 and inferred_1 != inferred_2:
                            g1_norm, g2_norm = inferred_1, inferred_2
                        elif inferred_1 and inferred_1 != g1_norm:
                            g1_norm = inferred_1
                            g2_norm = "male" if g1_norm == "female" else "female"
                        elif inferred_2 and inferred_2 != g2_norm:
                            g2_norm = inferred_2
                            g1_norm = "male" if g2_norm == "female" else "female"
                        elif g1_norm == g2_norm:
                            g2_norm = "male" if g1_norm == "female" else "female"

                        # Bind accent directly to the actual voice (voice_a=female, voice_b=male) that will speak for each character
                        acc_female = tts_service.get_accent(tts_service.voice_a)
                        acc_male = tts_service.get_accent(tts_service.voice_b)
                        a1_resolved = acc_female if g1_norm == "female" else acc_male
                        a2_resolved = acc_female if g2_norm == "female" else acc_male

                        r1_clean = str(raw_r1 or "").strip()
                        r2_clean = str(raw_r2 or "").strip()
                        if not r1_clean:
                            r1_clean = "Discussion Partner" if g1_norm == "female" else "Colleague"
                        if not r2_clean or r2_clean == r1_clean:
                            r2_clean = "Colleague" if r1_clean != "Colleague" else "Project Partner"
                        return g1_norm, g2_norm, a1_resolved, a2_resolved, r1_clean, r2_clean

                    cfg_s1 = kwargs.get("speaker_1_name") or tts_service.get_speaker_name(tts_service.voice_a)
                    cfg_s2 = kwargs.get("speaker_2_name") or tts_service.get_speaker_name(tts_service.voice_b)

                    if isinstance(quiz_obj, dict):
                        s1 = str(quiz_obj.get("speaker_1") or "").strip()
                        s2 = str(quiz_obj.get("speaker_2") or "").strip()
                        if not s1 or s1.lower() in ("speaker 1", "speaker1"):
                            s1 = cfg_s1
                        if not s2 or s2.lower() in ("speaker 2", "speaker2"):
                            s2 = cfg_s2
                        g1, g2, a1, a2, role1, role2 = _resolve_speaker_metadata(
                            s1, s2,
                            quiz_obj.get("speaker_1_gender"),
                            quiz_obj.get("speaker_2_gender"),
                            quiz_obj.get("speaker_1_role"),
                            quiz_obj.get("speaker_2_role"),
                        )
                        ordered_head = {
                            "title": quiz_obj.get("title", ""),
                            "topic": quiz_obj.get("topic", ""),
                            "speaker_1": s1,
                            "speaker_2": s2,
                            "speaker_1_gender": g1,
                            "speaker_2_gender": g2,
                            "speaker_1_role": role1,
                            "speaker_2_role": role2,
                            "speaker_1_accent": a1,
                            "speaker_2_accent": a2,
                        }
                        for k, v in list(quiz_obj.items()):
                            if k not in ordered_head:
                                ordered_head[k] = v
                        quiz_obj.clear()
                        quiz_obj.update(ordered_head)
                        script_dicts = quiz_obj["script"]
                    else:
                        s1 = str(quiz_obj.speaker_1 or "").strip()
                        s2 = str(quiz_obj.speaker_2 or "").strip()
                        if not s1 or s1.lower() in ("speaker 1", "speaker1"):
                            s1 = cfg_s1
                        if not s2 or s2.lower() in ("speaker 2", "speaker2"):
                            s2 = cfg_s2
                        quiz_obj.speaker_1 = s1
                        quiz_obj.speaker_2 = s2
                        g1, g2, a1, a2, role1, role2 = _resolve_speaker_metadata(
                            s1, s2,
                            quiz_obj.speaker_1_gender,
                            quiz_obj.speaker_2_gender,
                            quiz_obj.speaker_1_role,
                            quiz_obj.speaker_2_role,
                        )
                        quiz_obj.speaker_1_gender = g1
                        quiz_obj.speaker_2_gender = g2
                        quiz_obj.speaker_1_role = role1
                        quiz_obj.speaker_2_role = role2
                        quiz_obj.speaker_1_accent = a1
                        quiz_obj.speaker_2_accent = a2
                        script_dicts = [dataclasses.asdict(t) for t in quiz_obj.script]

                    audio_binary = tts_service.process_script(
                        script_dicts, 
                        return_binary=True,
                        speaker_1=s1,
                        speaker_2=s2,
                        speaker_1_gender=g1,
                        speaker_2_gender=g2
                    )
                    
                    if audio_binary:
                        b64_str = base64.b64encode(audio_binary).decode("utf-8")
                        audio_url = f"data:audio/mp3;base64,{b64_str}"

            # 6. Render and Save
            language = self.config.get("target_language") or "Chinese"
            html_content = self._render_handout(quiz_obj, template_name, audio_url, language=language)
            asset_stem = data.get("source_stem", core_name) if isinstance(data, dict) else core_name
            output_type = template_name
            handout_filename = f"{asset_stem}_{output_type}_quiz.html"
            handout_dir = self.config.wiki_content_path / core_name / "handouts"
            handout_dir.mkdir(parents=True, exist_ok=True)
            handout_path = handout_dir / handout_filename

            # -------------------------------------------------------------
            # Transparent Delivery: Handouts always ship to handouts/
            # Log audit recommendations if items were cured or discarded
            # -------------------------------------------------------------
            quiz_audit = None
            if hasattr(quiz_obj, "_expert_audit") and getattr(quiz_obj, "_expert_audit"):
                quiz_audit = getattr(quiz_obj, "_expert_audit")
            elif hasattr(quiz_obj, "audit") and getattr(quiz_obj, "audit"):
                quiz_audit = getattr(quiz_obj, "audit")
            elif isinstance(quiz_obj, dict):
                quiz_audit = quiz_obj.get("_expert_audit") or quiz_obj.get("audit")

            final_questions = getattr(quiz_obj, "questions", []) or (quiz_obj.get("questions", []) if isinstance(quiz_obj, dict) else [])
            if quiz_audit:
                pass_audit = quiz_audit.get("pass_audit", True)
                discarded_count = (quiz_audit.get("cure_stats") or {}).get("discard", 0)
                available_count = len(final_questions)
                if not pass_audit:
                    logger.warning(
                        f"⚠️ Handout {handout_filename} shipped with audit advisory: {pass_audit=}, "
                        f"{available_count=} questions available, {discarded_count=} discarded."
                    )

            with open(handout_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            return str(handout_path)

        except Exception as e:
            return f"Quiz Generation Error: {e}"

