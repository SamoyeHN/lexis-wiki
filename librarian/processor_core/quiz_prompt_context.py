"""Quiz prompt context and blueprint preparation service.

Constructs modality-specific prompt kwargs, blueprint specifications, and
lexical/grammatical constraints for:
- vocabulary (pre-computed skeletons, single-fit targets, slot frame rendering)
- reading (paragraph-numbered passage, discourse micro-tasks, CEFR descriptors)
- translation (syntactic/lexical L1 negative transfer error traps, minimal pairs)
- listening (speaker metadata, CA turn guidance, functional dialogue blueprints)
- video (timestamps, depth guidance, transcript)
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any, Dict, List, Optional

from ..linguistics import LinguisticEngine

logger = logging.getLogger("librarian.processor")


@dataclasses.dataclass
class QuizPromptContext:
    """Encapsulates prompt parameters, blueprint items, and lexical constraints for quiz generation."""

    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    unit_headwords: List[str] = dataclasses.field(default_factory=list)
    unit_grammar_patterns: List[str] = dataclasses.field(default_factory=list)
    banned_quiz_sentences: List[str] = dataclasses.field(default_factory=list)
    vocab_sense_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    has_prescribed_options: bool = False
    blueprint_specs: List[Dict[str, Any]] = dataclasses.field(default_factory=list)


def _build_vocabulary_context(
    processor: Any,
    data: Dict[str, Any],
    count: int,
    ctx: QuizPromptContext,
) -> None:
    raw_vocab = data.get("content", "")
    _, unit_headwords, banned_quiz_sentences = processor._sanitize_vocab_for_quiz(raw_vocab)
    ctx.unit_headwords = unit_headwords
    ctx.banned_quiz_sentences = banned_quiz_sentences

    skeletons = LinguisticEngine.build_precomputed_target_skeletons(
        raw_vocab, target_count=count, multi_word_ratio=0.0
    )
    effective_count = min(count, len(skeletons)) if skeletons else 0
    ctx.kwargs["count"] = effective_count
    ctx.has_prescribed_options = True
    logger.info(
        f"🎯 Built {len(skeletons)} pre-computed target skeletons "
        f"(Proficiency MCQ Mode, count capped at {effective_count})."
    )

    skeleton_bullets = []
    for idx, s in enumerate(skeletons[:effective_count], 1):
        tw_key = str(s.get("target_word") or "").strip().lower()
        if tw_key and s.get("definition"):
            ctx.vocab_sense_map[tw_key] = f"({s.get('part_of_speech', '')}) {s['definition']}".strip()
        prescribed_opts = s.get("prescribed_options", [])
        raw_frame_str = s.get("cloze_slot_frame") or s.get("best_pattern") or "... ____ ..."
        frame_str, surface_notes = processor._render_llm_surface_frame(raw_frame_str, s)
        ctx.blueprint_specs.append({
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

    ctx.kwargs["vocabulary_content"] = (
        f"### TARGET SPECIFICATIONS ({len(skeleton_bullets)} ITEMS) ###\n\n"
        + "\n\n".join(skeleton_bullets)
    )

    if ctx.blueprint_specs:
        selected_targets: List[str] = []
        seen_targets_low: set = set()
        for s_item in ctx.blueprint_specs:
            for k_name in ("target_word", "base_headword"):
                val = str(s_item.get(k_name) or "").strip()
                if val and val.lower() not in seen_targets_low:
                    seen_targets_low.add(val.lower())
                    selected_targets.append(val)
        if selected_targets:
            ctx.unit_headwords = selected_targets

    ctx.kwargs["cefr_level"] = data.get("cefr_level", "B2")


def _build_reading_context(
    data: Dict[str, Any],
    count: int,
    ctx: QuizPromptContext,
) -> None:
    ctx.kwargs["passage_content"] = LinguisticEngine.format_paragraph_numbered_passage(data["passage"])
    cefr = str(data.get("cefr_level", "B2")).upper()
    ctx.kwargs["cefr_level"] = cefr
    is_foundational = cefr in ("A1", "A2", "B1")
    ctx.kwargs["is_foundational"] = is_foundational
    if cefr in ("A1", "A2"):
        ctx.kwargs["cefr_descriptor"] = f"Foundational English (CEFR {cefr})"
    elif cefr == "B1":
        ctx.kwargs["cefr_descriptor"] = "Intermediate English (CEFR B1)"
    else:
        ctx.kwargs["cefr_descriptor"] = f"Advanced Academic English (CEFR {cefr})"

    try:
        reading_blueprints = LinguisticEngine.build_discourse_blueprints(
            data["passage"],
            count=count,
            cefr_level=cefr,
        )
        ctx.kwargs["reading_blueprints_block"] = LinguisticEngine.format_reading_micro_task_blueprints(
            reading_blueprints
        )
        ctx.blueprint_specs = [bp.to_micro_task() for bp in reading_blueprints]
        logger.info(
            f"🎯 Built {len(ctx.blueprint_specs)} symbolic reading discourse micro-task blueprints "
            f"({cefr=}, count={count})"
        )
    except (TypeError, ValueError, KeyError, RuntimeError) as e:
        logger.warning(f"Failed to build reading discourse blueprints: {e}")
        ctx.kwargs["reading_blueprints_block"] = ""


def _build_translation_context(
    processor: Any,
    data: Dict[str, Any],
    count: int,
    ctx: QuizPromptContext,
) -> None:
    raw_v = data.get("vocab_list", "")
    sanitized_v, unit_headwords, banned_quiz_sentences = processor._sanitize_vocab_for_quiz(raw_v)
    ctx.unit_headwords = unit_headwords
    ctx.banned_quiz_sentences = banned_quiz_sentences

    raw_g = data.get("grammar_list", "")
    sanitized_g, unit_grammar_patterns = processor._sanitize_grammar_for_quiz(raw_g)
    ctx.unit_grammar_patterns = unit_grammar_patterns

    target_lang = processor.config.get("target_language") or "Chinese"
    cefr = (data.get("cefr_level") or "B2").upper()
    ctx.kwargs["target_language"] = target_lang
    ctx.kwargs["language"] = target_lang
    ctx.kwargs["cefr_level"] = cefr

    translation_blueprints = LinguisticEngine.build_translation_micro_task_blueprints(
        raw_v,
        grammar_content=raw_g,
        target_count=count,
        target_language=target_lang,
        cefr_level=cefr,
    )
    if translation_blueprints:
        effective_count = min(count, len(translation_blueprints))
        ctx.kwargs["count"] = effective_count
        ctx.has_prescribed_options = True
        ctx.blueprint_specs = list(translation_blueprints[:effective_count])
        ctx.kwargs["vocabulary_content"] = "\n\n".join(
            bp["micro_task_block"] for bp in ctx.blueprint_specs if bp.get("micro_task_block")
        )
        ctx.kwargs["grammar_content"] = sanitized_g or "Included in each Item Specification above."
        logger.info(
            f"🎯 Built {len(ctx.blueprint_specs)} symbolic translation micro-task blueprints "
            f"(Decoupled Syntactic & Lexical L1 Negative Transfer Operators)."
        )
    else:
        ctx.kwargs["vocabulary_content"] = sanitized_v
        ctx.kwargs["grammar_content"] = sanitized_g
        if unit_headwords and unit_grammar_patterns:
            ctx.kwargs["count"] = min(count, len(unit_headwords), len(unit_grammar_patterns))
        elif unit_headwords:
            ctx.kwargs["count"] = min(count, len(unit_headwords))


def _build_listening_context(
    processor: Any,
    core_name: str,
    data: Dict[str, Any],
    count: int,
    ctx: QuizPromptContext,
    listening_archetype: Optional[str] = None,
    listening_scenario: Optional[str] = None,
) -> None:
    raw_v = data.get("vocab_list", "")
    sanitized_v, unit_headwords, banned_quiz_sentences = processor._sanitize_vocab_for_quiz(raw_v)
    ctx.unit_headwords = unit_headwords
    ctx.banned_quiz_sentences = banned_quiz_sentences
    ctx.kwargs["vocabulary_content"] = sanitized_v
    cefr = (data.get("cefr_level") or "B2").upper()
    ctx.kwargs["cefr_level"] = cefr

    from ..tts import tts_service
    tts_service._refresh()

    s1_name = tts_service.get_speaker_name(tts_service.voice_a)
    s2_name = tts_service.get_speaker_name(tts_service.voice_b)
    g1 = tts_service.get_gender(tts_service.voice_a)
    g2 = tts_service.get_gender(tts_service.voice_b)
    a1 = tts_service.get_accent(tts_service.voice_a)
    a2 = tts_service.get_accent(tts_service.voice_b)

    ctx.kwargs["speaker_1_name"] = s1_name
    ctx.kwargs["speaker_2_name"] = s2_name
    ctx.kwargs["speaker_1_gender"] = g1
    ctx.kwargs["speaker_2_gender"] = g2
    ctx.kwargs["speaker_1_accent"] = a1
    ctx.kwargs["speaker_2_accent"] = a2

    if cefr in ("A1", "A2", "A2+"):
        ctx.kwargs["dialogue_style_guidance"] = (
            "Conversational, everyday social or campus service interactions (e.g. discussing schedules, course plans, making arrangements, or solving daily problems). "
            "Keep speaker turns short and natural (8 to 15 words per turn), using accessible S1/S2 spoken phrasing and concrete dialogue markers."
        )
        ctx.kwargs["question_stem_guidance"] = (
            f"Use direct, unambiguous question stems referring to {s1_name} and {s2_name} by name "
            f"(e.g. 'What does {s1_name} want to do?', 'Why is {s2_name} concerned?'). "
            "Avoid complex speculative or hypothetical framing."
        )
        ctx.kwargs["option_complexity_guidance"] = (
            "strictly 12 words or fewer (ideally 3 to 8 words), using only basic everyday A1-A2 vocabulary "
            "such as 'work hours', 'schedule', 'sign in', 'next', 'cold', 'rest', 'water', 'doctor'"
        )
        ctx.kwargs["skill_distribution_guidance"] = (
            "     * `Detail`: concrete facts, explicitly stated intentions, or direct causes.\n"
            "     * `Main Idea`: overall topic or mutual agreement.\n"
            "     * `Inference`: a straightforward, obvious conclusion."
        )
    elif cefr in ("B1", "B1+"):
        ctx.kwargs["dialogue_style_guidance"] = (
            "Everyday study-group, campus, or practical colleague conversation using natural spoken English and contractions (`I'm`, `we've`, `that's`). "
            "Keep turns conversational and concise (10 to 16 words per turn), expressing clear opinions and practical suggestions without stiff formal jargon."
        )
        ctx.kwargs["question_stem_guidance"] = (
            "Use clear question stems assessing explicit facts, speaker reasoning, and immediate deductions."
        )
        ctx.kwargs["option_complexity_guidance"] = (
            "ideally 5 to 10 words, with natural everyday vocabulary"
        )
        ctx.kwargs["skill_distribution_guidance"] = (
            "     * `Detail`: specific rationale or facts stated by a speaker.\n"
            "     * `Main Idea`: core theme or opening purpose.\n"
            "     * `Inference`: speaker attitude or logical takeaway."
        )
    elif cefr == "B2":
        ctx.kwargs["dialogue_style_guidance"] = (
            "Natural, idiomatic spoken discussion between peers, student and advisor, or colleagues (e.g. IELTS Section 3 / university project review). "
            "Use high-frequency spoken English (S1/S2 vocabulary and natural contractions like `we've`, `that's`, `I'd`) at 15 to 22 words per turn (NEVER exceeding 28 words). "
            "Anchor the conversation to a tangible project artifact (e.g. presentation slide, survey sample size, lab booking, or report deadline). "
            "Difficulty at B2 comes from tracking pragmatic stance shifts, polite hedges, and conversational updates—NEVER from dense academic jargon. "
            "DO NOT use essay connectors like 'furthermore', 'moreover', or 'in conclusion'; use spoken signposts like 'to be fair', 'the thing is', or 'on the flip side'."
        )
        ctx.kwargs["question_stem_guidance"] = (
            "Use clear, natural question stems testing speaker intention, stance shifts, and agreed next steps."
        )
        ctx.kwargs["option_complexity_guidance"] = (
            "ideally 6 to 12 words, paraphrasing spoken meaning without overly formal or bookish vocabulary"
        )
        ctx.kwargs["skill_distribution_guidance"] = (
            "     * `Detail`: specific rationale or constraints stated by a speaker.\n"
            "     * `Main Idea`: opening purpose or overall topic.\n"
            "     * `Inference`: speaker attitude, hedged disagreement, or final outcome."
        )
    else:
        ctx.kwargs["dialogue_style_guidance"] = (
            "University seminar, tutorial consultation, or project review (e.g. IELTS Listening Section 3 / TOEFL academic discussion where two students, colleagues, or a tutor and student evaluate project trade-offs or methodology). "
            "Turns MUST be genuinely spoken at 18 to 25 words per turn (NEVER exceeding 30 words), anchored to a concrete project artifact (e.g. slide diagram, questionnaire numbers, or deadline constraint). "
            "When theoretical terms appear, unpack them with an everyday spoken analogy. "
            "NEVER use written essay connectors like 'furthermore', 'moreover', or 'in conclusion'; use natural spoken signposts like 'the bottom line is', 'to be fair', or 'having said that'."
        )
        ctx.kwargs["question_stem_guidance"] = (
            "Use rigorous assessment stems testing speaker stance, implicit qualifications, and analytical deductions."
        )
        ctx.kwargs["option_complexity_guidance"] = (
            "8 to 16 words, with nuanced and precise paraphrasing"
        )
        ctx.kwargs["skill_distribution_guidance"] = (
            "     * `Detail`: key nuances, qualifying conditions, or exact rationales.\n"
            "     * `Main Idea`: overarching academic purpose, consensus, or disagreement.\n"
            "     * `Inference`: deductions warranted by tone, stance, or logic."
        )

    summary_txt = data.get("summary_content", "")
    raw_src_txt = data.get("raw_source_content") or data.get("source_content") or ""
    core_unit = data.get("unit_core_name", core_name)

    thematic_concepts = []
    u_t = ""
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
    ctx.kwargs["thematic_topic"] = thematic_topic_str

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
    except (TypeError, ValueError, KeyError, RuntimeError) as e:
        logger.warning(f"Failed to build listening micro-task blueprints: {e}")
        listening_bp = None

    if isinstance(listening_bp, dict) and listening_bp.get("combined_prompt_block"):
        ctx.kwargs["vocabulary_content"] = listening_bp["combined_prompt_block"]
        ctx.blueprint_specs = listening_bp.get("questions") or []
        ctx.has_prescribed_options = True


def _build_video_context(
    data: Dict[str, Any],
    ctx: QuizPromptContext,
) -> None:
    ctx.kwargs["transcript_content"] = data["transcript"]
    ctx.kwargs["video_url"] = data["video_url"]
    ctx.kwargs["video_type"] = data["video_type"]
    cefr = (data.get("cefr_level") or "B2").upper()
    ctx.kwargs["cefr_level"] = cefr

    if cefr in ("A1", "A2"):
        ctx.kwargs["question_depth_guidance"] = (
            "Focus on direct, concrete observations and simple facts explicitly described in the video at the timestamp "
            "(e.g. 'What is shown at [MM:SS]?', 'What problem does the speaker describe?'). Avoid complex synthesis across long segments."
        )
        ctx.kwargs["option_complexity_guidance"] = (
            "Keep options short and straightforward (ideally 4 to 10 words). Avoid dense academic terminology in options."
        )
    elif cefr == "B1":
        ctx.kwargs["question_depth_guidance"] = (
            "Focus on clear causal relationships, operational mechanisms, and primary challenges discussed at the timestamp "
            "(e.g. 'Why was this method chosen?', 'How does the process function?')."
        )
        ctx.kwargs["option_complexity_guidance"] = (
            "Keep options clear and standard (ideally 6 to 14 words), using accessible technical vocabulary."
        )
    else:
        ctx.kwargs["question_depth_guidance"] = (
            "Focus on technical mechanisms, comparative evaluations, underlying controversies, or future projections discussed at the timestamp."
        )
        ctx.kwargs["option_complexity_guidance"] = (
            "Craft intellectually mature options (8 to 18 words) with nuanced distractors and academic phrasing."
        )


def build_quiz_prompt_context(
    processor: Any,
    template_name: str,
    core_name: str,
    data: Dict[str, Any],
    count: int,
    listening_archetype: Optional[str] = None,
    listening_scenario: Optional[str] = None,
) -> QuizPromptContext:
    """Builds prompt kwargs, blueprint specs, and constraint contexts for a given quiz modality."""
    ctx = QuizPromptContext(kwargs={"count": count})

    if template_name == "vocabulary":
        _build_vocabulary_context(processor, data, count, ctx)
    elif template_name == "reading":
        _build_reading_context(data, count, ctx)
    elif template_name == "translation":
        _build_translation_context(processor, data, count, ctx)
    elif template_name == "listening":
        _build_listening_context(
            processor,
            core_name,
            data,
            count,
            ctx,
            listening_archetype=listening_archetype,
            listening_scenario=listening_scenario,
        )
    elif template_name == "video":
        _build_video_context(data, ctx)

    return ctx
