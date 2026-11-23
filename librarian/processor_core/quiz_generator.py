"""Multi-modal quiz generation orchestrator (generate_quiz).

Composes:
- QuizShufflerMixin: Fisher-Yates option shuffling and explanation label remapping.
- QuizHydratorMixin: Blueprint field hydration and surface frame rendering.
- QuizSanitizerMixin: Syllabus and extraction sanitization.
- QuizRendererMixin: HTML template rendering with asset inlining.
- Sub-services:
  - QuizPromptContext & build_quiz_prompt_context: Prompt kwargs & blueprint preparation.
  - run_expert_audit_and_cure: Level-1 deterministic code gate & Level-2 surgical cure.
  - synthesize_quiz_audio: TTS dialogue script audio synthesis & base64 packaging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..linguistics import LinguisticEngine
from ..llm import llm
from ..prompts import Prompts
from .quiz_audio import synthesize_quiz_audio
from .quiz_expert_audit import run_expert_audit_and_cure
from .quiz_hydrators import QuizHydratorMixin
from .quiz_prompt_context import QuizPromptContext, build_quiz_prompt_context
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

    def generate_quiz(
        self,
        unit_name: str,
        count: int = 10,
        template_name: str = "vocabulary",
        audit_callback: Optional[Any] = None,
        listening_archetype: Optional[str] = None,
        listening_scenario: Optional[str] = None,
    ) -> str:
        """Generates a quiz handout based on extracted wiki data.

        Optional audit_callback(stage, progress, message, extra=None) for live dashboard progress.
        """
        if isinstance(count, str) and count.strip().lower() in (
            "vocabulary", "cloze", "reading", "translation", "listening", "video"
        ):
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

        if core_name.endswith(".md"):
            core_name = core_name[:-3]
        elif core_name.endswith(".txt"):
            core_name = core_name[:-4]

        for suffix in ["_vocabulary", "_grammar", "_concepts"]:
            if core_name.endswith(suffix):
                core_name = core_name[:-len(suffix)]

        # Pre-Flight LLM Availability Gate (Skip for 100% pure symbolic cloze):
        if template_name != "cloze":
            try:
                llm.ensure_available(
                    task_name=f"{core_name}_quiz_{template_name}",
                    prompt_summary=f"Quiz generation ({template_name}_quiz) pre-flight check for unit '{core_name}'.",
                )
            except (ConnectionError, RuntimeError, ValueError) as pre_err:
                logger.error(f"Pre-flight LLM check failed for {core_name}_quiz_{template_name}: {pre_err}")
                return f"Error: {pre_err}"

        # 2. Resolve Data
        try:
            data = self._load_wiki_data(core_name, template_name)
        except (OSError, ValueError, KeyError) as load_err:
            logger.error(f"Failed loading/extracting wiki data for {core_name} ({template_name}): {load_err}", exc_info=True)
            return f"Error: Failed to prepare source data for {core_name} ({template_name}): {load_err}"
        if not data:
            return f"Error: Missing prerequisite extraction for unit '{core_name}'. Please compile and extract knowledge nodes (extractions/{core_name}_vocabulary.md) before generating quizzes."

        # 3. Build Prompt Context & Blueprints
        prompt_ctx: QuizPromptContext = build_quiz_prompt_context(
            self,
            template_name,
            core_name,
            data,
            count,
            listening_archetype=listening_archetype,
            listening_scenario=listening_scenario,
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
                quiz_obj: Any = {
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
                prompt_template, schema_cls = Prompts.get(f"{template_name}_quiz")
                prompt = prompt_template.format(**prompt_ctx.kwargs)
                interpolated_schema = self._interpolate_schema(schema_cls, prompt_ctx.kwargs)
                quiz_obj = llm.chat(
                    [{"role": "user", "content": prompt}],
                    schema=interpolated_schema,
                    task_name=f"{core_name}_quiz_{template_name}",
                )

                if not quiz_obj:
                    return "Error: LLM returned empty quiz data."

            # 4.4. Deterministic Blueprint Hydration
            if prompt_ctx.blueprint_specs:
                if template_name == "translation":
                    quiz_obj = self.hydrate_translation_blueprint_items(quiz_obj, prompt_ctx.blueprint_specs)
                elif template_name == "listening":
                    quiz_obj = self.hydrate_listening_blueprint_items(
                        quiz_obj,
                        prompt_ctx.blueprint_specs,
                        speaker_1_name=prompt_ctx.kwargs.get("speaker_1_name", ""),
                        speaker_2_name=prompt_ctx.kwargs.get("speaker_2_name", ""),
                    )
                elif template_name == "reading":
                    quiz_obj = self.hydrate_reading_blueprint_items(quiz_obj, prompt_ctx.blueprint_specs)
                elif template_name == "vocabulary":
                    quiz_obj = self.hydrate_vocabulary_blueprint_items(quiz_obj, prompt_ctx.blueprint_specs)
                else:
                    quiz_obj = self.hydrate_vocabulary_blueprint_items(quiz_obj, prompt_ctx.blueprint_specs)
            elif template_name == "video":
                quiz_obj = self.hydrate_video_blueprint_items(quiz_obj)

            # 4.5. Randomize options (skip when blueprint prescribes CRC32-balanced options)
            quiz_obj = self._shuffle_quiz_options(quiz_obj, preserve_blueprint_order=prompt_ctx.has_prescribed_options)

            # Deterministically attach physical CEFR level to quiz_obj
            cefr_val = str(data.get("cefr_level") or prompt_ctx.kwargs.get("cefr_level") or "B2").upper()
            if isinstance(quiz_obj, dict):
                quiz_obj["cefr_level"] = cefr_val
            else:
                try:
                    setattr(quiz_obj, "cefr_level", cefr_val)
                except AttributeError:
                    pass

            # Modality context attachments
            if template_name == "reading":
                reading_vocab: List[Any] = []
                try:
                    reading_vocab = LinguisticEngine.extract_reading_vocabulary(
                        data.get("passage", ""),
                        target_count=8,
                        enable_llm_disambiguation=False,
                    )
                except (ValueError, TypeError, RuntimeError) as e:
                    logger.warning(f"Reading vocabulary extraction skipped: {e}")
                if isinstance(quiz_obj, dict):
                    quiz_obj["passage"] = data["passage"]
                    quiz_obj["vocabulary"] = reading_vocab
                else:
                    quiz_obj.passage = data["passage"]
                    quiz_obj.vocabulary = reading_vocab

            if template_name == "video":
                quiz_obj = self.hydrate_video_blueprint_items(quiz_obj)
                video_vocab: List[Any] = []
                try:
                    video_vocab = LinguisticEngine.extract_video_vocabulary(
                        data.get("transcript", ""),
                        target_count=12,
                        enable_llm_disambiguation=True,
                    )
                except (ValueError, TypeError, RuntimeError) as e:
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
            quiz_obj = run_expert_audit_and_cure(
                self,
                quiz_obj,
                template_name,
                core_name,
                data,
                prompt_ctx,
                audit_callback=audit_callback,
            )

            # 5. TTS for Listening Quiz (Base64 Embedding)
            audio_url = None
            if template_name == "listening":
                audio_url = synthesize_quiz_audio(
                    quiz_obj,
                    default_s1=prompt_ctx.kwargs.get("speaker_1_name", ""),
                    default_s2=prompt_ctx.kwargs.get("speaker_2_name", ""),
                    default_g1=prompt_ctx.kwargs.get("speaker_1_gender", "female"),
                    default_g2=prompt_ctx.kwargs.get("speaker_2_gender", "male"),
                )

            # 6. Render and Save
            language = self.config.get("target_language") or "Chinese"
            html_content = self._render_handout(quiz_obj, template_name, audio_url, language=language)
            asset_stem = data.get("source_stem", core_name) if isinstance(data, dict) else core_name
            output_type = template_name
            handout_filename = f"{asset_stem}_{output_type}_quiz.html"
            handout_dir = self.config.wiki_content_path / core_name / "handouts"
            handout_dir.mkdir(parents=True, exist_ok=True)
            handout_path = handout_dir / handout_filename

            quiz_audit = None
            if hasattr(quiz_obj, "_expert_audit") and getattr(quiz_obj, "_expert_audit"):
                quiz_audit = getattr(quiz_obj, "_expert_audit")
            elif hasattr(quiz_obj, "audit") and getattr(quiz_obj, "audit"):
                quiz_audit = getattr(quiz_obj, "audit")
            elif isinstance(quiz_obj, dict):
                quiz_audit = quiz_obj.get("_expert_audit") or quiz_obj.get("audit")

            final_questions = getattr(quiz_obj, "questions", []) or (
                quiz_obj.get("questions", []) if isinstance(quiz_obj, dict) else []
            )
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

        except (OSError, ValueError, TypeError, RuntimeError) as e:
            return f"Quiz Generation Error: {e}"
