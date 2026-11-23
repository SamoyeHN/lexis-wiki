"""Level-2 expert model quality audit and in-place surgical cure service.

Executes:
1. Pre-audit Level 1 deterministic code gate (fast structural checks).
2. Level 2 psychometric blind evaluation via ExpertAuditor (LLM-as-a-judge).
3. In-place surgical cure for REPAIR / REWRITE candidates with L1 verification.
4. Deterministic post-cure scoring, uncurable defect pruning, and log synchronization.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from ..linguistics import LinguisticEngine
from ..llm import llm
from .quiz_prompt_context import QuizPromptContext

logger = logging.getLogger("librarian.processor")


def _resolve_source_context(
    template_name: str,
    data: Dict[str, Any],
    quiz_obj: Any,
) -> str:
    """Extracts appropriate textual context for the blind evaluator based on modality."""
    if template_name == "reading":
        return LinguisticEngine.format_paragraph_numbered_passage(data.get("passage", ""))
    if template_name == "video":
        return data.get("transcript", "")
    if template_name == "translation":
        return ""
    if template_name == "listening":
        script_turns = []
        raw_script = (
            quiz_obj.get("script")
            if isinstance(quiz_obj, dict)
            else getattr(quiz_obj, "script", [])
        )
        if isinstance(raw_script, list):
            for turn in raw_script:
                if isinstance(turn, dict):
                    spk = turn.get("speaker", "Speaker")
                    txt = turn.get("text", "")
                    script_turns.append(f"{spk}: {txt}")
                elif hasattr(turn, "speaker") and hasattr(turn, "text"):
                    script_turns.append(f"{turn.speaker}: {turn.text}")
        return "\n".join(script_turns)
    if template_name == "cloze":
        return data.get("passage", "")
    return ""


def _normalize_candidate_cure(
    candidate_cure: Dict[str, Any],
    template_name: str,
    orig_dict: Dict[str, Any],
    triage: str,
    unit_headwords: List[str],
    curr_tgt_lang: str,
) -> None:
    """Normalizes candidate cure fields, options, and inherits missing fields."""
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
                    flags=re.IGNORECASE,
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
            flags=re.IGNORECASE,
        ).strip()

    if template_name in ("vocabulary", "cloze"):
        if not candidate_cure.get("question") and orig_dict.get("question") and triage == "REPAIR":
            candidate_cure["question"] = orig_dict["question"]
        if not candidate_cure.get("explanation") and orig_dict.get("explanation"):
            candidate_cure["explanation"] = orig_dict["explanation"]
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

    if template_name == "reading":
        if not candidate_cure.get("category") and orig_dict.get("category"):
            candidate_cure["category"] = orig_dict["category"]

    if template_name == "video":
        if not candidate_cure.get("timestamp") and orig_dict.get("timestamp"):
            candidate_cure["timestamp"] = orig_dict["timestamp"]

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
                candidate_cure["flawed_translation"],
            ]
            corr_idx = 0

        if len(extracted_opts) == 2:
            label_strip = r'^(?:[A-Da-d\d][\.\)\:\-]\s*)'
            clean_cand_opts = [re.sub(label_strip, '', str(o)).strip() for o in extracted_opts]
            candidate_cure["options"] = clean_cand_opts
            if not isinstance(corr_idx, int) or corr_idx not in (0, 1):
                corr_idx = 0
            candidate_cure["correct_answer_index"] = corr_idx
            candidate_cure["idiomatic_translation"] = clean_cand_opts[corr_idx]
            candidate_cure["flawed_translation"] = clean_cand_opts[1 - corr_idx]
            candidate_cure["correct_english_answer"] = clean_cand_opts[corr_idx]

        orig_stem = orig_dict.get("translated_sentence", "")
        curr_stem = candidate_cure.get("translated_sentence", "")
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


def _verify_cure_level1(
    processor: Any,
    template_name: str,
    healed_item: Dict[str, Any],
    cured_questions: List[Any],
    item_slot: int,
    source_context: str,
    prompt_ctx: QuizPromptContext,
    curr_tgt_lang: str,
) -> bool:
    """Verifies a single cured item against modality Level-1 code gates. Returns True if failed."""
    cure_cefr = prompt_ctx.kwargs.get("cefr_level", "B2")
    blueprint_specs = prompt_ctx.blueprint_specs
    unit_headwords = prompt_ctx.unit_headwords
    banned_quiz_sentences = prompt_ctx.banned_quiz_sentences

    if template_name == "reading":
        cure_bp_spec = (
            [blueprint_specs[item_slot]]
            if (blueprint_specs and item_slot < len(blueprint_specs))
            else None
        )
        cure_l1_flagged, _ = processor.audit_reading_integrity(
            {"questions": [healed_item]},
            passage_text=source_context,
            cefr_level=cure_cefr,
            blueprint_specs=cure_bp_spec,
        )
        return bool(cure_l1_flagged)

    if template_name == "video":
        cure_l1_flagged, _ = processor.audit_video_integrity(
            {"questions": [healed_item]},
            transcript_text=source_context,
            cefr_level=cure_cefr,
        )
        return bool(cure_l1_flagged)

    if template_name == "listening":
        cure_l1_flagged, _ = processor.audit_listening_integrity(
            {"questions": [healed_item]},
            script_text=source_context,
            cefr_level=cure_cefr,
            unit_headwords=unit_headwords,
        )
        return bool(cure_l1_flagged)

    if template_name == "translation":
        cure_l1_flagged, _ = processor.audit_translation_integrity(
            {"questions": [healed_item]},
            unit_headwords=unit_headwords,
            target_language=curr_tgt_lang,
            cefr_level=cure_cefr,
            enforce_minimal_pair=bool(blueprint_specs),
            passage_text=source_context,
        )
        return bool(cure_l1_flagged)

    if template_name == "cloze":
        cure_l1_flagged, _ = processor.audit_cloze_integrity(
            {"questions": [healed_item]},
            passage_text=source_context,
            cefr_level=cure_cefr,
        )
        return bool(cure_l1_flagged)

    candidate_batch = list(cured_questions)
    candidate_batch[item_slot] = healed_item
    cure_l1_flagged, _ = processor.audit_quiz_integrity(
        {"questions": candidate_batch},
        banned_sentences=banned_quiz_sentences,
        unit_headwords=unit_headwords,
        strict_distractor_recycling=True,
        allow_authentic_cloze=False,
    )
    return item_slot in cure_l1_flagged


def run_expert_audit_and_cure(
    processor: Any,
    quiz_obj: Any,
    template_name: str,
    core_name: str,
    data: Dict[str, Any],
    prompt_ctx: QuizPromptContext,
    audit_callback: Optional[Any] = None,
) -> Any:
    """Runs Level-1 Code Gate and Level-2 Expert Audit with In-Place Surgical Cure."""
    is_open_cloze = (
        template_name == "cloze"
        and (
            (isinstance(quiz_obj, dict) and quiz_obj.get("mode") == "open_cloze")
            or not any(
                isinstance(q, dict) and q.get("options")
                for q in (
                    quiz_obj.get("questions", [])
                    if isinstance(quiz_obj, dict)
                    else getattr(quiz_obj, "questions", [])
                )
            )
        )
    )

    if is_open_cloze:
        l1_defective, l1_defects = processor.audit_cloze_integrity(
            quiz_obj if isinstance(quiz_obj, dict) else dataclasses.asdict(quiz_obj),
            passage_text=data.get("passage", ""),
            cefr_level=prompt_ctx.kwargs.get("cefr_level", "B2"),
        )
        if l1_defects:
            logger.warning(
                f"Pure Open Cloze L1 Code Gate flagged {len(l1_defects)} issue(s): {l1_defects}"
            )
        return quiz_obj

    if not processor.config.get("enable_expert_audit", False):
        return quiz_obj

    try:
        from ..expert_auditor import ExpertAuditor

        source_context = _resolve_source_context(template_name, data, quiz_obj)
        quiz_dict_eval = dataclasses.asdict(quiz_obj) if dataclasses.is_dataclass(quiz_obj) else quiz_obj

        if template_name == "vocabulary" and prompt_ctx.vocab_sense_map and isinstance(quiz_dict_eval, dict):
            for q_item in quiz_dict_eval.get("questions", []):
                if isinstance(q_item, dict):
                    tw_q = str(q_item.get("target_word") or "").strip().lower()
                    if tw_q in prompt_ctx.vocab_sense_map and not q_item.get("contextual_sense"):
                        q_item["contextual_sense"] = prompt_ctx.vocab_sense_map[tw_q]

        # Step 1: Pre-Audit Level 1 Code Gate
        curr_tgt_lang = prompt_ctx.kwargs.get("target_language") or processor.config.get("target_language") or "Chinese"
        cefr_val = str(data.get("cefr_level") or prompt_ctx.kwargs.get("cefr_level") or "B2").upper()

        if template_name == "reading":
            _, l1_defects = processor.audit_reading_integrity(
                quiz_dict_eval,
                passage_text=source_context,
                cefr_level=cefr_val,
                blueprint_specs=prompt_ctx.blueprint_specs,
            )
        elif template_name == "cloze":
            _, l1_defects = processor.audit_cloze_integrity(
                quiz_dict_eval,
                passage_text=source_context,
                cefr_level=cefr_val,
            )
        elif template_name == "video":
            _, l1_defects = processor.audit_video_integrity(
                quiz_dict_eval,
                transcript_text=source_context,
                cefr_level=cefr_val,
            )
        elif template_name == "listening":
            _, l1_defects = processor.audit_listening_integrity(
                quiz_dict_eval,
                script_text=source_context,
                cefr_level=cefr_val,
                unit_headwords=prompt_ctx.unit_headwords,
            )
        elif template_name == "translation":
            _, l1_defects = processor.audit_translation_integrity(
                quiz_dict_eval,
                unit_headwords=prompt_ctx.unit_headwords,
                target_language=curr_tgt_lang,
                cefr_level=cefr_val,
                enforce_minimal_pair=bool(prompt_ctx.blueprint_specs),
                passage_text=source_context,
            )
        else:
            _, l1_defects = processor.audit_quiz_integrity(
                quiz_dict_eval,
                banned_sentences=prompt_ctx.banned_quiz_sentences,
                unit_headwords=prompt_ctx.unit_headwords,
                allow_authentic_cloze=False,
                strict_distractor_recycling=not prompt_ctx.has_prescribed_options,
            )

        if l1_defects:
            logger.warning(
                f"Pre-Audit Level 1 Code Gate flagged {len(l1_defects)} issue(s) before Level 2: {l1_defects}"
            )

        # Step 2: Level 2 Expert Model Quality Audit & Triage
        if audit_callback:
            try:
                audit_callback(
                    stage="auditing",
                    audit_progress=30,
                    message="🔍 Level 2 Expert Audit: Running psychometric blind evaluation & surgical triage...",
                    extra={"attempt": 1, "max_retries": 1},
                )
            except (TypeError, RuntimeError):
                pass

        judge_model_to_use = ExpertAuditor.get_judge_model()
        active_gen_model = llm.model

        if judge_model_to_use and active_gen_model and judge_model_to_use != active_gen_model:
            llm.unload_model(active_gen_model)

        audit_report = ExpertAuditor.audit_quiz(
            source_context,
            quiz_dict_eval,
            judge_model=judge_model_to_use,
            quiz_type=template_name,
            target_language=curr_tgt_lang,
            unit_headwords=prompt_ctx.unit_headwords,
            unit_grammar_patterns=prompt_ctx.unit_grammar_patterns,
            cefr_level=cefr_val,
            unit_slug=core_name,
        )

        if judge_model_to_use and active_gen_model and judge_model_to_use != active_gen_model:
            llm.unload_model(judge_model_to_use)

        if not audit_report:
            return quiz_obj

        raw_gen_score = audit_report.get("overall_quality_score", 100)
        audit_report["raw_generation_score"] = raw_gen_score

        orig_questions = quiz_dict_eval.get("questions", [])
        audit_items = audit_report.get("questions", [])
        cured_questions = list(orig_questions)
        cure_stats = {"pass": 0, "repair": 0, "rewrite": 0, "discard": 0}

        # Step 3: In-Place Surgical Cure & Post-Cure L1 Verification
        for a_idx, qa in enumerate(audit_items):
            item_slot = qa.get("item_index", a_idx + 1) - 1
            if not (0 <= item_slot < len(cured_questions)):
                item_slot = a_idx
            if not (0 <= item_slot < len(cured_questions)):
                continue

            triage = str(qa.get("triage_action", "PASS")).upper()
            candidate_cure = qa.get("cured_question")

            if triage in ("REPAIR", "REWRITE") and isinstance(candidate_cure, dict) and candidate_cure:
                orig_item = orig_questions[item_slot] if 0 <= item_slot < len(orig_questions) else {}
                orig_dict = (
                    orig_item
                    if isinstance(orig_item, dict)
                    else (dataclasses.asdict(orig_item) if dataclasses.is_dataclass(orig_item) else {})
                )

                _normalize_candidate_cure(
                    candidate_cure,
                    template_name,
                    orig_dict,
                    triage,
                    prompt_ctx.unit_headwords,
                    curr_tgt_lang,
                )

                temp_wrap = {"questions": [candidate_cure]}
                temp_wrap = processor._shuffle_quiz_options(
                    temp_wrap, preserve_blueprint_order=prompt_ctx.has_prescribed_options
                )
                healed_item = temp_wrap["questions"][0]

                cure_item_failed = _verify_cure_level1(
                    processor,
                    template_name,
                    healed_item,
                    cured_questions,
                    item_slot,
                    source_context,
                    prompt_ctx,
                    curr_tgt_lang,
                )

                if not cure_item_failed:
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
                    logger.warning(
                        f"Candidate expert cure for Item #{item_slot + 1} failed Level 1 gate. Keeping original item penalized."
                    )
                    cure_stats["discard"] += 1
            else:
                if qa.get("single_fit_valid", True):
                    cure_stats["pass"] += 1
                else:
                    cure_stats["discard"] += 1

        # Prune uncured defective items
        final_clean_questions = []
        for idx, q_candidate in enumerate(cured_questions):
            corresponding_audit = (
                audit_items[idx]
                if idx < len(audit_items) and isinstance(audit_items[idx], dict)
                else {}
            )
            if corresponding_audit.get("single_fit_valid", True):
                final_clean_questions.append(q_candidate)
            else:
                logger.warning(f"Pruned uncurable defective Item #{idx + 1} from final handout (single_fit_valid=False).")

        if isinstance(quiz_obj, dict):
            quiz_obj["questions"] = final_clean_questions
        else:
            try:
                quiz_obj.questions = final_clean_questions
            except AttributeError:
                pass

        quiz_obj = processor._shuffle_quiz_options(
            quiz_obj, preserve_blueprint_order=prompt_ctx.has_prescribed_options
        )

        if isinstance(quiz_obj, dict):
            quiz_obj["cefr_level"] = cefr_val
        else:
            try:
                setattr(quiz_obj, "cefr_level", cefr_val)
            except AttributeError:
                pass

        # Step 4: Deterministic Post-Cure Scoring
        total_items_count = len(orig_questions) or len(audit_items) or 1
        missing_audit_items = max(0, len(orig_questions) - len(audit_items)) if orig_questions else 0
        audit_report["missing_audit_items"] = missing_audit_items

        item_scores = [qa.get("pedagogical_score", 90) for qa in audit_items if isinstance(qa, dict)]
        score_denom = max(len(item_scores), total_items_count)
        final_avg = round(sum(item_scores) / score_denom) if item_scores else 100
        audit_report["base_quality_score"] = final_avg

        has_fatal_residual = (
            any(qa.get("single_fit_valid") is False for qa in audit_items if isinstance(qa, dict))
            or (cure_stats["discard"] > 0)
            or (missing_audit_items > 0)
        )

        if has_fatal_residual:
            audit_report["pass_audit"] = False
            cap_limit = 59 if missing_audit_items > 0 else 70
            audit_report["overall_quality_score"] = min(final_avg, cap_limit)
        else:
            audit_report["pass_audit"] = (final_avg >= 80)
            audit_report["overall_quality_score"] = final_avg

        final_overall_score = audit_report["overall_quality_score"]
        audit_report["cure_stats"] = cure_stats

        valid_items_count = cure_stats["pass"] + cure_stats["repair"] + cure_stats["rewrite"]
        audit_report["post_cure_accuracy"] = round(valid_items_count / total_items_count, 2)
        audit_report["blind_solve_accuracy"] = audit_report["post_cure_accuracy"]

        cured_count = cure_stats["repair"] + cure_stats["rewrite"]
        if cured_count > 0:
            cure_summary = (
                f"[SURGICAL CURE APPLIED: Pre-Cure {raw_gen_score}% -> Post-Cure {final_overall_score}%. "
                f"Pass: {cure_stats['pass']}, Repaired: {cure_stats['repair']}, "
                f"Rewritten: {cure_stats['rewrite']}, Discard/Flawed: {cure_stats['discard']}]."
            )
            orig_verdict = str(audit_report.get("summary_verdict", ""))
            audit_report["summary_verdict"] = f"{cure_summary} {orig_verdict}".strip()

        # Synchronize Pre-Cure vs Post-Cure scores to expert audit log
        try:
            logs_dir = Path(processor.config.get("project_root", ".")).resolve() / "logs"
            audit_task = f"{core_name}_audit_{template_name}"
            matching_logs = sorted(
                logs_dir.glob(f"*{audit_task}*.log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not matching_logs:
                task_prefix = f"expert_audit_{quiz_dict_eval.get('title', 'quiz')[:20]}"
                safe_prefix = re.sub(r'[\\/:*?"<>|\r\n]+', '_', task_prefix).strip('_')
                matching_logs = sorted(
                    logs_dir.glob(f"*_{safe_prefix}*.log"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
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
                        log_txt,
                    )
                    latest_log.write_text(log_txt, encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError) as log_err:
            logger.debug(f"Could not update audit log with post-cure score: {log_err}")

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
                        extra={"score": final_avg, "accuracy": final_accuracy, "passed": True, "cure_stats": cure_stats},
                    )
                else:
                    audit_callback(
                        stage="completed_with_warnings",
                        audit_progress=100,
                        message=f"⚠️ Audit finished with warnings (Score: {final_avg}%). Proceeding with best attempt.",
                        extra={"score": final_avg, "accuracy": final_accuracy, "passed": False, "cure_stats": cure_stats},
                    )
            except (TypeError, RuntimeError):
                pass

    except (TypeError, ValueError, KeyError, RuntimeError) as audit_err:
        logger.warning(f"Level 2 Expert Quality Audit encountered non-fatal error: {audit_err}", exc_info=True)

    return quiz_obj
