import re
import json
import logging
import dataclasses
from typing import Dict, List, Any, Optional, Union

from .config import config
from .schemas import QuizQualityAuditReport, QuestionAuditItem, DistractorAuditItem
from .llm import LLMClient
from .prompts import Prompts

logger = logging.getLogger("librarian.expert_auditor")


class ExpertAuditor:
    """Level 2 Expert Model Quality Audit (LLM-as-a-Judge).
    
    Conducts high-reasoning semantic evaluation focusing on:
    - Blind Solver Test (independent solving without knowing declared answer)
    - Absolute Single-Fit Validity (verifying exactly one defensible answer)
    - Cognitive Distractor Trap Quality (ensuring distractors represent authentic educational traps)
    """

    @classmethod
    def get_judge_model(cls) -> str:
        """Determines the model to use for judging.
        
        Prefers 'judge_model' from config, falls back to current active model.
        """
        judge_model = config.get("judge_model")
        if judge_model and str(judge_model).strip():
            return str(judge_model).strip()
        return config.get("model", "gemma4:12b")

    @classmethod
    def format_quiz_for_blind_audit(cls, source_text: str, questions: List[Dict[str, Any]], is_translation: bool = False) -> str:
        """Formats quiz items for expert audit.
        
        For standard reading/vocab/listening quizzes, correct answers are stripped out
        so the expert judge must solve each item blindly.
        For translation quizzes, translation pedagogy requires evaluating the declared target
        translation directly; thus, the declared answer is explicitly presented.
        """
        lines = []
        if source_text and source_text.strip():
            lines.extend([
                "### SOURCE MATERIAL:",
                source_text.strip(),
                ""
            ])

        if is_translation:
            lines.append("### TRANSLATION QUIZ ITEMS FOR PEDAGOGICAL AUDIT:")
        else:
            lines.append("### QUIZ ITEMS FOR EVALUATION (BLIND SOLVER TEST):")
        
        for idx, q in enumerate(questions):
            raw_text = q.get("question") or q.get("translated_sentence") or f"Question {idx + 1}"
            options = q.get("options", [])
            declared_idx = q.get("correct_answer_index")
            
            # 1. Clean stem: strip markdown emphasis (**bold**, `code`) and accidental answer leaks, while strictly preserving '____' blanks
            clean_stem = str(raw_text)
            clean_stem = re.sub(r"(?i)\s*[\(\[\{]?(?:correct\s*)?(?:answer|key|target)[\:\s\-]+[a-d\w\s]+[\)\]\}]?", "", clean_stem)
            clean_stem = re.sub(r"[*`]", "", clean_stem)
            # Strip standalone single underscore markdown italics (e.g. _word_) without touching multi-underscore blanks (____)
            clean_stem = re.sub(r"(?<!_)_(?!_)", "", clean_stem).strip()

            lines.append(f"Item #{idx + 1}:")
            category = q.get("category", "")
            if category:
                lines.append(f"Skill Category: {category}")
            tgt_word = q.get("target_word", "")
            if tgt_word and not is_translation:
                lines.append(f"Target Word to Assess: {tgt_word}")
            timestamp = q.get("timestamp", "")
            if timestamp:
                lines.append(f"Timestamp Segment: {timestamp}")
            lines.append(f"Stem: {clean_stem}")
            
            if is_translation:
                kw = q.get("target_keyword", "")
                grammar = q.get("target_grammar", "")
                flaw_type = q.get("flaw_type", "")
                skeleton = q.get("english_skeleton", "")
                audit_info = q.get("design_audit", "")
                if kw:
                    lines.append(f"Target Keyword: {kw}")
                if grammar:
                    lines.append(f"Target Grammar: {grammar}")
                if flaw_type:
                    lines.append(f"Declared Flaw Type: {flaw_type}")
                if skeleton:
                    lines.append(f"English Skeleton: {skeleton}")
                if audit_info:
                    lines.append(f"Design Intent & Grammar Trap Blueprint: {audit_info}")
                if isinstance(declared_idx, int) and 0 <= declared_idx < len(options):
                    declared_letter = chr(65 + declared_idx)
                    lines.append(f"Declared Target Translation: Option [{declared_letter}]")

            lines.append("Options:")
            for opt_idx, opt in enumerate(options):
                clean_opt = str(opt)
                # Strip accidental annotations like "(Correct Answer)" or "[Key]"
                clean_opt = re.sub(r"(?i)\s*[\(\[\{]?(?:correct\s*)?(?:key|answer)[\)\]\}]?", "", clean_opt)
                # Strip all markdown formatting to ensure identical plain presentation
                clean_opt = re.sub(r"[*_`]", "", clean_opt).strip()
                letter = chr(65 + opt_idx)
                if is_translation and opt_idx == declared_idx:
                    lines.append(f"  [{letter}] {clean_opt}  <-- [DECLARED TARGET TRANSLATION]")
                else:
                    lines.append(f"  [{letter}] {clean_opt}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def audit_quiz(
        cls,
        source_text: str,
        quiz_data: Dict[str, Any],
        judge_model: Optional[str] = None,
        quiz_type: Optional[str] = None,
        target_language: Optional[str] = None,
        unit_headwords: Optional[List[str]] = None,
        unit_grammar_patterns: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Executes Level 2 semantic audit on a quiz payload."""
        questions = quiz_data.get("questions", [])
        if not questions:
            logger.warning("No questions found in quiz data to evaluate.")
            return None

        # Detect quiz modality
        is_video = (quiz_type == "video") or any("timestamp" in q for q in questions if isinstance(q, dict))
        is_translation = (quiz_type == "translation") or any("translated_sentence" in q for q in questions if isinstance(q, dict))
        is_listening = (quiz_type == "listening")
        is_vocab = (quiz_type == "vocabulary") or (not is_translation and not is_video and not is_listening and any("target_word" in q for q in questions if isinstance(q, dict)))
        is_reading = (quiz_type == "reading")

        eval_model = judge_model or cls.get_judge_model()
        formatted_content = cls.format_quiz_for_blind_audit(source_text, questions, is_translation=is_translation)

        # Select specialized prompt template by modality
        if is_video:
            template_name = "expert_audit_video"
        elif is_listening:
            template_name = "expert_audit_listening"
        elif is_translation:
            template_name = "expert_audit_translation"
        elif is_vocab:
            template_name = "expert_audit_vocabulary"
        elif is_reading:
            template_name = "expert_audit_reading"
        else:
            template_name = "expert_audit"

        raw_prompt, schema_cls = Prompts.get(template_name)
        lang = target_language or "Simplified Chinese"
        
        # Build unit vocabulary list display if provided
        unit_vocab_str = ""
        if unit_headwords:
            clean_hws = [h.strip() for h in unit_headwords if h.strip()]
            unit_vocab_str = "     " + json.dumps(clean_hws)
        else:
            unit_vocab_str = "     (No explicit unit vocabulary list supplied. Rely on declared target keywords.)"

        # Build unit grammar patterns display if provided
        unit_grammar_str = ""
        if unit_grammar_patterns:
            unit_grammar_str = "\n".join(f"     {p}" for p in unit_grammar_patterns if p.strip())
        else:
            unit_grammar_str = "     (No explicit unit grammar list supplied. Rely on declared target grammar.)"

        format_kwargs = {
            "target_language": lang,
            "content": formatted_content,
            "total_items": len(questions),
            "unit_vocabulary_list": unit_vocab_str,
            "vocabulary_list": unit_vocab_str,
            "unit_grammar_list": unit_grammar_str,
            "grammar_list": unit_grammar_str
        }

        try:
            full_prompt = raw_prompt.format(**format_kwargs)
        except KeyError:
            # Fallback if template doesn't expect all keys
            full_prompt = raw_prompt
            for k, v in format_kwargs.items():
                full_prompt = full_prompt.replace(f"{{{k}}}", str(v))

        system_content = "You are an elite Psychometrician and Lead Assessment Auditor specializing in CEFR/TOEFL standardized language testing."
        user_content = full_prompt
        if "### SYSTEM ###" in full_prompt and "### USER ###" in full_prompt:
            parts = full_prompt.split("### USER ###", 1)
            system_content = parts[0].replace("### SYSTEM ###", "").strip()
            user_content = parts[1].strip()

        llm = LLMClient(model=eval_model)
        logger.info(f"Initiating Level 2 Expert Quality Audit via judge model: {eval_model}...")


        try:
            report_obj = llm.chat(
                [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                schema=schema_cls,
                task_name=f"expert_audit_{quiz_data.get('title', 'quiz')[:20]}",
                use_default_params=True,
                _disable_qa_retry=True # Avoid infinite recursive audits
            )

            if not report_obj:
                logger.error("Expert auditor returned empty report.")
                return None

            report_dict = dataclasses.asdict(report_obj) if dataclasses.is_dataclass(report_obj) else report_obj

            # ------------------------------------------------------------------
            # Post-audit blind-solve cross-check.
            #
            # The LLM judge is authoritative on pass/fail (overall_quality_score
            # + single_fit_valid). This block does NOT hard-veto on a raw
            # agreement ratio -- that would penalise good items simply because a
            # possibly-less-capable judge guessed wrong. Instead it:
            #   * aligns report items to the source by `item_index` (fallback:
            #     position) so comparisons are never misaligned;
            #   * skips items whose declared key is missing/out-of-range so we
            #     never fabricate a "match" or a "divergence";
            #   * records every divergence as evidence in `diagnostic_feedback`;
            #   * downgrades `pass_audit` ONLY on a *confident* ("Definite")
            #     divergence, the genuine double-key / key-leak signal.
            # ------------------------------------------------------------------
            audit_questions = report_dict.get("questions", []) or []
            if len(audit_questions) < len(questions):
                logger.warning(
                    f"[CARDINALITY DEFICIT] Level 2 Expert Audit returned only {len(audit_questions)}/{len(questions)} items! "
                    f"Possible LLM early truncation."
                )
            by_index = {}
            positional = []
            for qa in audit_questions:
                if not isinstance(qa, dict):
                    continue
                positional.append(qa)
                ii = qa.get("item_index")
                if isinstance(ii, int) and not isinstance(ii, bool) and ii not in by_index:
                    by_index[ii] = qa

            raw_indices = [qa.get("item_index") for qa in audit_questions if isinstance(qa.get("item_index"), int) and not isinstance(qa.get("item_index"), bool)]
            is_one_based = bool(raw_indices and (0 not in raw_indices or max(raw_indices) == len(questions)))

            def _opt_letter(idx, n):
                if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < n:
                    return chr(65 + idx)
                return str(idx)

            comparable = 0
            confirmed = 0
            confident_divergences = 0
            for idx, q in enumerate(questions):
                declared_raw = q.get("correct_answer_index")
                # Only trust a declared key that is a valid option index.
                if not isinstance(declared_raw, int) or isinstance(declared_raw, bool):
                    continue
                n_opts = len(q.get("options", []) or [])
                if n_opts > 0 and not (0 <= declared_raw < n_opts):
                    continue

                # Match audit item using detected indexing base (1-based vs 0-based), fallback to positional
                target_key = (idx + 1) if is_one_based else idx
                q_audit = by_index.get(target_key)
                if q_audit is None and idx < len(positional):
                    q_audit = positional[idx]
                if q_audit is None:
                    continue

                blind_raw = q_audit.get("blind_solved_index")
                if not isinstance(blind_raw, int) or isinstance(blind_raw, bool):
                    continue
                comparable += 1

                # Sanitize distractor trap_type assignments against declared key
                distractors = q_audit.get("distractors", [])
                if isinstance(distractors, list):
                    for d_idx, d_item in enumerate(distractors):
                        if not isinstance(d_item, dict):
                            continue
                        t_type = str(d_item.get("trap_type", ""))
                        if d_idx == declared_raw:
                            d_item["trap_type"] = "None (Correct Answer)"
                        elif "correct" in t_type.lower():
                            d_item["trap_type"] = "Plausible Real-World Distractor"

                if blind_raw == declared_raw:
                    confirmed += 1
                    continue

                # Divergence: blind solve disagrees with the declared key.
                confidence = str(q_audit.get("confidence", "")).strip()
                is_confident = confidence.lower() == "definite"
                if is_confident:
                    confident_divergences += 1

                declared_letter = _opt_letter(declared_raw, n_opts)
                blind_letter = _opt_letter(blind_raw, n_opts)
                feedback = str(q_audit.get("diagnostic_feedback", ""))
                if is_confident:
                    div_msg = f"[DIVERGENCE: Declared {declared_letter} vs Blind-Solved {blind_letter}, Confident (possible double-key)]"
                else:
                    div_msg = f"[DIVERGENCE: Declared {declared_letter} vs Blind-Solved {blind_letter}, Confidence={confidence or 'unknown'}]"
                if div_msg not in feedback:
                    q_audit["diagnostic_feedback"] = f"{div_msg} {feedback}".strip()
                logger.warning(
                    f"Item #{idx + 1} blind-solver divergence: declared [{declared_letter}] "
                    f"vs solved [{blind_letter}] (confidence={confidence or 'unknown'}); "
                    f"{'veto signal' if is_confident else 'evidence only'}."
                )

            # Accuracy is reported over the items we could actually compare.
            blind_accuracy = round(confirmed / comparable, 3) if comparable > 0 else 1.0
            report_dict["blind_solve_accuracy"] = blind_accuracy
            report_dict["blind_solve_comparable"] = comparable
            report_dict["blind_solve_confident_divergences"] = confident_divergences

            # Run deterministic scoring reconciliation
            cls.reconcile_report_scores(
                report_dict,
                quiz_type=quiz_type,
                confident_divergences=confident_divergences,
                total_items=len(questions)
            )

            # Synchronize penalized score into the written task log file if needed
            try:
                logs_dir = Path(config.project_root).resolve() / "logs"
                penalized_score = report_dict.get("overall_quality_score", 100)
                base_avg = report_dict.get("base_quality_score", penalized_score)
                flawed_count = len(report_dict.get("flawed_item_indices", [])) + confident_divergences
                task_prefix = f"expert_audit_{quiz_data.get('title', 'quiz')[:20]}"
                safe_prefix = re.sub(r'[\\/:*?"<>|\r\n]+', '_', task_prefix).strip('_')
                matching_logs = sorted(logs_dir.glob(f"*_{safe_prefix}*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
                if matching_logs:
                    latest_log = matching_logs[0]
                    log_txt = latest_log.read_text(encoding="utf-8")
                    if "=== COMPOSITE_SCORE:" in log_txt:
                        log_txt = re.sub(
                            r"=== COMPOSITE_SCORE:\s*[\d\.]+%\s*===",
                            f"=== COMPOSITE_SCORE: {penalized_score}.0% ===\n=== L2_PENALTY: Capped from Base {base_avg}% due to {flawed_count} Flaw(s) ===",
                            log_txt
                        )
                        latest_log.write_text(log_txt, encoding="utf-8")
            except Exception as log_sync_err:
                logger.debug(f"Could not sync penalized score to log file: {log_sync_err}")

            return report_dict

        except Exception as e:
            logger.error(f"Level 2 Expert Quality Audit failed with exception: {e}", exc_info=True)
            return None

    @classmethod
    def reconcile_report_scores(
        cls,
        report_dict: Dict[str, Any],
        quiz_type: str = "vocabulary",
        confident_divergences: int = 0,
        total_items: int = 0
    ) -> Dict[str, Any]:
        """Strict deterministic mathematical reconciliation of expert audit scores:
        1. Normalizes all boolean and string values.
        2. Detects fatal defects (double-keys, ungrammatical keys, reversed distractors).
        3. Forces defective items to score 20.
        4. Calculates base average.
        5. Enforces hard cap at 70 if any fatal flaw exists, setting pass_audit = False.
        """
        if not report_dict or not isinstance(report_dict, dict):
            return report_dict

        audit_questions = report_dict.get("questions", [])
        if not total_items and audit_questions:
            total_items = len(audit_questions)

        # 1. Deterministic Item Score Reconciliation & Robust Type Normalization
        for idx, qa in enumerate(audit_questions):
            if not isinstance(qa, dict):
                continue
            
            # Robust boolean normalization
            single_valid = qa.get("single_fit_valid", True)
            if isinstance(single_valid, str):
                single_valid = single_valid.strip().lower() in ("true", "yes", "1")
            elif not isinstance(single_valid, bool):
                single_valid = bool(single_valid)

            # Robust confidence normalization
            conf = qa.get("confidence")
            if isinstance(conf, str):
                conf = conf.strip().capitalize()
                qa["confidence"] = conf

            # Semantic fatality check in feedback (catch inverted keys or ungrammatical keys marked valid by accident)
            diag = str(qa.get("diagnostic_feedback", "")).lower()
            is_negated = any(neg in diag for neg in ["no double", "not a double", "avoids double", "neither double", "without double"])
            if not is_negated and any(kw in diag for kw in [
                "fatally flawed", "fatal flaw", "declared key is ungrammatical", "declared target translation is ungrammatical",
                "invalid double-key", "exhibits an invalid double", "presents a double-key", "is the to"
            ]):
                single_valid = False

            # Normalize triage_action
            triage = str(qa.get("triage_action", "PASS")).strip().upper()
            if triage not in ("PASS", "REPAIR", "REWRITE"):
                if not single_valid:
                    triage = "REWRITE"
                else:
                    triage = "PASS"
            # If item is fatally flawed but model declared PASS, elevate triage to REPAIR or REWRITE
            if not single_valid and triage == "PASS":
                triage = "REWRITE" if any(w in diag for w in ["fatal", "rewrite", "broken", "fidelity", "mismatch"]) else "REPAIR"
            qa["triage_action"] = triage

            qa["single_fit_valid"] = single_valid

            if not single_valid:
                # Fatal flaw penalty: 20 points
                qa["pedagogical_score"] = 20
            else:
                raw_ped = qa.get("pedagogical_score")
                if isinstance(raw_ped, (int, float)) and 0 <= raw_ped <= 100:
                    qa["pedagogical_score"] = int(round(raw_ped))
                else:
                    qa["pedagogical_score"] = 90

        # Calculate translation accuracy or blind accuracy
        if quiz_type == "translation":
            raw_model_acc = report_dict.get("blind_solve_accuracy")
            if isinstance(raw_model_acc, (int, float)) and 0 <= raw_model_acc <= 100:
                acc_val = raw_model_acc / 100.0 if raw_model_acc > 1.0 else raw_model_acc
                report_dict["blind_solve_accuracy"] = round(acc_val, 3)
            else:
                valid_items = sum(
                    1 for qa in audit_questions
                    if isinstance(qa, dict) and qa.get("single_fit_valid") is not False and qa.get("confidence") == "Definite"
                )
                report_dict["blind_solve_accuracy"] = round(valid_items / total_items, 3) if total_items else 1.0
        else:
            blind_acc = report_dict.get("blind_solve_accuracy")
            if isinstance(blind_acc, (int, float)):
                acc_val = blind_acc / 100.0 if blind_acc > 1.0 else blind_acc
                report_dict["blind_solve_accuracy"] = round(acc_val, 3)

        item_scores = [qa["pedagogical_score"] for qa in audit_questions if isinstance(qa, dict)]
        base_avg = round(sum(item_scores) / len(item_scores)) if item_scores else 100
        report_dict["base_quality_score"] = base_avg

        flawed_items = [
            qa.get("item_index", idx + 1)
            for idx, qa in enumerate(audit_questions)
            if isinstance(qa, dict) and not qa.get("single_fit_valid")
        ]
        report_dict["flawed_item_indices"] = flawed_items
        total_fatal_flaws = confident_divergences + len(flawed_items)

        if total_fatal_flaws > 0:
            report_dict["pass_audit"] = False
            penalized_score = min(int(base_avg), 70)
            report_dict["overall_quality_score"] = penalized_score

            # Reconcile summary verdict with transparent disclosure
            current_summary = str(report_dict.get("summary_verdict", ""))
            clean_summary = re.sub(
                r'(?i)(?:no\s+(?:fatal\s+flaws?|double[\-\s]*keys?)[^.\n]*[.\n]*)',
                '',
                current_summary
            ).strip()

            flaw_details = []
            if confident_divergences > 0:
                flaw_details.append(f"{confident_divergences} blind-solve divergence(s)")
            if flawed_items:
                flawed_str = ", ".join(f"#{i}" for i in flawed_items)
                flaw_details.append(f"single-fit failed in Item {flawed_str}")
            
            flag_notice = f"[REVIEW NEEDED: Base Quality {base_avg}%, capped to {penalized_score}% due to {'; '.join(flaw_details)}]."
            report_dict["summary_verdict"] = f"{flag_notice} {clean_summary}".strip()
            logger.warning(
                f"Level 2 Audit flagged for review: base {base_avg}% capped to {penalized_score}% "
                f"due to {total_fatal_flaws} flaw(s) ({', '.join(flaw_details)})."
            )
        else:
            report_dict["overall_quality_score"] = base_avg
            report_dict["pass_audit"] = (base_avg >= 80)

        return report_dict

    @classmethod
    def get_defective_item_indices(cls, audit_report: Dict[str, Any]) -> List[int]:
        """Identifies 0-indexed positions of questions that failed psychometric audit gates.
        A question is marked defective if:
        - single_fit_valid is False
        - pedagogical_score < 75
        - blind-solve divergence was detected ([DIVERGENCE:)
        """
        if not audit_report or not audit_report.get("questions"):
            return []

        raw_questions = audit_report.get("questions", [])
        # Determine if LLM used 1-based indexing (e.g. indices 1..N instead of 0..N-1)
        raw_indices = [q.get("item_index") for q in raw_questions if isinstance(q.get("item_index"), int) and not isinstance(q.get("item_index"), bool)]
        is_one_based = bool(raw_indices and (0 not in raw_indices or max(raw_indices) == len(raw_questions)))

        defective = []
        for idx, q_audit in enumerate(raw_questions):
            item_raw = q_audit.get("item_index")
            if isinstance(item_raw, int) and not isinstance(item_raw, bool):
                item_idx = (item_raw - 1) if is_one_based else item_raw
            else:
                item_idx = idx

            single_valid = q_audit.get("single_fit_valid", True)
            if isinstance(single_valid, str):
                single_valid = single_valid.strip().lower() in ("true", "yes", "1")
            
            q_score = q_audit.get("pedagogical_score", 100)
            diag = q_audit.get("diagnostic_feedback", "")

            # Check if this item failed core quality gates
            has_divergence = "[DIVERGENCE:" in diag
            has_low_score = q_score < 75
            has_invalid_single_fit = not single_valid

            if has_invalid_single_fit or has_divergence or has_low_score:
                defective.append(item_idx)

        return sorted(list(set(defective)))

    @classmethod
    def generate_surgical_critique(
        cls,
        audit_report: Dict[str, Any],
        full_quiz_questions: List[Dict[str, Any]],
        defective_indices: List[int],
        target_language: Optional[str] = None,
        template_name: str = "vocabulary",
        unit_headwords: Optional[List[str]] = None
    ) -> str:
        """Constructs focused critique feedback instructing the LLM to rewrite ONLY the defective items.
        Passes the exact defect reasons for each flagged item while locking approved items.
        """
        audit_questions = audit_report.get("questions", [])
        audit_map = {}
        for q in audit_questions:
            idx = q.get("item_index")
            if isinstance(idx, int):
                # Account for potential 1-based indexing in report
                zero_idx = (idx - 1) if (idx > 0 and 0 not in [x.get("item_index") for x in audit_questions]) else idx
                audit_map[zero_idx] = q

        lang = target_language or "Simplified Chinese"
        
        # Differentiate surgical mandate title based on quiz template
        t_lower = str(template_name or "vocabulary").lower()
        if "translation" in t_lower:
            mandate_title = f"SURGICAL TRANSLATION ASSESSMENT REPAIR ({lang}-to-English)"
        elif "reading" in t_lower:
            mandate_title = "SURGICAL READING COMPREHENSION REPAIR"
        elif "video" in t_lower or "listening" in t_lower:
            mandate_title = f"SURGICAL {t_lower.upper()} QUIZ REPAIR"
        else:
            mandate_title = "SURGICAL VOCABULARY ASSESSMENT REPAIR"

        feedback_lines = [
            f"## SECTION 1: AUDIT DIAGNOSTIC REPORT ({mandate_title})",
            f"The quality auditor verified your initial draft. Exactly {len(defective_indices)} defective item(s) require targeted repair.",
            "Review the specific defects and diagnostics below:\n"
        ]

        def _clean_diagnostic(raw_diag: str) -> str:
            """Strips internal auditor self-corrections, leaving only actionable defects."""
            text = str(raw_diag or "").strip()
            # Remove trailing CoT rumination like 'Wait, why... Let's re-evaluate...'
            text = re.sub(r'(?i)\b(?:wait[,\s]|let\'s\s+re-?evaluate|ah[,\s]|no[,\.\s]).*$', '', text)
            return text.strip() or raw_diag.strip()

        for def_idx in defective_indices:
            orig_q = full_quiz_questions[def_idx] if 0 <= def_idx < len(full_quiz_questions) else {}
            q_audit = audit_map.get(def_idx, {})

            item_display_num = def_idx + 1
            score = q_audit.get("pedagogical_score", 0)
            raw_diag = q_audit.get("diagnostic_feedback", "").strip()
            clean_diag = _clean_diagnostic(raw_diag)

            # Collect flawed distractor details
            flawed_reasons = []
            for d in q_audit.get("distractors", []):
                rating = d.get("plausibility_rating", "")
                rationale = d.get("elimination_rationale", "")
                letter = d.get("option_letter", "")
                if rating == "Low (Flawed)" or "double" in rationale.lower() or "synonym" in rationale.lower():
                    flawed_reasons.append(f"Option [{letter}]: {rationale}")

            defect_payload = {
                "item_index": item_display_num,
                "expert_audit_score": f"{score}/100",
                "expert_diagnostic": clean_diag,
                "flawed_distractors": flawed_reasons,
                "original_item_json": orig_q
            }

            feedback_lines.append(f"### [DEFECTIVE ITEM #{item_display_num}]")
            feedback_lines.append("```json")
            feedback_lines.append(json.dumps(defect_payload, ensure_ascii=False, indent=2))
            feedback_lines.append("```")

            # Concise item-specific repair directive
            if "translated_sentence" in orig_q:
                kw = orig_q.get("target_keyword", "")
                if "fidelity" in clean_diag.lower() or "mismatch" in clean_diag.lower() or "broken" in clean_diag.lower() or score <= 20:
                    feedback_lines.append(
                        f"👉 REQUIRED REVISION FOR ITEM #{item_display_num}: "
                        f"Rewrite a completely NEW, natural {lang} sentence that genuinely expresses '{kw}'. "
                        f"Create a matching English skeleton and 4 parallel options. DO NOT reuse the flawed original stem."
                    )
                else:
                    feedback_lines.append(
                        f"👉 REQUIRED REVISION FOR ITEM #{item_display_num}: "
                        f"Repair the flagged distractors for '{kw}'. Ensure the slot option seamlessly completes the sentence."
                    )
            else:
                target_w = orig_q.get("target_word", "") or orig_q.get("word", "")
                feedback_lines.append(
                    f"👉 REQUIRED REVISION FOR ITEM #{item_display_num} (Target Word: {target_w}): "
                    f"Keep target '{target_w}'. Sharpen stem context and provide 4 distinct options with exactly one valid key."
                )
            feedback_lines.append("")

        feedback_lines.append("## SECTION 2: REPAIR INSTRUCTIONS & MANDATES")
        feedback_lines.append("⚠️ CRITICAL: All repairs MUST strictly be grounded in the REFERENCE MATERIAL (VOCABULARY & GRAMMAR) provided in the first prompt above!")
        feedback_lines.append("1. GROUNDED FIDELITY: Every revised stem, keyword, and grammar formula MUST strictly derive from the provided material above.")
        feedback_lines.append("2. SELECTIVE SURGERY: Regenerate ONLY the defective items listed above. DO NOT touch, rewrite, or output the passed items.")
        feedback_lines.append("3. STRICT BOUNDARIES: Place both target keyword and grammar formula INSIDE the [ ____ ] slot. Never duplicate words outside the slot.")
        feedback_lines.append("4. ZERO-INDEX ALIGNMENT: 'correct_answer_index' must strictly be 0 for Option A, 1 for Option B, 2 for Option C, 3 for Option D.")
        feedback_lines.append("5. OUTPUT FORMAT: Output a valid JSON object matching the quiz schema containing ONLY the repaired items in the 'questions' list.")

        return "\n".join(feedback_lines)

    @classmethod
    def generate_critique_feedback(cls, audit_report: Dict[str, Any]) -> str:
        """Transforms a failed Level 2 Audit Report into surgical self-correction feedback for the LLM."""
        score = audit_report.get("overall_quality_score", 0)
        accuracy = audit_report.get("blind_solve_accuracy", 0.0) * 100
        verdict = audit_report.get("summary_verdict", "")

        feedback_lines = [
            f"### 🚨 [LEVEL 2 EXPERT QUALITY AUDIT FAILED - Score: {score}/100, Blind Solve Agreement: {accuracy:.1f}%]",
            f"Executive Verdict: {verdict}",
            "",
            "CRITICAL DEFECTS IDENTIFIED BY PSYCHOMETRIC AUDITOR:"
        ]

        for q_audit in audit_report.get("questions", []):
            item_num = q_audit.get("item_index", 0) + 1
            diag = q_audit.get("diagnostic_feedback", "").strip()
            single_valid = q_audit.get("single_fit_valid", True)
            q_score = q_audit.get("pedagogical_score", 100)

            flaws = []
            if not single_valid:
                flaws.append("Multiple defensible keys or key leakage detected (lacks unique single-fit answer)")
            if "[DIVERGENCE:" in diag:
                flaws.append("Blind-solver chose a different option; question stem or evidence is ambiguous")
            
            # Check for low quality distractors
            for d in q_audit.get("distractors", []):
                if d.get("plausibility_rating") == "Low (Flawed)":
                    flaws.append(f"Option [{d.get('option_letter')}] is a flawed/trivial giveaway: {d.get('elimination_rationale')}")

            if flaws or q_score < 80:
                feedback_lines.append(f"- Item #{item_num} (Score: {q_score}/100):")
                for f in flaws:
                    feedback_lines.append(f"    * {f}")
                if diag:
                    feedback_lines.append(f"    * Auditor Note: {diag}")

        feedback_lines.append("")
        feedback_lines.append("MANDATORY CORRECTION ACTIONS:")
        feedback_lines.append("1. Ambiguous Stems: Sharpen the question stem with unequivocal textual anchors from the source.")
        feedback_lines.append("2. Absolute Single-Fit: Ensure ONE and ONLY ONE option is defensible; eliminate accidental secondary keys.")
        feedback_lines.append("3. Plausible Distractors: Replace any trivial or absurd choices with authentic linguistic/reading traps.")
        feedback_lines.append("4. ZERO-INDEX ALIGNMENT: 'correct_answer_index' must be 0 for Option A, 1 for Option B, 2 for Option C, 3 for Option D. NEVER declare a mismatched index number.")
        feedback_lines.append("5. Please output the corrected, complete JSON object resolving all issues above.")

        return "\n".join(feedback_lines)


