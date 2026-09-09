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
    def format_quiz_for_blind_audit(cls, source_text: str, questions: List[Dict[str, Any]]) -> str:
        """Formats the source passage and quiz questions for blind evaluation.
        
        NOTE: Declared answers (correct_answer_index) and explanations are deliberately
        stripped out so the expert judge must solve each item blindly.
        """
        lines = [
            "### SOURCE MATERIAL:",
            source_text.strip(),
            "",
            "### QUIZ ITEMS FOR EVALUATION (BLIND SOLVER TEST):",
        ]
        
        for idx, q in enumerate(questions):
            q_text = q.get("question") or q.get("translated_sentence") or f"Question {idx + 1}"
            options = q.get("options", [])
            lines.append(f"Item #{idx + 1}:")
            lines.append(f"Stem: {q_text}")
            lines.append("Options:")
            for opt_idx, opt in enumerate(options):
                letter = chr(65 + opt_idx)
                lines.append(f"  [{letter}] {opt}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def audit_quiz(
        cls,
        source_text: str,
        quiz_data: Dict[str, Any],
        judge_model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Executes Level 2 semantic audit on a quiz payload."""
        questions = quiz_data.get("questions", [])
        if not questions:
            logger.warning("No questions found in quiz data to evaluate.")
            return None

        eval_model = judge_model or cls.get_judge_model()
        formatted_content = cls.format_quiz_for_blind_audit(source_text, questions)

        raw_prompt, schema_cls = Prompts.get("expert_audit")
        full_prompt = raw_prompt.format(content=formatted_content)

        # Parse SYSTEM and USER blocks
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
            by_index = {}
            positional = []
            for qa in audit_questions:
                if not isinstance(qa, dict):
                    continue
                positional.append(qa)
                ii = qa.get("item_index")
                if isinstance(ii, int) and not isinstance(ii, bool) and ii not in by_index:
                    by_index[ii] = qa

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

                # Match audit item by 1-based item_index (Item #1 -> idx + 1), fallback to 0-based index or positional
                q_audit = by_index.get(idx + 1)
                if q_audit is None:
                    q_audit = by_index.get(idx)
                if q_audit is None and idx < len(positional):
                    q_audit = positional[idx]
                if q_audit is None:
                    continue

                blind_raw = q_audit.get("blind_solved_index")
                if not isinstance(blind_raw, int) or isinstance(blind_raw, bool):
                    continue
                comparable += 1

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

            # Documented override: a *confident* blind-solve divergence is the
            # genuine double-key / key-leak signal the blind test exists to
            # catch, so it downgrades the verdict. Uncertain divergences are
            # recorded as evidence but do NOT veto (avoids penalising good
            # items for a judge's low-confidence guess).
            current_pass = report_dict.get("pass_audit", False)
            if isinstance(current_pass, str):
                current_pass = current_pass.strip().lower() in ("true", "yes", "1")
            if confident_divergences > 0 and current_pass:
                report_dict["pass_audit"] = False
                logger.warning(
                    f"Level 2 Audit pass_audit downgraded: {confident_divergences} confident "
                    f"blind-solve divergence(s) detected (possible double-key / key-leak)."
                )

            return report_dict

        except Exception as e:
            logger.error(f"Level 2 Expert Quality Audit failed with exception: {e}", exc_info=True)
            return None

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
        feedback_lines.append("4. Please output the corrected, complete JSON object resolving all issues above.")

        return "\n".join(feedback_lines)

