"""Log evaluation and batch log audit engine for Lexis Wiki.

Extracted from evaluator.py to separate log parsing & auditing
from zero-token code gate scoring heuristics.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import config
from .dimension_scorers import (
    _score_pedagogy,
    _score_schema,
    _score_uniqueness,
    _score_verbatim,
)
from .eval_helpers import (
    FATAL_QA_FLAGS,
    W_PEDAGOGY,
    W_SCHEMA,
    W_UNIQUENESS,
    W_VERBATIM,
    _clean_core,
    _detect_task_type,
    _extract_items,
    _extract_json,
    _extract_target_skeletons,
)


class LogEvaluator:
    """
    Evaluates LLM execution logs across 4 pedagogical quality dimensions:
    1. Schema & Structural Adherence (25%)
    2. Source Faithfulness & Verbatim Verification (30%)
    3. Pedagogical & Slot-Filling Quality (25%)
    4. Uniqueness & Deduplication (20%)

    Scoring convention:
    - Dimensions returning None (N/A) are excluded from composite calculation.
    - Composite Score is strictly normalized to 0–100 based on applicable dimensions.
    """

    # Cache & thread-lock for repeated audits (dashboard polls /api/hero-board).
    _AUDIT_LOCK = threading.Lock()
    _AUDIT_CACHE: Dict[str, Any] = {"key": None, "value": None}

    @classmethod
    def parse_log_file(cls, log_path: Path) -> Optional[Dict[str, Any]]:
        """Parses a single .log file and extracts metadata, status, prompt content, and JSON response."""
        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
        except Exception:
            return None

        task_match = re.search(r"=== TASK:\s*(.+?)\s*===", content)
        model_match = re.search(r"=== MODEL:\s*(.+?)\s*===", content)
        time_match = re.search(r"=== TIMESTAMP:\s*(.+?)\s*===", content)
        status_match = re.search(r"=== STATUS:\s*(.+?)\s*===", content)
        fail_cat_match = re.search(r"=== FAILURE_CATEGORY:\s*(.+?)\s*===", content)

        if not task_match or not model_match:
            return None

        task = task_match.group(1).strip()
        model = model_match.group(1).strip()
        timestamp = time_match.group(1).strip() if time_match else ""
        status = status_match.group(1).strip() if status_match else "SUCCESS"
        failure_category = fail_cat_match.group(1).strip() if fail_cat_match else None

        pre_cure_match = re.search(r"=== PRE_CURE_SCORE:\s*([\d\.]+)%\s*===", content)
        pre_cure_score = float(pre_cure_match.group(1)) if pre_cure_match else None

        post_cure_match = re.search(r"=== POST_CURE_SCORE:\s*([\d\.]+)%\s*===", content)
        post_cure_score = float(post_cure_match.group(1)) if post_cure_match else None

        # Extract SYSTEM PROMPT content block (fallback context)
        system_prompt = ""
        system_prompt_match = re.search(r"--- SYSTEM PROMPT ---\n(.*?)(?=\n--- USER PROMPT ---|\n--- RAW RESPONSE ---|\n=== END OF LOG ===|\Z)", content, re.DOTALL)
        if system_prompt_match:
            system_prompt = system_prompt_match.group(1).strip()

        # Extract USER PROMPT content block
        user_prompt = ""
        user_prompt_match = re.search(r"--- USER PROMPT ---\n(.*?)(?=\n--- CONSOLIDATED ITEM|\n--- RAW RESPONSE ---|\n=== END OF LOG ===|\Z)", content, re.DOTALL)
        if user_prompt_match:
            user_prompt = user_prompt_match.group(1).strip()

        # Extract RAW RESPONSE
        raw_response = ""
        response_match = re.search(r"--- RAW RESPONSE ---\n(.*?)(?=\n=== END OF LOG ===|\Z)", content, re.DOTALL)
        if response_match:
            raw_response = response_match.group(1).strip()

        # Parse JSON (direct, fenced ```json, or first balanced brace)
        parsed_json = _extract_json(raw_response)

        return {
            "log_name": log_path.name,
            "task": task,
            "model": model,
            "status": status,
            "failure_category": failure_category,
            "timestamp": timestamp,
            "pre_cure_score": pre_cure_score,
            "post_cure_score": post_cure_score,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_json": parsed_json,
        }

    @classmethod
    def evaluate_log(cls, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the 4-dimension audit on a parsed log entry."""
        log_name = log_data.get("log_name", "")
        task = log_data.get("task", "")
        model = log_data.get("model", "")
        status = log_data.get("status", "SUCCESS")
        failure_category = log_data.get("failure_category")
        pre_cure_score = log_data.get("pre_cure_score")
        post_cure_score = log_data.get("post_cure_score")
        user_prompt = log_data.get("user_prompt", "")
        system_prompt = log_data.get("system_prompt", "")
        context_prompt = log_data.get("context_prompt", "") or system_prompt
        parsed = log_data.get("parsed_json")

        raw_response = log_data.get("raw_response", "")
        task_type = _detect_task_type(task, parsed)
        items = _extract_items(parsed, task_type)

        # 0. Immediate Gate for WSD Arbitration Logs & Explicit LLM Execution Failures
        if task_type == "wsd_arbitration":
            n_err = int(parsed.get("errors") or 0) if isinstance(parsed, dict) else (1 if status == "FAILED" else 0)
            n_tot = int(parsed.get("total_arbitrated") or 0) if isinstance(parsed, dict) else 0
            if status == "FAILED" or n_err > 0:
                first_err = ""
                if isinstance(parsed, dict):
                    for dec in (parsed.get("decisions") or []):
                        if isinstance(dec, dict) and dec.get("error"):
                            first_err = str(dec.get("error"))
                            break
                if not first_err:
                    first_err = next((ln.strip() for ln in (raw_response or "").splitlines() if ln.strip()), "LLM unreachable or timed out")
                first_err = re.sub(r"^\[LLM_[A-Z_]+\]\s*", "", first_err)
                return {
                    "log_name": log_name,
                    "task": task,
                    "task_type": task_type,
                    "model": model,
                    "status": "FAILED",
                    "failure_category": failure_category or "WSD_ARBITRATION_FAILED",
                    "n_items": n_tot,
                    "composite_score": 0.0,
                    "pre_cure_score": pre_cure_score,
                    "post_cure_score": post_cure_score,
                    "scores": {
                        "schema_adherence": 0.0,
                        "verbatim_faithfulness": 0.0,
                        "pedagogical_quality": 0.0,
                        "uniqueness": 0.0,
                    },
                    "flags": [
                        f"❌ CRITICAL [LLM_EXECUTION_FAILED]: WSD sense arbitration failed ({n_err}/{max(n_tot, 1)} error(s), zero-fallback enforced) — {first_err[:180]}"
                    ],
                }
            return {
                "log_name": log_name,
                "task": task,
                "task_type": task_type,
                "model": model,
                "status": "SUCCESS",
                "failure_category": None,
                "n_items": n_tot,
                "composite_score": 100.0,
                "pre_cure_score": pre_cure_score,
                "post_cure_score": post_cure_score,
                "scores": {
                    "schema_adherence": W_SCHEMA,
                    "verbatim_faithfulness": W_VERBATIM,
                    "pedagogical_quality": W_PEDAGOGY,
                    "uniqueness": W_UNIQUENESS,
                },
                "flags": [f"✅ [WSD_ARBITRATION] {n_tot} polysemous item(s) disambiguated via LLM with 0 errors"],
            }

        if status == "FAILED":
            err_line = next((ln.strip() for ln in (raw_response or "").splitlines() if ln.strip()), "LLM request failed")
            err_line = re.sub(r"^\[LLM_[A-Z_]+\]\s*", "", err_line)
            cat_label = failure_category or "API_ERROR"
            return {
                "log_name": log_name,
                "task": task,
                "task_type": task_type,
                "model": model,
                "status": "FAILED",
                "failure_category": cat_label,
                "n_items": 0,
                "composite_score": 0.0,
                "pre_cure_score": pre_cure_score,
                "post_cure_score": post_cure_score,
                "scores": {
                    "schema_adherence": 0.0,
                    "verbatim_faithfulness": 0.0,
                    "pedagogical_quality": 0.0,
                    "uniqueness": 0.0,
                },
                "flags": [
                    f"❌ CRITICAL [LLM_EXECUTION_FAILED]: {cat_label} — {err_line[:180]}"
                ],
            }

        flags: List[str] = []
        schema_score, f1 = _score_schema(parsed, raw_response, task_type=task_type)
        verbatim_score, f2 = _score_verbatim(items, task_type, user_prompt, context_prompt=context_prompt)
        effective_prompt = user_prompt if user_prompt.strip() else context_prompt
        pedagogy_score, f3 = _score_pedagogy(items, task_type, user_prompt=effective_prompt)
        uniqueness_score, f4 = _score_uniqueness(items, task_type, user_prompt=effective_prompt)
        flags.extend(f1 + f2 + f3 + f4)

        # Dimension scores & weights map
        dim_results = [
            ("schema_adherence", schema_score, W_SCHEMA),
            ("verbatim_faithfulness", verbatim_score, W_VERBATIM),
            ("pedagogical_quality", pedagogy_score, W_PEDAGOGY),
            ("uniqueness", uniqueness_score, W_UNIQUENESS),
        ]

        applicable_weight = sum(w for _, score, w in dim_results if score is not None)
        earned_score = sum(score for _, score, _ in dim_results if score is not None)

        if applicable_weight > 0:
            composite_score = round((earned_score / applicable_weight) * 100.0, 1)
        else:
            composite_score = 0.0

        # Deterministic Target Coverage Gate: check if model delivered all requested target items
        if task_type in ("grammar", "expressions", "vocabulary"):
            skeletons = _extract_target_skeletons(effective_prompt, task_type)
            expected_count = len(skeletons)
            if expected_count > 0:
                delivered_count = len(items)
                if delivered_count < expected_count:
                    coverage_ratio = delivered_count / expected_count
                    # Detect whether targets come from auto-mining (AWL suggestions) vs teacher-provided syllabus
                    # "DETERMINISTIC TARGET VOCABULARY" = auto-mined AWL words (soft recommendations)
                    # "DETERMINISTIC TARGET PATTERNS"   = in-text grammar patterns (strict, they ARE in the text)
                    is_auto_mined = "DETERMINISTIC TARGET VOCABULARY" in (effective_prompt or "")

                    # Name the missing targets so the retry critique can target them specifically
                    name_key = "word" if task_type == "vocabulary" else "formula"
                    missing_names: List[str] = []
                    if name_key:
                        delivered_names = {
                            _clean_core(str(it.get(name_key) or it.get("word") or it.get("pattern_formula") or "")).lower()
                            for it in items
                        }
                        for sk in skeletons:
                            sk_name = str(sk.get(name_key) or "").strip()
                            if sk_name and _clean_core(sk_name).lower() not in delivered_names:
                                missing_names.append(sk_name)
                    missing_txt = f" — missing: {', '.join(missing_names[:12])}" if missing_names else ""

                    target_source_desc = "Auto-mined" if is_auto_mined else "Syllabus"
                    # Unified Target Coverage Gate (85% tolerance):
                    # Accounts for local LLM (e.g. gemma4) tendency to occasionally deliver e.g. 19/20 or 4/5 items.
                    # >= 85% coverage: Informational warning (non-fatal), slight score adjustment (at most 10%).
                    # < 85% coverage: FATAL defect (❌ [INCOMPLETE_COVERAGE]), triggers retry/correction and proportional score scaling.
                    if coverage_ratio >= 0.85:
                        flags.append(
                            f"⚠️ [PARTIAL_COVERAGE] {target_source_desc} target coverage: delivered {delivered_count}/{expected_count} targets{missing_txt}"
                        )
                        composite_score = max(0.0, round(composite_score * max(coverage_ratio, 0.9), 1))
                    else:
                        flags.append(
                            f"❌ [INCOMPLETE_COVERAGE] Incomplete target coverage: delivered only {delivered_count}/{expected_count} targets{missing_txt}"
                        )
                        composite_score = round(composite_score * coverage_ratio, 1)

        # Special handling for expert_audit task: align composite_score with deterministic code gate
        if task_type == "expert_audit" and isinstance(parsed, dict):
            try:
                from ..expert_auditor import ExpertAuditor
                infer_type = "vocabulary"
                task_lower = (task or "").lower()
                prompt_lower = (user_prompt or "").lower()
                if "translation" in task_lower or "translation" in prompt_lower:
                    infer_type = "translation"
                elif "reading" in task_lower or "reading" in prompt_lower:
                    infer_type = "reading"
                
                expected_items = len(re.findall(r"(?:^|\n)\s*Item\s+#\d+:", user_prompt or ""))
                divs = parsed.get("blind_solve_confident_divergences", 0)
                ExpertAuditor.reconcile_report_scores(
                    parsed,
                    quiz_type=infer_type,
                    confident_divergences=divs,
                    total_items=expected_items
                )
                raw_audited_score = round(float(parsed.get("overall_quality_score", composite_score)), 1)
                missing_audit_items = int(parsed.get("missing_audit_items", 0) or 0)
                
                if pre_cure_score is None or missing_audit_items > 0:
                    pre_cure_score = raw_audited_score
                if post_cure_score is not None and missing_audit_items == 0:
                    composite_score = post_cure_score
                else:
                    composite_score = raw_audited_score
                
                flaws = parsed.get("flawed_item_indices", [])
                if flaws:
                    flags.insert(0, f"❌ [FATAL FLAW] {len(flaws)} item(s) failed single-fit validation: #{', #'.join(str(i) for i in flaws)}")
                if missing_audit_items > 0:
                    audited_n = len(parsed.get("questions", []))
                    flags.insert(0, f"❌ [FATAL FLAW] Audit truncated early: only {audited_n}/{expected_items} item(s) evaluated ({missing_audit_items} missing)")
                if not parsed.get("pass_audit", False):
                    base = parsed.get("base_quality_score", raw_audited_score)
                    if base > raw_audited_score:
                        flags.insert(1, f"⚠️ [L2_PENALTY] Capped from Base {base}% to {raw_audited_score}% due to fatal flaw(s)")
                if divs > 0:
                    flags.insert(0, f"❌ [CRITICAL FLAW] {divs} confident double-key divergence(s) detected in blind solve")
                if post_cure_score is not None and post_cure_score > raw_audited_score and missing_audit_items == 0:
                    flags.insert(0, f"✅ [SURGICAL CURE] In-place healed from {pre_cure_score}% to Final {post_cure_score}%")
            except Exception as err:
                composite_score = round(float(parsed.get("overall_quality_score", composite_score)), 1)

        # Fatal QA Flag Gate: If fatal pedagogical flags are detected, composite score cannot claim passing (>=80%)
        # Automatically cap composite score below 60.0 to reflect failed/review status
        has_fatal_flags = any(
            any(fatal in f for fatal in FATAL_QA_FLAGS)
            for f in flags
        )
        if has_fatal_flags:
            composite_score = min(composite_score, 59.0)
            if status == "SUCCESS":
                status = "REVIEW_NEEDED"
                if not failure_category:
                    failure_category = "QA_FATAL_FLAG"

        # Sort flags: critical errors (❌) first, then warnings (⚠️)
        flags.sort(key=lambda x: (0 if x.startswith("❌") else (1 if x.startswith("⚠️") else 2)))

        return {
            "log_name": log_name,
            "task": task,
            "task_type": task_type,
            "model": model,
            "status": status,
            "failure_category": failure_category,
            "n_items": len(items),
            "composite_score": composite_score,
            "pre_cure_score": pre_cure_score,
            "post_cure_score": post_cure_score,
            "scores": {
                "schema_adherence": schema_score,
                "verbatim_faithfulness": verbatim_score,
                "pedagogical_quality": pedagogy_score,
                "uniqueness": uniqueness_score,
            },
            "flags": flags,
        }

    @classmethod
    def audit_all_logs(cls, logs_dir: Path = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Scans and audits all .log files in the logs/ directory.
        Returns the overall Hero Board leaderboard and individual log audits.
        Results are cached by MD5 hash of entries' metadata so repeated calls stay fast and thread-safe.
        """
        if logs_dir is None:
            logs_dir = config.project_root / "logs"
        logs_dir = Path(logs_dir)

        if not logs_dir.exists():
            return {"hero_board": [], "audits": []}

        entries = []
        for path in logs_dir.glob("*.log"):
            try:
                st = path.stat()
            except OSError:
                continue
            entries.append((path, st.st_mtime, st.st_size))

        if not entries:
            return {"hero_board": [], "audits": []}

        # Build compact hash-based cache key using sha256 (FIPS compliant)
        sig_data = f"{logs_dir.as_posix()}:{len(entries)}:" + ":".join(f"{p.name}:{mt}:{sz}" for (p, mt, sz) in entries)
        cache_key = hashlib.sha256(sig_data.encode("utf-8")).hexdigest()

        with cls._AUDIT_LOCK:
            if use_cache and cls._AUDIT_CACHE["key"] == cache_key and cls._AUDIT_CACHE["value"] is not None:
                return cls._AUDIT_CACHE["value"]

        # Newest first
        entries.sort(key=lambda e: e[1], reverse=True)

        audits: List[Dict[str, Any]] = []
        model_stats: Dict[str, Dict[str, Any]] = {}

        # Worker for ThreadPoolExecutor
        def _process_one(log_file: Path) -> Optional[Dict[str, Any]]:
            parsed = cls.parse_log_file(log_file)
            if not parsed:
                return None
            evaluation = cls.evaluate_log(parsed)
            if evaluation.get("task_type") == "intermediate_prose":
                return None
            return evaluation

        # Parallelize log parsing and evaluation (preserves recency order)
        with ThreadPoolExecutor() as executor:
            raw_evaluations = list(executor.map(_process_one, [e[0] for e in entries]))

        for evaluation in raw_evaluations:
            if not evaluation:
                continue

            audits.append(evaluation)

            if evaluation.get("task_type") == "wsd_arbitration" and evaluation.get("status") != "FAILED":
                continue

            model = evaluation["model"]
            if model not in model_stats:
                model_stats[model] = {
                    "model": model,
                    "runs": 0,
                    "success_runs": 0,
                    "failed_runs": 0,
                    "total_score": 0.0,
                    "schema_sum": 0.0,
                    "schema_runs": 0,
                    "verbatim_sum": 0.0,
                    "verbatim_runs": 0,
                    "pedagogy_sum": 0.0,
                    "pedagogy_runs": 0,
                    "uniqueness_sum": 0.0,
                    "uniqueness_runs": 0,
                }

            s = model_stats[model]
            s["runs"] += 1

            # Exclude FAILED or RETRYING status logs from the quality leaderboard composite score
            if evaluation.get("status") in ("FAILED", "RETRYING"):
                s["failed_runs"] += 1
                continue

            s["success_runs"] += 1
            s["total_score"] += evaluation["composite_score"]

            sc = evaluation["scores"]
            if sc["schema_adherence"] is not None:
                s["schema_sum"] += sc["schema_adherence"]
                s["schema_runs"] += 1
            if sc["verbatim_faithfulness"] is not None:
                s["verbatim_sum"] += sc["verbatim_faithfulness"]
                s["verbatim_runs"] += 1
            if sc["pedagogical_quality"] is not None:
                s["pedagogy_sum"] += sc["pedagogical_quality"]
                s["pedagogy_runs"] += 1
            if sc["uniqueness"] is not None:
                s["uniqueness_sum"] += sc["uniqueness"]
                s["uniqueness_runs"] += 1

        # Build Hero Board Leaderboard
        hero_board = []
        for model, s in model_stats.items():
            succ = s["success_runs"]
            if succ == 0:
                comp = 0.0
                sch_avg = vbt_avg = ped_avg = unq_avg = 0.0
            else:
                comp = round(s["total_score"] / succ, 1)
                sch_avg = round((s["schema_sum"] / (s["schema_runs"] * W_SCHEMA)) * 100.0, 1) if s["schema_runs"] else 100.0
                vbt_avg = round((s["verbatim_sum"] / (s["verbatim_runs"] * W_VERBATIM)) * 100.0, 1) if s["verbatim_runs"] else 100.0
                ped_avg = round((s["pedagogy_sum"] / (s["pedagogy_runs"] * W_PEDAGOGY)) * 100.0, 1) if s["pedagogy_runs"] else 100.0
                unq_avg = round((s["uniqueness_sum"] / (s["uniqueness_runs"] * W_UNIQUENESS)) * 100.0, 1) if s["uniqueness_runs"] else 100.0

            hero_board.append({
                "model": model,
                "runs": s["runs"],
                "success_runs": succ,
                "failed_runs": s["failed_runs"],
                "composite_score": comp,
                "schema_adherence_avg": sch_avg,
                "verbatim_faithfulness_avg": vbt_avg,
                "pedagogical_quality_avg": ped_avg,
                "uniqueness_avg": unq_avg,
            })

        # Sort Leaderboard descending by composite score
        hero_board.sort(key=lambda x: x["composite_score"], reverse=True)

        result = {"hero_board": hero_board, "audits": audits}
        with cls._AUDIT_LOCK:
            cls._AUDIT_CACHE["key"] = cache_key
            cls._AUDIT_CACHE["value"] = result
        return result
