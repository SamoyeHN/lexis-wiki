"""
CLI subparser and handler for audit, re-audit, and golden-set benchmarking.
"""

import argparse
import json
import sys
from pathlib import Path

from ..config import config
from ..llm import llm


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    # Audit / Hero Board command
    audit_parser = subparsers.add_parser("audit", help="Run audit assessment on log files and display the LLM Hero Board")
    audit_parser.add_argument("--json", action="store_true", help="Output audit results as JSON")
    audit_parser.set_defaults(handler=handle_audit)

    # Batch re-audit command (re-runs the Level 2 expert audit over existing handouts or extraction artifacts, read-only)
    reaudit_parser = subparsers.add_parser("re-audit", help="Re-run audit assessment over existing handouts or extraction artifacts")
    reaudit_parser.add_argument("units", nargs="*", help="Specific unit(s) to re-audit (default: all units)")
    reaudit_parser.add_argument("--target", choices=["quiz", "extraction", "all"], default="quiz", help="Target artifacts to re-audit: 'quiz', 'extraction', or 'all' (default: quiz)")
    reaudit_parser.add_argument("--template", "-t", choices=["vocabulary", "cloze", "reading", "translation", "listening", "video"], help="Only re-audit this quiz type")
    reaudit_parser.add_argument("--workers", "-w", type=int, default=2, help="Concurrent audit workers (default: 2)")
    reaudit_parser.add_argument("--llm", action="store_true", help="Enable Level-2 LLM semantic evaluation (e.g. WSD sense alignment for extractions)")
    reaudit_parser.add_argument("--fix", action="store_true", help="Automatically repair detected defects (e.g. in-place WSD sense cures)")
    reaudit_parser.add_argument("--json", action="store_true", help="Output the summary as JSON")
    reaudit_parser.set_defaults(handler=handle_reaudit)

    # Golden Set E2E regression benchmark command
    goldenset_parser = subparsers.add_parser("golden-set", help="Run E2E regression benchmarks over the 5 canonical Golden Set units (A1-C1)")
    goldenset_parser.add_argument("units", nargs="*", help="Specific Golden Set unit(s) to benchmark (default: all 5)")
    goldenset_parser.add_argument("--fast", action="store_true", help="Fast execution mode (L1 code gates only)")
    goldenset_parser.add_argument("--llm", action="store_true", help="Enable optional Level-2 LLM judge for extraction audit")
    goldenset_parser.add_argument("--json", action="store_true", help="Output benchmark results as JSON")
    goldenset_parser.set_defaults(handler=handle_golden_set)


def handle_audit(args: argparse.Namespace) -> None:
    from ..evaluator import LogEvaluator
    res = LogEvaluator.audit_all_logs()
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("\n==========================================")
        print("         LLM HERO BOARD LEADERBOARD       ")
        print("==========================================\n")
        board = res.get("hero_board", [])
        audits = res.get("audits", [])
        if not board and not audits:
            print("No log files found to evaluate in logs/ directory.")
        else:
            for idx, item in enumerate(board, 1):
                badge = f"#{idx}"
                failed_runs = item.get("failed_runs", 0)
                print(f"Rank {badge}: {item['model']}")
                print(f"   * Composite Quality Score: {item['composite_score']}%")
                print(f"   * Schema Adherence:       {item['schema_adherence_avg']}%")
                print(f"   * Verbatim Faithfulness:  {item['verbatim_faithfulness_avg']}%")
                print(f"   * Pedagogical Quality:    {item['pedagogical_quality_avg']}%")
                print(f"   * Uniqueness / No Dups:   {item['uniqueness_avg']}%")
                print(f"   * Total Evaluated Runs:   {item['runs']} (Failed/Aborted: {failed_runs})\n")

            failed_logs = [
                r for r in audits
                if r.get("status") == "FAILED" or r.get("composite_score") == 0.0
            ]
            if failed_logs:
                print("⚠️  RECENT LLM EXECUTION FAILURES (0.0% AUDIT ALERT):")
                for fl in failed_logs[:10]:
                    flags = "; ".join(fl.get("flags") or [])
                    print(f"   - {fl.get('log_name')} [{fl.get('task')}]: {flags}")
                print()


def handle_reaudit(args: argparse.Namespace) -> None:
    target = getattr(args, "target", "quiz")
    has_errors = False

    if target in ("extraction", "all"):
        from ..audit_extraction import batch_audit_extractions, format_extraction_summary
        enable_llm = getattr(args, "llm", False)
        fix = getattr(args, "fix", False)
        if enable_llm:
            mode_desc = "Level-2 semantic audit with LLM arbitration" + (", auto-cure enabled" if fix else ", read-only")
        else:
            mode_desc = "Level-1 deterministic code gate, read-only"
        print(f"Re-auditing extraction artifacts ({mode_desc})…")

        def _ext_progress(done, total, row):
            mark = "✓" if row.get("passed") else "✗"
            print(f"  [{done}/{total}] {mark} {row['unit']} score {row.get('score', 0):.1f}")

        ext_summary = batch_audit_extractions(
            unit_filter=args.units or None,
            max_workers=args.workers,
            progress_cb=_ext_progress,
            enable_llm=enable_llm,
            fix=fix,
        )
        if ext_summary["total"] == 0:
            print("No extraction artifacts found under wiki/ to re-audit.")
            if target == "extraction":
                sys.exit(1)
        elif args.json and target == "extraction":
            print(json.dumps(ext_summary, indent=2, ensure_ascii=False))
        else:
            print(format_extraction_summary(ext_summary))

        from ..audit_extraction import save_extraction_audit_log
        ext_log_paths = save_extraction_audit_log(ext_summary)
        try:
            rel_ext_log = ext_log_paths["log_path"].relative_to(Path(config.project_root))
        except ValueError:
            rel_ext_log = ext_log_paths["log_path"]
        print(f"📄 Extraction defect log saved to: {rel_ext_log}")

        if ext_summary.get("failed", 0) > 0:
            has_errors = True

    if target in ("quiz", "all"):
        from ..audit_batch import run_batch, format_summary, save_quiz_audit_log
        from ..llm import LLMError
        try:
            judge_m = config.get("judge_model") or config.get("model")
            llm.ensure_available(task_name="re_audit_batch", model_override=judge_m)
        except LLMError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

        print("Re-auditing quiz handouts (Level-2 expert audit, read-only)…")

        def _progress(done, total, row):
            mark = "⚠" if row.get("error") else ("✓" if row.get("passed") else "✗")
            acc = row.get("blind_solve_accuracy")
            acc_s = f"{acc * 100:.0f}%" if isinstance(acc, (int, float)) else "n/a"
            print(f"  [{done}/{total}] {mark} {row['unit']} [{row['template']}] blind {acc_s}")

        summary = run_batch(
            unit_filter=args.units or None,
            template_filter=args.template,
            max_workers=args.workers,
            progress_cb=_progress,
        )
        if summary["total"] == 0:
            print("No quiz handouts found under wiki/ to re-audit.")
            sys.exit(1)
        if args.json:
            output_payload = summary if target == "quiz" else {"quiz": summary, "extraction": ext_summary}
            print(json.dumps(output_payload, indent=2, ensure_ascii=False))
        else:
            print(format_summary(summary))

        quiz_log_paths = save_quiz_audit_log(summary)
        try:
            rel_quiz_log = quiz_log_paths["log_path"].relative_to(Path(config.project_root))
        except ValueError:
            rel_quiz_log = quiz_log_paths["log_path"]
        print(f"📄 Quiz defect log saved to: {rel_quiz_log}")

        if summary.get("errors", 0) > 0 or summary.get("failed", 0) > 0:
            has_errors = True

    if has_errors:
        sys.exit(1)


def handle_golden_set(args: argparse.Namespace) -> None:
    from ..golden_set import format_golden_set_summary, run_golden_set_benchmarks

    summary = run_golden_set_benchmarks(
        units=args.units or None,
        fast=args.fast,
        enable_llm=args.llm,
    )

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(format_golden_set_summary(summary))

    if summary.get("overall_status") != "passed":
        sys.exit(1)
