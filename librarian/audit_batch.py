"""Batch re-audit of existing quiz handouts using the Level-2 expert audit.

This is a **read-only** tool: it does NOT regenerate any content. For every rendered
quiz handout it finds (``wiki/<unit>/handouts/*_quiz.html``) it

  1. extracts the structured ``quizData`` (questions / options / correct_answer_index)
     that the handout embeds as ``const quizData = {...}``,
  2. loads the unit's source text (``wiki/<unit>/sources/<unit>.md``), and
  3. re-runs :meth:`ExpertAuditor.audit_quiz` (the de-biased Level-2 judge).

Because it only calls the *judge* model (not the generator), it is cheap and can be
run across the whole wiki to regression-check pass rates after a prompt, model, or
auditor change. Each result row also carries the previously embedded verdict
(``quizData._expert_audit``) so the summary can show ``before -> after`` for the fix.

The module is deliberately decoupled from the dashboard/CLI so it can be unit-tested
with a mocked judge and a temporary ``wiki/`` tree.
"""
import re
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import config, normalize_name
from .expert_auditor import ExpertAuditor

logger = logging.getLogger("librarian.audit_batch")

# Quiz templates that can appear in a handout filename ``<stem>_<template>_quiz.html``.
QUIZ_TEMPLATES = ("vocabulary", "reading", "translation", "listening", "video")

_FILENAME_RE = re.compile(
    r"^(?P<stem>.+)_(?P<template>vocabulary|reading|translation|listening|video)_quiz\.html$"
)

# Matches the JS injection point, e.g. ``const quizData = {"title": ...}``
_QUIZDATA_RE = re.compile(r"const\s+quizData\s*=\s*")


def _project_root(project_root: Optional[str]) -> Path:
    """Resolve the project root, preferring an explicit override (test-friendly)."""
    return Path(project_root) if project_root else Path(config.project_root)


def discover_quiz_units(project_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find every rendered quiz handout under ``wiki/*/handouts/*_quiz.html``.

    Returns a stable, sorted list of ``{"unit", "template", "path"}`` dicts where
    ``unit`` is the name of the unit directory (the handout's grandparent folder).
    """
    wiki_dir = _project_root(project_root) / "wiki"
    refs: List[Dict[str, Any]] = []
    if not wiki_dir.exists():
        return refs

    for handout in wiki_dir.glob("*/handouts/*_quiz.html"):
        if not handout.is_file():
            continue
        m = _FILENAME_RE.match(handout.name)
        if not m:
            continue
        unit_dir = handout.parent.parent  # wiki/<unit>
        refs.append({
            "unit": unit_dir.name,
            "template": m.group("template"),
            "path": str(handout),
        })

    refs.sort(key=lambda r: (r["unit"], r["template"]))
    return refs


def load_quiz_data(html_path: str, project_root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Extract the injected ``const quizData = {...}`` JSON object from a handout.

    Uses :meth:`json.JSONDecoder.raw_decode` from the injection point so nested braces
    and braces inside option/explanation text are handled correctly (a naive regex
    would break on the first ``}`` inside a string). Returns ``None`` if not found.
    """
    try:
        html = Path(html_path).read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Failed to read handout %s: %s", html_path, e)
        return None

    m = _QUIZDATA_RE.search(html)
    if not m:
        return None

    try:
        obj, _end = json.JSONDecoder().raw_decode(html, m.end())
    except Exception as e:
        logger.error("Failed to parse quizData in %s: %s", html_path, e)
        return None
    return obj if isinstance(obj, dict) else None


def resolve_source_text(unit: str, project_root: Optional[str] = None) -> str:
    """Load the unit's primary source text.

    Prefers ``wiki/<unit>/sources/<unit>.md`` and falls back to the first ``.md``/``.txt``
    in the unit's ``sources/`` directory. Returns an empty string if nothing is found
    (the judge can still evaluate items that are self-contained, e.g. pure vocabulary).
    """
    unit_dir = _project_root(project_root) / "wiki" / unit
    sources = unit_dir / "sources"
    if not sources.exists():
        return ""

    preferred = sources / f"{unit}.md"
    if preferred.exists() and preferred.is_file():
        try:
            return preferred.read_text(encoding="utf-8")
        except Exception:
            return ""

    for ext in (".md", ".txt"):
        candidates = sorted(sources.glob(f"*{ext}"))
        if candidates:
            try:
                return candidates[0].read_text(encoding="utf-8")
            except Exception:
                return ""
    return ""


def _as_bool(value: Any) -> Optional[bool]:
    """Coerce the (possibly string-typed) ``pass_audit`` value to a real bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return None


def audit_one(ref: Dict[str, Any], judge_model: Optional[str] = None,
              project_root: Optional[str] = None) -> Dict[str, Any]:
    """Re-run the Level-2 expert audit for a single handout and return a result row.

    The row also records the previously embedded verdict (``quizData._expert_audit``)
    so a summary can show ``before -> after`` for the de-biased auditor.
    """
    row: Dict[str, Any] = {
        "unit": ref["unit"],
        "template": ref["template"],
        "path": ref["path"],
        "item_count": 0,
        "passed": None,
        "blind_solve_accuracy": None,
        "overall_quality_score": None,
        "confident_divergences": None,
        "verdict": "",
        "prev_passed": None,
        "prev_blind_solve_accuracy": None,
        "changed": None,
        "error": None,
    }

    quiz_data = load_quiz_data(ref["path"], project_root)
    if not quiz_data or not quiz_data.get("questions"):
        row["error"] = "No quizData/questions found in handout"
        return row
    row["item_count"] = len(quiz_data.get("questions", []))

    # Previously embedded verdict, if present (computed by whatever auditor was active
    # when this handout was generated).
    prev = quiz_data.get("_expert_audit")
    if isinstance(prev, dict):
        row["prev_passed"] = _as_bool(prev.get("pass_audit"))
        row["prev_blind_solve_accuracy"] = prev.get("blind_solve_accuracy")

    source_text = resolve_source_text(ref["unit"], project_root)
    try:
        report = ExpertAuditor.audit_quiz(source_text or "", quiz_data, judge_model=judge_model)
    except Exception as e:
        logger.exception("Re-audit failed for %s [%s]", ref["unit"], ref["template"])
        row["error"] = f"Audit exception: {e}"
        return row

    if not isinstance(report, dict):
        row["error"] = "Audit returned no report"
        return row

    row["passed"] = _as_bool(report.get("pass_audit"))
    row["blind_solve_accuracy"] = report.get("blind_solve_accuracy")
    row["overall_quality_score"] = report.get("overall_quality_score")
    row["confident_divergences"] = report.get("blind_solve_confident_divergences")
    row["verdict"] = report.get("summary_verdict", "")

    if row["prev_passed"] is not None and row["passed"] is not None:
        row["changed"] = (row["prev_passed"] != row["passed"])
    return row


def run_batch(unit_filter: Optional[List[str]] = None,
              template_filter: Optional[str] = None,
              max_workers: int = 2,
              judge_model: Optional[str] = None,
              project_root: Optional[str] = None,
              progress_cb: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
              ) -> Dict[str, Any]:
    """Re-audit a set of quiz handouts concurrently and return an aggregate summary.

    :param unit_filter: optional list of unit names (matched case-insensitively).
    :param template_filter: optional single template name (e.g. ``"reading"``).
    :param max_workers: number of concurrent judge calls (LLM calls are I/O bound).
    :param judge_model: optional judge model override (defaults to the auditor's choice).
    :param project_root: optional project root override (test-friendly).
    :param progress_cb: optional ``cb(done, total, row)`` invoked as each unit finishes.
    """
    refs = discover_quiz_units(project_root)

    if unit_filter:
        wanted = {normalize_name(u) for u in unit_filter}
        refs = [r for r in refs if normalize_name(r["unit"]) in wanted]
    if template_filter:
        refs = [r for r in refs if r["template"] == template_filter]

    total = len(refs)
    summary: Dict[str, Any] = {
        "total": total,
        "audited": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "changed": 0,
        "avg_blind_solve_accuracy": None,
        "results": [],
    }
    if total == 0:
        return summary

    workers = max(1, int(max_workers))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(audit_one, ref, judge_model, project_root): ref for ref in refs}
        for future in as_completed(futures):
            row = future.result()
            summary["results"].append(row)
            if row["error"]:
                summary["errors"] += 1
            else:
                summary["audited"] += 1
                if row["passed"]:
                    summary["passed"] += 1
                else:
                    summary["failed"] += 1
                if row.get("changed"):
                    summary["changed"] += 1
            done += 1
            if progress_cb is not None:
                try:
                    progress_cb(done, total, row)
                except Exception:
                    logger.debug("progress_cb raised during batch re-audit", exc_info=True)

    accs = [r["blind_solve_accuracy"] for r in summary["results"]
            if isinstance(r["blind_solve_accuracy"], (int, float))]
    summary["avg_blind_solve_accuracy"] = round(sum(accs) / len(accs), 3) if accs else None
    summary["results"].sort(key=lambda r: (r["unit"], r["template"]))
    return summary


def format_summary(summary: Dict[str, Any]) -> str:
    """Render a human-readable console summary of a :func:`run_batch` result."""
    lines = [
        "=" * 64,
        "          LEVEL-2 EXPERT AUDIT — BATCH RE-AUDIT SUMMARY",
        "=" * 64,
        f"  Quizzes re-audited : {summary['audited']}/{summary['total']}",
        f"  Passed             : {summary['passed']}",
        f"  Failed             : {summary['failed']}",
    ]
    if summary["errors"]:
        lines.append(f"  Errors             : {summary['errors']}")
    if summary["avg_blind_solve_accuracy"] is not None:
        lines.append(f"  Avg blind-solve    : {summary['avg_blind_solve_accuracy'] * 100:.1f}%")
    if summary["changed"]:
        lines.append(f"  Verdicts flipped   : {summary['changed']}  (embedded -> fresh run)")
    lines.append("-" * 64)

    for r in summary["results"]:
        if r["error"]:
            lines.append(f"  !  {r['unit']} [{r['template']}]: {r['error']}")
            continue
        mark = "PASS" if r["passed"] else "FAIL"
        acc = r["blind_solve_accuracy"]
        acc_s = f"{acc * 100:.0f}%" if isinstance(acc, (int, float)) else "n/a"
        score = r["overall_quality_score"] if r["overall_quality_score"] is not None else "n/a"
        arrow = ""
        if r.get("prev_passed") is not None and r["passed"] is not None and r["prev_passed"] != r["passed"]:
            arrow = f"   (was {'PASS' if r['prev_passed'] else 'FAIL'} -> now {'PASS' if r['passed'] else 'FAIL'})"
        score_str = f"{score}%" if isinstance(score, (int, float)) else str(score)
        lines.append(f"  [{mark}] {r['unit']} [{r['template']}]  {score_str}  blind {acc_s}{arrow}")

    lines.append("=" * 64)
    return "\n".join(lines)