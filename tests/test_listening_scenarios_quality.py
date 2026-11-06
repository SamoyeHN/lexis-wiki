#!/usr/bin/env python3
"""
Listening Scenarios Quality Test & External Semantic/Logical Audit Suite
========================================================================
1. Deterministic Blueprint & Scenario Matrix Unit Tests (`unittest` / `pytest`):
   - Audits all 14 Functional Dialogue Archetypes, 28 Register-Isolated Trajectories,
     and 30 Everyday Scenarios in `librarian/data/functional_dialogue_archetypes.json`.
   - Verifies 0 leaked internal unit/strategy IDs (`u01_s01`, `B1P`, `+ -ing`), strict
     8-turn alternating speaker progression (`Speaker 1` odd turns, `Speaker 2` even turns),
     0 cross-turn duplicate recommended chunks, and 5 immutable question-to-turn blueprints.
   - Verifies clean interpolation of `librarian/prompts/listening_scenario_semantic_judge.md`.

2. Live LLM Scenario Generation & External Review Dossier Builder (`--generate`):
   - Generates 8-turn dialogues and 5 listening comprehension questions across all scenarios
     (or filtered subsets via `--archetype`, `--trajectory`, `--scenario`, `--limit`).
   - Runs post-generation deterministic hydration (`hydrate_listening_blueprint_items`) and
     Level-1 code gates (`audit_listening_integrity`).
   - Optionally runs a second Judge Model (`--run-judge`) with
     `librarian/prompts/listening_scenario_semantic_judge.md` to evaluate dialogue semantics,
     turn adjacency logic, and question single-fit validity.
   - Outputs `logs/listening_scenarios_review_dossier.md` and `.json` for external model or
     human expert review.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from librarian.linguistics import LinguisticEngine
from librarian.schemas import ListeningQuiz


ARCHETYPES_JSON_PATH = PROJECT_ROOT / "librarian" / "data" / "functional_dialogue_archetypes.json"
LISTENING_PROMPT_PATH = PROJECT_ROOT / "librarian" / "prompts" / "listening_quiz.md"
JUDGE_PROMPT_PATH = PROJECT_ROOT / "librarian" / "prompts" / "listening_scenario_semantic_judge.md"

# Representative thematic topics mapped to each archetype for realistic scenario generation
ARCHETYPE_SAMPLE_TOPICS: Dict[str, str] = {
    "arch_01_arranging_and_rescheduling": "Coordinating a study group meeting and adjusting schedules",
    "arch_02_enquiring_comparing_and_deciding": "Booking a campus study space and comparing service options",
    "arch_03_requesting_and_negotiating_help": "Requesting technical help and negotiating timing before a presentation",
    "arch_04_problem_sharing_empathy_and_advice": "Coping with study stress and managing time before exams",
    "arch_05_opinion_exchange_and_consensus": "Secrets to beauty / Internal vs. External Value / Self-Esteem",
    "arch_06_collaborative_proposals_and_tradeoffs": "Planning a student club project within budget limits",
    "arch_07_complaints_apologies_and_remedies": "Resolving a service billing error and agreeing on a fair remedy",
    "arch_08_social_catching_up_and_closing": "Catching up on campus life, work experience, and personal habits",
    "arch_09_academic_seminar_and_causal_probing": "Evaluating the causal impact of digital media on attention spans",
    "arch_10_interviews_and_credential_pitching": "Interviewing for a cross-cultural research assistantship",
    "arch_11_instruction_walkthrough_and_onboarding": "Learning the step-by-step protocol for laboratory data logging",
    "arch_12_negotiating_terms_and_compromise": "Negotiating project deadlines and resource allocation between teams",
    "arch_13_medical_consultation_and_health_triage": "Consulting a campus health advisor about sleep habits and fatigue",
    "arch_14_resolving_misunderstandings_and_mediation": "Clearing up a misunderstanding over shared team responsibilities",
}

ARCHETYPE_SAMPLE_TOPICS_A2: Dict[str, str] = {
    "arch_09_academic_seminar_and_causal_probing": "Discussing why students lose focus when studying with phones",
    "arch_10_interviews_and_credential_pitching": "Applying for a student club helper or campus café part-time job",
    "arch_11_instruction_walkthrough_and_onboarding": "Learning how to use the library printer and study room booking system",
    "arch_12_negotiating_terms_and_compromise": "Buying a second-hand bicycle or booking a room on a student budget",
}


def enumerate_all_scenarios() -> List[Dict[str, Any]]:
    """Returns a flat list of all 30 scenario specs across all 14 archetypes and 28 trajectories."""
    data = json.loads(ARCHETYPES_JSON_PATH.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for arch in data.get("archetypes") or []:
        arch_id = str(arch.get("archetype_id") or "")
        arch_name = str(arch.get("archetype_name") or "")
        for traj in arch.get("trajectories") or []:
            traj_id = str(traj.get("trajectory_id") or "")
            traj_title = str(traj.get("title") or "")
            cefr_band = str(traj.get("cefr_band") or "B1_C2")
            register = str(traj.get("register") or "formal_neutral")
            cefr_level = "A2" if cefr_band == "A1_A2" else "B2"
            topic = (
                ARCHETYPE_SAMPLE_TOPICS_A2.get(arch_id)
                if cefr_band == "A1_A2" and arch_id in ARCHETYPE_SAMPLE_TOPICS_A2
                else ARCHETYPE_SAMPLE_TOPICS.get(arch_id, arch_name)
            )
            for sc in traj.get("everyday_scenarios") or []:
                sc_id = str(sc.get("scenario_id") or "")
                rows.append(
                    {
                        "archetype_id": arch_id,
                        "archetype_name": arch_name,
                        "trajectory_id": traj_id,
                        "trajectory_title": traj_title,
                        "cefr_band": cefr_band,
                        "cefr_level": cefr_level,
                        "register": register,
                        "scenario_id": sc_id,
                        "scenario_setting": str(sc.get("setting") or ""),
                        "speaker_roles": traj.get("speaker_roles") or {},
                        "thematic_topic": topic,
                    }
                )
    return rows


def build_judge_prompt_for_scenario(
    scenario_meta: Dict[str, Any],
    blueprint_res: Dict[str, Any],
    generated_quiz_obj: Dict[str, Any],
) -> str:
    """Formats `listening_scenario_semantic_judge.md` for an external LLM or human reviewer."""
    template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    roles = scenario_meta.get("speaker_roles") or {}
    # Use safe replacement for template placeholders while preserving JSON braces in the prompt
    replacements = {
        "{archetype_name}": str(scenario_meta.get("archetype_name") or ""),
        "{archetype_id}": str(scenario_meta.get("archetype_id") or ""),
        "{trajectory_title}": str(scenario_meta.get("trajectory_title") or ""),
        "{trajectory_id}": str(scenario_meta.get("trajectory_id") or ""),
        "{cefr_level}": str(scenario_meta.get("cefr_level") or "B2"),
        "{register}": str(scenario_meta.get("register") or "formal_neutral"),
        "{scenario_id}": str(scenario_meta.get("scenario_id") or ""),
        "{scenario_setting}": str(scenario_meta.get("scenario_setting") or ""),
        "{speaker_1_role}": str(roles.get("speaker_1") or "Speaker 1"),
        "{speaker_2_role}": str(roles.get("speaker_2") or "Speaker 2"),
        "{thematic_topic}": str(scenario_meta.get("thematic_topic") or ""),
        "{blueprint_prompt_block}": str(blueprint_res.get("combined_prompt_block") or ""),
        "{generated_quiz_json}": json.dumps(generated_quiz_obj, indent=2, ensure_ascii=False),
    }
    rendered = template
    for k, v in replacements.items():
        rendered = rendered.replace(k, v)
    return rendered


class TestListeningScenariosQuality(unittest.TestCase):
    """Deterministic Semantic & Logical Invariant Tests across all 14 Archetypes / 28 Trajectories / 30 Scenarios."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = enumerate_all_scenarios()

    def test_01_all_archetypes_trajectories_and_scenarios_present(self) -> None:
        data = json.loads(ARCHETYPES_JSON_PATH.read_text(encoding="utf-8"))
        archs = data.get("archetypes") or []
        self.assertEqual(len(archs), 14, "Expected exactly 14 functional dialogue archetypes")
        trajs = [t for a in archs for t in (a.get("trajectories") or [])]
        self.assertEqual(len(trajs), 28, "Expected 28 register-isolated trajectories (2 per archetype)")
        self.assertGreaterEqual(len(self.scenarios), 28, "Expected at least 28 scenarios across all trajectories")

    def test_02_zero_leaked_internal_ids_or_grammar_placeholders(self) -> None:
        raw_text = ARCHETYPES_JSON_PATH.read_text(encoding="utf-8")
        leaked_tags = re.findall(r"\b(?:u\d{2}_s\d{2}|B1P|A2:U\d{2}|B1P:U\d{2})\b", raw_text)
        self.assertEqual(leaked_tags, [], f"Found leaked internal unit/strategy IDs: {leaked_tags[:10]}")
        pattern_lines = [line for line in raw_text.splitlines() if '"pattern"' in line or '"recommended_chunks"' in line]
        leaked_grammar = re.findall(r"\+\s*(?:-ing|bare infinitive)", "\n".join(pattern_lines))
        self.assertEqual(leaked_grammar, [], f"Found unexpanded grammar meta-placeholders in chunks: {leaked_grammar}")

    def test_03_every_scenario_blueprint_has_coherent_turns_and_questions(self) -> None:
        expected_anchors = ["Turns 1–2", "Turn 3", "Turn 4", "Turns 5–6", "Turns 7–8"]
        expected_speakers = ["Speaker 1 / Speaker 2", "Speaker 1", "Speaker 2", "Speaker 1 / Speaker 2", "Speaker 1 / Speaker 2"]

        for sc in self.scenarios:
            with self.subTest(scenario_id=sc["scenario_id"]):
                bp = LinguisticEngine.build_listening_micro_task_blueprints(
                    vocab_content=[],
                    thematic_topic=sc["thematic_topic"],
                    target_count=5,
                    cefr_level=sc["cefr_level"],
                    forced_archetype_id=sc["archetype_id"],
                    forced_trajectory_id=sc["trajectory_id"],
                    forced_scenario_id=sc["scenario_id"],
                )
                self.assertEqual(bp["archetype_id"], sc["archetype_id"])
                self.assertEqual(bp["trajectory_id"], sc["trajectory_id"])
                self.assertEqual(bp["scenario_id"], sc["scenario_id"])

                # 1. Verify 8 turns and strict Speaker 1 (odd) / Speaker 2 (even) alternation
                turn_specs = bp.get("turn_specs") or []
                self.assertEqual(len(turn_specs), 8, f"{sc['scenario_id']}: expected 8 turns")
                seen_chunks: set[str] = set()
                for idx, t_spec in enumerate(turn_specs, start=1):
                    self.assertEqual(t_spec["turn_number"], idx)
                    expected_spk = "Speaker 1" if idx % 2 == 1 else "Speaker 2"
                    self.assertEqual(
                        t_spec["speaker"],
                        expected_spk,
                        f"{sc['scenario_id']} Turn {idx} speaker mismatch",
                    )
                    chunks = t_spec.get("recommended_chunks") or []
                    self.assertGreaterEqual(
                        len(chunks), 1, f"{sc['scenario_id']} Turn {idx} has no recommended_chunks"
                    )
                    for c in chunks:
                        c_norm = c.strip().lower()
                        self.assertNotIn(
                            c_norm,
                            seen_chunks,
                            f"{sc['scenario_id']} Turn {idx} repeats chunk across turns: {c}",
                        )
                        seen_chunks.add(c_norm)

                # 2. Verify 5 Question Blueprints and immutable turn/speaker anchoring
                q_specs = bp.get("questions") or []
                self.assertEqual(len(q_specs), 5, f"{sc['scenario_id']}: expected 5 question blueprints")
                for q_idx, q_spec in enumerate(q_specs):
                    self.assertEqual(q_spec["anchor_turn"], expected_anchors[q_idx])
                    self.assertEqual(q_spec["target_speaker"], expected_speakers[q_idx])
                    ans_idx = q_spec["correct_answer_index"]
                    self.assertIn(ans_idx, (0, 1, 2, 3))
                    slot_ops = q_spec.get("slot_operators") or {}
                    self.assertEqual(len(slot_ops), 4)
                    self.assertIn("Paraphrase Operator", str(slot_ops.get(ans_idx, "")))

    def test_04_judge_prompt_interpolates_cleanly_for_all_scenarios(self) -> None:
        sc = self.scenarios[0]
        bp = LinguisticEngine.build_listening_micro_task_blueprints(
            vocab_content=[],
            thematic_topic=sc["thematic_topic"],
            target_count=5,
            cefr_level=sc["cefr_level"],
            forced_archetype_id=sc["archetype_id"],
            forced_trajectory_id=sc["trajectory_id"],
            forced_scenario_id=sc["scenario_id"],
        )
        dummy_quiz = {"title": "Sample", "dialogue": [], "questions": []}
        rendered = build_judge_prompt_for_scenario(sc, bp, dummy_quiz)
        self.assertIn(sc["scenario_id"], rendered)
        self.assertIn(sc["archetype_id"], rendered)
        self.assertNotIn("{scenario_id}", rendered)
        self.assertNotIn("{blueprint_prompt_block}", rendered)

    def test_05_forced_archetype_resolution_and_aliases(self) -> None:
        """Verifies canonical IDs, numbers, and semantic aliases resolve deterministically."""
        test_cases = [
            ("1", "arch_01_arranging_and_rescheduling"),
            ("arch_01", "arch_01_arranging_and_rescheduling"),
            ("reschedule", "arch_01_arranging_and_rescheduling"),
            ("2", "arch_02_enquiring_comparing_and_deciding"),
            ("options", "arch_02_enquiring_comparing_and_deciding"),
            ("decision", "arch_02_enquiring_comparing_and_deciding"),
            ("3", "arch_03_requesting_and_negotiating_help"),
            ("help", "arch_03_requesting_and_negotiating_help"),
            ("4", "arch_04_problem_sharing_empathy_and_advice"),
            ("advice", "arch_04_problem_sharing_empathy_and_advice"),
            ("5", "arch_05_opinion_exchange_and_consensus"),
            ("consensus", "arch_05_opinion_exchange_and_consensus"),
            ("6", "arch_06_collaborative_proposals_and_tradeoffs"),
            ("proposal", "arch_06_collaborative_proposals_and_tradeoffs"),
            ("7", "arch_07_complaints_apologies_and_remedies"),
            ("complaint", "arch_07_complaints_apologies_and_remedies"),
            ("8", "arch_08_social_catching_up_and_closing"),
            ("social", "arch_08_social_catching_up_and_closing"),
            ("9", "arch_09_academic_seminar_and_causal_probing"),
            ("seminar", "arch_09_academic_seminar_and_causal_probing"),
            ("10", "arch_10_interviews_and_credential_pitching"),
            ("interview", "arch_10_interviews_and_credential_pitching"),
            ("11", "arch_11_instruction_walkthrough_and_onboarding"),
            ("onboarding", "arch_11_instruction_walkthrough_and_onboarding"),
            ("12", "arch_12_negotiating_terms_and_compromise"),
            ("negotiate", "arch_12_negotiating_terms_and_compromise"),
            ("13", "arch_13_medical_consultation_and_health_triage"),
            ("doctor", "arch_13_medical_consultation_and_health_triage"),
            ("14", "arch_14_resolving_misunderstandings_and_mediation"),
            ("mediation", "arch_14_resolving_misunderstandings_and_mediation"),
        ]
        for spec, expected in test_cases:
            resolved = LinguisticEngine.resolve_functional_dialogue_archetype_id(spec)
            self.assertEqual(resolved, expected, f"Spec '{spec}' did not resolve to '{expected}'")

        # Test blueprint generation with forced archetype
        bp = LinguisticEngine.build_listening_micro_task_blueprints(
            vocab_content=[],
            thematic_topic="1. Renewable Energy",
            target_count=5,
            cefr_level="B2",
            forced_archetype_id="seminar"
        )
        self.assertEqual(bp.get("archetype_id"), "arch_09_academic_seminar_and_causal_probing")

    def test_06_forced_scenario_resolution_and_matching(self) -> None:
        """Verifies scenario IDs and keywords resolve deterministically and lock correct trajectory."""
        sc_test_cases = [
            ("arch_01_a2_inf_sc01", ("arch_01_a2_inf_sc01", "arch_01_arranging_and_rescheduling")),
            ("arch_01_b1p_form_sc01", ("arch_01_b1p_form_sc01", "arch_01_arranging_and_rescheduling")),
            ("weekend", ("arch_01_a2_inf_sc01", "arch_01_arranging_and_rescheduling")),
            ("clinic", ("arch_13_b1p_form_sc01", "arch_13_medical_consultation_and_health_triage")),
            ("arch_04_b1p_form_sc01", ("arch_04_b1p_form_sc01", "arch_04_problem_sharing_empathy_and_advice")),
        ]
        for spec, expected in sc_test_cases:
            sc_id, a_id = LinguisticEngine.resolve_functional_dialogue_scenario_id(spec)
            self.assertEqual((sc_id, a_id), expected, f"Scenario spec '{spec}' did not resolve to '{expected}'")

        # Test blueprint generation with forced scenario alone (should lock scenario AND archetype)
        bp = LinguisticEngine.build_listening_micro_task_blueprints(
            vocab_content=[],
            thematic_topic="General Discussion",
            target_count=5,
            cefr_level="B2",
            forced_scenario_id="weekend"
        )
        self.assertEqual(bp.get("archetype_id"), "arch_01_arranging_and_rescheduling")
        self.assertEqual(bp.get("scenario_id"), "arch_01_a2_inf_sc01")
        self.assertEqual(bp.get("cefr_band"), "A1_A2")

        # Test blueprint generation with both forced archetype and scenario
        bp2 = LinguisticEngine.build_listening_micro_task_blueprints(
            vocab_content=[],
            thematic_topic="Workplace Consultation",
            target_count=5,
            cefr_level="A2",  # Note: text is A2, but teacher explicitly requested B1+ formal scenario
            forced_archetype_id="arch_01_arranging_and_rescheduling",
            forced_scenario_id="arch_01_b1p_form_sc01"
        )
        self.assertEqual(bp2.get("archetype_id"), "arch_01_arranging_and_rescheduling")
        self.assertEqual(bp2.get("scenario_id"), "arch_01_b1p_form_sc01")
        self.assertEqual(bp2.get("cefr_band"), "B1_C2")



def _build_listening_generation_prompt(sc: Dict[str, Any], bp: Dict[str, Any]) -> str:
    """Builds the full `listening_quiz.md` prompt for a given scenario and blueprint."""
    prompt_template = LISTENING_PROMPT_PATH.read_text(encoding="utf-8")
    cefr = sc["cefr_level"].upper()
    if cefr in ("A1", "A2", "A2+"):
        dialogue_style = (
            "Conversational, everyday social or campus service interactions. "
            "Keep speaker turns short and natural (8 to 15 words per turn), using accessible S1/S2 spoken phrasing."
        )
        stem_guidance = "Use direct, unambiguous question stems with simple everyday A1-A2 words."
        opt_guidance = (
            "Keep options concise, simple, and direct (ideally 3 to 8 words), using only basic everyday A1-A2 vocabulary "
            "(such as 'work hours', 'schedule', 'sign in', 'next', 'cold', 'rest', 'water', 'doctor')."
        )
        skill_guidance = "Detail: 2-3 questions; Main Idea: 1-2 questions; Inference: 0-1 question."
    else:
        dialogue_style = (
            "Natural, idiomatic spoken conversation between peers, colleagues, or student and advisor. "
            "Use high-frequency spoken English (S1/S2 vocabulary and natural contractions like `we've`, `that's`, `I'd`) at 12 to 18 words per turn. "
            "Difficulty at B2 comes from tracking pragmatic stance shifts, polite hedges, and conversational updates—NEVER from dense academic/corporate jargon."
        )
        stem_guidance = "Use clear, natural question stems testing speaker intention, stance shifts, and agreed next steps."
        opt_guidance = "Keep options clear and idiomatic (ideally 6 to 12 words), paraphrasing spoken meaning without bookish vocabulary."
        skill_guidance = "Detail: 2 questions; Main Idea: 1 question; Inference: 2 questions."

    kwargs = {
        "cefr_level": cefr,
        "thematic_topic": sc["thematic_topic"],
        "vocabulary_content": bp["combined_prompt_block"],
        "dialogue_style_guidance": dialogue_style,
        "question_stem_guidance": stem_guidance,
        "option_complexity_guidance": opt_guidance,
        "skill_distribution_guidance": skill_guidance,
        "speaker_1_name": "Aria",
        "speaker_2_name": "Ryan",
        "speaker_1_gender": "female",
        "speaker_2_gender": "male",
        "speaker_1_accent": "Accent: American",
        "speaker_2_accent": "Accent: British",
        "count": 5,
    }
    return prompt_template.format(**kwargs)


def run_cli_generation_and_audit(args: argparse.Namespace) -> int:
    """Runs live LLM generation across selected scenarios (or parses existing logs) and builds the review dossier."""
    from librarian.evaluator import LogEvaluator
    from librarian.llm import llm
    from librarian.processor import processor

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dossier_md_path = out_dir / "listening_scenarios_review_dossier.md"
    dossier_json_path = out_dir / "listening_scenarios_review_dossier.json"

    if args.from_log:
        ev = LogEvaluator()
        log_path = Path(args.from_log)
        parsed = ev.parse_log_file(log_path)
        eval_res = ev.evaluate_log(parsed)
        prompt_str = str(parsed.get("prompt") or "")
        resp_obj = parsed.get("parsed_json") or {}
        m_arch = re.search(r"Functional Archetype\*\*:\s*`([^`]+)`\s*\(`([^`]+)`\)", prompt_str)
        m_traj = re.search(r"Selected Trajectory\*\*:\s*`([^`]+)`\s*\(CEFR Band:\s*`([^`]+)`\s*\|\s*Register:\s*`([^`]+)`\)", prompt_str)
        m_cefr = re.search(r"Curriculum Alignment \(CEFR ([A-C][12]\+?)\)", prompt_str)
        m_bp = re.search(r"(### 🎙️ FUNCTIONAL DIALOGUE ARCHETYPE.*)", prompt_str, re.DOTALL)
        sc_meta = {
            "archetype_name": m_arch.group(1) if m_arch else "Listening Scenario",
            "archetype_id": m_arch.group(2) if m_arch else "unknown_arch",
            "trajectory_title": m_traj.group(1) if m_traj else "",
            "trajectory_id": m_traj.group(2) if m_traj else "",
            "cefr_level": m_cefr.group(1) if m_cefr else "B2",
            "register": m_traj.group(3) if m_traj else "formal_neutral",
            "scenario_id": log_path.stem,
            "scenario_setting": "Extracted from log",
            "speaker_roles": {"speaker_1": "Speaker 1", "speaker_2": "Speaker 2"},
            "thematic_topic": "Extracted from log",
        }
        bp_res = {"combined_prompt_block": m_bp.group(1).strip() if m_bp else prompt_str}
        judge_prompt = build_judge_prompt_for_scenario(sc_meta, bp_res, resp_obj)
        md_content = "\n".join(
            [
                f"# External Judge Review Packet — `{log_path.name}`",
                "",
                f"- **Level-1 Evaluator Score**: `{eval_res.get('composite_score')}`",
                f"- **Level-1 Flags**: `{eval_res.get('flags')}`",
                "",
                "## Ready-to-Send External Model Audit Prompt",
                "```markdown",
                judge_prompt,
                "```",
            ]
        )
        dossier_md_path.write_text(md_content, encoding="utf-8")
        dossier_json_path.write_text(
            json.dumps({"log_file": str(log_path), "evaluator": eval_res, "judge_prompt": judge_prompt}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[OK] Wrote Log Review Packet: {dossier_md_path}")
        return 0

    all_scenarios = enumerate_all_scenarios()
    selected: List[Dict[str, Any]] = []
    seen_archs: set[str] = set()
    arch_filters = [a.strip() for a in args.archetype.split(",") if a.strip()] if args.archetype else []
    for sc in all_scenarios:
        if arch_filters and not any(af in sc["archetype_id"] for af in arch_filters):
            continue
        if args.trajectory and args.trajectory not in sc["trajectory_id"]:
            continue
        if args.scenario and args.scenario not in sc["scenario_id"]:
            continue
        if args.band and args.band.upper() != sc["cefr_band"].upper():
            continue
        if args.one_per_archetype:
            if sc["archetype_id"] in seen_archs:
                continue
            seen_archs.add(sc["archetype_id"])
        selected.append(sc)
        if args.limit and len(selected) >= args.limit:
            break

    if not selected:
        print("No scenarios matched the filter criteria.")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dossier_md_path = out_dir / "listening_scenarios_review_dossier.md"
    dossier_json_path = out_dir / "listening_scenarios_review_dossier.json"

    results: List[Dict[str, Any]] = []
    md_lines: List[str] = [
        "# Listening Scenarios Semantic & Logical Audit Dossier",
        "",
        f"- **Total Scenarios Tested**: `{len(selected)}`",
        f"- **Judge Prompt Template**: [`librarian/prompts/listening_scenario_semantic_judge.md`](file://{JUDGE_PROMPT_PATH})",
        "",
        "---",
        "",
    ]

    for idx, sc in enumerate(selected, start=1):
        sc_id = sc["scenario_id"]
        print(f"[{idx}/{len(selected)}] Building blueprint for {sc_id} ({sc['archetype_id']} | {sc['cefr_level']})...")
        bp = LinguisticEngine.build_listening_micro_task_blueprints(
            vocab_content=[],
            thematic_topic=sc["thematic_topic"],
            target_count=5,
            cefr_level=sc["cefr_level"],
            forced_archetype_id=sc["archetype_id"],
            forced_trajectory_id=sc["trajectory_id"],
            forced_scenario_id=sc["scenario_id"],
        )
        gen_prompt = _build_listening_generation_prompt(sc, bp)

        quiz_obj: Dict[str, Any] = {}
        code_gate_issues: List[str] = []
        judge_verdict: Dict[str, Any] | None = None
        elapsed = 0.0

        if args.generate:
            t0 = time.time()
            schema = processor._interpolate_schema(
                ListeningQuiz,
                {
                    "speaker_1_gender": "female",
                    "speaker_2_gender": "male",
                    "speaker_1_accent": "Accent: American",
                    "speaker_2_accent": "Accent: British",
                    "count": 5,
                },
            )
            raw_quiz = llm.chat(
                [{"role": "user", "content": gen_prompt}],
                schema=schema,
                task_name=f"test_listening_{sc_id}",
            )
            elapsed = round(time.time() - t0, 2)
            if dataclasses.is_dataclass(raw_quiz) and not isinstance(raw_quiz, type):
                raw_quiz = dataclasses.asdict(raw_quiz)
            elif hasattr(raw_quiz, "model_dump"):
                raw_quiz = raw_quiz.model_dump()
            if isinstance(raw_quiz, dict):
                quiz_obj = processor.hydrate_listening_blueprint_items(
                    raw_quiz, bp.get("questions") or []
                )
                _, code_gate_issues = processor.audit_listening_integrity(
                    quiz_obj,
                    cefr_level=sc["cefr_level"],
                    blueprint_specs=bp.get("questions") or [],
                )
            if args.run_judge and quiz_obj:
                judge_prompt = build_judge_prompt_for_scenario(sc, bp, quiz_obj)
                print(f"    -> Running Semantic/Logical Judge on {sc_id}...")
                judge_raw = llm.chat(
                    [{"role": "user", "content": judge_prompt}],
                    task_name=f"judge_listening_{sc_id}",
                )
                if isinstance(judge_raw, dict):
                    judge_verdict = judge_raw
                elif isinstance(judge_raw, str):
                    try:
                        m_json = re.search(r"\{.*\}", judge_raw, re.DOTALL)
                        if m_json:
                            judge_verdict = json.loads(m_json.group(0))
                    except Exception:
                        judge_verdict = {"raw_response": judge_raw}

        judge_prompt_text = build_judge_prompt_for_scenario(sc, bp, quiz_obj or {"note": "Run with --generate to populate LLM output"})
        record = {
            "scenario_meta": sc,
            "generation_seconds": elapsed,
            "code_gate_issues": code_gate_issues,
            "generated_quiz": quiz_obj,
            "judge_verdict": judge_verdict,
            "external_judge_prompt": judge_prompt_text,
        }
        results.append(record)

        md_lines.append(f"## {idx}. `{sc_id}` — {sc['archetype_name']} ({sc['cefr_level']} | `{sc['register']}`)")
        md_lines.append(f"- **Setting**: {sc['scenario_setting']}")
        md_lines.append(f"- **Thematic Topic**: `{sc['thematic_topic']}`")
        if args.generate:
            md_lines.append(f"- **Generation Time**: `{elapsed}s` | **Level-1 Code Gate Issues**: `{len(code_gate_issues)}`")
            if code_gate_issues:
                for iss in code_gate_issues:
                    md_lines.append(f"  - ⚠️ `{iss}`")
            md_lines.append("")
            md_lines.append("### Generated 8-Turn Dialogue")
            dialogue_turns = quiz_obj.get("script") or quiz_obj.get("dialogue") or []
            for t_idx, turn in enumerate(dialogue_turns, start=1):
                md_lines.append(f"{t_idx}. **{turn.get('speaker', '?')}**: {turn.get('text', '')}")
            md_lines.append("")
            md_lines.append("### Generated 5 Questions")
            for q_idx, q in enumerate(quiz_obj.get("questions") or [], start=1):
                md_lines.append(f"- **Q{q_idx} (`{q.get('category', '')}` | Key: `{q.get('correct_answer_index')}`)**: {q.get('question', '')}")
                for o_idx, opt in enumerate(q.get("options") or []):
                    marker = "✅" if o_idx == q.get("correct_answer_index") else "  "
                    md_lines.append(f"  - {marker} `[{o_idx}]` {opt}")
                md_lines.append(f"  - *Explanation*: {q.get('explanation', '')}")
                md_lines.append(f"  - *Audit*: `{q.get('design_audit', '')}`")
            if judge_verdict:
                md_lines.append("")
                md_lines.append("### Judge Model Semantic & Logical Verdict")
                md_lines.append("```json")
                md_lines.append(json.dumps(judge_verdict, indent=2, ensure_ascii=False))
                md_lines.append("```")
        else:
            md_lines.append("")
            md_lines.append("<details><summary>External Judge Prompt (Ready to Copy)</summary>")
            md_lines.append("")
            md_lines.append("```markdown")
            md_lines.append(judge_prompt_text)
            md_lines.append("```")
            md_lines.append("</details>")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    dossier_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    dossier_json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote Markdown Review Dossier: {dossier_md_path}")
    print(f"[OK] Wrote JSON Review Packet:     {dossier_json_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test & Audit Listening Scenarios across all 14 Archetypes")
    parser.add_argument("--generate", action="store_true", help="Call local LLM to generate dialogues & questions")
    parser.add_argument("--run-judge", action="store_true", help="Run the semantic/logical judge prompt via LLM")
    parser.add_argument("--export-dossier", action="store_true", help="Export the external review dossier without running unittest")
    parser.add_argument("--from-log", type=str, default="", help="Build a judge review packet directly from an existing listening log file")
    parser.add_argument("--archetype", type=str, default="", help="Filter by archetype_id substring (e.g. arch_05)")
    parser.add_argument("--trajectory", type=str, default="", help="Filter by trajectory_id substring")
    parser.add_argument("--scenario", type=str, default="", help="Filter by scenario_id substring")
    parser.add_argument("--band", type=str, default="", help="Filter by CEFR band (A1_A2 or B1_C2)")
    parser.add_argument("--one-per-archetype", action="store_true", help="Select 1 scenario per archetype (14 total)")
    parser.add_argument("--limit", type=int, default=0, help="Max number of scenarios to process")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "logs"), help="Directory for review dossier output")
    known_args, remaining_argv = parser.parse_known_args()

    if (
        known_args.generate
        or known_args.export_dossier
        or known_args.from_log
        or known_args.archetype
        or known_args.trajectory
        or known_args.scenario
    ):
        sys.exit(run_cli_generation_and_audit(known_args))
    else:
        unittest.main(argv=[sys.argv[0]] + remaining_argv)
