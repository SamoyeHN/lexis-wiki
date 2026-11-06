"""
Functional dialogue listening archetypes, trajectory selection, and micro-task blueprints.
"""

import datetime
import hashlib
import json
import os
import re
import sqlite3
import threading
import zlib
from contextlib import contextmanager
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class QuizListeningMixin:
    _FUNCTIONAL_ARCHETYPES_CACHE: Optional[Dict[str, Any]] = None

    ARCHETYPE_ALIASES: Dict[str, str] = {
        # Number / code shortcuts
        "1": "arch_01_arranging_and_rescheduling",
        "01": "arch_01_arranging_and_rescheduling",
        "arch_01": "arch_01_arranging_and_rescheduling",
        "arch1": "arch_01_arranging_and_rescheduling",
        "2": "arch_02_enquiring_comparing_and_deciding",
        "02": "arch_02_enquiring_comparing_and_deciding",
        "arch_02": "arch_02_enquiring_comparing_and_deciding",
        "arch2": "arch_02_enquiring_comparing_and_deciding",
        "3": "arch_03_requesting_and_negotiating_help",
        "03": "arch_03_requesting_and_negotiating_help",
        "arch_03": "arch_03_requesting_and_negotiating_help",
        "arch3": "arch_03_requesting_and_negotiating_help",
        "4": "arch_04_problem_sharing_empathy_and_advice",
        "04": "arch_04_problem_sharing_empathy_and_advice",
        "arch_04": "arch_04_problem_sharing_empathy_and_advice",
        "arch4": "arch_04_problem_sharing_empathy_and_advice",
        "5": "arch_05_opinion_exchange_and_consensus",
        "05": "arch_05_opinion_exchange_and_consensus",
        "arch_05": "arch_05_opinion_exchange_and_consensus",
        "arch5": "arch_05_opinion_exchange_and_consensus",
        "6": "arch_06_collaborative_proposals_and_tradeoffs",
        "06": "arch_06_collaborative_proposals_and_tradeoffs",
        "arch_06": "arch_06_collaborative_proposals_and_tradeoffs",
        "arch6": "arch_06_collaborative_proposals_and_tradeoffs",
        "7": "arch_07_complaints_apologies_and_remedies",
        "07": "arch_07_complaints_apologies_and_remedies",
        "arch_07": "arch_07_complaints_apologies_and_remedies",
        "arch7": "arch_07_complaints_apologies_and_remedies",
        "8": "arch_08_social_catching_up_and_closing",
        "08": "arch_08_social_catching_up_and_closing",
        "arch_08": "arch_08_social_catching_up_and_closing",
        "arch8": "arch_08_social_catching_up_and_closing",
        "9": "arch_09_academic_seminar_and_causal_probing",
        "09": "arch_09_academic_seminar_and_causal_probing",
        "arch_09": "arch_09_academic_seminar_and_causal_probing",
        "arch9": "arch_09_academic_seminar_and_causal_probing",
        "10": "arch_10_interviews_and_credential_pitching",
        "arch_10": "arch_10_interviews_and_credential_pitching",
        "arch10": "arch_10_interviews_and_credential_pitching",
        "11": "arch_11_instruction_walkthrough_and_onboarding",
        "arch_11": "arch_11_instruction_walkthrough_and_onboarding",
        "arch11": "arch_11_instruction_walkthrough_and_onboarding",
        "12": "arch_12_negotiating_terms_and_compromise",
        "arch_12": "arch_12_negotiating_terms_and_compromise",
        "arch12": "arch_12_negotiating_terms_and_compromise",
        "13": "arch_13_medical_consultation_and_health_triage",
        "arch_13": "arch_13_medical_consultation_and_health_triage",
        "arch13": "arch_13_medical_consultation_and_health_triage",
        "14": "arch_14_resolving_misunderstandings_and_mediation",
        "arch_14": "arch_14_resolving_misunderstandings_and_mediation",
        "arch14": "arch_14_resolving_misunderstandings_and_mediation",

        # Pedagogical / thematic aliases
        "arranging": "arch_01_arranging_and_rescheduling",
        "reschedule": "arch_01_arranging_and_rescheduling",
        "rescheduling": "arch_01_arranging_and_rescheduling",
        "appointment": "arch_01_arranging_and_rescheduling",

        "enquiry": "arch_02_enquiring_comparing_and_deciding",
        "enquiring": "arch_02_enquiring_comparing_and_deciding",
        "deciding": "arch_02_enquiring_comparing_and_deciding",
        "decision": "arch_02_enquiring_comparing_and_deciding",
        "options": "arch_02_enquiring_comparing_and_deciding",
        "compare": "arch_02_enquiring_comparing_and_deciding",

        "request": "arch_03_requesting_and_negotiating_help",
        "requesting": "arch_03_requesting_and_negotiating_help",
        "help": "arch_03_requesting_and_negotiating_help",

        "problem": "arch_04_problem_sharing_empathy_and_advice",
        "problems": "arch_04_problem_sharing_empathy_and_advice",
        "advice": "arch_04_problem_sharing_empathy_and_advice",
        "empathy": "arch_04_problem_sharing_empathy_and_advice",
        "trouble": "arch_04_problem_sharing_empathy_and_advice",

        "opinion": "arch_05_opinion_exchange_and_consensus",
        "opinions": "arch_05_opinion_exchange_and_consensus",
        "consensus": "arch_05_opinion_exchange_and_consensus",
        "debate": "arch_05_opinion_exchange_and_consensus",
        "views": "arch_05_opinion_exchange_and_consensus",

        "proposal": "arch_06_collaborative_proposals_and_tradeoffs",
        "proposals": "arch_06_collaborative_proposals_and_tradeoffs",
        "collaborative": "arch_06_collaborative_proposals_and_tradeoffs",
        "collaboration": "arch_06_collaborative_proposals_and_tradeoffs",
        "planning": "arch_06_collaborative_proposals_and_tradeoffs",

        "complaint": "arch_07_complaints_apologies_and_remedies",
        "complaints": "arch_07_complaints_apologies_and_remedies",
        "remedy": "arch_07_complaints_apologies_and_remedies",
        "remedies": "arch_07_complaints_apologies_and_remedies",
        "apology": "arch_07_complaints_apologies_and_remedies",
        "feedback": "arch_07_complaints_apologies_and_remedies",

        "social": "arch_08_social_catching_up_and_closing",
        "catching_up": "arch_08_social_catching_up_and_closing",
        "icebreak": "arch_08_social_catching_up_and_closing",
        "ice_breaking": "arch_08_social_catching_up_and_closing",
        "closing": "arch_08_social_catching_up_and_closing",

        "academic": "arch_09_academic_seminar_and_causal_probing",
        "seminar": "arch_09_academic_seminar_and_causal_probing",
        "causal": "arch_09_academic_seminar_and_causal_probing",
        "probing": "arch_09_academic_seminar_and_causal_probing",
        "hypothesis": "arch_09_academic_seminar_and_causal_probing",

        "interview": "arch_10_interviews_and_credential_pitching",
        "interviews": "arch_10_interviews_and_credential_pitching",
        "pitch": "arch_10_interviews_and_credential_pitching",
        "pitching": "arch_10_interviews_and_credential_pitching",

        "instruction": "arch_11_instruction_walkthrough_and_onboarding",
        "onboarding": "arch_11_instruction_walkthrough_and_onboarding",
        "walkthrough": "arch_11_instruction_walkthrough_and_onboarding",

        "negotiate": "arch_12_negotiating_terms_and_compromise",
        "negotiation": "arch_12_negotiating_terms_and_compromise",
        "bargain": "arch_12_negotiating_terms_and_compromise",
        "compromise": "arch_12_negotiating_terms_and_compromise",

        "medical": "arch_13_medical_consultation_and_health_triage",
        "health": "arch_13_medical_consultation_and_health_triage",
        "clinic": "arch_13_medical_consultation_and_health_triage",
        "doctor": "arch_13_medical_consultation_and_health_triage",
        "symptoms": "arch_13_medical_consultation_and_health_triage",

        "misunderstanding": "arch_14_resolving_misunderstandings_and_mediation",
        "mediation": "arch_14_resolving_misunderstandings_and_mediation",
        "clarification": "arch_14_resolving_misunderstandings_and_mediation",
    }

    @classmethod
    def resolve_functional_dialogue_archetype_id(cls, spec: str) -> str:
        """
        Normalizes an archetype specification into a canonical archetype_id
        (e.g., '4' -> 'arch_04_problem_sharing_empathy_and_advice',
               'advice' -> 'arch_04_problem_sharing_empathy_and_advice',
               'arch_04' -> 'arch_04_problem_sharing_empathy_and_advice').
        Returns empty string if spec is empty or cannot be resolved.
        """
        clean = str(spec or "").strip().lower()
        if not clean:
            return ""
        config = cls._load_functional_dialogue_archetypes()
        all_ids = {
            str(a.get("archetype_id")) for a in (config.get("archetypes") or []) if a.get("archetype_id")
        }
        if clean in all_ids:
            return clean
        if clean in cls.ARCHETYPE_ALIASES:
            return cls.ARCHETYPE_ALIASES[clean]
        for alias, aid in cls.ARCHETYPE_ALIASES.items():
            if clean == alias or clean.startswith(alias):
                return aid
        for aid in all_ids:
            if clean in aid:
                return aid
        return ""

    @classmethod
    def resolve_functional_dialogue_scenario_id(
        cls, spec: str, forced_archetype_id: str = ""
    ) -> Tuple[str, str]:
        """
        Normalizes a scenario specification into (canonical_scenario_id, canonical_archetype_id)
        (e.g., 'arch_01_a2_inf_sc01' -> ('arch_01_a2_inf_sc01', 'arch_01_arranging_and_rescheduling'),
               'weekend' -> ('arch_01_a2_inf_sc01', 'arch_01_arranging_and_rescheduling'),
               'clinic' -> ('arch_13_b1p_form_sc01', 'arch_13_medical_consultation_and_health_triage')).
        Returns ('', '') if spec is empty or cannot be resolved.
        """
        clean = str(spec or "").strip().lower()
        if not clean:
            return ("", "")
        config = cls._load_functional_dialogue_archetypes()
        archetypes = config.get("archetypes") or []

        # 1. Exact match on scenario_id
        for arch in archetypes:
            a_id = str(arch.get("archetype_id") or "")
            if forced_archetype_id and a_id != forced_archetype_id:
                continue
            for tr in arch.get("trajectories") or []:
                for sc in tr.get("everyday_scenarios") or []:
                    sc_id = str(sc.get("scenario_id") or "")
                    if clean == sc_id.lower():
                        return (sc_id, a_id)

        # 2. Substring match on scenario_id
        for arch in archetypes:
            a_id = str(arch.get("archetype_id") or "")
            if forced_archetype_id and a_id != forced_archetype_id:
                continue
            for tr in arch.get("trajectories") or []:
                for sc in tr.get("everyday_scenarios") or []:
                    sc_id = str(sc.get("scenario_id") or "")
                    if clean in sc_id.lower():
                        return (sc_id, a_id)

        # 3. Substring match on scenario setting / communicative goal
        for arch in archetypes:
            a_id = str(arch.get("archetype_id") or "")
            if forced_archetype_id and a_id != forced_archetype_id:
                continue
            for tr in arch.get("trajectories") or []:
                for sc in tr.get("everyday_scenarios") or []:
                    sc_id = str(sc.get("scenario_id") or "")
                    setting_text = f"{sc.get('setting', '')} {sc.get('communicative_goal', '')}".lower()
                    if clean in setting_text:
                        return (sc_id, a_id)

        return ("", "")

    @classmethod
    def _load_functional_dialogue_archetypes(cls) -> Dict[str, Any]:
        """
        Loads the unified COBUILD 8 Functional Dialogue Archetypes & Register-Isolated
        Pragmatic Trajectories configuration (`librarian/data/functional_dialogue_archetypes.json`)
        with in-memory caching.
        """
        if cls._FUNCTIONAL_ARCHETYPES_CACHE is not None:
            return cls._FUNCTIONAL_ARCHETYPES_CACHE

        data_dir = Path(__file__).resolve().parent.parent / "data"
        target_path = data_dir / "functional_dialogue_archetypes.json"
        config_data: Dict[str, Any] = {"archetypes": [], "micro_move_plugins": []}
        if target_path.exists():
            try:
                config_data = json.loads(target_path.read_text(encoding="utf-8"))
            except Exception:
                config_data = {"archetypes": [], "micro_move_plugins": []}

        cls._FUNCTIONAL_ARCHETYPES_CACHE = config_data
        return config_data

    @classmethod
    def _select_functional_dialogue_trajectory(
        cls,
        thematic_topic: str = "",
        cefr_level: str = "B2",
        seed_key: str = "",
        forced_archetype_id: str = "",
        forced_trajectory_id: str = "",
        forced_scenario_id: str = "",
    ) -> Dict[str, Any]:
        """
        Deterministically selects the best-matching Functional Dialogue Archetype (`arch_01`..`arch_14`),
        register-isolated 8-turn Trajectory (`A1_A2` vs `B1_C2`), Everyday Scenario (with optional
        scenario-bound lexis `optional=True`), and compatible Micro-Move Plugin from
        `functional_dialogue_archetypes.json`.
        """
        config_data = cls._load_functional_dialogue_archetypes()
        archetypes = config_data.get("archetypes") or []
        plugins = config_data.get("micro_move_plugins") or []
        plugin_map: Dict[str, Dict[str, Any]] = {
            str(p.get("plugin_id")): p for p in plugins if isinstance(p, dict) and p.get("plugin_id")
        }

        if not archetypes:
            return {}

        if forced_archetype_id:
            resolved_forced = cls.resolve_functional_dialogue_archetype_id(forced_archetype_id)
            if resolved_forced:
                forced_archetype_id = resolved_forced
            else:
                logger.warning(
                    f"Unknown forced_archetype_id '{forced_archetype_id}'; "
                    f"falling back to automatic archetype selection."
                )
                forced_archetype_id = ""

        if forced_scenario_id:
            resolved_sc_id, resolved_arch_id = cls.resolve_functional_dialogue_scenario_id(
                forced_scenario_id, forced_archetype_id
            )
            if resolved_sc_id:
                forced_scenario_id = resolved_sc_id
                if not forced_archetype_id and resolved_arch_id:
                    forced_archetype_id = resolved_arch_id
            else:
                logger.warning(
                    f"Unknown forced_scenario_id '{forced_scenario_id}'; "
                    f"falling back to automatic scenario selection."
                )
                forced_scenario_id = ""

        cefr_upper = str(cefr_level or "B2").strip().upper()
        target_band = "A1_A2" if cefr_upper in ("A1", "A2", "A2+", "B1") else "B1_C2"
        is_advanced_academic = cefr_upper in ("B2", "C1", "C2")

        topic_norm = str(thematic_topic or "").lower()
        topic_tokens = set(re.findall(r"[a-z]{3,}", topic_norm))
        rng = zlib.crc32((seed_key or f"{thematic_topic}|{cefr_upper}").encode("utf-8"))

        scored_archetypes: List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]] = []
        for a_idx, arch in enumerate(archetypes):
            if not isinstance(arch, dict):
                continue
            arch_id = str(arch.get("archetype_id") or "")
            if forced_archetype_id and arch_id != forced_archetype_id:
                continue
            trajs = arch.get("trajectories") or []
            matched_traj = None
            if forced_trajectory_id:
                for tr in trajs:
                    if isinstance(tr, dict) and str(tr.get("trajectory_id") or "") == forced_trajectory_id:
                        matched_traj = tr
                        break
                if not matched_traj:
                    continue
            elif forced_scenario_id:
                for tr in trajs:
                    if isinstance(tr, dict):
                        sc_ids = [str(s.get("scenario_id") or "") for s in (tr.get("everyday_scenarios") or [])]
                        if forced_scenario_id in sc_ids:
                            matched_traj = tr
                            break
                if not matched_traj:
                    for tr in trajs:
                        if isinstance(tr, dict) and str(tr.get("cefr_band") or "") == target_band:
                            matched_traj = tr
                            break
                    if not matched_traj and trajs:
                        matched_traj = trajs[0]
            else:
                for tr in trajs:
                    if isinstance(tr, dict) and str(tr.get("cefr_band") or "") == target_band:
                        matched_traj = tr
                        break
                if not matched_traj and trajs:
                    matched_traj = trajs[0]
            if not matched_traj:
                continue

            score = 0
            # 1. Keyword & phrase matching against thematic_topic
            kw_list = [str(k).lower().strip() for k in (arch.get("thematic_keywords") or []) if str(k).strip()]
            for kw in kw_list:
                if " " in kw:
                    if kw in topic_norm:
                        score += 14
                elif kw in topic_tokens:
                    score += 10
                elif any(
                    len(t) >= 6 and len(kw) >= 6 and t[:6] == kw[:6] and abs(len(t) - len(kw)) <= 4
                    for t in topic_tokens
                ):
                    score += 5

            # 2. Check scenario settings and communicative goals for thematic resonance
            for sc in matched_traj.get("everyday_scenarios") or []:
                sc_text = f"{sc.get('setting', '')} {sc.get('communicative_goal', '')}".lower()
                sc_tokens = set(re.findall(r"[a-z]{4,}", sc_text))
                overlap = topic_tokens & sc_tokens - {
                    "with", "from", "that", "this", "have", "what", "when", "where",
                    "their", "about", "friend", "friends", "student", "students", "colleague",
                }
                score += min(12, len(overlap) * 4)

            # 3. Everyday reflective/discussion affinity when no narrow transactional/specialized keyword hits:
            # Provide an equal baseline score across versatile conversational/academic archetypes and let
            # deterministic CRC32 tie-breaker rotate across units to prevent single-archetype monoculture.
            if score == 0:
                versatile_archs = {
                    "arch_02_enquiring_comparing_and_deciding",
                    "arch_04_problem_sharing_empathy_and_advice",
                    "arch_05_opinion_exchange_and_consensus",
                    "arch_06_collaborative_proposals_and_tradeoffs",
                    "arch_08_social_catching_up_and_closing",
                }
                if is_advanced_academic:
                    versatile_archs.add("arch_09_academic_seminar_and_causal_probing")
                if arch_id in versatile_archs:
                    score = 4

            # Deterministic tie-breaker rotation across units
            tie_break = (rng + a_idx * 2654435761) & 0x7FFFFFFF
            scored_archetypes.append((score, tie_break, arch, matched_traj))

        if not scored_archetypes:
            return {}

        scored_archetypes.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen_arch = scored_archetypes[0][2]
        chosen_traj = scored_archetypes[0][3]

        # Select the best-matching everyday_scenario within chosen_traj
        scenarios = chosen_traj.get("everyday_scenarios") or []
        chosen_scenario: Dict[str, Any] = {}
        if scenarios:
            if forced_scenario_id:
                for sc in scenarios:
                    if str(sc.get("scenario_id") or "") == forced_scenario_id:
                        chosen_scenario = sc
                        break
            if not chosen_scenario:
                best_sc_score = -1
                best_sc_tie = -1
                for s_idx, sc in enumerate(scenarios):
                    sc_text = f"{sc.get('setting', '')} {sc.get('communicative_goal', '')}".lower()
                    sc_toks = set(re.findall(r"[a-z]{4,}", sc_text))
                    sc_score = len(topic_tokens & sc_toks) * 5
                    sc_tie = (rng + s_idx * 1103515245) & 0x7FFFFFFF
                    if (sc_score, sc_tie) > (best_sc_score, best_sc_tie):
                        best_sc_score = sc_score
                        best_sc_tie = sc_tie
                        chosen_scenario = sc

        # Select 1 compatible micro-move plugin
        compat_plugin_ids = [
            str(pid) for pid in (chosen_arch.get("compatible_micro_plugins") or [])
            if str(pid) in plugin_map
        ]
        chosen_plugin: Dict[str, Any] = {}
        if compat_plugin_ids:
            chosen_pid = compat_plugin_ids[rng % len(compat_plugin_ids)]
            chosen_plugin = plugin_map[chosen_pid]

        return {
            "archetype": chosen_arch,
            "trajectory": chosen_traj,
            "scenario": chosen_scenario,
            "plugin": chosen_plugin,
            "cefr_band": chosen_traj.get("cefr_band", target_band),
        }

    @classmethod
    def _select_listening_vocab_by_pedlex_freq(
        cls,
        vocab_items: List[Dict[str, str]],
        cefr_level: str = "B2",
        max_items: int = 8,
    ) -> List[Dict[str, str]]:
        """
        Deterministically scores and selects 5–8 target vocabulary items for `listening_quiz`
        using PedLex (`pedlex.db`) 4-dimensional lexical metadata across CEFR A1–C2:
        1. Spoken Frequency (`spoken_freq`: S1 > S2 > S3) & Multi-Word Expressions (`phrasal verb`,
           `collocation`, `set phrase`, `idiom`) receive top priority across all CEFR levels.
        2. Core Lexicon Tier (`core_level`: 3 `●●●` > 2 `●●○` > 1 `●○○` > 0 unstarred).
        3. Written Frequency (`written_freq`: W1 > W2 > W3).
        4. CEFR-Calibrated Academic Word List (`is_awl`):
           - Activated as a strong positive signal (+22) at B2/C1/C2 to support university seminar /
             analytical discussions (`controversial`, `reliable`, `eventual`, `convert`, `derive`).
           - De-emphasized at A1/A2/B1 in favor of everyday S1–S3 and `●●●` core words.
        5. Sense-Index Discount & Narrow Technical / Literary Guards:
           - Discounts headword S/W frequency when the locked definition belongs to a secondary
             anatomical/technical sense (`sense_idx >= 2`, e.g. `nerve` as body-part fiber).
           - Penalizes `[formal]`/`[literary]`/`[written]` archaisms (`hitherto`, `insurgent`) and
             narrow physical-science/warfare nouns (`reactor`, `atomic`, `synthesis`, `particulates`)
             so dialogues remain natural spoken exchanges rather than forced engineering lectures.
        """
        if not vocab_items:
            return []

        cefr_upper = str(cefr_level or "B2").strip().upper()
        is_advanced_academic = cefr_upper in ("B2", "C1", "C2")
        _stop_tokens = {
            "a", "an", "the", "on", "in", "at", "to", "for", "from", "with",
            "into", "of", "up", "out", "off", "down", "over", "by", "as",
            "sb", "sth", "ones", "one", "somebody", "something", "someone",
        }
        _narrow_tech_re = re.compile(
            r"\b(?:nuclear|atom|atoms|chemical|biological|weapons|car\s+engines|"
            r"inside\s+your\s+body|animal\s+or\s+plant|thin\s+line\s+of|"
            r"piece\s+of\s+something\s+solid|fighting\s+against\s+the\s+government)\b",
            re.IGNORECASE,
        )

        scored: List[Tuple[int, int, Dict[str, str]]] = []
        for idx, it in enumerate(vocab_items):
            w = (it.get("word") or "").strip()
            pos = (it.get("pos") or it.get("part_of_speech") or "word").strip().lower()
            defn = (it.get("definition") or "").strip()
            defn_low = defn.lower()

            is_mwe = pos in ("phrasal verb", "collocation", "set phrase", "idiom") or (" " in w)
            clean_w = re.sub(r"\[.*?\]|\(.*?\)", "", w).strip()
            content_toks = [
                t for t in re.findall(r"[a-zA-Z]+", clean_w.lower())
                if t not in _stop_tokens
            ]
            lookup_w = content_toks[0] if content_toks else clean_w.lower()

            ent = cls.get_pedlex_entry(lookup_w) or {}
            homographs = ent.get("homographs") or []
            matched_h = None
            for h in homographs:
                h_pos = str(h.get("pos") or "").lower()
                if h_pos and (
                    h_pos in pos
                    or pos in h_pos
                    or (pos == "phrasal verb" and h_pos == "verb")
                ):
                    matched_h = h
                    break
            if not matched_h and homographs:
                matched_h = homographs[0]

            core_level = int((matched_h or ent).get("core_level") or 0)
            freq_tags = list((matched_h or ent).get("freq_tags") or ent.get("freq_tags") or [])
            spoken_freq = str((matched_h or {}).get("spoken_freq") or "")
            written_freq = str((matched_h or {}).get("written_freq") or "")
            if not spoken_freq:
                for t in freq_tags:
                    if str(t).startswith("S"):
                        spoken_freq = str(t)
                        break
            if not written_freq:
                for t in freq_tags:
                    if str(t).startswith("W"):
                        written_freq = str(t)
                        break
            is_awl = bool((matched_h or ent).get("is_awl") or ent.get("is_awl"))
            h_reg = str((matched_h or {}).get("register") or "").lower()

            # Check which sense index matches the locked definition
            sense_idx = 0
            senses = ent.get("senses") or []
            for s_i, s_obj in enumerate(senses):
                s_def = str(s_obj.get("definition") or "").lower()
                if s_def and (s_def[:30] in defn_low or defn_low[:30] in s_def):
                    sense_idx = s_i
                    break

            score = 0
            if is_mwe:
                score += 32

            # Discount headword S/W frequency if the locked sense is a secondary sense (sense_idx >= 2)
            freq_mult = 1.0 if (is_mwe or sense_idx <= 1) else 0.35
            if spoken_freq == "S1":
                score += int(45 * freq_mult)
            elif spoken_freq == "S2":
                score += int(38 * freq_mult)
            elif spoken_freq == "S3":
                score += int(32 * freq_mult)

            if written_freq == "W1":
                score += int(15 * freq_mult)
            elif written_freq == "W2":
                score += int(12 * freq_mult)
            elif written_freq == "W3":
                score += int(10 * freq_mult)

            if core_level == 3:
                score += 25
            elif core_level == 2:
                score += 20
            elif core_level == 1:
                score += (4 if is_advanced_academic else -5)
            else:
                score += (5 if is_mwe else -30)

            if is_advanced_academic:
                if is_awl and core_level >= 2:
                    score += 22
                elif is_awl and core_level == 1:
                    score += 8
            else:
                if core_level == 3 and not is_awl:
                    score += 10

            if pos in ("verb", "adjective", "phrasal verb", "collocation", "set phrase", "idiom"):
                score += 8

            if (
                any(lbl in defn_low or lbl in h_reg for lbl in ("[formal]", "formal", "[literary]", "literary", "[written]"))
                and not spoken_freq
            ):
                score -= 50

            if _narrow_tech_re.search(defn_low):
                score -= 25

            badge_parts: List[str] = []
            if spoken_freq:
                badge_parts.append(spoken_freq)
            if written_freq:
                badge_parts.append(written_freq)
            if core_level > 0:
                badge_parts.append("●" * core_level + "○" * (3 - core_level))
            if is_awl:
                badge_parts.append("AWL")
            it_copy = dict(it)
            it_copy["pedlex_badge"] = " ".join(badge_parts) if badge_parts else "MWE" if is_mwe else "Core"
            scored.append((score, -idx, it_copy))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        target_len = max(5, min(int(max_items or 8), len(scored)))
        return [entry_tuple[2] for entry_tuple in scored[:target_len]]

    @classmethod
    def build_listening_micro_task_blueprints(
        cls,
        vocab_content: Any = None,
        thematic_topic: str = "",
        target_count: int = 5,
        cefr_level: str = "B2",
        speaker_1_role: str = "Speaker 1",
        speaker_2_role: str = "Speaker 2",
        speaker_1_name: str = "",
        speaker_2_name: str = "",
        forced_archetype_id: str = "",
        forced_trajectory_id: str = "",
        forced_scenario_id: str = "",
    ) -> Dict[str, Any]:
        """
        Builds the Symbolic Functional Dialogue & Question Micro-Task Blueprint for `listening_quiz`
        grounded in `librarian/data/functional_dialogue_archetypes.json`:
        1. Zero Forced Passage Vocabulary: Reading passage vocabulary is NEVER forced into dialogue turns.
           Spoken turns are driven exclusively by communicative functions, register-isolated COBUILD
           functional chunks, and natural scenario flow.
        2. Scenario-Bound Optional Lexis (`optional: true`): Everyday scenario expressions (e.g.,
           `give someone a ring`, `pick up`, `postpone`) are surfaced strictly as optional idiomatic
           options bound to their specific scenario and register.
        3. Register-Isolated 8-Turn Functional Trajectory (`Turns 1–8`) + Micro-Move Plugin (`Turns 3–6`):
           Couples each turn to an explicit conversational move, pragmatic function, COBUILD spoken
           functional chunks, and pragmatic contrast guardrails.
        4. Deterministic Question-to-Turn Anchoring & Option-Slot Fallacy Operators (`0..3`):
           - Correct Slot (`Paraphrase Operator` with `Forbidden Verbatim Copy` guard)
           - Distractor Slots (`Speaker Attribution Swap`, `Unadopted Proposal / Cross-Turn Detail Trap`,
             `Pragmatic / Modality / Polarity Distortion`)
        """
        cefr_upper = str(cefr_level or "B2").strip().upper()
        topic_clean = " / ".join(line.strip() for line in str(thematic_topic or "").splitlines() if line.strip())

        # Include any unit headwords only as part of the deterministic RNG seed so different units
        # with similar themes rotate across trajectories/scenarios deterministically, NEVER forcing them into turns.
        seed_vocab_tokens: List[str] = []
        if vocab_content:
            for norm_it in cls._normalize_vocab_items_input(vocab_content):
                w = (norm_it.get("word") or "").strip()
                if w:
                    seed_vocab_tokens.append(w)
        seed_str = "|".join(seed_vocab_tokens[:8]) + f"|{topic_clean}|{cefr_upper}"
        rng = zlib.crc32((seed_str or "default_functional_listening").encode("utf-8"))

        selection = cls._select_functional_dialogue_trajectory(
            thematic_topic=topic_clean,
            cefr_level=cefr_upper,
            seed_key=seed_str,
            forced_archetype_id=forced_archetype_id,
            forced_trajectory_id=forced_trajectory_id,
            forced_scenario_id=forced_scenario_id,
        )
        arch = selection.get("archetype") or {}
        traj = selection.get("trajectory") or {}
        scenario = selection.get("scenario") or {}
        plugin = selection.get("plugin") or {}
        cefr_band = selection.get("cefr_band") or ("A1_A2" if cefr_upper in ("A1", "A2", "A2+") else "B1_C2")

        arch_id = str(arch.get("archetype_id") or "arch_01_arranging_and_rescheduling")
        arch_name = str(arch.get("archetype_name") or "Making Arrangements & Rescheduling")
        traj_id = str(traj.get("trajectory_id") or "arch_01_a2_informal")
        traj_title = str(traj.get("title") or arch_name)
        traj_register = str(traj.get("register") or "neutral")
        speaker_roles = traj.get("speaker_roles") or {}
        sc_speaker_roles = scenario.get("speaker_roles") or {}
        spk1_desc = str(sc_speaker_roles.get("speaker_1") or speaker_roles.get("speaker_1") or speaker_1_role or "Speaker 1")
        spk2_desc = str(sc_speaker_roles.get("speaker_2") or speaker_roles.get("speaker_2") or speaker_2_role or "Speaker 2")

        sc_setting = str(scenario.get("setting") or "Everyday or academic spoken interaction")
        sc_participants = str(scenario.get("participants") or f"{spk1_desc} and {spk2_desc}")
        sc_goal = str(scenario.get("communicative_goal") or arch.get("communicative_goal") or "Resolve a practical communicative task naturally.")
        opt_lexis_list = scenario.get("optional_scenario_lexis") or []

        state_machine_lines: List[str] = [
            "### 🎙️ FUNCTIONAL DIALOGUE ARCHETYPE & 8-TURN PRAGMATIC TRAJECTORY BLUEPRINT ###",
            f"- **Functional Archetype**: `{arch_name}` (`{arch_id}`)",
            f"- **Selected Trajectory**: `{traj_title}` (CEFR Band: `{cefr_band}` | Register: `{traj_register}`)",
        ]
        if topic_clean:
            if cefr_band == "A1_A2":
                state_machine_lines.append(
                    f"- **Macro Thematic Backdrop / Physical Shell**: `{topic_clean}` (Physical shell only: use this domain strictly as an external physical anchor, such as a facility/building name, department title, or item label. The dialogue MUST remain 100% focused on concrete everyday communicative actions — directions, times, borrowing, dining, or simple arrangements. NEVER discuss abstract academic theories or corporate mechanisms in A1-B1 turns)."
                )
            else:
                state_machine_lines.append(
                    f"- **Macro Thematic Backdrop**: `{topic_clean}` (Ground the academic/professional discussion in this domain, anchored to a concrete project artifact or real data point)."
                )

        if cefr_band == "B1_C2":
            state_machine_lines.append(
                "- **SLA Academic Concretization Mandate (Tangible Project Anchors)**:\n"
                "  * Anchor the discussion to a tangible project artifact or data point (e.g. presentation slide #3, survey sample size, lab equipment schedule, draft report word count).\n"
                "  * Keep turns between 18 and 25 words (NEVER exceeding 30 words). Use high-frequency S1/S2 spoken words and natural contractions.\n"
                "  * BANNED pseudo-written connectors: NEVER use 'furthermore', 'moreover', 'in conclusion', or 'it is noteworthy that'. Use natural spoken signposts: 'the thing is', 'to be fair', 'on the flip side', 'the bottom line is'.\n"
                "  * Circumlocution & Analogy: If an abstract theoretical concept appears, one speaker must briefly unpack it with an everyday analogy or concrete example."
            )
        state_machine_lines.extend([
            f"- **Everyday Scenario Setting**: {sc_setting}",
            f"- **Participant Roles**:",
            f"  * **Speaker 1**: {spk1_desc}",
            f"  * **Speaker 2**: {spk2_desc}",
            f"- **Communicative Goal**: {sc_goal}",
        ])

        if opt_lexis_list:
            lex_strs: List[str] = []
            for lex_it in opt_lexis_list:
                if isinstance(lex_it, dict) and lex_it.get("expression"):
                    expr = str(lex_it["expression"]).strip()
                    meaning = str(lex_it.get("meaning") or "").strip()
                    lex_strs.append(f'`"{expr}"` ({meaning})' if meaning else f'`"{expr}"`')
            if lex_strs:
                state_machine_lines.append(
                    f"- **Scenario-Bound Idiomatic Expressions (`optional: true` — use ONLY if they fit naturally; NEVER force)**: "
                    + "; ".join(lex_strs)
                )

        state_machine_lines.append(
            "\n#### 🔄 8-Turn Pragmatic Progression (Speaker 1 & Speaker 2) ####"
        )

        turn_specs: List[Dict[str, Any]] = []
        used_turn_chunks: Set[str] = set()
        turn_progression = traj.get("turn_progression") or []
        for t_obj in turn_progression:
            if not isinstance(t_obj, dict):
                continue
            t_num = int(t_obj.get("turn_number") or (len(turn_specs) + 1))
            spk_label = str(t_obj.get("speaker") or ("Speaker 1" if t_num % 2 == 1 else "Speaker 2"))
            phase_label = str(t_obj.get("phase") or f"Turn {t_num}")
            move_name = str(t_obj.get("move_name") or "Conversational Move")
            prag_func = str(t_obj.get("pragmatic_function") or "Advance the conversation naturally.")

            turn_overrides = scenario.get("turn_overrides") or {}
            sc_override = turn_overrides.get(str(t_num)) or turn_overrides.get(t_num)
            raw_fc = t_obj.get("functional_chunks") or []
            if isinstance(sc_override, dict):
                if sc_override.get("move_name"):
                    move_name = str(sc_override["move_name"])
                if sc_override.get("pragmatic_function"):
                    prag_func = str(sc_override["pragmatic_function"])
                if sc_override.get("functional_chunks"):
                    raw_fc = sc_override["functional_chunks"]
            chunk_patterns: List[str] = []
            for fc in raw_fc:
                if isinstance(fc, dict) and fc.get("pattern"):
                    pat = str(fc["pattern"]).strip()
                    if pat and pat not in chunk_patterns and pat not in used_turn_chunks:
                        chunk_patterns.append(pat)
                elif isinstance(fc, str) and fc.strip():
                    pat = fc.strip()
                    if pat not in chunk_patterns and pat not in used_turn_chunks:
                        chunk_patterns.append(pat)

            # Deterministically rotate up to 3 chunks so different units surface varied functional chunks
            if len(chunk_patterns) > 3:
                start_i = (rng + t_num * 5) % len(chunk_patterns)
                selected_chunks = (chunk_patterns[start_i:] + chunk_patterns[:start_i])[:3]
            else:
                selected_chunks = chunk_patterns[:3]

            for sc in selected_chunks:
                used_turn_chunks.add(sc)

            opt_exprs = [
                str(x).strip()
                for x in (t_obj.get("optional_expressions") or [])
                if str(x).strip() and str(x).strip() not in used_turn_chunks
            ]
            chunks_display = " / ".join(f'`{c}`' for c in selected_chunks) if selected_chunks else '`"Well, ..."`'
            opt_display = (
                f" | Optional idioms: " + " / ".join(f'`{e}`' for e in opt_exprs[:2])
                if opt_exprs
                else ""
            )

            state_machine_lines.append(
                f"- **Turn {t_num} ({spk_label} — {move_name})**: {prag_func} "
                f"Flexible Spoken Patterns: {chunks_display}.{opt_display}"
            )
            turn_specs.append({
                "turn_number": t_num,
                "speaker": spk_label,
                "phase": phase_label,
                "move_name": move_name,
                "strategy_name": move_name,
                "pragmatic_function": prag_func,
                "recommended_chunks": selected_chunks,
                "optional_expressions": opt_exprs[:2],
            })

        # Add Optional Micro-Move Conversational Plugin (Turns 3-6)
        if isinstance(plugin, dict) and plugin.get("plugin_name"):
            p_name = str(plugin.get("plugin_name"))
            p_turns = plugin.get("compatible_turns") or [3, 4, 5, 6]
            p_trigger = str(plugin.get("communicative_trigger") or "")
            p_chunk_key = "a1_a2_chunks" if cefr_band == "A1_A2" else "b1_c2_chunks"
            p_raw_chunks = plugin.get(p_chunk_key) or []
            p_pats: List[str] = []
            for pc in p_raw_chunks:
                if isinstance(pc, dict) and pc.get("pattern"):
                    p_pats.append(str(pc["pattern"]).strip())
            if p_pats:
                p_start = rng % len(p_pats)
                p_chosen = (p_pats[p_start:] + p_pats[:p_start])[:3]
                p_chunks_str = " / ".join(f'"{c}"' for c in p_chosen)
                turns_str = ", ".join(str(t) for t in p_turns)
                state_machine_lines.extend([
                    f"\n#### 🧩 Compatible Micro-Move Plugin (Optional Enrichment for Turns {turns_str}) ####",
                    f"- **Plugin**: `{p_name}` — {p_trigger}",
                    f"- **Optional Plugin Chunks**: {p_chunks_str}",
                ])

        # Add Pragmatic Contrasts & Communication Tips
        contrasts = traj.get("pragmatic_contrasts") or []
        comm_tips = traj.get("communication_tips") or []
        if contrasts or comm_tips:
            state_machine_lines.append("\n#### 🛡️ Spoken Register & Pragmatic Guardrails ####")
            for c_obj in contrasts[:2]:
                if not isinstance(c_obj, dict):
                    continue
                sit = str(c_obj.get("situation") or "").strip()
                pref = str(c_obj.get("prefer_authentic_spoken") or "").strip()
                avoid = str(c_obj.get("avoid_stiff_or_blunt") or "").strip()
                p_note = str(c_obj.get("pragmatic_note") or "").strip()
                if pref and avoid:
                    state_machine_lines.append(
                        f"- **[Spoken Register Guard — {sit}]**: Prefer `{pref}` instead of stiff/blunt `{avoid}`. ({p_note})"
                    )
            for tip in comm_tips[:2]:
                if str(tip).strip():
                    state_machine_lines.append(f"- **[Pragmatic Tip]**: {str(tip).strip()}")

        # Build Deterministic Question Blueprints from trajectory's question_blueprint_template
        effective_q_count = max(1, int(target_count or 5))
        q_templates = traj.get("question_blueprint_template") or []

        # Deterministic CRC32-balanced 4-option slots (0, 1, 2, 3) with zero adjacent duplicates
        rng_state = zlib.crc32((seed_str + "|q_slots").encode("utf-8"))
        base_slots: List[int] = []
        while len(base_slots) < effective_q_count:
            block = [0, 1, 2, 3]
            for i in range(3, 0, -1):
                rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
                j = rng_state % (i + 1)
                block[i], block[j] = block[j], block[i]
            if base_slots and block[0] == base_slots[-1]:
                block[0], block[1] = block[1], block[0]
            base_slots.extend(block)
        base_slots = base_slots[:effective_q_count]

        def _normalize_canonical_skill(raw_skill: str, q_i: int) -> str:
            low = str(raw_skill or "").lower()
            if "main idea" in low or "purpose" in low:
                return "Main Idea"
            if "inference" in low or "attitude" in low or "outcome" in low or "implication" in low:
                return "Inference"
            if "detail" in low or "constraint" in low or "negotiation" in low:
                return "Detail"
            return ("Main Idea", "Detail", "Inference", "Detail", "Inference")[q_i % 5]

        question_blueprints: List[Dict[str, Any]] = []
        question_lines = [
            "### 🎯 DIALOGUE-DRIVEN QUESTION BLUEPRINTS ###",
        ]

        option_letters = ["A", "B", "C", "D"]
        for q_idx in range(effective_q_count):
            q_tmpl = q_templates[q_idx % len(q_templates)] if q_templates else {}
            correct_slot = base_slots[q_idx]
            correct_letter = option_letters[correct_slot]

            raw_anchor_turns = q_tmpl.get("anchor_turns") or [1, 2]
            if len(raw_anchor_turns) == 1:
                anchor_turn_str = f"Turn {raw_anchor_turns[0]}"
            else:
                anchor_turn_str = f"Turns {raw_anchor_turns[0]}–{raw_anchor_turns[-1]}"
            primary_turn = int(raw_anchor_turns[0]) if raw_anchor_turns else 1

            target_speaker = str(q_tmpl.get("target_speaker") or ("Speaker 1" if primary_turn % 2 == 1 else "Speaker 2"))
            raw_skill_label = str(q_tmpl.get("target_skill") or "Detail")
            canonical_skill = _normalize_canonical_skill(raw_skill_label, q_idx)
            comp_focus = str(
                q_tmpl.get("comprehension_focus")
                or f"the key communicative point or decision around {anchor_turn_str}"
            )
            op_dict = q_tmpl.get("option_slot_operators") or {}
            corr_op_desc = str(
                op_dict.get("correct_operator")
                or f"Paraphrase Operator (paraphrase what is actually stated or implied around {anchor_turn_str} using fresh wording)"
            )
            d1_op_desc = str(
                op_dict.get("distractor_1_operator")
                or "Speaker / Role Swap (attribute a detail or statement from the other speaker to the target speaker)"
            )
            d2_op_desc = str(
                op_dict.get("distractor_2_operator")
                or "Opposite Detail / Condition Flip (flip the polarity, timeframe, or condition of what was actually said)"
            )
            d3_op_desc = str(
                op_dict.get("distractor_3_operator")
                or "Cross-Turn Detail Mix-up (borrow a detail from another part of the dialogue out of context)"
            )

            def _split_op_label(desc_str: str, default_label: str) -> Tuple[str, str]:
                if "(" in desc_str:
                    prefix, rest = desc_str.split("(", 1)
                    return prefix.strip() or default_label, rest.rstrip(")").strip()
                return default_label, desc_str

            _, corr_instruction = _split_op_label(corr_op_desc, "Paraphrase Operator")
            corr_instruction = re.sub(
                r"\s*without copying \d+\+ consecutive words.*$",
                "",
                corr_instruction,
                flags=re.IGNORECASE,
            ).strip()
            d1_name, d1_instruction = _split_op_label(d1_op_desc, "Distractor Trap 1")
            d2_name, d2_instruction = _split_op_label(d2_op_desc, "Distractor Trap 2")
            d3_name, d3_instruction = _split_op_label(d3_op_desc, "Distractor Trap 3")

            distractor_slots = [s for s in (0, 1, 2, 3) if s != correct_slot]
            trap_specs = {
                distractor_slots[0]: (d1_name, d1_instruction),
                distractor_slots[1]: (d2_name, d2_instruction),
                distractor_slots[2]: (d3_name, d3_instruction),
            }

            slot_operators_meta: Dict[int, str] = {}
            for s_i in range(4):
                if s_i == correct_slot:
                    slot_operators_meta[s_i] = "Paraphrase Operator (Correct Answer)"
                else:
                    op_name, _ = trap_specs[s_i]
                    slot_operators_meta[s_i] = op_name

            dist_slots_str = ", ".join(
                f"Option {chr(65 + s)} (`options[{s}]`)" for s in distractor_slots
            )

            q_block_lines = [
                f"#### Question {q_idx + 1} Blueprint ####",
                f"- Target Exchange & Focus: `{anchor_turn_str}` (`{target_speaker}`) — *{comp_focus}*",
                f"- Required Category (`category`): `{canonical_skill}` | 🔒 Prescribed Correct Answer Index (`correct_answer_index`): `{correct_slot}` (Option {correct_letter})",
                f"- Correct Answer (Option {correct_letter} / `options[{correct_slot}]`): {corr_instruction}.",
                f"- Distractor Traps ({dist_slots_str}): `{d1_name}` / `{d2_name}` / `{d3_name}`",
            ]
            question_lines.append("\n".join(q_block_lines))

            question_blueprints.append({
                "item_index": q_idx + 1,
                "anchor_turn": anchor_turn_str,
                "primary_turn": primary_turn,
                "target_speaker": target_speaker,
                "target_skill": canonical_skill,
                "detailed_skill": raw_skill_label,
                "comprehension_focus": comp_focus,
                "correct_answer_index": correct_slot,
                "slot_operators": slot_operators_meta,
                "micro_task_block": "\n".join(q_block_lines),
            })

        state_machine_str = "\n".join(state_machine_lines)
        questions_str = "\n\n".join(question_lines)
        combined_prompt_block = state_machine_str + "\n\n" + questions_str

        s1_clean = str(speaker_1_name or "").strip()
        s2_clean = str(speaker_2_name or "").strip()
        if s1_clean and s1_clean.lower() not in ("speaker 1", "speaker1"):
            state_machine_str = re.sub(r"\bSpeaker\s*1\b", s1_clean, state_machine_str)
            questions_str = re.sub(r"\bSpeaker\s*1\b", s1_clean, questions_str)
            combined_prompt_block = re.sub(r"\bSpeaker\s*1\b", s1_clean, combined_prompt_block)
            for qb in question_blueprints:
                qb["micro_task_block"] = re.sub(r"\bSpeaker\s*1\b", s1_clean, qb.get("micro_task_block", ""))
        if s2_clean and s2_clean.lower() not in ("speaker 2", "speaker2"):
            state_machine_str = re.sub(r"\bSpeaker\s*2\b", s2_clean, state_machine_str)
            questions_str = re.sub(r"\bSpeaker\s*2\b", s2_clean, questions_str)
            combined_prompt_block = re.sub(r"\bSpeaker\s*2\b", s2_clean, combined_prompt_block)
            for qb in question_blueprints:
                qb["micro_task_block"] = re.sub(r"\bSpeaker\s*2\b", s2_clean, qb.get("micro_task_block", ""))

        return {
            "cefr_level": cefr_upper,
            "cefr_band": cefr_band,
            "archetype_id": arch_id,
            "archetype_name": arch_name,
            "trajectory_id": traj_id,
            "trajectory_title": traj_title,
            "scenario_id": scenario.get("scenario_id", ""),
            "turn_specs": turn_specs,
            "questions": question_blueprints,
            "state_machine_block": state_machine_str,
            "questions_block": questions_str,
            "combined_prompt_block": combined_prompt_block,
        }
