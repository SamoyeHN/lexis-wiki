"""Blueprint item hydration and LLM surface frame rendering."""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..linguistics import LinguisticEngine

logger = logging.getLogger("librarian.processor.hydrators")


class QuizHydratorMixin:
    """Mixin providing blueprint field hydration and surface frame rendering across quiz modalities."""

    @classmethod
    def hydrate_translation_blueprint_items(
        cls,
        quiz_obj: Any,
        blueprint_specs: List[Dict[str, Any]],
    ) -> Any:
        """
        Deterministically hydrates blueprint-owned fields on TranslationQuiz items:
        - Binds `target_keyword`, `target_grammar`, and `flaw_type` from `blueprint_specs`.
        - Constructs `options` from `[idiomatic_translation, flawed_translation]` ordered according
          to `prescribed_correct_index` (CRC32-locked 0 or 1).
        - Sets `correct_answer_index`, `correct_english_answer`, and hydrates `explanation` from
          `diagnostic_critique`.
        """
        if not quiz_obj or not blueprint_specs:
            return quiz_obj

        q_list = (
            quiz_obj.get("questions", [])
            if isinstance(quiz_obj, dict)
            else getattr(quiz_obj, "questions", [])
        )
        if not isinstance(q_list, list):
            return quiz_obj

        for q_i, q_item in enumerate(q_list):
            if q_i >= len(blueprint_specs):
                break
            spec = blueprint_specs[q_i] or {}
            is_dict = isinstance(q_item, dict)

            def _get(field: str, default: Any = "") -> Any:
                return q_item.get(field, default) if is_dict else getattr(q_item, field, default)

            def _set(field: str, val: Any) -> None:
                if is_dict:
                    q_item[field] = val
                else:
                    setattr(q_item, field, val)

            if spec.get("target_keyword"):
                _set("target_keyword", spec["target_keyword"])
            if spec.get("target_grammar"):
                _set("target_grammar", spec["target_grammar"])
            if spec.get("flaw_type"):
                _set("flaw_type", spec["flaw_type"])

            # Deterministically hydrate design_audit (<0.01ms, 0 tokens)
            t_name = spec.get("target_grammar") or spec.get("target_keyword") or "Translation Target"
            f_name = spec.get("flaw_type") or "L1 Negative Transfer Trap"
            audit_str = f"AUDIT: [{t_name}] -> [{f_name}]"
            cur_audit = str(_get("design_audit") or "").strip()
            if not cur_audit or not cur_audit.startswith("AUDIT:"):
                _set("design_audit", audit_str)

            idiomatic = str(_get("idiomatic_translation", "") or "").strip()
            flawed = str(_get("flawed_translation", "") or "").strip()
            critique = str(_get("diagnostic_critique", "") or "").strip()

            if idiomatic and flawed:
                prescribed_idx = spec.get("prescribed_correct_index")
                if prescribed_idx is None:
                    prescribed_idx = spec.get("correct_answer_index")
                if prescribed_idx not in (0, 1):
                    import zlib
                    seed = f"{spec.get('target_keyword', '')}:{q_i}".encode("utf-8")
                    prescribed_idx = zlib.crc32(seed) % 2

                if prescribed_idx == 1:
                    opts = [flawed, idiomatic]
                else:
                    opts = [idiomatic, flawed]

                _set("options", opts)
                _set("correct_answer_index", prescribed_idx)
                _set("correct_english_answer", idiomatic)

            if critique:
                _set("explanation", critique)

        return quiz_obj

    @classmethod
    def hydrate_listening_blueprint_items(
        cls,
        quiz_obj: Any,
        blueprint_specs: List[Dict[str, Any]],
        speaker_1_name: str = "",
        speaker_2_name: str = "",
    ) -> Any:
        """
        Deterministically hydrates and reconciles blueprint-owned fields on ListeningQuiz items:
        - Enforces `speaker_1`, `speaker_2`, and alternating turn `speaker` names (`speaker_1` on odd turns,
          `speaker_2` on even turns), replacing any residual `'Speaker 1'` / `'Speaker 2'` across `question`,
          `options`, and `explanation`.
        - Enforces `category` from `spec["target_skill"]` ("Detail", "Main Idea", "Inference").
        - Reconciles the true option position with the CRC32-prescribed `spec["correct_answer_index"]`:
          if the LLM placed the true answer at a different index `actual_idx`, swaps
          `options[actual_idx]` <-> `options[prescribed_idx]`, sets `correct_answer_index = prescribed_idx`,
          and atomically remaps `Option A/B/C/D` references in `explanation`.
        - Intentionally does NOT attach `target_word` onto question items so `evaluator.py`
          treats them as listening comprehension questions rather than fill-in-the-blank items.
        """
        if not quiz_obj or not blueprint_specs:
            return quiz_obj

        # 0. Synchronize speaker_1, speaker_2, and script turn speakers
        spk1 = str(
            speaker_1_name
            or (quiz_obj.get("speaker_1") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "speaker_1", ""))
            or ""
        ).strip()
        spk2 = str(
            speaker_2_name
            or (quiz_obj.get("speaker_2") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "speaker_2", ""))
            or ""
        ).strip()

        if not spk1 or spk1.lower() in ("speaker 1", "speaker1") or not spk2 or spk2.lower() in ("speaker 2", "speaker2"):
            try:
                from ..tts import tts_service
                tts_service._refresh()
                if not spk1 or spk1.lower() in ("speaker 1", "speaker1"):
                    spk1 = tts_service.get_speaker_name(tts_service.voice_a)
                if not spk2 or spk2.lower() in ("speaker 2", "speaker2"):
                    spk2 = tts_service.get_speaker_name(tts_service.voice_b)
            except Exception:
                pass

        script_list = (
            quiz_obj.get("script", [])
            if isinstance(quiz_obj, dict)
            else getattr(quiz_obj, "script", [])
        )
        if isinstance(script_list, list) and script_list:
            for t_idx, t_item in enumerate(script_list):
                t_spk = str(
                    (t_item.get("speaker") if isinstance(t_item, dict) else getattr(t_item, "speaker", ""))
                    or ""
                ).strip()
                if t_spk and t_spk.lower() not in ("speaker 1", "speaker1", "speaker 2", "speaker2"):
                    if t_idx % 2 == 0 and not spk1:
                        spk1 = t_spk
                    elif t_idx % 2 == 1 and not spk2:
                        spk2 = t_spk
            spk1 = spk1 or "Aria"
            spk2 = spk2 or "Ryan"
            if isinstance(quiz_obj, dict):
                quiz_obj["speaker_1"] = spk1
                quiz_obj["speaker_2"] = spk2
            else:
                setattr(quiz_obj, "speaker_1", spk1)
                setattr(quiz_obj, "speaker_2", spk2)

            for t_idx, t_item in enumerate(script_list):
                t_is_dict = isinstance(t_item, dict)
                t_spk = str(
                    (t_item.get("speaker") if t_is_dict else getattr(t_item, "speaker", ""))
                    or ""
                ).strip()
                expected_spk = spk1 if (t_idx % 2 == 0) else spk2
                if (
                    not t_spk
                    or t_spk.lower() in ("speaker 1", "speaker1", "speaker 2", "speaker2")
                    or (speaker_1_name and speaker_2_name)
                ):
                    if t_is_dict:
                        t_item["speaker"] = expected_spk
                    else:
                        setattr(t_item, "speaker", expected_spk)

        def _replace_speaker_labels(txt: str) -> str:
            if not txt:
                return txt
            out = txt
            if spk1 and spk1 != "Speaker 1":
                out = re.sub(r"\bSpeaker\s*1\b", spk1, out, flags=re.IGNORECASE)
            if spk2 and spk2 != "Speaker 2":
                out = re.sub(r"\bSpeaker\s*2\b", spk2, out, flags=re.IGNORECASE)
            return out

        q_list = (
            quiz_obj.get("questions", [])
            if isinstance(quiz_obj, dict)
            else getattr(quiz_obj, "questions", [])
        )
        if not isinstance(q_list, list):
            return quiz_obj

        for q_i, q_item in enumerate(q_list):
            if q_i >= len(blueprint_specs):
                break
            spec = blueprint_specs[q_i] or {}
            is_dict = isinstance(q_item, dict)

            def _get(field: str, default: Any = "") -> Any:
                return q_item.get(field, default) if is_dict else getattr(q_item, field, default)

            def _set(field: str, val: Any) -> None:
                if is_dict:
                    q_item[field] = val
                else:
                    setattr(q_item, field, val)

            # 1. Enforce prescribed category ("Detail", "Main Idea", "Inference") and sanitize stem
            target_skill = str(spec.get("target_skill") or "").strip()
            if target_skill in ("Detail", "Main Idea", "Inference"):
                _set("category", target_skill)

            raw_stem = str(_get("question", "") or "").strip()
            if raw_stem:
                clean_stem = re.sub(r"\s+in\s+[Tt]urns?\s+\d+(?:\s*[–-]\s*\d+)?", "", raw_stem)
                clean_stem = _replace_speaker_labels(clean_stem)
                clean_stem = re.sub(r"\s{2,}", " ", clean_stem).strip()
                if clean_stem != raw_stem:
                    _set("question", clean_stem)

            # 2. Reconcile true answer slot with CRC32-prescribed correct_answer_index
            raw_opts = _get("options", [])
            if not isinstance(raw_opts, list) or len(raw_opts) != 4:
                continue
            opts = [_replace_speaker_labels(str(o or "")) for o in raw_opts]
            if opts != list(raw_opts):
                _set("options", opts)

            prescribed_idx = spec.get("correct_answer_index")
            if not isinstance(prescribed_idx, int) or prescribed_idx not in (0, 1, 2, 3):
                continue

            declared_idx = _get("correct_answer_index", prescribed_idx)
            try:
                declared_idx = int(declared_idx)
            except (TypeError, ValueError):
                declared_idx = prescribed_idx
            if declared_idx not in (0, 1, 2, 3):
                declared_idx = prescribed_idx

            cur_exp = _replace_speaker_labels(str(_get("explanation", "") or ""))
            if cur_exp != str(_get("explanation", "") or ""):
                _set("explanation", cur_exp)
            cur_audit = str(_get("design_audit", "") or "")
            full_text = f"{cur_audit}\n{cur_exp}"

            # Check if explanation explicitly names a specific option letter as the true answer
            actual_idx = declared_idx
            m_lead_opt = re.search(
                r"\bOption\s+([A-D])\b[^\.\n]*?\b(?:is\s+the\s+only|correctly\s+(?:captures|identifies|paraphrases|states|reflects|matches|summarizes)|is\s+(?:the\s+)?correct\b)",
                cur_exp,
                re.IGNORECASE,
            )
            inferred_letter = None
            if m_lead_opt:
                inferred_letter = m_lead_opt.group(1).upper()
            else:
                m_opt = re.search(
                    r"(?:(?:correct\s+answer\s+is|supporting|matches|aligned\s+with|only)\s+Option\s+([A-D])\b)",
                    full_text,
                    re.IGNORECASE,
                )
                if m_opt:
                    inferred_letter = m_opt.group(1).upper()

            inferred_idx = (ord(inferred_letter) - 65) if inferred_letter else None
            if inferred_idx is not None and 0 <= inferred_idx < 4:
                actual_idx = inferred_idx

            if actual_idx != prescribed_idx:
                # Swap options so the true option physically resides at prescribed_idx
                opts[actual_idx], opts[prescribed_idx] = opts[prescribed_idx], opts[actual_idx]
                _set("options", opts)
                if cur_exp:
                    swap_map = {actual_idx: prescribed_idx, prescribed_idx: actual_idx}
                    _set("explanation", cls._remap_explanation_labels(cur_exp, swap_map))

            _set("correct_answer_index", prescribed_idx)

            # 3. Deterministically hydrate design_audit (<0.01ms, 0 tokens)
            target_spk = spec.get("target_speaker") or (spk1 if (spec.get("primary_turn", 1) % 2 == 1) else spk2)
            if target_spk in ("Speaker 1", "speaker 1", "speaker1"):
                target_spk = spk1
            elif target_spk in ("Speaker 2", "speaker 2", "speaker2"):
                target_spk = spk2
            turn_str = spec.get("anchor_turn") or "Turn 1"
            skill_str = target_skill or "Detail"
            ops = [v for k, v in (spec.get("slot_operators") or {}).items() if "Correct Answer" not in v]
            trap_str = " + ".join(ops[:2]) if ops else "Distractor Traps"
            audit_str = f"AUDIT: [{target_spk}, {turn_str}] -> [{skill_str}] -> [{trap_str}]"
            cur_audit = str(_get("design_audit") or "").strip()
            if not cur_audit or not cur_audit.startswith("AUDIT:"):
                _set("design_audit", audit_str)

        return quiz_obj

    @classmethod
    def hydrate_reading_blueprint_items(
        cls,
        quiz_obj: Any,
        blueprint_specs: List[Dict[str, Any]],
    ) -> Any:
        """
        Deterministically hydrates and reconciles blueprint-owned fields on ReadingQuiz items:
        - Reconciles question category with canonical Reading schema literal:
          `macro_gist` -> "Main Idea", `causal_chain` / `factual_mechanism` / `rhetorical_contrast` -> "Detail/Recall",
          `pragmatic_inference` -> "Inference", `modality_hedging` -> "Author's Tone/Purpose".
        - Enforces canonical design_audit tag format: `AUDIT: [AnchorID] -> [Category] -> [Operators]`.
        - Fallback repairs question stem if empty or trivially corrupt.
        """
        if not quiz_obj or not blueprint_specs:
            return quiz_obj

        q_list = quiz_obj.get("questions", []) if isinstance(quiz_obj, dict) else getattr(quiz_obj, "questions", [])
        if not isinstance(q_list, list):
            return quiz_obj

        CATEGORY_MAP = {
            "macro_gist": "Main Idea",
            "causal_chain": "Detail/Recall",
            "factual_mechanism": "Detail/Recall",
            "pragmatic_inference": "Inference",
            "modality_hedging": "Author's Tone/Purpose",
            "rhetorical_contrast": "Detail/Recall",
        }

        used_section_titles = set()
        for q_i, q_item in enumerate(q_list):
            if q_i >= len(blueprint_specs):
                break
            spec = blueprint_specs[q_i] or {}
            is_dict = isinstance(q_item, dict)

            def _get(field: str, default: Any = "") -> Any:
                if is_dict:
                    return q_item.get(field, default)
                return getattr(q_item, field, default)

            def _set(field: str, val: Any) -> None:
                if is_dict:
                    q_item[field] = val
                else:
                    setattr(q_item, field, val)

            skill = str(spec.get("skill_type") or "macro_gist").lower()
            cat = CATEGORY_MAP.get(skill, "Main Idea")
            _set("category", cat)

            ops = [d.get("operator", "") for d in spec.get("prescribed_distractor_operators", []) if isinstance(d, dict)]
            ops_str = "/".join(ops) if ops else "fallacy_operators"
            anchor_id = spec.get("anchor_id", f"S-{q_i + 1}")
            audit_str = f"AUDIT: [{anchor_id}] -> [{cat}] -> [{ops_str}]"

            cur_audit = str(_get("design_audit") or "").strip()
            if not cur_audit or not cur_audit.startswith("AUDIT:"):
                _set("design_audit", audit_str)

            cur_stem = str(_get("question") or "").strip()
            sec_title = spec.get("section_title", "")
            has_custom_title = bool(sec_title and not sec_title.lower().startswith("paragraph "))

            needs_stem_repair = False
            if not cur_stem or len(cur_stem) < 8:
                needs_stem_repair = True
            elif has_custom_title:
                stem_lower = cur_stem.lower()
                sec_lower = sec_title.lower()
                collides_with_other = any(other_t.lower() in stem_lower for other_t in used_section_titles)
                if collides_with_other or (sec_lower not in stem_lower):
                    needs_stem_repair = True

            if needs_stem_repair and spec.get("question_stem"):
                _set("question", spec.get("question_stem"))
                cur_stem = spec.get("question_stem")

            if has_custom_title:
                used_section_titles.add(sec_title)

        # Cross-item distractor deduplication guard
        seen_distractors: Dict[str, Tuple[int, int]] = {}  # norm_text -> (q_idx, opt_idx)
        for q_i, q_item in enumerate(q_list):
            is_dict = isinstance(q_item, dict)
            opts = q_item.get("options") if is_dict else getattr(q_item, "options", [])
            c_idx = q_item.get("correct_answer_index", 0) if is_dict else getattr(q_item, "correct_answer_index", 0)
            if not isinstance(opts, list):
                continue
            opts_copy = list(opts)
            changed = False
            for opt_i, opt in enumerate(opts_copy):
                if opt_i == c_idx:
                    continue
                opt_norm = str(opt).strip().lower()
                if len(opt_norm) >= 6:
                    if opt_norm in seen_distractors:
                        # Collision with earlier distractor!
                        spec = blueprint_specs[q_i] if q_i < len(blueprint_specs) else {}
                        alt_ref = ""
                        for d_spec in spec.get("prescribed_distractor_operators", []):
                            if d_spec.get("source_text"):
                                alt_ref = d_spec["source_text"].strip().rstrip(".")
                                break
                        if alt_ref and alt_ref.lower() != opt_norm:
                            opts_copy[opt_i] = f"{alt_ref}."
                        else:
                            opts_copy[opt_i] = f"Specifically, {str(opt).strip()}"
                        changed = True
                    else:
                        seen_distractors[opt_norm] = (q_i, opt_i)
            if changed:
                if is_dict:
                    q_item["options"] = opts_copy
                else:
                    setattr(q_item, "options", opts_copy)

        return quiz_obj

    @classmethod
    def hydrate_vocabulary_blueprint_items(
        cls,
        quiz_obj: Any,
        blueprint_specs: List[Dict[str, Any]],
    ) -> Any:
        """
        Deterministically hydrates and reconciles blueprint-owned fields on VocabularyQuiz items:
        - Reconciles target_word, definition, prescribed options, and correct_answer_index.
        - Automatically repairs question stem for bound prepositions if dropped.
        - Enforces canonical design_audit tag format: `AUDIT: [Target Word] -> [Part of Speech] -> [Exact Blanked Frame in Stem]`.
        """
        if not quiz_obj or not blueprint_specs:
            return quiz_obj

        q_list = quiz_obj.get("questions", []) if isinstance(quiz_obj, dict) else getattr(quiz_obj, "questions", [])
        if not isinstance(q_list, list):
            return quiz_obj

        for q_i, q_item in enumerate(q_list):
            if q_i >= len(blueprint_specs):
                break
            spec = blueprint_specs[q_i] or {}
            is_dict = isinstance(q_item, dict)

            def _get(field: str, default: Any = "") -> Any:
                if is_dict:
                    return q_item.get(field, default)
                return getattr(q_item, field, default)

            def _set(field: str, val: Any) -> None:
                if is_dict:
                    q_item[field] = val
                else:
                    setattr(q_item, field, val)

            # 1. Stem post-repair for bound prepositions
            stem_str = str(_get("question", "") or "")
            micro_task_rule = str(spec.get("micro_task") or "")
            m_bound_prep = re.search(r"____\s+([a-z]{2,10})\b", micro_task_rule)
            _other_preps = {
                "in", "on", "at", "to", "for", "from", "with", "by", "of",
                "into", "onto", "upon", "about", "against", "under", "over",
                "through", "between", "among", "as", "towards", "toward",
            }
            if m_bound_prep and m_bound_prep.group(1).lower() in _other_preps and "____" in stem_str:
                req_prep = m_bound_prep.group(1).lower()
                if (
                    not re.search(rf"____\s+{re.escape(req_prep)}\b", stem_str, flags=re.IGNORECASE)
                    and not re.search(rf"____\b[^.?!;]*\b{re.escape(req_prep)}\b", stem_str, flags=re.IGNORECASE)
                ):
                    m_after = re.search(r"____\s+([A-Za-z]+)\b", stem_str)
                    after_tok = m_after.group(1).lower() if m_after else ""
                    if after_tok and after_tok not in _other_preps:
                        stem_str = re.sub(r"____", f"____ {req_prep}", stem_str, count=1)
                        _set("question", stem_str)

            # 2. Reconcile blueprint metadata
            tw = spec.get("target_word") or _get("target_word", "")
            if spec.get("target_word"):
                _set("target_word", spec["target_word"])
            if spec.get("definition"):
                _set("definition", spec["definition"])
            if spec.get("options"):
                _set("options", list(spec["options"]))
            if spec.get("correct_answer_index") is not None:
                _set("correct_answer_index", spec["correct_answer_index"])

            # 3. Deterministically hydrate design_audit
            pos = spec.get("part_of_speech") or "word"
            blank_frame = ""
            if "____" in stem_str:
                m_frame = re.search(r"((?:\b[A-Za-z'-]+\b\s+){0,2}____(?:\s+\b[A-Za-z'-]+\b){0,2})", stem_str)
                if m_frame:
                    blank_frame = m_frame.group(1).strip()
            if not blank_frame:
                blank_frame = spec.get("cloze_slot_frame") or "... ____ ..."

            audit_str = f"AUDIT: [{tw}] -> [{pos}] -> [{blank_frame}]"
            cur_audit = str(_get("design_audit") or "").strip()
            if not cur_audit or not cur_audit.startswith("AUDIT:"):
                _set("design_audit", audit_str)

        return quiz_obj

    @staticmethod
    def _render_llm_surface_frame(frame_str: str, skeleton: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        De-metasyntaxify a symbolic `cloze_slot_frame` into a natural-English surface frame
        plus concise human-readable rules for small local LLMs, without mutating internal
        symbolic skeleton fields.
        """
        if not frame_str:
            return "... ____ ...", []
        surface = str(frame_str).strip()
        notes: List[str] = []
        example_txt = str(skeleton.get("example_sentence") or skeleton.get("authentic_example") or "")

        # 1. Expand ontology hints `[category, e.g. sample1 / sample2]` -> concrete sample1 + separate rule
        def _repl_ontology(m: re.Match) -> str:
            cat = m.group(1).strip()
            examples_raw = m.group(2).strip()
            first_ex = re.split(r"\s*/\s*", examples_raw)[0].strip()
            if first_ex:
                notes.append(f"Use a concrete {cat} noun (e.g. '{first_ex}' or similar).")
                return first_ex
            return "the issue"

        surface = re.sub(
            r"\[([^,\]]+),\s*e\.g\.\s*([^\]]+)\]",
            _repl_ontology,
            surface,
            flags=re.IGNORECASE,
        )

        # 2. Replace `to do [sth]` / `to do something` / `to [verb]` with a concrete infinitive or rule
        if re.search(r"\bto\s+(?:do\s+(?:\[sth\]|something)|\[verb\])", surface, flags=re.IGNORECASE):
            m_inf = re.search(r"\bto\s+([a-z]{3,})\b", example_txt.lower())
            inf_v = m_inf.group(1) if (m_inf and m_inf.group(1) not in ("do", "be", "have", "get", "the", "a")) else ""
            if inf_v:
                surface = re.sub(r"\bto\s+(?:do\s+(?:\[sth\]|something)|\[verb\])", f"to {inf_v}", surface, flags=re.IGNORECASE)
            else:
                surface = re.sub(r"\bto\s+(?:do\s+(?:\[sth\]|something)|\[verb\])", "to", surface, flags=re.IGNORECASE)
                notes.append("Follow 'to' with a specific action verb (never write the literal phrase 'to do something').")

        # 3. Replace mandative subjunctive `that [sb] (should) [V-base]` with a natural model
        if re.search(r"that\s+\[sb\]\s*\(should\)\s*\[V-base\]", surface, flags=re.IGNORECASE):
            surface = re.sub(
                r"that\s+\[sb\]\s*\(should\)\s*\[V-base\]",
                "that the committee review",
                surface,
                flags=re.IGNORECASE,
            )
            notes.append("Use a subjunctive base-form verb after 'that' (never write '(should)' or brackets).")

        # 4. Resolve remaining metasyntax brackets and number-neutral slash copulas
        surface = re.sub(r"\[sb\]", "someone", surface, flags=re.IGNORECASE)
        surface = re.sub(r"\[sth\]", "the project", surface, flags=re.IGNORECASE)
        surface = re.sub(r"\[(?:sb/sth|sth/sb)\]", "the issue", surface, flags=re.IGNORECASE)
        surface = re.sub(r"\[V-base\]", "act", surface, flags=re.IGNORECASE)
        surface = re.sub(r"\[noun\]", "system", surface, flags=re.IGNORECASE)
        surface = re.sub(r"\[doing\s+sth\]|\bdoing\s+something\b", "working", surface, flags=re.IGNORECASE)
        surface = re.sub(
            r"\b([a-zA-Z]+)\s+is/are\b",
            lambda m: f"{m.group(1)} {'are' if (m.group(1).lower().endswith('s') and not m.group(1).lower().endswith(('ss', 'us', 'is'))) else 'is'}",
            surface,
            flags=re.IGNORECASE,
        )
        surface = re.sub(
            r"\b([a-zA-Z]+)\s+was/were\b",
            lambda m: f"{m.group(1)} {'were' if (m.group(1).lower().endswith('s') and not m.group(1).lower().endswith(('ss', 'us', 'is'))) else 'was'}",
            surface,
            flags=re.IGNORECASE,
        )
        surface = re.sub(r"\bis/are\b", "is", surface, flags=re.IGNORECASE)
        surface = re.sub(r"\bwas/were\b", "was", surface, flags=re.IGNORECASE)

        # 5. Lock passive auxiliary for VBN targets when example uses passive `be + VBN`
        v_tag = str(skeleton.get("verb_form_tag") or "").upper()
        tw_low = str(skeleton.get("target_word") or "").strip().lower()
        if (
            v_tag == "VBN"
            and tw_low
            and not re.search(r"\b(?:is|are|was|were|be|been|being|has|have|had|get|got)\s+_{2,}", surface, flags=re.IGNORECASE)
            and re.search(rf"\b(?:is|are|was|were|be|been|get|got)\s+(?:\w+\s+)?{re.escape(tw_low)}\b", example_txt, flags=re.IGNORECASE)
        ):
            surface = re.sub(r"____", "was ____", surface, count=1)
            notes.append("Keep a passive auxiliary ('was/were/is/are') directly before '____'.")

        surface = re.sub(r"\s+", " ", surface).strip()
        return surface, notes

    @classmethod
    def hydrate_video_blueprint_items(
        cls,
        quiz_obj: Any,
    ) -> Any:
        """
        Deterministically hydrates design_audit for VideoQuiz items (<0.01ms, 0 tokens):
        - Enforces canonical tag format: `AUDIT: [{timestamp}] -> [Comprehension] -> [Distractor Traps]`.
        """
        if not quiz_obj:
            return quiz_obj

        questions = quiz_obj.get("questions") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "questions", None)
        if not questions or not isinstance(questions, list):
            return quiz_obj

        for idx, q_item in enumerate(questions):
            is_dict = isinstance(q_item, dict)

            def _get(field: str, default: Any = "") -> Any:
                if is_dict:
                    return q_item.get(field, default)
                return getattr(q_item, field, default)

            def _set(field: str, val: Any) -> None:
                if is_dict:
                    q_item[field] = val
                else:
                    setattr(q_item, field, val)

            cur_audit = str(_get("design_audit") or "").strip()
            if not cur_audit or not cur_audit.startswith("AUDIT:"):
                ts = str(_get("timestamp") or f"Segment #{idx + 1}").strip()
                audit_str = f"AUDIT: [{ts}] -> [Comprehension] -> [Distractor Traps]"
                _set("design_audit", audit_str)

        return quiz_obj

