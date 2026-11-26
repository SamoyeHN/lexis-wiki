"""Two-stage WSD logging session and 1-token LLM arbitration mixin."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("librarian.linguistics")


class WsdArbitrationMixin:
    """WSD session logging and Top-K candidate 1-token local LLM arbitration."""

    _wsd_session_local = threading.local()

    @classmethod
    @contextmanager
    def wsd_log_session(cls, unit_name: str = "unit"):
        """
        Context manager that aggregates all Top-K LLM WSD arbitration calls within a unit
        into a single consolidated log file (`logs/<timestamp>_extract_wsd_<unit>_TEXT_MODE.log`).
        Supports re-entrant nesting so outer unit sessions absorb inner vocabulary/expression calls.
        """
        active = getattr(cls._wsd_session_local, "session", None)
        if active is not None:
            if unit_name and unit_name not in ("unit", "vocabulary", "expressions") and active.get("unit_name") in ("unit", "vocabulary", "expressions"):
                active["unit_name"] = unit_name
            yield active
            return

        session = {
            "unit_name": unit_name or "unit",
            "model": None,
            "records": [],
            "start_time": datetime.datetime.now(),
        }
        cls._wsd_session_local.session = session
        try:
            yield session
        finally:
            cls._wsd_session_local.session = None
            records = session.get("records", [])
            # Write consolidated log only when at least one non-cached LLM arbitration ran
            has_new_llm_call = any(not r.get("cached", False) for r in records)
            if records and has_new_llm_call:
                end_time = datetime.datetime.now()
                duration = (end_time - session["start_time"]).total_seconds()
                try:
                    from librarian.config import config
                    from librarian.logger import log_wsd_session
                    log_wsd_session(
                        unit_name=session.get("unit_name") or "unit",
                        model=session.get("model") or config.get("model") or "unknown",
                        records=records,
                        duration=duration,
                        start_time=session["start_time"],
                        end_time=end_time,
                    )
                except Exception:
                    pass

    @classmethod
    def disambiguate_via_llm(
        cls,
        item_name: str,
        sentence: str,
        candidates: List[Dict[str, Any]],
        model: Optional[str] = None,
        discourse_context: str = "",
        target_pos: Optional[str] = None,
    ) -> int:
        """
        Neuro-Symbolic Zero-Hallucination LLM Reranker.
        Given an authentic sentence and 2-4 candidates from PedLex/PedLex,
        prompts the local LLM to select the single best option letter (A/B/C/D).
        Returns chosen candidate index (0-based).
        """
        if len(candidates) <= 1:
            return 0

        from librarian.config import config
        active_model = str(model or config.get("wsd_model") or config.get("judge_model") or config.get("model") or "").strip()

        letters = [chr(ord("A") + i) for i in range(min(len(candidates), 20))]
        unique_poses = {
            str(c.get("pos") or "").strip()
            for c in candidates[:len(letters)]
            if str(c.get("pos") or "").strip()
        }
        eff_pos_target = str(target_pos or "").strip().lower()
        shared_pos = next(iter(unique_poses)) if len(unique_poses) == 1 else ""
        if not eff_pos_target and shared_pos:
            eff_pos_target = shared_pos.lower()

        options_text = []
        cand_sig_parts = []
        for idx, cand in enumerate(candidates[:len(letters)]):
            pos_str = f"({cand.get('pos', '')}) " if (cand.get("pos") and (not shared_pos or len(unique_poses) > 1)) else ""
            raw_s = cand.get("raw_sense") if isinstance(cand.get("raw_sense"), dict) else {}
            defn_str = cand.get('definition', '').strip()
            signpost_raw = str(cand.get("signpost") or raw_s.get("signpost") or "").strip()
            signpost_str = (
                f"[{signpost_raw.upper()}] "
                if signpost_raw and f"[{signpost_raw.lower()}]" not in defn_str.lower()
                else ""
            )
            gram_raw = str(raw_s.get("grammar") or "").strip()
            gram_str = (
                f"{gram_raw} "
                if any(
                    tag in gram_raw.lower()
                    for tag in (
                        "plural",
                        "singular",
                        "only before noun",
                        "not before noun",
                        "uncountable",
                        "countable",
                        "intransitive",
                        "transitive",
                        "disapproving",
                        "approving",
                        "figurative",
                        "technical",
                        "literary",
                        "humorous",
                    )
                )
                and gram_raw.lower() not in defn_str.lower()
                else ""
            )
            pats = list(cand.get('patterns') or ([cand['pattern']] if cand.get('pattern') else []))
            if not pats and raw_s.get("collocations"):
                for col_item in (raw_s.get("collocations") or [])[:2]:
                    col_p = col_item.get("pattern") if isinstance(col_item, dict) else str(col_item or "")
                    if col_p:
                        pats.append(str(col_p))
            pat_str = f" [Pattern: {', '.join(str(p) for p in pats[:2] if p)}]" if pats else ""
            exs = cand.get('examples') or []
            ex_str = f" (e.g. \"{str(exs[0]).strip()}\")" if exs and str(exs[0]).strip() else ""
            options_text.append(f"{letters[idx]}. {pos_str}{signpost_str}{gram_str}{defn_str}{pat_str}{ex_str}")
            cand_sig_parts.append(f"{cand.get('pos', '')}:{signpost_str}{gram_str}{defn_str}:{pat_str}")

        ctx_clean = str(discourse_context or "").strip()
        pos_header = f" ({eff_pos_target or shared_pos})" if (eff_pos_target or shared_pos) else ""
        if ctx_clean and ">>" in ctx_clean and "<<" in ctx_clean:
            context_section = (
                f"Select the single best dictionary sense for '{item_name}'{pos_header}.\n\n"
                f"Passage Context (target sentence in >>...<<):\n"
                f"\"{ctx_clean}\"\n\n"
            )
        else:
            ctx_block = f"Surrounding Context:\n\"{ctx_clean}\"\n\n" if (ctx_clean and ctx_clean != sentence.strip()) else ""
            context_section = (
                f"Select the single best dictionary sense for '{item_name}'{pos_header} as used in this sentence:\n"
                f"\"{sentence}\"\n\n"
                f"{ctx_block}"
            )

        pos_constraint_instruction = ""
        if eff_pos_target:
            pos_constraint_instruction = (
                f"- MANDATORY PART-OF-SPEECH INVARIANT: '{item_name}' functions as a {eff_pos_target.upper()} in this context. "
                f"You MUST select a candidate sense matching this exact part of speech ({eff_pos_target}). "
                f"NEVER choose a verb definition for a noun context or a noun definition for a verb context.\n"
            )

        prompt = (
            f"{context_section}"
            f"Candidate Senses:\n" + "\n".join(options_text) + "\n\n"
            f"Instructions:\n"
            f"{pos_constraint_instruction}"
            f"- Match the candidate whose definition, [SIGNPOST], and [Pattern: ...] best fit how '{item_name}' is used in the target sentence.\n"
            f"- Match the semantic domain of the governing verb or argument in the sentence (e.g., speaking/writing/expressing an argument vs. thinking/remembering vs. seeing/hearing), rather than merely matching a preposition in an example sentence.\n"
            f"- Respect selectional restrictions (e.g., 'if a [subject] ...', animate vs. inanimate arguments) and register/domain tags ([formal], [informal], [spoken], [technical], [disapproving]).\n"
            f"- Prefer core general senses over narrow technical, legal, medical, or [spoken]/[informal] senses unless the context clearly requires them.\n"
            f"Reply with ONLY the single capital letter ({letters[0]}-{letters[-1]}):"
        )

        cache_raw = f"{active_model}|v6|{item_name.strip().lower()}|{eff_pos_target}|{sentence.strip()}|{ctx_clean}|{'||'.join(cand_sig_parts)}"
        cache_key = hashlib.sha1(cache_raw.encode("utf-8")).hexdigest()

        active_session = getattr(cls._wsd_session_local, "session", None)
        if cache_key in cls._wsd_llm_cache:
            cached_idx, cached_raw = cls._wsd_llm_cache[cache_key]
            if 0 <= cached_idx < len(candidates):
                if active_session is not None:
                    active_session["model"] = active_session.get("model") or active_model
                    active_session["records"].append({
                        "item_name": item_name,
                        "sentence": sentence,
                        "discourse_context": ctx_clean,
                        "candidates": candidates,
                        "options_text": options_text,
                        "prompt": prompt,
                        "chosen_idx": cached_idx,
                        "raw_reply": cached_raw,
                        "error": None,
                        "duration": 0.0,
                        "cached": True,
                    })
                return cached_idx

        chosen_idx: Optional[int] = None
        raw = ""
        err_msg = None
        call_dur = 0.0
        t0 = datetime.datetime.now()
        try:
            from librarian.llm import llm, LLMError
            resp = llm.chat(
                [{"role": "user", "content": prompt}],
                model=active_model or None,
                json_format=False,
                temperature=0.0,
                skip_log=True,
            )
            call_dur = (datetime.datetime.now() - t0).total_seconds()
            raw = (resp or "").strip()

            # 1. If wrapped in JSON like {"answer": "A"} or {"choice": "A"}
            parsed_idx = None
            if raw.startswith("{") and raw.endswith("}"):
                try:
                    d = json.loads(raw)
                    for k in ("answer", "choice", "selected", "letter"):
                        val = str(d.get(k, "")).strip().upper()
                        if val in letters:
                            parsed_idx = letters.index(val)
                            break
                    if parsed_idx is None:
                        # If JSON has only one key that is one of the valid letters: {"A": "..."}
                        matching_keys = [k.strip().upper() for k in d.keys() if k.strip().upper() in letters]
                        if len(matching_keys) == 1:
                            parsed_idx = letters.index(matching_keys[0])
                except Exception:
                    pass

            # 2. Plain text / single token matching:
            if parsed_idx is None:
                raw_upper = raw.upper()
                # Guard against refusal sentences ("Neither A nor B", "None of the options") falsely matching letter 'A'
                is_refusal = any(ref_kw in raw_upper for ref_kw in ("NEITHER", "NONE OF", "NO OPTION", "NOT LISTED", "CANNOT"))
                if not is_refusal:
                    m = re.search(r'\b(?:OPTION\s+)?([A-T])\b', raw_upper)
                    if m and m.group(1) in letters:
                        parsed_idx = letters.index(m.group(1))
                    else:
                        for l in letters:
                            if raw_upper.startswith(l):
                                parsed_idx = letters.index(l)
                                break

            if parsed_idx is None:
                raise LLMError(
                    f"WSD LLM arbitration returned unparseable option '{raw[:80]}' for '{item_name}' (valid options: {', '.join(letters)})"
                )
            chosen_idx = parsed_idx
        except Exception as exc:
            call_dur = (datetime.datetime.now() - t0).total_seconds()
            err_msg = str(exc)

        if err_msg is None and chosen_idx is not None:
            cls._wsd_llm_cache[cache_key] = (chosen_idx, raw)
        if active_session is not None:
            active_session["model"] = active_session.get("model") or active_model
            active_session["records"].append({
                "item_name": item_name,
                "sentence": sentence,
                "discourse_context": ctx_clean,
                "candidates": candidates,
                "options_text": options_text,
                "prompt": prompt,
                "chosen_idx": chosen_idx,
                "raw_reply": raw,
                "error": err_msg,
                "duration": call_dur,
                "cached": False,
            })
        if err_msg is not None or chosen_idx is None:
            logger.warning(
                f"WSD LLM arbitration rejected/failed for '{item_name}' (skipping candidate): {err_msg or 'No valid option selected'}"
            )
            return -1
        return chosen_idx


__all__ = ["WsdArbitrationMixin"]
