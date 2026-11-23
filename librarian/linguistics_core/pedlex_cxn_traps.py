"""
PedLex construction tagging specs, contrast distractors, and minimal-pair error traps.
"""

import datetime
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import zlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics")


class PedlexCxnTrapsMixin:
    # ------------------------------------------------------------------
    # Section 8: Two-Tier Construction Tagging (`pedlex_cxn_index`)
    # & Single-Point Minimal-Pair Error Traps (`pedlex_error_traps`)
    # ------------------------------------------------------------------
    @classmethod
    def get_word_cxn_specs(
        cls,
        word: str,
        pos: Optional[str] = None,
        sense_idx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve construction specifications (`cxn_spec`) from `pedlex_cxn_index` (or entry JSON fallback).
        When `pos` is provided:
          - strictly filters for matching POS so cross-homograph entries (e.g. 'attempt' noun vs verb) never collide.
        When `sense_idx` is provided:
          - returns entry-level (`sense_idx = -1`) specs plus sense-specific (`sense_idx == sense_idx`) specs.
        When `sense_idx` is None:
          - returns all construction rows registered for `word`."""
        w_clean = (word or "").strip().lower()
        if not w_clean or not cls._ensure_ldoce_conn():
            return []
        p_code = cls._pos_code(pos) if pos else ""
        p_norm = cls.normalize_pos(pos) if pos else ""
        q_words = list(cls.orthographic_variants(w_clean)) or [w_clean]
        # Also resolve multi-word preparatory-object idioms like 'make it possible' -> 'make'
        m_it = re.match(r"^(make|find|keep|leave|consider|think|deem|regard)\s+it\s+[a-z]+\b", w_clean)
        if m_it and m_it.group(1) not in q_words:
            q_words.append(m_it.group(1))

        out: List[Dict[str, Any]] = []
        seen_cxn_sigs: Set[Tuple[str, int, str]] = set()
        if cls._ldoce_table_exists("pedlex_cxn_index"):
            try:
                for qw in q_words:
                    if qw == m_it.group(1) if m_it else False:
                        if out:
                            break
                    sql_conds = ["word = ?"]
                    sql_params: List[Any] = [qw]
                    if p_norm:
                        sql_conds.append("(pos = ? OR pos = '' OR pos = 'other')")
                        sql_params.append(p_norm)
                    if sense_idx is not None:
                        sql_conds.append("(sense_idx = -1 OR sense_idx = ?)")
                        sql_params.append(int(sense_idx))
                    sql_query = (
                        "SELECT subcat_group, family_id, word, pos, sense_idx, cefr_int, zipf_score, "
                        "canonical_pattern, canonical_blank, cloze_frame, contrast_groups, ban_groups "
                        f"FROM pedlex_cxn_index WHERE {' AND '.join(sql_conds)}"
                    )
                    rows = cls._ldoce_conn.execute(sql_query, sql_params).fetchall()
                    for (
                        r_subcat, r_fam, r_word, r_pos, r_sidx, r_cefr, r_zipf,
                        r_pat, r_blank, r_cloze, r_contrasts, r_bans
                    ) in rows:
                        norm_pat = re.sub(rf"\b{re.escape(qw)}\b", w_clean, str(r_pat or ""))
                        sig = (str(r_subcat or ""), int(r_sidx if r_sidx is not None else -1), norm_pat)
                        if sig in seen_cxn_sigs:
                            continue
                        seen_cxn_sigs.add(sig)
                        c_list = [g.strip() for g in (r_contrasts or "").split(",") if g.strip()]
                        b_list = [g.strip() for g in (r_bans or "").split(",") if g.strip()]
                        out.append({
                            "word": w_clean if qw != (m_it.group(1) if m_it else "") else r_word,
                            "sense_idx": r_sidx,
                            "pos": r_pos,
                            "subcat_group": r_subcat,
                            "cxn_family": r_fam,
                            "family_id": r_fam,
                            "cefr_int": r_cefr,
                            "zipf_score": r_zipf,
                            "canonical_pattern": norm_pat,
                            "canonical_blank": r_blank,
                            "cloze_frame": r_cloze,
                            "contrast_groups": c_list,
                            "ban_groups": b_list,
                        })
            except Exception:
                pass
        if not out:
            entry = cls.get_ldoce_entry(w_clean)
            if entry:
                for sp in (entry.get("cxn_specs") or []):
                    if isinstance(sp, dict):
                        spec = dict(sp)
                        spec.setdefault("word", w_clean)
                        spec.setdefault("sense_idx", -1)
                        spec.setdefault("cxn_family", spec.get("family_id", ""))
                        if isinstance(spec.get("contrast_groups"), str):
                            spec["contrast_groups"] = [g.strip() for g in spec["contrast_groups"].split(",") if g.strip()]
                        if isinstance(spec.get("ban_groups"), str):
                            spec["ban_groups"] = [g.strip() for g in spec["ban_groups"].split(",") if g.strip()]
                        out.append(spec)
                for idx, s in enumerate(entry.get("senses") or []):
                    if sense_idx is not None and idx != sense_idx:
                        continue
                    for sp in (s.get("cxn_specs") or []):
                        if isinstance(sp, dict):
                            spec = dict(sp)
                            spec.setdefault("word", w_clean)
                            spec.setdefault("sense_idx", idx)
                            spec.setdefault("cxn_family", spec.get("family_id", ""))
                            if isinstance(spec.get("contrast_groups"), str):
                                spec["contrast_groups"] = [g.strip() for g in spec["contrast_groups"].split(",") if g.strip()]
                            if isinstance(spec.get("ban_groups"), str):
                                spec["ban_groups"] = [g.strip() for g in spec["ban_groups"].split(",") if g.strip()]
                            out.append(spec)
        return out

    @classmethod
    def query_cxn_contrast_distractors(
        cls,
        contrast_groups: List[str],
        ban_groups: Optional[List[str]] = None,
        pos: Optional[str] = None,
        exclude_words: Optional[Set[str]] = None,
        limit: int = 12,
    ) -> List[str]:
        """Execute <0.2ms set-difference SQL query on `pedlex_cxn_index`:
        Selects headwords/phrases in `contrast_groups` that NEVER appear in `ban_groups` across any sense."""
        if not contrast_groups or not cls._ensure_ldoce_conn() or not cls._ldoce_table_exists("pedlex_cxn_index"):
            return []
        bans = [g for g in (ban_groups or []) if g]
        contrasts = [g for g in contrast_groups if g]
        if not contrasts:
            return []
        excl = {w.strip().lower() for w in (exclude_words or set()) if w}
        try:
            c_placeholders = ",".join("?" for _ in contrasts)
            params: List[Any] = list(contrasts)
            pos_clause = ""
            norm_p = cls.normalize_pos(pos) if pos else ""
            if norm_p and norm_p not in ("phrase", "other"):
                pos_clause = " AND pos = ?"
                params.append(norm_p)
            ban_clause = ""
            if bans:
                b_placeholders = ",".join("?" for _ in bans)
                ban_clause = f" AND word NOT IN (SELECT word FROM pedlex_cxn_index WHERE subcat_group IN ({b_placeholders}))"
                params.extend(bans)
            sql = (
                f"SELECT DISTINCT word FROM pedlex_cxn_index "
                f"WHERE subcat_group IN ({c_placeholders}){pos_clause}{ban_clause} "
                f"ORDER BY zipf_score DESC"
            )
            rows = cls._ldoce_conn.execute(sql, params).fetchall()
            cands = [r[0] for r in rows if r and r[0] and r[0].lower() not in excl]
            return cands[:limit]
        except Exception:
            return []

    @classmethod
    def get_word_error_traps(
        cls,
        word: str,
        pos: Optional[str] = None,
        sense_idx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve pre-mined lexical/syntactic minimal-pair error traps from `pedlex_error_traps`
        for `translation_quiz` and `vocabulary_quiz` distractor/error-injection specs.
        Enforces strict POS and `sense_idx` isolation so cross-homograph / cross-sense
        `<gram>` properties (e.g., `interior` adj `[only before noun]` vs `interior` noun,
        or `cram` sense 3 `[intransitive]` vs sense 0 `[transitive]`) never contaminate
        the locked sense."""
        w_clean = (word or "").strip().lower()
        if not w_clean or not cls._ensure_ldoce_conn():
            return []
        p_low = cls.normalize_pos(pos) if pos else ""
        out: List[Dict[str, Any]] = []
        seen_trap_sigs: Set[Tuple[str, int, str]] = set()
        q_vars = list(cls.orthographic_variants(w_clean)) or [w_clean]
        var_alt_re = re.compile(r"\b(?:" + "|".join(re.escape(v) for v in q_vars) + r")\b", re.IGNORECASE)
        if cls._ldoce_table_exists("pedlex_error_traps"):
            try:
                for qv in q_vars:
                    if p_low:
                        rows = cls._ldoce_conn.execute(
                            "SELECT word, pos, sense_idx, trap_type, forbidden_frame, bad_example, good_example, rule_note "
                            "FROM pedlex_error_traps WHERE word = ? AND (pos = ? OR pos = '' OR pos = 'other')",
                            (qv, p_low),
                        ).fetchall()
                    else:
                        rows = cls._ldoce_conn.execute(
                            "SELECT word, pos, sense_idx, trap_type, forbidden_frame, bad_example, good_example, rule_note "
                            "FROM pedlex_error_traps WHERE word = ?",
                            (qv,),
                        ).fetchall()
                    for r in rows:
                        r_pos = str(r[1] or "").strip().lower()
                        r_sidx = r[2]
                        t_type = str(r[3] or "").strip()
                        # Sense-level isolation: sense-specific <gram> traps must match the locked sense_idx
                        if sense_idx is not None and r_sidx is not None and r_sidx >= 0 and r_sidx != sense_idx:
                            continue
                        # Strict POS compatibility guard for <gram> property traps
                        if p_low:
                            if t_type in ("GRAM_ONLY_BEFORE_NOUN", "GRAM_ATTRIBUTIVE_ONLY", "GRAM_NOT_BEFORE_NOUN") and p_low != "adjective":
                                continue
                            if t_type == "GRAM_UNCOUNTABLE" and p_low != "noun":
                                continue
                            if t_type == "GRAM_INTRANSITIVE" and p_low not in ("verb", "phrasal verb"):
                                continue
                        norm_type = "GRAM_ATTRIBUTIVE_ONLY" if t_type == "GRAM_ONLY_BEFORE_NOUN" else t_type
                        norm_forb = var_alt_re.sub(w_clean, str(r[4] or ""))
                        norm_rule = var_alt_re.sub(w_clean, str(r[7] or ""))
                        sig = (norm_type, int(r_sidx if r_sidx is not None else -1), norm_forb or norm_rule)
                        if sig in seen_trap_sigs:
                            continue
                        seen_trap_sigs.add(sig)
                        if t_type in ("GRAM_ONLY_BEFORE_NOUN", "GRAM_ATTRIBUTIVE_ONLY"):
                            c_frame = f"{w_clean} + noun"
                        elif t_type == "GRAM_NOT_BEFORE_NOUN":
                            c_frame = f"be/seem + {w_clean}"
                        elif t_type == "GRAM_UNCOUNTABLE":
                            c_frame = f"{w_clean} [uncountable]"
                        elif t_type == "GRAM_INTRANSITIVE":
                            c_frame = f"{w_clean} [intransitive]"
                        else:
                            c_frame = w_clean
                        out.append({
                            "word": w_clean,
                            "pos": r_pos,
                            "sense_idx": r_sidx,
                            "trap_type": norm_type,
                            "forbidden_frame": norm_forb,
                            "bad_example": r[5],
                            "good_example": r[6],
                            "rule_note": norm_rule,
                            "correct_frame": c_frame,
                            "wrong_frame": r[5] or norm_forb,
                            "warning_note": norm_rule,
                            "authentic_example": r[6],
                            "model_example": r[6],
                        })
            except Exception:
                pass
        # Supplement with deterministic <gram>, valency, and L1 negative-transfer traps when not in grambox table
        if not out:
            entry = cls.get_ldoce_entry(w_clean)
            if entry:
                for s_idx, s in enumerate(entry.get("senses") or []):
                    if sense_idx is not None and s_idx != sense_idx:
                        continue
                    s_pos = str(s.get("pos") or entry.get("pos") or "other").lower()
                    if p_low and s_pos not in (p_low, "other", ""):
                        continue
                    gram = str(s.get("grammar") or "").lower()
                    pats = s.get("patterns") or []
                    exs = s.get("examples") or []
                    ex0 = exs[0] if exs else ""
                    if "only before noun" in gram and s_pos == "adjective":
                        out.append({
                            "word": w_clean, "pos": s_pos, "sense_idx": s_idx,
                            "trap_type": "GRAM_ATTRIBUTIVE_ONLY",
                            "forbidden_frame": f"*be {w_clean}*",
                            "bad_example": f"*The system is {w_clean}.*",
                            "good_example": ex0 or f"a {w_clean} requirement",
                            "rule_note": f"'{w_clean}' is [only before noun] and cannot be used predicatively after 'be'.",
                            "correct_frame": f"{w_clean} + noun",
                            "wrong_frame": f"*be + {w_clean}*",
                            "warning_note": f"'{w_clean}' is [only before noun].",
                            "authentic_example": ex0,
                        })
                    elif "uncountable" in gram and "countable" not in gram.replace("uncountable", "") and s_pos == "noun":
                        out.append({
                            "word": w_clean, "pos": s_pos, "sense_idx": s_idx,
                            "trap_type": "GRAM_UNCOUNTABLE",
                            "forbidden_frame": f"*a/an {w_clean}*, *{w_clean}s*",
                            "bad_example": f"*a good {w_clean}*",
                            "good_example": ex0 or f"some {w_clean}",
                            "rule_note": f"'{w_clean}' is [uncountable] and never takes 'a/an' or a plural '-s'.",
                            "correct_frame": f"{w_clean} [uncountable]",
                            "wrong_frame": f"*a {w_clean} / {w_clean}s*",
                            "warning_note": f"'{w_clean}' is [uncountable].",
                            "authentic_example": ex0,
                        })
                    elif w_clean in ("although", "though", "even though", "whereas", "while"):
                        out.append({
                            "word": w_clean, "pos": "conjunction", "sense_idx": s_idx,
                            "trap_type": "L1_CONJUNCTION_REDUNDANCY",
                            "forbidden_frame": f"*{w_clean.capitalize()} ..., but ...*",
                            "bad_example": f"*{w_clean.capitalize()} it was raining, but we went out.*",
                            "good_example": ex0 or f"{w_clean.capitalize()} it was raining, we went out.",
                            "rule_note": f"Never combine subordinating conjunction '{w_clean}' with coordinating conjunction 'but' in the same sentence.",
                            "correct_frame": f"{w_clean.capitalize()} + Clause, Main Clause",
                            "wrong_frame": f"*{w_clean.capitalize()} + Clause, but Main Clause*",
                            "warning_note": f"Do not use 'but' in a sentence introduced by '{w_clean}'.",
                            "authentic_example": ex0,
                        })
                    elif w_clean in ("despite", "in spite of"):
                        out.append({
                            "word": w_clean, "pos": "preposition", "sense_idx": s_idx,
                            "trap_type": "CLAUSE_PREP_CONFUSION",
                            "forbidden_frame": f"*{w_clean} + finite clause (Subj + V)*",
                            "bad_example": f"*{w_clean.capitalize()} he was tired, he kept working.*",
                            "good_example": ex0 or f"{w_clean.capitalize()} being tired, he kept working.",
                            "rule_note": f"'{w_clean}' is a preposition and must be followed by a noun phrase, gerund (-ing), or 'the fact that', never a bare finite clause.",
                            "correct_frame": f"{w_clean} + NP / V-ing / the fact that",
                            "wrong_frame": f"*{w_clean} + Subject + finite Verb*",
                            "warning_note": f"'{w_clean}' takes a noun phrase or V-ing, not a finite clause.",
                            "authentic_example": ex0,
                        })
                    elif pats and s_pos in ("verb", "adjective"):
                        var_lead_pat = r"^(?:" + "|".join(re.escape(v) for v in q_vars) + r")"
                        for raw_p in pats:
                            p0 = re.sub(r"\s+\betc\.?\b", "", str(raw_p)).strip()
                            if not p0 or len(p0.split()) > 5 or "/" in p0:
                                continue
                            p0_low = p0.lower()
                            if re.search(rf"{var_lead_pat}\s+to\s+do\b", p0_low):
                                norm_p0 = var_alt_re.sub(w_clean, p0)
                                out.append({
                                    "word": w_clean, "pos": s_pos, "sense_idx": s_idx,
                                    "trap_type": "SYNTAX_COMP_TRAP",
                                    "forbidden_frame": f"*{w_clean} + V-ing (gerund)*",
                                    "bad_example": f"*{w_clean} doing something*",
                                    "good_example": ex0 or norm_p0,
                                    "rule_note": f"'{w_clean}' governs a 'to'-infinitive ('{norm_p0}'), never a bare '-ing' gerund (*{w_clean} doing*).",
                                    "correct_frame": norm_p0,
                                    "wrong_frame": f"*{w_clean} + V-ing* (using an `-ing` gerund instead of `to` + base verb)",
                                    "warning_note": f"Required frame: {norm_p0} (never *{w_clean} + V-ing*)",
                                    "authentic_example": ex0,
                                })
                                break
                            elif re.search(rf"{var_lead_pat}\s+doing\b", p0_low):
                                norm_p0 = var_alt_re.sub(w_clean, p0)
                                out.append({
                                    "word": w_clean, "pos": s_pos, "sense_idx": s_idx,
                                    "trap_type": "SYNTAX_COMP_TRAP",
                                    "forbidden_frame": f"*{w_clean} + to-infinitive*",
                                    "bad_example": f"*{w_clean} to do something*",
                                    "good_example": ex0 or norm_p0,
                                    "rule_note": f"'{w_clean}' governs an '-ing' gerund complement ('{norm_p0}'), never a 'to'-infinitive (*{w_clean} to do*).",
                                    "correct_frame": norm_p0,
                                    "wrong_frame": f"*{w_clean} + to-infinitive* (using `to` + base verb instead of `-ing` gerund)",
                                    "warning_note": f"Required frame: {norm_p0} (never *{w_clean} to do*)",
                                    "authentic_example": ex0,
                                })
                                break
                            else:
                                m_gov_prep = re.match(
                                    rf"{var_lead_pat}\s+(in|on|at|to|with|for|from|of|into|about|over|against)\s+(?:somebody|something|sb|sth)\b",
                                    p0_low,
                                )
                                if m_gov_prep:
                                    prep_tok = m_gov_prep.group(1)
                                    alt_prep = {
                                        "in": "on/at", "into": "at/on", "with": "to", "on": "in/of",
                                        "to": "for/with", "of": "from/about", "from": "of", "for": "to",
                                        "about": "on/of", "over": "on", "against": "with", "at": "in/to",
                                    }.get(prep_tok, "to")
                                    norm_p0 = var_alt_re.sub(w_clean, p0)
                                    out.append({
                                        "word": w_clean, "pos": s_pos, "sense_idx": s_idx,
                                        "trap_type": "PREP_COLLOC_TRAP",
                                        "forbidden_frame": f"*{w_clean} {alt_prep}*",
                                        "bad_example": f"*{w_clean} {alt_prep} something*",
                                        "good_example": ex0 or norm_p0,
                                        "rule_note": f"Use '{norm_p0}', not '*{w_clean} {alt_prep}*'.",
                                        "correct_frame": norm_p0,
                                        "wrong_frame": f"*{w_clean} {alt_prep}*",
                                        "warning_note": f"Required frame: {norm_p0}",
                                        "authentic_example": ex0,
                                    })
                                    break
                    if out:
                        break
        return out

