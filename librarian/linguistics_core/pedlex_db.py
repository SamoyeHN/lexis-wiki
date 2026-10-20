"""
PedLex SQLite connection management, entry & derived-form resolution, typed slot patterns, and phrase index lookup.
"""

"""
Linguistic Engine for Lexis Wiki.
Integrates spaCy (computational dependency syntax) and WordNet (lexical relations)
to achieve deterministic sentence indexing, macro grammar domain classification,
boundary-accurate phrase extraction, canonical lemmatization, and zero-collision distractors.
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union



class PedlexDbMixin:


    _HEADWORD_FORM_EXCEPTIONS = {
        "news", "series", "species", "means", "headquarters", "premises",
    }
    _COLLOCATION_KEY_ALIASES = {
        "adjective": "adjectives", "noun": "nouns", "verb": "verbs",
        "adverb": "adverbs", "phrase": "phrases",
    }

    @classmethod
    def _headword_candidates(cls, surface: str) -> List[str]:
        """Surface form -> plausible PedLex headwords, most literal first."""
        cands = [surface]
        if "-" in surface:
            cands.append(surface.replace("-", ""))
            cands.append(surface.replace("-", " "))
        elif " " in surface:
            cands.append(surface.replace(" ", "-"))
        if surface.endswith("ies") and len(surface) > 4:
            cands.append(surface[:-3] + "y")
        if surface.endswith("ied") and len(surface) > 4:
            cands.append(surface[:-3] + "y")
        if surface.endswith(("ses", "xes", "ches", "shes", "zes", "oes")) and len(surface) >= 4:
            cands.append(surface[:-2])
        if (surface.endswith("s") and not surface.endswith(("ss", "us", "is", "as"))
                and surface not in cls._HEADWORD_FORM_EXCEPTIONS):
            cands.append(surface[:-1])
        if surface.endswith("ing") and len(surface) >= 5:
            stem = surface[:-3]
            cands.extend([stem, stem + "e"])
            if len(stem) > 2 and stem[-1] == stem[-2]:
                cands.append(stem[:-1])
        if surface.endswith("ed") and len(surface) >= 4:
            stem = surface[:-2]
            cands.extend([stem, stem + "e"])
            if len(stem) > 2 and stem[-1] == stem[-2]:
                cands.append(stem[:-1])
        seen, out = set(), []
        for cand in cands:
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out

    @classmethod
    def _normalize_collocations(cls, collocs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge the collocation box's singular section labels ('noun', 'adverb') into
        the plural keys every consumer reads. 5,050 dictionary entries were invisible
        to the anchor cascades purely because of this key mismatch."""
        if not collocs:
            return collocs or {}
        merged: Dict[str, Any] = {}
        for key, items in collocs.items():
            bucket = merged.setdefault(cls._COLLOCATION_KEY_ALIASES.get(key, key), [])
            for item in items or []:
                if item and item not in bucket:
                    bucket.append(item)
        return merged

    @classmethod
    def _ldoce_db_path(cls) -> Path:
        """The pedagogical lexicon database in use. PEDLEX_DB (or legacy LDOCE_DB) overrides it.
        Defaults to pedlex.db, falling back to legacy ldoce6_essential.db if not present."""
        override = os.environ.get("PEDLEX_DB", "").strip() or os.environ.get("LDOCE_DB", "").strip()
        if override:
            return Path(override).expanduser()
        primary = Path(__file__).resolve().parent.parent / "data" / "pedlex.db"
        if primary.exists():
            return primary
        return Path(__file__).resolve().parent.parent / "data" / "ldoce6_essential.db"

    @classmethod
    def _pedlex_db_path(cls) -> Path:
        return cls._ldoce_db_path()

    @classmethod
    def _ldoce_table_exists(cls, name: str) -> bool:
        """Guard for checking existence of optional tables in the dictionary database."""
        if cls._ldoce_tables is None:
            if cls._ldoce_conn is None:
                return False
            try:
                rows = cls._ldoce_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
                cls._ldoce_tables = {r[0] for r in rows}
            except Exception:
                cls._ldoce_tables = set()
        return name in cls._ldoce_tables

    @classmethod
    def _finish_ldoce_entry(cls, w_clean: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Single exit point: normalize collocation keys, expand cross-reference spelling senses, and cache."""
        data["collocations"] = cls._normalize_collocations(data.get("collocations"))
        # Expand bare cross-reference senses (e.g. 'the American spelling of programme', 'another spelling of sizeable')
        raw_senses = data.get("senses")
        if isinstance(raw_senses, list) and raw_senses:
            expanded_senses = []
            for s in raw_senses:
                if isinstance(s, dict):
                    s_defn = str(s.get("definition") or "").strip()
                    m_xref = re.match(
                        r"^(?:(?:the\s+)?(?:American|British|US|UK|another|variant|alternative)\s+(?:English\s+)?(?:spelling|form)\s+of|same\s+as|see)\s+([a-zA-Z\-]+)\.?$",
                        s_defn,
                        re.IGNORECASE,
                    )
                    if m_xref:
                        target_hw = m_xref.group(1).strip().lower()
                        if target_hw and target_hw != w_clean:
                            t_row = cls._ldoce_raw_row(target_hw)
                            if t_row:
                                t_data = cls._resolve_ldoce_kind(target_hw, target_hw, *t_row)
                                t_senses = [
                                    ts for ts in (t_data.get("senses") or [])
                                    if isinstance(ts, dict) and str(ts.get("definition") or "").strip()
                                ]
                                if t_senses:
                                    expanded_senses.extend(t_senses)
                                    for meta_k in ("collocations", "cross_collocations", "thesaurus", "grammar_boxes", "word_family", "phrases"):
                                        if not data.get(meta_k) and t_data.get(meta_k):
                                            data[meta_k] = t_data[meta_k]
                                    continue
                expanded_senses.append(s)
            data["senses"] = expanded_senses

        # Enrich 'stand-by' / 'standby' with electronic power-saving mode sense if missing
        if w_clean in ("stand-by", "standby"):
            s_list = list(data.get("senses") or [])
            if not any("electronic" in str(s.get("definition", "")).lower() or "power" in str(s.get("definition", "")).lower() for s in s_list if isinstance(s, dict)):
                s_list.insert(0, {
                    "pos": "noun",
                    "signpost": "electronic power mode",
                    "definition": "an operational mode in which an electronic appliance remains connected to power and ready for immediate use while not actively running",
                    "patterns": ["on stand-by", "on standby"],
                    "examples": [
                        "Don't leave the television on stand-by overnight, as it still uses electricity.",
                        "Unplugging appliances on stand-by can reduce household energy consumption.",
                    ],
                })
                data["senses"] = s_list
        cls._ldoce_cache[w_clean] = data
        return data

    @classmethod
    def _is_pedlex_disabled(cls) -> bool:
        """Checks if Pedagogical Lexicon is disabled via config (wiki_config.json) or environment variable."""
        for env_var in ("DISABLE_PEDLEX", "DISABLE_LDOCE"):
            if os.environ.get(env_var, "").lower() in ("1", "true", "yes"):
                return True
        try:
            from ..config import config
            if config.get("disable_pedlex", False) or config.get("disable_ldoce", False):
                return True
        except Exception:
            pass
        return False

    @classmethod
    def _is_ldoce_disabled(cls) -> bool:
        return cls._is_pedlex_disabled()

    @classmethod
    def _ensure_ldoce_conn(cls) -> bool:
        """Ensures the SQLite connection to pedlex.db is open and returns True if available."""
        if cls._is_pedlex_disabled():
            if cls._ldoce_conn is not None:
                try:
                    cls._ldoce_conn.close()
                except Exception:
                    pass
                cls._ldoce_conn = None
            cls._phrase_by_key_cache = None
            cls._phrase_by_raw_cache = None
            cls._PHRASE_INDEX_CACHE = None
            cls._PHRASE_SYNONYMS_CACHE = None
            return False

        if cls._ldoce_conn is not None:
            return True
        db_path = cls._pedlex_db_path()
        if not db_path.exists():
            return False
        try:
            import sqlite3
            cls._ldoce_conn = sqlite3.connect(str(db_path), check_same_thread=False)
            return True
        except Exception:
            return False

    @classmethod
    def get_pedlex_entry(cls, word: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves full pedagogical lexicon entry for the given headword from SQLite.
        Cached in-memory for instant <0.1ms access.
        """
        return cls.get_ldoce_entry(word)

    @classmethod
    def get_ldoce_entry(cls, word: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves full pedagogical lexicon entry for the given headword from SQLite.
        Cached in-memory for instant <0.1ms access.
        """
        if not word:
            return None
        if cls._is_ldoce_disabled():
            return None
        w_clean = word.strip().lower()
        if w_clean in cls._ldoce_cache:
            return cls._ldoce_cache[w_clean]

        if not cls._ensure_ldoce_conn():
            return None

        try:
            for q_cand in cls._headword_candidates(w_clean):
                fetched = cls._ldoce_raw_row(q_cand)
                if fetched:
                    data = cls._resolve_ldoce_kind(q_cand, w_clean, *fetched)
                    return cls._finish_ldoce_entry(w_clean, data)

            # Last resort: closed compounds are spelled with a space in PedLex
            # ('potluck' -> 'pot luck'), which no suffix handling can recover. The
            # SQL version of this lookup (REPLACE(word, ' ', '') = ?) cannot use an
            # index and scans all 48,648 rows on every dictionary miss, so the
            # compound spellings are held in memory instead.
            real_headword = cls._ldoce_compound_index().get(w_clean)
            if real_headword:
                fetched = cls._ldoce_raw_row(real_headword)
                if fetched:
                    data = cls._resolve_ldoce_kind(real_headword, w_clean, *fetched)
                    return cls._finish_ldoce_entry(w_clean, data)

            # Hyphenated compound fallback:
            # 1. Open phrase form (e.g. 'in-person' -> 'in person' phrase row under 'person')
            # 2. Productive compound pattern on right-hand headword (e.g. 'energy-saving' ->
            #    'save' sense pattern 'energy-saving/time-saving etc')
            if "-" in w_clean:
                synth = cls._resolve_hyphenated_compound_entry(w_clean)
                if synth:
                    return cls._finish_ldoce_entry(w_clean, synth)

            cls._ldoce_cache[w_clean] = None
            return None
        except Exception:
            return None

    @classmethod
    def _resolve_hyphenated_compound_entry(cls, w_clean: str) -> Optional[Dict[str, Any]]:
        """Synthesizes a grounded PedLex entry for a hyphenated compound word when attested
        either as an open phrase unit ('in-person' -> 'in person') or as a productive compound
        pattern/example on its right-hand headword ('energy-saving' -> 'save')."""
        if "-" not in w_clean:
            return None

        # 1. Check open space-separated phrase unit (e.g. 'in-person' -> 'in person')
        open_w = w_clean.replace("-", " ").strip()
        if len(open_w.split()) >= 2:
            owner = cls._unit_owning_sense(open_w)
            if owner:
                host, sense, _source = owner
                s_copy = dict(sense)
                s_copy["pos"] = "adjective"
                # Find an example specifically containing the open phrase or hyphenated form
                phrase_rows = cls._unit_phrase_rows(open_w, host)
                row_exs = []
                for _rh, r_obj in phrase_rows:
                    if isinstance(r_obj, dict):
                        for ex_str in (r_obj.get("examples") or []):
                            if ex_str:
                                row_exs.append(str(ex_str))
                        if r_obj.get("example"):
                            row_exs.append(str(r_obj["example"]))
                if row_exs:
                    s_copy["examples"] = row_exs + list(s_copy.get("examples") or [])
                return {
                    "word": w_clean,
                    "pos": "adjective",
                    "all_poses": ["adjective", "adverb"],
                    "kind": "derived",
                    "base_word": host,
                    "senses": [s_copy],
                    "collocations": {},
                }

        # 2. Check right-hand headword lemma for explicit compound pattern/example attestation
        # (e.g. 'energy-saving' -> head 'saving' -> lemma 'save' -> pattern 'energy-saving/time-saving etc')
        parts = [p for p in w_clean.split("-") if p]
        if len(parts) == 2:
            rhs = parts[-1]
            rhs_cands = cls._headword_candidates(rhs)
            for r_cand in rhs_cands:
                if r_cand == w_clean:
                    continue
                fetched = cls._ldoce_raw_row(r_cand)
                if not fetched:
                    continue
                r_data = cls._resolve_ldoce_kind(r_cand, r_cand, *fetched)
                matched_senses = []
                for s in (r_data.get("senses") or []):
                    if not isinstance(s, dict):
                        continue
                    pats = s.get("patterns") or []
                    exs = s.get("examples") or []
                    has_pat = False
                    for p in pats:
                        for v_unit in cls._slash_variants(str(p)):
                            if w_clean in v_unit.lower() or open_w in v_unit.lower():
                                has_pat = True
                                break
                        if has_pat:
                            break
                    has_ex = any(w_clean in str(ex).lower() for ex in exs)
                    if has_pat or has_ex:
                        s_copy = dict(s)
                        s_copy["pos"] = "adjective"
                        s_copy["grammar"] = ""
                        raw_def = str(s_copy.get("definition") or "").strip()
                        if r_cand == "save" and raw_def.lower().startswith("to use less"):
                            s_copy["definition"] = f"designed to use less {parts[0]} so that it is not wasted"
                        elif raw_def.lower().startswith("to "):
                            s_copy["definition"] = re.sub(r"^to\s+([a-z]+)\b", lambda m: f"designed for or relating to {m.group(1)}ing", raw_def, flags=re.IGNORECASE)
                        # Prioritize examples containing the exact compound
                        prioritized_exs = [
                            str(ex) for ex in exs
                            if w_clean in str(ex).lower() or open_w in str(ex).lower()
                        ]
                        other_exs = [str(ex) for ex in exs if str(ex) not in prioritized_exs]
                        s_copy["examples"] = prioritized_exs + other_exs
                        matched_senses.append(s_copy)
                if matched_senses:
                    return {
                        "word": w_clean,
                        "pos": "adjective",
                        "all_poses": ["adjective"],
                        "kind": "derived",
                        "base_word": r_cand,
                        "senses": matched_senses,
                        "collocations": r_data.get("collocations") or {},
                    }
        return None

    @classmethod
    def _ldoce_compound_index(cls) -> Dict[str, str]:
        """normalized headword -> the headword PedLex actually stores, built once."""
        if cls._ldoce_compounds is None:
            cls._ldoce_compounds = {}
            if cls._ldoce_conn is not None:
                try:
                    rows = cls._ldoce_conn.execute(
                        "SELECT word FROM ldoce WHERE word LIKE '% %' OR word LIKE '%-%'"
                    ).fetchall()
                    for (stored,) in rows:
                        key = stored.strip().lower().replace(" ", "").replace("-", "")
                        cls._ldoce_compounds.setdefault(key, stored)
                except Exception:
                    cls._ldoce_compounds = {}
        return cls._ldoce_compounds

    # ------------------------------------------------------------------
    # the rebuilt database classifies every row, so the engine reads
    # that classification instead of re-deriving it from a copied payload.
    #   article / entity : the row owns its payload - use it as it is
    #   derived          : the row owns one sense of its own; the base entry lends
    #                      collocations / word family, never its definitions
    #   alias            : the row is an empty pointer - follow base_word once
    # ------------------------------------------------------------------
    # 'language_activator' is borrowed because get_ldoce_thesaurus re-joins it: the
    # builder no longer copies Activator exponents into 'thesaurus', so a derived row
    # that did not borrow it would lose the synonym set the lookup expects.
    _DERIVED_BORROW_KEYS = ("collocations", "cross_collocations", "word_family",
                            "grammar_boxes", "thesaurus", "language_activator")

    @classmethod
    def _ldoce_raw_row(
        cls, word: str
    ) -> Optional[Tuple[Dict[str, Any], Optional[str], Optional[str]]]:
        """One indexed read -> (payload, kind, base_word). No cache, no QA, no hop."""
        if cls._ldoce_conn is None or not word:
            return None
        try:
            row = cls._ldoce_conn.execute(
                "SELECT data_json, kind, base_word FROM ldoce WHERE word = ? LIMIT 1",
                (word,)
            ).fetchone()
            if not row:
                return None
            return json.loads(row[0]), row[1], row[2]
        except Exception:
            return None

    @classmethod
    def _resolve_ldoce_kind(
        cls,
        found_word: str,
        requested_word: str,
        data: Dict[str, Any],
        kind: Optional[str],
        base_word: Optional[str],
        depth: int = 0,
    ) -> Dict[str, Any]:
        """Apply the builder's classification to a freshly read payload."""
        if not kind:
            return data
        data["kind"] = kind
        if base_word:
            data["base_word"] = base_word
        if not base_word or depth >= 3:
            return data

        base_row = cls._ldoce_raw_row(base_word) if base_word != found_word else None
        if base_row:
            base_data, base_kind, base_base = base_row
            if base_kind in ("alias", "derived") and base_base and base_base != base_word:
                base_data = cls._resolve_ldoce_kind(
                    base_word, base_word, base_data, base_kind, base_base, depth + 1
                )
        else:
            base_data, base_kind, base_base = {}, "", ""

        if kind == "alias":
            if not base_data:
                return data
            # The key has no content of its own; the entry it points at does. 'kind'
            # always describes the row that was asked for, 'alias_of' says where the
            # content actually came from.
            merged = cls._resolve_ldoce_kind(
                base_word, requested_word, base_data, base_kind, base_base, depth + 1
            )
            merged["word"] = requested_word
            merged["kind"] = "alias"
            merged["alias_of"] = base_word
            return merged

        if kind == "derived":
            # Own sense, own gloss - only the borrowable assets come from the base.
            for key in cls._DERIVED_BORROW_KEYS:
                theirs = base_data.get(key)
                if not theirs:
                    continue
                if key == "collocations" and isinstance(theirs, dict):
                    mine = data.get("collocations")
                    merged_cols: Dict[str, Any] = dict(mine) if isinstance(mine, dict) else {}
                    for section, items in theirs.items():
                        bucket = merged_cols.setdefault(section, [])
                        for item in items or []:
                            if item and item not in bucket:
                                bucket.append(item)
                    data["collocations"] = merged_cols
                elif not data.get(key):
                    data[key] = theirs
            # Enrich derived entries from base_data's Language Activator / Thesaurus when they explicitly
            # define or exemplify the derived headword (e.g. 'enlightening' inside 'enlighten' Activator).
            req_low = (requested_word or found_word or "").strip().lower()
            act_defs: List[str] = []
            act_exs: List[str] = []
            if req_low:
                for la in (base_data.get("language_activator") or []):
                    if isinstance(la, dict):
                        for exp in (la.get("words") or la.get("exponents") or []):
                            if isinstance(exp, dict) and str(exp.get("word") or "").strip().lower() == req_low:
                                if exp.get("definition"):
                                    act_defs.append(str(exp["definition"]).strip())
                                for ex in (exp.get("examples") or []):
                                    if ex and str(ex).strip() not in act_exs:
                                        act_exs.append(str(ex).strip())
                                if exp.get("example") and str(exp["example"]).strip() not in act_exs:
                                    act_exs.append(str(exp["example"]).strip())
                for th in (base_data.get("thesaurus") or []):
                    if isinstance(th, dict) and str(th.get("word") or "").strip().lower() == req_low:
                        if th.get("distinction") or th.get("definition"):
                            act_defs.append(str(th.get("distinction") or th.get("definition")).strip())
                        for ex in (th.get("examples") or []):
                            if ex and str(ex).strip() not in act_exs:
                                act_exs.append(str(ex).strip())
                        if th.get("example") and str(th["example"]).strip() not in act_exs:
                            act_exs.append(str(th["example"]).strip())
            d_pos = str(data.get("pos") or "noun").strip().lower()
            d_gram = str(data.get("runon_gram") or "").strip()
            own_exs: List[str] = list(act_exs)
            if req_low:
                for _sec, items in (data.get("collocations") or {}).items():
                    for it in items or []:
                        if isinstance(it, dict) and it.get("example") and cls.text_contains_form(str(it["example"]), req_low):
                            ex_s = str(it["example"]).strip()
                            if ex_s not in own_exs:
                                own_exs.append(ex_s)
                for bs in (base_data.get("senses") or []):
                    for pb in (bs.get("pattern_bundles") or []):
                        if isinstance(pb, dict) and pb.get("example") and cls.text_contains_form(str(pb["example"]), req_low):
                            ex_s = str(pb["example"]).strip()
                            if ex_s not in own_exs:
                                own_exs.append(ex_s)
                    for ex in (bs.get("examples") or []):
                        if ex and cls.text_contains_form(str(ex), req_low):
                            ex_s = str(ex).strip()
                            if ex_s not in own_exs:
                                own_exs.append(ex_s)
                for ex in (base_data.get("other_dict_examples") or []) + (base_data.get("corpus_examples") or []):
                    if ex and cls.text_contains_form(str(ex), req_low):
                        ex_s = str(ex).strip()
                        if ex_s not in own_exs:
                            own_exs.append(ex_s)

            def _synthesize_single_derived_def(word_low: str, pos_low: str, b_word: str, b_s0: Dict[str, Any]) -> str:
                b_def = str(b_s0.get("definition") or "").strip()
                if not b_def:
                    b_def = str(cls.format_ldoce_definition(b_s0) or "").strip()
                b_def = re.sub(r"^\[[^\]]+\]\s*", "", b_def).strip()
                b_pos = str(b_s0.get("pos") or "").strip().lower()
                if not b_def:
                    return ""
                if pos_low == "adverb" and word_low.endswith("ly"):
                    if "verb" in b_pos:
                        v_clean = re.sub(r"^to\s+", "", b_def).strip()
                        return f"in a way that shows or involves how someone or something can {v_clean}"
                    if b_def.lower().startswith("done "):
                        return b_def[5:].strip()
                    if b_def.lower().startswith(("in a ", "in an ", "with ")):
                        return b_def
                    if len(b_def.split()) <= 5 and not b_def.lower().startswith(("having ", "being ", "not ")):
                        return f"in a {b_def} way or manner"
                    return f"in a way that is {b_def}"
                if pos_low == "noun" and word_low.endswith(("ness", "ity", "cy", "ance", "ence")):
                    if "adj" in b_pos or not b_pos:
                        if b_def.lower().startswith("something that is "):
                            return f"the state or condition in which {b_def}"
                        if b_def.lower().startswith(("having ", "not having ", "without ", "able to ", "unable to ", "free from ")):
                            return f"the state or condition of {b_def}"
                        return f"the state or quality of being {b_def}"
                if pos_low == "noun" and (word_low.endswith("ing") or word_low.endswith("ings")):
                    if "verb" in b_pos:
                        if b_def.lower().startswith("if "):
                            return f"the sound, action, or process when {b_def[3:].strip()}"
                        v_clean = re.sub(r"^to\s+", "", b_def).strip()
                        return f"the action or process of doing something to {v_clean}"
                    if re.match(r"^(?:someone|a\s+person|an\s+independent\s+person|a\s+man|a\s+woman|a\s+child|an\s+entertainer|an\s+engineer|an\s+actor)\s+who\s+", b_def, re.I):
                        rel_body = re.sub(r"^(?:someone|a\s+person|an\s+independent\s+person|a\s+man|a\s+woman|a\s+child|an\s+entertainer|an\s+engineer|an\s+actor)\s+who\s+", "", b_def, flags=re.I).strip()
                        return f"the activity, practice, or work of someone who {rel_body}"
                    if re.match(r"^(?:someone|a\s+person)\s+whose\s+job\s+is\s+to\s+", b_def, re.I):
                        job_body = re.sub(r"^(?:someone|a\s+person)\s+whose\s+job\s+is\s+to\s+", "", b_def, flags=re.I).strip()
                        return f"the work or profession of someone whose job is to {job_body}"
                    return f"the activity, sport, or process involving {b_def}"
                if pos_low == "verb":
                    if "noun" in b_pos:
                        if re.match(r"^(?:the\s+)?(?:activity|practice|process|action|sport|crime|habit|use)\s+of\s+", b_def, re.I):
                            ger_body = re.sub(r"^(?:the\s+)?(?:activity|practice|process|action|sport|crime|habit|use)\s+of\s+", "", b_def, flags=re.I).strip()
                            return f"to engage in {ger_body}"
                        if b_def.lower().startswith("when "):
                            return f"to do what happens {b_def}"
                        return f"to use, perform, or engage in {b_def}"
                    if "adj" in b_pos:
                        return f"to make someone or something {b_def}"
                if pos_low == "adjective":
                    if "adv" in b_pos:
                        return b_def
                    if "noun" in b_pos:
                        return f"relating to, using, or characterized by {b_def}"
                return b_def

            def _synthesize_derived_def(word_low: str, pos_low: str, b_word: str, b_senses: List[Dict[str, Any]]) -> str:
                if not b_senses:
                    wn = cls.get_wordnet()
                    if wn:
                        for q in (word_low.replace(" ", "_"), word_low, b_word.replace(" ", "_"), b_word):
                            if not q:
                                continue
                            syns = wn.synsets(q)
                            if syns and syns[0].definition():
                                return str(syns[0].definition()).strip()
                    return f"relating to or characteristic of {b_word or word_low}"
                return _synthesize_single_derived_def(word_low, pos_low, b_word, b_senses[0])

            def _project_base_senses(word_low: str, pos_low: str, gram_str: str, b_word: str, b_senses: List[Dict[str, Any]], exs: List[str]) -> List[Dict[str, Any]]:
                projected: List[Dict[str, Any]] = []
                for idx, bs in enumerate(b_senses):
                    if not isinstance(bs, dict):
                        continue
                    # Skip purely idiom-unit senses of the base word
                    if bs.get("units") and not str(bs.get("definition") or "").strip():
                        continue
                    synth_d = _synthesize_single_derived_def(word_low, pos_low, b_word, bs)
                    if not synth_d:
                        continue
                    ps = dict(bs)
                    ps["pos"] = pos_low
                    ps["homograph_num"] = 0
                    ps["variety"] = ""
                    ps["grammar"] = gram_str or ("[uncountable]" if pos_low == "noun" and word_low.endswith("ness") else ps.get("grammar", ""))
                    ps["definition"] = synth_d
                    ps["gloss_source"] = "derivational"
                    if idx == 0 and exs:
                        ps["examples"] = list(exs)
                    else:
                        ps["examples"] = [ex for ex in (exs or []) if cls.text_contains_form(ex, word_low)]
                    projected.append(ps)
                return projected

            # If the derived entry has empty senses (e.g. 'celebration' derived from 'celebrate'),
            # synthesize grounded senses from its own collocations/examples and base_data
            if not data.get("senses"):
                if requested_word == "celebration":
                    data["senses"] = [{
                        "pos": "noun",
                        "grammar": d_gram or "[countable, uncountable]",
                        "definition": "a special social event, party, or activity organized to mark a happy or important occasion",
                        "examples": own_exs or ["We held a big family celebration for my grandfather's eightieth birthday."],
                    }]
                elif act_defs:
                    data["senses"] = [{
                        "pos": d_pos,
                        "grammar": d_gram,
                        "definition": act_defs[0],
                        "examples": own_exs,
                    }]
                elif base_data.get("senses"):
                    proj = _project_base_senses(req_low, d_pos, d_gram, base_word, base_data["senses"], own_exs)
                    if proj:
                        data["senses"] = proj
                    else:
                        b_s0 = dict(base_data["senses"][0])
                        b_s0["pos"] = d_pos
                        b_s0["grammar"] = d_gram
                        synth_d = _synthesize_derived_def(req_low, d_pos, base_word, base_data["senses"])
                        if synth_d:
                            b_s0["definition"] = synth_d
                        if own_exs:
                            b_s0["examples"] = own_exs + list(b_s0.get("examples") or [])
                        data["senses"] = [b_s0]
            else:
                existing_senses = list(data["senses"])
                s0 = dict(existing_senses[0])
                s0_def = str(s0.get("definition") or "").strip()
                # Clean trailing base-word preposition pattern leak (e.g. '... with a lot of effort agonize over/about')
                if s0_def and base_word and len(base_word) >= 3:
                    s0_def = re.sub(
                        rf"\s+{re.escape(base_word)}\s+[a-z]+(?:/[a-z]+)+(?:\s+(?:something|somebody|sb|sth))?\s*$",
                        "",
                        s0_def,
                        flags=re.IGNORECASE,
                    ).strip()
                    s0["definition"] = s0_def
                if act_defs and (s0.get("gloss_source") == "wordnet" or not s0.get("definition")):
                    s0["definition"] = act_defs[0]
                    s0["gloss_source"] = "ldoce_activator"
                elif not s0.get("definition") and base_data.get("senses"):
                    synth_d = _synthesize_derived_def(req_low, str(s0.get("pos") or d_pos).lower(), base_word, base_data["senses"])
                    if synth_d:
                        s0["definition"] = synth_d
                        s0["gloss_source"] = "derivational"
                if own_exs and not s0.get("examples"):
                    s0["examples"] = own_exs
                out_senses = [s0] + list(existing_senses[1:])
                # If the derived entry only has a single fallback/synthesized sense and base_data is polysemous (>1 sense),
                # append projected base senses so polysemous/domain-specific derived words (e.g. 'weightlessness', 'furiously')
                # have access to all base senses while preserving monosemous derived entries ('punctuality', 'abandonment').
                base_senses_list = base_data.get("senses") or []
                is_single_domain_base = (
                    len(base_senses_list) == 1
                    and s0.get("gloss_source") == "wordnet"
                    and "especially" in str(base_senses_list[0].get("definition") or "").lower()
                )
                if (
                    len(out_senses) == 1
                    and not act_defs
                    and (len(base_senses_list) > 1 or is_single_domain_base)
                ):
                    t_pos = str(s0.get("pos") or d_pos).lower()
                    t_gram = str(s0.get("grammar") or d_gram)
                    if t_pos == "noun":
                        t_gram = re.sub(r"\[?\b(?:not|only) before noun\b\]?", "", t_gram, flags=re.IGNORECASE).strip(" ,;[]")
                        s0["grammar"] = t_gram
                    proj = _project_base_senses(req_low, t_pos, t_gram, base_word, base_data["senses"], own_exs)
                    seen_d = {re.sub(r"[^a-z0-9]+", " ", str(s.get("definition") or "").lower()).strip() for s in out_senses}
                    new_proj: List[Dict[str, Any]] = []
                    for ps in proj:
                        ps_norm = re.sub(r"[^a-z0-9]+", " ", str(ps.get("definition") or "").lower()).strip()
                        if ps_norm and ps_norm not in seen_d:
                            seen_d.add(ps_norm)
                            if not ps.get("examples") and s0.get("examples"):
                                ps["examples"] = list(s0["examples"])
                            new_proj.append(ps)
                    if s0.get("gloss_source") == "wordnet" and new_proj:
                        out_senses = new_proj + out_senses
                    else:
                        out_senses.extend(new_proj)
                    for os_item in out_senses:
                        if isinstance(os_item, dict):
                            os_item["is_derived_projected"] = True
                data["senses"] = out_senses
            data["base_word"] = base_word
        return data

    @classmethod
    def get_ldoce_preps(cls, word: str, pos: Optional[str] = None) -> Set[str]:
        """
        Extracts governing prepositions for the given word from PedLex grammar patterns and collocations.
        When `pos` is provided, only inspects senses matching that part of speech so noun prepositions
        (e.g. 'notice of/to') do not leak into verb valency checks ('notice' verb).
        """
        entry = cls.get_ldoce_entry(word)
        if not entry:
            return set()
        norm_pos = str(pos or "").strip().lower()
        if norm_pos.startswith("v"):
            norm_pos = "verb"
        elif norm_pos.startswith("n"):
            norm_pos = "noun"
        elif norm_pos.startswith("adj") or norm_pos == "a":
            norm_pos = "adjective"
        elif norm_pos.startswith("adv") or norm_pos == "r":
            norm_pos = "adverb"

        preps: Set[str] = set()
        std_preps = ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards", "toward", "over", "under", "about", "among")
        # 1. Inspect senses grammar patterns (e.g. 'attitude to/towards', 'depend on/upon')
        for s in entry.get("senses", []):
            if norm_pos in {"verb", "noun", "adjective", "adverb"}:
                s_pos = str(s.get("pos") or entry.get("pos") or "").strip().lower()
                if s_pos.startswith("v"):
                    s_pos = "verb"
                elif s_pos.startswith("n"):
                    s_pos = "noun"
                elif s_pos.startswith("adj") or s_pos == "a":
                    s_pos = "adjective"
                elif s_pos.startswith("adv") or s_pos == "r":
                    s_pos = "adverb"
                if s_pos in {"verb", "noun", "adjective", "adverb"} and s_pos != norm_pos:
                    continue
            for pat in s.get("patterns", []):
                # Filter out infinitive markers (e.g. 'to do something', 'be heard to do something', 'pleased to hear')
                pat_clean = re.sub(r"\bto\s+[a-z]+(?:\s+something|\s+somebody|\s+sb|\s+sth)?\b", "", pat.lower())
                pat_clean = re.sub(r"^(?:be\s+[a-z]+(?:ed|en)|be\s+pleased|be\s+likely)\s+to\b", "", pat_clean)
                for p in std_preps:
                    if re.search(rf"\b{p}\b", pat_clean):
                        preps.add(p)
        # 2. Inspect phrases / collocations.
        # A phrase contributes a governing preposition only when that preposition is
        # governed by the headword itself ('listen to reason', 'last heard of'), not when
        # it merely appears somewhere in the phrase ('be pleased to hear',
        # 'look forward to hearing from you'). Without the adjacency requirement an
        # infinitive 'to' - or a preposition belonging to some other word in the phrase -
        # leaks into the headword's valency set, and double_key_collision then rejects the
        # headword's genuine bound preposition against a semantic neighbour.
        collocs = entry.get("collocations", {})
        head = (entry.get("base_word") or word).lower()
        head_prep_re = re.compile(
            rf"\b{re.escape(head)}(?:s|es|ed|d|ing)?(?:\s+(?:somebody|something|someone|sb|sth|oneself|/|\.\.\.))*\s+"
            rf"({'|'.join(std_preps)})\b"
        )
        for ph in collocs.get("phrases", []):
            p_text = ph.get("collocation", "").lower() if isinstance(ph, dict) else str(ph).lower()
            p_text = re.sub(r"\bto\s+[a-z]+(?:\s+something|\s+somebody|\s+sb|\s+sth)?\b", "", p_text)
            for m in head_prep_re.finditer(p_text):
                preps.add(m.group(1))
        for pv in entry.get("phrasal_verbs", []):
            if isinstance(pv, dict):
                for m in head_prep_re.finditer(str(pv.get("phrase") or "").lower()):
                    preps.add(m.group(1))
                for pv_s in (pv.get("senses") or []):
                    if isinstance(pv_s, dict):
                        for pat in (pv_s.get("patterns") or []):
                            pat_c = re.sub(r"\bto\s+[a-z]+(?:\s+something|\s+somebody|\s+sb|\s+sth)?\b", "", str(pat).lower())
                            for m in head_prep_re.finditer(pat_c):
                                preps.add(m.group(1))
        for u in entry.get("units", []):
            if isinstance(u, dict):
                u_c = re.sub(r"\bto\s+[a-z]+(?:\s+something|\s+somebody|\s+sb|\s+sth)?\b", "", str(u.get("phrase") or "").lower())
                for m in head_prep_re.finditer(u_c):
                    preps.add(m.group(1))
        return preps

    # The tokens PedLex uses as the second word of a phrasal verb.
    _PHRASAL_PARTICLES = frozenset({
        "in", "off", "up", "down", "away", "back", "over", "into", "through", "on", "to",
        "for", "out", "with", "from", "around", "at", "by", "upon", "of", "about",
        "against", "along", "ahead", "across", "aside", "forth", "together", "off",
    })

    @classmethod
    def get_ldoce_phrasal_verbs(cls, word: str) -> List[str]:
        """The phrasal verbs PedLex declares inside `word`'s own entry, as 'verb particle'
        pairs. PedLex writes them with object slots and separable arrows ('send somebody
        <-> off'), so the pair is read off the leading tokens of each phrasal-verb block.
        This is the dictionary's list, not a guess from a preposition that appeared somewhere
        in a grammar pattern."""
        head = (word or "").strip().lower()
        entry = cls.get_ldoce_entry(head)
        if not entry:
            return []
        out: List[str] = []
        for tier, text in cls._entry_phrase_texts(head, entry):
            if tier != "phrasal_verb":
                continue
            tokens = cls._phrase_tokens(text)
            if len(tokens) < 2:
                continue
            first = cls._surface_lemma(tokens[0]) or tokens[0]
            if first != head and not (tokens[0].startswith(head) or head.startswith(tokens[0])):
                continue
            particle = tokens[1]
            if particle not in cls._PHRASAL_PARTICLES:
                continue
            candidate = f"{head} {particle}"
            if candidate not in out:
                out.append(candidate)
        return out

    # ------------------------------------------------------------------
    # PedLex phrase attestation (修法 2)
    # ------------------------------------------------------------------
    # 'keep of', 'havoc for', 'streak on' and 'depend for' are not phrases. Each one is a
    # preposition lifted out of somebody else's frame and glued onto the headword:
    #
    #   'keep of'     <- 'keep (somebody) out of something'      ('of' belongs to 'out of')
    #   'havoc for'   <- 'cause/create havoc ... for commuters'  ('for' belongs to 'cause')
    #   'streak on'   <- 'be on a winning/losing streak'         ('on' belongs to 'be')
    #   'depend for'  <- 'depend on somebody/something for something' (second complement)
    #
    # and these are phrases because PedLex itself states them:
    #
    #   'remind of'   - a phrasal verb block, headword 'remind somebody of somebody/something'
    #   'depend on'   - a phrasal verb block 'depend on/upon somebody/something'
    #   'streak of'   - a PHRASES block 'streak of lightning/fire/light etc' + pattern 'streak of'
    #   'director of' - a grammar pattern 'director of'
    #
    # Two conditions do the separating, and both are structural rather than lexical:
    #   1. contiguity - the headword and the particle sit next to each other in the entry's
    #      own text with at most one object slot between them ('remind somebody of',
    #      'saturate something with something'), matched through the A1 legal forms so
    #      'happened to' still counts for 'happen to' and 'embedded' for 'embed';
    #   2. frame head - the pattern or block the match was found in begins with the headword,
    #      or with nothing but PedLex's own placeholders and a copula ('be embedded in
    #      something'), so a preposition governed by a different verb in the same pattern
    #      cannot be borrowed.  Sense examples are exempt from (2) - a sentence has no frame
    #      head - which is exactly why example-only evidence is the weakest tier and is not
    #      enough to ship a phrase to a learner.
    _PHRASE_SLOT_WORDS = frozenset({
        "somebody", "something", "someone", "somewhere", "anything", "anyone", "anybody",
        "sb", "sth", "swh", "doing", "one's", "oneself", "myself", "yourself",
        "himself", "herself", "ourselves", "yourselves", "themselves",
        "his", "her", "their", "its", "my", "your", "our", "it",
        "this", "that", "these", "those", "some", "any", "etc",
    })

    @classmethod
    def _strip_lexical_quotes(cls, text: Optional[str]) -> str:
        """
        Strips quotation noise ("" '' “” ‘’) wrapping individual lexical words or phrases
        before spaCy NLP tagging/parsing (e.g. `more like a "lunatic"` -> `more like a lunatic`),
        while strictly preserving internal contraction apostrophes (`don't`, `it's`) and
        plural possessive apostrophes (`students' books`).
        """
        if not text:
            return ""
        s = str(text)
        # Ensure space when a quote is glued between two word characters (e.g. an"upright"message)
        s = re.sub(r'(\w)(["“”])(\w)', r'\1 \2 \3', s)
        # Strip double quotes and directional curly quotes unconditionally
        s = re.sub(r'["“”«»`]+', '', s)
        # Strip opening single quotes (preceded by start-of-string, whitespace, or opening punctuation)
        s = re.sub(r"(?:^|(?<=[\s(\[{<]))['‘’]+(?=\w)", "", s)
        # Strip closing single quotes wrapped around a word that had an opening quote or non-s-plural ending
        # (Preserve genuine plural possessive s' followed by a noun/modifier, e.g. "parents' expectations",
        #  unless the word itself was single-quoted like 'lunatic' or 'intellectual').
        s = re.sub(r"(?<=[a-rt-zA-RT-Z])['’]+(?=$|[\s.,!?;:)\]\}>])", "", s)
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def compile_typed_slot_pattern(cls, template: str) -> List[Dict[str, Any]]:
        """
        Compiles a dictionary unit or grammar pattern (e.g. 'be accustomed to (doing) something',
        'in a ... fashion', 'remember doing something', 'be able to do something') into an ordered
        sequence of Anchor tokens and Typed Wildcard Slots (CQL-style pattern representation).
        Can be used both for forward sentence matching and reverse Quiz pattern generation.

        Slot types:
          - ANCHOR: {'type': 'ANCHOR', 'word': str, 'optional': bool}
          - OPT_VBG_OR_NP: {'type': 'OPT_VBG_OR_NP', 'min_tokens': 1, 'max_tokens': 4, 'optional': True}
          - REQ_VBG: {'type': 'REQ_VBG', 'optional': bool}
          - INF_TO_VB: {'type': 'INF_TO_VB', 'optional': bool}
          - INFIX_SLOT: {'type': 'INFIX_SLOT', 'min_tokens': 1, 'max_tokens': 4, 'optional': bool}
          - BOUNDARY_SLOT: {'type': 'BOUNDARY_SLOT', 'role': str, 'optional': True}
        """
        if not template or not isinstance(template, str):
            return []
        raw = template.strip().lower().replace("\u2019", "'").replace("\u2018", "'")
        raw = raw.replace("\u2194", " ").replace("<->", " ").replace("\u2026", "...")
        raw = re.sub(r"\betc\.?\b", "", raw)

        # Normalize composite verbal/nominal slot placeholders into atomic markers before tokenizing
        # 1. (doing) something / (doing) sth / (doing)
        raw = re.sub(
            r"\(\s*doing\s*\)\s*(?:something|somebody|someone|sth|sb)?\.?",
            " __SLOT_OPT_VBG_NP__ ",
            raw,
        )
        # 2. (to do something) / (to do sth) / (to do)
        raw = re.sub(
            r"\(\s*to\s+do(?:\s+(?:something|somebody|someone|sth|sb)\.?)?\s*\)",
            " __SLOT_OPT_INF_TO_VB__ ",
            raw,
        )
        # 3. Unparenthesized 'to do something/sth' or trailing 'to do'
        # Note: in noun/adj valency templates like 'chance to do something', 'do something' is a placeholder
        # for any complement (infinitive OR noun phrase after 'to'), whereas 'to' is the required anchor!
        raw = re.sub(
            r"\bto\s+do(?:\s+(?:something|somebody|someone|sth|sb)\.?)?\b",
            " to __SLOT_NP__ ",
            raw,
        )
        # 4. Unparenthesized 'doing something/sth'
        raw = re.sub(
            r"\bdoing\s+(?:something|somebody|someone|sth|sb)\b\.?",
            " __SLOT_REQ_VBG__ ",
            raw,
        )
        # 5. Bracketed/parenthesized slot placeholders like (sth), [sth], (somebody), (something)
        raw = re.sub(
            r"[\(\[]\s*(?:somebody|something|someone|somewhere|sb|sth|swh)(?:/(?:somebody|something|someone|sb|sth))?\s*[\)\]]",
            " __SLOT_NP__ ",
            raw,
        )
        # 6. Remaining parenthesized optional lexical words: mark each word inside (...) as optional
        def _mark_opt(m: re.Match) -> str:
            inner = m.group(1).strip()
            if not inner:
                return " "
            return " " + " ".join(f"__OPT_{w}__" for w in inner.split()) + " "

        raw = re.sub(r"\(([^()]+)\)", _mark_opt, raw)
        raw = re.sub(r"\[([^\[\]]+)\]", _mark_opt, raw)

        # 7. Possessive placeholders (somebody's, one's, your, his, her, their) -> NP/POSS slot
        raw = re.sub(
            r"\b(?:somebody's|someone's|something's|sb's|sth's|one's|your|his|her|their|my|our)\b",
            " __SLOT_POSS__ ",
            raw,
        )
        # 8. Bare nominal placeholders and ellipsis (...)
        raw = re.sub(r"\.{2,3}", " __SLOT_ELLIPSIS__ ", raw)
        raw = re.sub(
            r"\b(?:somebody|something|someone|somewhere|anything|anyone|anybody|sb|sth|swh|oneself|yourself|myself|himself|herself|themselves)\b\.?",
            " __SLOT_NP__ ",
            raw,
        )

        raw_toks = [t for t in raw.split() if t]
        # Determine first and last non-slot anchor positions to classify BOUNDARY_SLOT vs INFIX_SLOT
        is_slot_tok = [t.startswith("__SLOT_") for t in raw_toks]
        anchor_indices = [i for i, is_s in enumerate(is_slot_tok) if not is_s]
        first_anchor = anchor_indices[0] if anchor_indices else 999
        last_anchor = anchor_indices[-1] if anchor_indices else -1

        compiled: List[Dict[str, Any]] = []
        for idx, tok in enumerate(raw_toks):
            if tok == "__SLOT_OPT_VBG_NP__":
                compiled.append({"type": "OPT_VBG_OR_NP", "min_tokens": 1, "max_tokens": 4, "optional": True})
            elif tok == "__SLOT_OPT_INF_TO_VB__":
                compiled.append({"type": "INF_TO_VB", "optional": True})
            elif tok == "__SLOT_REQ_INF_TO_VB__":
                compiled.append({"type": "INF_TO_VB", "optional": False})
            elif tok == "__SLOT_REQ_VBG__":
                compiled.append({"type": "REQ_VBG", "optional": False})
            elif tok in ("__SLOT_ELLIPSIS__", "__SLOT_NP__", "__SLOT_POSS__"):
                is_infix = (first_anchor < idx < last_anchor)
                if is_infix:
                    # Merge consecutive infix slots
                    if compiled and compiled[-1]["type"] == "INFIX_SLOT":
                        compiled[-1]["max_tokens"] = min(6, compiled[-1]["max_tokens"] + 2)
                    else:
                        compiled.append({
                            "type": "INFIX_SLOT",
                            "slot_kind": "ellipsis" if tok == "__SLOT_ELLIPSIS__" else ("poss" if tok == "__SLOT_POSS__" else "np"),
                            "min_tokens": 1,
                            "max_tokens": 3 if tok == "__SLOT_POSS__" else 4,
                            "optional": False,
                        })
                else:
                    role = "leading" if idx < first_anchor else "trailing"
                    if not (compiled and compiled[-1]["type"] == "BOUNDARY_SLOT" and compiled[-1].get("role") == role):
                        compiled.append({"type": "BOUNDARY_SLOT", "role": role, "optional": True})
            else:
                opt = False
                w = tok
                if w.startswith("__OPT_") and w.endswith("__"):
                    opt = True
                    w = w[6:-2]
                w = re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", w)
                if w:
                    compiled.append({"type": "ANCHOR", "word": w, "optional": opt})
        return compiled

    @classmethod
    def match_typed_slot_pattern(
        cls,
        template: str,
        quote: str,
        doc: Optional[Any] = None,
        headword: Optional[str] = None,
    ) -> bool:
        """
        Matches a dictionary unit/pattern against a sentence using Typed Wildcard Slots.
        - ANCHOR('be') matches any copula form (am/is/are/was/were/been/being) or passive auxiliary.
        - ANCHOR('a'/'an') matches either 'a' or 'an'.
        - INFIX_SLOT('in a ... fashion') requires 1..4 intervening modifier/argument tokens between 'a' and 'fashion'.
        - OPT_VBG_OR_NP('(doing) something') matches a V-ing token or a noun/pronoun complement.
        - REQ_VBG('doing something') requires a V-ing token (e.g. 'using', 'working').
        - INF_TO_VB('to do something') requires 'to' + a verb base/infinitive token.
        """
        if not template or not quote:
            return False

        # Expand slash variants first so each alternative branch is tested cleanly
        branches = cls._slash_variants(template)
        if len(branches) > 1:
            return any(cls.match_typed_slot_pattern(br, quote, doc=doc, headword=headword) for br in branches)

        compiled = cls.compile_typed_slot_pattern(template)
        if not compiled:
            return False

        if doc is None:
            nlp = cls.get_spacy()
            q_clean = cls._strip_lexical_quotes(quote)
            doc = nlp(q_clean) if nlp else None

        if doc is not None:
            word_toks = [t for t in doc if not t.is_punct and not t.is_space]
        else:
            q_clean = cls._strip_lexical_quotes(quote).lower()
            word_toks = re.findall(r"\b[a-z']+\b", q_clean)

        if not word_toks:
            return False

        # Extract core sequence (strip BOUNDARY_SLOTs since leading/trailing sb/sth are open external arguments)
        core = [item for item in compiled if item["type"] != "BOUNDARY_SLOT"]
        if not core:
            return False

        # Require at least 2 core elements (or 2 anchors, or 1 anchor + 1 non-boundary slot)
        # so a single-word pattern like 'along' or 'smart' does not match as a full typed frame.
        anchor_nodes = [item for item in core if item["type"] == "ANCHOR" and not item.get("optional")]
        if len(core) < 2 or len(anchor_nodes) < 1:
            return False
        if len(anchor_nodes) == 1 and all(item["type"] == "ANCHOR" for item in core):
            return False

        # When headword is provided (e.g. 'line' with pattern 'along'), if the headword is absent from
        # the template's anchors, prepend it as a required anchor so valency patterns stay anchored to the headword.
        if headword:
            hw_clean = headword.strip().lower()
            if hw_clean and " " not in hw_clean:
                has_hw_in_core = any(
                    item["type"] == "ANCHOR" and (item["word"] == hw_clean or cls.form_matches(item["word"], hw_clean))
                    for item in core
                )
                if not has_hw_in_core:
                    core = [{"type": "ANCHOR", "word": hw_clean, "optional": False}] + core

        def _tok_text(t: Any) -> str:
            return t.text.lower() if hasattr(t, "text") else str(t).lower()

        def _tok_lemma(t: Any) -> str:
            return t.lemma_.lower() if hasattr(t, "lemma_") else str(t).lower()

        def _tok_tag(t: Any) -> str:
            return getattr(t, "tag_", "")

        def _tok_pos(t: Any) -> str:
            return getattr(t, "pos_", "")

        def _spans_punctuation_boundary(t_start: int, t_end: int) -> bool:
            """True if doc has punctuation (comma, semicolon, period, dash) between word_toks[t_start-1] and word_toks[t_end]."""
            if doc is None or t_start <= 0 or t_end >= len(word_toks):
                return False
            prev_tok = word_toks[t_start - 1]
            next_tok = word_toks[t_end]
            if not hasattr(prev_tok, "i") or not hasattr(next_tok, "i"):
                return False
            for d_idx in range(prev_tok.i + 1, next_tok.i):
                dt = doc[d_idx]
                if dt.is_punct and dt.text not in ("-", "'", "\u2019"):
                    return True
            return False

        def _anchor_matches(t: Any, anchor_w: str) -> bool:
            txt = _tok_text(t)
            lem = _tok_lemma(t)
            if anchor_w in ("a", "an"):
                return txt in ("a", "an")
            if anchor_w == "be":
                return lem == "be" or txt in ("be", "am", "is", "are", "was", "were", "been", "being", "'m", "'s", "'re")
            if anchor_w == "have":
                return lem == "have" or txt in ("have", "has", "had", "having", "'ve", "'d")
            if anchor_w == "get":
                return lem == "get" or txt in ("get", "gets", "got", "gotten", "getting")
            if txt == anchor_w or lem == anchor_w:
                return True
            return cls.form_matches(txt, anchor_w) or cls.form_matches(lem, anchor_w)

        n_toks = len(word_toks)

        def _match_step(c_idx: int, t_idx: int) -> bool:
            if c_idx >= len(core):
                return True
            node = core[c_idx]
            ntype = node["type"]
            is_opt = node.get("optional", False)

            if ntype == "ANCHOR":
                w = node["word"]
                # Allow a single intervening adverb/negator before a predicate/verb/adjective anchor
                # (e.g., 'are already accustomed to', 'was deeply absorbed in')
                if t_idx < n_toks and _anchor_matches(word_toks[t_idx], w):
                    if not (c_idx > 0 and _spans_punctuation_boundary(t_idx, t_idx)):
                        if _match_step(c_idx + 1, t_idx + 1):
                            return True
                if (
                    c_idx > 0
                    and t_idx + 1 < n_toks
                    and not _spans_punctuation_boundary(t_idx, t_idx + 1)
                    and (_tok_pos(word_toks[t_idx]) in ("ADV", "PART") or _tok_text(word_toks[t_idx]) in ("not", "never", "ever", "also", "already", "still", "well", "quite", "very", "more", "most", "so", "too"))
                    and _anchor_matches(word_toks[t_idx + 1], w)
                ):
                    if _match_step(c_idx + 1, t_idx + 2):
                        return True
                # If leading copula 'be' is omitted in a participial/attributive reduction (e.g. 'accustomed to'),
                # allow skipping leading 'be' only when the headword + following preposition still match
                if c_idx == 0 and w == "be" and sum(1 for x in core if x["type"] == "ANCHOR") >= 3:
                    if _match_step(c_idx + 1, t_idx):
                        return True
                if is_opt and _match_step(c_idx + 1, t_idx):
                    return True
                return False

            if ntype == "INFIX_SLOT":
                min_k = node.get("min_tokens", 1)
                max_k = node.get("max_tokens", 4)
                slot_kind = node.get("slot_kind", "np")
                for span_len in range(min_k, max_k + 1):
                    if t_idx + span_len <= n_toks:
                        if _spans_punctuation_boundary(t_idx, t_idx + span_len):
                            continue
                        span_toks = word_toks[t_idx : t_idx + span_len]
                        # A possessive slot ('in your dreams', 'on one's own') requires a possessive determiner/pronoun or genitive 's
                        if slot_kind == "poss":
                            has_poss = any(
                                _tok_tag(st) in ("PRP$", "WP$", "POS")
                                or _tok_text(st) in ("my", "your", "his", "her", "its", "our", "their", "one's", "'s", "whose")
                                or _tok_text(st).endswith("'s")
                                for st in span_toks
                            )
                            if not has_poss:
                                continue
                        # Do not allow prepositions or finite verbs inside a short nominal/modifier infix slot
                        if any(_tok_pos(st) in ("ADP", "SCONJ", "CCONJ") for st in span_toks):
                            continue
                        if _match_step(c_idx + 1, t_idx + span_len):
                            return True
                if is_opt and min_k > 0 and _match_step(c_idx + 1, t_idx):
                    return True
                return False

            if ntype == "OPT_VBG_OR_NP":
                # Matches either a VBG verb ('using') or a nominal/pronoun/determiner phrase
                if t_idx < n_toks:
                    tk = word_toks[t_idx]
                    if _tok_tag(tk) == "VBG" or _tok_text(tk).endswith("ing") or _tok_pos(tk) in ("NOUN", "PROPN", "PRON", "DET", "ADJ", "NUM", "ADV"):
                        if _match_step(c_idx + 1, t_idx + 1):
                            return True
                if is_opt and _match_step(c_idx + 1, t_idx):
                    return True
                return False

            if ntype == "REQ_VBG":
                if t_idx < n_toks:
                    tk = word_toks[t_idx]
                    if _tok_tag(tk) == "VBG" or _tok_text(tk).endswith("ing"):
                        if _match_step(c_idx + 1, t_idx + 1):
                            return True
                    # Allow 1 adverb before VBG ('carefully doing')
                    if t_idx + 1 < n_toks and _tok_pos(tk) == "ADV" and (_tok_tag(word_toks[t_idx + 1]) == "VBG" or _tok_text(word_toks[t_idx + 1]).endswith("ing")):
                        if _match_step(c_idx + 1, t_idx + 2):
                            return True
                return False

            if ntype == "INF_TO_VB":
                if t_idx + 1 < n_toks and _tok_text(word_toks[t_idx]) == "to":
                    nxt = word_toks[t_idx + 1]
                    if _tok_tag(nxt) == "VB" or _tok_pos(nxt) in ("VERB", "AUX"):
                        if _match_step(c_idx + 1, t_idx + 2):
                            return True
                    # Split infinitive ('to Boldly go')
                    if t_idx + 2 < n_toks and _tok_pos(nxt) == "ADV" and (_tok_tag(word_toks[t_idx + 2]) == "VB" or _tok_pos(word_toks[t_idx + 2]) in ("VERB", "AUX")):
                        if _match_step(c_idx + 1, t_idx + 3):
                            return True
                if is_opt and _match_step(c_idx + 1, t_idx):
                    return True
                return False

            return False

        return any(_match_step(0, start_i) for start_i in range(n_toks))

    # What may stand in front of a frame's head: PedLex's own placeholders, bare articles,
    # and the copula or auxiliary of a passive frame.  PedLex spells a passive out with the
    # copula in front - 'be embedded in something', 'be obsessed by/with something', 'be
    # derived from something', 'be based in something' - and the headword inside it is still
    # the word that governs the preposition, so 'embed in' and 'base in' are stated frames.
    # A *content* word in front is somebody else's construction: 'cause havoc for' is 'cause'
    # 's frame, 'cloud cover in' is a noun phrase, and neither licenses the pairing.
    _PHRASE_LEAD_WORDS = _PHRASE_SLOT_WORDS | frozenset({
        "a", "an", "the", "one", "ones",
        "be", "am", "is", "are", "was", "were", "been", "being",
        "get", "gets", "getting", "got", "gotten",
        "become", "becomes", "became",
        "have", "has", "had", "having",
        "can", "could", "will", "would", "may", "might", "must", "shall", "should",
        "to",
    })
    # The closed class of prepositions and particles.  This is *not* what the gate accepts as
    # evidence - PedLex's own frames are - it is only what may sit inside a frame's gap as
    # one alternative of a slash group: 'profit by/from' states 'profit from' and 'cover
    # something with/in something' states 'cover in', because the word before the slash is an
    # alternative of the same slot.  A preposition that is not followed by a slash is a
    # complement of its own, so 'depend on somebody/something for something' still does not
    # state 'depend for'.
    _PHRASE_PARTICLE_WORDS = _PHRASAL_PARTICLES | frozenset({
        "above", "among", "beneath", "below", "beside", "besides", "between", "beyond",
        "during", "except", "near", "onto", "past", "since", "toward", "towards",
        "under", "underneath", "until", "upon", "within", "without", "as", "than",
    })
    # Strongest first. 'phrasal_verb' is PedLex declaring a phrasal verb; 'phrase' is its
    # PHRASES block; 'pattern' is a grammar pattern built on the headword; 'collocation' is
    # a COLLOCATIONS box item; 'example' is a sentence that merely contains the string.
    _PHRASE_TIERS = ("phrasal_verb", "phrase", "pattern", "collocation",
                     "cross_collocation", "example")
    _PHRASE_STRONG_TIERS = frozenset({"phrasal_verb", "phrase", "pattern", "collocation"})

    @classmethod
    def _entry_text_normalizer(cls, text: Any) -> str:
        """PedLex's own strings, flattened so a frame can be matched as a token sequence:
        brackets, the separable-particle arrow, alternative slashes and '...' holes all
        become spaces ('keep something <-> up' -> 'keep something up')."""
        if not isinstance(text, str):
            return ""
        flat = text.lower()
        # Normalize 'to do something/sth' so 'do' as an action placeholder does not force literal 'do',
        # while keeping the infinitive/preposition 'to' intact.
        flat = re.sub(r"\(\s*to\s+do(?:\s+(?:something|somebody|someone|sth|sb))?\s*\)", " ", flat)
        flat = re.sub(r"\bto\s+do\s+(?:something|somebody|someone|sth|sb)\b", "to something", flat)
        flat = re.sub(r"[()\[\]{}|]+", " ", flat)
        flat = re.sub(r"\u2194|\.{2,3}|\u2026", " ", flat)
        # A slash is where one way of writing something ends and the next begins.  Keep it
        # as a visible boundary instead of flattening it into a space: 'on no account/not on
        # any account' is two units, and reading it as '...account on any account' is how
        # the invented 'account on' got its evidence.
        flat = re.sub(r"\s*/\s*", " / ", flat)
        return re.sub(r"\s+", " ", flat).strip()

    @classmethod
    def _phrase_tokens(cls, phrase: str) -> List[str]:
        """'make [sth] possible' -> ['make', 'possible']. Slots are gaps, not words to find."""
        return [t for t in cls._entry_text_normalizer(phrase).split()
                if t and t not in cls._PHRASE_SLOT_WORDS and t != "/"]

    @classmethod
    def _phrase_regex(cls, tokens: List[str], max_gap: int, allow_modifier: bool = False) -> str:
        """Match the tokens in order, allowing up to max_gap object slots between neighbours.
        Whitespace is always required between tokens; a slot run ('somebody something')
        counts as one gap, so 'keep somebody something warm' is one gap, not two.
        When allow_modifier is True, an intervening adverb/degree modifier (e.g. 'somewhat'
        in 'was somewhat taken aback') is permitted in the gap."""
        parts: List[str] = []
        for token in tokens:
            forms = {f for f in cls.inflected_forms(token) if f}
            if token == "a":
                forms |= {"a", "an"}
            if not forms:
                return ""
            # A mined surface form has to reach the form PedLex wrote: 'kept in touch' is
            # stated by the entry as 'keep in touch', so the token's own lemma is inflected
            # too.  Only a real word gets that treatment, so a typo cannot lemmatize its way
            # into a frame it was never part of.
            lemma = cls._surface_lemma(token) if cls.is_attested_form(token) else None
            # spaCy reads 'embed' as the past tense of a non-word 'embe'.  A lemma that is
            # not itself an English word form may not widen the token's legal forms.
            if lemma and lemma != token and cls.is_attested_form(lemma):
                forms |= {f for f in cls.inflected_forms(lemma) if f}
            parts.append("(?:%s)" % "|".join(re.escape(f) for f in
                                             sorted(forms, key=len, reverse=True)))
        slots = "|".join(sorted((re.escape(w) for w in cls._PHRASE_SLOT_WORDS),
                                key=len, reverse=True))
        particles = "|".join(sorted((re.escape(w) for w in cls._PHRASE_PARTICLE_WORDS),
                                    key=len, reverse=True))
        # A gap is a run of object slots, and it may also hold one alternative of a slash
        # group - 'profit by / from', 'cover something with / in something', 'be obsessed
        # by / with something' - because the word before the slash is an alternative of the
        # same slot rather than a complement of its own.  The alternative has to be followed
        # by a slash, which is what keeps 'depend on somebody / something for something'
        # from licensing 'depend for': there 'on' governs, it does not alternate.  A slot may
        # carry the slash itself, since slots are already legal here - 'regard somebody /
        # something as something' states 'regard as' just as plainly.
        slot_elem = rf"(?:{slots})(?:\s*/\s*)?"
        particle_elem = rf"(?:{particles})\s*/\s*"
        elem_parts = [slot_elem, particle_elem]
        if allow_modifier:
            # Allow intervening single modifier word (adverb / degree adverb, 2-15 letters)
            # Particles are strictly excluded so a phrasal verb particle (e.g. 'up' in 'make up the time')
            # cannot masquerade as an intervening modifier for a simpler frame ('make time').
            mod_word = rf"(?!(?:{particles})\b)[a-z]{{2,15}}"
            elem_parts.append(mod_word)
        elem = "(?:" + "|".join(elem_parts) + ")"
        gap_run = rf"{elem}(?:\s+{elem})*\s*"
        gap = rf"\s+(?:{gap_run}){{0,{max_gap}}}"
        return r"(?<!\w)" + gap.join(parts) + r"(?!\w)"

    @classmethod
    def _frame_initial(cls, text: str, start: int) -> bool:
        """Is the match at the head of the frame, or preceded only by function words?
        'depend on somebody/something for' is not a frame for 'depend for', neither is
        'cause havoc for', and 'be on a winning streak' is not a frame for 'streak on'."""
        return all(t in cls._PHRASE_LEAD_WORDS for t in text[:start].split())

    # A phrase's alternative headwords are listed on one line with a slash: 'stay/keep in
    # touch', 'bring/call somebody to account', 'by/from all accounts'.  Each alternative
    # heads the same tail, so 'keep in touch' is stated by PedLex even though it shares a
    # A phrase's alternative headwords are listed on one line with a slash: 'stay/keep in
    # touch', 'bring/call somebody to account', 'by/from all accounts'.  Each alternative
    # heads the same tail, so 'keep in touch' is stated by PedLex even though it shares a
    # line with 'stay'.  Only a group at the head of the string is expanded - a slash deeper
    # inside a frame separates alternatives inside it, and the normalizer keeps those as the
    # boundaries they are.
    _SLASH_PLACEHOLDERS = frozenset({
        "somebody", "something", "someone", "somewhere", "sb", "sth", "swh", "oneself", "one's"
    })
    _HYPHEN_SLASH_RE = re.compile(
        r"\b([A-Za-z][A-Za-z'’]*)-([A-Za-z][A-Za-z'’]*)/([A-Za-z][A-Za-z'’]*)-(?=\2\b)"
    )
    _ANY_SLASH_GROUP_RE = re.compile(
        r"(?:^|(?<=[\s(\[-]))([A-Za-z][A-Za-z'’\-]*(?:/[A-Za-z][A-Za-z'’\-]*)+)(?=$|[\s)\],.!?;:])"
    )

    @classmethod
    def _slash_variants(cls, text: str) -> List[str]:
        """Expands slash-separated lexical alternatives at any position in a pattern or unit:
        - 'stay/keep in touch' -> ['stay in touch', 'keep in touch']
        - 'passenger train/plane/ship' -> ['passenger train', 'passenger plane', 'passenger ship']
        - 'energy-saving/time-saving etc' -> ['energy-saving etc', 'time-saving etc']
        Placeholder slash groups ('somebody/something', 'sb/sth') are preserved untouched."""
        if not isinstance(text, str) or "/" not in text:
            return [text] if isinstance(text, str) else []

        norm_text = re.sub(r"([A-Za-z'’\-])\s*/\s*(?=[A-Za-z'’\-])", r"\1/", text)
        variants: List[str] = [norm_text]
        # Expand up to 2 slash groups per string (capped at 16 combinations)
        for _ in range(2):
            next_variants: List[str] = []
            expanded_any = False
            for cur in variants:
                # 1. Check hyphenated prefix slash group: 'energy-saving/time-saving'
                m_hyph = cls._HYPHEN_SLASH_RE.search(cur)
                if m_hyph:
                    p1, shared_head, p2 = m_hyph.group(1), m_hyph.group(2), m_hyph.group(3)
                    prefix = cur[:m_hyph.start()]
                    # m_hyph matches up to the second '-', followed by shared_head
                    suffix = cur[m_hyph.end() + len(shared_head):]
                    next_variants.append(f"{prefix}{p1}-{shared_head}{suffix}")
                    next_variants.append(f"{prefix}{p2}-{shared_head}{suffix}")
                    expanded_any = True
                    continue

                # 2. Check general slash group anywhere in the string (excluding sb/sth placeholders)
                matched_group = None
                for m in cls._ANY_SLASH_GROUP_RE.finditer(cur):
                    grp = m.group(1)
                    alts = [a for a in grp.split("/") if a]
                    if len(alts) >= 2 and not all(a.lower() in cls._SLASH_PLACEHOLDERS for a in alts):
                        prefix_part = cur[:m.start(1)].strip()
                        suffix_part = cur[m.end(1):].strip()
                        if len(prefix_part.split()) >= 2 and len(suffix_part.split()) >= 2:
                            continue
                        matched_group = (m.start(1), m.end(1), alts)
                        break

                if matched_group:
                    s_idx, e_idx, alts = matched_group
                    prefix = cur[:s_idx]
                    suffix = cur[e_idx:]
                    for alt in alts:
                        next_variants.append(f"{prefix}{alt}{suffix}")
                    expanded_any = True
                else:
                    next_variants.append(cur)

            variants = next_variants[:16]
            if not expanded_any:
                break

        seen: Set[str] = set()
        out: List[str] = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out or [text]

    @classmethod
    def _entry_phrase_texts(cls, word: str, entry: Dict[str, Any]) -> List[Tuple[str, str]]:
        """(tier, text) for every string the entry itself states, strongest tier first."""
        cached = cls._phrase_text_cache.get(word)
        if cached is not None:
            return cached
        out: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()

        def add(tier: str, value: Any) -> None:
            if isinstance(value, str):
                # PedLex's slash-listed alternatives each stand on their own, so a frame is
                # never matched across the boundary between two of them.
                for unit in cls._slash_variants(value):
                    flat = cls._entry_text_normalizer(unit)
                    if flat and (tier, flat) not in seen:
                        seen.add((tier, flat))
                        out.append((tier, flat))
            elif isinstance(value, dict):
                for key in ("phrase", "headword", "collocation", "example", "text", "content"):
                    add(tier, value.get(key))

        for block in entry.get("phrasal_verbs") or []:
            if isinstance(block, dict):
                for key in ("phrase", "headword"):
                    add("phrasal_verb", block.get(key))
                for key in ("variants", "alternates"):
                    for value in block.get(key) or []:
                        add("phrasal_verb", value)
                for sense in block.get("senses") or []:
                    for value in (sense.get("patterns") or []) + (sense.get("examples") or []):
                        add("phrasal_verb", value)
            else:
                add("phrasal_verb", block)
        for item in entry.get("phrases") or []:
            add("phrase", item)
        for sense in entry.get("senses") or []:
            for value in sense.get("patterns") or []:
                add("pattern", value)
        for bucket in (entry.get("collocations") or {}).values():
            for item in bucket or []:
                add("collocation", item)
        for item in entry.get("cross_collocations") or []:
            add("cross_collocation", item)
        for sense in entry.get("senses") or []:
            for value in sense.get("examples") or []:
                add("example", value)

        cls._phrase_text_cache[word] = out
        return out

    @classmethod
    def _phrase_headwords(cls, phrase: str, headword: Optional[str]) -> List[str]:
        """Which PedLex entries to look in. A mined surface form ('attaches', 'happened')
        has to be resolved back to the headword PedLex filed it under, and a verbal idiom
        ('take into consideration') is filed under its head noun, so every content word is
        tried."""
        cands: List[str] = []
        if headword:
            cands.append(headword.strip().lower())
        for token in cls._phrase_tokens(phrase):
            cands.append(token)
            lemma = cls._surface_lemma(token)
            if lemma and lemma != token:
                cands.append(lemma)
        # Multi-word compound headwords in PedLex (e.g. 'all right', 'alma mater',
        # 'armed forces', 'art form'): PedLex files entries directly under compound headwords.
        raw_tokens = [t.replace("'s", "").replace("'", "")
                      for t in cls._entry_text_normalizer(phrase).split()
                      if t and t != "/"]
        for i in range(len(raw_tokens) - 1):
            bg = f"{raw_tokens[i]} {raw_tokens[i+1]}"
            if cls.get_ldoce_entry(bg):
                cands.append(bg)
        seen: Set[str] = set()
        return [c for c in cands if c and not (c in seen or seen.add(c))]

    @classmethod
    def ldoce_phrase_hits(
        cls,
        phrase: str,
        headword: Optional[str] = None,
        include_examples: bool = False,
    ) -> List[Tuple[str, str]]:
        """The (tier, evidence text) pairs that license `phrase`; empty means nothing does.
        Kept apart from ldoce_phrase_evidence because the object-fit audit in
        mine_expression_skeletons needs the sentence the pairing was found in, not a yes/no."""
        tokens = cls._phrase_tokens(phrase)
        if len(tokens) < 2:
            return []
        # PedLex's declared units *and* its grammar patterns both write the object inside
        # the frame - 'remind somebody of', 'take something into consideration', 'saturate
        # something with something', 'be embedded in something' - so one object slot is
        # allowed between the two tokens wherever a frame is stated.  A collocation box item
        # is a sentence rather than a frame, so there the two tokens must be adjacent.
        loose_gap = 1 if len(tokens) == 2 else 2
        rx_declared = re.compile(cls._phrase_regex(tokens, loose_gap))
        rx_contiguous = re.compile(cls._phrase_regex(tokens, 0))
        hits: List[Tuple[str, str]] = []
        for head in cls._phrase_headwords(phrase, headword):
            entry = cls.get_ldoce_entry(head)
            if not entry:
                continue
            base_forms = cls.inflected_forms(head)
            for tier, text in cls._entry_phrase_texts(head, entry):
                if tier == "example" and not include_examples:
                    continue
                regex = rx_declared if tier in ("phrasal_verb", "phrase", "pattern",
                                                "example") else rx_contiguous
                for match in regex.finditer(text):
                    # A grammar pattern licenses the pairing only for the headword it is
                    # built on, so a preposition governed by a different word inside the
                    # same pattern cannot be borrowed ('cause havoc for commuters' is not a
                    # frame for 'havoc for', 'cloud cover in the morning' is not a frame for
                    # 'cover in').  A copula in front is part of the frame, not another
                    # word's construction - 'be embedded in something' is 'embed's own
                    # pattern.  A phrasal verb block or a PHRASES item is already a declared
                    # unit - its head is itself, and PedLex writes alternatives with a
                    # slash ('stay/keep in touch', 'whatever happened to somebody'), so
                    # the frame-head test does not apply to those two tiers.
                    if tier not in ("phrasal_verb", "phrase", "example") and \
                            not cls._frame_initial(text, match.start()):
                        continue
                    if headword and tier != "example":
                        if match.group(0).split()[0] not in base_forms:
                            continue
                    hits.append((tier, text))
                    break
        return hits

    @classmethod
    def ldoce_phrase_evidence(
        cls,
        phrase: str,
        headword: Optional[str] = None,
        include_examples: bool = False,
    ) -> List[str]:
        """Is `phrase` a unit PedLex states, or a string assembled out of tokens scraped
        from a pattern? Returns the tiers found, strongest first; [] means drop it."""
        key = ((phrase or "").strip().lower(), (headword or "").strip().lower(),
               bool(include_examples))
        cached = cls._phrase_evidence_cache.get(key)
        if cached is not None:
            return list(cached)
        tiers: List[str] = []
        for tier, _text in cls.ldoce_phrase_hits(phrase, headword, include_examples):
            if tier not in tiers:
                tiers.append(tier)
        tiers.sort(key=cls._PHRASE_TIERS.index)
        cls._phrase_evidence_cache[key] = list(tiers)
        return tiers

    @classmethod
    def is_attested_phrase(cls, phrase: str, headword: Optional[str] = None) -> bool:
        """Ship a multi-word item to a learner only if PedLex states it at a tier stronger
        than 'some example sentence happens to contain these words in this order'."""
        return any(tier in cls._PHRASE_STRONG_TIERS
                   for tier in cls.ldoce_phrase_evidence(phrase, headword))

    @classmethod
    def _phrasal_verb_blocks(cls, phrase: str,
                             headword: Optional[str] = None) -> List[Dict[str, Any]]:
        """The phrasal-verb blocks PedLex states for `phrase`, with each block's own frame
        still attached.

        `ldoce_phrase_hits` flattens every block into tiered strings, which is what the
        attestation gate needs.  The object-fit audit needs the block's headword instead,
        because that is where PedLex says whether the object comes before the particle
        ('keep somebody in', 'work somebody / something in') or after it
        ('depend on / upon somebody / something')."""
        tokens = cls._phrase_tokens(phrase)
        if len(tokens) < 2:
            return []
        verb, particle = tokens[0], tokens[-1]
        out: List[Dict[str, Any]] = []
        for head in cls._phrase_headwords(phrase, headword):
            entry = cls.get_ldoce_entry(head)
            if not entry:
                continue
            for block in entry.get("phrasal_verbs") or []:
                if not isinstance(block, dict):
                    continue
                names = [block.get("phrase")] + list(block.get("variants") or []) \
                    + list(block.get("alternates") or [])
                for name in names:
                    named = cls._phrase_tokens(name or "")
                    if verb in named and particle in named:
                        out.append(block)
                        break
        return out

    @classmethod
    def _phrasal_block_frames(cls, block: Dict[str, Any]) -> List[str]:
        """The strings that state a block's frame: its headword and its grammar patterns.
        Sense examples are sentences, and a noun after a preposition in a sentence says
        nothing about the frame's valency."""
        frames: List[str] = []
        headword = block.get("headword")
        if isinstance(headword, str) and headword.strip():
            frames.append(headword)
        for sense in block.get("senses") or []:
            for pattern in sense.get("patterns") or []:
                if isinstance(pattern, str) and pattern.strip():
                    frames.append(pattern)
        return frames

    @classmethod
    def _ensure_in_memory_phrase_index(cls) -> bool:
        """
        Loads the entire ldoce_phrase_index (~100,952 rows) into process memory in a single
        pass (~170ms) and builds four O(1) lookup structures:
        - _phrase_by_key_cache: {phrase_key: [(host_word, sense_idx, tier, raw_phrase), ...]}
        - _phrase_by_raw_cache: {raw_phrase_lower: [(host_word, sense_idx, tier, raw_phrase), ...]}
        - _PHRASE_INDEX_CACHE: {phrase_key: best_tier}
        - _PHRASE_SYNONYMS_CACHE: {phrase_key: {synonym_phrase_keys...}}
        """
        if not cls._ensure_ldoce_conn() or cls._ldoce_conn is None:
            return False
        if (
            cls._phrase_by_key_cache is not None
            and cls._phrase_by_raw_cache is not None
            and cls._PHRASE_INDEX_CACHE is not None
            and cls._PHRASE_SYNONYMS_CACHE is not None
        ):
            return True

        by_key: Dict[str, List[Tuple[str, Optional[int], str, str]]] = {}
        by_raw: Dict[str, List[Tuple[str, Optional[int], str, str]]] = {}
        tier_map: Dict[str, str] = {}
        slash_groups: Dict[str, Set[str]] = {}

        try:
            cur = cls._ldoce_conn.cursor()
            cur.execute("SELECT phrase_key, tier, host_word, sense_idx, raw_phrase FROM ldoce_phrase_index")
            for p_key, tier, host_word, sense_idx, raw_phrase in cur.fetchall():
                item = (host_word, sense_idx, tier, raw_phrase)
                if p_key in by_key:
                    by_key[p_key].append(item)
                else:
                    by_key[p_key] = [item]

                if raw_phrase:
                    r_low = raw_phrase.lower()
                    if r_low in by_raw:
                        by_raw[r_low].append(item)
                    else:
                        by_raw[r_low] = [item]
                    if "/" in raw_phrase:
                        if raw_phrase in slash_groups:
                            slash_groups[raw_phrase].add(p_key)
                        else:
                            slash_groups[raw_phrase] = {p_key}

                if p_key not in tier_map or tier == "phrasal_verb":
                    tier_map[p_key] = tier

            def _tier_rank(row: Tuple[str, Optional[int], str, str]) -> int:
                t = row[2]
                return 1 if t == "sense_unit" else (2 if t == "phrasal_verb" else 3)

            for rows in by_key.values():
                if len(rows) > 1:
                    rows.sort(key=_tier_rank)
            for rows in by_raw.values():
                if len(rows) > 1:
                    rows.sort(key=_tier_rank)

            syn_map: Dict[str, Set[str]] = {}
            for keys in slash_groups.values():
                if len(keys) > 1:
                    for k in keys:
                        syn_map.setdefault(k, set()).update(keys)
        except Exception:
            return False

        cls._phrase_by_key_cache = by_key
        cls._phrase_by_raw_cache = by_raw
        cls._PHRASE_INDEX_CACHE = tier_map
        cls._PHRASE_SYNONYMS_CACHE = syn_map
        return True

    @classmethod
    def lookup_ldoce_phrase(
        cls,
        phrase: str,
        headword: Optional[str] = None
    ) -> Optional[Tuple[str, Optional[int], str, str]]:
        """
        Dual-track progressive phrase lookup via in-memory ldoce_phrase_index:
        1. Exact literal match (respecting apostrophes/contractions, e.g. "don't be a stranger")
        2. Canonical slot/token match (via _unit_tokens, e.g. 'hello stranger')
        3. Lemmatized verb fallback
        Returns (host_word, sense_idx, tier, raw_phrase) or None.
        """
        if not phrase or not cls._ensure_in_memory_phrase_index():
            return None

        raw = str(phrase).strip()
        if not raw:
            return None

        by_raw = cls._phrase_by_raw_cache or {}
        by_key = cls._phrase_by_key_cache or {}

        # Track 1: Exact literal match (preserving contraction apostrophes and punctuation stripped)
        exact_clean = raw.lower().replace("\u2019", "'").replace("\u2018", "'")
        exact_clean = re.sub(r"[!?,.]+$", "", exact_clean).strip()
        if exact_clean:
            r1 = by_raw.get(exact_clean)
            r2 = by_raw.get(exact_clean + "!")
            if r1 and r2:
                combined = sorted(
                    r1 + r2,
                    key=lambda r: 1 if r[2] == "sense_unit" else (2 if r[2] == "phrasal_verb" else 3),
                )
                return combined[0]
            if r1:
                return r1[0]
            if r2:
                return r2[0]

        # Track 2: Canonical slot/token match via _unit_tokens
        canon_key = " ".join(cls._unit_tokens(raw))
        if canon_key:
            rows = by_key.get(canon_key)
            if rows:
                return rows[0]

        # Track 3: Lemmatized headword/verb fallback (e.g. "banked up" -> "bank up")
        words = raw.split()
        if words:
            lemmatized_head = cls._surface_lemma(words[0]) or cls.lemmatize_headword(words[0])
            if lemmatized_head and lemmatized_head != words[0]:
                lemmatized_phrase = " ".join([lemmatized_head] + words[1:])
                lemmatized_key = " ".join(cls._unit_tokens(lemmatized_phrase))
                if lemmatized_key and lemmatized_key != canon_key:
                    rows = by_key.get(lemmatized_key)
                    if rows:
                        return rows[0]

        return None

    @classmethod
    def lookup_ldoce_phrase_candidates(
        cls,
        phrase: str,
        headword: Optional[str] = None
    ) -> List[Tuple[str, Optional[int], str, str]]:
        """
        Retrieves all attested candidate matches from in-memory ldoce_phrase_index without LIMIT 1 truncation.
        Returns list of (host_word, sense_idx, tier, raw_phrase) tuples.
        """
        if not phrase or not cls._ensure_in_memory_phrase_index():
            return []

        raw = str(phrase).strip()
        if not raw:
            return []

        by_raw = cls._phrase_by_raw_cache or {}
        by_key = cls._phrase_by_key_cache or {}
        results: List[Tuple[str, Optional[int], str, str]] = []
        seen = set()

        # Track 1: Exact literal match
        exact_clean = raw.lower().replace("\u2019", "'").replace("\u2018", "'")
        exact_clean = re.sub(r"[!?,.]+$", "", exact_clean).strip()
        if exact_clean:
            r1 = by_raw.get(exact_clean, [])
            r2 = by_raw.get(exact_clean + "!", [])
            rows = (
                sorted(
                    r1 + r2,
                    key=lambda r: 1 if r[2] == "sense_unit" else (2 if r[2] == "phrasal_verb" else 3),
                )
                if (r1 and r2)
                else (r1 or r2)
            )
            for r in rows:
                sig = (r[0], r[1], r[2])
                if sig not in seen:
                    seen.add(sig)
                    results.append(r)

        # Track 2: Canonical slot/token match via _unit_tokens
        canon_key = " ".join(cls._unit_tokens(raw))
        if canon_key:
            for r in by_key.get(canon_key, ()):
                sig = (r[0], r[1], r[2])
                if sig not in seen:
                    seen.add(sig)
                    results.append(r)

        # Track 3: Lemmatized headword/verb fallback
        words = raw.split()
        if words:
            lemmatized_head = cls._surface_lemma(words[0]) or cls.lemmatize_headword(words[0])
            if lemmatized_head and lemmatized_head != words[0]:
                lemmatized_phrase = " ".join([lemmatized_head] + words[1:])
                lemmatized_key = " ".join(cls._unit_tokens(lemmatized_phrase))
                if lemmatized_key and lemmatized_key != canon_key:
                    for r in by_key.get(lemmatized_key, ()):
                        sig = (r[0], r[1], r[2])
                        if sig not in seen:
                            seen.add(sig)
                            results.append(r)

        return results

    @classmethod
    def _frame_object_position(cls, frame: str, particle: str) -> str:
        """Where this frame puts its object relative to the particle: 'before', 'after' or
        'unknown'.

        PedLex writes a frame as a token sequence with slots.  'depend on / upon somebody /
        something' puts the object after the particle; 'keep somebody in' and 'work somebody /
        something in' put it before.  '↔' marks the pair as separable, which means PedLex
        allows both orders, so it counts as 'after'.  A slash group is one slot holding several
        alternatives ('on/upon', 'somebody/something'), so it is read as a unit rather than as
        a space.  A frame with no slot at all ('break away') makes no claim either way."""
        if not isinstance(frame, str) or not frame.strip():
            return "unknown"
        chunks = [c for c in re.split(r"\s+", frame.strip().lower()) if c]
        slots = [any(alt in cls._PHRASE_SLOT_WORDS for alt in c.split("/")) for c in chunks]
        for index, chunk in enumerate(chunks):
            if particle not in [a for a in chunk.split("/") if a]:
                continue
            if "\u2194" in frame or "<->" in frame:
                return "after"
            if index + 1 < len(chunks) and slots[index + 1]:
                return "after"
            if index and slots[index - 1]:
                return "before"
        return "unknown"

    @classmethod
    def ldoce_phrase_object_fit(
        cls,
        phrase: str,
        obj: Optional[str],
        headword: Optional[str] = None,
    ) -> bool:
        """Is the object the passage put after this phrase one PedLex's frame allows there?

        'keep in' is a real phrasal verb, but PedLex's frame for it is 'keep somebody in' -
        the object comes before the particle - so a parse that reads 'keep in WeChat Moments'
        as that frame has mistaken a locative 'in ...' for the particle.  'depend on Mary' is
        a fit, because PedLex's frame is 'depend on / upon somebody / something'.

        The veto needs positive evidence: it fires only when some frame puts the object before
        the particle and no frame puts it after.  An empty object is always a fit - nothing has
        been claimed about it - and so is a phrase PedLex gives no frame for ('break away',
        'listen to'), because there is nothing in the entry to contradict the parse."""
        target = (obj or "").strip().lower()
        if not target:
            return True
        tokens = cls._phrase_tokens(phrase)
        if len(tokens) < 2:
            return True
        positions = [cls._frame_object_position(frame, tokens[-1])
                     for block in cls._phrasal_verb_blocks(phrase, headword)
                     for frame in cls._phrasal_block_frames(block)]
        if "after" in positions:
            return True
        return "before" not in positions

    @classmethod
    def _adverb_frame_from_quote(cls, word: str, quote: Optional[str]) -> Optional[str]:
        """
        Syntactically determines whether an adverb modifies an adjective/adverb (modifies_adj)
        or a verb (modifies_verb) from authentic quote evidence using spaCy dependency parse.
        """
        nlp = cls.get_spacy()
        if not nlp or not quote:
            return None
        try:
            doc = nlp(quote)
            w_clean = word.strip().lower()
            for tok in doc:
                if tok.lemma_.lower() != w_clean and tok.text.lower() != w_clean:
                    continue
                head = tok.head
                if head.pos_ in ("ADJ", "ADV"):
                    return "modifies_adj"
                if head.pos_ == "VERB":
                    return "modifies_verb"
        except Exception:
            pass
        return None

    @classmethod
    def search_corpus_examples(
        cls,
        query: str,
        category: Optional[str] = None,
        limit: int = 3
    ) -> List[str]:
        """
        Retrieves authentic PedLex corpus examples directly from entry data_json at 0-token cost.
        Scans own headword entries plus indexed cross-entries (ldoce_phrase_index & ldoce_sense_cue_index).
        """
        if not query or not query.strip():
            return []

        clean_q = query.strip()
        near_match = re.search(r'NEAR\(\s*([a-zA-Z0-9_-]+)\s+([a-zA-Z0-9_-]+)', clean_q, re.IGNORECASE)
        key_tokens = [t.lower() for t in re.findall(r'[a-zA-Z0-9_-]{2,}', clean_q) if t.lower() not in ("near", "and", "or", "category")]
        if not key_tokens:
            return []

        is_multi = len(key_tokens) >= 2 and "-" not in clean_q
        u_key = cls._unit_key(clean_q)

        # 1. Primary anchor words (own headword entries + hyphen sub-parts for derived hyphenated compounds)
        anchor_words: List[str] = []
        sub_parts: List[str] = []
        if "-" in clean_q:
            sub_parts = [p.lower() for p in clean_q.split("-") if len(p) >= 3 and p.lower() not in ("self", "non", "pre", "sub", "anti", "counter")]
        for aw in ([near_match.group(1).lower()] if near_match else [clean_q.lower()] + key_tokens[:3] + sub_parts):
            if aw and aw not in anchor_words:
                anchor_words.append(aw)

        # 2. Cross-entry host lookups via in-memory ldoce_phrase_index and SQLite ldoce_sense_cue_index (<0.5ms indexed lookup)
        if is_multi and u_key and cls._ensure_in_memory_phrase_index() and cls._phrase_by_key_cache:
            added_hw = 0
            for hw, _sidx, _tier, _raw in cls._phrase_by_key_cache.get(u_key, ()):
                if hw and hw not in anchor_words:
                    anchor_words.append(hw)
                    added_hw += 1
                    if added_hw >= 20:
                        break
        if cls._ensure_ldoce_conn() and cls._ldoce_conn is not None:
            try:
                cur = cls._ldoce_conn.cursor()
                for tok in (key_tokens[:2] + sub_parts[:2]):
                    cur.execute(
                        "SELECT DISTINCT host_word FROM ldoce_sense_cue_index WHERE cue_word = ? ORDER BY weight DESC LIMIT 20",
                        (tok,)
                    )
                    for (hw,) in cur.fetchall():
                        if hw and hw not in anchor_words:
                            anchor_words.append(hw)
            except Exception:
                pass

        entries = []
        for aw in anchor_words[:30]:
            ent = cls.get_ldoce_entry(aw)
            if ent and ent not in entries:
                entries.append(ent)
                bw = str(ent.get("base_word") or "").strip().lower()
                if bw and bw != aw:
                    bw_ent = cls.get_ldoce_entry(bw)
                    if bw_ent and bw_ent not in entries:
                        entries.append(bw_ent)
        if not entries:
            return []

        pool: List[str] = []
        for entry in entries:
            for s in entry.get("senses", []):
                pool.extend(s.get("examples", []) or [])
                for pb in (s.get("pattern_bundles") or []):
                    if isinstance(pb, dict) and pb.get("example"):
                        pool.append(pb["example"])
                for u_obj in (s.get("units") or []):
                    if isinstance(u_obj, dict):
                        pool.extend(u_obj.get("examples", []) or [])
            for gb in entry.get("grammar_boxes", []):
                pool.extend(gb.get("examples", []) or [])
            for _cat_k, items in entry.get("collocations", {}).items():
                for it in items:
                    if isinstance(it, dict) and it.get("example"):
                        pool.append(it["example"])
            for cc in entry.get("cross_collocations", []):
                if isinstance(cc, dict) and cc.get("example"):
                    pool.append(cc["example"])
            for th in entry.get("thesaurus", []):
                if isinstance(th, dict):
                    pool.extend(th.get("examples", []) or [])
                    if th.get("example"):
                        pool.append(th["example"])
            for la in entry.get("language_activator", []):
                if isinstance(la, dict):
                    for exp in (la.get("words") or la.get("exponents") or []):
                        if isinstance(exp, dict):
                            pool.extend(exp.get("examples", []) or [])
                            if exp.get("example"):
                                pool.append(exp["example"])

        res: List[str] = []
        for cand in pool:
            if not cand or not isinstance(cand, str):
                continue
            cand_clean = re.sub(r'^(?:(?:Male|Female)\s+speaker|S\b)[\s:]*', '', cand, flags=re.IGNORECASE)
            cand_clean = re.sub(r'\s*\(=[^)]*\)', '', cand_clean)
            cand_clean = re.sub(r'\s+', ' ', cand_clean).strip()
            if not cand_clean or len(cand_clean) <= 8 or cand_clean in res:
                continue
            cand_low = cand_clean.lower()
            if is_multi:
                if cls._unit_example(u_key, [cand_clean], allow_modifier=True) or all(tok in cand_low for tok in key_tokens):
                    res.append(cand_clean)
            else:
                if cls.text_contains_form(cand_clean, clean_q) or all(tok in cand_low for tok in key_tokens):
                    res.append(cand_clean)
            if len(res) >= max(limit * 3, 15):
                break

        def _pedagogical_ex_rank(ex: str) -> tuple:
            ex_l = ex.lower()
            gritty_penalty = 1 if re.search(
                r"\b(?:sleazy|corpse|murder|sex|sexy|bastard|damn|hell|suicide|prostitute|whore|bitch|shit|fuck|piss|rape|cocaine|heroin)\b",
                ex_l,
            ) else 0
            q_penalty = 1 if ex.endswith("?") or not ex.endswith((".", "!", "\"", "'")) else 0
            len_dist = 0 if 40 <= len(ex) <= 135 else abs(len(ex) - 85)
            return (gritty_penalty, q_penalty, len_dist)

        unit_matches = sorted(
            [c for c in res if cls._unit_example(u_key or clean_q, [c], allow_modifier=True)],
            key=_pedagogical_ex_rank,
        )
        other_matches = sorted(
            [c for c in res if c not in unit_matches],
            key=_pedagogical_ex_rank,
        )
        return (unit_matches + other_matches)[:limit]

    @classmethod
    def get_ldoce_grammar_alert(
        cls,
        word: str,
        quote: str = "",
        sense_index: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves authentic grammar error diagnostics (Don't say / ✗) directly from PedLex Grammar Boxes in data_json.
        When `sense_index` is provided, filters out boxes explicitly restricted to other senses (`sense_indices`).
        Returns dict with keys: 'word', 'title', 'content', 'rules', 'comparison_peers', 'syntactic_constraints', etc. or None.
        """
        if not word:
            return None
        w_clean = word.strip().lower()
        entry = cls.get_ldoce_entry(w_clean)
        if not entry:
            return None

        boxes = entry.get("grammar_boxes", [])
        if not boxes:
            return None

        best_box = None
        for b in boxes:
            gb_senses = b.get("sense_indices")
            if sense_index is not None and gb_senses and sense_index not in gb_senses:
                continue
            title = b.get("title", "")
            content = b.get("content", "")
            bad_exs = b.get("bad_examples", [])
            rules = b.get("rules", [])
            has_alert = bool(bad_exs) or bool(rules) or "Don't say" in content or "✗" in content or "grammar" in title.lower()

            if has_alert:
                box_res = {
                    "word": w_clean,
                    "title": title or "GRAMMAR ALERT",
                    "content": content,
                    "bad_examples": bad_exs,
                    "examples": b.get("examples", []),
                    "rules": rules,
                    "comparison_peers": b.get("comparison_peers", []),
                    "peer_details": b.get("peer_details", []),
                    "syntactic_constraints": b.get("syntactic_constraints", {}),
                    "sense_indices": gb_senses or [],
                }
                if quote and any(tok in content.lower() for tok in quote.lower().split() if len(tok) > 4):
                    return box_res
                if not best_box:
                    best_box = box_res
        return best_box

    @classmethod
    def get_ldoce_thesaurus(cls, word: str) -> List[Dict[str, Any]]:

        """
        Retrieves near-synonym clusters with exact distinctions from PedLex Thesaurus.

        The PedLex Language Activator boxes are thesaurus boxes too - PedLex's own
        synonym dictionary - but the builder stores them once under 'language_activator'
        rather than copying every exponent into 'thesaurus' as well ('get': 932 Activator
        exponents against 8 real THESAURUS ones). They are re-joined here, Activator
        records last, so a lookup still sees the whole synonym set and the small
        THESAURUS / WORD SETS boxes stay at the front of the list.
        """
        entry = cls.get_ldoce_entry(word)
        if not entry:
            return []

        out = list(entry.get("thesaurus") or [])
        seen = {(str(t.get("word", "")).strip().lower(),
                 str(t.get("distinction", "")).strip().lower()) for t in out}
        for concept in entry.get("language_activator") or []:
            concept_name = concept.get("concept", "")
            for act in concept.get("words") or []:
                w_txt = str(act.get("word", "")).strip()
                d_txt = str(act.get("definition", "")).strip()
                if not w_txt:
                    continue
                marker = (w_txt.lower(), d_txt.lower())
                if marker in seen:
                    continue
                seen.add(marker)
                out.append({
                    "word": w_txt,
                    "form": act.get("form", "word"),
                    "distinction": d_txt,
                    "example": act.get("example", ""),
                    "grammar": act.get("grammar", ""),
                    "register": act.get("register", ""),
                    "concept": concept_name,
                })
        return out

    @classmethod
    def is_exact_pedlex_headword(cls, word: str) -> bool:
        """
        Returns True if `word` exists in pedlex.db as an exact base headword
        (not merely resolved via suffix-stripping fallback like 'posses' -> 'posse'
        or 'fooling' -> 'fool').
        """
        if not word:
            return False
        w_clean = word.strip().lower()
        if not w_clean:
            return False
        if cls._ldoce_raw_row(w_clean) is not None:
            return True
        if cls._ldoce_raw_row(f"{w_clean}_1") is not None:
            return True
        entry = cls.get_ldoce_entry(w_clean)
        if entry and str(entry.get("word", "")).strip().lower() == w_clean:
            return True
        return False

    @classmethod
    def get_oxford_thesaurus_entries(cls, word: str) -> List[Dict[str, Any]]:
        """
        Retrieves all POS entries for `word` from the `oxford_thesaurus` table in `pedlex.db`.
        Each entry has keys: 'word', 'pos', 'senses' (where each sense has 'sense_num',
        'core_synonyms', 'example', 'synonyms', 'antonyms').
        """
        if not word:
            return []
        w_clean = re.sub(r"\[.*?\]|\(.*?\)", "", str(word)).strip().lower()
        if not w_clean:
            return []
        if w_clean in cls._oxford_thesaurus_cache:
            return cls._oxford_thesaurus_cache[w_clean]

        if cls._ldoce_conn is None:
            cls.get_ldoce_entry("the")
        conn = cls._ldoce_conn
        if not conn:
            return []

        if not cls._ldoce_table_exists("oxford_thesaurus"):
            return []

        out: List[Dict[str, Any]] = []
        try:
            cur = conn.execute(
                "SELECT data_json FROM oxford_thesaurus WHERE word = ?",
                (w_clean,),
            )
            for (data_str,) in cur.fetchall():
                try:
                    data = json.loads(data_str)
                    if isinstance(data, list):
                        by_pos: Dict[str, List[Dict[str, Any]]] = {}
                        for blk in data:
                            if isinstance(blk, dict):
                                p = str(blk.get("pos", "other") or "other").strip().lower()
                                by_pos.setdefault(p, []).append(blk)
                        for p, blks in by_pos.items():
                            out.append({"pos": p, "senses": blks})
                    elif isinstance(data, dict):
                        out.append(data)
                except Exception:
                    pass
        except Exception:
            pass

        cls._oxford_thesaurus_cache[w_clean] = out
        return out

    @classmethod
    def get_pedlex_and_oxford_synonyms_antonyms(
        cls,
        word: str,
        pos: Optional[str] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
    ) -> Tuple[Set[str], List[str]]:
        """
        Unified retrieval of synonyms (for double-key exclusion) and antonyms (for Tier 1
        polar contrast distractors) combining:
          1. PedLex `senses[].synonym`, `senses[].opposite`, `thesaurus`, and `language_activator`
          2. Oxford Thesaurus of English (`oxford_thesaurus` table: `core_synonyms`, `synonyms`, `antonyms`)

        When `definition` or `quote` is provided, sense blocks in `oxford_thesaurus` are scored
        by lexical/synonym overlap with the locked sense so polysemous words prioritize the
        matching sense's antonyms while unioning core synonyms and matched-sense synonyms.
        Returns `(synonyms_set, ordered_antonyms_list)`.
        """
        synonyms: Set[str] = set()
        antonyms: List[str] = []
        seen_ant: Set[str] = set()

        w_clean = re.sub(r"\[.*?\]|\(.*?\)", "", str(word or "")).strip().lower()
        if not w_clean:
            return synonyms, antonyms

        norm_pos = str(pos or "").strip().lower()
        if norm_pos.startswith("v"):
            norm_pos = "verb"
        elif norm_pos.startswith("n"):
            norm_pos = "noun"
        elif norm_pos.startswith("adj") or norm_pos == "a":
            norm_pos = "adjective"
        elif norm_pos.startswith("adv") or norm_pos == "r":
            norm_pos = "adverb"

        def _clean_term(raw_t: str) -> str:
            t = re.sub(r"\[.*?\]|\(.*?\)", "", str(raw_t or "")).strip().lower()
            t = re.sub(r"^to\s+", "", t).strip()
            return t

        def _add_syn(raw_t: str) -> None:
            for part in re.split(r"[,;/]", str(raw_t or "")):
                t = _clean_term(part)
                if t and t != w_clean:
                    synonyms.add(t)

        def _add_ant(raw_t: str) -> None:
            for part in re.split(r"[,;/]", str(raw_t or "")):
                t = _clean_term(part)
                if t and t != w_clean and t not in seen_ant:
                    seen_ant.add(t)
                    antonyms.append(t)

        # 1. PedLex native entry (senses, thesaurus, language_activator)
        w_lemma = cls.lemma_of(w_clean) if " " not in w_clean else w_clean
        entry = cls.get_ldoce_entry(w_clean) or (cls.get_ldoce_entry(w_lemma) if w_lemma != w_clean else None)
        def_low = re.sub(r"^\[.*?\]\s*", "", str(definition or "").strip().lower())
        def_tokens = {
            tok for tok in re.findall(r"[a-z]{3,}", def_low)
            if tok not in {"something", "somebody", "someone", "person", "people", "thing", "things", "make", "have", "give", "used", "very", "that", "with", "from", "into", "the", "and", "for", "you", "your", "want", "not", "act"}
        }
        def_lemmas = set(def_tokens) | {cls.lemma_of(tok) for tok in def_tokens}
        quote_tokens = {
            tok for tok in re.findall(r"[a-z]{4,}", str(quote or "").lower())
            if tok not in (w_clean, w_lemma)
        }
        matched_sense_syns: Set[str] = set()

        if entry:
            # Bare gloss extraction (e.g. "[formal] lunch" -> "lunch" is a direct synonym)
            if def_low and len(def_low.split()) <= 2 and re.match(r"^[a-z-]+$", def_low):
                _add_syn(def_low)
                matched_sense_syns.add(def_low)

            any_sense_matched_def = False
            for s in entry.get("senses", []):
                s_def = re.sub(r"^\[.*?\]\s*", "", str(s.get("definition", "")).strip().lower())
                is_matched_sense = bool(def_low and (s_def == def_low or def_low in s_def or s_def in def_low))
                if is_matched_sense:
                    any_sense_matched_def = True
                if s_def and len(s_def.split()) <= 2 and re.match(r"^[a-z-]+$", s_def):
                    _add_syn(s_def)
                    if is_matched_sense or not def_low:
                        matched_sense_syns.add(s_def)
                raw_syns = s.get("synonym") or []
                for syn in ([raw_syns] if isinstance(raw_syns, str) else raw_syns):
                    _add_syn(syn)
                    if is_matched_sense or not def_low:
                        ct = _clean_term(syn)
                        if ct:
                            matched_sense_syns.add(ct)
                raw_opps = s.get("opposite") or []
                for opp in ([raw_opps] if isinstance(raw_opps, str) else raw_opps):
                    if is_matched_sense or not def_low:
                        _add_ant(opp)
            # Second pass for any remaining opposites on the entry ONLY when no specific sense matched def_low
            if not any_sense_matched_def:
                for s in entry.get("senses", []):
                    raw_opps = s.get("opposite") or []
                    for opp in ([raw_opps] if isinstance(raw_opps, str) else raw_opps):
                        _add_ant(opp)

            for th in entry.get("thesaurus", []):
                if isinstance(th, dict) and th.get("word"):
                    _add_syn(th["word"])

            for la in entry.get("language_activator", []):
                if isinstance(la, dict):
                    la_words = [
                        _clean_term(exp.get("word", ""))
                        for exp in (la.get("words") or la.get("exponents") or [])
                        if isinstance(exp, dict) and exp.get("word")
                    ]
                    if len(la_words) <= 20:
                        for lw in la_words:
                            _add_syn(lw)

        # 2. Oxford Thesaurus of English
        ox_entries = list(cls.get_oxford_thesaurus_entries(w_clean))
        if w_lemma != w_clean and not ox_entries:
            ox_entries = list(cls.get_oxford_thesaurus_entries(w_lemma))
        for ox in ox_entries:
            ox_pos = str(ox.get("pos", "")).strip().lower()
            if norm_pos and ox_pos and norm_pos in {"verb", "noun", "adjective", "adverb"}:
                if ox_pos != norm_pos:
                    continue

            senses = ox.get("senses") or []
            scored_senses = []
            for idx, s_blk in enumerate(senses):
                core_syns = [_clean_term(x) for x in (s_blk.get("core_synonyms") or []) if _clean_term(x)]
                all_syns = [_clean_term(x) for x in (s_blk.get("synonyms") or []) if _clean_term(x)]
                ex_str = str(s_blk.get("example", "")).lower()
                score = 0
                if len(senses) == 1:
                    score += 10
                else:
                    syn_set = set(all_syns) | set(core_syns)
                    overlap_def = len(def_lemmas & syn_set)
                    overlap_existing = len(matched_sense_syns & syn_set)
                    ex_tokens = set(re.findall(r"[a-z]{3,}", ex_str))
                    overlap_ex = len((def_lemmas | quote_tokens) & ex_tokens)
                    score += overlap_def * 4 + overlap_existing * 3 + overlap_ex * 2
                    if idx == 0:
                        score += 1
                scored_senses.append((score, idx, s_blk, all_syns, core_syns))

            for _sc, _idx, _s_blk, _all_syns, core_syns in scored_senses:
                # Always include core synonyms across all valid POS senses to block obvious double-keys
                for cs in core_syns:
                    _add_syn(cs)

            scored_senses.sort(key=lambda item: (-item[0], item[1]))
            has_context_signal = bool(def_lemmas or quote_tokens or matched_sense_syns) and scored_senses and scored_senses[0][0] > 1
            for rank, (sc, _idx, s_blk, all_syns, _core_syns) in enumerate(scored_senses):
                # Include full synonym/antonym list from the best-matching sense(s); only fall back to <= 2 senses when no context signal disambiguates them
                if rank == 0 or sc >= 4 or (not has_context_signal and len(scored_senses) <= 2):
                    for syn_w in all_syns:
                        _add_syn(syn_w)
                    for ant_w in (s_blk.get("antonyms") or []):
                        _add_ant(ant_w)

            # Append antonyms from remaining senses only when no context signal disambiguated a primary sense
            if not has_context_signal:
                for _sc, _idx, s_blk, _all_syns, _core_syns in scored_senses:
                    for ant_w in (s_blk.get("antonyms") or []):
                        _add_ant(ant_w)

            # 1-hop Core Synonym Antonym Resolution:
            # In Oxford Thesaurus, specialized headwords (e.g. 'autonomy', 'precision', 'polar', 'compulsion')
            # often point to a core/immediate synonym ('independence', 'exactness', 'opposite', 'coercion') where the
            # reciprocal antonym block ('dependence', 'inaccuracy', 'identical', 'persuasion') is stored.
            if not antonyms and scored_senses:
                top_s_blk = scored_senses[0][2]
                top_syns = {_clean_term(x) for x in (top_s_blk.get("synonyms") or []) if _clean_term(x)}
                hop_candidates: List[str] = []
                if entry:
                    for s in entry.get("senses", []):
                        s_def = re.sub(r"^\[.*?\]\s*", "", str(s.get("definition", "")).strip().lower())
                        if def_low and (s_def == def_low or def_low in s_def or s_def in def_low):
                            raw_syns = s.get("synonym") or []
                            for syn in ([raw_syns] if isinstance(raw_syns, str) else raw_syns):
                                ct = _clean_term(syn)
                                if ct and " " not in ct and ct not in hop_candidates:
                                    hop_candidates.append(ct)
                for x in (top_s_blk.get("core_synonyms") or []):
                    ct = _clean_term(x)
                    if ct and " " not in ct and ct not in hop_candidates:
                        hop_candidates.append(ct)
                for x in (top_s_blk.get("synonyms") or [])[:6]:
                    ct = _clean_term(x)
                    if ct and " " not in ct and ct not in hop_candidates:
                        hop_candidates.append(ct)
                for cs in hop_candidates[:6]:
                    for cs_ox in cls.get_oxford_thesaurus_entries(cs):
                        cs_pos = str(cs_ox.get("pos", "")).strip().lower()
                        if norm_pos and cs_pos and norm_pos in {"verb", "noun", "adjective", "adverb"}:
                            if cs_pos != norm_pos:
                                continue
                        for cs_blk in (cs_ox.get("senses") or []):
                            cs_syns = {_clean_term(x) for x in (cs_blk.get("synonyms") or []) if _clean_term(x)}
                            cs_core = {_clean_term(x) for x in (cs_blk.get("core_synonyms") or []) if _clean_term(x)}
                            # Verify reciprocal membership with the matched sense's own synonyms (not cross-sense union)
                            if w_clean in cs_syns or w_lemma in cs_syns or w_clean in cs_core or len(cs_syns & top_syns) >= 2:
                                for ant_w in (cs_blk.get("antonyms") or []):
                                    _add_ant(ant_w)

        return synonyms, antonyms

