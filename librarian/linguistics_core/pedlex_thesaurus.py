"""
PedLex authentic corpus example search, grammar alerts, and LDOCE/Oxford thesaurus integration.
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


class PedlexThesaurusMixin:
    _exact_headword_set: Optional[FrozenSet[str]] = None
    _exact_headword_conn_id: Optional[int] = None

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
                if is_multi and (clean_q.lower() != u_key or any(t.endswith("ed") for t in key_tokens)):
                    cur.execute(
                        "SELECT word FROM ldoce WHERE data_json LIKE ? LIMIT 15",
                        (f"%{clean_q.lower()}%",),
                    )
                    for (hw,) in cur.fetchall():
                        if hw and hw not in anchor_words:
                            anchor_words.insert(0, hw)
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
            for pv in entry.get("phrasal_verbs", []):
                if isinstance(pv, dict):
                    for pvs in (pv.get("senses") or []):
                        if isinstance(pvs, dict):
                            pool.extend(pvs.get("examples", []) or [])
                            for pb in (pvs.get("pattern_bundles") or []):
                                if isinstance(pb, dict) and pb.get("example"):
                                    pool.append(pb["example"])
            for ph in entry.get("phrases", []):
                if isinstance(ph, dict):
                    for phs in (ph.get("senses") or []):
                        if isinstance(phs, dict):
                            pool.extend(phs.get("examples", []) or [])
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
            exact_surf_miss = 0 if (is_multi and clean_q.lower() in ex_l) else 1
            gritty_penalty = 1 if re.search(
                r"\b(?:sleazy|corpse|murder|sex|sexy|bastard|damn|hell|suicide|prostitute|whore|bitch|shit|fuck|piss|rape|cocaine|heroin)\b",
                ex_l,
            ) else 0
            q_penalty = 1 if ex.endswith("?") or not ex.endswith((".", "!", "\"", "'")) else 0
            len_dist = 0 if 40 <= len(ex) <= 135 else abs(len(ex) - 85)
            return (exact_surf_miss, gritty_penalty, q_penalty, len_dist)

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
        sense_index: Optional[int] = None,
        pos: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves authentic grammar error diagnostics (Don't say / ✗) directly from PedLex Grammar Boxes in data_json.
        When `sense_index` or `pos` is provided, filters out boxes explicitly restricted to other senses or POS categories.
        Returns dict with keys: 'word', 'title', 'content', 'rules', 'comparison_peers', 'syntactic_constraints', etc. or None.
        """
        if not word:
            return None
        w_clean = word.strip().lower()
        entry = cls.get_ldoce_entry(w_clean)
        if not entry and " " in w_clean:
            entry = cls.get_ldoce_entry(w_clean.split()[0])
        if not entry:
            return None

        boxes = entry.get("grammar_boxes", [])
        if not boxes:
            return None

        p_norm = (pos or "").strip().lower()
        if "phrasal" in p_norm:
            p_norm = "verb"
        elif p_norm.startswith("n"):
            p_norm = "noun"
        elif p_norm.startswith("v"):
            p_norm = "verb"
        elif p_norm.startswith("adj") or p_norm == "a":
            p_norm = "adjective"
        elif p_norm.startswith("adv") or p_norm == "r":
            p_norm = "adverb"

        all_senses = entry.get("senses") or []
        best_box = None
        for b in boxes:
            b_pv = (b.get("phrasal_verb") or "").strip().lower()
            if b_pv:
                if " " not in w_clean and b_pv != w_clean:
                    continue
                if " " in w_clean and b_pv != w_clean and not w_clean.startswith(b_pv):
                    continue
            elif " " in w_clean:
                continue
            gb_senses = b.get("sense_indices")
            if sense_index is not None and gb_senses and sense_index not in gb_senses:
                continue
            if sense_index is not None and not gb_senses and 0 <= sense_index < len(all_senses):
                s_hom = all_senses[sense_index].get("homograph_num")
                b_hom = b.get("homograph_num")
                if s_hom and b_hom and s_hom != b_hom:
                    continue
            title = b.get("title", "")
            content = b.get("content", "")
            bad_exs = b.get("bad_examples", [])
            rules = b.get("rules", [])
            has_alert = bool(bad_exs) or bool(rules) or "Don't say" in content or "✗" in content or "grammar" in title.lower()
            if not has_alert:
                continue

            if p_norm:
                b_pos = (b.get("pos") or "").strip().lower()
                if b_pos and b_pos != "other":
                    b_pos_norm = (
                        "noun" if b_pos.startswith("n")
                        else "verb" if b_pos.startswith("v")
                        else "adjective" if (b_pos.startswith("adj") or b_pos == "a")
                        else "adverb" if (b_pos.startswith("adv") or b_pos == "r")
                        else b_pos
                    )
                    if b_pos_norm != p_norm:
                        # Also verify whether any linked sense matches p_norm
                        linked_match = False
                        for s_i in (gb_senses or []):
                            if 0 <= s_i < len(all_senses):
                                sp = (all_senses[s_i].get("pos") or "").lower()
                                if sp.startswith(p_norm[:3]):
                                    linked_match = True
                                    break
                        if not linked_match:
                            continue

            if p_norm in ("noun", "verb"):
                c_low = content.lower()
                forb = (b.get("syntactic_constraints") or {}).get("forbidden_frames") or []
                is_noun_box = bool(
                    "countable_use" in forb
                    or re.search(rf"\b{re.escape(w_clean)}\s+is\s+(?:always\s+)?an?\s+(?:uncountable|countable|singular|plural)\s+noun\b", c_low)
                )
                is_verb_box = bool(
                    not is_noun_box
                    and re.search(rf"\b(?:you|he|she|we|they)\s+{re.escape(w_clean)}(?:s|es|ed)?\s+(?:in|at|for|on|as|with|to)\b", c_low)
                )
                if p_norm == "noun" and is_verb_box:
                    continue
                if p_norm == "verb" and is_noun_box:
                    continue

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

        if cls._ldoce_conn is None:
            cls.get_ldoce_entry("the")
        conn = cls._ldoce_conn
        if conn is not None:
            conn_id = id(conn)
            if cls._exact_headword_set is None or cls._exact_headword_conn_id != conn_id:
                try:
                    rows = conn.execute("SELECT word FROM ldoce").fetchall()
                    hw_set: Set[str] = set()
                    for (raw_w,) in rows:
                        if raw_w:
                            wl = str(raw_w).strip().lower()
                            if wl:
                                hw_set.add(wl)
                                hw_set.add(re.sub(r"_\d+$", "", wl))
                    cls._exact_headword_set = frozenset(hw_set)
                    cls._exact_headword_conn_id = conn_id
                except Exception:
                    cls._exact_headword_set = None
            if cls._exact_headword_set is not None:
                return w_clean in cls._exact_headword_set

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
        by_pos_merged: Dict[str, List[Dict[str, Any]]] = {}
        seen_sense_sigs: Set[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = set()
        try:
            for q_var in sorted(set(cls.orthographic_variants(w_clean))):
                cur = conn.execute(
                    "SELECT data_json FROM oxford_thesaurus WHERE word = ?",
                    (q_var,),
                )
                rows = cur.fetchall()
                if not rows:
                    continue
                for (data_str,) in rows:
                    try:
                        data = json.loads(data_str)
                        blks_to_add: List[Dict[str, Any]] = []
                        if isinstance(data, list):
                            blks_to_add = [b for b in data if isinstance(b, dict)]
                        elif isinstance(data, dict):
                            p_top = str(data.get("pos", "other") or "other").strip().lower()
                            for b in (data.get("senses") or []):
                                if isinstance(b, dict):
                                    b_copy = dict(b)
                                    b_copy.setdefault("pos", p_top)
                                    blks_to_add.append(b_copy)
                        for blk in blks_to_add:
                            p = str(blk.get("pos", "other") or "other").strip().lower()
                            core_sig = tuple(str(x).lower() for x in (blk.get("core_synonyms") or []))
                            syn_sig = tuple(str(x).lower() for x in (blk.get("synonyms") or [])[:5])
                            sig = (p, core_sig, syn_sig)
                            if sig not in seen_sense_sigs:
                                seen_sense_sigs.add(sig)
                                by_pos_merged.setdefault(p, []).append(blk)
                    except Exception:
                        pass
            for p, blks in by_pos_merged.items():
                out.append({"pos": p, "senses": blks})
        except Exception:
            pass

        cls._oxford_thesaurus_cache[w_clean] = out
        return out

    _pedlex_ox_syn_ant_cache: Dict[Tuple[str, str, str, str], Tuple[FrozenSet[str], Tuple[str, ...]]] = {}

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

        def_key = re.sub(r"^\[.*?\]\s*", "", str(definition or "").strip().lower())
        quote_key = str(quote or "").strip().lower()[:120]
        cache_key = (w_clean, norm_pos, def_key, quote_key)
        cached_sa = cls._pedlex_ox_syn_ant_cache.get(cache_key)
        if cached_sa is not None:
            return set(cached_sa[0]), list(cached_sa[1])

        w_lemma = cls.lemma_of(w_clean) if " " not in w_clean else w_clean
        w_ortho_set: Set[str] = set(cls.orthographic_variants(w_clean)) | set(cls.orthographic_variants(w_lemma))

        def _clean_term(raw_t: str) -> str:
            t = re.sub(r"\[.*?\]|\(.*?\)", "", str(raw_t or "")).strip().lower()
            t = re.sub(r"^to\s+", "", t).strip()
            return t

        def _add_syn(raw_t: str) -> None:
            raw_s = str(raw_t or "")
            if "/" in raw_s and any(" " in seg.strip() for seg in raw_s.split("/")):
                t = _clean_term(raw_s)
                if t and t not in w_ortho_set and "/" not in t:
                    synonyms.add(t)
                return
            for part in re.split(r"[,;/]", raw_s):
                t = _clean_term(part)
                if t and t not in w_ortho_set:
                    synonyms.add(t)

        def _add_ant(raw_t: str) -> None:
            raw_s = str(raw_t or "")
            if "/" in raw_s and any(" " in seg.strip() for seg in raw_s.split("/")):
                t = _clean_term(raw_s)
                if t and t not in w_ortho_set and "/" not in t and t not in seen_ant:
                    seen_ant.add(t)
                    antonyms.append(t)
                return
            for part in re.split(r"[,;/]", raw_s):
                t = _clean_term(part)
                if t and t not in w_ortho_set and t not in seen_ant:
                    seen_ant.add(t)
                    antonyms.append(t)

        # 1. PedLex native entry (senses, thesaurus, language_activator)
        entry = cls.get_ldoce_entry(w_clean) or (cls.get_ldoce_entry(w_lemma) if w_lemma != w_clean else None)
        def_low = re.sub(r"^\[.*?\]\s*", "", str(definition or "").strip().lower())
        def_tokens = {
            tok for tok in re.findall(r"[a-z]{3,}", def_low)
            if tok not in {"something", "somebody", "someone", "person", "people", "thing", "things", "make", "have", "give", "used", "very", "that", "with", "from", "into", "the", "and", "for", "you", "your", "want", "not", "act"}
        }
        def_lemmas = set(def_tokens) | {cls.lemma_of(tok) for tok in def_tokens}
        quote_tokens = {
            tok for tok in re.findall(r"[a-z]{4,}", str(quote or "").lower())
            if tok not in w_ortho_set
        }
        matched_sense_syns: Set[str] = set()

        if entry:
            # Bare gloss extraction (e.g. "[formal] lunch" -> "lunch" is a direct synonym)
            if def_low and len(def_low.split()) <= 2 and re.match(r"^[a-z-]+$", def_low):
                _add_syn(def_low)
                matched_sense_syns.add(def_low)

            ped_senses = entry.get("senses", [])
            any_sense_matched_def = any(
                bool(
                    def_low
                    and (
                        re.sub(r"^\[.*?\]\s*", "", str(s.get("definition", "")).strip().lower()) == def_low
                        or def_low in re.sub(r"^\[.*?\]\s*", "", str(s.get("definition", "")).strip().lower())
                        or (
                            len(re.sub(r"^\[.*?\]\s*", "", str(s.get("definition", "")).strip().lower())) >= 10
                            and re.sub(r"^\[.*?\]\s*", "", str(s.get("definition", "")).strip().lower()) in def_low
                        )
                    )
                )
                for s in ped_senses
            )
            unmatched_opps: Set[str] = set()
            for s in ped_senses:
                s_def = re.sub(r"^\[.*?\]\s*", "", str(s.get("definition", "")).strip().lower())
                is_matched_sense = bool(
                    def_low
                    and (
                        s_def == def_low
                        or def_low in s_def
                        or (len(s_def) >= 10 and s_def in def_low)
                    )
                )
                s_def_toks = set(re.findall(r"[a-z]{3,}", s_def))
                allow_sense_syn = (
                    not any_sense_matched_def
                    or is_matched_sense
                    or len(s_def_toks & def_tokens) >= 2
                )
                if s_def and len(s_def.split()) <= 2 and re.match(r"^[a-z-]+$", s_def) and allow_sense_syn:
                    _add_syn(s_def)
                    if is_matched_sense or not def_low:
                        matched_sense_syns.add(s_def)
                raw_syns = s.get("synonym") or []
                for syn in ([raw_syns] if isinstance(raw_syns, str) else raw_syns):
                    if allow_sense_syn:
                        _add_syn(syn)
                    if is_matched_sense or not def_low:
                        ct = _clean_term(syn)
                        if ct:
                            matched_sense_syns.add(ct)
                raw_opps = s.get("opposite") or []
                for opp in ([raw_opps] if isinstance(raw_opps, str) else raw_opps):
                    if is_matched_sense or not def_low:
                        _add_ant(opp)
                    else:
                        ct_opp = _clean_term(opp)
                        if ct_opp:
                            unmatched_opps.add(ct_opp)
            # Second pass for any remaining opposites on the entry ONLY when no specific sense matched def_low
            if not any_sense_matched_def:
                for s in ped_senses:
                    raw_opps = s.get("opposite") or []
                    for opp in ([raw_opps] if isinstance(raw_opps, str) else raw_opps):
                        _add_ant(opp)

            thes_sections: Dict[Any, List[Dict[str, Any]]] = {}
            for th in entry.get("thesaurus", []):
                if isinstance(th, dict) and th.get("word"):
                    _add_syn(th["word"])
                    sec_k = th.get("section_index", 0)
                    thes_sections.setdefault(sec_k, []).append(th)
            if def_tokens and thes_sections:
                for _sec_k, sec_items in thes_sections.items():
                    self_dists = [
                        str(it.get("distinction", "")).lower()
                        for it in sec_items
                        if _clean_term(it.get("word", "")) in w_ortho_set
                    ]
                    if len(thes_sections) == 1 or any(
                        set(re.findall(r"[a-z]{3,}", sd)) & def_tokens for sd in self_dists
                    ):
                        for it in sec_items:
                            ct_th = _clean_term(it.get("word", ""))
                            if ct_th and ct_th not in w_ortho_set:
                                matched_sense_syns.add(ct_th)

            for la in entry.get("language_activator", []):
                if isinstance(la, dict):
                    la_concept = str(la.get("concept") or "").strip().lower()
                    # Skip negated/opposite activator concepts (e.g. 'not intelligent' attached to 'smart')
                    if la_concept.startswith(("not ", "no ", "without ")) and not w_clean.startswith(("un", "in", "im", "ir", "il", "non", "dis")):
                        continue
                    la_words = [
                        _clean_term(exp.get("word", ""))
                        for exp in (la.get("words") or la.get("exponents") or [])
                        if isinstance(exp, dict) and exp.get("word")
                    ]
                    # Only include cluster if target headword/lemma (or its orthographic variant) is an actual standalone exponent in the cluster
                    _max_la_cluster = 45 if norm_pos == "noun" else 20
                    if bool(w_ortho_set & set(la_words)) and len(la_words) <= _max_la_cluster:
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
                reg_syns_map = s_blk.get("register_synonyms") or {}
                reg_syns_flat = [
                    _clean_term(x)
                    for r_list in (reg_syns_map.values() if isinstance(reg_syns_map, dict) else [])
                    for x in (r_list if isinstance(r_list, list) else [])
                    if _clean_term(x)
                ]
                all_syns = [_clean_term(x) for x in (s_blk.get("synonyms") or []) if _clean_term(x)] + reg_syns_flat
                s_ants = {_clean_term(x) for x in (s_blk.get("antonyms") or []) if _clean_term(x)}
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
                    if any_sense_matched_def and unmatched_opps and (s_ants & unmatched_opps):
                        score -= 5
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
            # (or when the primary sense had no antonyms at all, fall back to other senses of the same headword)
            if not has_context_signal or not antonyms:
                for _sc, _idx, s_blk, _all_syns, _core_syns in scored_senses:
                    for ant_w in (s_blk.get("antonyms") or []):
                        _add_ant(ant_w)

            # 1-hop Core Synonym Antonym Resolution (disabled for verbs because action/injury verbs
            # like 'scar' -> 'blemish' hop into appearance-verb antonyms like 'beautify'/'embellish'):
            # In Oxford Thesaurus, specialized headwords (e.g. 'autonomy', 'precision', 'polar', 'compulsion', 'threat')
            # often point to a core/immediate synonym ('independence', 'exactness', 'opposite', 'coercion', 'danger') where the
            # reciprocal antonym block ('dependence', 'inaccuracy', 'identical', 'persuasion', 'safety') is stored.
            if not antonyms and scored_senses and norm_pos != "verb":
                viable_s_blks = (
                    [
                        item[2]
                        for rank, item in enumerate(scored_senses)
                        if rank == 0
                        or (item[0] >= 2 if has_context_signal else item[0] >= 1)
                        or (not has_context_signal and len(scored_senses) <= 3)
                    ]
                    if norm_pos == "noun"
                    else [scored_senses[0][2]]
                )
                top_syns = {
                    _clean_term(x)
                    for sb in viable_s_blks
                    for x in ((sb.get("synonyms") or []) + (sb.get("core_synonyms") or []))
                    if _clean_term(x)
                }
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
                for sb in viable_s_blks:
                    for x in (sb.get("core_synonyms") or []):
                        ct = _clean_term(x)
                        if ct and " " not in ct and ct not in hop_candidates:
                            hop_candidates.append(ct)
                for sb in viable_s_blks:
                    for x in (sb.get("synonyms") or [])[:6]:
                        ct = _clean_term(x)
                        if ct and " " not in ct and ct not in hop_candidates:
                            hop_candidates.append(ct)
                max_hops = 10 if norm_pos == "noun" else 6
                # Pass 1: Strict reciprocal membership (target appears in hop candidate's synonym/core_synonym list)
                for cs in hop_candidates[:max_hops]:
                    for cs_ox in cls.get_oxford_thesaurus_entries(cs):
                        cs_pos = str(cs_ox.get("pos", "")).strip().lower()
                        if norm_pos and cs_pos and norm_pos in {"verb", "noun", "adjective", "adverb"}:
                            if cs_pos != norm_pos:
                                continue
                        for cs_blk in (cs_ox.get("senses") or []):
                            cs_syns = {_clean_term(x) for x in (cs_blk.get("synonyms") or []) if _clean_term(x)}
                            cs_core = {_clean_term(x) for x in (cs_blk.get("core_synonyms") or []) if _clean_term(x)}
                            if bool(w_ortho_set & cs_syns) or bool(w_ortho_set & cs_core):
                                for ant_w in (cs_blk.get("antonyms") or []):
                                    _add_ant(ant_w)
                # Pass 2: Fallback shared-synonym overlap only if strict reciprocal hop found no antonyms
                if not antonyms:
                    for cs in hop_candidates[:max_hops]:
                        for cs_ox in cls.get_oxford_thesaurus_entries(cs):
                            cs_pos = str(cs_ox.get("pos", "")).strip().lower()
                            if norm_pos and cs_pos and norm_pos in {"verb", "noun", "adjective", "adverb"}:
                                if cs_pos != norm_pos:
                                    continue
                            for cs_blk in (cs_ox.get("senses") or []):
                                cs_syns = {_clean_term(x) for x in (cs_blk.get("synonyms") or []) if _clean_term(x)}
                                if len(cs_syns & top_syns) >= 2:
                                    for ant_w in (cs_blk.get("antonyms") or []):
                                        _add_ant(ant_w)

            # 1-hop Reciprocal Antonym Cluster Expansion for Nouns:
            # In Oxford Thesaurus, nouns often list only 1 headword antonym (e.g., consensus -> disagreement,
            # shortage -> abundance, threat -> safety, recession -> boom). Expand verified direct noun antonyms
            # into the reciprocal opposite sense's core_synonyms so multiple distinct opposite-polarity nouns
            # are available when slots or cross-item deduplication require them.
            if norm_pos == "noun" and antonyms:
                reciprocal_targets = set(w_ortho_set) | matched_sense_syns
                if scored_senses:
                    for _sc, _idx, _sb, _asyns, _csyns in scored_senses[:2]:
                        reciprocal_targets.update(_csyns)
                seed_ants = list(antonyms)[:2]
                for ant_seed in seed_ants:
                    if " " in ant_seed:
                        continue
                    for a_ox in cls.get_oxford_thesaurus_entries(ant_seed):
                        if str(a_ox.get("pos", "")).strip().lower() != "noun":
                            continue
                        a_senses = a_ox.get("senses") or []
                        for a_blk in a_senses:
                            a_ants = {_clean_term(x) for x in (a_blk.get("antonyms") or []) if _clean_term(x)}
                            if (a_ants & reciprocal_targets) or (len(a_senses) == 1 and not a_ants):
                                for ac in (a_blk.get("core_synonyms") or []):
                                    ct_ac = _clean_term(ac)
                                    if ct_ac and " " not in ct_ac and ct_ac not in synonyms:
                                        _add_ant(ct_ac)
                                for asyn in (a_blk.get("synonyms") or [])[:4]:
                                    ct_as = _clean_term(asyn)
                                    if ct_as and " " not in ct_as and ct_as not in synonyms:
                                        _add_ant(ct_as)

        # Antonyms take precedence: never classify an attested antonym as a synonym
        synonyms -= seen_ant
        cls._pedlex_ox_syn_ant_cache[cache_key] = (frozenset(synonyms), tuple(antonyms))
        return synonyms, antonyms

    @classmethod
    def get_strict_pedlex_and_oxford_synonyms(
        cls,
        word: str,
        pos: Optional[str] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
    ) -> Set[str]:
        """Return strict lexical synonyms from PedLex (`senses[].synonym` + bare-gloss definitions)
        and `oxford_thesaurus` (`core_synonyms` + sense-matched `synonyms`), excluding broad contrastive
        `ldoce` `thesaurus` / `language_activator` clusters."""
        w_clean = (word or "").strip().lower()
        if not w_clean:
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
        def_low = re.sub(r"^\[.*?\]\s*", "", (definition or "").strip().lower())
        def_tokens = set(re.findall(r"[a-z]{3,}", def_low)) - {
            "the", "and", "for", "that", "with", "from", "into", "someone", "something",
            "somebody", "person", "people", "thing", "things", "way", "time", "part", "place",
            "used", "make", "have", "give", "take", "very", "more", "most", "much", "many",
        }
        w_lemma = cls.lemma_of(w_clean) if hasattr(cls, "lemma_of") and " " not in w_clean else w_clean
        w_ortho_set: Set[str] = set(cls.orthographic_variants(w_clean)) | set(cls.orthographic_variants(w_lemma))
        strict_syns: Set[str] = set()

        def _clean_t(t: str) -> str:
            t = re.sub(r"\(.*?\)|\[.*?\]", "", str(t or "")).strip().lower()
            t = re.sub(r"\b(?:something|somebody|someone|sth|sb)\b", "", t).strip()
            return re.sub(r"\s+", " ", t).strip()

        def_lemmas = set(def_tokens) | (
            {cls.lemma_of(tok) for tok in def_tokens} if hasattr(cls, "lemma_of") else set()
        )
        quote_tokens = {
            tok for tok in re.findall(r"[a-z]{4,}", str(quote or "").lower())
            if tok not in w_ortho_set
        }
        entry = cls.get_ldoce_entry(w_clean) or (cls.get_ldoce_entry(w_lemma) if w_lemma != w_clean else None)
        if entry:
            ped_senses = entry.get("senses", [])
            any_strict_sense_matched = any(
                bool(
                    def_low
                    and (
                        re.sub(r"^\[.*?\]\s*", "", str(s.get("definition", "")).strip().lower()) == def_low
                        or def_low in re.sub(r"^\[.*?\]\s*", "", str(s.get("definition", "")).strip().lower())
                        or (
                            len(re.sub(r"^\[.*?\]\s*", "", str(s.get("definition", "")).strip().lower())) >= 10
                            and re.sub(r"^\[.*?\]\s*", "", str(s.get("definition", "")).strip().lower()) in def_low
                        )
                    )
                )
                for s in ped_senses
            )
            for s in ped_senses:
                s_pos = str(s.get("pos", "")).strip().lower()
                if s_pos.startswith("v"):
                    s_pos = "verb"
                elif s_pos.startswith("n"):
                    s_pos = "noun"
                elif s_pos.startswith("adj") or s_pos == "a":
                    s_pos = "adjective"
                elif s_pos.startswith("adv") or s_pos == "r":
                    s_pos = "adverb"
                if norm_pos and s_pos and norm_pos in {"verb", "noun", "adjective", "adverb"} and s_pos != norm_pos:
                    continue
                s_def = re.sub(r"^\[.*?\]\s*", "", str(s.get("definition", "")).strip().lower())
                is_matched_sense = bool(
                    def_low
                    and (
                        s_def == def_low
                        or def_low in s_def
                        or (len(s_def) >= 10 and s_def in def_low)
                    )
                )
                s_def_toks = set(re.findall(r"[a-z]{3,}", s_def))
                if any_strict_sense_matched and not is_matched_sense and len(s_def_toks & def_tokens) < 2:
                    continue
                if s_def and len(s_def.split()) <= 2 and re.match(r"^[a-z-]+$", s_def) and s_def not in w_ortho_set:
                    strict_syns.add(s_def)
                for syn in ([s.get("synonym")] if isinstance(s.get("synonym"), str) else (s.get("synonym") or [])):
                    ct = _clean_t(syn)
                    if ct and ct not in w_ortho_set:
                        strict_syns.add(ct)

        ox_entries = list(cls.get_oxford_thesaurus_entries(w_clean))
        if w_lemma != w_clean and not ox_entries:
            ox_entries = list(cls.get_oxford_thesaurus_entries(w_lemma))
        for ox in ox_entries:
            ox_pos = str(ox.get("pos", "")).strip().lower()
            if norm_pos and ox_pos and norm_pos in {"verb", "noun", "adjective", "adverb"} and ox_pos != norm_pos:
                continue
            senses = ox.get("senses") or []
            scored = []
            for idx, s_blk in enumerate(senses):
                core_s = [_clean_t(x) for x in (s_blk.get("core_synonyms") or []) if _clean_t(x)]
                all_s = [_clean_t(x) for x in (s_blk.get("synonyms") or []) if _clean_t(x)]
                ex_tokens = set(re.findall(r"[a-z]{3,}", str(s_blk.get("example", "")).lower()))
                ov = (
                    len(def_lemmas & (set(core_s) | set(all_s))) * 4
                    + len((def_lemmas | quote_tokens) & ex_tokens) * 2
                    + (1 if idx == 0 else 0)
                )
                scored.append((ov, idx, core_s, all_s))
            scored.sort(key=lambda x: (-x[0], x[1]))
            has_sig = bool(def_tokens) and scored and scored[0][0] > 1
            for rank, (sc, _idx, core_s, all_s) in enumerate(scored):
                if rank == 0 or sc >= 4 or (not has_sig and len(scored) <= 2):
                    for cs in core_s:
                        if cs not in w_ortho_set:
                            strict_syns.add(cs)
                    for sw in all_s:
                        if sw not in w_ortho_set:
                            strict_syns.add(sw)

        _, ants = cls.get_pedlex_and_oxford_synonyms_antonyms(word, pos=pos, definition=definition, quote=quote)
        strict_syns -= set(ants)
        return strict_syns

