"""
PedLex SQLite connection management, orthographic variant resolution, and entry completion.
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


class PedlexCoreMixin:


    _HEADWORD_FORM_EXCEPTIONS = {
        "news", "series", "species", "means", "headquarters", "premises",
    }
    _COLLOCATION_KEY_ALIASES = {
        "adjective": "adjectives", "noun": "nouns", "verb": "verbs",
        "adverb": "adverbs", "phrase": "phrases",
    }
    _DERIVED_BORROW_KEYS = (
        "collocations", "cross_collocations", "word_family",
        "grammar_boxes", "thesaurus", "language_activator"
    )

    _NATIVE_LL_STEMS = frozenset({
        "fill", "will", "bill", "kill", "drill", "chill", "skill", "spill", "thrill",
        "grill", "mill", "pill", "till", "still", "distill", "instill", "fulfill",
        "spell", "shell", "smell", "swell", "dwell", "tell", "sell", "bell", "well", "yell",
        "call", "fall", "hall", "mall", "stall", "tall", "wall", "install", "recall", "appall", "enthrall",
        "poll", "roll", "enroll", "unroll", "scroll", "stroll", "toll", "troll",
        "dull", "full", "hull", "lull", "mull", "null", "pull", "skull",
    })

    @classmethod
    def _pure_orthographic_variants(cls, word: str) -> List[str]:
        """
        Pure rule-based AmE <-> BrE orthographic spelling alternations for a single token `word`.
        Returns an ordered list starting with `w` followed by its regional spelling variants.
        """
        st = (word or "").strip().lower()
        if not st or len(st) < 4 or " " in st:
            return [st] if st else []

        ortho_cands: List[str] = [st]
        # 1. -ll <-> -l root verbs & -ment derivatives ('enroll'/'enrol', 'fulfill'/'fulfil', 'enrollment'/'enrolment')
        if st.endswith(("enroll", "unroll", "fulfill", "instill", "distill", "appall", "enthrall")):
            ortho_cands.append(st[:-1])
        elif st.endswith(("enrol", "unrol", "fulfil", "instil", "distil", "appal", "enthral")):
            ortho_cands.append(st + "l")
        elif st.endswith(("enrollment", "unrollment", "fulfillment", "instillment", "distillment", "installment")):
            ortho_cands.append(st[:-5] + "ment")
        elif st.endswith(("enrolment", "unrolment", "fulfilment", "instilment", "distilment", "instalment")):
            ortho_cands.append(st[:-4] + "lment")
        if "llful" in st:
            ortho_cands.append(st.replace("llful", "lful"))
        elif "lful" in st and "llful" not in st:
            ortho_cands.append(st.replace("lful", "llful"))

        # 2. Single -l- <-> Double -ll- before vowel suffixes ('traveler'/'traveller', 'counselor'/'counsellor', 'marvelous'/'marvellous')
        if len(st) >= 6:
            m_ll = re.match(r"^([a-z]{2,}[eaoiu]l)l(ed|ing|er|ers|or|ors|ous|ist|ists|ery)$", st)
            if m_ll and (m_ll.group(1) + "l") not in cls._NATIVE_LL_STEMS:
                ortho_cands.append(m_ll.group(1) + m_ll.group(2))
            m_l = re.match(
                r"^([a-z]{2,}(?:vel|sel|cel|pel|tel|del|nal|gal|tal|kal|ral|ual|oil|cil|rel|nel|bel))(?<!ll)(ed|ing|er|ers|or|ors|ous|ist|ists)$",
                st,
            )
            if m_l:
                ortho_cands.append(m_l.group(1) + "l" + m_l.group(2))
        if st == "jewelry":
            ortho_cands.append("jewellery")
        elif st == "jewellery":
            ortho_cands.append("jewelry")
        elif st == "woolen":
            ortho_cands.append("woollen")
        elif st == "woollen":
            ortho_cands.append("woolen")

        # 3. Mute -e retention/drop ('sizable'/'sizeable', 'likable'/'likeable', 'judgment'/'judgement', 'aging'/'ageing')
        for am_e, br_e in (
            ("sizable", "sizeable"), ("sizably", "sizeably"),
            ("likable", "likeable"), ("lovable", "loveable"),
            ("movable", "moveable"), ("shakable", "shakeable"),
            ("usable", "useable"), ("livable", "liveable"),
            ("salable", "saleable"), ("blamable", "blameable"),
            ("unmistakable", "unmistakeable"), ("unmistakably", "unmistakeably"),
            ("judgment", "judgement"), ("judgments", "judgements"), ("judgmental", "judgemental"),
            ("acknowledgment", "acknowledgement"), ("acknowledgments", "acknowledgements"),
            ("abridgment", "abridgement"), ("lodgment", "lodgement"),
            ("aging", "ageing"),
        ):
            if st == am_e:
                ortho_cands.append(br_e)
            elif st == br_e:
                ortho_cands.append(am_e)

        # 4. -or <-> -our & productive derivatives ('color'/'colour', 'colorful'/'colourful', 'favorable'/'favourable')
        _non_our_stems = (
            "door", "floor", "poor", "moor", "prior", "major", "minor", "senior", "junior",
            "author", "doctor", "actor", "factor", "sector", "motor", "editor", "visitor",
            "mirror", "error", "terror", "horror", "sensor", "pastor", "victor", "mentor",
            "tutor", "sponsor", "cursor", "tensor", "vector", "donor", "tenor", "liquor",
        )
        m_or = re.match(r"^([a-z]{2,}?)or(s|ed|ing|able|ably|ite|ites|less|ful|fully|al|ally)?$", st)
        if m_or and len(st) >= 5:
            base_or = m_or.group(1) + "or"
            suf_or = m_or.group(2) or ""
            if not base_or.endswith(_non_our_stems):
                ortho_cands.append(m_or.group(1) + "our" + suf_or)
        m_our = re.match(r"^([a-z]{2,}?)our(s|ed|ing|able|ably|ite|ites|less|ful|fully|al|ally)?$", st)
        if m_our and len(st) >= 6:
            base_our = m_our.group(1) + "our"
            suf_our = m_our.group(2) or ""
            if not base_our.endswith(("four", "hour", "pour", "tour", "sour", "flour", "devour", "scour")):
                ortho_cands.append(m_our.group(1) + "or" + suf_our)

        # 5. -er <-> -re ('center'/'centre', 'theater'/'theatre', 'fiber'/'fibre')
        if len(st) >= 5:
            m_er = re.match(r"^([a-z]*(?:cent|theat|fib|met|lit|calib|sab|somb|spect|lust|meag|manoeuv|maneuv|och|reconnoit|sepulch|mit|goit))er(s|ed)?$", st)
            if m_er:
                ortho_cands.append(m_er.group(1) + "re" + (m_er.group(2) or ""))
            elif len(st) >= 6 and st.endswith(("ter", "ber", "ger", "ver", "ker")):
                ortho_cands.append(st[:-2] + "re")
            m_re = re.match(r"^([a-z]*(?:cent|theat|fib|met|lit|calib|sab|somb|spect|lust|meag|manoeuv|maneuv|och|reconnoit|sepulch|mit|goit))re(s|d)?$", st)
            if m_re:
                suf_re = "ed" if m_re.group(2) == "d" else (m_re.group(2) or "")
                ortho_cands.append(m_re.group(1) + "er" + suf_re)
            elif len(st) >= 6 and st.endswith(("tre", "bre", "gre", "vre", "kre")):
                ortho_cands.append(st[:-2] + "er")

        # 6. -og <-> -ogue & -gram <-> -gramme ('catalog'/'catalogue', 'dialog'/'dialogue', 'program'/'programme')
        if len(st) >= 6 and st.endswith("og"):
            ortho_cands.append(st + "ue")
        elif len(st) >= 7 and st.endswith("ogs"):
            ortho_cands.append(st[:-1] + "ues")
        elif len(st) >= 8 and st.endswith("ogue"):
            ortho_cands.append(st[:-2])
        elif len(st) >= 9 and st.endswith("ogues"):
            ortho_cands.append(st[:-3] + "s")
        if st.endswith(("program", "kilogram", "milligram")):
            ortho_cands.append(st + "me")
        elif st.endswith(("programs", "kilograms", "milligrams")):
            ortho_cands.append(st[:-1] + "mes")
        elif st.endswith(("programme", "kilogramme", "milligramme")):
            ortho_cands.append(st[:-2])
        elif st.endswith(("programmes", "kilogrammes", "milligrammes")):
            ortho_cands.append(st[:-3] + "s")

        # 7. -ize <-> -ise & -yze <-> -yse (including inflections and -ization/-isation)
        m_ize = re.match(r"^([a-z]{3,})(iz|is)(e|es|ed|ing|er|ers|ation|ations|able)$", st)
        if m_ize:
            alt_z = "is" if m_ize.group(2) == "iz" else "iz"
            ortho_cands.append(m_ize.group(1) + alt_z + m_ize.group(3))
        m_yze = re.match(r"^([a-z]{2,})(yz|ys)(e|es|ed|ing|er|ers)$", st)
        if m_yze:
            alt_y = "ys" if m_yze.group(2) == "yz" else "yz"
            ortho_cands.append(m_yze.group(1) + alt_y + m_yze.group(3))

        # 8. -ense <-> -ence ('defense'/'defence', 'offense'/'offence', 'pretense'/'pretence', 'license'/'licence')
        if len(st) >= 6:
            if st.endswith("ense"):
                ortho_cands.append(st[:-4] + "ence")
            elif st.endswith("enses"):
                ortho_cands.append(st[:-5] + "ences")
            elif st.endswith("ence"):
                ortho_cands.append(st[:-4] + "ense")
            elif st.endswith("ences"):
                ortho_cands.append(st[:-5] + "enses")

        # 9. Classical/stem prefix & root alternations ('skeptic'/'sceptic', 'gray'/'grey', 'inquire'/'enquire', etc.)
        for am_p, br_p in (
            ("skeptic", "sceptic"), ("artifact", "artefact"), ("gray", "grey"),
            ("inquir", "enquir"), ("glamor", "glamour"), ("mustache", "moustache"), ("sulfur", "sulphur"),
            ("esthetic", "aesthetic"), ("anesthe", "anaesthe"), ("archeolog", "archaeolog"),
            ("paleontolog", "palaeontolog"), ("encyclopedi", "encyclopaedi"),
            ("pediatric", "paediatric"), ("orthopedic", "orthopaedic"),
            ("gynecolog", "gynaecolog"), ("hemorrhag", "haemorrhag"),
            ("hemoglobin", "haemoglobin"), ("leukemia", "leukaemia"),
            ("diarrhea", "diarrhoea"), ("esophag", "oesophag"), ("estrogen", "oestrogen"),
            ("fetus", "foetus"), ("fetal", "foetal"), ("medieval", "mediaeval"),
            ("maneuver", "manoeuvre"), ("mold", "mould"), ("plow", "plough"),
            ("smolder", "smoulder"), ("molt", "moult"), ("cozy", "cosy"),
            ("math", "maths"), ("pajamas", "pyjamas"), ("aluminum", "aluminium"),
            ("specialty", "speciality"), ("specialties", "specialities"),
        ):
            if st.startswith(am_p):
                ortho_cands.append(br_p + st[len(am_p):])
            elif st.startswith(br_p):
                ortho_cands.append(am_p + st[len(br_p):])

        seen: Set[str] = set()
        out: List[str] = []
        for cand in ortho_cands:
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out

    @classmethod
    def orthographic_variants(cls, word: str) -> List[str]:
        """
        Full Orthographic Equivalence Closure for `word`:
        Combines rule-based AmE <-> BrE variants (`_pure_orthographic_variants`) with
        any cross-linked `payload_headword` / `alias_of` / `base_word` recorded in `pedlex.db`.
        """
        w_clean = re.sub(r"\[.*?\]|\(.*?\)", "", str(word or "")).strip().lower()
        if not w_clean:
            return []
        if " " in w_clean:
            toks = w_clean.split()
            tok_vars = [cls._pure_orthographic_variants(t) for t in toks]
            out_phrases = [w_clean]
            for idx, v_list in enumerate(tok_vars):
                if len(v_list) > 1:
                    for v in v_list[1:]:
                        cand_p = " ".join(toks[:idx] + [v] + toks[idx + 1 :])
                        if cand_p not in out_phrases:
                            out_phrases.append(cand_p)
            return out_phrases

        cands = list(cls._pure_orthographic_variants(w_clean))
        try:
            entry = cls.get_ldoce_entry(w_clean)
            if entry:
                for k in ("word", "payload_headword", "base_word", "alias_of"):
                    val = str(entry.get(k) or "").strip().lower()
                    val = re.sub(r"_\d+$", "", val)
                    if val and " " not in val and val not in cands:
                        # Only include if it is an orthographic variant or shares the same lemma/stem
                        if (
                            val in cls._pure_orthographic_variants(w_clean)
                            or w_clean in cls._pure_orthographic_variants(val)
                            or (hasattr(cls, "are_spelling_variants") and cls.are_spelling_variants(w_clean, val))
                        ):
                            cands.append(val)
                            for sub_v in cls._pure_orthographic_variants(val):
                                if sub_v not in cands:
                                    cands.append(sub_v)
        except Exception:
            pass
        return cands

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

        # Systematic AmE <-> BrE headword spelling alternations (evaluated after literal stems)
        ortho_cands: List[str] = []
        for st in list(cands):
            for ov in cls._pure_orthographic_variants(st):
                if ov != st:
                    ortho_cands.append(ov)
        cands.extend(ortho_cands)

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
        legacy = Path(__file__).resolve().parent.parent / "data" / "ldoce6_essential.db"
        if legacy.exists():
            return legacy
        return primary

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

    _XREF_SPELLING_RE = re.compile(
        r"^(?:(?:a|an|the)\s+)?(?:(?:usual|old|old-fashioned|informal)\s+)?(?:American|British|US|UK|another|variant|alternative)\s+(?:English\s+)?(?:spelling|form)\s+of\s+([a-zA-Z\-]+?)(?:\d+|\(\d+\))?(?:\s*\(.*\))?\.?$|^(?:same\s+as|see)\s+([a-zA-Z\-]+)\.?$",
        re.IGNORECASE,
    )

    @classmethod
    def _norm_def_for_ortho_dedup(cls, text: str) -> str:
        s = re.sub(r"\s+", " ", (text or "").strip().lower())
        if hasattr(cls, "_normalize_bre_ame"):
            return " ".join(
                cls._normalize_bre_ame(tok) if tok.isalpha() else tok
                for tok in re.findall(r"[a-z]+|[^a-z]+", s)
            )
        return s

    @classmethod
    def _finish_ldoce_entry(cls, w_clean: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Single exit point: normalize collocation keys, expand cross-reference spelling senses,
        merge regional AmE <-> BrE orthographic variant payloads, and cache."""
        data["collocations"] = cls._normalize_collocations(data.get("collocations"))

        def _expand_xref_senses(entry_dict: Dict[str, Any], owner_w: str) -> List[Dict[str, Any]]:
            raw_s = entry_dict.get("senses")
            if not isinstance(raw_s, list) or not raw_s:
                return []
            expanded: List[Dict[str, Any]] = []
            for s in raw_s:
                if isinstance(s, dict):
                    s_defn = str(s.get("definition") or "").strip()
                    m_xref = cls._XREF_SPELLING_RE.match(s_defn)
                    if m_xref:
                        target_hw = (m_xref.group(1) or m_xref.group(2) or "").strip().lower()
                        if target_hw and target_hw != owner_w:
                            t_row = cls._ldoce_raw_row(target_hw)
                            if t_row:
                                t_data = cls._resolve_ldoce_kind(target_hw, target_hw, *t_row)
                                t_senses = [
                                    ts for ts in (t_data.get("senses") or [])
                                    if isinstance(ts, dict)
                                    and str(ts.get("definition") or "").strip()
                                    and not cls._XREF_SPELLING_RE.match(str(ts.get("definition") or "").strip())
                                ]
                                if t_senses:
                                    expanded.extend(t_senses)
                                    for meta_k in ("collocations", "cross_collocations", "thesaurus", "language_activator", "grammar_boxes", "word_family", "phrases", "phrasal_verbs", "units", "all_poses"):
                                        if not entry_dict.get(meta_k) and t_data.get(meta_k):
                                            entry_dict[meta_k] = t_data[meta_k]
                                    continue
                expanded.append(s)
            return expanded

        orig_had_only_xref = bool(
            data.get("senses")
            and all(
                isinstance(s, dict) and bool(cls._XREF_SPELLING_RE.match(str(s.get("definition") or "").strip()))
                for s in (data.get("senses") or [])
            )
        )
        data["senses"] = _expand_xref_senses(data, w_clean)

        # Orthographic Equivalence Closure across AmE <-> BrE rows in `ldoce`:
        # When both regional spellings exist as rows in `ldoce` (e.g. `favor` has only noun senses while
        # `favour` has noun + verb senses and 13 `language_activator` clusters; `analyze` and `analyse`
        # each hold 1 distinct `language_activator` cluster), merge their senses and lexicographic compartments
        # so AmE and BrE queries observe identical lexical knowledge.
        if " " not in w_clean:
            ortho_peers = [ov for ov in cls._pure_orthographic_variants(w_clean) if ov != w_clean]
            if ortho_peers:
                cur_senses = list(data.get("senses") or [])
                sense_by_sig: Dict[Tuple[str, str], Dict[str, Any]] = {}
                for s in cur_senses:
                    if isinstance(s, dict):
                        sp = str(s.get("pos") or data.get("pos") or "").strip().lower()
                        sd = cls._norm_def_for_ortho_dedup(str(s.get("definition") or ""))
                        if sd:
                            sense_by_sig.setdefault((sp, sd), s)

                for ov in ortho_peers:
                    ov_row = cls._ldoce_raw_row(ov)
                    if not ov_row:
                        continue
                    ov_data = cls._resolve_ldoce_kind(ov, ov, *ov_row)
                    ov_senses = _expand_xref_senses(ov_data, ov)
                    for os_item in ov_senses:
                        if not isinstance(os_item, dict):
                            continue
                        os_def = str(os_item.get("definition") or "").strip()
                        if not os_def or cls._XREF_SPELLING_RE.match(os_def):
                            continue
                        os_pos = str(os_item.get("pos") or ov_data.get("pos") or "").strip().lower()
                        sig = (os_pos, cls._norm_def_for_ortho_dedup(os_def))
                        existing_s = sense_by_sig.get(sig)
                        if existing_s is None:
                            s_copy = dict(os_item)
                            if not s_copy.get("pos") and os_pos:
                                s_copy["pos"] = os_pos
                            cur_senses.append(s_copy)
                            sense_by_sig[sig] = s_copy
                        else:
                            for list_k in ("patterns", "units", "examples"):
                                if os_item.get(list_k):
                                    merged_l = list(existing_s.get(list_k) or [])
                                    for elem in os_item[list_k]:
                                        if elem and elem not in merged_l:
                                            merged_l.append(elem)
                                    existing_s[list_k] = merged_l
                            for syn_k in ("synonym", "opposite"):
                                if os_item.get(syn_k) and not existing_s.get(syn_k):
                                    existing_s[syn_k] = os_item[syn_k]
                    data["senses"] = cur_senses

                    # Merge all_poses
                    all_p = list(data.get("all_poses") or ([data["pos"]] if data.get("pos") else []))
                    for op in (ov_data.get("all_poses") or ([ov_data["pos"]] if ov_data.get("pos") else [])):
                        if op and op not in all_p:
                            all_p.append(op)
                    if all_p:
                        data["all_poses"] = all_p
                    if orig_had_only_xref and ov_data.get("pos"):
                        data["pos"] = ov_data["pos"]

                    # Merge collocations
                    ov_collocs = cls._normalize_collocations(ov_data.get("collocations"))
                    if ov_collocs:
                        cur_collocs = data.get("collocations") or {}
                        for ck, citems in ov_collocs.items():
                            cbucket = cur_collocs.setdefault(ck, [])
                            for ci in citems or []:
                                if ci and ci not in cbucket:
                                    cbucket.append(ci)
                        data["collocations"] = cur_collocs

                    # Merge language_activator
                    if ov_data.get("language_activator"):
                        cur_la = list(data.get("language_activator") or [])
                        seen_la = {str(x.get("concept") or "").strip().lower() for x in cur_la if isinstance(x, dict)}
                        for la_item in ov_data["language_activator"]:
                            if isinstance(la_item, dict):
                                c_key = str(la_item.get("concept") or "").strip().lower()
                                if c_key not in seen_la:
                                    seen_la.add(c_key)
                                    cur_la.append(la_item)
                        data["language_activator"] = cur_la

                    # Merge thesaurus
                    if ov_data.get("thesaurus"):
                        cur_th = list(data.get("thesaurus") or [])
                        seen_th = {
                            (cls._norm_def_for_ortho_dedup(str(x.get("word") or "")), x.get("section_index", 0))
                            for x in cur_th if isinstance(x, dict)
                        }
                        for th_item in ov_data["thesaurus"]:
                            if isinstance(th_item, dict):
                                t_key = (cls._norm_def_for_ortho_dedup(str(th_item.get("word") or "")), th_item.get("section_index", 0))
                                if t_key not in seen_th:
                                    seen_th.add(t_key)
                                    cur_th.append(th_item)
                        data["thesaurus"] = cur_th

                    # Merge grammar_boxes, phrases, phrasal_verbs, units
                    for blk_k, id_field in (
                        ("grammar_boxes", "heading"),
                        ("phrases", "phrase"),
                        ("phrasal_verbs", "phrase"),
                        ("units", "phrase"),
                    ):
                        if ov_data.get(blk_k):
                            cur_blks = list(data.get(blk_k) or [])
                            seen_blks = {
                                cls._norm_def_for_ortho_dedup(str(b.get(id_field) or b.get("headword") or b.get("title") or ""))
                                for b in cur_blks if isinstance(b, dict)
                            }
                            for b_item in ov_data[blk_k]:
                                if isinstance(b_item, dict):
                                    b_key = cls._norm_def_for_ortho_dedup(str(b_item.get(id_field) or b_item.get("headword") or b_item.get("title") or ""))
                                    if b_key and b_key not in seen_blks:
                                        seen_blks.add(b_key)
                                        cur_blks.append(b_item)
                            data[blk_k] = cur_blks

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

        # Promote phrasal_verbs senses into entry["senses"] immediately after base verb senses
        # (or appended at the end if 0 verb senses exist, e.g. 'account').
        # Setting sensenum="" and from_phrasal_verb=True activates _lock_sense's Phrasal Verb Particle Gate (pv_sense_map)
        # and keeps entry["senses"][idx] valid for disambiguation.
        pv_blocks = data.get("phrasal_verbs")
        if isinstance(pv_blocks, list) and pv_blocks and not any(isinstance(s, dict) and s.get("from_phrasal_verb") for s in (data.get("senses") or [])):
            cur_senses = list(data.get("senses") or [])
            last_v_idx = -1
            for idx, s in enumerate(cur_senses):
                if isinstance(s, dict) and cls._pos_code(s.get("pos")) == "v":
                    last_v_idx = idx

            existing_hnums = [
                int(s["homograph_num"])
                for s in cur_senses
                if isinstance(s, dict) and isinstance(s.get("homograph_num"), int) and s.get("homograph_num") > 0
            ]
            next_hnum = (max(existing_hnums) + 1) if existing_hnums else None

            promoted_senses = []
            for pv_blk in pv_blocks:
                if not isinstance(pv_blk, dict):
                    continue
                pv_phrase = str(pv_blk.get("phrase") or pv_blk.get("headword") or "").strip()
                for pv_s in (pv_blk.get("senses") or []):
                    if not isinstance(pv_s, dict) or not str(pv_s.get("definition") or "").strip():
                        continue
                    promoted = dict(pv_s)
                    promoted["pos"] = "verb"
                    promoted["sensenum"] = ""
                    promoted["homograph_num"] = next_hnum
                    promoted["from_phrasal_verb"] = True
                    if pv_phrase:
                        units = list(promoted.get("units") or [])
                        if pv_phrase not in units:
                            units.insert(0, pv_phrase)
                        promoted["units"] = units
                        pats = list(promoted.get("patterns") or [])
                        if pv_phrase not in pats:
                            pats.insert(0, pv_phrase)
                        promoted["patterns"] = pats
                    promoted_senses.append(promoted)

            if promoted_senses:
                if last_v_idx >= 0:
                    data["senses"] = cur_senses[:last_v_idx + 1] + promoted_senses + cur_senses[last_v_idx + 1:]
                else:
                    data["senses"] = cur_senses + promoted_senses

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
            try:
                from .pedlex_dist import ensure_pedlex_database
                ensure_pedlex_database(db_path)
            except Exception:
                pass
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
        if w_clean in cls._ldoce_in_progress:
            return None

        if not cls._ensure_ldoce_conn():
            return None

        cls._ldoce_in_progress.add(w_clean)
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

            cls._ldoce_cache[w_clean] = None
            return None
        except Exception:
            return None
        finally:
            cls._ldoce_in_progress.discard(w_clean)

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
    #   derived          : run-on derivative without independent dictionary definition/examples -> excluded
    #   alias            : the row is an empty pointer - follow base_word once
    # ------------------------------------------------------------------

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

        if kind == "derived":
            # Own sense, own gloss - only the borrowable assets come from the base.
            if base_data:
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
            return data

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

        return data

