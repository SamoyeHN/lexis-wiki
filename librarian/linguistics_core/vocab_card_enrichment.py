"""
Vocabulary card enrichment, semantic discriminator generation, and batch-safe reanchoring.
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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics")


class VocabCardEnrichmentMixin:
    @classmethod
    def _contrast_strategy_clause(
        cls,
        target_word: str,
        definition: Optional[str],
        dist_list: List[str],
        dist_meta: Optional[Dict[str, str]],
        anchor_type: Optional[str] = None
    ) -> str:
        """
        Sense-aware contrast instruction .

        The blueprint pasted the same 'Prefer a concessive or cause-and-effect turn (e.g.
        Although/Despite...)' into every noun item whatever the distractors were. A concessive
        turn only discriminates when the distractor set actually contains a polar opposite;
        for near-synonym clusters ('earnest' vs 'serious') the discriminator is precision of
        meaning, and for taxonomically distant distractors no contrast is needed at all. The
        instruction is now derived from the distractor mechanisms the generator actually used.
        """
        dists = [d for d in (dist_list or []) if d]
        if not dists:
            return ""
        meta = dist_meta or {}
        tw = (target_word or "").strip().lower()
        antonyms = [d for d in dists if meta.get(d) == "antonym"]
        near_synonyms = [d for d in dists if meta.get(d) in
                         ("ldoce_thesaurus", "satellite_synonym", "synonym", "troponym",
                          "coordinate", "attribute_coordinate", "preposition_valency")]
        if antonyms:
            return (f" Establish an explicit polarity cue (an evaluative word or a situational outcome) "
                    f"so the opposite '{antonyms[0]}' is logically excluded.")
        if near_synonyms:
            named = ", ".join(f"'{d}'" for d in near_synonyms[:3])
            return (f" Establish a precision cue — the exact scale, register, or "
                    f"collocational frame that separates '{tw}' from {named}.")
        return " Establish situational and definitional clues that make the distinction decisive."

    @staticmethod
    def _definition_overlap_ratio(text_a: Optional[str], text_b: Optional[str]) -> float:
        """Overlap between two definition texts, used to detect self-referential notes."""
        _DEF_STOP = {
            "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with", "from",
            "into", "onto", "upon", "about", "as", "or", "and", "that", "which", "who",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "can", "could", "will", "would", "way", "used", "especially",
            "someone", "something", "somebody", "things", "person", "people", "your", "you",
            "their", "them", "they", "it", "its", "not", "very", "more", "most", "much", "many"
        }
        raw_a = set(re.findall(r"[a-z]+", (text_a or "").lower()))
        raw_b = set(re.findall(r"[a-z]+", (text_b or "").lower()))
        if not raw_a or not raw_b:
            return 0.0
        ta = raw_a - _DEF_STOP
        tb = raw_b - _DEF_STOP
        if not ta or not tb:
            ta, tb = raw_a, raw_b
        return len(ta & tb) / float(min(len(ta), len(tb)))

    @staticmethod
    def _thesaurus_variant_matches(th_word: Optional[str], word: Optional[str]) -> bool:
        """
        PedLex thesaurus headwords can be multi-form ('unwilling/not willing',
        'be loath to do something'), so match the plain distractor lemma against
        any of the recorded variants rather than the raw string.
        """
        t = (th_word or "").strip().lower()
        w = (word or "").strip().lower()
        if not t or not w:
            return False
        if t == w:
            return True
        variants = [v.strip() for v in re.split(r"[/|,]", t) if v.strip()]
        variants += [re.sub(r"^(be|get|have) ", "", v).strip() for v in variants]
        if w in variants:
            return True
        # For single-word candidate w, only match if variant is single-word or explicitly parenthesized
        for v in variants:
            if not v:
                continue
            # Strip parenthesized optional parts: e.g. '(be) willing' -> 'willing'
            v_clean = re.sub(r"\(.*?\)", "", v).strip()
            if v_clean == w:
                return True
            # Strip trailing sth/sb: e.g. 'command of sth' -> 'command of'
            v_nobound = re.sub(r"\b(sth|sb|something|someone)\b", "", v).strip()
            if v_nobound == w:
                return True
            # Strip infinitive tails: e.g. 'loath to do something' -> 'loath'
            v_inf = re.sub(r"\bto do (something|sth)\b|\bto (sth|sb|something|someone)\b", "", v).strip()
            if v_inf == w:
                return True
        return False

    @classmethod
    def get_vocab_card_enrichment(
        cls,
        word: str,
        pos: str = "",
        definition: str = "",
        quote: str = ""
    ) -> Dict[str, str]:
        """
        Deterministically extracts zero-token pedagogical enrichment for a vocabulary card:
        - `collocations`: Top sense-locked collocations / syntactic patterns from `pedlex.db`.
        - `grammar_usage`: Sense-locked `<gram>` tags (e.g. `[only before noun]`, `[not before noun]`,
          `[uncountable]`, `[countable]`), key syntactic frame, and sense-locked `grammar_boxes` rules (`✗ Don't say: ...`).
        - `thesaurus_note`: Sense-aligned `THESAURUS` synonym distinctions and `[grammar]` constraints.
        """
        out: Dict[str, str] = {
            "collocations": "",
            "grammar_usage": "",
            "thesaurus_note": "",
        }
        raw_w = str(word or "").strip()
        if " " in raw_w:
            return out
        w_clean = re.sub(r"\[.*?\]|\(.*?\)", "", raw_w).strip().lower()
        if not w_clean or " " in w_clean:
            return out

        entry = cls.get_ldoce_entry(w_clean)
        if not entry:
            return out

        out["core_dots"] = str(entry.get("core_dots") or "")
        out["freq_tags"] = " ".join(entry.get("freq_tags") or [])

        senses = entry.get("senses") or []
        t_s_idx = None
        locked_sense = None
        if senses:
            t_s_idx = cls._lock_sense(
                entry,
                definition=definition,
                quote=quote,
                target_pos=pos
            )
            if t_s_idx is not None and 0 <= t_s_idx < len(senses):
                locked_sense = senses[t_s_idx]

        p_code = cls._pos_code(pos or (locked_sense.get("pos") if locked_sense else entry.get("pos")))

        def _clean_pattern_text(pat_str: str) -> str:
            s = str(pat_str or "").strip()
            if not s:
                return ""
            m_emb = re.search(r"^(.*?)\s+([A-Z][\w\W]*\b[\.\?!])$", s)
            if not m_emb:
                m_etc = re.search(r"^(.*?\betc\.?)\s+([A-Z].*\s+.*)$", s)
                if m_etc:
                    m_emb = m_etc
            if m_emb:
                s = m_emb.group(1).strip()
            s = re.sub(r"\b(sth|sb|something|somebody)\s+[A-Z][a-z]+.*$", r"\1", s).strip()
            s = re.sub(r"\bsth\b", "something", s)
            s = re.sub(r"\bsb\b", "somebody", s)
            s = re.sub(
                r"\s*\b(?:(?:especially|mainly|chiefly|mostly|usually)\s+)?(?:British|American|Australian|Canadian|North American|Scottish|Irish)\s+English\b[,;:\.]*",
                "",
                s,
                flags=re.IGNORECASE,
            ).strip(" ,;:")
            return s

        # 1. Grammar Tag (<gram> tags for inline header) & Pattern Note (key pattern + grammar_boxes rule)
        gram_tag_str = ""
        pattern_parts: List[str] = []
        if locked_sense:
            raw_gram = str(locked_sense.get("grammar") or "").strip()
            if raw_gram and raw_gram.strip("[]` ").lower() not in ("noun", "verb", "adjective", "adverb", "preposition", "conjunction", "interjection"):
                gram_tag_str = raw_gram.strip("`").strip()
            if p_code in ("a", "adj") and "not before noun" in gram_tag_str.lower():
                att_texts = [quote] if quote else []
                att_texts.extend(str(ex) for ex in (locked_sense.get("examples") or [])[:3] if ex)
                has_attr_use = False
                nlp_inst = cls.get_spacy()
                for atxt in att_texts:
                    adoc = nlp_inst(atxt)
                    if any(t.lower_ == w_clean and t.dep_ == "amod" and t.head.pos_ in ("NOUN", "PROPN") for t in adoc):
                        has_attr_use = True
                        break
                if has_attr_use:
                    gram_tag_str = re.sub(r"\[?\bnot before noun\b\]?", "", gram_tag_str, flags=re.IGNORECASE).strip(" ,;[]")
            elif p_code in ("a", "adj") and "only before noun" in gram_tag_str.lower() and quote:
                nlp_inst = cls.get_spacy()
                qdoc = nlp_inst(quote)
                has_pred_use = any(
                    t.lower_ == w_clean and (
                        t.dep_ in ("acomp", "oprd")
                        or (t.head != t and t.head.lemma_.lower() in ("be", "become", "seem", "feel", "get", "remain", "stay", "grow", "turn", "appear", "look", "sound") and t.dep_ != "amod")
                    )
                    for t in qdoc
                )
                if has_pred_use:
                    gram_tag_str = re.sub(r"\bonly before noun\b", "usually before noun", gram_tag_str, flags=re.IGNORECASE)
            raw_pats_and_units = list(dict.fromkeys(
                list(locked_sense.get("syntactic_patterns") or locked_sense.get("patterns") or [])
                + [
                    u for u in (locked_sense.get("units") or [])
                    if not (p_code == "v" and re.search(rf"\b(?:a|an|the|to a)\s+{re.escape(w_clean)}\b", str(u).lower()))
                ]
            ))
            pats = [
                _clean_pattern_text(p) for p in raw_pats_and_units
                if _clean_pattern_text(p) and _clean_pattern_text(p).lower() != w_clean
            ]
            if pats:
                pattern_parts.append(f"*{pats[0]}*")

        gb_alert = cls.get_ldoce_grammar_alert(w_clean, quote=quote, sense_index=t_s_idx, pos=p_code)
        if gb_alert:
            rules = []
            for r in (gb_alert.get("rules") or []):
                r_s = str(r.get("text") if isinstance(r, dict) else r or "").strip()
                r_s = re.sub(r"^[•·\-\*]\s*", "", r_s).strip()
                r_s = re.sub(r"\s*Grammar guide\s*[‒–—-]\s*[A-Z\s]+$", "", r_s).strip()
                if r_s:
                    rules.append(r_s)
            bad_exs = [str(b).strip() for b in (gb_alert.get("bad_examples") or []) if str(b).strip()]
            if rules:
                r_txt = rules[0]
                if bad_exs and "don't say" not in r_txt.lower() and bad_exs[0].lower() not in r_txt.lower():
                    r_txt += f" (✗ *{bad_exs[0]}*)"
                pattern_parts.append(r_txt)
            elif bad_exs:
                pattern_parts.append(f"✗ Don't say: *{bad_exs[0]}*")

        out["grammar_tag"] = gram_tag_str
        out["pattern_note"] = " — ".join(pattern_parts) if pattern_parts else ""
        gram_legacy_parts: List[str] = []
        if gram_tag_str:
            gram_legacy_parts.append(f"`{gram_tag_str}`")
        if pattern_parts:
            first_pat = pattern_parts[0]
            if first_pat.startswith("*") and first_pat.endswith("*"):
                gram_legacy_parts.append(f"Pattern: {first_pat}")
            else:
                gram_legacy_parts.append(first_pat)
            gram_legacy_parts.extend(pattern_parts[1:])
        if gram_legacy_parts:
            out["grammar_usage"] = " — ".join(gram_legacy_parts)

        # 2. Zero-Token Pedagogical Collocations (100% authentic dictionary collocations/patterns; zero fabricated splices)
        colloc_items: List[str] = []
        locked_homograph = locked_sense.get("homograph_num") if locked_sense else None
        locked_def_kws: Set[str] = set()
        other_sense_kws: Set[str] = set()
        _colloc_stop = {
            "with", "from", "into", "over", "under", "more", "most", "very", "some", "such",
            "that", "this", "what", "when", "where", "which", "while", "able", "used", "make",
            "made", "have", "much", "many", "something", "somebody", "someone", "particular",
            "especially", "usually", "often", "always", "never", "other", "another", "between",
            "through", "without", "within", "before", "after", "about", "because", "though",
        }
        if locked_sense and len(senses) > 1:
            l_signpost = str(locked_sense.get("signpost") or "")
            l_syns = " ".join(str(x) for x in (locked_sense.get("synonyms") or []))
            l_exs = " ".join(str(x) for x in (locked_sense.get("examples") or [])[:4])
            l_units = " ".join(str(x) for x in (locked_sense.get("units") or []) + (locked_sense.get("patterns") or []) + (locked_sense.get("collocations") or []))
            l_def_str = f"{l_signpost} {l_syns} {definition or cls.format_ldoce_definition(locked_sense) or ''} {l_exs} {l_units}".lower()
            for lw in re.findall(r"[a-z]{4,}", l_def_str):
                if lw != w_clean and lw not in _colloc_stop and not cls.form_matches(lw, w_clean):
                    locked_def_kws.add(lw)
                    lw_lem = cls._surface_lemma(lw)
                    if lw_lem and lw_lem not in _colloc_stop:
                        locked_def_kws.add(lw_lem)
            for s_other in senses:
                if s_other is locked_sense:
                    continue
                o_signpost = str(s_other.get("signpost") or "")
                o_syns = " ".join(str(x) for x in (s_other.get("synonyms") or []))
                o_exs = " ".join(str(x) for x in (s_other.get("examples") or [])[:4])
                o_units = " ".join(str(x) for x in (s_other.get("units") or []) + (s_other.get("patterns") or []) + (s_other.get("collocations") or []))
                o_def_str = f"{o_signpost} {o_syns} {cls.format_ldoce_definition(s_other) or ''} {o_exs} {o_units}".lower()
                for ow in re.findall(r"[a-z]{4,}", o_def_str):
                    if ow != w_clean and ow not in _colloc_stop and not cls.form_matches(ow, w_clean):
                        ow_lem = cls._surface_lemma(ow) or ow
                        if ow not in locked_def_kws and ow_lem not in locked_def_kws:
                            other_sense_kws.add(ow)
                            if ow_lem not in _colloc_stop:
                                other_sense_kws.add(ow_lem)

        entry_p_code = cls._pos_code(entry.get("pos"))
        all_poses = [cls._pos_code(p) for p in (entry.get("all_poses") or []) if cls._pos_code(p)]
        is_multi_pos = len(set(all_poses)) > 1
        verb_only_preps: Set[str] = set()
        noun_only_preps: Set[str] = set()
        if is_multi_pos and "v" in all_poses and "n" in all_poses:
            v_preps = set(cls.get_ldoce_preps(w_clean, pos="verb"))
            n_preps = set(cls.get_ldoce_preps(w_clean, pos="noun"))
            verb_only_preps = v_preps - n_preps
            noun_only_preps = n_preps - v_preps

        def _sense_pos_matches(s_obj: Dict[str, Any]) -> bool:
            if not p_code:
                return True
            s_p = cls._pos_code(s_obj.get("pos"))
            if s_p and s_p != p_code:
                return False
            if locked_homograph is not None and s_obj.get("homograph_num") is not None:
                if s_obj.get("homograph_num") != locked_homograph:
                    return False
            return True

        def _add_colloc(cand_str: str, check_cross_sense: bool = False, extra_context: str = "", require_sense_overlap: bool = False) -> None:
            c_s = _clean_pattern_text(cand_str)
            if not c_s or " " not in c_s or "..." in c_s:
                return
            c_low = c_s.lower()
            if c_low == w_clean or c_low in (f"the {w_clean}", f"a {w_clean}", f"an {w_clean}"):
                return
            if p_code == "v" and (
                re.search(rf"\b(?:a|an|the|to a|under|out of|in|beyond|within|without)\s+{re.escape(w_clean)}\b", c_low)
                or re.match(r"^(?:a|an|the)\s+[a-z]+(?:ing|ed)\b", c_low)
            ):
                return
            c_words = c_low.split()
            if p_code == "n" and len(c_words) == 2 and c_words[1] == w_clean and re.match(r"^[a-z]+ing$", c_words[0]):
                return
            if is_multi_pos and p_code == "n" and "v" in all_poses and c_words and c_words[0] == w_clean:
                # Guard against verb-initial frames leaking onto noun cards (e.g., "study to be...", "study law/business/...", "study for an exam")
                if len(c_words) >= 2 and (
                    c_words[1] == "to"
                    or "/" in c_words[1]
                    or c_words[1] in verb_only_preps
                ):
                    return
            if is_multi_pos and p_code == "v" and "n" in all_poses and c_words and c_words[0] == w_clean:
                if len(c_words) >= 2 and c_words[1] in noun_only_preps:
                    return
            if check_cross_sense and (other_sense_kws or (len(senses) > 1 and t_s_idx is not None)):
                c_toks: Set[str] = set()
                probe_low = f"{c_low} {extra_context.lower()}" if extra_context else c_low
                for cw in re.findall(r"[a-z]{4,}", probe_low):
                    if cw != w_clean and cw not in _colloc_stop and not cls.form_matches(cw, w_clean):
                        c_toks.add(cw)
                        cw_lem = cls._surface_lemma(cw)
                        if cw_lem and cw_lem not in _colloc_stop:
                            c_toks.add(cw_lem)
                if c_toks & other_sense_kws and len(c_toks & other_sense_kws) >= len(c_toks & locked_def_kws):
                    return
                if len(senses) > 1 and t_s_idx is not None and (require_sense_overlap or (t_s_idx > 0 and (c_toks & other_sense_kws))) and not (c_toks & locked_def_kws):
                    return
            for idx_e, ex_item in enumerate(colloc_items):
                ex_low = ex_item.lower()
                if c_low == ex_low:
                    return
                if c_low in ex_low:
                    return
                if ex_low in c_low:
                    colloc_items[idx_e] = c_s
                    return
            colloc_items.append(c_s)

        # Tier A: Locked sense authentic collocations, pattern bundles, patterns, and lexical units
        if locked_sense:
            for c in locked_sense.get("collocations") or []:
                _add_colloc(str(c))
            for pb in locked_sense.get("pattern_bundles") or []:
                _add_colloc(str(pb.get("pattern") or ""))
            for pat in locked_sense.get("patterns") or []:
                _add_colloc(str(pat))
            for u in locked_sense.get("units") or []:
                _add_colloc(str(u))

        # Tier B: Authentic multi-word dictionary Collocations Box entries (source != "corpus")
        colloc_box_pos_ok = (
            not entry_p_code
            or not p_code
            or entry_p_code == p_code
            or not is_multi_pos
        )
        if len(colloc_items) < 5:
            raw_collocs = entry.get("collocations")
            if isinstance(raw_collocs, dict):
                allowed_cats = ("phrases", "verbs", "adjectives", "nouns", "adverbs")
                if p_code == "v":
                    allowed_cats = ("adverbs", "phrases", "nouns")
                elif p_code == "n":
                    allowed_cats = ("verbs", "adjectives", "phrases", "nouns")
                elif p_code == "adj":
                    allowed_cats = ("adverbs", "nouns", "phrases")
                elif p_code == "adv":
                    allowed_cats = ("verbs", "adjectives", "phrases")
                for cat_k in allowed_cats:
                    for it_c in (raw_collocs.get(cat_k) or []):
                        if not isinstance(it_c, dict) or it_c.get("source") == "corpus":
                            continue
                        it_p_code = cls._pos_code(str(it_c.get("pos") or ""))
                        if p_code and it_p_code in ("n", "v", "adj", "adv"):
                            if it_p_code != p_code:
                                continue
                        elif not colloc_box_pos_ok:
                            continue
                        it_s_indices = it_c.get("sense_indices")
                        if isinstance(it_s_indices, list) and it_s_indices and t_s_idx is not None:
                            if t_s_idx not in it_s_indices:
                                continue
                            has_direct_sense_link = True
                        else:
                            has_direct_sense_link = False
                        c_val = _clean_pattern_text(str(it_c.get("collocation") or ""))
                        if not c_val or " " not in c_val:
                            continue
                        it_ex = str(it_c.get("example") or "").strip()
                        if c_val[0].isupper() and it_ex.startswith(c_val):
                            continue
                        if not has_direct_sense_link and len(senses) > 1 and t_s_idx is not None:
                            for g_box in (entry.get("grammar_boxes") or []):
                                g_s_idxs = g_box.get("sense_indices")
                                if isinstance(g_s_idxs, list) and g_s_idxs and t_s_idx not in g_s_idxs:
                                    g_exs = [str(gx).strip() for gx in (g_box.get("examples") or []) if gx]
                                    if it_ex and it_ex in g_exs:
                                        c_val = ""
                                        break
                            if not c_val:
                                continue
                        _add_colloc(
                            c_val,
                            check_cross_sense=not has_direct_sense_link,
                            extra_context=it_ex,
                            require_sense_overlap=(not has_direct_sense_link and len(senses) > 1 and t_s_idx is not None and t_s_idx > 0),
                        )
                        if len(colloc_items) >= 5:
                            break
                    if len(colloc_items) >= 5:
                        break

        # Tier C: Same-POS/same-homograph sibling senses' patterns and lexical units (only when not conflicting with locked sense)
        if len(colloc_items) < 5 and len(senses) <= 1:
            for s_obj in senses:
                if s_obj is locked_sense or not _sense_pos_matches(s_obj):
                    continue
                for pb in s_obj.get("pattern_bundles") or []:
                    _add_colloc(str(pb.get("pattern") or ""), check_cross_sense=True)
                for pat in s_obj.get("patterns") or []:
                    _add_colloc(str(pat), check_cross_sense=True)
                for u in s_obj.get("units") or []:
                    _add_colloc(str(u), check_cross_sense=True)
                if len(colloc_items) >= 5:
                    break

        # Tier D: Entry-level fixed phrases & cross-referenced dictionary collocations (when单POS or needed)
        is_ed_adj = (p_code == "adj" and w_clean.endswith("ed"))
        allow_unindexed_tier_d = (len(senses) <= 1 or len(colloc_items) == 0)
        if len(colloc_items) < 5 and colloc_box_pos_ok and not is_ed_adj and allow_unindexed_tier_d:
            for ph_obj in (entry.get("phrases") or []):
                ph_str = ph_obj.get("phrase") or ph_obj.get("title") if isinstance(ph_obj, dict) else str(ph_obj or "")
                ph_ctx = ""
                if isinstance(ph_obj, dict):
                    ph_ctx = f"{ph_obj.get('resolved_definition') or ''} {' '.join(str(x) for x in (ph_obj.get('examples') or [])[:2])}"
                _add_colloc(str(ph_str or ""), check_cross_sense=True, extra_context=ph_ctx, require_sense_overlap=(len(senses) > 1))
                if len(colloc_items) >= 5:
                    break
        if len(colloc_items) < 5 and not is_multi_pos and not is_ed_adj and allow_unindexed_tier_d:
            for cr_obj in (entry.get("cross_collocations") or []):
                if isinstance(cr_obj, dict):
                    cr_ctx = f"{cr_obj.get('gloss') or ''} {cr_obj.get('example') or ''}"
                    _add_colloc(str(cr_obj.get("collocation") or ""), check_cross_sense=True, extra_context=cr_ctx)
                    if len(colloc_items) >= 5:
                        break

        if colloc_items:
            out["collocations"] = "; ".join(colloc_items[:5])

        # 3. Sense-Aligned THESAURUS Note (exclude target word itself; rank peers by semantic overlap & proximity)
        thes_list = entry.get("thesaurus") or []
        if thes_list:
            def _thes_item_matches_pos(t_obj: Dict[str, Any]) -> bool:
                if not is_multi_pos or not p_code:
                    return True
                t_pos_code = cls._pos_code(str(t_obj.get("pos") or ""))
                if t_pos_code in ("n", "v", "adj", "adv") and t_pos_code != p_code:
                    return False
                tw_low = str(t_obj.get("word") or "").strip().lower()
                tg_low = str(t_obj.get("grammar") or "").lower()
                td_low = re.sub(r"^[:\s]+", "", str(t_obj.get("distinction") or "")).strip().lower()
                has_n_gram = any(k in tg_low for k in ("countable", "uncountable", "singular", "plural", "noun"))
                has_v_gram = any(k in tg_low for k in ("transitive", "intransitive", "phrasal verb"))
                starts_v_def = bool(re.match(r"^(?:to\s+[a-z]+|if\s+you\s+[a-z]+)\b", td_low))
                starts_n_def = bool(
                    re.match(r"^(?:a|an|the|one\s+of\s+the|something\s+that|information\s+about|someone\s+who)\b", td_low)
                )
                is_reflexive_vp = any(r in tw_low for r in ("yourself", "himself", "herself", "oneself", "themselves"))
                if p_code == "v":
                    if has_n_gram or starts_n_def or (not has_v_gram and not starts_v_def):
                        return False
                elif p_code == "n":
                    if has_v_gram or is_reflexive_vp or (starts_v_def and not has_n_gram):
                        return False
                elif p_code in ("adj", "adv"):
                    if has_n_gram or has_v_gram or starts_v_def or is_reflexive_vp:
                        return False
                return True

            locked_def = (definition or (cls.format_ldoce_definition(locked_sense) if locked_sense else "")).strip()
            # When thesaurus contains multiple section_index clusters (e.g. 'save' vs 'spend' under economize),
            # isolate the single section whose polarity and concept align with the locked sense.
            pos_matched_thes = [t for t in thes_list if _thes_item_matches_pos(t)]
            sec_indices = sorted({int(t.get("section_index", 0) or 0) for t in pos_matched_thes})
            if len(sec_indices) > 1:
                _downward_mods = {"less", "fewer", "reduce", "reducing", "save", "saving", "not", "never", "avoid", "stop", "without"}
                best_sec_idx = sec_indices[0]
                best_sec_score = -999.0
                sec_scores = {}
                for _sec_i in sec_indices:
                    _sec_items = [t for t in pos_matched_thes if int(t.get("section_index", 0) or 0) == _sec_i]
                    if not _sec_items:
                        continue
                    _sec_head = str(_sec_items[0].get("section_headword") or "").strip().lower()
                    _sec_concept = str(_sec_items[0].get("concept") or "").strip().lower()
                    _sec_self = next((t for t in _sec_items if cls._thesaurus_variant_matches(t.get("word"), w_clean)), None)
                    _sec_score = 0.15 if _sec_i == 0 else 0.0
                    if cls._thesaurus_variant_matches(_sec_head, w_clean):
                        _sec_score += 2.0
                    if _sec_self and locked_def:
                        _s_dist = re.sub(r"^[:\s]+", "", str(_sec_self.get("distinction") or "")).strip()
                        _sec_score += cls._definition_overlap_ratio(_s_dist, locked_def)
                        # Detect contrastive/opposite inclusion (e.g. 'economize: to spend less money' inside 'spend' section)
                        _s_dist_toks = set(re.findall(r"\b[a-z]+\b", _s_dist.lower()))
                        _concept_body = _sec_concept.split(":", 1)[-1] if ":" in _sec_concept else _sec_concept
                        _concept_toks = set(re.findall(r"\b[a-z]+\b", _concept_body))
                        if _sec_head and _sec_head in _s_dist_toks and (_s_dist_toks & _downward_mods) and not (_concept_toks & _downward_mods):
                            _sec_score -= 2.0
                    if _sec_concept and locked_def:
                        _sec_score += 0.5 * cls._definition_overlap_ratio(_sec_concept, locked_def)
                    sec_scores[_sec_i] = _sec_score
                    if _sec_score > best_sec_score:
                        best_sec_score = _sec_score
                        best_sec_idx = _sec_i
                if any(s < 0 for s in sec_scores.values()):
                    thes_list = [t for t in thes_list if int(t.get("section_index", 0) or 0) == best_sec_idx]

            self_indices = [
                idx_t for idx_t, t in enumerate(thes_list)
                if cls._thesaurus_variant_matches(t.get("word"), w_clean) and _thes_item_matches_pos(t)
            ]
            self_exp = thes_list[self_indices[0]] if self_indices else None
            # Only surface THESAURUS box when its cluster aligns with the locked sense (or entry has 1 POS)
            box_matches_sense = True
            if is_multi_pos and not self_indices:
                # Verify that at least one peer matches the requested POS
                if not any(_thes_item_matches_pos(t) for t in thes_list):
                    box_matches_sense = False
            if locked_def and len(senses) > 1:
                if self_exp:
                    self_dist = re.sub(r"^[:\s]+", "", str(self_exp.get("distinction") or "")).strip()
                    if self_dist and cls._definition_overlap_ratio(self_dist, locked_def) < 0.25:
                        box_matches_sense = False
                thes_blob = " ".join(
                    re.sub(r"^[:\s]+", "", str(t.get("distinction") or "")).strip()
                    for t in thes_list if _thes_item_matches_pos(t)
                ).strip()
                if thes_blob:
                    locked_ov = cls._definition_overlap_ratio(thes_blob, locked_def)
                    if not self_exp and locked_ov < 0.20:
                        box_matches_sense = False
                    elif t_s_idx is not None:
                        for s_i_o, s_other in enumerate(senses):
                            if s_i_o == t_s_idx or not _sense_pos_matches(s_other):
                                continue
                            o_def = (cls.format_ldoce_definition(s_other) or "").strip()
                            if o_def and cls._definition_overlap_ratio(thes_blob, o_def) > locked_ov + 0.08:
                                box_matches_sense = False
                                break
            if box_matches_sense:
                # Reference text combines the locked definition and the target's own thesaurus distinction(s)
                ref_texts = [locked_def] if locked_def else []
                for s_idx in self_indices:
                    s_dist = re.sub(r"^[:\s]+", "", str(thes_list[s_idx].get("distinction") or "")).strip()
                    if s_dist:
                        ref_texts.append(s_dist)
                ref_combined = " ".join(ref_texts)

                ld_low = locked_def.lower()
                is_nonhuman_object_def = (
                    ("for something" in ld_low or "in something" in ld_low or "about something" in ld_low)
                    and not any(k in ld_low for k in ("someone", "somebody", "person", "people", "sexual", "romantic"))
                )

                scored_peers: List[Tuple[float, int, str]] = []
                seen_peer_words: Set[str] = set()
                for idx_t, t in enumerate(thes_list):
                    if not _thes_item_matches_pos(t):
                        continue
                    ew = str(t.get("word") or "").strip()
                    ed = re.sub(r"^[:\s]+", "", str(t.get("distinction") or "")).strip()
                    eg = str(t.get("grammar") or "").strip()
                    er = str(t.get("register") or "").strip()
                    if not ew or not ed:
                        continue
                    # Never list the target headword itself in Thesaurus Note (already defined above)
                    if cls._thesaurus_variant_matches(ew, w_clean) or ew.lower() == w_clean:
                        continue
                    if is_nonhuman_object_def:
                        ed_low = ed.lower()
                        if any(k in ed_low for k in ("sexually attracted", "wife, husband", "boyfriend", "girlfriend", "romantic")):
                            continue
                    ew_key = ew.lower()
                    if ew_key in seen_peer_words:
                        continue
                    seen_peer_words.add(ew_key)

                    # Score peer by lexical overlap with target definition/distinction + box proximity to self_indices
                    overlap_score = cls._definition_overlap_ratio(ed, ref_combined) if ref_combined else 0.0
                    prox_bonus = 0.0
                    if self_indices:
                        min_dist = min(abs(idx_t - s_i) for s_i in self_indices)
                        if min_dist == 1:
                            prox_bonus = 0.12
                        elif min_dist == 2:
                            prox_bonus = 0.05
                    total_score = overlap_score + prox_bonus

                    badges: List[str] = []
                    if eg:
                        badges.append(eg if eg.startswith("[") else f"[{eg}]")
                    if er:
                        badges.append(er if er.startswith("[") else f"[{er}]")
                    badge_str = f" `{' '.join(badges)}`" if badges else ""
                    chunk_str = f"    **{ew}**{badge_str} {ed}"
                    scored_peers.append((total_score, -idx_t, chunk_str))

                scored_peers.sort(key=lambda item: (item[0], item[1]), reverse=True)
                contrast_chunks = [item[2] for item in scored_peers[:3]]
                if contrast_chunks:
                    out["thesaurus_note"] = "\n" + "\n".join(contrast_chunks)

        return out

    @classmethod
    def _semantic_discriminator_note(
        cls,
        target_word: str,
        target_definition: Optional[str],
        dist_list: List[str],
        dist_meta: Optional[Dict[str, str]] = None,
        pos: Optional[str] = None,
        sense_index: Optional[int] = None,
        quote: Optional[str] = None
    ) -> str:
        """
        Target-anchored semantic & grammatical discriminator (+ <GRAM>/Grammar-Box Slot Guard).

        Combines:
        1. Explicit `<gram>` / `grammar_boxes` syntactic slot clashes (`syntactic_slot_clash:*`,
           `grambox_peer:*`, `[only before noun]`, `[not before noun]`, `[uncountable]`, `forbidden_frames`).
        2. Sense-locked `grammar_boxes` usage rules (`✗ Don't say: ...`).
        3. PedLex `THESAURUS` exponent distinctions (including exponent `[grammar]` tags).
        """
        tw = (target_word or "").strip().lower()
        dists = [d for d in (dist_list or []) if d]
        if not tw or not dists:
            return ""

        meta = dist_meta or {}
        notes: List[str] = []

        # 1. Check explicit grammatical/syntactic slot clashes & grammar_boxes peer rules first
        thes_items = cls.get_ldoce_thesaurus(tw)
        for dist in dists:
            d_low = dist.strip().lower()
            mechanism = meta.get(dist) or meta.get(d_low) or ""
            # Check THESAURUS grammar badge on distractor
            th_gram = ""
            th_dist = ""
            for th in thes_items:
                if cls._thesaurus_variant_matches(th.get("word"), dist):
                    th_gram = (th.get("grammar") or "").strip()
                    th_dist = (th.get("distinction") or "").strip()
                    break

            if mechanism.startswith("syntactic_slot_clash:") or mechanism.startswith("grambox_peer:"):
                reason_code = mechanism.split(":", 1)[1]
                if "only_before_noun" in reason_code:
                    notes.append(
                        f"Contrast: vs '{dist}' (`[only before noun]` — grammatically barred from predicative slot after a linking verb)."
                    )
                elif "not_before_noun" in reason_code:
                    notes.append(
                        f"Contrast: vs '{dist}' (`[not before noun]` — grammatically barred from attributive pre-noun slot)."
                    )
                elif "uncountable" in reason_code:
                    notes.append(
                        f"Contrast: vs '{dist}' (`[uncountable]` — cannot take countable determiners or plural form)."
                    )
                elif "forbidden_frame" in reason_code:
                    frame_lbl = reason_code.replace("forbidden_frame_", "")
                    d_alert = cls.get_ldoce_grammar_alert(d_low)
                    bad_ex = ""
                    if d_alert and d_alert.get("bad_examples"):
                        bad_ex = f", ✗ Don't say: {d_alert['bad_examples'][0]}"
                    notes.append(
                        f"Contrast: vs '{dist}' (Grammar Box: barred from '{frame_lbl}' complement frame{bad_ex})."
                    )
                elif "no_passive" in reason_code or "intransitive" in reason_code:
                    notes.append(
                        f"Contrast: vs '{dist}' (intransitive / no-passive constraint — cannot occupy this syntactic voice/object slot)."
                    )
                else:
                    notes.append(
                        f"Contrast: vs '{dist}' (Grammar Box contrast: {reason_code})."
                    )
                break

            # Also check if distractor carries a strict positional <gram> tag that contrasts with target
            d_prof = cls.get_word_gram_profile(d_low, pos=pos)
            t_prof = cls.get_word_gram_profile(tw, pos=pos)
            if (d_prof.get("only_before_noun") or "only before noun" in th_gram.lower()) and not t_prof.get("only_before_noun"):
                nuance_tail = f"; nuance: {th_dist}" if th_dist else ""
                notes.append(
                    f"Contrast: vs '{dist}' (`[only before noun]` constraint{nuance_tail})."
                )
                break
            if (d_prof.get("not_before_noun") or "not before noun" in th_gram.lower()) and not t_prof.get("not_before_noun"):
                nuance_tail = f"; nuance: {th_dist}" if th_dist else ""
                notes.append(
                    f"Contrast: vs '{dist}' (`[not before noun]` predicative-only constraint{nuance_tail})."
                )
                break

        # 2. Standard THESAURUS / antonym discriminator if no grammatical clash note fired yet
        if not notes:
            cluster_nuance = ""
            for th in thes_items:
                if cls._thesaurus_variant_matches(th.get("word"), tw):
                    cluster_nuance = (th.get("distinction") or "").strip()
                    break
            locked = (target_definition or "").strip()
            target_side = locked
            if cluster_nuance and (not locked or
                                   cls._definition_overlap_ratio(cluster_nuance, locked) >= 0.7):
                target_side = cluster_nuance
            if target_side:
                for dist in dists:
                    dist_note = ""
                    dist_gram = ""
                    for th in thes_items:
                        if cls._thesaurus_variant_matches(th.get("word"), dist):
                            dist_note = (th.get("distinction") or "").strip()
                            dist_gram = (th.get("grammar") or "").strip()
                            break
                    mechanism = meta.get(dist) or meta.get(dist.lower()) or ""
                    if not dist_note:
                        if mechanism == "antonym":
                            notes.append(f"Contrast: vs '{dist}' (polar opposite of target meaning).")
                            break
                        if mechanism == "orthographic_paronym":
                            notes.append(f"Contrast: vs '{dist}' (orthographic look-alike / paronym with distinct meaning).")
                            break
                        continue
                    if cls._definition_overlap_ratio(dist_note, target_side) >= 0.8 and not dist_gram:
                        continue  # no real contrast in this cluster pairing
                    if dist_gram:
                        hinge = f"polar opposite; `{dist_gram}`" if mechanism == "antonym" else f"`{dist_gram}`"
                    else:
                        hinge = "polar opposite" if mechanism == "antonym" else "different degree/scale or collocational frame"
                    notes.append(f"Contrast: vs '{dist}' ({hinge}).")
                    break
            else:
                for dist in dists:
                    mechanism = meta.get(dist) or meta.get(dist.lower()) or ""
                    if mechanism == "orthographic_paronym":
                        notes.append(f"Contrast: vs '{dist}' (orthographic look-alike / paronym with distinct meaning).")
                        break

        # 3. Target-side sense-locked Grammar Box rule (if any)
        gb_alert = cls.get_ldoce_grammar_alert(tw, quote=quote or "", sense_index=sense_index, pos=pos or "")
        if gb_alert:
            rules = []
            for r in (gb_alert.get("rules") or []):
                r_s = str(r.get("text") if isinstance(r, dict) else r or "").strip()
                r_s = re.sub(r"\s*Grammar guide\s*[‒–—-]\s*[A-Z\s]+$", "", r_s).strip()
                if r_s:
                    rules.append(r_s)
            bad_exs = [str(b).strip() for b in (gb_alert.get("bad_examples") or []) if str(b).strip()]
            if rules:
                r_short = rules[0]
                if bad_exs and "don't say" not in r_short.lower() and bad_exs[0].lower() not in r_short.lower():
                    r_short += f" (✗ Don't say: {bad_exs[0]})"
                notes.append(f"Grammar Box: {r_short}")

        return " ".join(notes)

    @classmethod
    def _batch_safe_reanchor(
        cls,
        target_word: str,
        canonical_pos: str,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        forbidden: Optional[Set[str]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Re-picks a collocational anchor drawn strictly from OUTSIDE the batch target set.
        Returns (anchor, anchor_type), or (None, None) when nothing outside the batch fits,
        which lets the micro-task planner degrade to its neutral semantic-field template
        rather than welding the anchor-presence gate and the cross-target leakage gate
        into an unsatisfiable deadlock.
        """
        tw = (target_word or "").lower().strip()
        if not tw or not forbidden:
            return None, None

        entry_pos = {"verb": "verb", "noun": "noun", "adj": "adj"}.get(canonical_pos)
        entry = cls.get_pedlex_collocations(tw, pos=entry_pos) if entry_pos else {}

        candidates: List[str] = []
        if entry_pos == "verb":
            candidates = entry.get("colloc_nouns", []) + entry.get("noun_after", [])
        elif entry_pos == "adj":
            candidates = entry.get("noun_after", []) + entry.get("colloc_nouns", [])
        # Noun micro-tasks only consume a bound preposition; an adjective or governing
        # verb anchor falls through to the neutral template regardless, so skip them.

        anchor = cls._select_sense_aligned_anchor(
            candidates,
            definition=definition,
            quote=quote,
            target_word=tw,
            forbidden=forbidden
        )
        if anchor:
            return anchor, ("object" if entry_pos == "verb" else "modified_noun")

        p_cand = cls._pick_anchor_token(entry.get("prep", []), target_word=tw, forbidden=forbidden)
        if p_cand:
            return p_cand, "prep"
        return None, None

