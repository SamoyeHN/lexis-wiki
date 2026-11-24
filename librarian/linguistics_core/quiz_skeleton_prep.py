"""
Vocabulary target preprocessing and multi-word construction decomposition.
Handles Family 1A, Family 1B, Family 1C, Family 2, Family 6, discontinuous frames,
and sense locking.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from .quiz_skeleton_context import SkeletonContext


class QuizSkeletonPrepMixin:
    @classmethod
    def _preprocess_skeleton_target(
        cls,
        it: Dict[str, Any],
        single_word_targets: Set[str],
        batch_target_words: Set[str],
        nlp: Any,
        has_multiple_candidates: bool = True,
        passage_level: Optional[str] = None,
    ) -> Optional[SkeletonContext]:
        """
        Stage 1: Preprocess vocabulary target item, decompose multi-word constructions,
        and lock authoritative pedagogical sense before anchor resolution.
        """
        # Resolve passage CEFR level from item metadata, explicit argument, or active ContextVar
        passage_cefr = (
            it.get("passage_cefr")
            or it.get("overall_cefr_level")
            or it.get("cefr_level")
            or passage_level
            or cls.get_current_passage_cefr()
        )
        if passage_cefr:
            passage_cefr = str(passage_cefr).strip().upper()

        anchor = None
        anchor_type = None
        anchor_source = "curriculum_collocation"
        w = it["word"]
        raw_w_item = it.get("raw_word") or w
        w_lower = w.lower()
        raw_pos = it["part_of_speech"].lower()

        # Gate: Drop proper nouns / brand names (e.g. 'coca-cola') when attested in context as capitalized PROPN+NER
        _ctx_for_propn = str(it.get("quote") or it.get("raw_quote") or it.get("example") or "").strip()
        if nlp and _ctx_for_propn and " " not in w_lower:
            _w_parts = [p for p in re.split(r"[-\s]+", w_lower) if p]
            _ctx_doc = nlp(_ctx_for_propn)
            _matched_propn = [t for t in _ctx_doc if t.text.lower() in _w_parts]
            if _matched_propn and all(
                t.text[:1].isupper()
                and t.pos_ == "PROPN"
                and t.ent_type_ in ("ORG", "PERSON", "GPE", "LOC", "PRODUCT", "WORK_OF_ART", "NORP", "EVENT")
                for t in _matched_propn
            ):
                logging.getLogger(__name__).info("[DROP_PROPER_NOUN] Dropping proper noun / brand target '%s'", w)
                return None

        # Discontinuous Verb + [sb/sth] + Prep Frame Resolution (e.g. "remind [sb] of [sb] / [sth]"):
        # Stripping internal object slots leaves an ungrammatical adjacent string ('remind of') that cannot
        # fill a single blank in English ('*reminds of me'). Decompose into head verb ('remind') + bound
        # preposition anchor ('of') so the blank tests the head verb in '... ____ [sb/sth] of ...'.
        discontinuous_prep = None
        causative_extraposition_pat = None
        causative_extraposition_ex = None
        correlative_second_arm = None
        correlative_pat = None
        correlative_frame = None
        lv_triad_prep = None
        lv_triad_pat = None
        lv_triad_frame = None
        cxn_override_pat = None
        cxn_override_frame = None

        _ditrans_prep_verbs = {
            "remind", "convince", "inform", "warn", "assure", "rob", "deprive", "rid", "cure",
            "prevent", "stop", "prohibit", "discourage", "deter", "accuse", "suspect",
            "congratulate", "blame", "thank", "forgive", "talk", "trick", "deceive", "fool",
            "equate", "compare", "liken", "confuse", "conflate", "associate", "connect",
            "link", "relate", "attribute", "ascribe", "credit", "expose", "subject",
            "confine", "restrict", "limit", "reduce", "convert", "transform", "translate",
            "integrate", "incorporate", "merge", "blend", "combine", "adapt", "adjust",
            "tailor", "accustom", "reconcile", "resign", "dedicate", "devote", "commit",
            "apply", "attach", "fasten", "tie", "bind", "fix", "add", "append",
            "define", "describe", "regard", "view", "see", "treat", "use", "hail",
            "dismiss", "denounce", "condemn", "portray", "depict", "characterize",
            "classify", "categorize", "label", "brand", "cast", "cite", "quote",
            "base", "model", "ground", "center", "centre", "focus", "concentrate",
            "hinge", "depend", "rely", "count", "bank", "pride", "avail", "endow",
            "inflict", "impose", "lavish", "bestow", "confer",
        }
        orig_disc_prep = None
        v_obj_pp_tail = None
        m_v_obj_pp = re.match(
            r'^([a-z]+)\s+(?:\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+((?:of|to|with|from|into|for|on|in|about|against|at|upon|over|as|under|out\s+of|beyond|within)\s+(?!sb\b|sth\b|somebody\b|something\b)[a-z]+(?:\s+[a-z]+)?)$',
            raw_w_item.strip(),
            re.IGNORECASE,
        )
        if m_v_obj_pp:
            head_v_pp = m_v_obj_pp.group(1).lower()
            if head_v_pp not in single_word_targets:
                w = head_v_pp
                w_lower = head_v_pp
                raw_pos = "verb"
                v_obj_pp_tail = re.sub(r"\s+", " ", m_v_obj_pp.group(2).lower()).strip()
                cxn_override_pat = f"{head_v_pp} [sth] {v_obj_pp_tail}"
                cxn_override_frame = f"... ____ [sth] {v_obj_pp_tail}"
                cur_v_def = str(it.get("definition") or "").strip()
                if cur_v_def and v_obj_pp_tail not in cur_v_def.lower():
                    it["definition"] = f"{cur_v_def} (in the phrase '{head_v_pp} something {v_obj_pp_tail}')"

        m_disc = None
        if not v_obj_pp_tail:
            m_disc = re.match(
                r'^([a-z]+)\s+(?:\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+(of|to|with|from|into|for|on|in|about|against|at|upon|over|as)\s+(?:\[[^\]]+\]|\([^)]+\)|\b(?:sb|sth|somebody|something)\b)\s*$',
                raw_w_item.strip(),
                re.IGNORECASE
            )
            if not m_disc:
                m_disc_cand = re.match(
                    r'^([a-z]+)\s+(?:\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+(of|to|with|from|into|for|on|in|about|against|at|upon|over|as)\s*$',
                    raw_w_item.strip(),
                    re.IGNORECASE
                )
                if m_disc_cand and (
                    m_disc_cand.group(1).lower() in _ditrans_prep_verbs
                    or m_disc_cand.group(2).lower() in ("of", "to", "with", "from", "into", "for", "about", "against", "at", "upon", "as")
                ):
                    m_disc = m_disc_cand
            if not m_disc and str(raw_pos or "").strip().lower() != "phrasal verb" and len(w_lower.split()) == 2:
                w_p0, w_p1 = w_lower.split()
                _contiguous_intrans_prep_verbs = {
                    "talk", "speak", "think", "dream", "hear", "know", "complain", "boast",
                    "depend", "rely", "count", "bank", "hinge", "focus", "concentrate",
                    "insist", "elaborate", "embark", "impinge", "infringe", "intrude",
                }
                _is_contiguous_raw = bool(
                    re.search(rf"\b{re.escape(w_p0)}\s+{re.escape(w_p1)}\b", raw_w_item.lower())
                )
                if (
                    w_p1 in ("of", "from", "into", "for", "on", "to", "with")
                    and w_p0 in _ditrans_prep_verbs
                    and not (_is_contiguous_raw and w_p0 in _contiguous_intrans_prep_verbs)
                ):
                    m_disc = re.match(r'^([a-z]+)\s+([a-z]+)$', w_lower)
        if m_disc:
            head_v = m_disc.group(1).lower()
            disc_p = m_disc.group(2).lower()
            # Only decompose if head_v is not already a standalone single-word target in the batch
            if head_v not in single_word_targets:
                w = head_v
                w_lower = head_v
                discontinuous_prep = disc_p
                orig_disc_prep = disc_p
                raw_pos = "verb"

        # Family 1B: Catenative Verb + `to do (sth)` Placeholder Decomposition
        # (e.g. 'tend to do (sth)' where 'do (sth)' is a dictionary placeholder for '<infinitive verb>'
        # and the passage quote uses 'tend to think', not literal 'tend to do')
        m_cat_to_do = re.match(
            r"^([a-z]+)\s+to\s+do(?:\s+[\(\[][^\)\]]+[\)\]])?$",
            raw_w_item.strip(),
            re.IGNORECASE,
        )
        if m_cat_to_do and "to do" not in (it.get("quote") or "").lower():
            cat_v = m_cat_to_do.group(1).lower()
            if cat_v not in single_word_targets:
                w = cat_v
                w_lower = cat_v
                raw_pos = "verb"

        # Family 1C: Copular Adjective Phrase Decomposition (e.g. "be full of (sth)", "be short of (sth)", "be proud of", "be fond of")
        # In dictionaries/syllabi, adjective complementation is often recorded with a pedagogical placeholder 'be'
        # (e.g. 'be full of (sth)'). In authentic English usage and psychometric assessment, the copula is finite
        # ('was full of', 'are short of') and the adjective phrase ('full of', 'short of') occupies the predicate.
        # Decouple 'be + <Adj> + <Prep>' into '<Adj> + <Prep>' as a predicate adjective phrase, preserving
        # the finite copula in the stem's cloze slot frame ('... was/is ____ [something]') and generating
        # syntactically parallel adjective phrase distractors (e.g. 'short of', 'devoid of', 'aware of').
        # Guard: Distinguish from 3-word passive verbal constructions ('be equipped with', 'be based on', 'be associated with')
        # where the head word is a passive participle verb.
        m_cop_adj_prep = re.match(
            r"^be\s+([a-z]+)\s+([a-z]+)(?:\s+[\(\[][^\)\]]+[\)\]])?$",
            raw_w_item.strip(),
            re.IGNORECASE,
        )
        if not m_cop_adj_prep and re.match(r"^be\s+([a-z]+)\s+([a-z]+)$", w_lower):
            m_cop_adj_prep = re.match(r"^be\s+([a-z]+)\s+([a-z]+)$", w_lower)
        if m_cop_adj_prep:
            cand_adj_w = m_cop_adj_prep.group(1).lower()
            cand_prep_w = m_cop_adj_prep.group(2).lower()
            cand_adj_entry = cls.get_ldoce_entry(cand_adj_w)
            cand_pos = str((cand_adj_entry or {}).get("pos") or "").lower()
            is_true_adj = not (cand_adj_w.endswith("ed") or cand_adj_w.endswith("en")) and (
                cand_pos == "adjective" or bool(cls._wordnet_pos(cand_adj_w) & {"a", "s"})
            )
            if is_true_adj:
                w = f"{cand_adj_w} {cand_prep_w}"
                w_lower = w
                raw_pos = "phrase"

        # Family 1A: Discontinuous Causative Verb + [sth/it] + Adjective Extraposition Frame Resolution
        # (e.g. "make [sth] possible", "find [sth] easier", "keep it simple", "leave it clear"):
        # Stripping '[sth]' into bare 2-word 'make possible' produces an ungrammatical contiguous string ('*make possible it').
        # Transform into the single-blank preparatory-object extraposition construction '(make|find|keep|leave) it <adj>'
        # with frame '<verb> it <adj> ... to do [sth]'.
        _comp_adj_to_base = {
            "easier": "easy",
            "harder": "hard",
            "clearer": "clear",
            "simpler": "simple",
            "safer": "safe",
        }
        m_caus_adj = re.match(
            r'^(make|find|keep|leave)\s+(?:it|\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+(possible|impossible|clear|clearer|easy|easier|difficult|hard|harder|certain|plain|simple|simpler|safe|safer|worthwhile|necessary|unlikely|likely)$',
            raw_w_item.strip(),
            re.IGNORECASE
        )
        if not m_caus_adj and re.match(
            r'^(make|find|keep|leave)\s+(?:it\s+)?(possible|impossible|clear|clearer|easy|easier|difficult|hard|harder|certain|plain|simple|simpler|safe|safer)$',
            w_lower,
        ):
            m_caus_adj = re.match(r'^(make|find|keep|leave)\s+(?:it\s+)?([a-z]+)$', w_lower)
        if m_caus_adj:
            caus_v = m_caus_adj.group(1).lower()
            raw_adj_w = m_caus_adj.group(2).lower()
            adj_w = _comp_adj_to_base.get(raw_adj_w, raw_adj_w)
            w = f"{caus_v} it {adj_w}"
            w_lower = w
            raw_pos = "collocation"
            causative_extraposition_pat = f"{caus_v} it {adj_w} ... to do [sth]"
            adj_entry = cls.get_ldoce_entry(adj_w)
            if adj_entry and adj_entry.get("senses"):
                for s_cand in adj_entry["senses"]:
                    if not isinstance(s_cand, dict):
                        continue
                    for pb in (s_cand.get("pattern_bundles") or []):
                        if not isinstance(pb, dict):
                            continue
                        pb_pat = str(pb.get("pattern") or "").lower()
                        if re.search(rf"\b{re.escape(caus_v)}s?\s+it\s+{re.escape(adj_w)}\b.*\bto\b", pb_pat):
                            raw_s_def = cls.format_ldoce_definition(s_cand)
                            comp_def = cls._compose_collocation_definition(
                                f"{caus_v} {adj_w}", "collocation", adj_w, s_cand, raw_s_def, it.get("quote")
                            )
                            if comp_def:
                                it["definition"] = comp_def
                            causative_extraposition_ex = str(pb.get("example") or "").strip() or None
                            break
                    if causative_extraposition_ex:
                        break

        # Family 2: Discontinuous Paired Correlatives ('not only ... but (also)', 'neither ... nor', 'no sooner ... than', etc.)
        # Decompose discontinuous correlative headwords into the first arm as the single-blank target while locking the second arm
        # in the stem's Cloze Slot Frame ('... ____ ... but also ...').
        _correl_decomp_map = [
            (r"^not\s+only\b.*\bbut(?:\s+also)?\b", "not only", "but also", "not only ... but also ...", "... ____ ... but also ..."),
            (r"^neither\b.*\bnor\b", "neither", "nor", "neither ... nor ...", "... ____ ... nor ..."),
            (r"^either\b.*\bor\b", "either", "or", "either ... or ...", "... ____ ... or ..."),
            (r"^whether\b.*\bor\b", "whether", "or", "whether ... or ...", "... ____ ... or ..."),
            (r"^no\s+sooner\b.*\bthan\b", "no sooner", "than", "no sooner ... than ...", "... ____ ... than ..."),
            (r"^hardly\b.*\bwhen\b", "hardly", "when", "hardly ... when ...", "... ____ ... when ..."),
            (r"^scarcely\b.*\bwhen\b", "scarcely", "when", "scarcely ... when ...", "... ____ ... when ..."),
            (r"^would\s+rather\b.*\bthan\b", "would rather", "than", "would rather ... than ...", "... ____ ... than ..."),
        ]
        for c_rx, first_arm, sec_arm, c_pat, c_frame in _correl_decomp_map:
            if re.search(c_rx, raw_w_item.strip(), re.IGNORECASE) or re.search(c_rx, w_lower, re.IGNORECASE):
                w = first_arm
                w_lower = first_arm
                raw_pos = "phrase"
                correlative_second_arm = sec_arm
                correlative_pat = c_pat
                correlative_frame = c_frame
                break
        if not correlative_frame and w_lower in ("not only", "neither", "either", "no sooner", "hardly", "scarcely", "would rather") and it.get("quote"):
            q_low_corr = it["quote"].lower()
            _quote_correl_map = {
                "not only": ("but", "not only ... but also ...", "... ____ ... but also ..."),
                "neither": ("nor", "neither ... nor ...", "... ____ ... nor ..."),
                "either": ("or", "either ... or ...", "... ____ ... or ..."),
                "no sooner": ("than", "no sooner ... than ...", "... ____ ... than ..."),
                "hardly": ("when", "hardly ... when ...", "... ____ ... when ..."),
                "scarcely": ("when", "scarcely ... when ...", "... ____ ... when ..."),
                "would rather": ("than", "would rather ... than ...", "... ____ ... than ..."),
            }
            sec_arm, c_pat, c_frame = _quote_correl_map[w_lower]
            if re.search(rf"\b{re.escape(w_lower)}\b[^.?!;]*\b{re.escape(sec_arm)}\b", q_low_corr):
                raw_pos = "phrase"
                correlative_second_arm = sec_arm
                correlative_pat = c_pat
                correlative_frame = c_frame

        # Family 6: Light-Verb + (Det/Adj) + Noun + Prep Triad Internal-Modifier Decomposition
        # When the passage quote inserts an internal adjective modifier before the triad head noun
        # (e.g. 'pay close attention to', 'have a profound impact on', 'play a vital role in'),
        # a contiguous 3-word phrase option cannot fit without colliding with or dropping the modifier.
        # Decompose to the triad head noun ('attention', 'impact', 'role') with locked triad frame.
        _lv_triads = {
            "pay attention to": ("pay", "attention", "to"),
            "have an impact on": ("have", "impact", "on"),
            "have an effect on": ("have", "effect", "on"),
            "play a role in": ("play", "role", "in"),
            "play a part in": ("play", "part", "in"),
            "take advantage of": ("take", "advantage", "of"),
            "make use of": ("make", "use", "of"),
        }
        if w_lower in _lv_triads and it.get("quote"):
            lv_v, lv_n, lv_p = _lv_triads[w_lower]
            m_lv_mod = re.search(
                rf"\b({re.escape(lv_v)}(?:s|es|ed|d|ing)?|had|has|having|paid|paying|played|playing|took|taken|taking|made|making)\s+(?:(a|an|the)\s+)?([a-z]+)\s+{re.escape(lv_n)}\s+{re.escape(lv_p)}\b",
                it["quote"],
                re.IGNORECASE,
            )
            if m_lv_mod:
                det_part = (m_lv_mod.group(2) or "").lower()
                adj_mod = m_lv_mod.group(3).lower()
                if adj_mod not in ("a", "an", "the", "no", "any", "some", "much", "more", "little", "less") and lv_n not in single_word_targets:
                    w = lv_n
                    w_lower = lv_n
                    raw_pos = "noun"
                    lv_triad_prep = lv_p
                    prefix_str = f"{lv_v} {det_part} [{adj_mod}]".replace("  ", " ") if det_part else f"{lv_v} [{adj_mod}]"
                    frame_prefix = f"{lv_v} {det_part} {adj_mod}".replace("  ", " ") if det_part else f"{lv_v} {adj_mod}"
                    lv_triad_pat = f"{prefix_str} {lv_n} {lv_p} [sth]"
                    lv_triad_frame = f"... {frame_prefix} ____ {lv_p} [sth]"

        # Prepositional Phrase with Obligatory Internal Modifier Slot (e.g. `on a (sth) scale` -> `on a large scale`):
        # Stripping `(sth)` leaves an ungrammatical bare singular count noun (`*on a scale`).
        # Realize the authentic 4-word phrase from the passage quote or dictionary example usage.
        _m_prep_mod_slot = re.match(
            r"^((?:on|in|at|to|by|under|with|from)\s+(?:a|an|the))\s+(?:\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+([a-z]+)$",
            raw_w_item,
            re.IGNORECASE,
        )
        if _m_prep_mod_slot:
            _pm_prefix = _m_prep_mod_slot.group(1).lower()
            _pm_prep = _pm_prefix.split()[0]
            _pm_tail = _m_prep_mod_slot.group(2).lower()
            for _src_txt in (it.get("quote") or "", it.get("example_usage") or ""):
                if not _src_txt:
                    continue
                _m_realized = re.search(
                    rf"\b({re.escape(_pm_prep)}\s+(?:a|an)\s+[a-z]+\s+{re.escape(_pm_tail)})\b",
                    _src_txt,
                    re.IGNORECASE,
                )
                if _m_realized:
                    w = _m_realized.group(1).lower()
                    w_lower = w
                    break

        # Map POS category
        # Idiomatic Slot Gate: If single word is embedded in a fixed idiomatic construction in authentic quote
        # (e.g. 'granted' in 'take ... for granted'), prioritize the authentic idiomatic slot over isolated POS/function-word
        is_idiom_slot = False
        if w_lower == "granted" and it.get("quote") and re.search(r'\btake\b(?:\W+\w+){0,6}\W+for\W+granted\b', it["quote"], re.IGNORECASE):
            is_idiom_slot = True
            canonical_pos = "idiom_slot"
            anchor = "take for"
            anchor_type = "idiom"

        if not is_idiom_slot:
            if discontinuous_prep:
                canonical_pos = "verb"
            elif correlative_frame:
                canonical_pos = "phrase"
            elif lv_triad_frame:
                canonical_pos = "noun"
            else:
                resolved = cls.resolve_item_pos(w_lower, it.get("quote"), manual_pos=raw_pos)
                if resolved in ("phrase", "collocation", "phrasal verb", "set phrase", "idiom"):
                    canonical_pos = "phrase"
                elif resolved == "function_word" or cls.is_function_word(w_lower):
                    canonical_pos = "function_word"
                elif resolved == "verb":
                    canonical_pos = "verb"
                elif resolved in ("adjective", "adj"):
                    canonical_pos = "adj"
                elif resolved in ("adverb", "adv"):
                    canonical_pos = "adv"
                else:
                    canonical_pos = "noun"

        # Evidence-Grade Gate (P2): a curriculum 'Quoted Sentence' licenses an anchor or a
        # valency frame only when it actually contains a clause. Fragments lifted from
        # headings ('Punctuality Pays!') were treated as authentic usage, so the blueprint
        # welded anchors and frames onto a two-word title that never demonstrated them.
        # A weak quote is demoted to display-only evidence: it stays visible to the writer
        # as candidate_quote, but every anchor must then come from PedLex instead.
        quote_provenance, quote_wordcount = cls._quote_evidence_strength(it.get("quote"))
        it["raw_quote"] = it.get("quote") or ""
        if quote_provenance == "weak":
            it["quote"] = ""

        # 0. Sense-First Grounding (Sense Primacy — GEMINI.md §1.10):
        # Lock the authoritative Pedagogical Sense BEFORE anchor selection or distractor generation.
        target_p_code = cls._pos_code(canonical_pos)
        entry_base = cls.get_ldoce_entry(w_lower)
        target_entry = entry_base
        entry_p_code = cls._pos_code((target_entry or {}).get("pos")) if target_entry else None
        if (
            not target_entry
            or not target_entry.get("senses")
            or (target_p_code and entry_p_code and entry_p_code != target_p_code)
        ) and " " not in w_lower:
            lem_w = cls.lemma_of(w_lower)
            if lem_w and lem_w != w_lower and ("-" not in w_lower or "-" in lem_w):
                lem_entry = cls.get_ldoce_entry(lem_w)
                if lem_entry and lem_entry.get("kind") != "derived" and lem_entry.get("senses"):
                    target_entry = lem_entry
        locked_sense = None
        t_s_idx = None
        use_pv_blocks = canonical_pos == "phrase" and (
            not target_entry
            or raw_pos == "phrasal verb"
            or str(target_entry.get("word", "")).strip().lower() != w_lower
        )
        if use_pv_blocks:
            pv_blocks = cls._phrasal_verb_blocks(w_lower)
            pv_senses = []
            if pv_blocks:
                for blk in pv_blocks:
                    blk_hw = str(blk.get("headword") or "").strip()
                    for s_obj in (blk.get("senses") or []):
                        if isinstance(s_obj, dict):
                            s_copy = dict(s_obj)
                            if blk_hw and "phrasal_headword" not in s_copy:
                                s_copy["phrasal_headword"] = blk_hw
                            pv_senses.append(s_copy)
            # Unified Literal + Figurative Phrasal Verb Sense Pool:
            # In LDOCE/PedLex, literal physical senses of separable phrasal verbs (e.g. 'nail something down'
            # under 'nail' Sense 1) are often stored inside the main verb entry's `pattern_bundles`/`patterns`
            # rather than the figurative `phrasal_verbs` sub-block. Fuse matching main-verb senses into `pv_senses`
            # so `_lock_sense` can lock onto the literal physical sense whenever attested by the passage/definition.
            w_parts_pv = w_lower.split()
            is_passive_3w = (
                len(w_parts_pv) == 3
                and w_parts_pv[0] in ("be", "get", "feel")
                and (w_parts_pv[1].endswith("ed") or w_parts_pv[1].endswith("en") or cls.lemma_of(w_parts_pv[1]) != w_parts_pv[1])
            )
            if len(w_parts_pv) == 2 or is_passive_3w:
                if is_passive_3w:
                    v_head_pv, part_w_pv = cls.lemma_of(w_parts_pv[1]), w_parts_pv[2]
                    pv_pat_re = re.compile(
                        rf"\b(?:{re.escape(v_head_pv)}|{re.escape(w_parts_pv[1])})\b(?:\s+(?:something|somebody|sb|sth|oneself|it|them|up|down|out|in|on|off|away|back|/|\s)*)?\b{re.escape(part_w_pv)}\b",
                        re.IGNORECASE,
                    )
                else:
                    v_head_pv, part_w_pv = w_parts_pv[0], w_parts_pv[1]
                    pv_pat_re = re.compile(
                        rf"\b{re.escape(v_head_pv)}\b(?:\s+(?:something|somebody|sb|sth|oneself|it|them|up|down|out|in|on|off|away|back|/|\s)*)?\b{re.escape(part_w_pv)}\b",
                        re.IGNORECASE,
                    )
                base_v_ent = cls.get_ldoce_entry(v_head_pv)
                for cand_ent in ([base_v_ent] if base_v_ent else []):
                    for s_obj in (cand_ent.get("senses") or []):
                        if not isinstance(s_obj, dict):
                            continue
                        if "verb" not in str(s_obj.get("pos") or cand_ent.get("pos") or "").lower():
                            continue
                        s_pats = [str(p) for p in (s_obj.get("patterns") or [])] + [
                            str(pb.get("pattern") or "")
                            for pb in (s_obj.get("pattern_bundles") or [])
                            if isinstance(pb, dict)
                        ]
                        if any(pv_pat_re.search(p_str) for p_str in s_pats):
                            s_copy = dict(s_obj)
                            match_pat = next((p_str for p_str in s_pats if pv_pat_re.search(p_str)), f"{v_head_pv} something {part_w_pv}")
                            s_copy["phrasal_headword"] = match_pat
                            # Filter examples/pattern_bundles on this copy to those attesting the particle if available
                            part_re = re.compile(rf"\b{re.escape(part_w_pv)}\b", re.IGNORECASE)
                            matching_pbs = [
                                pb for pb in (s_obj.get("pattern_bundles") or [])
                                if isinstance(pb, dict) and (
                                    pv_pat_re.search(str(pb.get("pattern") or ""))
                                    or any(part_re.search(str(ex)) for ex in (pb.get("examples") or []))
                                )
                            ]
                            matching_exs = [
                                ex for ex in (s_obj.get("examples") or [])
                                if part_re.search(str(ex))
                            ]
                            for pb in matching_pbs:
                                for ex in (pb.get("examples") or []):
                                    if part_re.search(str(ex)) and ex not in matching_exs:
                                        matching_exs.append(ex)
                            if matching_pbs:
                                s_copy["pattern_bundles"] = matching_pbs
                            if matching_exs:
                                s_copy["examples"] = matching_exs
                            pv_senses.append(s_copy)
            if pv_senses:
                synth_pv_entry = {"word": w_lower, "pos": "phrasal verb", "senses": pv_senses}
                target_entry = synth_pv_entry
                t_s_idx = cls._lock_sense(
                    synth_pv_entry,
                    definition=it.get("definition"),
                    quote=it.get("quote") or it.get("raw_quote"),
                    target_pos="phrase"
                )
                if t_s_idx is not None and 0 <= t_s_idx < len(pv_senses):
                    locked_sense = pv_senses[t_s_idx]
        if locked_sense is None and canonical_pos == "phrase" and len(w_lower.split()) == 2:
            w_p0, w_p1 = w_lower.split()
            p0_entry = cls.get_ldoce_entry(w_p0)
            p0_pos = str((p0_entry or {}).get("pos") or "").lower()
            p0_is_adj = not (w_p0.endswith("ed") or w_p0.endswith("en")) and (
                p0_pos == "adjective" or bool(cls._wordnet_pos(w_p0) & {"a", "s"})
            )
            if p0_is_adj and p0_entry and p0_entry.get("senses"):
                target_entry = p0_entry
                t_s_idx = cls._lock_sense(
                    p0_entry,
                    definition=it.get("definition"),
                    quote=it.get("quote") or it.get("raw_quote"),
                    target_pos="adj"
                )
                if t_s_idx is not None and 0 <= t_s_idx < len(p0_entry.get("senses", [])):
                    locked_sense = p0_entry["senses"][t_s_idx]
        if locked_sense is None and target_entry and target_entry.get("senses"):
            # When decomposed from a discontinuous trivalent frame ('V + (sth/sb) + Prep'),
            # include transitive phrasal_verbs senses (for verbs like 'rid', 'endow', 'deprive')
            # alongside transitive main-entry verb senses, and filter out [intransitive, no passive] senses.
            if discontinuous_prep:
                trans_pool = []
                for s_obj in (target_entry.get("senses") or []):
                    if not isinstance(s_obj, dict):
                        continue
                    s_gram = f"{s_obj.get('grammar') or ''} {s_obj.get('syntax_codes') or ''}".lower()
                    if "intransitive" in s_gram and "transitive" not in s_gram.replace("intransitive", ""):
                        continue
                    trans_pool.append(s_obj)
                for blk in cls._phrasal_verb_blocks(f"{w_lower} {discontinuous_prep}"):
                    blk_hw_l = str(blk.get("headword") or "").lower()
                    for s_obj in (blk.get("senses") or []):
                        if not isinstance(s_obj, dict):
                            continue
                        s_gram = f"{s_obj.get('grammar') or ''} {s_obj.get('syntax_codes') or ''}".lower()
                        if ("intransitive" in s_gram and "transitive" not in s_gram.replace("intransitive", "")) or "no passive" in s_gram:
                            continue
                        if trans_pool and not any(
                            re.search(
                                rf"\b{re.escape(w_lower)}\s+(?:somebody|something|someone|sb|sth)\b.*\b{re.escape(discontinuous_prep)}\b",
                                str(ph).lower(),
                            )
                            for ph in ([blk_hw_l] + list(s_obj.get("patterns") or []))
                        ):
                            continue
                        trans_pool.append(s_obj)
                if trans_pool:
                    synth_trans_entry = dict(target_entry, senses=trans_pool)
                    t_s_idx = cls._lock_sense(
                        synth_trans_entry,
                        definition=it.get("definition"),
                        quote=it.get("quote") or it.get("raw_quote"),
                        target_pos=canonical_pos,
                    )
                    if t_s_idx is not None and 0 <= t_s_idx < len(trans_pool):
                        locked_sense = trans_pool[t_s_idx]
                        # Replace stale intransitive phrasal-verb definition (e.g. 'to be equal to something' for 'equate')
                        locked_def_fmt = cls.format_ldoce_definition(locked_sense)
                        if locked_def_fmt:
                            it["definition"] = locked_def_fmt
            if locked_sense is None:
                t_s_idx = cls._lock_sense(
                    target_entry,
                    definition=it.get("definition"),
                    quote=it.get("quote") or it.get("raw_quote"),
                    target_pos=canonical_pos
                )
                if t_s_idx is None and target_entry is not entry_base:
                    t_s_idx = cls._lock_sense(
                        target_entry,
                        definition=it.get("definition"),
                        quote=it.get("quote") or it.get("raw_quote"),
                        target_pos=None
                    )
                if t_s_idx is not None and 0 <= t_s_idx < len(target_entry["senses"]):
                    locked_sense = target_entry["senses"][t_s_idx]
                    # Guard against stale single-word definitions in cached markdown when the quote
                    # explicitly attests a multi-word sense unit (e.g. 'attaches great importance to' -> 'attach importance/significance etc to something')
                    q_raw_for_unit = (it.get("quote") or it.get("raw_quote") or "").strip()
                    if q_raw_for_unit and not locked_sense.get("units") and any(s_c.get("units") for s_c in target_entry["senses"]):
                        q_only_idx = cls._lock_sense(
                            target_entry,
                            definition=None,
                            quote=q_raw_for_unit,
                            target_pos=canonical_pos,
                        )
                        if q_only_idx is not None and q_only_idx != t_s_idx and 0 <= q_only_idx < len(target_entry["senses"]):
                            cand_u_sense = target_entry["senses"][q_only_idx]
                            for u_str in (cand_u_sense.get("units") or []):
                                u_brs = cls.clean_and_expand_pattern_slashes(str(u_str).lower()) or [str(u_str).lower()]
                                if any(
                                    len([w for w in re.findall(r"[a-z]{2,}", br) if w not in ("something", "somebody", "someone", "sb", "sth", "etc", "one", "ones", "do", "doing") and w != w_lower]) >= 2
                                    and all(
                                        cls.text_contains_form(q_raw_for_unit, w)
                                        for w in re.findall(r"[a-z]{2,}", br)
                                        if w not in ("something", "somebody", "someone", "sb", "sth", "etc", "one", "ones", "do", "doing")
                                    )
                                    for br in u_brs
                                ):
                                    t_s_idx = q_only_idx
                                    locked_sense = cand_u_sense
                                    it["definition"] = cls.format_ldoce_definition(locked_sense)
                                    break
        if locked_sense and not it.get("definition"):
            it["definition"] = cls.format_ldoce_definition(locked_sense)

        # Enrich bare-gloss definitions (e.g. 'luncheon' -> '[formal] lunch') using PedLex Activator/Thesaurus distinction
        raw_def_curr = str(it.get("definition") or "").strip()
        def_body_curr = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", raw_def_curr).strip()
        if def_body_curr and len(def_body_curr.split()) <= 2 and target_entry:
            for la in (target_entry.get("language_activator") or []):
                if isinstance(la, dict):
                    for exp in (la.get("words") or la.get("exponents") or []):
                        if isinstance(exp, dict) and str(exp.get("word", "")).strip().lower() == w_lower:
                            la_def = str(exp.get("definition", "")).strip()
                            if la_def and len(la_def.split()) > 3 and la_def.lower() not in raw_def_curr.lower():
                                it["definition"] = f"{raw_def_curr} ({la_def})"
                                break
                if "(" in str(it.get("definition", "")):
                    break
            if "(" not in str(it.get("definition", "")):
                for th in (target_entry.get("thesaurus") or []):
                    if isinstance(th, dict) and str(th.get("word", "")).strip().lower() == w_lower:
                        if th.get("pos") and canonical_pos in ("noun", "verb", "adj", "adv") and cls._pos_code(th["pos"]) != cls._pos_code(canonical_pos):
                            continue
                        th_def = str(th.get("distinction", "")).strip()
                        if th_def and len(th_def.split()) > 3 and th_def.lower() not in raw_def_curr.lower():
                            it["definition"] = f"{raw_def_curr} ({th_def})"
                            break

        # 0.1 Pre-bind Atomic Sense Pattern Bundle (Pattern + Corpus Example + Bound Preposition)
        # before anchor selection and distractor generation so that sense-native prepositions
        # participate directly in Step 1 anchor resolution and Step 3 double-key clearance.
        ldoce_pattern_example = None
        ldoce_cloze_frame = None
        pattern_anchor = None
        best_pattern = None
        initial_atomic_bundle = cls.pick_atomic_pattern_bundle(
            locked_sense, w_lower, quote=it.get("quote"), preferred_anchor=discontinuous_prep
        )
        if initial_atomic_bundle and initial_atomic_bundle.get("prep"):
            sense_native_prep = initial_atomic_bundle["prep"]
            if discontinuous_prep and discontinuous_prep != sense_native_prep:
                discontinuous_prep = sense_native_prep
        else:
            sense_native_prep = discontinuous_prep

        # 1. Retrieve Authentic Oxford Collocational Anchor (Sense-Guided)
        if not is_idiom_slot:
            anchor = discontinuous_prep
            anchor_type = "prep" if discontinuous_prep else "collocation"
        prep_obj = None
        quote_governed_prep: Optional[str] = discontinuous_prep if discontinuous_prep else None
        quote_is_transitive = False

        if correlative_frame:
            anchor = correlative_second_arm
            anchor_type = "correl_pair"
            cxn_override_pat = correlative_pat
            cxn_override_frame = correlative_frame
        elif lv_triad_frame:
            anchor = lv_triad_prep
            anchor_type = "lv_triad"
            cxn_override_pat = lv_triad_pat
            cxn_override_frame = lv_triad_frame
        elif v_obj_pp_tail:
            anchor = v_obj_pp_tail
            anchor_type = "v_obj_pp"

        word_cxn_specs_raw = cls.get_word_cxn_specs(w_lower, pos=canonical_pos)
        word_cxn_specs = [
            sp for sp in word_cxn_specs_raw
            if int(sp.get("sense_idx", -1)) < 0
            or t_s_idx is None
            or int(sp.get("sense_idx", -1)) == int(t_s_idx)
        ]
        word_subcats = {str(sp.get("subcat_group") or "") for sp in word_cxn_specs}

        target_inflections_set = {w_lower} | set(cls.inflected_forms(w_lower))

        return SkeletonContext(
            it=it,
            w=w,
            raw_w_item=raw_w_item,
            w_lower=w_lower,
            raw_pos=raw_pos,
            canonical_pos=canonical_pos,
            passage_cefr=passage_cefr,
            target_entry=target_entry,
            entry_base=entry_base,
            locked_sense=locked_sense,
            t_s_idx=t_s_idx,
            target_p_code=target_p_code,
            quote_provenance=quote_provenance,
            quote_wordcount=quote_wordcount,
            discontinuous_prep=discontinuous_prep,
            orig_disc_prep=orig_disc_prep,
            v_obj_pp_tail=v_obj_pp_tail,
            causative_extraposition_pat=causative_extraposition_pat,
            causative_extraposition_ex=causative_extraposition_ex,
            correlative_second_arm=correlative_second_arm,
            correlative_pat=correlative_pat,
            correlative_frame=correlative_frame,
            lv_triad_prep=lv_triad_prep,
            lv_triad_pat=lv_triad_pat,
            lv_triad_frame=lv_triad_frame,
            cxn_override_pat=cxn_override_pat,
            cxn_override_frame=cxn_override_frame,
            is_idiom_slot=is_idiom_slot,
            m_caus_adj=m_caus_adj,
            initial_atomic_bundle=initial_atomic_bundle,
            sense_native_prep=sense_native_prep,
            anchor=anchor,
            anchor_type=anchor_type,
            anchor_source=anchor_source,
            prep_obj=prep_obj,
            quote_governed_prep=quote_governed_prep,
            quote_is_transitive=quote_is_transitive,
            word_subcats=word_subcats,
            target_inflections_set=target_inflections_set,
            has_multiple_candidates=has_multiple_candidates,
        )
