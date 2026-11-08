"""
Translation micro-task blueprints, minimal pairs, and lexical/syntactic traps.
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


# -----------------------------------------------------------------------------
# High-Yield L1 False Collocation & Delexical Calque Catalog (Chinese -> English)
# Grounded in Contrastive SLA Error Taxonomies & PedLex Collocations
# -----------------------------------------------------------------------------
L1_FALSE_COLLOCATION_CATALOG: Dict[str, Dict[str, str]] = {
    "knowledge": {
        "correct_frame": "acquire / gain knowledge",
        "wrong_frame": "*learn knowledge*",
        "pedagogical_note": "In English, 'knowledge' collocates with 'acquire' or 'gain', never '*learn knowledge*' (a direct calque of Chinese '学知识').",
        "model_example": "Interdisciplinary collaboration helps young scholars acquire broader specialized knowledge.",
        "zh_cue": "学/汲取知识",
    },
    "acquire": {
        "correct_frame": "acquire knowledge",
        "wrong_frame": "*learn knowledge*",
        "pedagogical_note": "Pair 'acquire' with abstract nouns like 'knowledge' or 'skills'; avoid literal L1 translation '*learn knowledge*'.",
        "model_example": "Young researchers must acquire specialized knowledge through empirical practice.",
        "zh_cue": "汲取/学到知识",
    },
    "suggestion": {
        "correct_frame": "make a suggestion",
        "wrong_frame": "*raise / propose suggestions*",
        "pedagogical_note": "English conventionally uses the delexical verb 'make a suggestion'; '*raise suggestions*' is an unidiomatic literal transfer of Chinese '提建议'.",
        "model_example": "During the review meeting, the consultant made several viable suggestions regarding cost reduction.",
        "zh_cue": "提建议",
    },
    "effort": {
        "correct_frame": "make an effort",
        "wrong_frame": "*pay effort / take effort*",
        "pedagogical_note": "In English, one 'makes an effort'; '*pay effort*' is an unidiomatic literal transfer of Chinese '付出努力'.",
        "model_example": "Both departments made a concerted effort to meet the project deadline.",
        "zh_cue": "做出/付出努力",
    },
    "role": {
        "correct_frame": "play a role in",
        "wrong_frame": "*act a role in / take a role in*",
        "pedagogical_note": "Use 'play a role in' (or 'play a part in'); '*act a role*' is an unidiomatic calque of Chinese '扮演/起角色作用'.",
        "model_example": "Early childhood education plays a crucial role in cognitive development.",
        "zh_cue": "起……作用 / 扮演……角色",
    },
    "measure": {
        "correct_frame": "take measures to",
        "wrong_frame": "*adopt measures / make measures*",
        "pedagogical_note": "English conventionally uses 'take measures'; '*adopt measures*' or '*make measures*' is an unidiomatic transfer of Chinese '采取措施'.",
        "model_example": "The municipal authority took decisive measures to curb traffic congestion.",
        "zh_cue": "采取措施",
    },
    "progress": {
        "correct_frame": "make progress",
        "wrong_frame": "*get progress / gain progress*",
        "pedagogical_note": "'Progress' is uncountable and pairs with 'make' ('make significant progress'); avoid '*get / gain progress*' (transfer of Chinese '取得进步').",
        "model_example": "The research team has made substantial progress in renewable battery storage.",
        "zh_cue": "取得进步",
    },
    "contribution": {
        "correct_frame": "make a contribution to",
        "wrong_frame": "*do a contribution to / give a contribution to*",
        "pedagogical_note": "Use the delexical verb 'make a contribution'; '*do / give a contribution*' is an unidiomatic calque of Chinese '做贡献'.",
        "model_example": "Pioneering scientists made an enduring contribution to theoretical physics.",
        "zh_cue": "做贡献",
    },
    "damage": {
        "correct_frame": "cause / do damage to",
        "wrong_frame": "*make damage to*",
        "pedagogical_note": "In English, one 'causes damage' or 'does damage to'; '*make damage*' is an unidiomatic transfer of Chinese '造成/制造破坏'.",
        "model_example": "Severe coastal storms caused extensive damage to public infrastructure.",
        "zh_cue": "造成破坏/损害",
    },
    "decision": {
        "correct_frame": "make a decision",
        "wrong_frame": "*do a decision*",
        "pedagogical_note": "English uses 'make a decision'; '*do a decision*' is a literal transfer of Chinese '做决定'.",
        "model_example": "The executive committee made a prudent decision after reviewing the quarterly audit.",
        "zh_cue": "做决定",
    },
    "mistake": {
        "correct_frame": "make a mistake",
        "wrong_frame": "*do a mistake / commit a mistake*",
        "pedagogical_note": "Pair 'mistake' with 'make' ('make a careless mistake'); avoid '*do a mistake*'.",
        "model_example": "Even experienced translators can make a subtle mistake in register.",
        "zh_cue": "犯错误",
    },
    "price": {
        "correct_frame": "high / low price",
        "wrong_frame": "*expensive / cheap price*",
        "pedagogical_note": "In English, goods are 'expensive/cheap', but prices are 'high/low'; '*expensive price*' is a direct calque of Chinese '价格贵'.",
        "model_example": "Commodity goods remained at a relatively high price throughout the winter quarter.",
        "zh_cue": "价格高/昂贵",
    },
    "traffic": {
        "correct_frame": "heavy / light traffic",
        "wrong_frame": "*big / crowded / many traffic*",
        "pedagogical_note": "Traffic volume is described as 'heavy' or 'light'; '*big / crowded traffic*' is an unidiomatic calque of Chinese '交通大/拥挤'.",
        "model_example": "Urban commuters encountered exceptionally heavy traffic during morning rush hour.",
        "zh_cue": "交通拥堵/车流量大",
    },
    "attention": {
        "correct_frame": "pay attention to",
        "wrong_frame": "*give attention to / take attention to*",
        "pedagogical_note": "English idiom requires 'pay attention to'; avoid '*give / take attention to*' (transfer of Chinese '给予注意').",
        "model_example": "Engineers must pay close attention to structural tolerances under thermal stress.",
        "zh_cue": "注意/予以关注",
    },
    "importance": {
        "correct_frame": "attach importance to",
        "wrong_frame": "*pay importance to / give importance to*",
        "pedagogical_note": "Use 'attach importance to' (or 'place importance on'); '*pay importance*' is an unidiomatic transfer of Chinese '给予/重视'.",
        "model_example": "Leading universities attach great importance to interdisciplinary research ethics.",
        "zh_cue": "重视/予以高度重视",
    },
    "research": {
        "correct_frame": "conduct / carry out research",
        "wrong_frame": "*make research / do research*",
        "pedagogical_note": "In formal English, one 'conducts' or 'carries out research'; avoid informal or unidiomatic '*make research*'.",
        "model_example": "Independent laboratories conducted extensive research on the environmental impact.",
        "zh_cue": "开展/进行研究",
    },
    "habit": {
        "correct_frame": "develop / form a habit",
        "wrong_frame": "*keep a habit / raise a habit*",
        "pedagogical_note": "In English, one 'develops' or 'forms' a habit; '*raise a habit*' is a calque of Chinese '养成习惯'.",
        "model_example": "Early reading programs help children develop a lifelong reading habit.",
        "zh_cue": "养成习惯",
    },
    "problem": {
        "correct_frame": "solve / address a problem",
        "wrong_frame": "*resolve a question / answer a problem*",
        "pedagogical_note": "Pair 'solve / address' with 'problem', and 'answer' with 'question'; avoid confusing 'problem' and 'question'.",
        "model_example": "Multilateral diplomacy helped address the cross-border environmental problem.",
        "zh_cue": "解决问题",
    },
}


class QuizTranslationMixin:
    # =========================================================================
    # SYMBOLIC TRANSLATION MICRO-TASK BLUEPRINTS (translation_quiz)
    # =========================================================================

    @classmethod
    def _classify_vocab_register(cls, word: str, pos: str = "", definition: str = "") -> str:
        """
        Deterministically classifies a vocabulary item's pragmatic register into
        'formal', 'informal', or 'neutral' using PedLex register_tags, definition labels,
        phrasal verb morphology, and CEFR/Latinate profile.
        """
        w_clean = re.sub(r"\[.*?\]|\(.*?\)", "", str(word or "")).strip().lower()
        w_clean = re.sub(r"\s+", " ", w_clean)
        def_low = str(definition or "").lower()
        pos_low = str(pos or "").lower()

        # 1. Explicit bracketed register tags in definition
        if any(t in def_low for t in ("[spoken]", "[informal]", "[slang]", "[colloquial]", "spoken,", "informal]")):
            return "informal"
        if any(t in def_low for t in ("[formal]", "[written]", "[technical]", "[literary]", "[law]", "[medical]")):
            return "formal"

        # 2. PedLex entry & sense register_tags lookup
        entry = cls.get_ldoce_entry(w_clean) or (cls.get_ldoce_entry(w_clean.split()[0]) if " " in w_clean else None)
        if isinstance(entry, dict):
            senses = entry.get("senses") or []
            matched_sense = None
            if def_low:
                norm_target = re.sub(r"[^a-z0-9 ]", "", def_low).strip()
                for s in senses:
                    s_def = re.sub(r"[^a-z0-9 ]", "", str(s.get("definition") or "").lower()).strip()
                    if norm_target and s_def and (norm_target in s_def or s_def in norm_target):
                        matched_sense = s
                        break
            target_senses = [matched_sense] if matched_sense else senses[:2]
            for s in target_senses:
                if not isinstance(s, dict):
                    continue
                reg_tags = [str(r).lower() for r in (s.get("register_tags") or [])]
                dom_tags = [str(d).lower() for d in (s.get("domain_tags") or [])]
                s_def_txt = str(s.get("definition") or "").lower()
                if any(r in reg_tags for r in ("spoken", "informal", "slang", "colloquial", "everyday")) or "[spoken]" in s_def_txt or "[informal]" in s_def_txt:
                    return "informal"
                if any(r in reg_tags for r in ("formal", "written", "technical", "literary", "academic")) or any(d in dom_tags for d in ("law", "science", "medicine", "economics")) or "[formal]" in s_def_txt:
                    return "formal"

        # 3. Multi-word phrasal verbs with conversational particles default to informal
        conversational_particles = {"up", "out", "off", "down", "around", "away", "back", "over", "along", "in", "through"}
        tokens = w_clean.split()
        if pos_low == "phrasal verb" or (len(tokens) == 2 and tokens[1] in conversational_particles):
            # Exclude formal prepositional verbs like "originate in", "comply with", "consist of", "result in"
            formal_prep_verbs = {"originate", "comply", "consist", "culminate", "pertain", "adhere", "concur", "abstain", "deviate", "reside", "prevail"}
            if tokens[0] not in formal_prep_verbs:
                return "informal"

        # 4. Formal academic / Latinate headwords
        formal_academic_roots = {
            "originate", "regulatory", "comply", "subsequent", "consequently", "notwithstanding",
            "constitute", "implement", "predominant", "facilitate", "mandate", "jurisdiction",
            "stipulate", "commence", "endeavor", "ascertain", "proliferate", "mitigate",
            "exacerbate", "unprecedented", "indispensable", "autonomous", "empirical", "paradigm",
        }
        if w_clean in formal_academic_roots:
            return "formal"

        if " " not in w_clean and len(w_clean) >= 8:
            cefr_lvl = cls.get_word_cefr(w_clean)
            if cefr_lvl in ("C1", "C2") and w_clean.endswith(("ate", "ory", "tion", "ive", "ity", "ence", "ance", "ous")):
                return "formal"

        return "neutral"

    @classmethod
    def _classify_grammar_register(cls, name: str, formula: str = "", pedagogical_function: str = "") -> str:
        """
        Deterministically classifies a grammar pattern's stylistic register into
        'formal', 'informal', or 'neutral'.
        """
        combined = f"{name or ''} {formula or ''} {pedagogical_function or ''}".lower()

        # Check informal / conversational markers first
        if any(k in combined for k in (
            "[spoken]", "[informal]", "conversational", "spoken", "informal",
            "question tag", "ellipsis", "get-passive", "everyday coordination",
            "simple coordination", "colloquial", "casual"
        )):
            return "informal"

        # Check formal / elevated academic syntax
        if any(k in combined for k in (
            "inversion", "not only", "seldom", "rarely", "never ", "little + aux",
            "only when", "only by", "only after", "participial", "participle",
            "adjunct", "having +", "v-ing,", "v-ed,", "shell noun", "appositive",
            "cleft", "nominalization", "extraposition", "preposition + which",
            "preposition + whom", "formal", "academic", "elevated"
        )):
            return "formal"

        return "neutral"

    @classmethod
    def _resolve_translation_lexical_trap(
        cls,
        word: str,
        pos: str = "",
        definition: str = "",
        quote: str = "",
        example_usage: str = "",
    ) -> Dict[str, Any]:
        """
        Resolves the target keyword's Required Lexical Frame, Model Example, and
        optional single-point Lexical Error Trap from `pedlex_error_traps` (with
        deterministic symbolic fallback when pedlex.db is absent or unindexed).
        Enforces strict POS and `sense_idx` isolation.
        """
        w_clean = re.sub(r"\[.*?\]|\(.*?\)", "", str(word or "")).strip()
        w_clean = re.sub(r"\s+", " ", w_clean)
        def_str = str(definition or "").strip()
        pos_str = str(pos or "").strip()

        # 1. Query atomic pattern bundle for canonical pattern & authentic example
        locked_sense = None
        s_idx: Optional[int] = None
        entry = cls.get_ldoce_entry(w_clean)
        if entry and entry.get("senses"):
            s_idx = cls._lock_sense(
                entry,
                definition=def_str,
                quote=quote,
                target_pos=pos_str or None,
            )
            if s_idx is not None and 0 <= s_idx < len(entry["senses"]):
                locked_sense = entry["senses"][s_idx]

        atomic_bundle = (
            cls.pick_atomic_pattern_bundle(locked_sense, w_clean, quote=quote)
            if locked_sense
            else None
        )
        bundle_pat = str((atomic_bundle.get("pattern") if isinstance(atomic_bundle, dict) else "") or "").strip()
        bundle_ex = str((atomic_bundle.get("example") if isinstance(atomic_bundle, dict) else "") or "").strip()
        if bundle_pat:
            bundle_pat = re.sub(r"\s+\betc\.?(?:\s+.*)?$", "", bundle_pat).strip()

        if not bundle_pat and isinstance(locked_sense, dict):
            sense_pats = [
                re.sub(r"\s+\betc\.?(?:\s+.*)?$", "", str(p)).strip()
                for p in (locked_sense.get("patterns") or [])
                if str(p).strip()
            ]
            if sense_pats:
                q_low = f"{quote} {example_usage}".lower()
                matched_pats = [p for p in sense_pats if p.lower() in q_low]
                bundle_pat = matched_pats[0] if matched_pats else sense_pats[0]

        # Filter out conversational speech-act formulas, personal deictic phrases,
        # or noisy proper-noun exemplar strings so narrative sentences are clean.
        _dialogic_deictic_re = re.compile(
            r"\b(?:thanks|thank\s+you|hello|hi|sorry|please|excuse\s+me|pardon|your|my|our|you're|i'm|i\s+am|don't\s+worry|having\s+us)\b",
            re.IGNORECASE,
        )
        if bundle_pat and (
            _dialogic_deictic_re.search(bundle_pat)
            or "/" in bundle_pat and any(c.isupper() for c in bundle_pat)
            or len(bundle_pat.split()) > 5
        ):
            bundle_pat = ""

        # Ensure model_example actually contains the target word stem and is not a dialogic greeting/thanks formula
        first_tok = w_clean.split()[0].lower() if w_clean.split() else w_clean.lower()
        w_stem = first_tok[: max(3, min(len(first_tok), 5))]
        candidate_examples = [bundle_ex, example_usage, quote]
        if isinstance(locked_sense, dict):
            candidate_examples.extend(str(ex) for ex in (locked_sense.get("examples") or []) if str(ex).strip())
        model_example = ""
        for cand_ex in candidate_examples:
            cand_s = str(cand_ex or "").strip()
            if not cand_s:
                continue
            if w_stem in cand_s.lower() and not _dialogic_deictic_re.search(cand_s):
                model_example = cand_s
                break
        if not model_example:
            for cand_ex in candidate_examples:
                cand_s = str(cand_ex or "").strip()
                if cand_s and not _dialogic_deictic_re.search(cand_s):
                    model_example = cand_s
                    break

        # For nouns, do not force a bare generic genitive 'Noun + of' frame (e.g. 'emblem of', 'population of')
        # as Required Lexical Frame, since standalone modified nouns ('its official emblem', 'the growing population')
        # would otherwise force the LLM to invent tautological 'of + NP' tails ('population of residents').
        _generic_noun_of_re = re.compile(
            rf"^{re.escape(w_clean)}\s+of$",
            re.IGNORECASE,
        )
        if pos_str.lower() in ("noun", "n") and bundle_pat and _generic_noun_of_re.match(bundle_pat.strip()):
            bundle_pat = ""

        if not bundle_pat:
            ctx_sample = f"{model_example} {quote} {definition}"
            if pos_str.lower() == "noun" and re.search(rf"\bthe\s+{re.escape(w_clean)}\s+of\b", ctx_sample, re.IGNORECASE):
                bundle_pat = f"the {w_clean} of [sth]"
            elif pos_str.lower() == "verb":
                m_vp = re.search(rf"\b{re.escape(w_clean)}(?:s|ed|ing|med|ming)?\s+(into|in|on|to|with|from|for|of)\b", ctx_sample, re.IGNORECASE)
                if m_vp:
                    bundle_pat = f"{w_clean} {m_vp.group(1).lower()} [sth]"
        required_frame = bundle_pat or w_clean

        # 1.5 Check L1 False Collocation Catalog (direct SLA contrastive catalog)
        best_trap: Optional[Dict[str, Any]] = None
        w_low = w_clean.lower()
        cat_key = None
        if w_low in L1_FALSE_COLLOCATION_CATALOG:
            cat_key = w_low
        else:
            for tok in w_low.split():
                if tok in L1_FALSE_COLLOCATION_CATALOG:
                    cat_key = tok
                    break

        if cat_key:
            cat_entry = L1_FALSE_COLLOCATION_CATALOG[cat_key]
            best_trap = {
                "trap_type": "L1_COLLOC_FALSE_FRIEND",
                "correct_frame": cat_entry["correct_frame"],
                "wrong_frame": cat_entry["wrong_frame"],
                "pedagogical_note": cat_entry["pedagogical_note"],
                "model_example": cat_entry["model_example"],
                "zh_cue": cat_entry.get("zh_cue", ""),
            }
            required_frame = cat_entry["correct_frame"]
            if not model_example or not re.search(r"[a-z]", model_example):
                model_example = cat_entry["model_example"]

        # 2. Query pedlex_error_traps with strict POS and sense_idx isolation
        # Focus strictly on high-diagnostic collocation, dependent preposition, and authentic usage traps
        # (Exclude morphological trivia like uncountable noun pluralization '*packagings*' which degrade translation into spelling checks)
        if not best_trap:
            traps = cls.get_word_error_traps(w_clean, pos=pos_str or None, sense_idx=s_idx)
            if traps:
                priority_order = {
                    "GRAM_BOX_DONTSAY": 0,
                    "GRAMBOX_BADEXA": 0,
                    "GRAMBOX_RULE_TRAP": 1,
                    "PREP_COLLOC_TRAP": 2,
                    "SYNTAX_COMP_TRAP": 3,
                    "LEXICAL_VALENCY_FRAME": 4,
                }
                def _is_invalid_prep_or_dialogic_trap(t_obj: Dict[str, Any]) -> bool:
                    t_type = str(t_obj.get("trap_type") or "")
                    cf = str(t_obj.get("correct_frame") or "").strip()
                    cf_low = cf.lower()
                    wf_low = str(t_obj.get("wrong_frame") or "").strip().lower()
                    if _dialogic_deictic_re.search(cf):
                        return True
                    if "wrong preposition/complement" in wf_low or len(cf.split()) > 6:
                        return True
                    if t_type != "PREP_COLLOC_TRAP":
                        return False
                    # Reject generic nominal 'Noun + of' preposition traps (e.g. 'emblem of' vs '*emblem from*',
                    # 'population of' vs '*population from*'), which force artificial tautologies on standalone nouns.
                    if pos_str.lower() in ("noun", "n") and _generic_noun_of_re.match(cf):
                        return True
                    if not re.search(r"\bto\b", cf_low):
                        return False
                    # If 'to' is at the end of the frame (e.g. 'Is it possible to', 'able to') or followed by a bare verb ('to do', 'to be'),
                    # it is an infinitive marker rather than a preposition governing a noun phrase.
                    if re.search(r"\bto\s*$", cf_low) or re.search(r"\bto\s+(?:do|be|have|get|make|take|go|see|find|say|know)\b", cf_low):
                        return True
                    if ("for/with" in wf_low or "for" in wf_low) and pos_str.lower() in ("adjective", "adj") and not re.search(r"\bto\s+(?:somebody|something|sb|sth|\[)", cf_low):
                        return True
                    return False

                valid_traps = [
                    t for t in traps
                    if str(t.get("trap_type") or "") in priority_order and not _is_invalid_prep_or_dialogic_trap(t)
                ]
                if valid_traps:
                    sorted_traps = sorted(
                        valid_traps,
                        key=lambda t: priority_order.get(str(t.get("trap_type") or ""), 99)
                    )
                    best_trap = dict(sorted_traps[0])
                    c_f = str(best_trap.get("correct_frame") or "").strip()
                    t_type_sel = str(best_trap.get("trap_type") or "")
                    if (
                        c_f
                        and required_frame == w_clean
                        and t_type_sel in ("PREP_COLLOC_TRAP", "SYNTAX_COMP_TRAP")
                        and "[" not in c_f
                        and not c_f.endswith((".", "!", "?"))
                        and not _dialogic_deictic_re.search(c_f)
                        and not (pos_str.lower() in ("noun", "n") and _generic_noun_of_re.match(c_f))
                        and len(c_f.split()) <= 6
                    ):
                        required_frame = c_f
                    trap_mod_ex = str(best_trap.get("model_example") or "").strip()
                    if trap_mod_ex and not model_example and not _dialogic_deictic_re.search(trap_mod_ex):
                        model_example = trap_mod_ex

        # 3. Deterministic fallback for common bound-preposition collocation frames
        if not best_trap:
            prep_fallbacks = {
                "comply": ("comply with [rule/law]", "*comply to / comply by*", "All companies must comply with the new safety regulations."),
                "depend": ("depend on [sb/sth]", "*depend of / depend from*", "Our success depends on everyone's cooperation."),
                "rely": ("rely on [sb/sth]", "*rely in / rely at*", "Many rural communities rely on public transport."),
                "consist": ("consist of [parts]", "*be consisted of / consist in*", "The committee consists of ten senior members."),
                "result": ("result in [outcome]", "*result to*", "The policy change resulted in a sharp drop in emissions."),
                "contribute": ("contribute to [sth/V-ing]", "*contribute for*", "Regular exercise contributes to better mental health."),
                "adapt": ("adapt to [new situation]", "*adapt with*", "Species must adapt to changing environments."),
                "participate": ("participate in [activity]", "*participate on / participate to*", "Students are encouraged to participate in seminar discussions."),
                "succeed": ("succeed in [sth/V-ing]", "*succeed to do*", "She succeeded in persuading the board."),
                "insist": ("insist on [sth/V-ing]", "*insist to do*", "He insisted on paying for the meal."),
            }
            w_low = w_clean.lower()
            if w_low in prep_fallbacks:
                c_frame, w_frame, mod_ex = prep_fallbacks[w_low]
                if required_frame == w_clean or required_frame.startswith(f"{w_low} "):
                    required_frame = c_frame
                if not model_example:
                    model_example = mod_ex
                best_trap = {
                    "trap_type": "PREP_COLLOC_TRAP",
                    "correct_frame": c_frame,
                    "wrong_frame": w_frame,
                    "pedagogical_note": f"Use the collocation '{c_frame}', not {w_frame}.",
                }
            elif bundle_pat and " " in bundle_pat:
                m_prep = re.search(rf"\b{re.escape(w_low)}\s+(in|on|at|to|with|for|from|of|into|about|over|against)\b", bundle_pat.lower())
                if m_prep:
                    prep_tok = m_prep.group(1)
                    bp_low = bundle_pat.strip().lower()
                    is_inf_to = (
                        prep_tok == "to"
                        and (
                            bp_low.endswith(" to")
                            or bool(re.search(r"\bto\s+(?:do|be|have|get|make|take|go|see|find|say|know)\b", bp_low))
                            or (pos_str.lower() in ("adjective", "adj") and not re.search(r"\bto\s+(?:somebody|something|sb|sth|\[)", bp_low))
                        )
                    )
                    is_generic_noun_of = pos_str.lower() in ("noun", "n") and prep_tok == "of"
                    if not is_inf_to and not is_generic_noun_of:
                        alt_prep = {"in": "from/at", "into": "at/on", "with": "to", "on": "of", "to": "for/with", "of": "from", "from": "of", "for": "to"}.get(prep_tok, "to")
                        best_trap = {
                            "trap_type": "PREP_COLLOC_TRAP",
                            "correct_frame": bundle_pat,
                            "wrong_frame": f"*{w_clean} {alt_prep}*",
                            "pedagogical_note": f"Use '{bundle_pat}' rather than substituting '*{w_clean} {alt_prep}*'.",
                        }

        # Format human-readable lexical flaw label and corruption instruction
        lex_flaw_label = ""
        lex_corruption_step = ""
        if best_trap:
            t_type = str(best_trap.get("trap_type") or "")
            c_frame = str(best_trap.get("correct_frame") or required_frame).strip()
            w_frame = str(best_trap.get("wrong_frame") or "").strip()
            dont_say = str(best_trap.get("dont_say") or best_trap.get("bad_example") or "").strip()
            if t_type == "L1_COLLOC_FALSE_FRIEND":
                lex_flaw_label = f"L1 False Collocation / Calque Trap (`{c_frame}` vs `{w_frame}`)"
                lex_corruption_step = f"using the literal L1 calque `{w_frame}` instead of `{c_frame}`"
            elif t_type == "PREP_COLLOC_TRAP":
                lex_flaw_label = f"Lexical Valency / Preposition Trap (`{c_frame}` vs `{w_frame}`)"
                lex_corruption_step = f"substituting the wrong preposition `{w_frame}` instead of `{c_frame}`"
            elif t_type == "SYNTAX_COMP_TRAP":
                lex_flaw_label = f"Subcategorization Complement Trap (`{c_frame}` vs `{w_frame}`)"
                lex_corruption_step = f"replacing the required complement frame `{c_frame}` with `{w_frame}`"
            elif t_type in ("GRAM_BOX_DONTSAY", "GRAMBOX_BADEXA", "GRAMBOX_RULE_TRAP"):
                bad_ex = f" (Do NOT say: {dont_say})" if dont_say else f" (`{w_frame}`)"
                lex_flaw_label = f"Lexical Collocation / Usage Trap on `{w_clean}`{bad_ex}"
                lex_corruption_step = f"introducing the authentic learner error `{w_frame or dont_say}` instead of `{c_frame}`"
            elif t_type == "LEXICAL_VALENCY_FRAME":
                lex_flaw_label = f"Lexical Valency / Subcategorization Frame Trap (`{c_frame}` vs `{w_frame}`)"
                lex_corruption_step = f"violating the required valency frame `{c_frame}` by using `{w_frame}`"

        return {
            "required_frame": required_frame,
            "model_example": model_example,
            "trap": best_trap,
            "flaw_label": lex_flaw_label,
            "corruption_step": lex_corruption_step,
            "zh_cue": best_trap.get("zh_cue", "") if best_trap else "",
        }

    @classmethod
    def _resolve_translation_syntactic_trap(cls, gram_item: Dict[str, str]) -> Dict[str, str]:
        """
        Deterministically extracts a Single-Point Syntactic L1 Negative-Transfer Trap
        from a grammar pattern and its `Grammar Warning (<TOPIC>)` common mistake.
        Matches strictly against the pattern name, formula, and warning topic header
        (never matching incidental prose words inside the explanation body).
        """
        name = str(gram_item.get("name") or "Target Grammar Pattern").strip()
        formula = str(gram_item.get("formula") or name).strip()
        mistakes = str(gram_item.get("common_mistakes") or "").strip()

        m_topic = re.search(r"(?:Grammar\s+Warning\s*)?\(([^)]+)\)", mistakes, re.IGNORECASE)
        warn_topic = m_topic.group(1).strip() if m_topic else ""
        header_low = f"{name} {formula} {warn_topic}".lower()
        name_form_low = f"{name} {formula}".lower()
        form_low = formula.lower()

        if any(k in header_low for k in ("although", "even though", "though")) and "despite" not in header_low:
            return {
                "flaw_label": "L1 Conjunction Redundancy (`Although ... but ...` double conjunction)",
                "corruption_step": "adding redundant `but` before the main clause in an `Although / Even though` sentence",
                "constraint_note": "NO 'but' in main clause",
                "zh_cue": "虽然...，但...",
            }
        if (
            any(k in header_low for k in ("because", "since", "as a result of", "causal", "cause and effect", "reason clause", "explanatory causal"))
            and "because of" not in header_low
        ):
            return {
                "flaw_label": "L1 Conjunction Redundancy (`Because / Since ... so ...` double conjunction)",
                "corruption_step": "adding redundant `so` before the main clause in a `Because / Since` causal sentence (e.g., `*Because ..., so ...*`)",
                "constraint_note": "NO 'so' in main clause",
                "zh_cue": "因为...，所以...",
            }
        if (
            any(k in header_low for k in ("inanimate", "enablement", "propositional packaging"))
            or "enable" in form_low
            or ("demonstrate" in form_low and "that" in form_low)
            or ("reveal" in form_low and "that" in form_low)
        ):
            if "enable" in form_low or "allow" in form_low or "permit" in form_low or "causation" in header_low or "enablement" in header_low:
                return {
                    "flaw_label": "Inanimate Causation vs. Instrumental Prepositional Fronting Trap (`With / By + [Tool], [Subject] can...` instead of `[Tool] enables [Object] to do`)",
                    "corruption_step": "reverting the inanimate causal subject to literal Chinese circumstantial fronting (e.g. `With / By + [NP], people / doctors can ...` instead of `[NP] enables / allows ... to ...`)",
                    "constraint_note": "use an Inanimate Subject + enable/allow + [Object] + to-do (NO 'With ..., [Subject] can ...')",
                    "zh_cue": "凭借/借助……，人们能够……",
                }
            return {
                "flaw_label": "Inanimate Subject Packaging vs. Circumstantial Personal Fronting Trap (`Through + [Study/Data], we / researchers found...` instead of Inanimate Subject)",
                "corruption_step": "reverting the inanimate reporting subject to literal Chinese circumstantial personal fronting (e.g. `Through + [Study/Data], we / researchers found that ...` instead of `The study reveals / demonstrates that ...`)",
                "constraint_note": "deploy the Inanimate Research / Data Subject as the main clause subject (NO 'Through ..., researchers found ...')",
                "zh_cue": "通过……，研究人员发现……",
            }
        if (
            any(k in header_low for k in ("parataxis", "verb-stack", "verb stack", "serial verb", "verb stacking", "hierarchical particip", "subordination vs. coordination"))
            or ("particip" in header_low and ("stack" in header_low or "serial" in header_low or "hierarchy" in header_low or "hierarchical" in header_low))
            or "and [v3]" in form_low
            or "and v3" in form_low
        ):
            return {
                "flaw_label": "Parataxis Verb-Stack Trap (flat coordinate verb chain instead of participial subordination)",
                "corruption_step": "flattening the hierarchical participial clause into a mechanical coordinate verb chain (e.g., 'Subject + V1, V2 and V3' or 'Subject + V1 and V2 and V3' instead of 'V-ing..., Subject + V2 and V3')",
                "constraint_note": "subordinate the first action into a fronted participial clause 'V-ing / Having V-ed..., Subject + V2 and V3' (NO flat coordinate verb stacking 'Subject + V1 and V2 and V3')",
                "zh_cue": "……（动作一），……（动作二）并……（动作三）",
            }
        if any(k in header_low for k in ("despite", "in spite of")):
            return {
                "flaw_label": "Clause-Preposition Confusion (`Despite / In spite of + *finite clause*`)",
                "corruption_step": "placing a bare finite clause directly after `Despite / In spite of` instead of a noun phrase, `-ing` gerund, or `the fact that`",
                "constraint_note": "followed by NP / V-ing / the fact that, NOT a bare finite clause",
                "zh_cue": "尽管...，...",
            }
        if (
            "resultative particip" in name_form_low
            or "post-positioned present particip" in name_form_low
            or "comma splice" in header_low
            or bool(re.search(r"\[vp\]\s*,\s*\[v-ing", form_low))
        ):
            return {
                "flaw_label": "Resultative Participle vs. Comma Splice / Voice Trap (replacing `, [V-ing]` with passive `*, V-ed*` or an uncoordinated finite verb)",
                "corruption_step": "replacing the post-positioned present participle `, [V-ing]` with a passive participle `*, V-ed*` or a bare finite verb (comma splice)",
                "constraint_note": "attach `, [V-ing Phrase]` directly after the main clause without a conjunction",
                "zh_cue": "……，从而造成了/象征着……",
            }
        if (
            "passive" in name_form_low
            or "unaccusative" in name_form_low
            or "be/modal be" in form_low
            or bool(re.search(r"\[be[^\]]*\]\s*\+\s*\[v(?:-ed|3)\]", form_low))
        ):
            return {
                "flaw_label": "Passive Voice / Unaccusative Verb Trap (malformed passive auxiliary/participle or passivized intransitive verb)",
                "corruption_step": "corrupting the passive verb form (`[be] + [V-ed]` -> active/base verb or missing `be`) or passivizing an intransitive verb",
                "constraint_note": "use `[be/modal be] + past participle [V-ed]`",
                "zh_cue": "被……/由……",
            }
        if (
            "causative" in header_low
            or "bare infinitive" in header_low
            or "bare verb" in header_low
            or bool(re.search(r"\b(?:let|make|have)\s*\+\s*\[", form_low))
        ):
            return {
                "flaw_label": "Causative Infinitive Trap (`let/make/have + Object + *to do*` instead of bare infinitive)",
                "corruption_step": "inserting ungrammatical `*to*` before the infinitive verb after causative `let / make / have` (e.g., `*let sb. to do*`)",
                "constraint_note": "use a bare infinitive (NO 'to') after `let / make / have + Object`",
                "zh_cue": "让/使某人做……",
            }
        if (
            (
                any(k in header_low for k in ("inversion", "negative adv", "seldom", "rarely", "little", "only when", "only by", "only after"))
                or bool(re.search(r"\bnever\b", header_low))
            )
            and "nevertheless" not in header_low
            or ("not only" in header_low and "correlative" not in header_low and "aux" in form_low)
        ):
            return {
                "flaw_label": "Failed Subject-Auxiliary Inversion (keeping declarative Subject + Verb order)",
                "corruption_step": "failing to invert the auxiliary verb and subject after the fronted negative/restrictive adverbial",
                "constraint_note": "invert Auxiliary + Subject in the main/fronted clause",
                "zh_cue": "不仅/很少/只有...",
            }
        if "correlative" in header_low or ("not only" in header_low and "but also" in header_low):
            return {
                "flaw_label": "Correlative Parallelism Misplacement (`not only ... but also` unbalanced constituent slots)",
                "corruption_step": "breaking parallel constituent structure after `not only` and `but also`",
                "constraint_note": "place parallel grammatical constituents immediately after `not only` and `but also`",
                "zh_cue": "不仅……，而且……",
            }
        if (
            "past particip" in name_form_low
            or bool(re.search(r"\[v(?:-ed|3)\s+phrase\]", form_low))
        ):
            return {
                "flaw_label": "Participle Voice Trap in Post-Positioned Adjunct (using active `*, [V-ing]*` instead of passive `, [V-ed]`)",
                "corruption_step": (
                    "replacing the past participle `, [V-ed]` in the post-positioned adjunct with an ungrammatical active present participle "
                    "`*, [V-ing]*` (e.g., `*following by*` instead of `followed by`, or `*organizing by*` instead of `organized by`)"
                ),
                "constraint_note": (
                    "attach a passive past participial phrase `, [V-ed Phrase]` directly after a comma (e.g., `, followed by ...`, `, organized by ...`, `, held at ...`) "
                    "— NEVER an active `-ing` phrase or nominative absolute (`*each person bringing*`)"
                ),
                "zh_cue": "……，由……组织 / 随后是……",
            }
        if any(k in header_low for k in ("particip", "v-ing", "v-ed", "adjunct", "having +")):
            return {
                "flaw_label": "Dangling Participle / Participle Voice Confusion",
                "corruption_step": "misusing the participle voice (`-ing` vs `-ed`) or inserting a redundant conjunction (`and`/`so`) before the main clause",
                "constraint_note": "align participle subject with main-clause subject without redundant conjunction",
                "zh_cue": "伴随/分词状语...",
            }
        if any(k in header_low for k in ("shell noun", "appositive")):
            return {
                "flaw_label": "Appositive Clause Connector Trap (using `*which*` instead of appositive `that`)",
                "corruption_step": "replacing the appositive complementizer `that` with `*which*` after the shell noun",
                "constraint_note": "use appositive `that` + complete clause",
                "zh_cue": "……这一事实/观点",
            }
        if any(k in header_low for k in (", which", "non-restrictive", "anaphoric", "sentential relative")):
            return {
                "flaw_label": "Non-Restrictive Relative Clause Trap (`*, that*` or comma splice)",
                "corruption_step": "replacing `, which` with ungrammatical `*, that*` or omitting the relative pronoun in a comma splice",
                "constraint_note": "use `, which` + singular verb",
                "zh_cue": "……，这意味着/表明……",
            }
        if (
            "restrictive relative" in name_form_low
            or "relative clause" in name_form_low
            or bool(re.search(r"\[np\]\s*\+\s*(?:that|who|which)\s*\+\s*\[vp\]", form_low))
        ):
            return {
                "flaw_label": "Relative Clause Verb Agreement Trap (mismatching verb number agreement with the antecedent `[NP]` inside the `that/who/which` clause)",
                "corruption_step": "mismatching singular/plural verb agreement inside the relative clause (`that/who/which + *singular/plural verb mismatch*`) against the antecedent `[NP]`",
                "constraint_note": "strictly match relative-clause verb number agreement to the antecedent `[NP]` in `[NP] + that/who/which + [VP]`",
                "zh_cue": "……的（定语从句）",
            }
        if any(k in header_low for k in ("conditional", "unless", "as long as", "provided that", "stipulative")) or form_low.startswith("if "):
            return {
                "flaw_label": "Conditional Tense / Modal Trap (inserting modal `*will / would*` inside the conditional clause)",
                "corruption_step": "inserting modal `*will*` or `*would*` inside the conditional subordinate clause",
                "constraint_note": "use present/past indicative tense (NOT 'will/would') inside the conditional clause",
                "zh_cue": "如果/除非……，就……",
            }
        if any(k in header_low for k in ("cleft", "extraposition", "dummy-it", "it + [be]", "it is")):
            return {
                "flaw_label": "Extraposition / Cleft Structure Trap (missing dummy `It` or malformed connector)",
                "corruption_step": "omitting dummy subject `It` or replacing the required `that/to` complementizer",
                "constraint_note": "preserve `It is ... that/to` frame",
                "zh_cue": "正是/……是必要的",
            }
        if (
            any(k in header_low for k in ("preparatory", "object complement", "complex-transitive", "complex transitive"))
            or bool(re.search(r"\b(?:find|make|keep|consider|think|deem)\s*\+\s*\[(?:np|it|object)\]\s*\+\s*\[adj", form_low))
        ):
            return {
                "flaw_label": "Complex-Transitive / Preparatory-It Trap (omitting dummy object `it` before `[adj] + to-V` or using an `*-ly*` adverb instead of `[adj]`)",
                "corruption_step": "omitting dummy object `it` in `find/make it + [adj] + to do` (e.g., `*find hard to do*`) or replacing the object-complement adjective `[adj]` with an `*-ly*` adverb",
                "constraint_note": "keep `find/make + [NP / dummy it] + [adj]` intact",
                "zh_cue": "觉得/发现……是……的",
            }
        if any(
            k in header_low
            for k in (
                "conjunctive adverb",
                "transition adverb",
                "discourse stance",
                "however,",
                "therefore,",
                "nevertheless,",
                "consequently,",
                "moreover,",
            )
        ):
            return {
                "flaw_label": "Conjunctive Adverb Comma Splice (using a comma instead of `;` or `.` before `however / therefore / nevertheless` when joining two independent clauses)",
                "corruption_step": "joining two independent clauses with only a comma before `however / therefore / nevertheless` (comma splice) or adding a redundant conjunction (`*However, ... but ...*`)",
                "constraint_note": "use a period or semicolon before `However, / Therefore,` when linking two independent clauses",
                "zh_cue": "然而/因此，……",
            }

        clean_warn = re.sub(r"^(?:Grammar\s+Warning\s*)?\([^)]*\)\s*:?\s*", "", mistakes, flags=re.IGNORECASE).strip()
        if clean_warn:
            # Split at sentence boundary, colon+quote example, or example markers without splitting on 'sb.', 'sth.', 'vs.', 'e.g.'
            parts = re.split(
                r"(?<!\bvs)(?<!\bsb)(?<!\bsth)(?<!\be\.g)(?<!\bi\.e)\.\s+|:\s*[\"'\u201c\u2018]|\s+;\s+|\n|\[INCORRECT:|✗",
                clean_warn,
            )
            short_warn = parts[0].strip().rstrip(":;,./ ")
            # Strip trailing ', or ...' clause if it pushes the label past 100 chars
            if len(short_warn) > 95 and ", or " in short_warn:
                short_warn = short_warn.split(", or ")[0].strip()
            topic_label = f"{warn_topic}: {short_warn}" if warn_topic and warn_topic.lower() not in short_warn.lower() else short_warn
            if len(topic_label) > 120:
                topic_label = topic_label[:120].rsplit(" ", 1)[0].rstrip(",;:./ ")
            # Balance or strip any unclosed quotes and parentheses
            if topic_label.count("'") % 2 == 1:
                topic_label = topic_label.rstrip("'/ ") + "'"
            if topic_label.count('"') % 2 == 1:
                topic_label = topic_label.rstrip('"/ ') + '"'
            if topic_label.count("(") > topic_label.count(")"):
                topic_label += ")" * (topic_label.count("(") - topic_label.count(")"))
            return {
                "flaw_label": f"Syntactic Structure Trap ({topic_label})",
                "corruption_step": f"violating the `{name}` rule ({topic_label})",
                "constraint_note": f"strictly follow `{formula}`",
                "zh_cue": "",
            }

        return {
            "flaw_label": f"Syntactic Formula Breakdown (`{name}`)",
            "corruption_step": f"distorting the required syntactic formula `{formula}` while keeping all other words identical",
            "constraint_note": f"strictly follow `{formula}`",
            "zh_cue": "",
        }

    @classmethod
    def calculate_minimal_pair_similarity(cls, idiomatic: str, flawed: str) -> float:
        """
        Computes the word-level overlap similarity (0.0 to 1.0) between
        `idiomatic_translation` and `flawed_translation` to verify the Minimal-Pair
        corruption contract (≥85% shared non-target wording).
        """
        import difflib
        from collections import Counter

        t1 = re.findall(r"[a-z0-9']+", str(idiomatic or "").lower())
        t2 = re.findall(r"[a-z0-9']+", str(flawed or "").lower())
        if not t1 or not t2:
            return 0.0

        c1, c2 = Counter(t1), Counter(t2)
        shared = sum((c1 & c2).values())
        max_len = max(len(t1), len(t2))
        min_len = min(len(t1), len(t2))
        # If one sentence is a pure prefix/suffix extension of the other by >= 2 tokens
        # (e.g. tacking 'at the church' onto the end of an untouched sentence), normalize by max_len
        if min_len >= 4 and (max_len - min_len) >= 2 and (t2[:min_len] == t1[:min_len] or t2[-min_len:] == t1[-min_len:]):
            denom = max(max_len, 1)
        else:
            denom = max(min_len, 1)
        multiset_overlap = shared / denom
        seq_ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
        raw_sim = round(min(1.0, max(multiset_overlap, seq_ratio)), 4)
        # In short A1/A2 sentences (4-7 words), a strict single-locus minimal-pair edit
        # (e.g. 5/6 shared = 0.8333, or passive 'be found' -> active '*find*') is a valid Minimal Pair.
        if min_len >= 4 and abs(len(t1) - len(t2)) <= 1:
            if (max_len - shared) <= 1:
                return max(0.86, raw_sim)
            if min_len <= 7 and (max_len - shared) <= 2:
                return max(0.86, raw_sim)

        # Single-constituent minimal-pair recognition for complex/academic sentences (B2-C2):
        # If all non-matching tokens are concentrated in a single contiguous edit locus
        # (bracketed by identical prefix and/or suffix) and shared wording >= 72%,
        # it is a verified single-constituent Minimal Pair (e.g. nominalization 'the effective utilization of'
        # vs clause 'how to effectively utilize' sharing 14/18 = 77.8% words).
        blocks = [b for b in difflib.SequenceMatcher(None, t1, t2).get_matching_blocks() if b.size > 0]
        is_single_locus = False
        if len(blocks) == 1:
            b = blocks[0]
            if (b.a == 0 and b.b == 0) or (b.a + b.size == len(t1) and b.b + b.size == len(t2)):
                is_single_locus = True
        elif len(blocks) == 2:
            b1, b2 = blocks[0], blocks[1]
            if (b1.a == 0 and b1.b == 0) and (b2.a + b2.size == len(t1) and b2.b + b2.size == len(t2)):
                is_single_locus = True

        if is_single_locus and raw_sim >= 0.72 and min_len >= 8:
            return max(0.86, raw_sim)

        return raw_sim

    @classmethod
    def detect_comparative_translation_defect(
        cls,
        idiomatic: str,
        flawed: str,
        target_grammar: str = "",
    ) -> Optional[str]:
        """
        Deterministically detects structural / syntactic minimal-pair defects in
        Comparative Translation Appraisal (`idiomatic_translation` vs `flawed_translation`):
        1. Pure Tail / Prefix Phrase Padding (`t2` leaves `t1` 100% intact and merely appends/drops
           a >=2-word prepositional/adverbial tail like 'at the church').
        2. Stacked Preposition + To-Infinitive Collision (e.g. '*volunteered with to help*' where a
           preposition is stacked directly in front of 'to + verb').
        3. Post-Positioned Past Participial Adjunct Mismatch (`[V-ed Phrase]` required by `target_grammar`,
           yet the post-comma adjunct in `idiomatic_translation` uses an active `-ing` clause like
           'each person bringing a dish' with no past participle).
        """
        id_str = str(idiomatic or "").strip()
        fl_str = str(flawed or "").strip()
        if not id_str or not fl_str:
            return None

        t1 = re.findall(r"[a-z0-9']+", id_str.lower())
        t2 = re.findall(r"[a-z0-9']+", fl_str.lower())
        if not t1 or not t2:
            return None

        # 1. Pure Tail Padding: flawed_translation starts with the exact tokens of idiomatic_translation
        #    and merely appends >= 2 trailing tokens (e.g. 'at the church'), or vice versa.
        min_l = min(len(t1), len(t2))
        max_l = max(len(t1), len(t2))
        if min_l >= 4 and (max_l - min_l) >= 2:
            if t2[:min_l] == t1[:min_l]:
                extra_tail = " ".join(t2[min_l:] if len(t2) > len(t1) else t1[min_l:])
                return (
                    f"pure tail/head padding in flawed_translation ('{extra_tail}' appended/dropped at end of sentence "
                    f"without introducing a minimal-pair grammatical error inside the sentence)"
                )

        # 2. Stacked Preposition + To-Infinitive Collision (e.g. '*volunteered with to help*')
        _prep_to_verb_re = re.compile(
            r"\b(with|for|in|on|at|of|from|by|into|over)\s+to\s+([a-z]{2,})\b",
            re.IGNORECASE,
        )
        m_stack = _prep_to_verb_re.search(fl_str)
        if m_stack and not _prep_to_verb_re.search(id_str):
            return (
                f"stacked preposition + to-infinitive collision '*{m_stack.group(0)}*' in flawed_translation "
                f"(replace 'to {m_stack.group(2)}' with an '-ing' gerund or a single wrong preposition instead of stacking both)"
            )

        # 3. Post-Positioned Past Participial Adjunct (`[V-ed Phrase]`) Voice Verification
        g_low = str(target_grammar or "").lower()
        if "past particip" in g_low or bool(re.search(r"\bv(?:-ed|3)\s+phrase\b", g_low)):
            post_seg = id_str.rsplit(",", 1)[1].strip() if "," in id_str else id_str
            post_low = post_seg.lower()
            _irreg_vbn = {
                "been", "born", "borne", "beaten", "become", "begun", "bent", "bitten", "blown",
                "broken", "brought", "built", "burnt", "bought", "caught", "chosen", "come", "cost",
                "cut", "dealt", "done", "drawn", "drunk", "driven", "eaten", "fallen", "fed", "felt",
                "fought", "found", "flown", "forgotten", "forgiven", "frozen", "given", "gone", "grown",
                "hung", "had", "heard", "hidden", "hit", "held", "hurt", "kept", "known", "laid",
                "led", "left", "lent", "let", "lain", "lit", "lost", "made", "meant", "met", "paid",
                "put", "read", "ridden", "rung", "risen", "run", "said", "seen", "sought", "sold",
                "sent", "set", "shaken", "shone", "shot", "shown", "shut", "sung", "sunk", "sat",
                "slept", "slid", "spoken", "spent", "spilt", "split", "spread", "stood", "stolen",
                "stuck", "stung", "struck", "sworn", "swept", "swum", "taken", "taught", "torn",
                "told", "thought", "thrown", "understood", "woken", "worn", "won", "written",
            }
            post_toks = re.findall(r"[a-z]+", post_low)
            has_past_part = any(
                (w.endswith("ed") and len(w) >= 4) or w in _irreg_vbn
                for w in post_toks
            )
            has_ing_part = any(
                w.endswith("ing") and len(w) >= 5 and w not in ("during", "nothing", "something", "anything", "everything", "morning", "evening", "building", "painting", "meeting")
                for w in post_toks
            )
            if not has_past_part and has_ing_part:
                return (
                    f"post-comma adjunct must use a passive past participle '[V-ed Phrase]' (e.g. ', followed by...', ', organized by...'), "
                    f"not an active '-ing' phrase ('{post_seg}')"
                )

        return None

    @classmethod
    def build_translation_micro_task_blueprints(
        cls,
        vocab_content: Any,
        grammar_content: Any = None,
        target_count: int = 5,
        target_language: str = "Chinese",
        cefr_level: str = "B2",
    ) -> List[Dict[str, Any]]:
        """
        Builds Symbolic Translation Micro-Task Blueprints (`translation_quiz`):
        1. Step 1 — Register-Aligned Lexical & Syntactic Pairing (`Symbolic Pair Matching`):
           Pairs each `Target Keyword` with a register-compatible `Target Grammar Pattern`
           (preventing awkward forced marriages like `[spoken]` phrasal verbs with formal inversions).
        2. Step 2 — Single-Point Minimal-Pair Corruption Operator (`pedlex_error_traps` & `Grammar Warning`):
           Assigns a deterministic single-point flaw operator and CRC32-balanced `correct_answer_index` (0 or 1).
        3. Step 3 — Formats the canonical `### Item N Specification ###` with `🎯 Micro-Task for LLM`.
        """
        # 1. Parse vocabulary items (structured list or Markdown string)
        vocab_items: List[Dict[str, str]] = []
        for norm_it in cls._normalize_vocab_items_input(vocab_content):
            vocab_items.append({
                "word": norm_it.get("word") or "",
                "pos": norm_it.get("part_of_speech") or "word",
                "definition": norm_it.get("definition") or "",
                "quote": norm_it.get("quote") or "",
                "example_usage": norm_it.get("example_usage") or "",
            })

        if not vocab_items:
            return []

        # 2. Parse grammar items (structured list or Markdown string)
        grammar_items: List[Dict[str, str]] = []
        if isinstance(grammar_content, (list, tuple)):
            for git in grammar_content:
                if isinstance(git, dict):
                    explicit_topic = str(git.get("syntax_topic") or git.get("name") or "").strip()
                    raw_f = str(git.get("pattern_formula") or explicit_topic or "").strip()
                    explicit_form = str(git.get("formula") or git.get("pattern_formula") or "").strip()
                    ped_f = str(git.get("pedagogical_function") or "").strip()
                    mist = str(git.get("common_mistakes") or git.get("mistakes") or "").strip()
                else:
                    explicit_topic = str(getattr(git, "syntax_topic", "") or getattr(git, "name", "") or "").strip()
                    raw_f = str(getattr(git, "pattern_formula", "") or explicit_topic or "").strip()
                    explicit_form = str(getattr(git, "formula", "") or getattr(git, "pattern_formula", "") or "").strip()
                    ped_f = str(getattr(git, "pedagogical_function", "") or "").strip()
                    mist = str(getattr(git, "common_mistakes", "") or getattr(git, "mistakes", "") or "").strip()
                if not raw_f:
                    continue
                if explicit_topic and explicit_form and explicit_topic != explicit_form:
                    g_name, g_form = explicit_topic, explicit_form
                elif ":" in raw_f:
                    g_name, g_form = [p.strip() for p in raw_f.split(":", 1)]
                else:
                    g_name, g_form = raw_f, explicit_form or raw_f
                grammar_items.append({
                    "name": g_name,
                    "formula": g_form,
                    "pedagogical_function": ped_f,
                    "common_mistakes": mist,
                })
        elif isinstance(grammar_content, str) and grammar_content.strip():
            g_blocks = re.split(r"(?=^##\s*\[\[)", grammar_content.strip(), flags=re.MULTILINE)
            for gb in g_blocks:
                m_gn = re.search(r"^##\s*\[\[(.*)\]\]\s*$", gb, re.MULTILINE)
                if not m_gn:
                    continue
                g_name = m_gn.group(1).strip()
                m_gf = re.search(r"-\s*\*\*Pattern Formula\*\*:\s*(.*)", gb, re.IGNORECASE)
                g_form = m_gf.group(1).strip() if m_gf else g_name
                m_gp = re.search(r"-\s*\*\*Pedagogical Function\*\*:\s*(.*)", gb, re.IGNORECASE)
                ped_f = m_gp.group(1).strip() if m_gp else ""
                m_gm = re.search(r"-\s*\*\*Common Mistakes\*\*:\s*(.*)", gb, re.IGNORECASE)
                mist = m_gm.group(1).strip() if m_gm else ""
                grammar_items.append({
                    "name": g_name,
                    "formula": g_form,
                    "pedagogical_function": ped_f,
                    "common_mistakes": mist,
                })

        # Fallback grammar pool if none provided
        if not grammar_items:
            grammar_items = [
                {
                    "name": "Concessive Clause",
                    "formula": "Although / Even though + Subject + Verb, Main Clause",
                    "pedagogical_function": "Frames a contrast between an obstacle and an outcome.",
                    "common_mistakes": "Grammar Warning (CONCESSIVE CONNECTIVES): Never pair 'Although / Even though' with 'but' in the same sentence.",
                },
                {
                    "name": "Explanatory Causal Clause",
                    "formula": "Because / Since + [Cause Clause], [Main Clause]",
                    "pedagogical_function": "Connects an empirical justification to an outcome without correlative conjunctions.",
                    "common_mistakes": "Grammar Warning (L1 Double Conjunction Calque): Never pair 'Because' or 'Since' with 'so' in the same sentence.",
                },
                {
                    "name": "Inanimate Subject Propositional Packaging",
                    "formula": "[Inanimate/Abstract Subject] + [reveal/demonstrate/show] + that + [Clause]",
                    "pedagogical_function": "Packages empirical findings into an authoritative inanimate subject.",
                    "common_mistakes": "Grammar Warning (Inanimate Subject vs. Circumstantial Fronting): Do NOT revert to 'Through [study], researchers found...'.",
                },
                {
                    "name": "Inanimate Causation and Enablement",
                    "formula": "[Inanimate Subject] + enable/allow/permit + [Object] + to + [VP]",
                    "pedagogical_function": "Expresses causal enablement with an inanimate instrument as grammatical subject.",
                    "common_mistakes": "Grammar Warning (Instrumental Fronting vs. Inanimate Causation): Do NOT use 'With [Tool], people can...'.",
                },
                {
                    "name": "Hierarchical Participial Subordination",
                    "formula": "[V-ing / Having V-ed Phrase], [Subject] + [V2] and [V3]",
                    "pedagogical_function": "Subordinates the initial sequential action into a fronted participial adjunct ('V-ing... / Having V-ed...'), creating a two-tier syntactic hierarchy instead of flat Chinese parataxis.",
                    "common_mistakes": "Grammar Warning (Parataxis Verb-Stacking vs. Hierarchical Participle): Learners frequently produce literal L1 transfer by chaining coordinate verbs flatly: '[INCORRECT: The researchers entered the lab, sanitized their equipment, and commenced the experiment -> CORRECT: Entering the lab, the researchers sanitized their equipment and commenced the experiment].' (Note: Subordinate the preparatory action into a fronted participle 'V-ing...' to establish hypotactic hierarchy).",
                },
            ]

        # 3. Step 1: Discover valid Lexical Collocation / Valency Traps and Syntactic Traps
        valid_lex_items: List[Dict[str, Any]] = []
        for v_it in vocab_items:
            w = v_it["word"]
            p = v_it["pos"]
            d = v_it["definition"]
            v_it["register"] = cls._classify_vocab_register(w, p, d)
            lex_info = cls._resolve_translation_lexical_trap(
                w, pos=p, definition=d, quote=v_it.get("quote", ""), example_usage=v_it.get("example_usage", "")
            )
            trap = lex_info.get("trap")
            trap_type = str(trap.get("trap_type") or "") if isinstance(trap, dict) else ""
            if trap and lex_info.get("flaw_label") and trap_type in (
                "L1_COLLOC_FALSE_FRIEND",
                "PREP_COLLOC_TRAP",
                "SYNTAX_COMP_TRAP",
                "LEXICAL_VALENCY_FRAME",
                "GRAM_BOX_DONTSAY",
                "GRAMBOX_BADEXA",
                "GRAMBOX_RULE_TRAP",
            ):
                v_copy = dict(v_it)
                v_copy["lex_info"] = lex_info
                valid_lex_items.append(v_copy)

        # Supplement high-yield L1 collocation items from catalog if unit vocabulary lacks sufficient traps
        desired_lex_quota = max(1, target_count // 2)
        if len(valid_lex_items) < desired_lex_quota:
            existing_words = {v["word"].lower() for v in valid_lex_items}
            for cat_k, cat_val in L1_FALSE_COLLOCATION_CATALOG.items():
                if len(valid_lex_items) >= desired_lex_quota:
                    break
                if cat_k not in existing_words:
                    sup_lex_info = cls._resolve_translation_lexical_trap(cat_k)
                    if sup_lex_info.get("trap"):
                        valid_lex_items.append({
                            "word": cat_k,
                            "pos": "phrase" if " " in cat_val["correct_frame"] else "noun",
                            "definition": cat_val["correct_frame"],
                            "quote": cat_val["model_example"],
                            "example_usage": cat_val["model_example"],
                            "register": "neutral",
                            "lex_info": sup_lex_info,
                        })
                        existing_words.add(cat_k)

        valid_syn_items: List[Dict[str, Any]] = []
        for g_it in grammar_items:
            g_it["register"] = cls._classify_grammar_register(g_it["name"], g_it["formula"], g_it["pedagogical_function"])
            syn_info = cls._resolve_translation_syntactic_trap(g_it)
            g_copy = dict(g_it)
            g_copy["syn_info"] = syn_info
            valid_syn_items.append(g_copy)

        # 4. Decouple Word & Syntax: Balance Syntactic L1 items (~60%) and Lexical Collocation items (~40%)
        effective_count = max(1, target_count)
        if valid_lex_items:
            target_lex_count = min(len(valid_lex_items), max(1, effective_count // 2))
            target_syn_count = effective_count - target_lex_count
        else:
            target_lex_count = 0
            target_syn_count = effective_count

        final_items: List[Dict[str, Any]] = []
        s_i = 0
        l_i = 0
        for step in range(effective_count):
            if s_i < target_syn_count and (l_i >= target_lex_count or (step + 1) % 2 != 0 or s_i == 0):
                g_cand = valid_syn_items[s_i % len(valid_syn_items)]
                s_i += 1
                final_items.append({"type": "syntactic", "grammar_item": g_cand})
            elif l_i < target_lex_count:
                v_cand = valid_lex_items[l_i % len(valid_lex_items)]
                l_i += 1
                final_items.append({"type": "lexical", "vocab_item": v_cand})
            else:
                g_cand = valid_syn_items[s_i % len(valid_syn_items)]
                s_i += 1
                final_items.append({"type": "syntactic", "grammar_item": g_cand})

        # 5. Compute deterministic CRC32-balanced 2-option slots (0 vs 1, 50/50 split)
        n_items = len(final_items)
        slot_0_count = n_items // 2 + (n_items % 2)
        slot_1_count = n_items // 2
        balanced_slots = [0] * slot_0_count + [1] * slot_1_count
        seed_str = "|".join(
            it["grammar_item"]["name"] if it["type"] == "syntactic" else it["vocab_item"]["word"]
            for it in final_items
        )
        rng_state = zlib.crc32(seed_str.encode("utf-8"))
        for i in range(len(balanced_slots) - 1, 0, -1):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            j = rng_state % (i + 1)
            balanced_slots[i], balanced_slots[j] = balanced_slots[j], balanced_slots[i]

        # 6. Build Single-Point Minimal-Pair Corruption Operator & Micro-Task Blueprint
        blueprints: List[Dict[str, Any]] = []
        cefr_upper = str(cefr_level or "B2").upper()

        for idx, it in enumerate(final_items, 1):
            corr_slot = balanced_slots[idx - 1]

            if it["type"] == "syntactic":
                g_it = it["grammar_item"]
                g_name = g_it["name"]
                g_formula = g_it["formula"]
                g_reg = g_it.get("register", "neutral")
                syn_info = g_it["syn_info"]
                target_grammar_display = (
                    f"{g_name} (`{g_formula}`)"
                    if g_formula and g_formula.lower() != g_name.lower()
                    else g_name
                )
                prescribed_flaw_type = syn_info["flaw_label"]
                primary_corruption = syn_info["corruption_step"]
                gram_constraint = f" ({syn_info['constraint_note']})" if syn_info.get("constraint_note") else ""
                critique_focus = (
                    f"pointing out the exact `{prescribed_flaw_type}` error and confirming the `{g_name}` structure"
                )

                zh_cue_str = syn_info.get("zh_cue", "")
                zh_cue_guide = f" (e.g. expressing an idea such as: \"{zh_cue_str}\")" if zh_cue_str else ""

                micro_task_lines = [
                    f"### Item {idx} Specification ###",
                    f"- Focus Type: Syntactic L1 Negative Transfer Trap",
                    f"- Target Grammar Pattern: {target_grammar_display}",
                    f"- Prescribed Flaw Type: {prescribed_flaw_type}",
                    f"- 🎯 Micro-Task for LLM:",
                    f"  1. `translated_sentence`: Write a natural, polished, and idiomatic {target_language} sentence{zh_cue_guide} expressing a complete thought that naturally requires `{g_name}`. (Do NOT invent convoluted reverse-translationese; make it sound authentic to native {target_language} speakers).",
                    f"  2. `idiomatic_translation`: Translate into natural CEFR {cefr_upper} English faithfully applying `{g_formula}`{gram_constraint}.",
                    f"  3. `flawed_translation`: Keep the sentence nearly identical as a Minimal Pair (≥85% shared wording), ONLY introducing the prescribed syntactic flaw ({primary_corruption}).",
                    f"  4. `diagnostic_critique`: Contrast the two versions by {critique_focus}.",
                ]
                micro_task_block = "\n".join(micro_task_lines)

                blueprints.append({
                    "item_index": idx,
                    "focus_type": "syntactic",
                    "target_keyword": "",
                    "part_of_speech": "",
                    "definition": "",
                    "vocab_register": "",
                    "required_lexical_frame": "",
                    "model_example": "",
                    "target_grammar": f"{g_name}: {g_formula}" if g_formula and g_formula.lower() != g_name.lower() else g_name,
                    "target_grammar_name": g_name,
                    "target_grammar_formula": g_formula,
                    "grammar_register": g_reg,
                    "flaw_type": prescribed_flaw_type,
                    "flaw_category": "syntactic",
                    "lexical_trap": None,
                    "syntactic_trap": syn_info,
                    "correct_answer_index": corr_slot,
                    "micro_task_block": micro_task_block,
                })
            else:
                v_it = it["vocab_item"]
                word = v_it["word"]
                pos = v_it["pos"]
                defn = v_it["definition"]
                v_reg = v_it.get("register", "neutral")
                lex_info = v_it["lex_info"]
                lex_trap = lex_info.get("trap")
                req_frame = lex_info["required_frame"] or word
                model_ex = lex_info["model_example"]
                frame_line = f"{req_frame} (Model: \"{model_ex}\")" if model_ex else req_frame

                prescribed_flaw_type = lex_info["flaw_label"]
                primary_corruption = lex_info["corruption_step"]
                critique_focus = (
                    f"pointing out the exact `{prescribed_flaw_type}` error and confirming the `{req_frame}` collocation"
                )

                lex_zh_cue = (lex_trap.get("zh_cue") or lex_info.get("zh_cue") or "") if isinstance(lex_trap, dict) else (lex_info.get("zh_cue") or "")
                zh_cue_guide = f" (e.g. expressing an idea such as: \"{lex_zh_cue}\")" if lex_zh_cue else ""

                micro_task_lines = [
                    f"### Item {idx} Specification ###",
                    f"- Focus Type: Lexical Collocation & Valency L1 Trap",
                    f"- Target Collocation / Lexical Frame: {word} ({pos}) — \"{defn}\" -> `{frame_line}`",
                    f"- Prescribed Flaw Type: {prescribed_flaw_type}",
                    f"- 🎯 Micro-Task for LLM:",
                    f"  1. `translated_sentence`: Write a natural, polished, and idiomatic {target_language} sentence{zh_cue_guide} requiring the specific meaning and context of `{req_frame}`.",
                    f"  2. `idiomatic_translation`: Translate into natural CEFR {cefr_upper} English correctly using the collocation `{req_frame}`.",
                    f"  3. `flawed_translation`: Keep the sentence nearly identical as a Minimal Pair (≥85% shared wording), ONLY introducing the prescribed collocation/preposition flaw ({primary_corruption}).",
                    f"  4. `diagnostic_critique`: Contrast the two versions by {critique_focus}.",
                ]
                micro_task_block = "\n".join(micro_task_lines)

                blueprints.append({
                    "item_index": idx,
                    "focus_type": "lexical",
                    "target_keyword": word,
                    "part_of_speech": pos,
                    "definition": defn,
                    "vocab_register": v_reg,
                    "required_lexical_frame": req_frame,
                    "model_example": model_ex,
                    "target_grammar": "",
                    "target_grammar_name": "",
                    "target_grammar_formula": "",
                    "grammar_register": "",
                    "flaw_type": prescribed_flaw_type,
                    "flaw_category": "lexical",
                    "lexical_trap": lex_trap,
                    "syntactic_trap": None,
                    "correct_answer_index": corr_slot,
                    "micro_task_block": micro_task_block,
                })

        return blueprints
