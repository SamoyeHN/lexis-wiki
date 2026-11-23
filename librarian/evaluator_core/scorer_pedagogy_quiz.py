"""Item-level pedagogical and psychometric validation for assessment (quiz) items."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..linguistics import LinguisticEngine
from .eval_helpers import (
    STOP_SLOTS,
    _clean_core,
    _form_in_text,
    _safe_str,
    _source_token_variants,
)


def validate_quiz_item(
    item: Dict[str, Any],
    prompt_headwords: Set[str],
    prompt_blueprints: Dict[str, Dict[str, Any]],
    src_cefr_level: Optional[str] = None,
    src_cefr_rank: int = 6,
    src_passage_words: Optional[Set[str]] = None,
) -> Tuple[bool, List[str]]:
    """Validate a single quiz question item against SLA and psychometric criteria."""
    flags: List[str] = []
    if src_passage_words is None:
        src_passage_words = set()

    options = item.get("options", [])
    idx = item.get("correct_answer_index")
    explanation = _safe_str(item.get("explanation") or item.get("diagnostic_critique"))
    question = _safe_str(item.get("question") or item.get("translated_sentence") or item.get("source_sentence"))
    target = _safe_str(item.get("target_word") or item.get("target_keyword") or item.get("word")).strip().lower()
    is_translation = bool(
        item.get("translated_sentence")
        or item.get("source_sentence")
        or item.get("english_skeleton")
        or item.get("target_keyword")
        or item.get("idiomatic_translation")
    )
    skeleton = _safe_str(item.get("english_skeleton"))
    is_comparative_translation = bool(item.get("idiomatic_translation") and item.get("flawed_translation"))

    if is_comparative_translation:
        if not options:
            options = [
                _safe_str(item.get("idiomatic_translation")),
                _safe_str(item.get("flawed_translation")),
            ]
        if idx is None:
            idx = 0
        if not target:
            audit_str = _safe_str(item.get("design_audit"))
            m_aud = re.search(
                r"AUDIT:\s*\[\s*([^\]]+?)\s*\]",
                audit_str,
                re.IGNORECASE,
            )
            if m_aud:
                cand_target = m_aud.group(1).strip().lower()
                if cand_target in prompt_headwords or any(
                    cand_target.startswith(hw) or hw.startswith(cand_target) for hw in prompt_headwords
                ):
                    target = cand_target

    is_comprehension = not target and not is_translation
    expected_opt_count = 2 if is_comparative_translation else 4

    # 1. Option count and distinctness
    is_list_valid = isinstance(options, list) and len(options) == expected_opt_count
    has_no_duplicates = is_list_valid and len(set(str(o).strip().lower() for o in options)) == expected_opt_count

    # 2. Correct answer index validity
    valid_idx = isinstance(idx, int) and 0 <= idx < expected_opt_count

    # 3. Target word must be present in options and match the correct answer slot options[idx]
    target_in_options = True
    if target and is_list_valid and valid_idx:
        clean_target = _clean_core(target)
        if is_comparative_translation:
            idiomatic_ans = _clean_core(_safe_str(item.get("idiomatic_translation")).lower())
            target_toks = [t for t in clean_target.split() if t and t not in STOP_SLOTS]
            is_ly_adv = " " not in clean_target and len(clean_target) >= 5 and clean_target.endswith("ly")
            target_in_options = (
                bool(re.search(rf"\b{re.escape(clean_target)}\b", idiomatic_ans))
                if is_ly_adv
                else (
                    clean_target in idiomatic_ans
                    or (len(clean_target) >= 4 and clean_target[:4] in idiomatic_ans)
                    or (bool(target_toks) and all(_form_in_text(t, idiomatic_ans) for t in target_toks))
                )
            )
        elif is_translation:
            full_ans = _clean_core(_safe_str(item.get("correct_english_answer")).lower())
            skel_ans = _clean_core(_safe_str(item.get("english_skeleton")).lower())
            opt_ans = _clean_core(_safe_str(options[idx]).lower())
            combined_ans = f"{full_ans} {skel_ans} {opt_ans}"
            target_toks = [t for t in clean_target.split() if t and t not in STOP_SLOTS]
            target_in_options = (
                clean_target in full_ans
                or clean_target in skel_ans
                or clean_target in opt_ans
                or (len(clean_target) >= 4 and any(clean_target[:4] in text for text in [full_ans, skel_ans, opt_ans]))
                or (bool(target_toks) and all(_form_in_text(t, combined_ans) for t in target_toks))
            )
        else:
            selected_opt = _clean_core(str(options[idx]))
            target_in_options = (
                (clean_target == selected_opt)
                or (selected_opt in clean_target.split())
                or (
                    len(clean_target) >= 4
                    and len(selected_opt) >= 4
                    and (
                        selected_opt.startswith(clean_target[:4])
                        or clean_target.startswith(selected_opt[:4])
                        or (clean_target in selected_opt)
                    )
                )
            )

    # 4. Blank verification for fill-in-the-blank questions
    has_blank_when_expected = True
    is_single_blank = True
    no_stem_leak = True
    no_article_leak = True
    no_scaffold_leak = True
    no_verbatim_example = True
    verbatim_label = "dictionary example"
    options_match_blueprint = True
    index_match_blueprint = True
    no_verb_inflection_clash = True
    no_comparative_defect = True
    comp_defect_reason = ""
    leaked_words: List[str] = []

    if is_comparative_translation:
        has_blank_when_expected = True
        idiom_txt = _safe_str(item.get("idiomatic_translation"))
        flawed_txt = _safe_str(item.get("flawed_translation"))
        t_gram = _safe_str(item.get("target_grammar") or item.get("design_audit"))
        comp_defect = LinguisticEngine.detect_comparative_translation_defect(
            idiom_txt, flawed_txt, target_grammar=t_gram
        )
        if comp_defect:
            no_comparative_defect = False
            comp_defect_reason = comp_defect
            flags.append(
                f"❌ Quiz item '{target or question[:25]}': Comparative translation defect — {comp_defect}"
            )
        else:
            mp_sim = LinguisticEngine.calculate_minimal_pair_similarity(idiom_txt, flawed_txt)
            if mp_sim < 0.75:
                no_comparative_defect = False
                comp_defect_reason = f"Minimal-Pair contract violation ({mp_sim:.0%} shared wording < 75%)"
                flags.append(
                    f"❌ Quiz item '{target or question[:25]}': Minimal-Pair contract violation — idiomatic_translation and flawed_translation share only {mp_sim:.0%} wording"
                )
    elif is_comprehension:
        has_blank_when_expected = True
    elif is_translation:
        if skeleton:
            has_blank_when_expected = bool(re.search(r"\[\s*_{2,}\s*\]|_{2,}", skeleton))
    elif target and question:
        blanks = re.findall(r"_{2,}", question)
        if len(blanks) == 0:
            has_blank_when_expected = False
        elif len(blanks) > 1:
            is_single_blank = False
            flags.append(
                f"❌ Quiz item '{target}': Multiple blanks ({len(blanks)}) detected in question stem (only exactly ONE blank permitted)"
            )

        # Scaffolding & Meta-Instruction Leakage Gate
        scaffold_leak = re.search(
            r"(?:\b(?:Fit into frame|Fit into the Pattern|Plural noun slot required|Complete predicate required|Write a natural sentence|Model:)\b"
            r"|\b(?:to\s+do|doing)\s+something\b"
            r"|\b(?:is/are|was/were|sb/sth|sth/sb)\b"
            r"|\(\s*should\s*\)"
            r"|\[(?:concrete\s+physical|physical\s+(?:object|person)|sb|sth|doing\s+sth|someone|something|V-base|verb|noun)[^\]]*\])",
            question,
            flags=re.IGNORECASE,
        )
        if scaffold_leak:
            no_scaffold_leak = False
            flags.append(
                f"❌ Quiz item '{target}': Prompt scaffolding leaked into question stem: '{scaffold_leak.group(0)}'"
            )

        # Target Stem Leakage Gate
        clean_target = _clean_core(target)
        target_stem = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', clean_target)
        stem_no_blank = re.sub(r'_{2,}', ' ', question)
        stem_words = re.findall(r'\b[a-zA-Z]+\b', stem_no_blank)
        leaked_words = [
            w for w in stem_words
            if w.lower() == clean_target
            or (len(target_stem) >= 4 and re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', w.lower()) == target_stem)
            or (" " not in clean_target and len(w) >= 4 and LinguisticEngine.shares_derivational_root(w.lower(), clean_target))
        ]
        if leaked_words:
            no_stem_leak = False
            flags.append(
                f"❌ Quiz item '{target}': Target word leaks verbatim into question stem outside blank: {leaked_words}"
            )

        # Distractor Stem Echo Gate
        if is_list_valid and options and valid_idx and not is_translation and len(question) <= 350:
            _stem_words_low = {w.lower() for w in stem_words}
            _stem_lemmas_low = {
                (LinguisticEngine.lemma_of(w.lower()) or w.lower())
                for w in stem_words if len(w) >= 4
            }
            _echo_exempt = {
                "within", "beyond", "across", "around", "behind", "below", "beside", "between",
                "during", "except", "inside", "outside", "through", "throughout", "toward",
                "towards", "under", "until", "without", "against", "about", "above", "after",
                "along", "among", "before", "beneath", "despite",
                "make", "take", "give", "have", "keep", "find", "hold", "come", "turn", "bring",
                "leave", "work", "play", "show", "call", "help", "need", "want", "feel", "seem",
                "become", "start", "begin", "know", "think", "tell", "mean", "send", "use",
                "also", "even", "just", "only", "very", "much", "more", "most", "some", "many",
                "such", "than", "then", "that", "this", "with", "from", "into", "onto", "over",
                "when", "what", "which", "where", "while", "will", "would", "could", "should",
                "must", "been", "being", "were",
            }
            echoed_dists = []
            for o_i, opt in enumerate(options):
                if o_i == idx:
                    continue
                d_clean = _clean_core(opt)
                if not d_clean or " " in d_clean or len(d_clean) < 4 or d_clean in _echo_exempt:
                    continue
                d_lem = LinguisticEngine.lemma_of(d_clean) or d_clean
                if d_lem in _echo_exempt:
                    continue
                if d_clean in _stem_words_low or (len(d_lem) >= 4 and d_lem in _stem_lemmas_low):
                    echoed_dists.append(str(opt).strip())
            if echoed_dists:
                no_stem_leak = False
                flags.append(
                    f"❌ Quiz item '{target}': Distractor option(s) echo verbatim in question stem outside blank: {echoed_dists}"
                )

        # Article Leakage Gate
        no_article_leak = True
        if re.search(r"\ba\s*(?:\(\s*n\s*\)|/\s*an)\s*_{2,}\b", question, flags=re.IGNORECASE):
            no_article_leak = False
            flags.append(
                f"❌ Quiz item '{target}': Question stem uses artificial 'a(n) ____' or 'a/an ____' before blank; use 'the ____', a possessive, an intervening adjective ('a remarkable ____'), or plural instead."
            )
        elif options and re.search(r"\b(?:a|an)\s+_{2,}\b", question, flags=re.IGNORECASE):
            initials = {str(opt).strip()[:1].lower() for opt in options if str(opt).strip()}
            has_vowel_init = any(init in 'aeiou' for init in initials)
            has_cons_init = any(init.isalpha() and init not in 'aeiou' for init in initials)
            if has_vowel_init and has_cons_init:
                no_article_leak = False
                flags.append(
                    f"❌ Quiz item '{target}': Question stem has indefinite article ('a/an') immediately preceding blank with mixed vowel/consonant options (fatal test-wiseness leakage); use 'the ____', plural, or possessive instead."
                )

        bp_info = prompt_blueprints.get(target.lower(), {}) if prompt_blueprints else {}
        bp_pos = bp_info.get("part_of_speech", "")

        # Verb Option Morphological Agreement Gate
        if is_list_valid and options and len(options) == 4:
            t_parts = target.strip().split()
            t_head = t_parts[0].lower()
            t_base = LinguisticEngine.lemma_of(t_head) or t_head
            t_is_adj = any(a in bp_pos for a in ("adj", "adjective"))
            t_is_verb = not t_is_adj and (
                bp_pos in ("verb", "phrasal verb")
                or (not bp_pos and LinguisticEngine._is_base_verb_lemma(t_base))
            )
            if t_is_verb and t_head != t_base:
                t_verb_tag = None
                for cand_tag in ("VBD", "VBG", "VBZ", "VBN"):
                    if (LinguisticEngine._inflect_verb(t_base, cand_tag) or "").lower() == t_head:
                        t_verb_tag = cand_tag
                        break
                if t_verb_tag in ("VBD", "VBN", "VBG", "VBZ"):
                    base_clashing = []
                    for o in options:
                        o_str = str(o).strip()
                        o_parts = o_str.split()
                        if not o_parts:
                            continue
                        o_head = o_parts[0].lower()
                        o_base = o_head if LinguisticEngine._is_base_verb_lemma(o_head) else (LinguisticEngine.lemma_of(o_head) or o_head)
                        o_can_be_verb = (
                            LinguisticEngine._is_base_verb_lemma(o_base)
                            or bool(LinguisticEngine._wordnet_pos(o_base) & {"v"})
                        )
                        if o_can_be_verb and o_head == o_base and o_head != t_head:
                            inflected_form = LinguisticEngine._inflect_verb(o_base, t_verb_tag)
                            if inflected_form and inflected_form.lower() == o_head:
                                continue
                            base_clashing.append(o_str)
                    if base_clashing:
                        no_verb_inflection_clash = False
                        flags.append(
                            f"❌ Quiz item '{target}': Verb option inflection mismatch (动词同型错误) — target is inflected ({t_verb_tag}), but options contain base-form verbs: {base_clashing}"
                        )

        # Auxiliary & Modal Agreement Gate
        q_text = str(item.get("question") or "")
        if is_list_valid and options and len(options) == 4 and "____" in q_text:
            m_left_aux = re.search(r"\b(has|have|had)(?:\s+been)?\s+____", q_text, re.IGNORECASE)
            if m_left_aux:
                aux_word = m_left_aux.group(1).lower()
                vb_clashing = []
                for o in options:
                    o_head = str(o).strip().split()[0].lower()
                    o_base = LinguisticEngine.lemma_of(o_head) or o_head
                    if o_head == o_base and LinguisticEngine._is_base_verb_lemma(o_base):
                        vbn_form = (LinguisticEngine._inflect_verb(o_base, "VBN") or "").lower()
                        if vbn_form and vbn_form != o_base and o_head != vbn_form:
                            vb_clashing.append(o)
                if vb_clashing:
                    no_verb_inflection_clash = False
                    flags.append(
                        f"❌ Quiz item '{target}': Auxiliary agreement clash (助动词语法冲突) — stem has '{aux_word} ____' requiring past participle (VBN), but options contain base-form verbs: {vb_clashing}"
                    )

            m_modal = re.search(r"\b(can|could|will|would|shall|should|may|might|must|did|does|do)\s+____", q_text, re.IGNORECASE)
            if m_modal:
                modal_word = m_modal.group(1).lower()
                infl_clashing = []
                for o in options:
                    o_head = str(o).strip().split()[0].lower()
                    o_base = LinguisticEngine.lemma_of(o_head) or o_head
                    if o_head != o_base and LinguisticEngine._is_base_verb_lemma(o_base):
                        infl_clashing.append(o)
                if infl_clashing:
                    no_verb_inflection_clash = False
                    flags.append(
                        f"❌ Quiz item '{target}': Modal agreement clash (情态动词语法冲突) — stem has '{modal_word} ____' requiring base verb (VB), but options contain inflected verbs: {infl_clashing}"
                    )

            m_to_inf = re.search(r"\bto\s+____", q_text, re.IGNORECASE)
            if m_to_inf and bp_pos in ("verb", "phrase", "phrasal verb"):
                infl_clashing = []
                for o in options:
                    o_head = str(o).strip().split()[0].lower()
                    o_base = LinguisticEngine.lemma_of(o_head) or o_head
                    if o_head != o_base and LinguisticEngine._is_base_verb_lemma(o_base):
                        infl_clashing.append(o)
                if infl_clashing:
                    no_verb_inflection_clash = False
                    flags.append(
                        f"❌ Quiz item '{target}': Infinitive agreement clash (不定式语法冲突) — stem has 'to ____' requiring base verb (VB), but options contain inflected verbs: {infl_clashing}"
                    )

            # Uniform Option Inflection Gate
            vbn_count = 0
            vb_count = 0
            vb_opts = []
            vbn_opts = []
            for o in options:
                o_head = str(o).strip().split()[0].lower()
                o_base = LinguisticEngine.lemma_of(o_head) or o_head
                if LinguisticEngine._is_base_verb_lemma(o_base):
                    vbn_form = (LinguisticEngine._inflect_verb(o_base, "VBN") or "").lower()
                    if o_head == o_base and vbn_form != o_base:
                        vb_count += 1
                        vb_opts.append(o)
                    elif o_head == vbn_form and vbn_form != o_base:
                        vbn_count += 1
                        vbn_opts.append(o)
            if vb_count > 0 and vbn_count > 0 and (vb_count + vbn_count == 4):
                no_verb_inflection_clash = False
                flags.append(
                    f"❌ Quiz item '{target}': Mixed verb inflections among options (选项曲折形式混杂) — base verbs: {vb_opts}, past participles: {vbn_opts}"
                )

        # Prescribed Options Contract Gate
        prescribed_opts = bp_info.get("prescribed_options") or []
        prescribed_idx = bp_info.get("prescribed_index")
        if prescribed_opts and is_list_valid:
            wanted_opts = [str(o).strip().lower() for o in prescribed_opts]
            shipped_opts = [str(o).strip().lower() for o in options]
            if shipped_opts != wanted_opts:
                options_match_blueprint = False
                substituted = sorted(set(wanted_opts) - set(shipped_opts))
                introduced = sorted(set(shipped_opts) - set(wanted_opts))
                flags.append(
                    f"❌ Quiz item '{target}': Prescribed options altered — blueprint prescribed "
                    f"[{', '.join(str(o) for o in prescribed_opts)}], item shipped "
                    f"[{', '.join(str(o) for o in options)}]"
                    + (f" (substituted: {substituted}; introduced: {introduced})" if substituted or introduced else "")
                )
            if isinstance(prescribed_idx, int) and isinstance(idx, int) and idx != prescribed_idx:
                index_match_blueprint = False
                flags.append(
                    f"❌ Quiz item '{target}': Prescribed answer index altered — blueprint index "
                    f"{prescribed_idx}, item shipped {idx}"
                )

        # Verbatim Copy Gate
        no_verbatim_example = True
        verbatim_label = "dictionary example"
        verbatim_hit = ""
        stem_clean = re.sub(r"_{2,}", " ", question.lower())
        stem_tokens = re.sub(r"[^\w\s]", " ", stem_clean).split()
        stem_5grams = {tuple(stem_tokens[i:i + 5]) for i in range(len(stem_tokens) - 4)}
        for _label, _source in (
            ("dictionary example", bp_info.get("example", "")),
            ("curriculum quote", bp_info.get("licensed_quote", "")),
        ):
            for src_clean in _source_token_variants(_source, target):
                if len(src_clean) < 5 or not stem_5grams:
                    continue
                src_5grams = {tuple(src_clean[i:i + 5]) for i in range(len(src_clean) - 4)}
                overlap_5grams = src_5grams.intersection(stem_5grams)
                if overlap_5grams:
                    no_verbatim_example = False
                    verbatim_label = _label
                    verbatim_hit = " ".join(next(iter(overlap_5grams)))
                    break
            if not no_verbatim_example:
                break
        weak_quote = (
            bp_info.get("candidate_quote", "")
            if bp_info.get("quote_provenance") == "weak" else ""
        )
        weak_tokens = re.sub(r"[^\w\s]", " ", weak_quote.lower()).split()
        if no_verbatim_example and weak_tokens and len(weak_tokens) < 5:
            if " ".join(weak_tokens) in " ".join(stem_tokens):
                no_verbatim_example = False
                verbatim_label = "curriculum quote"
                verbatim_hit = " ".join(weak_tokens)
        if not no_verbatim_example:
            flags.append(
                f"❌ Quiz item '{target}': Stem verbatim from {verbatim_label} "
                f"(detected n-gram overlap: '{verbatim_hit}...')!"
            )

        # Stem Length Check
        if len(stem_words) < 4:
            flags.append(f"⚠️ Quiz item '{target}': Inadequate context stem (< 4 words)")

    # 5. In-List Distractor Recycling check
    recycled_in_distractors = []
    bp_for_item = prompt_blueprints.get(target.lower(), {}) if (prompt_blueprints and target) else {}
    if prompt_headwords and is_list_valid and valid_idx and not bp_for_item.get("prescribed_options"):
        clean_target = _clean_core(target) if target else ""
        clean_hw_set = {_clean_core(hw) for hw in prompt_headwords if hw}

        for o_idx, opt in enumerate(options):
            if o_idx == idx:
                continue
            opt_clean = str(opt).strip().lower()
            opt_core = _clean_core(opt_clean)

            if clean_target and (
                opt_core == clean_target
                or clean_target in opt_clean
                or (len(clean_target) >= 4 and opt_clean.startswith(clean_target[:4]))
            ):
                continue

            if opt_clean in prompt_headwords or (opt_core and opt_core in clean_hw_set and opt_core != clean_target):
                recycled_in_distractors.append(opt_clean)
            elif not is_translation:
                opt_lemma = LinguisticEngine.lemma_of(opt_clean) if " " not in opt_clean else opt_clean
                for hw in prompt_headwords:
                    hw_clean = str(hw).strip().lower()
                    if clean_target and (hw_clean == clean_target or _clean_core(hw_clean) == clean_target):
                        continue
                    hw_lemma = LinguisticEngine.lemma_of(hw_clean) if " " not in hw_clean else hw_clean
                    if (
                        (opt_lemma and hw_lemma and opt_lemma == hw_lemma)
                        or (" " not in hw_clean and " " not in opt_clean and LinguisticEngine.are_same_word_family(opt_clean, hw_clean))
                    ):
                        recycled_in_distractors.append(opt_clean)
                        break

    no_in_list_recycling = len(recycled_in_distractors) == 0

    # 6. CEFR Lexical Ceiling Gate
    if src_cefr_level and src_cefr_rank <= 3 and not bp_for_item.get("prescribed_options"):
        cefr_outliers = []
        for o_idx, opt in enumerate(options if isinstance(options, list) else []):
            if o_idx == idx:
                continue
            opt_w = str(opt).strip().lower()
            if (
                " " not in opt_w
                and opt_w.isalpha()
                and len(opt_w) >= 5
                and opt_w not in src_passage_words
                and opt_w not in prompt_headwords
            ):
                w_lvl = LinguisticEngine.get_word_cefr(opt_w)
                if w_lvl and LinguisticEngine.CEFR_ORDER.get(w_lvl, 0) >= max(5, src_cefr_rank + 2):
                    cefr_outliers.append(f"{opt_w} ({w_lvl})")
        if cefr_outliers:
            flags.append(
                f"⚠️ Quiz item '{target or question[:25]}': Distractor exceeds passage CEFR ceiling ({src_cefr_level}): {', '.join(cefr_outliers)}"
            )

    item_passed = (
        is_list_valid
        and has_no_duplicates
        and valid_idx
        and target_in_options
        and bool(explanation)
        and bool(question)
        and has_blank_when_expected
        and is_single_blank
        and no_stem_leak
        and no_scaffold_leak
        and no_article_leak
        and no_verb_inflection_clash
        and no_verbatim_example
        and no_in_list_recycling
        and options_match_blueprint
        and index_match_blueprint
        and no_comparative_defect
    )

    if item_passed:
        return True, flags

    reasons = []
    if not is_list_valid:
        reasons.append(f"options count != {expected_opt_count}")
    elif not has_no_duplicates:
        reasons.append("duplicate options detected")
    if not valid_idx:
        reasons.append("invalid answer index")
    if not target_in_options:
        reasons.append(f"target '{target}' not matching options[{idx}]")
    if not explanation:
        reasons.append("missing explanation")
    if not question:
        reasons.append("missing question text")
    if not no_scaffold_leak:
        reasons.append("prompt scaffolding leaked into question stem")
    if not has_blank_when_expected:
        reasons.append("missing fill-in-the-blank slot (____)")
    if not is_single_blank:
        reasons.append("multiple blanks detected in stem")
    if not no_stem_leak:
        reasons.append("target word leaks into question stem")
    if not no_article_leak:
        reasons.append("indefinite article leakage before blank")
    if not no_verb_inflection_clash:
        reasons.append("non-base verb inflection clash before blank")
    if not no_verbatim_example:
        reasons.append(f"stem verbatim from {verbatim_label}")
    if not no_comparative_defect:
        reasons.append(comp_defect_reason or "comparative translation defect")
    if not options_match_blueprint:
        reasons.append(f"prescribed options altered from blueprint {bp_info.get('prescribed_options')}")
    if not index_match_blueprint:
        reasons.append(f"prescribed answer index altered from blueprint index {bp_info.get('prescribed_index')}")

    if not no_in_list_recycling:
        reasons.append(f"recycles study list headwords in distractors {recycled_in_distractors}")
        flags.append(f"❌ Quiz item '{target or question[:25]}' in-list distractor recycling: {recycled_in_distractors}")
    elif not has_blank_when_expected:
        flags.append(f"❌ Quiz item '{target}': missing fill-in-the-blank slot (____)")
    elif not is_single_blank:
        flags.append(f"❌ Quiz item '{target}': Multiple blanks detected in stem")
    elif not no_stem_leak:
        flags.append(f"❌ Quiz item '{target}': Target word leaks into question stem: {leaked_words}")
    elif not no_article_leak:
        flags.append(f"❌ Quiz item '{target}': indefinite article leakage before blank")
    elif not no_verbatim_example:
        flags.append(f"❌ Quiz item '{target}': Stem verbatim from {verbatim_label}")
    else:
        flags.append(f"⚠️ Quiz question failed pedagogy check: {', '.join(reasons)}")

    return False, flags
