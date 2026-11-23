"""Inflection screening, voice tags, and target reconciliation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class DistractorInflectionsMixin:

    @classmethod
    def screen_inflected_options(
        cls,
        inflected: List[str],
        spare: List[str],
        final_target: str,
        count: int = 3,
        report: Optional[Dict[str, object]] = None,
    ) -> List[str]:
        """
        compliance gates must see the form the student actually reads.

        Distractors were screened before inflection; 'costliness' passed the length and
        CEFR gates, then 'pluralize_noun' turned it into 'costlinesses', which no gate
        ever looked at. Each inflected candidate is re-checked against the final target
        surface form; a failing candidate is replaced by the next one from the pool
        rather than repaired into another shape.
        """
        target = (final_target or "").strip().lower()
        pool: List[str] = []
        for cand in list(inflected) + list(spare):
            c = (cand or "").strip().lower()
            if c and c != target and c not in pool:
                pool.append(c)

        kept: List[str] = []
        soft_rejected: List[str] = []   # real words sitting at the wrong CEFR level
        hard_rejected: List[str] = []   # spellings that are not English word forms
        reasons: Dict[str, str] = {}
        for cand in pool:
            if not cls.is_attested_form(cand):
                hard_rejected.append(cand)
                reasons[cand] = "unattested_form"
                continue
            if not cls.is_cefr_compliant_distractor(cand, target):
                soft_rejected.append(cand)
                reasons[cand] = "cefr_ceiling"
                continue
            kept.append(cand)

        # A four-option item is a hard schema requirement, so a screened-out candidate is
        # replaced by another candidate - never by a repaired spelling. When the compliant
        # pool runs dry, padding prefers a real word at the wrong level over a spelling
        # that is not a word at all: the item degrades to 'too hard', not to 'wrong English'.
        if len(kept) < count:
            for cand in soft_rejected + hard_rejected:
                if len(kept) >= count:
                    break
                kept.append(cand)

        if report is not None:
            report.update({
                "target": target,
                "kept": list(kept),
                "rejected": reasons,
                "padded": len(kept) > len([k for k in kept if k not in soft_rejected + hard_rejected]),
            })
        return kept[:count]

    @classmethod
    def _example_target_tag(cls, example: Optional[str], target: str) -> Optional[str]:
        """
        Reads the verb form the authentic example actually uses for the target
        ('What annoyed him most...' -> VBD). Returns None when the target is absent.
        """
        ex = (example or "").strip()
        tw = (target or "").strip().lower()
        if not ex or not tw:
            return None
        nlp = cls.get_spacy()
        if not nlp:
            return None
        try:
            doc = nlp(ex)
        except Exception:
            return None
        for tok in doc:
            if tok.text.lower() == tw or tok.lemma_.lower() == tw:
                tag_up = tok.tag_.upper()
                if tag_up == "VBD" and any(
                    ch.dep_ in ("aux", "auxpass", "aux:pass") and ch.lemma_.lower() in ("be", "have", "get")
                    for ch in tok.children
                ):
                    return "VBN"
                return tag_up
        return None

    @classmethod
    def _is_verbal_phrase_passive_in_example(cls, example: Optional[str], target_phrase: str) -> bool:
        """
        Returns True when the verbal phrase (`nail down`, `brick up`, `chain up`, `pin down`)
        is used in passive voice (`VBN` with passive auxiliary `be`/`get` or `nsubjpass`/`auxpass`)
        in the anchoring example.
        """
        ex = (example or "").strip()
        tw_words = (target_phrase or "").strip().lower().split()
        if not ex or len(tw_words) < 2:
            return False
        head_w, tail_w = tw_words[0], tw_words[-1]
        nlp = cls.get_spacy()
        if not nlp:
            return False
        try:
            doc = nlp(ex)
        except Exception:
            return False
        head_forms = {head_w} | {f.lower() for f in cls.inflected_forms(head_w)}
        for tok in doc:
            if tok.text.lower() in head_forms or tok.lemma_.lower() == head_w:
                if not any(t.text.lower() == tail_w for t in doc[tok.i + 1 : min(len(doc), tok.i + 6)]):
                    continue
                if any(ch.dep_ in ("auxpass", "aux:pass", "nsubjpass", "nsubj:pass") for ch in tok.children):
                    return True
                if tok.tag_.upper() in ("VBN", "VBD") and any(
                    ch.dep_ == "aux" and ch.lemma_.lower() in ("be", "get")
                    for ch in tok.children
                ):
                    return True
        return False

    @classmethod
    def _verbal_phrase_target_tag(cls, example: Optional[str], target_phrase: str) -> Optional[str]:
        """
        Reads the verb inflection tag (VBN, VBD, VBZ, VBG, VB) that the authentic example
        actually uses for the verbal phrase (e.g. 'shut down', 'nail down', 'set out').
        Returns None when the verbal phrase or its head verb cannot be identified.
        """
        ex = (example or "").strip()
        tw_words = (target_phrase or "").strip().lower().split()
        if not ex or len(tw_words) < 2:
            return None
        head_w, tail_w = tw_words[0], tw_words[-1]
        nlp = cls.get_spacy()
        if not nlp:
            return None
        try:
            doc = nlp(ex)
        except Exception:
            return None
        head_forms = {head_w} | {f.lower() for f in cls.inflected_forms(head_w)}
        for tok in doc:
            if tok.text.lower() in head_forms or tok.lemma_.lower() == head_w:
                if not any(t.text.lower() == tail_w for t in doc[tok.i + 1 : min(len(doc), tok.i + 6)]):
                    continue
                tag_up = tok.tag_.upper()
                has_aux = (
                    any(ch.dep_ in ("auxpass", "aux:pass", "nsubjpass", "nsubj:pass") for ch in tok.children)
                    or any(
                        ch.dep_ in ("aux", "auxpass") and ch.lemma_.lower() in ("have", "be", "get")
                        for ch in tok.children
                    )
                )
                if tok.i > 0 and doc[tok.i - 1].text.lower() in (
                    "to", "will", "would", "could", "can", "might", "may", "should", "must", "did", "do", "does", "'ll", "'d"
                ):
                    return "VB"
                if tag_up == "VBG":
                    return "VBG"
                if has_aux:
                    return "VBN"
                if tag_up in ("VBD", "VBZ", "VB"):
                    return tag_up
                return tag_up
        return None

    @classmethod
    def _reconcile_prescribed_inflection(
        cls,
        base_options: List[str],
        target_index: int,
        target_word: str,
        canonical_pos: Optional[str],
        inflection_desc: str,
        authentic_example: Optional[str],
        quote: Optional[str] = None,
    ) -> Tuple[str, List[str], str, str]:
        """
        Prescribed-Inflection Concordance .

        The blueprint declares `inflection` and the LLM must obey it, so the declaration
        must come from the same evidence the item is built on. For verbs it was always
        'base form' even when the authentic PedLex example that supplied the frame read
        'What annoyed him most was that he had received no apology.' — a past-tense frame.
        The blueprint then demanded a base-form verb inside a past-tense cloze, which is
        unanswerable. When the anchoring example is inflected, every option is inflected to
        the identical form so frame, declared inflection and options agree.
        For multi-word idioms containing `one's`, synchronizes all 4 options to a concrete
        3rd-person possessive determiner (`her`, `his`, `their`, `its`) and matching verb tense.
        Returns (inflection_desc, options, target_word, form_tag).
        """
        pos_low = (canonical_pos or "").lower()
        tw_words = (target_word or "").strip().lower().split()
        if pos_low == "phrase" and "one's" in tw_words:
            poss_idx = tw_words.index("one's")
            re_parts: List[str] = []
            for idx_w, w_tok in enumerate(tw_words):
                if idx_w == poss_idx:
                    re_parts.append(r"(?P<poss>\b(?:my|your|his|her|its|our|their|one's|(?:(?:a|an|the)\s+)?[a-z]+['’]s)\b)")
                elif idx_w == 0:
                    f_forms = sorted({w_tok} | set(cls.inflected_forms(w_tok)), key=len, reverse=True)
                    re_parts.append(r"(?P<head>\b(?:" + "|".join(re.escape(f) for f in f_forms if f) + r")\b)")
                elif idx_w == len(tw_words) - 1:
                    l_forms = sorted({w_tok} | set(cls.inflected_forms(w_tok)), key=len, reverse=True)
                    re_parts.append(r"\b(?:" + "|".join(re.escape(f) for f in l_forms if f) + r")\b")
                else:
                    re_parts.append(rf"\b{re.escape(w_tok)}\b")
            idiom_re = re.compile(r"\s+".join(re_parts), flags=re.IGNORECASE)
            m_quote = idiom_re.search(quote or "")
            m_ex = idiom_re.search(authentic_example or "")

            resolved_poss = None
            for m_cand in (m_quote, m_ex):
                if m_cand:
                    p_val = m_cand.group("poss").lower()
                    if p_val in ("her", "his", "their", "its"):
                        resolved_poss = p_val
                        break
            combined_ctx = f"{quote or ''} {authentic_example or ''}"
            if not resolved_poss:
                if re.search(r"\b(?:she|her|woman|mother|girl|sister|wife|daughter)\b", combined_ctx, re.IGNORECASE):
                    resolved_poss = "her"
                elif re.search(r"\b(?:he|his|him|man|father|boy|brother|husband|son)\b", combined_ctx, re.IGNORECASE):
                    resolved_poss = "his"
                elif re.search(r"\b(?:they|their|them)\b", combined_ctx, re.IGNORECASE):
                    resolved_poss = "their"
                else:
                    resolved_poss = "his"

            # Singular countable head-noun agreement guard: avoid 'a roof over their head' / 'a feather in their cap'
            if resolved_poss == "their":
                has_singular_head = tw_words[0] in ("a", "an")
                if has_singular_head:
                    resolved_poss = (
                        "her"
                        if re.search(r"\b(?:she|her|woman|mother|girl|sister|wife|daughter)\b", combined_ctx, re.IGNORECASE)
                        else "his"
                    )

            head_w = tw_words[0]
            _head_ent = cls.get_ldoce_entry(head_w)
            _head_pos = str((_head_ent or {}).get("pos") or "").strip().lower()
            is_verbal_idiom = (
                head_w not in {"a", "an", "the", "in", "on", "at", "by", "for", "with", "from", "to", "of", "over", "under"}
                and _head_pos not in ("noun", "adjective", "adverb", "preposition", "determiner")
                and bool(cls._wordnet_pos(head_w) & {"v"})
            )

            tag = "VB"
            if is_verbal_idiom:
                nlp = cls.get_spacy()

                def _detect_idiom_verb_tag(text_src: Optional[str], m_obj: Optional[re.Match]) -> Optional[str]:
                    if not text_src or not m_obj:
                        return None
                    prefix_str = text_src[: m_obj.start()].strip().lower()
                    if re.search(
                        r"(?:\b(?:to|will|would|can|could|may|might|should|must|do|does|did|let|help|make)|'ll|'d)\s*$",
                        prefix_str,
                    ):
                        return "VB"
                    if nlp:
                        try:
                            doc = nlp(text_src)
                            head_start = m_obj.start("head")
                            for tok in doc:
                                if tok.idx == head_start:
                                    t_tag = tok.tag_.upper()
                                    if t_tag == "VBN":
                                        if re.search(
                                            r"(?:\b(?:have|has|had|having|be|am|is|are|was|were|been|being|get|gets|got|gotten)|'ve|'s|'d)\s+(?:[a-z]+ly\s+)?$",
                                            prefix_str,
                                        ):
                                            return "VBN"
                                        return "VBD"
                                    if t_tag in ("VBD", "VBZ", "VBG"):
                                        return t_tag
                                    return "VB"
                        except Exception:
                            pass
                    head_surf = m_obj.group("head").lower()
                    for cand_tag in ("VBD", "VBZ", "VBG"):
                        if head_surf == (cls._inflect_verb(head_w, cand_tag) or "").lower() and head_surf != head_w:
                            return cand_tag
                    return "VB"

                tag = _detect_idiom_verb_tag(authentic_example, m_ex) or _detect_idiom_verb_tag(quote, m_quote) or "VB"

            reconciled_opts: List[str] = []
            if is_verbal_idiom and tag in ("VBD", "VBZ", "VBG", "VBN"):
                temp_opts: List[str] = []
                all_ok = True
                for opt in base_options:
                    ow = opt.strip().split()
                    if not ow:
                        all_ok = False
                        break
                    v_inf = cls._inflect_verb(ow[0].lower(), tag)
                    if not v_inf:
                        all_ok = False
                        break
                    inf_phrase = " ".join([v_inf] + ow[1:])
                    inf_phrase = re.sub(r"\b(?:one's|your)\b", resolved_poss, inf_phrase, flags=re.IGNORECASE)
                    temp_opts.append(inf_phrase)
                if all_ok and len(temp_opts) == len(base_options):
                    reconciled_opts = temp_opts
                else:
                    tag = "VB"

            if not reconciled_opts:
                reconciled_opts = [
                    re.sub(r"\b(?:one's|your)\b", resolved_poss, opt.strip(), flags=re.IGNORECASE)
                    for opt in base_options
                ]

            new_target = reconciled_opts[target_index] if 0 <= target_index < len(reconciled_opts) else target_word
            new_desc = (
                f"{cls._VERB_FORM_LABELS.get(tag, 'base form')} (with '{resolved_poss}')"
                if is_verbal_idiom
                else f"exact phrase form (with '{resolved_poss}')"
            )
            return new_desc, reconciled_opts, new_target, (tag if is_verbal_idiom else None)

        if pos_low == "phrase" and len(tw_words) >= 2 and inflection_desc == "base form":
            head_w = tw_words[0]
            tail_w = tw_words[-1]
            _head_ent = cls.get_ldoce_entry(head_w)
            _head_pos = str((_head_ent or {}).get("pos") or "").strip().lower()
            is_verbal_phrase = (
                head_w not in {"a", "an", "the", "in", "on", "at", "by", "for", "with", "from", "to", "of", "over", "under", "as", "even", "so", "now"}
                and _head_pos not in ("adjective", "adverb", "preposition", "determiner")
                and bool(cls._wordnet_pos(head_w) & {"v"})
            )
            if is_verbal_phrase:
                pv_tag = cls._verbal_phrase_target_tag(authentic_example, target_word)
                if not pv_tag and quote:
                    pv_tag = cls._verbal_phrase_target_tag(quote, target_word)
                if pv_tag in ("VBN", "VBD", "VBZ", "VBG"):
                    inflected_pv: List[str] = []
                    all_pv_ok = True
                    for opt in base_options:
                        ow = opt.strip().split()
                        if len(ow) < 2:
                            all_pv_ok = False
                            break
                        ow_base = ow[0].lower() if cls._is_base_verb_lemma(ow[0].lower()) else (cls.lemma_of(ow[0].lower()) or ow[0].lower())
                        v_form = cls._inflect_verb(ow_base, pv_tag)
                        if not v_form or not cls._morph_form_valid(ow_base, v_form, pv_tag):
                            all_pv_ok = False
                            break
                        inflected_pv.append(" ".join([v_form] + ow[1:]))
                    if all_pv_ok and len(inflected_pv) == len(base_options):
                        new_pv_target = inflected_pv[target_index] if 0 <= target_index < len(inflected_pv) else target_word
                        return cls._VERB_FORM_LABELS.get(pv_tag, f"{pv_tag} form"), inflected_pv, new_pv_target, pv_tag

        if pos_low != "verb":
            return inflection_desc, base_options, target_word, None
        if inflection_desc != "base form":
            return inflection_desc, base_options, target_word, "VB"
        tag = cls._example_target_tag(authentic_example, target_word)
        if tag not in ("VBD", "VBZ", "VBG", "VBN") and not cls._is_base_verb_lemma(target_word):
            if quote:
                tag = cls._example_target_tag(quote, target_word)
            if tag not in ("VBD", "VBZ", "VBG", "VBN"):
                tw_base = cls.lemma_of(target_word)
                if tw_base and tw_base != target_word.lower():
                    for cand_tag in ("VBD", "VBG", "VBZ", "VBN"):
                        if (cls._inflect_verb(tw_base, cand_tag) or "").lower() == target_word.lower():
                            tag = cand_tag
                            break
        if tag not in ("VBD", "VBZ", "VBG", "VBN"):
            return inflection_desc, base_options, target_word, "VB"
        inflected: List[str] = []
        for opt in base_options:
            opt_base = opt if cls._is_base_verb_lemma(opt) else (cls.lemma_of(opt) or opt)
            form = cls._inflect_verb(opt_base, tag)
            if not form or not cls._morph_form_valid(opt_base, form, tag):
                return inflection_desc, base_options, target_word, "VB"  # unsafe to rewrite
            inflected.append(form)
        new_target = inflected[target_index] if 0 <= target_index < len(inflected) else target_word

        # Inflection Suffix Symmetry Guard:
        # If target verb is past tense / participle ending in '-ed', distractors whose past form
        # equals their base lemma (zero-marked irregulars e.g. 'cost', 'cast', 'hit', 'cut', 'put', 'set')
        # cause morphological confusion. Replace zero-marked distractors with overt past forms.
        if tag in ("VBD", "VBN") and new_target.endswith("ed"):
            has_zero_marked = any(
                idx != target_index and inflected[idx].lower() == (base_options[idx] if cls._is_base_verb_lemma(base_options[idx]) else (cls.lemma_of(base_options[idx]) or base_options[idx])).lower()
                for idx in range(len(inflected))
            )
            if has_zero_marked:
                fb_verbs = cls.get_corpus_fallback_candidates(
                    target_word=target_word, pos="v", target_count=10, exclude_words=set(base_options) | set(inflected) | {target_word}
                )
                fb_idx = 0
                for idx in range(len(inflected)):
                    if idx == target_index:
                        continue
                    b_lem = base_options[idx] if cls._is_base_verb_lemma(base_options[idx]) else (cls.lemma_of(base_options[idx]) or base_options[idx])
                    if inflected[idx].lower() == b_lem.lower():
                        while fb_idx < len(fb_verbs):
                            cand_v = fb_verbs[fb_idx]
                            fb_idx += 1
                            cand_inf = cls._inflect_verb(cand_v, tag)
                            if cand_inf and cand_inf.endswith("ed") and cand_inf.lower() not in [x.lower() for x in inflected]:
                                inflected[idx] = cand_inf
                                break

        # A1: an inflected target that is not an attested English form must never ship,
        # even when spaCy was willing to lemmatize it back to the base verb.
        if not cls.is_attested_form(new_target):
            return inflection_desc, base_options, target_word, "VB"

        # Lexical POS Boundary Guard: reject inflection ONLY if the inflected form is an independent
        # dictionary headword entry (`word == new_target`) with a non-verb POS (e.g. annoy[verb] -> annoyed[adjective entry]),
        # or has an independent adjective homograph (e.g. swell[verb] -> swollen[adjective entry]).
        # Note: get_ldoce_entry('pledged') falls back to the lemma entry 'pledge' (homograph 1 = noun),
        # whose `word` is 'pledge', NOT 'pledged' — that is a lemmatized fallback, not a dedicated participial-adjective entry!
        _inflected_entry = cls.get_ldoce_entry(new_target)
        if (
            _inflected_entry
            and _inflected_entry.get("kind") == "article"
            and (_inflected_entry.get("word") or "").strip().lower() == new_target.lower()
        ):
            _inflected_pos = (_inflected_entry.get("pos") or "").lower()
            all_poses = [p.lower() for p in _inflected_entry.get("all_poses", [])]
            has_adj = "adjective" in all_poses or any(
                (h.get("pos") or "").lower() == "adjective" for h in _inflected_entry.get("homographs", [])
            )
            if (_inflected_pos and _inflected_pos not in ("verb", "v")) or (tag == "VBN" and has_adj):
                return inflection_desc, base_options, target_word, "VB"

        return cls._VERB_FORM_LABELS.get(tag, "base form"), inflected, new_target, tag

    @classmethod
    def inflect_verb_like_target(cls, cand_base: str, target_word: str, base_verb: str) -> str:
        """Morphologically inflects `cand_base` to match the exact surface form/tag of `target_word`."""
        t_clean = (target_word or "").strip().lower()
        b_clean = (base_verb or "").strip().lower()
        c_clean = (cand_base or "").strip().lower()
        if not t_clean or not c_clean or t_clean == b_clean:
            return c_clean

        tag = cls.verb_form_tag_of(t_clean)
        if not tag:
            if t_clean.endswith("s") and (t_clean in (b_clean + "s", b_clean + "es") or b_clean in ("have", "be", "do")):
                tag = "VBZ"
            elif t_clean.endswith("ing"):
                tag = "VBG"
            elif t_clean.endswith("ed"):
                tag = "VBD"

        if tag:
            inf = cls.verb_form_for_tag(c_clean, tag) or cls._inflect_verb(c_clean, tag)
            if inf:
                return inf

        return c_clean
