"""
Cloze slot frame derivation from authentic dictionary examples.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


class QuizFramesSlotSlicingMixin:
    _NON_GRADABLE_ADJS = frozenset({
        "regulatory", "compensatory", "rival", "primary", "secondary", "chief", "major", "minor",
        "senior", "junior", "initial", "final", "previous", "current", "future", "former", "main",
        "only", "same", "single", "double", "triple", "unique", "impossible", "alive", "dead",
        "perfect", "total", "whole", "entire", "exact", "equal", "annual", "daily", "weekly",
        "monthly", "electric", "electronic", "digital", "online", "offline", "wooden", "plastic",
        "metal", "golden", "nuclear", "solar", "medical", "legal", "financial", "economic",
        "local", "national", "international", "public", "private", "ancient", "historical",
    })

    @classmethod
    def derive_cloze_slot_frame_from_example(
        cls,
        example_text: Optional[str],
        final_target: str,
        base_word: str,
        canonical_pos: str,
        anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        is_copula_adj: bool = False,
        is_attributive_only_adj: bool = False,
        is_plural_noun: bool = False,
        is_pure_uncountable_target: bool = False,
        is_target_strictly_countable: bool = False,
        is_singular_only_noun: bool = False,
        has_mixed_vc_initials: bool = False,
        has_uncountable_contrast_foil: bool = False,
        verb_form_tag: Optional[str] = None,
        forbidden_words: Optional[Set[str]] = None,
        patient_ontology_slot: Optional[str] = None,
    ) -> Optional[str]:
        """
        Slices a compact, authentic 2-to-5-word `cloze_slot_frame` directly from the winning example's
        syntactic neighborhood around the target word, guaranteeing at least one concrete left or right
        context word/POS anchor (`... their ____ achievement`, `... take out a ____`, `... ____ to [verb]`,
        `... ____ for`, `... is ____ in`) and zero bare `None` frames.
        """
        if not example_text or not final_target:
            return None
        nlp = cls.get_spacy()
        if not nlp:
            return None

        ex_clean = re.sub(r"\s*\(=[^)]*\)", "", str(example_text)).strip()
        ft_low = final_target.strip().lower()
        bw_low = (base_word or ft_low).strip().lower()
        t_forms = {ft_low, bw_low} | set(cls.inflected_forms(bw_low)) | set(cls.inflected_forms(ft_low))

        def _is_forbidden(word_str: str) -> bool:
            if not word_str or not forbidden_words:
                return False
            return cls._in_forbidden_word_set(word_str.lower(), forbidden_words)

        def _normalize_det(det_str: str, det_tok_obj=None) -> str:
            if det_tok_obj is not None and det_tok_obj.dep_ == "poss" and det_tok_obj.pos_ in ("NOUN", "PROPN", "ADJ"):
                return "their"
            d_l = (det_str or "").strip().lower()
            if not d_l:
                return ""
            if d_l.endswith("'s") or d_l.endswith("’s"):
                return "their"
            if canonical_pos == "noun":
                if is_pure_uncountable_target or is_plural_noun:
                    if d_l in ("a", "an", "each", "every", "one"):
                        if is_plural_noun:
                            return "several" if has_uncountable_contrast_foil else "the"
                        return ""
                    if is_plural_noun and has_uncountable_contrast_foil and d_l in ("the", "some", "any"):
                        return "several"
                elif has_uncountable_contrast_foil and not is_plural_noun:
                    if d_l in ("a", "an", "the"):
                        if is_singular_only_noun:
                            return "the" if has_mixed_vc_initials else "a"
                        return "each" if has_mixed_vc_initials else "a"
                elif has_mixed_vc_initials and d_l in ("a", "an"):
                    return "each" if (has_uncountable_contrast_foil and not is_singular_only_noun) else "the"
            elif canonical_pos == "adj":
                if has_mixed_vc_initials and d_l in ("a", "an"):
                    is_non_gradable = (
                        is_attributive_only_adj
                        or bw_low in cls._NON_GRADABLE_ADJS
                        or ft_low in cls._NON_GRADABLE_ADJS
                    )
                    return "the" if is_non_gradable else "a very"
            return d_l

        def _tok_surface(t) -> str:
            """Preserve original casing for proper nouns and acronyms; lowercase common words."""
            if t is None:
                return ""
            if t.pos_ == "PROPN" or (t.i > 0 and t.text and t.text[0].isupper()):
                return t.text
            return t.text.lower()

        def _pobj_phrase(prep_tok, max_words: int = 5) -> str:
            """Extract a concise, non-forbidden prepositional object NP to anchor bare prep frames."""
            if prep_tok is None:
                return ""
            pobj = next(
                (c for c in prep_tok.children if c.dep_ == "pobj" and c.pos_ in ("NOUN", "PROPN", "PRON")),
                None,
            )
            if pobj is None:
                pcomp = next((c for c in prep_tok.children if c.dep_ == "pcomp" and c.pos_ == "VERB"), None)
                if pcomp is not None and not _is_forbidden(pcomp.lemma_.lower()) and pcomp.lemma_.lower() not in cls._DELEXICAL_VERBS:
                    return f" {_tok_surface(pcomp)}"
                return ""
            if pobj.pos_ == "PRON" or pobj.ent_type_ == "PERSON" or _is_forbidden(pobj.lemma_.lower()):
                is_human = (
                    pobj.text.lower() in ("him", "her", "me", "us", "you", "them", "someone", "somebody", "anyone", "everyone")
                    or pobj.ent_type_ == "PERSON"
                )
                return " [someone]" if is_human else " [something]"
            parts: List[str] = []
            det_c = next((c for c in pobj.children if c.dep_ in ("det", "poss") and prep_tok.i < c.i < pobj.i), None)
            if det_c is not None:
                d_txt = det_c.text.strip().lower()
                if d_txt.endswith("'s") or d_txt.endswith("’s") or (det_c.dep_ == "poss" and det_c.pos_ in ("NOUN", "PROPN")):
                    parts.append("their")
                elif d_txt in ("a", "an", "the", "this", "that", "these", "those", "some", "any", "my", "your", "his", "her", "its", "our", "their"):
                    parts.append(d_txt)
            mod_cs = [
                c for c in pobj.children
                if c.dep_ in ("amod", "compound")
                and prep_tok.i < c.i < pobj.i
                and c.is_alpha
                and not _is_forbidden(c.lemma_.lower())
            ]
            mod_cs.sort(key=lambda c: c.i)
            for c in mod_cs:
                parts.append(_tok_surface(c))
            parts.append(_tok_surface(pobj))
            return " " + " ".join(parts[-max_words:])

        try:
            doc = nlp(ex_clean)
        except Exception:
            return None

        _cop_contraction_map = {
            "'s": "is", "’s": "is",
            "'re": "are", "’re": "are",
            "'m": "am", "’m": "am",
            "'ll": "will", "’ll": "will",
            "'d": "would", "’d": "would",
        }

        # 1. Multi-word phrase / expression
        if " " in ft_low or canonical_pos == "phrase":
            ft_clean_for_toks = re.sub(
                r"\b(?:one's|oneself|someone's|somebody's|my|your|his|her|its|our|their)\b",
                " ",
                ft_low,
            )
            ft_toks = [t for t in re.findall(r"[a-z]+", ft_clean_for_toks) if t not in ("sb", "sth", "somebody", "something", "one", "ones")]
            if ft_toks:
                # 1.0 Passive 3-Word Verbal Phrase ('be equipped with', 'be associated with', 'be based on')
                is_passive_3w_phrase = (
                    len(ft_toks) >= 3
                    and ft_toks[0] in ("be", "get", "feel")
                    and (ft_toks[1].endswith("ed") or ft_toks[1].endswith("en") or cls.lemma_of(ft_toks[1]) != ft_toks[1])
                )
                if is_passive_3w_phrase:
                    base_v = cls.lemma_of(ft_toks[1])
                    prep_w = ft_toks[-1]
                    v_tok = None
                    for tok in doc:
                        if (tok.lemma_.lower() == base_v or tok.text.lower() == ft_toks[1]) and tok.pos_ in ("VERB", "AUX"):
                            v_tok = tok
                            break
                    p_tok = None
                    if v_tok:
                        for tok in doc:
                            if tok.i > v_tok.i and tok.text.lower() == prep_w and tok.pos_ in ("ADP", "SCONJ"):
                                p_tok = tok
                                break
                    if v_tok and p_tok:
                        is_ex_passive = (
                            any(c.dep_ in ("auxpass", "nsubjpass") for c in v_tok.children)
                            or (v_tok.i > 0 and doc[v_tok.i - 1].text.lower() in ("is", "are", "was", "were", "been", "be", "get", "got"))
                        )
                        prep_anim = cls._resolve_object_animacy_from_example(ex_clean, prep_w, default_tag="something")
                        if is_ex_passive:
                            pass_subj = next((c for c in v_tok.children if c.dep_ in ("nsubjpass", "nsubj") and c.i < v_tok.i), None)
                            s_text = pass_subj.text.lower() if pass_subj else "something"
                            is_human = (
                                cls._resolve_object_animacy_from_example(ex_clean, base_v) == "someone"
                                or s_text in ("students", "teachers", "people", "student", "teacher", "children", "workers")
                            )
                            slot_label = f"[someone, e.g. {s_text}]" if is_human else f"[something, e.g. {s_text}]"
                            aux_toks = [c.text.lower() for c in v_tok.children if (c.dep_ in ("aux", "auxpass") or c.pos_ == "AUX") and c.i < v_tok.i]
                            aux_str = (" ".join(aux_toks) + " ") if aux_toks else ("will " if ft_toks[0] == "be" else "were/are ")
                            if ft_toks[0] == "be" and any(w in aux_str for w in ("is", "are", "was", "were", "been")):
                                aux_str = "will "
                            return f"... {slot_label} {aux_str}____ {prep_anim}"
                        else:
                            # Active diathesis: Direct object of base_v is the Patient (becomes passive subject)
                            dobj = next((c for c in v_tok.children if c.dep_ in ("dobj", "obj")), None)
                            if not dobj:
                                for tok in doc[v_tok.i + 1 : p_tok.i]:
                                    if tok.pos_ in ("NOUN", "PROPN", "PRON"):
                                        dobj = tok
                                        break
                            d_text = dobj.text.lower() if dobj else "someone"
                            is_human = (
                                cls._resolve_object_animacy_from_example(ex_clean, base_v) == "someone"
                                or d_text in ("students", "teachers", "people", "student", "teacher", "children", "workers")
                            )
                            slot_label = f"[someone, e.g. {d_text}]" if is_human else f"[something, e.g. {d_text}]"
                            aux_str = "will " if ft_toks[0] == "be" else "were "
                            return f"... {slot_label} {aux_str}____ {prep_anim}"

                first_lem = nlp(ft_toks[0])[0].lemma_.lower() if nlp else ft_toks[0]
                first_forms = {ft_toks[0], first_lem} | set(cls.inflected_forms(ft_toks[0])) | set(cls.inflected_forms(first_lem))
                last_w = ft_toks[-1]
                start_i, end_i = None, None
                for tok in doc:
                    if tok.text.lower() in first_forms or tok.lemma_.lower() in (ft_toks[0], first_lem):
                        if ft_toks[0] in ("a", "an", "the") and len(ft_toks) >= 2:
                            if tok.i + 1 >= len(doc) or (
                                doc[tok.i + 1].text.lower() != ft_toks[1]
                                and doc[tok.i + 1].lemma_.lower() != ft_toks[1]
                            ):
                                continue
                        for j in range(tok.i + len(ft_toks) - 1, min(len(doc), tok.i + len(ft_toks) + 4)):
                            if doc[j].text.lower() == last_w or doc[j].lemma_.lower() == last_w:
                                start_i, end_i = tok.i, j + 1
                                break
                        if start_i is not None:
                            break
                if start_i is not None and end_i is not None:
                    _head_ent = cls.get_ldoce_entry(ft_toks[0])
                    _head_primary_pos = str((_head_ent or {}).get("pos") or "").strip().lower()
                    _is_binomial_or_of_np = (
                        (
                            len(ft_toks) == 3
                            and ft_toks[1] in ("and", "or", "by", "to")
                            and (
                                ft_toks[0] == ft_toks[2]
                                or ft_toks[0] in ("back", "up", "down", "in", "out", "on", "off", "more", "sooner", "now", "first", "day", "time", "side", "face", "hand", "step", "word", "pros", "ups", "odds", "ins")
                            )
                        )
                        or (len(ft_toks) >= 3 and ft_toks[1] == "of")
                    )
                    is_verbal_phrase = (
                        not _is_binomial_or_of_np
                        and (
                            doc[start_i].pos_ == "VERB"
                            or (
                                bool(cls._wordnet_pos(ft_toks[0]) & {"v"})
                                and _head_primary_pos not in ("noun", "adjective", "adverb", "preposition", "determiner")
                            )
                        )
                    )
                    if verb_form_tag is None and is_verbal_phrase and doc[start_i].tag_ in ("VBG", "VBN", "VBD", "VBZ", "VBP", "VB"):
                        verb_form_tag = doc[start_i].tag_
                    left_aux = ""
                    if is_verbal_phrase and start_i > 0 and verb_form_tag not in ("VBD", "VBZ", "VBG", "VBN"):
                        prev_raw = doc[start_i - 1].text.lower()
                        if prev_raw == "'ll":
                            left_aux = "will "
                        elif prev_raw == "'d":
                            left_aux = "would "
                        elif prev_raw in ("to", "will", "would", "could", "can", "might", "may", "should", "must", "did", "do", "does"):
                            if prev_raw == "to" and start_i > 1:
                                prev_verb = doc[start_i - 2]
                                if (
                                    prev_verb.pos_ in ("VERB", "ADJ", "NOUN")
                                    and prev_verb.is_alpha
                                    and not _is_forbidden(prev_verb.lemma_.lower())
                                ):
                                    left_aux = f"{prev_verb.text.lower()} to "
                                else:
                                    left_aux = "to "
                            else:
                                left_aux = f"{prev_raw} "
                    # Copular Adjective Phrase: when phrase head is an adjective followed by a preposition
                    # (e.g. 'full of', 'short of', 'proud of', 'fond of'), recover the finite copula from the example
                    _head_is_adj = not is_verbal_phrase and not (ft_toks[0].endswith("ed") or ft_toks[0].endswith("en")) and (
                        _head_primary_pos == "adjective" or bool(cls._wordnet_pos(ft_toks[0]) & {"a", "s"})
                    )
                    if _head_is_adj and not left_aux:
                        target_tok = doc[start_i]
                        cop_word = None
                        if target_tok.head.pos_ in ("AUX", "VERB") and target_tok.head.lemma_.lower() in ("be", "seem", "feel", "become", "look", "sound", "taste", "smell", "remain", "stay") and target_tok.head.i < start_i:
                            cop_word = target_tok.head.text.lower()
                        if not cop_word:
                            for c in target_tok.children:
                                if (c.dep_ == "cop" or (c.dep_ == "aux" and c.lemma_.lower() in ("be", "seem", "feel", "become", "look", "sound", "remain", "stay"))) and c.i < start_i:
                                    cop_word = c.text.lower()
                                    break
                        if not cop_word:
                            for anc in target_tok.ancestors:
                                if anc.pos_ in ("AUX", "VERB") and anc.lemma_.lower() in ("be", "seem", "feel", "become", "look", "sound", "taste", "smell", "remain", "stay") and anc.i < start_i:
                                    cop_word = anc.text.lower()
                                    break
                        if not cop_word:
                            cop_word = "was"
                        cop_word = _cop_contraction_map.get(cop_word, cop_word)
                        left_aux = f"{cop_word} "

                    # Check if a direct object sits inside a separable phrasal verb ('turned the whole house over', 'wake you up')
                    mid_toks = doc[start_i + 1 : end_i - 1]
                    if len(ft_toks) == 2 and last_w in ("up", "down", "out", "in", "off", "on", "away", "back", "over", "through", "around", "along", "aside", "apart") and any(t.pos_ in ("PRON", "NOUN", "PROPN") for t in mid_toks):
                        anim = patient_ontology_slot or cls._resolve_object_animacy_from_example(ex_clean, ft_toks[0], default_tag="[sb/sth]")
                        return f"... {left_aux}____ {anim}"
                    # If the phrase ends with a transitive preposition ('set out on', 'catch up with', 'listen to', 'take pleasure in')
                    if last_w in ("with", "to", "for", "of", "into", "from", "on", "at", "about", "against", "upon", "over", "after") or (last_w == "in" and len(ft_toks) >= 3):
                        anim = patient_ontology_slot or cls._resolve_object_animacy_from_example(ex_clean, ft_low, default_tag="[sb/sth]")
                        return f"... {left_aux}____ {anim}"
                    # Check if the phrase starts with a verb and takes a direct object or prepositional complement right after `end_i`
                    if is_verbal_phrase and end_i < len(doc):
                        nxt_t = doc[end_i]
                        _is_2w_adv_pv = (
                            len(ft_toks) == 2
                            and last_w in ("up", "down", "out", "in", "off", "on", "away", "back", "over", "through", "around", "along", "aside", "apart")
                        )
                        # Prepositional complement / locative adjunct:
                        # Find adjacent or closely-following prepositional child (e.g. 'lie down on the bed', 'look back on those days')
                        nxt_prep = None
                        if (
                            nxt_t.pos_ == "ADP"
                            and nxt_t.text.lower() not in ("between", "during", "since", "until", "before", "after", "throughout")
                        ):
                            nxt_prep = nxt_t
                        elif _is_2w_adv_pv:
                            nxt_prep = next(
                                (
                                    c for c in doc[start_i].children
                                    if c.dep_ == "prep"
                                    and c.i >= end_i
                                    and c.pos_ == "ADP"
                                    and c.text.lower() not in ("between", "during", "since", "until", "before", "after", "throughout")
                                ),
                                None,
                            )
                        if nxt_prep is not None:
                            pobj_tail = _pobj_phrase(nxt_prep)
                            if not _is_2w_adv_pv or nxt_prep.text.lower() == str(anchor or "").strip().lower() or pobj_tail:
                                return f"... {left_aux}____ {nxt_prep.text.lower()}{pobj_tail}".strip()
                        has_dobj_after = (
                            nxt_t.dep_ in ("dobj", "obj")
                            or (
                                nxt_t.pos_ in ("DET", "PRON", "NOUN", "PROPN")
                                and nxt_t.text.lower() not in ("all", "every", "each", "both", "today", "yesterday", "tomorrow")
                                and any(t.dep_ in ("dobj", "obj") for t in doc[end_i : min(len(doc), end_i + 4)])
                            )
                        )
                        if has_dobj_after:
                            anim = patient_ontology_slot or cls._resolve_object_animacy_from_example(ex_clean, ft_low, default_tag="[sb/sth]")
                            return f"... {left_aux}____ {anim}"
                    if is_verbal_phrase:
                        is_pv_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in doc[start_i].children)
                        if not is_pv_passive and start_i > 0 and doc[start_i - 1].text.lower() in ("was", "were", "is", "are", "been", "be", "got", "get"):
                            is_pv_passive = True
                        pv_pass_subj = next(
                            (
                                c for c in doc[start_i].children
                                if c.dep_ in ("nsubjpass", "nsubj") and c.i < start_i and c.pos_ in ("NOUN", "PROPN") and c.is_alpha and not _is_forbidden(c.lemma_.lower())
                            ),
                            None,
                        )
                        # Passive-Preserving Frame for VBN verbal phrases ('The lid was firmly nailed down.', 'The elephants were chained up by their legs.'):
                        if is_pv_passive and verb_form_tag == "VBN":
                            aux_start = pv_pass_subj.i + 1 if pv_pass_subj is not None else max(0, start_i - 3)
                            aux_toks = [
                                t.text.lower()
                                for t in doc[aux_start : start_i]
                                if t.pos_ == "AUX"
                                or t.dep_ in ("aux", "auxpass", "neg")
                                or t.text.lower() in ("was", "were", "is", "are", "been", "be", "has", "have", "had", "got", "get")
                            ]
                            mid_pv_aux = (" ".join(aux_toks) + " ") if aux_toks else "was/were "
                            if patient_ontology_slot:
                                mid_pv_aux_norm = re.sub(r"\b(?:was|were)\b", "was/were", mid_pv_aux)
                                mid_pv_aux_norm = re.sub(r"\b(?:is|are)\b", "is/are", mid_pv_aux_norm)
                                mid_pv_aux_norm = re.sub(r"\b(?:has|have)\b", "has/have", mid_pv_aux_norm)
                                return f"... {patient_ontology_slot} {mid_pv_aux_norm}____"
                            if pv_pass_subj is not None:
                                s_low = pv_pass_subj.text.lower()
                                det_cands = [d for d in pv_pass_subj.children if d.dep_ in ("det", "poss") and d.i < pv_pass_subj.i and d.is_alpha]
                                if det_cands:
                                    d_low = det_cands[-1].text.lower()
                                    if d_low in ("my", "your", "our"):
                                        d_low = "a" if pv_pass_subj.tag_ == "NN" else "the"
                                    return f"... {d_low} {s_low} {mid_pv_aux}____"
                                if pv_pass_subj.tag_ == "NN":
                                    return f"... a {s_low} {mid_pv_aux}____"
                                return f"... {s_low} {mid_pv_aux}____"
                        # Deep Passive Projection fallback when options are in base form ('nail down'):
                        if is_pv_passive and verb_form_tag not in ("VBD", "VBZ", "VBG", "VBN"):
                            if patient_ontology_slot:
                                return f"... {left_aux or 'to '}____ {patient_ontology_slot}"
                            if pv_pass_subj is not None:
                                s_low = pv_pass_subj.text.lower()
                                det_cands = [d for d in pv_pass_subj.children if d.dep_ in ("det", "poss") and d.i < pv_pass_subj.i and d.is_alpha]
                                if det_cands:
                                    d_low = det_cands[-1].text.lower()
                                    if d_low in ("my", "your", "our"):
                                        d_low = "a" if pv_pass_subj.tag_ == "NN" else "the"
                                    obj_np = f"{d_low} {s_low}"
                                elif pv_pass_subj.tag_ == "NN":
                                    obj_np = f"the {s_low}"
                                else:
                                    obj_np = s_low
                                return f"... {left_aux or 'to '}____ {obj_np}"
                        if left_aux:
                            return f"... {left_aux}____"
                        if verb_form_tag in ("VBD", "VBZ", "VBP", "VBG", "VBN"):
                            if end_i < len(doc) and doc[end_i].is_alpha and not _is_forbidden(doc[end_i].lemma_.lower()):
                                return f"... ____ {doc[end_i].text.lower()}"
                            subj_cands = [
                                c for c in doc[start_i].children
                                if c.dep_ in (("nsubj", "nsubjpass") if verb_form_tag == "VBN" else ("nsubj",)) and c.i < start_i and c.is_alpha
                            ]
                            subj_tok = subj_cands[-1] if subj_cands else (
                                doc[start_i - 1]
                                if not is_pv_passive and start_i > 0 and doc[start_i - 1].is_alpha and doc[start_i - 1].pos_ in ("NOUN", "PROPN", "PRON")
                                else None
                            )
                            if subj_tok and subj_tok.pos_ in ("NOUN", "PROPN") and not _is_forbidden(subj_tok.lemma_.lower()):
                                s_low = subj_tok.text.lower()
                                mid_pv_aux = ""
                                if verb_form_tag == "VBN" and subj_tok.i + 1 < start_i:
                                    aux_toks = [
                                        t.text.lower()
                                        for t in doc[subj_tok.i + 1 : start_i]
                                        if t.pos_ in ("AUX", "PART", "ADV") or t.dep_ in ("aux", "auxpass", "neg", "advmod")
                                    ]
                                    if aux_toks:
                                        mid_pv_aux = " ".join(aux_toks) + " "
                                det_cands = [d for d in subj_tok.children if d.dep_ == "det" and d.i < subj_tok.i and d.is_alpha]
                                if det_cands:
                                    d_low = det_cands[-1].text.lower()
                                    if d_low in ("my", "your", "our"):
                                        d_low = "a" if subj_tok.tag_ == "NN" else "the"
                                    return f"... {d_low} {s_low} {mid_pv_aux}____"
                                if subj_tok.tag_ == "NN":
                                    return f"... a {s_low} {mid_pv_aux}____"
                                return f"... {s_low} {mid_pv_aux}____"
                            if subj_tok and subj_tok.pos_ == "PRON" and not _is_forbidden(subj_tok.lemma_.lower()):
                                mid_pv_aux = ""
                                if verb_form_tag == "VBN" and subj_tok.i + 1 < start_i:
                                    aux_toks = [
                                        t.text.lower()
                                        for t in doc[subj_tok.i + 1 : start_i]
                                        if t.pos_ in ("AUX", "PART", "ADV") or t.dep_ in ("aux", "auxpass", "neg", "advmod")
                                    ]
                                    if aux_toks:
                                        mid_pv_aux = " ".join(aux_toks) + " "
                                return f"... {subj_tok.text.lower()} {mid_pv_aux}____"
                            if verb_form_tag == "VBN":
                                pre_aux = [
                                    t.text.lower()
                                    for t in doc[max(0, start_i - 2) : start_i]
                                    if (t.pos_ in ("AUX", "VERB") and t.lemma_.lower() in ("have", "be", "get")) or t.dep_ in ("aux", "auxpass")
                                ]
                                if pre_aux:
                                    return f"... {' '.join(pre_aux)} ____"
                            if re.search(r"\bher\b", ft_low):
                                return "... she ____"
                            if re.search(r"\bhis\b", ft_low):
                                return "... he ____"
                            if re.search(r"\btheir\b", ft_low):
                                return "... they ____"
                            return "... someone ____"
                        return "... to ____"
                    is_clausal_connector_phrase = (
                        (len(ft_toks) == 3 and ft_toks[0] == "as" and ft_toks[2] == "as")
                        or ft_low in ("even though", "so that", "in order that", "now that", "provided that", "providing that", "as if", "as though")
                    )
                    if is_clausal_connector_phrase and end_i < len(doc) and doc[end_i].is_alpha and not _is_forbidden(doc[end_i].lemma_.lower()):
                        return f"... ____ {doc[end_i].text.lower()}"
                    is_self_contained_pp = (
                        ft_toks[0] in ("in", "on", "at", "by", "for", "under", "over", "from", "within", "without", "beyond", "behind", "between", "among", "through", "throughout", "around", "across")
                        or (len(ft_toks) >= 2 and (ft_toks[0], ft_toks[1]) in (("all", "over"), ("out", "of"), ("up", "to"), ("apart", "from"), ("instead", "of"), ("according", "to")))
                    )
                    if start_i > 0 and doc[start_i - 1].is_alpha and not _is_forbidden(doc[start_i - 1].lemma_.lower()):
                        prev_tok = doc[start_i - 1]
                        prev_w = prev_tok.text.lower()
                        if prev_w in ("and", "or", "got"):
                            return "... have ____"

                        # Mechanism C: Copular PP set phrase subject/complement anchoring
                        if is_self_contained_pp and prev_tok.lemma_.lower() in ("be", "seem", "remain", "stay", "look", "feel", "become"):
                            # Check trailing PP/adjunct after end_i (e.g. 'on the shelf behind them')
                            if end_i < len(doc) and doc[end_i].pos_ == "ADP" and not _is_forbidden(doc[end_i].lemma_.lower()):
                                trail_tail = _pobj_phrase(doc[end_i])
                                if trail_tail:
                                    return f"... {prev_w} ____ {doc[end_i].text.lower()}{trail_tail}".strip()
                            # Backtrack to subject of copula (e.g. 'the documents were on the shelf')
                            cop_subj = next(
                                (
                                    c for c in prev_tok.children
                                    if c.dep_ in ("nsubj", "nsubjpass") and c.i < prev_tok.i and c.pos_ in ("NOUN", "PROPN") and not _is_forbidden(c.lemma_.lower())
                                ),
                                None,
                            )
                            if cop_subj is None and prev_tok.dep_ == "cop" and prev_tok.head:
                                cop_subj = next(
                                    (
                                        c for c in prev_tok.head.children
                                        if c.dep_ in ("nsubj", "nsubjpass") and c.i < prev_tok.i and c.pos_ in ("NOUN", "PROPN") and not _is_forbidden(c.lemma_.lower())
                                    ),
                                    None,
                                )
                            if cop_subj is not None:
                                subj_det = next(
                                    (c.text.lower() for c in cop_subj.children if c.dep_ in ("det", "poss") and c.i < cop_subj.i and not _is_forbidden(c.lemma_.lower())),
                                    "",
                                )
                                subj_phrase = f"{subj_det} {cop_subj.text.lower()}".strip()
                                return f"... {subj_phrase} {prev_w} ____".strip()

                        # Mechanism D: Governing head backtracking for noun idioms / binomials preceded by article
                        if prev_w in ("the", "a", "an"):
                            if start_i > 1:
                                tok_before_det = doc[start_i - 2]
                                if tok_before_det.pos_ == "ADP" and start_i > 2:
                                    gov_tok = doc[start_i - 3]
                                    if gov_tok.pos_ in ("VERB", "NOUN", "ADJ") and not _is_forbidden(gov_tok.lemma_.lower()):
                                        gov_phrase = f"{gov_tok.text.lower()} {tok_before_det.text.lower()}"
                                        return f"... {gov_phrase} {prev_w} ____"
                                if tok_before_det.pos_ in ("VERB", "ADP") and not _is_forbidden(tok_before_det.lemma_.lower()):
                                    return f"... {tok_before_det.text.lower()} {prev_w} ____"

                        if not (is_self_contained_pp and prev_tok.pos_ in ("ADP", "ADJ", "ADV", "DET", "CCONJ", "SCONJ")):
                            return f"... {prev_w} ____"

                    # Trailing context after end_i with prepositional object closure
                    if end_i < len(doc) and doc[end_i].is_alpha and not _is_forbidden(doc[end_i].lemma_.lower()):
                        nxt_after = doc[end_i]
                        pobj_tail = _pobj_phrase(nxt_after) if nxt_after.pos_ == "ADP" else ""
                        return f"... ____ {nxt_after.text.lower()}{pobj_tail}".strip()
                    if is_self_contained_pp:
                        for v_tok in doc[ : start_i]:
                            if v_tok.pos_ in ("VERB", "AUX") and v_tok.is_alpha and not _is_forbidden(v_tok.lemma_.lower()):
                                return f"... {v_tok.text.lower()} ... ____"
                        for v_tok in doc[end_i : ]:
                            if v_tok.pos_ in ("VERB", "AUX") and v_tok.is_alpha and not _is_forbidden(v_tok.lemma_.lower()):
                                return f"... ____ ... {v_tok.text.lower()}"
            return None

        # Locate target token / span in `doc` (supporting hyphenated multi-token compounds like `brand-new`)
        t_start_i, t_end_i, t_tok = cls._find_target_span_in_doc(
            doc, ft_low, base_str=bw_low, t_forms=t_forms
        )
        if t_tok is None or t_start_i is None or t_end_i is None:
            return None

        if verb_form_tag is None and canonical_pos == "verb" and t_tok.tag_ in ("VBG", "VBN", "VBD", "VBZ", "VBP", "VB"):
            verb_form_tag = t_tok.tag_

        # 2. Adjective (`canonical_pos == "adj"`)
        if canonical_pos == "adj":
            eff_head = t_tok
            while eff_head.dep_ == "conj" and eff_head.head and eff_head.head.pos_ == "ADJ" and eff_head.head.i < eff_head.i:
                eff_head = eff_head.head
            _cop_contraction_map = {
                "'s": "is", "’s": "is",
                "'re": "are", "’re": "are",
                "'m": "am", "’m": "am",
                "'ll": "will", "’ll": "will",
                "'d": "would", "’d": "would",
            }
            # Check for bound preposition immediately following target span (excluding 'as')
            prep_child = next(
                (
                    c for c in list(t_tok.children) + (list(eff_head.children) if eff_head != t_tok else [])
                    if c.dep_ == "prep"
                    and c.i >= t_end_i
                    and c.text.lower() in ("in", "of", "to", "for", "with", "about", "at", "on", "from", "by", "into", "over", "towards", "against")
                ),
                None,
            )
            if prep_child is None and t_end_i < len(doc) and doc[t_end_i].pos_ == "ADP" and doc[t_end_i].text.lower() in (
                "in", "of", "to", "for", "with", "about", "at", "on", "from", "by", "into", "over", "towards", "against"
            ) and eff_head.dep_ != "amod":
                prep_child = doc[t_end_i]
            if (
                is_copula_adj
                or anchor_type == "verb_copula"
                or (prep_child is not None and eff_head.dep_ != "amod")
                or eff_head.dep_ in ("acomp", "oprd", "xcomp", "ccomp")
            ):
                cop_word = anchor.lower() if (anchor_type == "verb_copula" and anchor) else "be"
                gov_v = None
                if eff_head.head and eff_head.head.pos_ in ("AUX", "VERB") and eff_head.head.i < t_start_i:
                    raw_cop = eff_head.head.text.lower()
                    cop_word = _cop_contraction_map.get(raw_cop, raw_cop)
                    gov_v = eff_head.head
                else:
                    for c in list(eff_head.children) + list(t_tok.children):
                        if c.dep_ in ("cop", "aux", "auxpass") and c.i < t_start_i:
                            raw_cop = c.text.lower()
                            cop_word = _cop_contraction_map.get(raw_cop, raw_cop)
                            gov_v = c
                            break

                obj_slot = ""
                if gov_v is not None and gov_v.pos_ == "VERB":
                    interv_obj = next(
                        (
                            c for c in gov_v.children
                            if c.dep_ in ("dobj", "obj") and gov_v.i <= c.i < t_start_i
                        ),
                        None,
                    )
                    if interv_obj is None:
                        interv_obj = next(
                            (
                                c for c in eff_head.children
                                if c.dep_ in ("nsubj", "dobj") and gov_v.i <= c.i < t_start_i
                            ),
                            None,
                        )
                    if interv_obj is not None:
                        is_human_obj = (
                            interv_obj.text.lower() in ("someone", "somebody", "sb", "her", "him", "them", "us", "me")
                            or interv_obj.ent_type_ == "PERSON"
                            or interv_obj.lemma_.lower() in {
                                "person", "people", "man", "men", "woman", "women", "child", "children", "kid", "boy", "girl",
                                "student", "teacher", "professor", "worker", "employee", "staff", "colleague", "customer", "client"
                            }
                        )
                        obj_slot = "[someone] " if is_human_obj else "[something] "

                # Subject anchor for copula adjectives ('The rocks are slippery' -> '... the rocks are ____')
                subj_prefix = ""
                if gov_v is not None and not obj_slot:
                    subj_cand = next((c for c in gov_v.children if c.dep_ in ("nsubj", "nsubjpass") and c.i < t_start_i), None)
                    if subj_cand is None and eff_head.head:
                        subj_cand = next((c for c in eff_head.children if c.dep_ in ("nsubj", "nsubjpass") and c.i < t_start_i), None)
                    if subj_cand is not None:
                        if subj_cand.pos_ in ("NOUN", "PROPN") and not _is_forbidden(subj_cand.lemma_.lower()):
                            s_det = next((c.text.lower() for c in subj_cand.children if c.dep_ in ("det", "poss") and c.i < subj_cand.i), "")
                            if not s_det and subj_cand.tag_ == "NN":
                                s_det = "the"
                            subj_prefix = f"{s_det} {subj_cand.text.lower()} ".strip() + " "

                adv_mod = next((c for c in eff_head.children if c.dep_ == "advmod" and c.i < t_start_i and c.text.lower() in ("so", "very", "quite", "extremely", "similarly", "increasingly", "highly", "fairly", "rather")), None)
                if adv_mod and not obj_slot:
                    cop_word = f"{cop_word} {adv_mod.text.lower()}"

                has_prev_as = (t_start_i > 0 and doc[t_start_i - 1].text.lower() == "as")
                has_next_as = (t_end_i < len(doc) and doc[t_end_i].text.lower() == "as")
                if has_prev_as and has_next_as:
                    correl_tail = "as"
                    if t_end_i + 1 < len(doc) and doc[t_end_i + 1].is_alpha:
                        correl_tail = f"as {doc[t_end_i + 1].text.lower()}"
                    return f"... {subj_prefix}{cop_word} {obj_slot}as ____ {correl_tail}"

                if prep_child is not None and (anchor_type != "verb_copula" or cop_word in ("be", "is", "are", "was", "were", "been")):
                    return f"... {subj_prefix}{cop_word} {obj_slot}____ {prep_child.text.lower()}{_pobj_phrase(prep_child)}"
                if anchor_type == "prep" and anchor and anchor.lower() != "as":
                    return f"... {subj_prefix}{cop_word} {obj_slot}____ {anchor.lower()}"
                return f"... {subj_prefix}{cop_word} {obj_slot}____".strip()

            # Attributive pre-nominal `Adj + Noun` (`eff_head.dep_ == "amod"`)
            if eff_head.dep_ in ("amod", "compound") and eff_head.head and eff_head.head.pos_ in ("NOUN", "PROPN") and t_end_i <= eff_head.head.i:
                head_n = eff_head.head
                head_txt = head_n.text.lower()
                if _is_forbidden(head_n.lemma_.lower()):
                    head_txt = "[noun]"
                # Secondary dependency 1: Trailing preposition after head noun (e.g. 'the overall cost of', 'a leading expert in')
                head_prep = next((c for c in head_n.children if c.dep_ == "prep" and c.i == head_n.i + 1 and c.text.lower() in ("of", "for", "in", "to", "on", "with", "against", "about", "towards", "from", "into")), None)
                if head_prep is not None:
                    head_txt = f"{head_txt} {head_prep.text.lower()}"

                det_tok = next(
                    (c for c in head_n.children if c.dep_ in ("det", "poss") and c.i < t_start_i),
                    None,
                )
                det_str = _normalize_det(det_tok.text if det_tok else "", det_tok_obj=det_tok)

                # Secondary dependency 2: Governing verb if head noun is direct object and frame lacks prep (e.g. 'reduce the overall cost')
                v_pref = ""
                if head_prep is None and head_n.dep_ in ("dobj", "obj") and head_n.head and head_n.head.pos_ == "VERB" and head_n.head.i < t_start_i:
                    v_gov = head_n.head
                    if v_gov.lemma_.lower() not in cls._DELEXICAL_VERBS and not _is_forbidden(v_gov.lemma_.lower()):
                        v_pref = f"{v_gov.text.lower()} "

                if det_str:
                    return f"... {v_pref}{det_str} ____ {head_txt}".strip()
                if t_start_i > 0 and doc[t_start_i - 1].pos_ == "ADP" and doc[t_start_i - 1].is_alpha:
                    return f"... {doc[t_start_i - 1].text.lower()} ____ {head_txt}"
                return f"... {v_pref}____ {head_txt}".strip()

            # Fallback if immediately followed by a noun in surface order
            if t_end_i < len(doc) and doc[t_end_i].pos_ in ("NOUN", "PROPN"):
                nxt_n = doc[t_end_i]
                head_txt = "[noun]" if _is_forbidden(nxt_n.lemma_.lower()) else nxt_n.text.lower()
                prev_det = doc[t_start_i - 1].text if (t_start_i > 0 and doc[t_start_i - 1].pos_ in ("DET", "PRON")) else ""
                det_str = _normalize_det(prev_det)
                if det_str:
                    return f"... {det_str} ____ {head_txt}"
                if t_start_i > 0 and doc[t_start_i - 1].pos_ == "ADP" and doc[t_start_i - 1].is_alpha:
                    return f"... {doc[t_start_i - 1].text.lower()} ____ {head_txt}"
                return f"... ____ {head_txt}"

            if prep_child is not None:
                return f"... ____ {prep_child.text.lower()}"
            if t_start_i > 0 and doc[t_start_i - 1].is_alpha and doc[t_start_i - 1].pos_ not in ("NOUN", "PROPN"):
                prev_w = _cop_contraction_map.get(doc[t_start_i - 1].text.lower(), doc[t_start_i - 1].text.lower())
                if prev_w in ("and", "or") and eff_head.head and eff_head.head.pos_ in ("AUX", "VERB"):
                    prev_w = _cop_contraction_map.get(eff_head.head.text.lower(), eff_head.head.text.lower())
                return f"... {prev_w} ____"
            return "... the ____ [noun]"

        # 3. Verb (`canonical_pos == "verb"`)
        if canonical_pos == "verb":
            _has_verb_dobj_child = any(c.dep_ in ("dobj", "obj") and c.i > t_tok.i for c in t_tok.children)
            if t_tok.pos_ in ("NOUN", "PROPN") and t_tok.dep_ in ("compound", "pobj", "dobj", "attr") and not _has_verb_dobj_child:
                if t_tok.dep_ == "pobj" and t_tok.head and t_tok.head.pos_ == "ADP" and (verb_form_tag == "VBG" or ft_low.endswith("ing")):
                    _p_gov = t_tok.head.head.text.lower() if (t_tok.head.head and t_tok.head.head.is_alpha) else ""
                    return f"... {_p_gov + ' ' if _p_gov else ''}{t_tok.head.text.lower()} ____"
                if t_tok.dep_ == "compound" and t_tok.head and t_tok.head.pos_ in ("NOUN", "PROPN"):
                    return f"... ____ {t_tok.head.text.lower()}"
                if t_tok.dep_ in ("dobj", "obj") and t_tok.head and t_tok.head.pos_ in ("VERB", "AUX"):
                    return f"... {t_tok.head.text.lower()} ____"
                if patient_ontology_slot:
                    return f"... ____ {patient_ontology_slot}"
                return "... ____"
            left_aux = ""
            _aux_link_verbs = (
                "has", "have", "had", "is", "are", "was", "were", "been", "being", "be",
                "get", "got", "seem", "seemed", "seems", "become", "became", "becomes",
                "appear", "appeared", "appears", "feel", "felt", "look", "looked", "remain", "remained",
            )
            if t_tok.i > 0:
                prev_t = doc[t_tok.i - 1]
                if (
                    verb_form_tag in ("VBN", "VBG")
                    and (prev_t.pos_ in ("AUX", "ADP") or prev_t.text.lower() in _aux_link_verbs)
                ) or (
                    verb_form_tag == "VBD"
                    and prev_t.text.lower() in _aux_link_verbs
                ):
                    if prev_t.text.lower() in ("been", "being") and t_tok.i > 1 and doc[t_tok.i - 2].text.lower() in ("has", "have", "had", "is", "are", "was", "were"):
                        left_aux = f"{doc[t_tok.i - 2].text.lower()} {prev_t.text.lower()} "
                    elif (
                        verb_form_tag == "VBG"
                        and prev_t.pos_ == "ADP"
                        and t_tok.i > 1
                        and doc[t_tok.i - 2].pos_ in ("VERB", "ADJ", "NOUN", "ADV")
                        and doc[t_tok.i - 2].is_alpha
                        and not _is_forbidden(doc[t_tok.i - 2].lemma_.lower())
                    ):
                        left_aux = f"{doc[t_tok.i - 2].text.lower()} {prev_t.text.lower()} "
                    else:
                        left_aux = prev_t.text.lower() + " "
                elif (
                    verb_form_tag in ("VBN", "VBG")
                    and prev_t.pos_ == "ADV"
                    and t_tok.i > 1
                    and (doc[t_tok.i - 2].pos_ == "AUX" or doc[t_tok.i - 2].text.lower() in _aux_link_verbs)
                    and not _is_forbidden(prev_t.lemma_.lower())
                ):
                    left_aux = f"{doc[t_tok.i - 2].text.lower()} {prev_t.text.lower()} "

            # Intransitive / Ergative Inchoative / Passive verb flag
            is_tok_passive = (
                any(c.dep_ in ("auxpass", "nsubjpass") for c in t_tok.children)
                or (t_tok.tag_ == "VBN" and any(c.dep_ in ("aux", "auxpass") and c.lemma_.lower() in ("be", "get") for c in t_tok.children))
                or (verb_form_tag == "VBN" and any(c.dep_ in ("auxpass", "nsubjpass") or (c.dep_ == "aux" and c.lemma_.lower() in ("be", "get")) for c in t_tok.children))
            )

            # 3a. Infinitive complement (`to + V-base`)
            if (
                t_tok.i + 2 < len(doc)
                and doc[t_tok.i + 1].text.lower() == "to"
                and (doc[t_tok.i + 1].tag_ == "TO" or doc[t_tok.i + 2].pos_ in ("VERB", "AUX"))
            ):
                return f"... {left_aux}____ to [verb]"

            # 3b. Bound preposition complement (`pose for`, `correlate with`, `originate in`, `remind sb of`, `endow sb with`)
            _temporal_adj_nouns = {
                "time", "year", "years", "month", "months", "week", "weeks",
                "day", "days", "decade", "decades", "century", "centuries",
                "night", "morning", "evening", "hour", "hours", "moment",
            }
            prep_c = next(
                (
                    c for c in t_tok.children
                    if (c.dep_ == "prep" or (c.dep_ == "agent" and anchor_type == "prep"))
                    and not (is_tok_passive and c.text.lower() == "by" and anchor_type != "prep")
                    and c.i > t_tok.i
                    and c.text.lower() in ("for", "with", "in", "into", "to", "from", "on", "of", "at", "about", "against", "over", "under", "between", "as", "upon", "towards", "by")
                    and not (
                        c.text.lower() in ("over", "for", "in", "at")
                        and not any(mc.dep_ in ("dobj", "obj") and t_tok.i < mc.i < c.i for mc in t_tok.children)
                        and any(pc.dep_ == "pobj" and pc.lemma_.lower() in _temporal_adj_nouns for pc in c.children)
                    )
                ),
                None,
            )
            if (
                prep_c is not None
                and anchor_type != "verb_subject"
                and (anchor_type == "prep" or not any(c.dep_ in ("dobj", "obj") and c.i < prep_c.i for c in t_tok.children))
            ):
                mid_dobj = next(
                    (c for c in t_tok.children if c.dep_ in ("dobj", "obj") and t_tok.i < c.i < prep_c.i),
                    None,
                )
                if mid_dobj is not None:
                    _md_low = mid_dobj.text.lower()
                    if patient_ontology_slot:
                        _mid_slot = patient_ontology_slot
                    elif _md_low.endswith(("self", "selves")):
                        _mid_slot = "oneself"
                    elif _md_low in ("me", "you", "him", "her", "us", "whom", "someone", "somebody", "everyone", "anyone", "nobody") or mid_dobj.ent_type_ == "PERSON":
                        _mid_slot = "[person/group]"
                    else:
                        _mid_slot = cls._resolve_object_animacy_from_example(ex_clean, ft_low, default_tag="something")
                        if _mid_slot == "someone":
                            _mid_slot = "[person/group]"
                    _prep_surf = (
                        "as to"
                        if prep_c.text.lower() == "as" and prep_c.i + 1 < len(doc) and doc[prep_c.i + 1].text.lower() == "to"
                        else prep_c.text.lower()
                    )
                    return f"... {left_aux}____ {_mid_slot} {_prep_surf}"
                _p_aux = left_aux
                if is_tok_passive and not _p_aux:
                    _p_aux = "be " if verb_form_tag == "VBN" else "was/were "
                return f"... {_p_aux}____ {prep_c.text.lower()}{_pobj_phrase(prep_c)}"

            # 3c. Direct object (`dobj` / `obj` / `oprd`)
            _temporal_np_nouns = {
                "winter", "summer", "spring", "autumn", "fall", "morning", "afternoon",
                "evening", "night", "day", "week", "month", "year", "time", "hour", "minute",
            }
            dobj_c = (
                next(
                    (
                        c for c in t_tok.children
                        if c.dep_ in ("dobj", "obj", "oprd")
                        and c.i > t_tok.i
                        and c.lemma_.lower() not in _temporal_np_nouns
                    ),
                    None,
                )
                or next(
                    (
                        c for c in t_tok.children
                        if c.dep_ in ("dobj", "obj", "oprd")
                        and c.i > t_tok.i
                    ),
                    None,
                )
                if anchor_type != "verb_subject"
                else None
            )
            if dobj_c is not None:
                _partitive_quant_lemmas = {
                    "ton", "pile", "load", "heap", "stack", "hundred", "thousand", "million",
                    "lot", "plenty", "piece", "bit", "drop", "slice", "amount", "quantity",
                    "number", "type", "kind", "sort", "series", "range", "variety",
                }
                partitive_of_tail = ""
                if dobj_c.lemma_.lower() in _partitive_quant_lemmas:
                    prep_of_c = next(
                        (
                            ch for ch in dobj_c.children
                            if ch.dep_ == "prep" and ch.text.lower() == "of" and ch.i == dobj_c.i + 1
                        ),
                        None,
                    )
                    if prep_of_c is not None:
                        pobj_of_c = next(
                            (
                                pch for pch in prep_of_c.children
                                if pch.dep_ == "pobj" and pch.pos_ in ("NOUN", "PROPN") and not _is_forbidden(pch.lemma_.lower())
                            ),
                            None,
                        )
                        if pobj_of_c is not None:
                            partitive_of_tail = f" of {_tok_surface(pobj_of_c)}"
                # Check for a trailing particle (`tip his chair back`) or governed trivalent preposition (`attach importance to`)
                trail_prt_or_prep = ""
                if not partitive_of_tail and dobj_c.i + 1 < len(doc):
                    nxt_after_obj = doc[dobj_c.i + 1]
                    if nxt_after_obj.dep_ == "prt" and nxt_after_obj.head == t_tok:
                        trail_prt_or_prep = f" {nxt_after_obj.text.lower()}"
                    elif (
                        nxt_after_obj.dep_ == "prep"
                        and (nxt_after_obj.head == t_tok or nxt_after_obj.head == dobj_c)
                        and nxt_after_obj.text.lower() in ("to", "into", "onto", "on", "from", "with", "for", "against", "in", "at", "upon", "towards", "as", "about", "between", "over", "under", "through", "behind")
                    ):
                        nxt_low = nxt_after_obj.text.lower()
                        if nxt_low == "as" and dobj_c.i + 2 < len(doc) and doc[dobj_c.i + 2].text.lower() == "to":
                            trail_prt_or_prep = " as to"
                        else:
                            trail_prt_or_prep = f" {nxt_low}"
                if dobj_c.pos_ == "PRON" or dobj_c.ent_type_ == "PERSON" or _is_forbidden(dobj_c.lemma_.lower()):
                    if patient_ontology_slot and not partitive_of_tail:
                        return f"... {left_aux}____ {patient_ontology_slot}{trail_prt_or_prep}"
                    anim = cls._resolve_object_animacy_from_example(ex_clean, ft_low, default_tag="[sb/sth]")
                    if anim == "someone":
                        anim = "[person/group]"
                    return f"... {left_aux}____ {anim}{trail_prt_or_prep}"
                if dobj_c.pos_ in ("NOUN", "PROPN"):
                    # Collect determiners/quantifiers/numbers between `t_tok` and `dobj_c`
                    raw_span = list(doc[t_tok.i + 1 : dobj_c.i + 1])
                    poss_idx = next((idx for idx, t in enumerate(raw_span[:-1]) if t.dep_ == "poss" and t.pos_ in ("NOUN", "PROPN")), None)
                    if poss_idx is not None:
                        after_poss = [_tok_surface(t) for t in raw_span[poss_idx + 1 :] if (t.is_alpha or t.like_num) and not _is_forbidden(t.lemma_.lower())]
                        if after_poss and (
                            cls._is_body_part_noun(dobj_c.lemma_.lower())
                            or dobj_c.tag_ in ("NNS", "NNPS")
                            or cls._is_primarily_uncountable_noun(dobj_c.lemma_.lower())
                        ):
                            span_toks = after_poss
                        else:
                            span_toks = ["the"] + after_poss if after_poss else [_tok_surface(dobj_c)]
                    else:
                        span_toks = [
                            _tok_surface(t)
                            for t in raw_span
                            if (t.is_alpha or t.like_num) and not _is_forbidden(t.lemma_.lower())
                        ]

                    # Enrich short verb frames or inanimate causal subjects with authentic subject noun/pronoun
                    subj_prefix = ""
                    if not left_aux:
                        subj_tok = next((c for c in t_tok.children if c.dep_ in ("nsubj", "nsubjpass") and c.i < t_tok.i), None)
                        if subj_tok is not None and not _is_forbidden(subj_tok.lemma_.lower()):
                            is_short_obj_subj = not trail_prt_or_prep and len(span_toks) <= 2
                            is_causal_inanimate = False
                            if (
                                not is_short_obj_subj
                                and subj_tok.pos_ in ("NOUN", "PROPN", "ADJ")
                                and len(span_toks) <= 3
                                and subj_tok.lemma_.lower() not in {"thing", "things", "way", "time", "fact", "case", "part", "point"}
                            ):
                                wn_s = cls.get_wordnet()
                                s_syns = wn_s.synsets(subj_tok.lemma_.lower(), "n") if wn_s else []
                                if s_syns and s_syns[0].lexfile() in {
                                    "noun.substance", "noun.phenomenon", "noun.process",
                                    "noun.event", "noun.state", "noun.attribute", "noun.cognition",
                                }:
                                    is_causal_inanimate = True
                            if is_short_obj_subj or is_causal_inanimate:
                                if subj_tok.pos_ in ("NOUN", "PROPN") or is_causal_inanimate:
                                    s_det = next((c.text.lower() for c in subj_tok.children if c.dep_ in ("det", "poss") and c.i < subj_tok.i and not _is_forbidden(c.lemma_.lower())), "")
                                    s_surf = subj_tok.text.lower() if (subj_tok.pos_ in ("NOUN", "ADJ")) else _tok_surface(subj_tok)
                                    s_str = f"{s_det} {s_surf}".strip()
                                    subj_prefix = f"{s_str} "
                                elif is_short_obj_subj and subj_tok.text.lower() in ("they", "we", "he", "she"):
                                    subj_prefix = f"{subj_tok.text.lower()} "

                    if 1 <= len(span_toks) <= 4:
                        return f"... {subj_prefix}{left_aux}____ {' '.join(span_toks)}{partitive_of_tail}{trail_prt_or_prep}".strip()
                    return f"... {subj_prefix}{left_aux}____ {_tok_surface(dobj_c)}{partitive_of_tail}{trail_prt_or_prep}".strip()

            # 3d. Intransitive / Ergative Inchoative / Passive verb with subject (`the custom ____`, `the lake had ____ overnight`)
            is_tok_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in t_tok.children)
            subj_c = next(
                (c for c in t_tok.children if c.dep_ in ("nsubj", "nsubjpass") and c.i < t_tok.i),
                None,
            )
            if (
                subj_c is None
                and t_tok.dep_ == "amod"
                and t_tok.head.dep_ == "ROOT"
                and t_tok.head.pos_ in ("NOUN", "PROPN")
                and t_tok.i > t_tok.head.i
            ):
                subj_c = t_tok.head
            xcomp_bridge = ""
            if subj_c is None and t_tok.dep_ == "xcomp" and t_tok.head.pos_ in ("VERB", "AUX") and t_tok.head.i < t_tok.i:
                subj_c = next(
                    (c for c in t_tok.head.children if c.dep_ in ("nsubj", "nsubjpass") and c.i < t_tok.head.i),
                    None,
                )
                if subj_c is not None:
                    bridge_toks = [t.text.lower() for t in doc[t_tok.head.i : t_tok.i] if (t.is_alpha or "'" in t.text) and not _is_forbidden(t.lemma_.lower())]
                    if 1 <= len(bridge_toks) <= 3:
                        xcomp_bridge = " ".join(bridge_toks) + " "
            if subj_c is not None and not _is_forbidden(subj_c.lemma_.lower()):
                mid_aux = left_aux
                if not mid_aux and not xcomp_bridge and subj_c.i + 1 < t_tok.i:
                    aux_span = [
                        t.text.lower()
                        for t in doc[subj_c.i + 1 : t_tok.i]
                        if (t.pos_ in ("AUX", "PART") or t.dep_ in ("aux", "auxpass", "neg"))
                    ]
                    if aux_span:
                        mid_aux = "".join(
                            (tok_s if tok_s.startswith("'") or tok_s == "n't" else f" {tok_s}")
                            for tok_s in aux_span
                        ).strip() + " "
                post_adv = ""
                if t_tok.i + 1 < len(doc):
                    nxt_t = doc[t_tok.i + 1]
                    if (
                        nxt_t.dep_ in ("prt", "advmod")
                        and nxt_t.is_alpha
                        and not _is_forbidden(nxt_t.lemma_.lower())
                    ):
                        post_adv = f" {nxt_t.text.lower()}"
                # Deep Passive Projection: if the verb in the example is passive (`subj_c` is Patient / 受动者)
                # but the quiz options are in active base form (`verb_form_tag == "VB"`), never place the Patient
                # before a base-form verb (which would invert 受动者 into 施动者); project Patient into the object slot.
                if is_tok_passive and verb_form_tag not in ("VBD", "VBZ", "VBG", "VBN"):
                    if patient_ontology_slot:
                        return f"... {left_aux}____ {patient_ontology_slot}{post_adv}"
                    if subj_c.pos_ in ("NOUN", "PROPN"):
                        s_det = next((c.text.lower() for c in subj_c.children if c.dep_ in ("det", "poss") and c.i < subj_c.i), "")
                        if not s_det and subj_c.tag_ == "NN":
                            s_det = "the"
                        s_prefix = f"{s_det} {subj_c.text.lower()}".strip()
                        return f"... {left_aux}____ {s_prefix}{post_adv}"
                # Active Transitive Relative-Clause Guard: if `anchor_type == "object"` in active voice
                # (meaning the direct object was fronted in a relative/wh clause and `subj_c` is the active Agent / 施动者),
                # do not slice a dangling `... [Agent] ____` frame without its object.
                elif not is_tok_passive and anchor_type == "object":
                    if patient_ontology_slot:
                        return f"... {left_aux}____ {patient_ontology_slot}{post_adv}"
                    if (
                        t_tok.dep_ == "relcl"
                        and t_tok.head.pos_ in ("NOUN", "PROPN")
                        and not _is_forbidden(t_tok.head.lemma_.lower())
                    ):
                        rel_det = "the " if t_tok.head.tag_ == "NN" else ""
                        return f"... {left_aux}____ {rel_det}{t_tok.head.text.lower()}{post_adv}"
                    if anchor and not _is_forbidden(str(anchor).lower()):
                        return f"... {left_aux}____ the {str(anchor).lower()}{post_adv}"
                else:
                    if is_tok_passive and verb_form_tag == "VBN" and patient_ontology_slot:
                        _raw_p_aux = xcomp_bridge or mid_aux or "was/were "
                        _raw_p_aux = re.sub(r"\b(?:was|were)\b", "was/were", _raw_p_aux)
                        _raw_p_aux = re.sub(r"\b(?:is|are)\b", "is/are", _raw_p_aux)
                        _raw_p_aux = re.sub(r"\b(?:has|have)\b", "has/have", _raw_p_aux)
                        return f"... {patient_ontology_slot} {_raw_p_aux}____{post_adv}"
                    if subj_c.pos_ == "PRON":
                        if patient_ontology_slot:
                            _aux_norm = xcomp_bridge or mid_aux or ""
                            if is_tok_passive:
                                _aux_norm = _aux_norm or "was/were "
                                _aux_norm = re.sub(r"\b(?:was|were)\b", "was/were", _aux_norm)
                                _aux_norm = re.sub(r"\b(?:is|are)\b", "is/are", _aux_norm)
                                _aux_norm = re.sub(r"\b(?:has|have)\b", "has/have", _aux_norm)
                            return f"... {patient_ontology_slot} {_aux_norm}____{post_adv}".replace("  ", " ")
                        return f"... {subj_c.text.lower()} {xcomp_bridge or mid_aux}____"
                    s_det = next((c.text.lower() for c in subj_c.children if c.dep_ in ("det", "poss") and c.i < subj_c.i), "")
                    s_prefix = f"{s_det} {_tok_surface(subj_c)}".strip()
                    return f"... {s_prefix} {xcomp_bridge or mid_aux}____{post_adv}"

            if left_aux:
                return f"... {left_aux}____"

            if prep_c is not None:
                return f"... {left_aux}____ {prep_c.text.lower()}"
            if (
                t_tok.i + 1 < len(doc)
                and doc[t_tok.i + 1].is_alpha
                and doc[t_tok.i + 1].pos_ in ("ADP", "PART", "ADV")
                and doc[t_tok.i + 1].text.lower() not in ("next", "last", "this", "that", "every", "each", "all", "both", "today", "yesterday", "tomorrow")
                and not _is_forbidden(doc[t_tok.i + 1].lemma_.lower())
            ):
                return f"... {left_aux}____ {doc[t_tok.i + 1].text.lower()}"
            if anchor and anchor_type == "prep" and not _is_forbidden(str(anchor).lower()):
                return f"... {left_aux}____ {str(anchor).lower()}"
            if (
                t_tok.i > 0
                and not (t_tok.i >= 2 and doc[t_tok.i - 2].text in ("-", "–"))
                and doc[t_tok.i - 1].is_alpha
                and doc[t_tok.i - 1].pos_ in ("AUX", "ADV", "PART")
                and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower())
            ):
                return f"... {doc[t_tok.i - 1].text.lower()} ____"
            if (
                t_tok.dep_ == "amod"
                and t_tok.head.pos_ in ("NOUN", "PROPN")
                and t_tok.head.i > t_tok.i
                and not _is_forbidden(t_tok.head.lemma_.lower())
            ):
                return f"... the {t_tok.head.text.lower()} was ____"
            if left_aux:
                if patient_ontology_slot:
                    return f"... {left_aux}____ {patient_ontology_slot}"
                return f"... {left_aux}____".strip()
            return f"... ____ {patient_ontology_slot or '[object]'}"

        # 4. Noun (`canonical_pos == "noun"`)
        if canonical_pos == "noun":
            det_tok = next(
                (c for c in t_tok.children if c.dep_ in ("det", "poss") and c.i < t_tok.i),
                None,
            )
            amod_tok = next(
                (c for c in t_tok.children if c.dep_ in ("amod", "compound") and c.i < t_tok.i and c.is_alpha and not _is_forbidden(c.lemma_.lower())),
                None,
            )
            if amod_tok is None and t_tok.i > 0 and doc[t_tok.i - 1].pos_ == "ADJ" and doc[t_tok.i - 1].is_alpha and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower()):
                amod_tok = doc[t_tok.i - 1]
            amod_txt = ""
            amod_first_i = amod_tok.i if amod_tok is not None else t_tok.i
            if amod_tok is not None:
                if (
                    amod_tok.i >= 2
                    and doc[amod_tok.i - 1].text in ("-", "–")
                    and (doc[amod_tok.i - 2].is_alpha or doc[amod_tok.i - 2].like_num)
                    and not _is_forbidden(doc[amod_tok.i - 2].lemma_.lower())
                ):
                    amod_txt = f"{doc[amod_tok.i - 2].text.lower()}-{amod_tok.text.lower()}"
                    amod_first_i = amod_tok.i - 2
                else:
                    amod_txt = amod_tok.text.lower()
            if det_tok is None:
                prev_det_i = (amod_first_i - 1) if amod_tok is not None else (t_tok.i - 1)
                if prev_det_i >= 0 and doc[prev_det_i].text.lower() in ("a", "an", "the", "each", "every", "his", "her", "their", "its", "our", "my", "your", "no"):
                    det_tok = doc[prev_det_i]
            default_noun_det = (
                ("several" if is_plural_noun else ("the" if (has_mixed_vc_initials and is_singular_only_noun) else ("each" if has_mixed_vc_initials else "a")))
                if has_uncountable_contrast_foil
                else ("a" if is_target_strictly_countable and not is_plural_noun and not has_mixed_vc_initials else ("the" if not is_pure_uncountable_target else ""))
            )
            det_str = _normalize_det(
                det_tok.text if det_tok else default_noun_det,
                det_tok_obj=det_tok,
            )
            det_for_amod = ""
            if amod_txt:
                if det_tok is not None:
                    raw_d = det_tok.text.strip().lower()
                    if raw_d.endswith("'s") or raw_d.endswith("’s") or (det_tok.dep_ == "poss" and det_tok.pos_ in ("NOUN", "PROPN", "ADJ")):
                        det_for_amod = "their"
                    elif raw_d in ("a", "an"):
                        if is_pure_uncountable_target or is_plural_noun:
                            det_for_amod = ("several" if has_uncountable_contrast_foil else "the") if is_plural_noun else ""
                        else:
                            det_for_amod = "an" if amod_txt[0] in "aeiou" else "a"
                    else:
                        det_for_amod = _normalize_det(raw_d, det_tok_obj=det_tok)
                elif not is_pure_uncountable_target and not is_plural_noun:
                    det_for_amod = ("an" if amod_txt[0] in "aeiou" else "a") if (is_target_strictly_countable or has_uncountable_contrast_foil) else "the"
            prep_after = next(
                (c for c in t_tok.children if c.dep_ == "prep" and c.i == t_tok.i + 1 and c.text.lower() in ("of", "for", "in", "to", "on", "with", "over", "between", "from", "into", "against", "towards", "about", "behind", "throughout", "under")),
                None,
            )
            if prep_after is None and t_tok.i + 1 < len(doc) and doc[t_tok.i + 1].pos_ == "ADP" and doc[t_tok.i + 1].text.lower() in ("of", "for", "in", "to", "on", "with", "over", "between", "from", "into", "against", "towards", "about", "behind", "throughout", "under"):
                prep_after = doc[t_tok.i + 1]

            # 4.0 Noun + infinitive complement (`a ____ to [verb]`, `several ____ to detect`, or support verb + noun + infinitive `made a ____ to visit`)
            if (
                t_tok.i + 2 < len(doc)
                and doc[t_tok.i + 1].text.lower() == "to"
                and (doc[t_tok.i + 1].tag_ == "TO" or doc[t_tok.i + 2].pos_ in ("VERB", "AUX"))
            ):
                d_shell = det_str or ("a" if not is_pure_uncountable_target and not is_plural_noun and not has_mixed_vc_initials else "the")
                v_inf_tok = doc[t_tok.i + 2]
                v_inf_lem = v_inf_tok.lemma_.lower()
                inf_tail = f"to {v_inf_tok.text.lower()}" if (anchor_type != "shell_to_do" and v_inf_lem not in cls._DELEXICAL_VERBS and not _is_forbidden(v_inf_lem)) else "to [verb]"

                # If the target noun is simultaneously the direct object of a support/lexical verb ('made a promise to visit', 'take steps to reduce', 'make an effort to improve')
                # preserve the governing support verb to lock the collocational trajectory!
                _SUPPORT_VERBS_NOUN = {
                    "make", "take", "give", "pay", "hold", "lose", "gain", "keep",
                    "put", "set", "bear", "lay", "draw", "reach", "strike", "pose", "stand", "fulfill", "fulfil", "break",
                }
                if t_tok.dep_ in ("dobj", "obj") and t_tok.head and t_tok.head.pos_ == "VERB":
                    v_head = t_tok.head
                    v_lem = v_head.lemma_.lower()
                    if (
                        v_lem not in cls._DELEXICAL_VERBS
                        or v_lem in _SUPPORT_VERBS_NOUN
                    ) and not _is_forbidden(v_lem):
                        prt_tok = next((c.text.lower() for c in v_head.children if c.dep_ == "prt" and c.i < t_tok.i), "")
                        v_surface = v_head.text.lower() if v_head.tag_ in ("VBD", "VBN", "VB", "VBP") else v_lem
                        v_phrase = f"{v_surface} {prt_tok}".strip()
                        np_mid = f"{d_shell} " if d_shell else ""
                        return f"... {v_phrase} {np_mid}____ {inf_tail}"

                return f"... {d_shell + ' ' if d_shell else ''}____ {inf_tail}"

            # 4a. Direct object of a lexical or support verb (`take out a ____`, `make a large ____ to`, `take a ten-minute ____`)
            _SUPPORT_VERBS_NOUN = {
                "make", "take", "give", "pay", "hold", "lose", "gain", "keep",
                "put", "set", "bear", "lay", "draw", "reach", "strike", "pose", "stand",
            }
            if t_tok.dep_ in ("dobj", "obj") and t_tok.head and t_tok.head.pos_ == "VERB":
                v_head = t_tok.head
                v_lem = v_head.lemma_.lower()
                if (
                    v_lem not in cls._DELEXICAL_VERBS
                    or v_lem in _SUPPORT_VERBS_NOUN
                    or (v_lem in ("have", "get") and (amod_txt or prep_after is not None))
                ) and not _is_forbidden(v_lem):
                    prt_tok = next((c.text.lower() for c in v_head.children if c.dep_ == "prt" and c.i < t_tok.i), "")
                    v_surface = v_head.text.lower() if v_head.tag_ in ("VBD", "VBN", "VB", "VBP") else v_lem
                    v_phrase = f"{v_surface} {prt_tok}".strip()
                    if amod_txt:
                        np_mid_str = f"{det_for_amod} {amod_txt}".strip()
                        np_mid = f"{np_mid_str} " if np_mid_str else ""
                    else:
                        np_mid = f"{det_str} " if det_str else ""
                    if prep_after is not None:
                        return f"... {v_phrase} {np_mid}____ {prep_after.text.lower()}"
                    return f"... {v_phrase} {np_mid}____"

            # 4a2. Predicate nominal / copular complement (`is a ____`, `was a ____`, `became a ____`)
            if t_tok.dep_ in ("attr", "oprd") and t_tok.head and t_tok.head.pos_ in ("VERB", "AUX"):
                v_head = t_tok.head
                cop_txt = v_head.text.lower() if v_head.tag_ in ("VBD", "VBZ", "VBP", "VBN", "VB") else v_head.lemma_.lower()
                subj_tok = next((c for c in v_head.children if c.dep_ in ("nsubj", "nsubjpass", "expl") and c.i < t_tok.i), None)
                s_str = ""
                if subj_tok is not None:
                    if subj_tok.dep_ == "expl":
                        s_str = "there "
                    elif subj_tok.pos_ in ("NOUN", "PROPN") and not _is_forbidden(subj_tok.lemma_.lower()):
                        s_det = next((c.text.lower() for c in subj_tok.children if c.dep_ in ("det", "poss") and c.i < subj_tok.i), "")
                        s_str = f"{s_det} {_tok_surface(subj_tok)} ".strip() + " "
                d_str = f"{det_str} " if det_str else ""
                if prep_after is not None:
                    return f"... {s_str}{cop_txt} {d_str}____ {prep_after.text.lower()}{_pobj_phrase(prep_after)}"
                return f"... {s_str}{cop_txt} {d_str}____".strip()

            # 4b. Pre-nominal adjective modifier + noun (+ optional prep) (`a personal ____ for`, `an economic ____`)
            if amod_tok is not None and amod_txt:
                np_mid = f"{det_for_amod} {amod_txt}".strip()
                if prep_after is not None:
                    return f"... {np_mid} ____ {prep_after.text.lower()}"
                # If target noun is direct object of a verb, preserve governing verb (e.g. 'replaces plastic ____', 'boost retail ____')
                if t_tok.dep_ in ("dobj", "obj") and t_tok.head and t_tok.head.pos_ == "VERB" and t_tok.head.i < t_tok.i:
                    v_head = t_tok.head
                    if v_head.lemma_.lower() not in cls._DELEXICAL_VERBS and not _is_forbidden(v_head.lemma_.lower()):
                        v_str = v_head.text.lower()
                        return f"... {v_str} {np_mid} ____".strip()
                return f"... {np_mid} ____"

            # 4b2. Compound noun pre-modifier / adjunct (`____ sales`, `the ____ industry`, `____ demand for`)
            if t_tok.dep_ in ("compound", "amod") and t_tok.head and t_tok.head.pos_ in ("NOUN", "PROPN") and t_tok.head.i > t_tok.i:
                head_n = t_tok.head
                # Inherit determiner from head noun if it occurs before target token
                head_det = next((c for c in head_n.children if c.dep_ in ("det", "poss") and c.i < t_tok.i), None)
                if head_det is None and det_tok is not None and det_tok.i < t_tok.i:
                    head_det = det_tok
                
                det_prefix = ""
                if head_det is not None:
                    d_norm = _normalize_det(head_det.text, det_tok_obj=head_det)
                    if d_norm:
                        det_prefix = f"{d_norm} "

                head_prep = next((c for c in head_n.children if c.dep_ == "prep" and c.i == head_n.i + 1 and c.text.lower() in ("for", "of", "in", "to", "on", "with", "against")), None)
                mid_comp = [_tok_surface(c) for c in head_n.children if c.dep_ in ("compound", "amod") and t_tok.i < c.i < head_n.i]
                mid_str = f"{' '.join(mid_comp)} " if mid_comp else ""
                head_tail = f"{mid_str}{_tok_surface(head_n)} {head_prep.text.lower()}" if head_prep is not None else f"{mid_str}{_tok_surface(head_n)}"
                return f"... {det_prefix}____ {head_tail}".strip()

            # 4c. Noun + bound preposition (`well in advance of the meeting`, `in advance of`, `the ____ of the copyright laws`, `a ____ of doctors`)
            if prep_after is not None:
                p_left = t_tok.head if (t_tok.dep_ == "pobj" and t_tok.head and t_tok.head.pos_ in ("ADP", "SCONJ") and t_tok.head.i < t_tok.i) else None
                if p_left is not None and not _is_forbidden(p_left.lemma_.lower()):
                    adv_left = next((c.text.lower() for c in p_left.children if c.dep_ == "advmod" and c.i < p_left.i), "")
                    if not adv_left and p_left.i > 0 and doc[p_left.i - 1].text.lower() in ("well", "far", "way", "right", "just", "long"):
                        adv_left = doc[p_left.i - 1].text.lower()
                    p_str = f"{adv_left} {p_left.text.lower()}".strip()
                    if det_tok is not None:
                        d_real = _normalize_det(det_tok.text, det_tok_obj=det_tok)
                        p_str = f"{p_str} {d_real}"
                    p_gov = p_left.head if (p_left.head and p_left.head.pos_ in ("VERB", "AUX") and p_left.head.i < p_left.i and not _is_forbidden(p_left.head.lemma_.lower())) else None
                    if p_gov is not None and p_left.i - p_gov.i <= 3:
                        gov_txt = p_gov.text.lower()
                        p_str = f"{gov_txt} {p_str}"
                    return f"... {p_str} ____ {prep_after.text.lower()}{_pobj_phrase(prep_after)}"
                d_pref = det_str or (default_noun_det if not is_pure_uncountable_target else "")
                return f"... {d_pref + ' ' if d_pref else ''}____ {prep_after.text.lower()}{_pobj_phrase(prep_after)}"

            # 4d. Subject of a lexical verb (`the ____ unfolds`, `____ will soon be paying higher airfares`)
            if t_tok.dep_ in ("nsubj", "nsubjpass") and t_tok.head and t_tok.head.pos_ in ("VERB", "AUX") and t_tok.head.i > t_tok.i:
                v_head = t_tok.head
                if v_head.pos_ == "AUX" or v_head.lemma_.lower() in ("be", "have", "do"):
                    # Sentence-initial existential/cataphoric 'The <noun> is that ...': preserve full clause-opener frame
                    # without leading '...' so the LLM never embeds it mid-clause (e.g. 'in the ____ is').
                    if (
                        (det_tok is not None and det_tok.i == 0 or t_tok.i == 0)
                        and v_head.lemma_.lower() == "be"
                        and v_head.i + 1 < len(doc)
                        and doc[v_head.i + 1].text.lower() == "that"
                    ):
                        d_cap = (det_str or "the").capitalize()
                        return f"{d_cap} ____ {v_head.text.lower()} that ..."
                elif v_head.lemma_.lower() not in cls._DELEXICAL_VERBS and not _is_forbidden(v_head.lemma_.lower()):
                    is_plural = is_plural_noun or t_tok.tag_ in ("NNS", "NNPS")
                    if det_tok is not None:
                        d_pref = _normalize_det(det_tok.text, det_tok_obj=det_tok)
                    elif is_plural or is_pure_uncountable_target:
                        d_pref = ""
                    else:
                        d_pref = det_str or "the"

                    # Collect auxiliary / modal / adverb chain between subject and verb
                    aux_mid = ""
                    if t_tok.i + 1 < v_head.i and v_head.i - t_tok.i <= 5:
                        mid_span = doc[t_tok.i + 1 : v_head.i]
                        if all(
                            t.pos_ in ("AUX", "PART", "ADV")
                            or t.dep_ in ("aux", "auxpass", "neg", "advmod")
                            for t in mid_span
                        ):
                            aux_mid = " ".join(t.text.lower() for t in mid_span) + " "

                    # Check for direct object / complement of v_head to avoid bleached frames like "... ____ paying"
                    tail_comp = ""
                    dobj = next((c for c in v_head.children if c.dep_ in ("dobj", "obj") and c.i > v_head.i), None)
                    if dobj is not None and dobj.is_alpha and not _is_forbidden(dobj.lemma_.lower()):
                        obj_mods = [
                            _tok_surface(c)
                            for c in dobj.children
                            if c.dep_ in ("det", "amod", "compound") and v_head.i < c.i < dobj.i
                        ]
                        obj_phrase = f"{' '.join(obj_mods)} {_tok_surface(dobj)}".strip()
                        tail_comp = f" {obj_phrase}"
                    elif prep_after_v := next((c for c in v_head.children if c.dep_ == "prep" and c.i == v_head.i + 1 and c.text.lower() in ("for", "to", "on", "in", "with", "from", "about", "against")), None):
                        tail_comp = f" {prep_after_v.text.lower()}"

                    prefix_d = f"{d_pref} " if d_pref else ""
                    return f"... {prefix_d}____ {aux_mid}{v_head.text.lower()}{tail_comp}"

            # 4e. Object of a preposition (`in ____`, `on the ____`, `under ____`)
            if t_tok.dep_ == "pobj" and t_tok.head and t_tok.head.dep_ == "prep":
                p_left = t_tok.head.text.lower()
                det_in_ex = _normalize_det(det_tok.text, det_tok_obj=det_tok) if det_tok is not None else ""
                # Sentence-initial prepositional adjunct followed by comma: "By convention, ...", "In theory, ..."
                is_initial_prep = (
                    t_tok.head.i == 0
                    or getattr(t_tok.head, "is_sent_start", False)
                    or (t_tok.head.i > 0 and doc[t_tok.head.i - 1].text in (".", "!", "?", ";"))
                )
                if is_initial_prep and t_end_i < len(doc) and doc[t_end_i].text in (",", ";"):
                    return f"{p_left.capitalize()} {det_in_ex + ' ' if det_in_ex else ''}____, ..."
                _gov = t_tok.head.head if (t_tok.head and t_tok.head.head and t_tok.head.head != t_tok.head) else None
                if (
                    _gov is not None
                    and _gov.pos_ in ("VERB", "ADJ", "NOUN")
                    and _gov.is_alpha
                    and _gov.lemma_.lower() not in cls._DELEXICAL_VERBS
                    and _gov.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                    and not _is_forbidden(_gov.lemma_.lower())
                ):
                    d_gov = t_tok.head.i - _gov.i
                    if 0 < d_gov <= 2:
                        return f"... {_gov.text.lower()} {p_left} {det_in_ex + ' ' if det_in_ex else ''}____"
                    if 2 < d_gov <= 5 and _gov.pos_ == "VERB":
                        # Check for intervening direct object: verb + dobj + prep + ____ (e.g. raised £200 for charity)
                        dobj = next((c for c in _gov.children if c.dep_ in ("dobj", "obj") and _gov.i < c.i < t_tok.head.i), None)
                        if dobj is not None and not _is_forbidden(dobj.lemma_.lower()):
                            obj_det = next((c.text.lower() for c in dobj.children if c.dep_ in ("det", "poss") and c.i < dobj.i), "")
                            obj_str = f"{obj_det} {_tok_surface(dobj)}".strip()
                            return f"... {_gov.text.lower()} {obj_str} {p_left} {det_in_ex + ' ' if det_in_ex else ''}____"
                return f"... {p_left} {det_in_ex + ' ' if det_in_ex else ''}____"

            is_plural = is_plural_noun or t_tok.tag_ in ("NNS", "NNPS")
            if det_tok is not None:
                d_pref = _normalize_det(det_tok.text, det_tok_obj=det_tok)
            elif is_plural or is_pure_uncountable_target:
                d_pref = ""
            else:
                d_pref = det_str or "the"

            _acl_v = next(
                (
                    c for c in t_tok.children
                    if c.dep_ in ("acl", "relcl") and c.pos_ == "VERB" and c.i > t_tok.i
                    and c.lemma_.lower() not in cls._DELEXICAL_VERBS
                    and not _is_forbidden(c.lemma_.lower())
                ),
                None,
            )
            if _acl_v is not None:
                rel_prp = next((c.text.lower() for c in _acl_v.children if c.dep_ in ("nsubj", "nsubjpass") and t_tok.i < c.i < _acl_v.i and c.text.lower() in ("who", "which", "that")), "")
                rel_pref = f"{rel_prp} " if rel_prp else ""
                acl_dobj = next((c for c in _acl_v.children if c.dep_ in ("dobj", "obj") and c.i > _acl_v.i), None)
                acl_tail = f" {_tok_surface(acl_dobj)}" if (acl_dobj and acl_dobj.is_alpha and not _is_forbidden(acl_dobj.lemma_.lower())) else ""
                pref_str = f"{d_pref} " if d_pref else ""
                return f"... {pref_str}____ {rel_pref}{_acl_v.text.lower()}{acl_tail}"
            pref_str = f"{d_pref} " if d_pref else ""
            return f"... {pref_str}____".strip()

        # 5. Adverb (`canonical_pos == "adv"`)
        if canonical_pos == "adv":
            # 5a. Sentence-initial discourse/conjunctive adverb followed by comma ('Similarly, we find...')
            if (
                (
                    t_tok.i == 0
                    or getattr(t_tok, "is_sent_start", False)
                    or (t_tok.i > 0 and doc[t_tok.i - 1].text in (".", "!", "?", ";"))
                )
                and t_tok.i + 1 < len(doc)
                and doc[t_tok.i + 1].text in (",", ";")
            ):
                return "____, ..."
            # 5b. `hence + the/poss + NP` ('hence the tight security' -> '... ____ the tight security')
            if (
                ft_low in ("hence", "thus", "notably", "especially", "particularly", "mainly", "mostly", "chiefly", "even", "only")
                and t_tok.i + 2 < len(doc)
                and doc[t_tok.i + 1].text.lower() in ("the", "his", "her", "their", "our", "its", "this", "that", "these", "those")
            ):
                _det_t = doc[t_tok.i + 1].text.lower()
                if doc[t_tok.i + 2].pos_ in ("NOUN", "PROPN") and not _is_forbidden(doc[t_tok.i + 2].lemma_.lower()):
                    return f"... ____ {_det_t} {doc[t_tok.i + 2].text.lower()}"
                if (
                    t_tok.i + 3 < len(doc)
                    and doc[t_tok.i + 2].pos_ == "ADJ"
                    and doc[t_tok.i + 3].pos_ in ("NOUN", "PROPN")
                    and not _is_forbidden(doc[t_tok.i + 3].lemma_.lower())
                ):
                    return f"... ____ {_det_t} {doc[t_tok.i + 2].text.lower()} {doc[t_tok.i + 3].text.lower()}"
            # 5c. Immediately following adjective (`looked vaguely familiar` -> `... ____ familiar`) or adjacent modified head (`Adv + Adj/Verb` or `Verb + Adv`)
            if (
                t_tok.i + 1 < len(doc)
                and doc[t_tok.i + 1].is_alpha
                and doc[t_tok.i + 1].pos_ == "ADJ"
                and not doc[t_tok.i + 1].text[:1].isupper()
                and doc[t_tok.i + 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS
                and not _is_forbidden(doc[t_tok.i + 1].lemma_.lower())
            ):
                return f"... ____ {doc[t_tok.i + 1].text.lower()}"
            if (
                t_tok.head
                and t_tok.head != t_tok
                and t_tok.head.is_alpha
                and t_tok.head.pos_ in ("VERB", "ADJ")
                and t_tok.head.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                and not _is_forbidden(t_tok.head.lemma_.lower())
                and abs(t_tok.i - t_tok.head.i) == 1
            ):
                if t_tok.i < t_tok.head.i:
                    return f"... ____ {t_tok.head.text.lower()}"
                return f"... {t_tok.head.text.lower()} ____"
            # 5d. Degree/temporal modifier immediately preceding and modifying adverb ('shortly thereafter' -> '... shortly ____')
            if (
                t_tok.i > 0
                and doc[t_tok.i - 1].is_alpha
                and doc[t_tok.i - 1].pos_ == "ADV"
                and (doc[t_tok.i - 1].head == t_tok or t_tok.head == doc[t_tok.i - 1])
                and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower())
                and doc[t_tok.i - 1].text.lower() not in ("not", "never", "also", "even", "just", "only")
            ):
                return f"... {doc[t_tok.i - 1].text.lower()} ____"
            # 5e. `Verb + Object (Pronoun/NP) + Clause-Final Adverb` ('We must stop him somehow.' -> '... stop someone ____', 'handle the chemical responsibly' -> '... handle something ____')
            if (
                t_tok.head
                and t_tok.head != t_tok
                and t_tok.head.is_alpha
                and t_tok.head.pos_ == "VERB"
                and t_tok.head.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                and not _is_forbidden(t_tok.head.lemma_.lower())
                and t_tok.i > t_tok.head.i
            ):
                _mid_obj = next(
                    (
                        c for c in t_tok.head.children
                        if c.dep_ in ("dobj", "obj", "attr") and t_tok.head.i < c.i < t_tok.i
                    ),
                    None,
                )
                if _mid_obj is not None:
                    _obj_slot = (
                        "[person/group]"
                        if (
                            _mid_obj.text.lower() in ("him", "her", "me", "us", "you", "someone", "somebody", "anyone", "everyone")
                            or _mid_obj.ent_type_ == "PERSON"
                        )
                        else "something"
                    )
                    return f"... {t_tok.head.text.lower()} {_obj_slot} ____"
                return f"... {t_tok.head.text.lower()} ____"
            # 5f. Adjective modifier immediately preceding adverb
            if (
                t_tok.i > 0
                and doc[t_tok.i - 1].is_alpha
                and doc[t_tok.i - 1].pos_ in ("ADV", "ADJ")
                and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower())
                and doc[t_tok.i - 1].text.lower() not in ("not", "never", "also", "even", "just", "only")
            ):
                return f"... {doc[t_tok.i - 1].text.lower()} ____"
            if (
                t_tok.i + 1 < len(doc)
                and doc[t_tok.i + 1].is_alpha
                and doc[t_tok.i + 1].pos_ in ("VERB", "ADJ")
                and doc[t_tok.i + 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS
                and not _is_forbidden(doc[t_tok.i + 1].lemma_.lower())
            ):
                return f"... ____ {doc[t_tok.i + 1].text.lower()}"
            if (
                t_tok.i > 0
                and doc[t_tok.i - 1].is_alpha
                and doc[t_tok.i - 1].pos_ in ("VERB", "ADJ")
                and doc[t_tok.i - 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS
                and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower())
            ):
                return f"... {doc[t_tok.i - 1].text.lower()} ____"
            if t_tok.i + 1 < len(doc) and doc[t_tok.i + 1].is_alpha and doc[t_tok.i + 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS and not _is_forbidden(doc[t_tok.i + 1].lemma_.lower()):
                return f"... ____ {doc[t_tok.i + 1].text.lower()}"
            if t_tok.i > 0 and doc[t_tok.i - 1].is_alpha and doc[t_tok.i - 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower()):
                return f"... {doc[t_tok.i - 1].text.lower()} ____"

        return None


