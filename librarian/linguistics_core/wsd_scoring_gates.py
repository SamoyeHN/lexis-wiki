"""WSD scoring stage 1: Invariant gates, register, domain, and lexical overlap."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from .wsd_scoring_context import WsdContext, WsdSenseInfo


class WsdScoringGatesMixin:
    """Stage 1 scoring: homograph block, particles, morphology, register, domain, and token overlap."""

    @classmethod
    def _quote_has_form(cls, word_form: str, ctx: WsdContext) -> bool:
        if not word_form:
            return False
        if re.search(rf"{re.escape(word_form)}", ctx.q_low_all):
            return True
        return any(tok in cls.inflected_forms(word_form) for tok in ctx.q_word_tokens)

    @classmethod
    def _score_sense_gates_and_lexical(
        cls, s: Dict[str, Any], i: int, ctx: WsdContext
    ) -> Tuple[float, WsdSenseInfo]:
        entry = ctx.entry
        target_pos = ctx.target_pos
        target_p = ctx.target_p
        definition = ctx.definition
        quote = ctx.quote
        frame_evidence = ctx.frame_evidence
        stopwords = ctx.stopwords
        q_tokens = ctx.q_tokens
        d_tokens = ctx.d_tokens
        q_low_all = ctx.q_low_all
        q_word_tokens = ctx.q_word_tokens
        q_seq = ctx.q_seq
        block_poses = ctx.block_poses
        target_w_clean = ctx.target_w_clean
        headword_tokens = ctx.headword_tokens
        target_inflections = ctx.target_inflections
        has_pron_obj = ctx.has_pron_obj
        target_tok_morph_number = ctx.target_tok_morph_number
        target_tok_is_predicative = ctx.target_tok_is_predicative
        target_tok_is_attributive = ctx.target_tok_is_attributive
        target_v_has_dobj = ctx.target_v_has_dobj
        target_v_is_bare_active_intrans = ctx.target_v_is_bare_active_intrans
        doc_q = ctx.doc_q
        meta_stopwords = ctx.meta_stopwords
        cue_scores = ctx.cue_scores
        generic_signpost_stopwords = ctx.generic_signpost_stopwords
        second_order_domain_tokens = ctx.second_order_domain_tokens
        discourse_tokens = ctx.discourse_tokens
        pv_sense_map = ctx.pv_sense_map
        senses = ctx.senses
        discourse_context = ctx.discourse_context

        s_pos = cls._pos_code(s.get("pos"))

        def _quote_has_form(word_form: str) -> bool:
            if not word_form:
                return False
            if re.search(rf"\b{re.escape(word_form)}\b", q_low_all):
                return True
            return any(tok in cls.inflected_forms(word_form) for tok in q_word_tokens)

        eff_pos = target_pos or s.get("pos")
        eff_p = target_p or s_pos

        score = 0

        # 0. Homograph block agreement: the sense that lives in the block which
        # declares the target POS outranks a same-POS sense borrowed from another
        # block, instead of losing to whichever sense happens to come first.
        if target_p and target_p in block_poses.get(s.get("homograph_num"), set()):
            score += 10
        elif target_p and s_pos == target_p:
            score += 4

        # 0a. Phrasal Verb Particle Gate:
        # If this sense belongs to a phrasal verb block with a mandatory particle (e.g. 'pay back' -> 'back', 'pay off' -> 'off', 'pay out' -> 'out'),
        # the quote MUST contain the particle within a local window of the verb; otherwise this phrasal sense cannot hijack base verb usages.
        if i in pv_sense_map:
            req_particles = pv_sense_map[i]
            if all(
                any(cls._quote_has_frame(q_seq, [target_w_clean, alt], window=4) for alt in rp.split("/") if alt)
                for rp in req_particles
            ):
                score += 45
            else:
                score -= 60
        elif eff_p == "v":
            # Also guard verb senses whose definition starts with 'if ... <verb>s <particle>' when the particle is absent
            m_pv_def = re.match(
                rf"^if\s+[^,]+?\s+(?:{re.escape(target_w_clean)}[a-z]*)\s+(up|down|out|off|away|back|over|through|around|round|apart|aside|ahead|along|about)\b",
                (s.get("definition") or "").strip().lower(),
            )
            if m_pv_def and not any(cls.form_matches(tok, m_pv_def.group(1)) for tok in q_seq):
                score -= 60

        # 0b. Morphological Number / Countability & Syntactic Slot Gate:
        # If target noun in quote is physically plural ('causes', 'customs'),
        # it cannot match an exclusively [uncountable] sense if a [countable] sense is available.
        # Guard invariant -s / -ics nouns ('electronics', 'physics', 'economics', 'news', 'means', 'species')
        # where spaCy tags Number=Plur on morphologically invariant -s headwords.
        s_gram = (s.get("grammar") or "").lower()
        if target_tok_morph_number and "Plur" in target_tok_morph_number and not target_w_clean.endswith("s"):
            if "[uncountable]" in s_gram and "[countable]" not in s_gram:
                score -= 60
            if "[singular]" in s_gram and "[countable]" not in s_gram and "[plural]" not in s_gram:
                score -= 60
        elif target_tok_morph_number and "Sing" in target_tok_morph_number:
            if "[plural]" in s_gram and "[singular]" not in s_gram and "[countable]" not in s_gram:
                score -= 60
        # For -ics nouns ('electronics', 'mechanics', 'statistics', 'politics', 'ethics', 'acoustics'),
        # disambiguate between [uncountable] academic discipline/industry/science vs [plural] concrete objects/facts
        if eff_p == "n" and target_w_clean.endswith("ics"):
            _ics_def_low = (s.get("definition") or "").lower()
            _has_academic_or_industry_ctx = bool(
                q_tokens & {
                    "student", "students", "engineer", "engineers", "engineering", "science", "sciences",
                    "professor", "department", "major", "degree", "course", "courses", "study", "studies",
                    "industry", "company", "companies", "firm", "firms", "sector", "field", "research",
                    "technology", "specialist", "specialists", "expert", "experts", "technician",
                }
            )
            if _has_academic_or_industry_ctx:
                if "[uncountable]" in s_gram or any(k in _ics_def_low for k in ("the study", "the science", "industry", "scientific study", "subject")):
                    score += 55
                elif "[plural]" in s_gram and "[uncountable]" not in s_gram:
                    score -= 45
        if target_tok_is_predicative and "[only before noun]" in s_gram:
            s_ex_pat_text = " ".join(list(s.get("examples") or []) + list(s.get("patterns") or [])).lower()
            s_ex_pat_toks = set(re.findall(r"\b[a-z]{4,}\b", s_ex_pat_text)) - meta_stopwords
            for _t_ex in list(s_ex_pat_toks):
                _l_ex = cls._surface_lemma(_t_ex)
                if _l_ex and len(_l_ex) >= 4:
                    s_ex_pat_toks.add(_l_ex)
            if not (s_ex_pat_toks & (q_tokens - meta_stopwords)):
                score -= 60
        elif target_tok_is_attributive and "[not before noun]" in s_gram:
            score -= 60
        if eff_p == "v" and s_gram:
            _s_has_intrans = bool(re.search(r"\bintransitive\b", s_gram))
            _s_has_trans = bool(re.search(r"\btransitive\b", s_gram))
            if target_v_has_dobj and _s_has_intrans and not _s_has_trans:
                score -= 45
            elif target_v_is_bare_active_intrans and _s_has_trans and not _s_has_intrans:
                score -= 35

        # 0c. Spoken Register Invariant Gate:
        # If sense is marked as [spoken] / conversational formula, penalize in standard narrative or exposition
        # unless the quote is explicitly in quotation marks or dialogue.
        if ("spoken" in (s.get("register") or "").lower() or "[spoken]" in (s.get("grammar") or "").lower()):
            is_direct_speech = any(q_char in quote for q_char in ('"', "'", "“", "”", "‘", "’")) if quote else False
            if not is_direct_speech:
                score -= 50

        # 0d. Rare / Technical Domain Gate (Medical / Law / Technical):
        # Penalize highly technical domain senses (e.g. 'illness/symptom' for present, chemistry/physics senses)
        # unless the context quote actually contains relevant domain lexical triggers.
        s_reg = (s.get("register") or "").lower()
        s_signpost = (s.get("signpost") or "").lower()
        s_gram_str = (s.get("grammar") or "").lower()
        is_med_sense = s_reg == "medical" or "[medical]" in s_gram_str or s_signpost in ("illness", "medical", "disease")
        if is_med_sense:
            has_med_trigger = any(w in q_low_all for w in (
                "illness", "disease", "symptom", "patient", "doctor", "hospital", "medicine",
                "syndrome", "fever", "pain", "clinic", "clinical", "infection", "treatment",
                "suffer", "suffered", "suffering", "mild", "severe", "fatal", "heart", "brain",
                "blood", "attack", "paralyz", "recover", "health"
            ))
            if not has_med_trigger:
                score -= 50

        is_law_sense = s_reg in ("law", "legal") or "[law]" in s_gram_str or "[legal]" in s_gram_str or s_signpost in ("law", "legal", "court")
        if is_law_sense:
            has_law_trigger = any(w in q_low_all for w in ("court", "judge", "jury", "trial", "lawyer", "attorney", "lawsuit", "legal", "statute", "prosecution", "defendant", "verdict"))
            if not has_law_trigger:
                score -= 50

        is_tech_sense = (
            s_reg in ("chemistry", "physics", "biology", "computing")
            or (s_reg == "technical" and i > 0)
            or any(tag in s_gram_str for tag in ("[chemistry]", "[physics]", "[computing]"))
            or ("[technical]" in s_gram_str and i > 0)
            or s_signpost in ("chemistry", "physics", "technical", "chemical")
        )
        if is_tech_sense:
            has_tech_trigger = any(w in q_low_all for w in (
                "chemical", "acid", "substance", "atom", "molecule", "reaction", "physics",
                "laboratory", "voltage", "circuit", "alkali", "gas", "metal", "compound",
                "technology", "technological", "scientific", "science", "engineering",
                "robot", "robotics", "inventor", "inventors", "technician", "technicians",
                "experiment", "printing", "device", "machine", "software", "hardware",
                "digital", "electronic", "electronics", "computer"
            ))
            if not has_tech_trigger:
                score -= 50

        is_grammar_sense = (
            s_reg in ("grammar", "linguistics")
            or "[grammar]" in s_gram_str
            or s_signpost in ("grammar", "linguistics")
        )
        if is_grammar_sense:
            has_grammar_trigger = any(w in q_low_all for w in (
                "grammar", "grammatical", "sentence", "verb", "noun", "adjective", "adverb",
                "pronoun", "preposition", "clause", "phrase", "tense", "plural", "singular",
                "syntax", "linguistic", "linguistics", "conjugat", "inflect"
            ))
            if not has_grammar_trigger:
                score -= 60

        # 0e. Pejorative / Disapproval Stance Gate:
        # Senses explicitly marked 'used to show disapproval' (e.g. 'commercialize' sense 0: 'to be more concerned
        # with making money from something than about its quality – used to show disapproval') require a critical/pejorative
        # context or default absence of constructive enabling framing ('help ... design, produce, and commercialize their ideas').
        s_defn_raw_low = (s.get("definition") or "").lower()
        if "show disapproval" in s_defn_raw_low or "disapproving" in s_reg:
            has_pejorative_cue = any(w in q_low_all for w in (
                "too", "overly", "merely", "corrupt", "ruin", "ruined", "spoil", "spoiled", "spoilt",
                "cheap", "greed", "greedy", "profit", "money", "exploit", "bad", "poor", "excessive", "shameless"
            ))
            has_constructive_frame = any(w in q_low_all for w in (
                "help", "helps", "helped", "helping", "enable", "enables", "enabled", "allow", "allows",
                "support", "encourage", "develop", "design", "produce", "innovate", "innovation", "launch"
            ))
            if has_constructive_frame and not has_pejorative_cue:
                score -= 50

        # 1. Quote token overlap against sense examples, patterns & units
        ex_tokens = set()
        for ex in s.get("examples", []):
            ex_tokens.update(re.findall(r"\b[a-zA-Z]{3,}\b", ex.lower()))
        for pat in list(s.get("patterns") or []) + list(s.get("units") or []):
            ex_tokens.update(re.findall(r"\b[a-zA-Z]{3,}\b", pat.lower()))
        
        # meta_stopwords is computed once above, before the cue-index query, so
        # that the cue layer and the overlap families delexicalise identically.

        ex_overlap = len((q_tokens - meta_stopwords) & (ex_tokens - meta_stopwords))
        score += ex_overlap * 8

        # 2. Quote token overlap against sense definition itself
        s_defn = s.get("definition", "").lower()
        if re.match(r"^(?:the\s+)?(?:past\s+participle|past\s+tense|present\s+participle|plural|comparative|superlative)\s+of\b", s_defn.strip()):
            score -= 120
        s_def_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", s_defn)) - meta_stopwords
        # A definition that merely restates the headword ('a smart person is wearing neat
        # attractive clothes') proves nothing about which sense the quote uses, yet every
        # sense written that way collected the same 6 points for it.
        headword = (entry.get("word") or "").lower()
        if headword:
            s_def_tokens = {t for t in s_def_tokens if not cls.form_matches(t, headword) and t not in target_inflections}
        q_def_overlap = len((q_tokens - meta_stopwords) & s_def_tokens)
        score += q_def_overlap * 6

        # 3. Input definition token overlap (if provided)
        if definition:
            in_def_norm = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", definition.strip()).strip().lower()
            s_fmt_norm = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", (cls.format_ldoce_definition(s) or "").strip()).strip().lower()
            if in_def_norm and (in_def_norm == s_defn.strip() or in_def_norm == s_fmt_norm):
                score += 200
            elif d_tokens and s_def_tokens:
                overlap_ratio = len(d_tokens & s_def_tokens) / max(1, min(len(d_tokens), len(s_def_tokens)))
                if overlap_ratio >= 0.75 and len(d_tokens & s_def_tokens) >= 2:
                    score += 100
                else:
                    score += len(d_tokens & s_def_tokens) * 5
        else:
            def_overlap = len(d_tokens & s_def_tokens)
            score += def_overlap * 5

        # 4. Signpost & Taxonomic Genus match (1st-order quote/def + convergent 2nd-order collocate & discourse domain match)
        convergent_domain_tokens = second_order_domain_tokens & discourse_tokens
        sp = (s.get("signpost") or "").lower().strip()
        if sp:
            sp_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", sp)) - meta_stopwords
            sp_lemmas = {cls._surface_lemma(t) or t for t in sp_tokens}
            all_sp_tokens = (sp_tokens | sp_lemmas) - meta_stopwords - generic_signpost_stopwords
            # If the entire signpost is a single domain word like 'time' or 'place', preserve it
            if not all_sp_tokens and sp in ("time", "place", "state", "form", "number", "amount", "part"):
                all_sp_tokens = {sp}
            if all_sp_tokens & (q_tokens | d_tokens):
                score += 15
            elif all_sp_tokens & convergent_domain_tokens:
                # Convergent 2nd-Order Collocate + Discourse Context Window match:
                # Both sentence collocates (e.g. 'wild', 'crossbred', 'domesticated', 'variety') AND
                # neighboring discourse sentences (+/- 2 sentences) project the exact same domain token (e.g. 'plant'/'animal')
                score += 24
        # Taxonomic genus reinforcement (e.g. "a type of animal, plant, or disease") when 2nd-order collocates and discourse agree
        m_genus = re.match(r"^a\s+(?:type|kind|breed|variety|species|form)\s+of\s+([a-z,\s]+?)(?:\bthat\b|\bwhich\b|\bwith\b|\bin\b|$)", s_defn.strip())
        if m_genus and convergent_domain_tokens:
            genus_toks = set(re.findall(r"\b[a-z]{3,}\b", m_genus.group(1))) - meta_stopwords - generic_signpost_stopwords - {"or", "and", "something", "someone", "person", "thing", "things", "people"}
            if genus_toks and (genus_toks & convergent_domain_tokens):
                score += 12
        if (
            second_order_domain_tokens
            and s_def_tokens
            and eff_p == "n"
            and (
                bool(m_genus)
                or (sp and sp not in ("controlled by a computer", "computer", "machine"))
                or bool(convergent_domain_tokens)
            )
        ):
            _tech_bio_dom = second_order_domain_tokens & {
                "computer", "computers", "electronic", "screen", "screens", "device", "devices",
                "phone", "telephone", "mobile", "digital", "internet", "software", "wireless",
                "plant", "plants", "animal", "animals", "crop", "crops", "disease", "diseases",
                "space", "astronaut", "astronauts", "float", "floating", "gravity", "orbit", "spacecraft",
            }
            if _tech_bio_dom & s_def_tokens:
                score += 14 * len(_tech_bio_dom & s_def_tokens)


        s_info = WsdSenseInfo(
            s_pos=s_pos,
            eff_pos=eff_pos,
            eff_p=eff_p,
            s_defn=s_defn,
            s_def_tokens=s_def_tokens,
            sp=sp,
            s_gram=s_gram,
        )
        return float(score), s_info


__all__ = ["WsdScoringGatesMixin"]
