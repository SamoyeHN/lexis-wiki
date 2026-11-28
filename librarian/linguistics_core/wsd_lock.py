"""Neuro-symbolic WSD sense locking orchestrator mixin."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .wsd_scoring import WsdContext


class WsdLockMixin:
    """Determines the authoritative sense index in PedLex entry aligned to quote and definition."""

    @classmethod
    def _build_wsd_context(
        cls,
        entry: Dict[str, Any],
        target_pos: Optional[str] = None,
        target_p: Optional[str] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        frame_evidence: Optional[str] = None,
        discourse_context: Optional[str] = None,
        senses: Optional[List[Dict[str, Any]]] = None,
    ) -> WsdContext:
        if senses is None:
            senses = entry.get("senses", [])
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'so', 'because', 'as', 'until', 'while',
            'when', 'where', 'why', 'how', 'which', 'what', 'who', 'whom', 'whose',
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'upon', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
            'around', 'round', 'away', 'back', 'along', 'across', 'through', 'into', 'past', 'onto', 'alongside', 'behind',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'would', 'should', 'could', 'ought', 'can', 'may', 'might', 'will', 'shall', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'theirs',
            'his', 'her', 'hers', 'its', 'our', 'ours', 'your', 'yours', 'me', 'him', 'us', 'this', 'that', 'these', 'those',
            'more', 'most', 'many', 'much', 'some', 'any', 'other', 'another', 'such', 'very', 'even', 'also', 'too',
            'not', 'no', 'nor', 'neither', 'either', 'never'
        }

        q_tokens = (set(re.findall(r"\b[a-zA-Z]{3,}\b", (quote or "").lower())) - stopwords) if quote else set()
        d_tokens = (set(re.findall(r"\b[a-zA-Z]{3,}\b", (definition or "").lower())) - stopwords) if definition else set()

        # A1: a pattern word is evidenced by the quote when the quote carries the word
        # OR any legal inflection of it. 'firm ... attaches itself to' satisfies the
        # PedLex pattern 'attach something to something'; the old \battach\b regex did not.
        q_low_all = (quote or "").lower()
        q_word_tokens = set(re.findall(r"\b[a-z]+\b", q_low_all))
        # Ordered copy of the same tokens: a pattern frame is judged by how far apart its
        # words sit in the sentence, which a set cannot say.
        q_seq = re.findall(r"\b[a-z]+\b", q_low_all)

        def _quote_has_form(word_form: str) -> bool:
            if not word_form:
                return False
            if re.search(rf"\b{re.escape(word_form)}\b", q_low_all):
                return True
            return any(tok in cls.inflected_forms(word_form) for tok in q_word_tokens)

        best_idx = None
        best_score = -9999
        second_score = -9999
        eligible_senses = 0
        score_trace: List[Tuple[int, float]] = []

        # Homograph blocks : one row can hold 'right1 adj', 'right2 noun',
        # 'right3 verb'. A sense that sits in the block declaring the target POS is the
        # primary one for this query, so it must not lose a tie to ordering alone.
        block_poses: Dict[Any, Set[str]] = {}
        for h in entry.get("homographs") or []:
            h_num = h.get("homograph_num")
            h_codes = {cls._pos_code(h.get("pos"))} if h.get("pos") else set()
            for hp in (h.get("all_poses") or []):
                hpc = cls._pos_code(hp)
                if hpc:
                    h_codes.add(hpc)
            block_poses[h_num] = {c for c in h_codes if c}

        target_w_clean = (entry.get("word") or "").lower()
        headword_tokens = set(re.findall(r"\b[a-z]+\b", target_w_clean))
        target_inflections = set(cls.inflected_forms(target_w_clean)) | headword_tokens | {target_w_clean}
        for hw_t in headword_tokens:
            target_inflections.update(cls.inflected_forms(hw_t))

        has_pron_obj = False
        target_tok_morph_number = None
        target_tok_is_predicative = False
        target_tok_is_attributive = False
        target_v_has_dobj = False
        target_v_is_bare_active_intrans = False
        doc_q = None
        local_q_tokens: Set[str] = set(q_tokens)
        if quote and len(senses) > 1:
            try:
                nlp = cls.get_spacy()
                doc_q = nlp(cls._strip_lexical_quotes(quote))
                doc = doc_q
                matching_toks = [
                    t for t in doc
                    if t.lemma_.lower() == target_w_clean or t.text.lower() == target_w_clean or
                    cls.form_matches(t.text.lower(), target_w_clean) or
                    cls.form_matches(t.lemma_.lower(), target_w_clean) or
                    (headword_tokens and (t.lemma_.lower() in headword_tokens or t.text.lower() in headword_tokens))
                ]
                if matching_toks:
                    mt0 = matching_toks[0]
                    if len(doc) > 12:
                        local_idxs = {t.i for t in doc if abs(t.i - mt0.i) <= 5}
                        local_idxs.add(mt0.head.i)
                        local_idxs.update(ch.i for ch in mt0.children)
                        local_idxs.update(ch.i for ch in mt0.head.children)
                        local_idxs.add(mt0.head.head.i)
                        _local = {
                            w
                            for t in doc
                            if t.i in local_idxs
                            for w in (
                                t.text.lower(),
                                t.lemma_.lower(),
                                cls._surface_lemma(t.text.lower()) or "",
                            )
                            if w and w in q_tokens
                        }
                        if _local:
                            local_q_tokens = _local
                    has_pron_obj = any(ch.dep_ in ('dobj', 'obj') and ch.pos_ == 'PRON' for ch in mt0.children)
                    if mt0.pos_ in ('VERB', 'AUX'):
                        _measure_nouns = {'time', 'times', 'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years', 'hour', 'hours', 'minute', 'minutes', 'moment', 'way', 'ways', 'mile', 'miles', 'step', 'steps', 'lot', 'bit'}
                        _dobj_ch = [ch for ch in mt0.children if ch.dep_ in ('dobj', 'obj', 'dative') and ch.lemma_.lower() not in _measure_nouns]
                        if _dobj_ch or any(ch.dep_ in ('nsubjpass', 'nsubj:pass', 'auxpass') for ch in mt0.children):
                            target_v_has_dobj = True
                        elif (
                            mt0.dep_ not in ('relcl', 'acl', 'amod')
                            and not any(ch.dep_ in ('dobj', 'obj', 'dative', 'ccomp', 'xcomp', 'oprd', 'nsubjpass', 'auxpass', 'agent') for ch in mt0.children)
                        ):
                            target_v_is_bare_active_intrans = True
                    if mt0.pos_ == 'NOUN':
                        target_tok_morph_number = mt0.morph.get('Number')
                    if mt0.dep_ in ('acomp', 'oprd') or (
                        mt0.head != mt0
                        and mt0.head.lemma_.lower() in ('be', 'become', 'seem', 'feel', 'get', 'remain', 'stay', 'grow', 'turn', 'appear', 'look', 'sound')
                        and mt0.dep_ != 'amod'
                    ):
                        target_tok_is_predicative = True
                    elif mt0.dep_ == 'amod' and mt0.head.pos_ in ('NOUN', 'PROPN') and mt0.i < mt0.head.i:
                        target_tok_is_attributive = True
            except Exception:
                pass

        # Meta stopwords: filter out generic PedLex dictionary-definition noise from overlap counts
        #
        # Ultra-generic words carry no sense information in either direction: 'new' in
        # the quote and 'new' in the example 'smart new offices' is not evidence that
        # 'make smart choices' is about clothing, yet it was worth 8 points to the
        # clothes sense and helped it beat 'intelligent or sensible'.
        meta_stopwords = stopwords | target_inflections | {'time', 'likely', 'someone', 'something', 'somebody', 'things', 'people', 'used', 'make',
                                                          'new', 'old', 'own', 'real', 'thing', 'kind', 'sort', 'part', 'side', 'end',
                                                          'only', 'just', 'friend', 'friends'}

        # Query pedagogical sense cue index for authoritative contextual domain clues.
        #
        # The cue query is delexicalised with the SAME meta_stopwords as the
        # definition / example / signpost overlap families below.  Cue rows are mined
        # out of PedLex's own example sentences, so a token like 'new' or 'likely' is an
        # artefact of the dictionary corpus rather than evidence about the sense: 'new'
        # alone is attached to 2,207 cue rows spread over 1,859 headwords.  Querying the
        # cue index with such a token re-awards it at 15 points to whichever sense
        # happens to carry it, which is 15/8 the weight the example-overlap family gives
        # it and 15/6 the weight the definition-overlap family gives it.  That is what
        # made 'smart' score the CLOTHES sense in "make smart choices" (cue 'new'->#3),
        # 'streamline' score the SHAPE sense in "streamline the urban cluster"
        # (cue 'new'->#1), 'freedom' score the CHOICE sense (cue 'new'->#4) and
        # 'serious' score the ROMANCE sense (cue 'likely'->#6, a word that is also in
        # that sense's own definition and already paid for by q_def_overlap).
        cue_scores: Dict[int, float] = {}
        if quote and target_w_clean and not entry.get("_is_synthetic_pool"):
            cue_query_tokens = (local_q_tokens - meta_stopwords) - target_inflections
            if cue_query_tokens:
                cue_scores = cls.get_sense_cue_scores(target_w_clean, cue_query_tokens)

        # Precompute 2nd-Order Collocate Projection + Discourse Context Window tokens
        # Only activated when a sense carries a domain signpost (e.g. 'PLANT/ANIMAL') or taxonomic genus ('a type of ...')
        generic_signpost_stopwords = {
            "way", "saying", "make", "making", "give", "giving", "take", "taking",
            "get", "getting", "put", "putting", "use", "using", "have", "having",
            "keep", "keeping", "change", "changing", "move", "moving", "go", "going",
            "come", "coming", "do", "doing", "say", "speak", "speaking", "talk",
            "talking", "tell", "telling", "show", "showing", "start", "starting",
            "stop", "stopping", "good", "bad", "big", "small", "high", "low",
            "part", "piece", "amount", "number", "state", "time", "place", "form",
        }
        has_any_signpost_or_genus = any(
            bool((s_cand.get("signpost") or "").strip())
            or bool(re.match(r"^a\s+(?:type|kind|breed|variety|species|form|group)\s+of\b", (s_cand.get("definition") or "").strip().lower()))
            for s_cand in senses
        )
        second_order_domain_tokens: Set[str] = set()
        discourse_tokens: Set[str] = set()
        if len(senses) > 1 and has_any_signpost_or_genus:
            if discourse_context:
                for dt in re.findall(r"\b[a-zA-Z]{3,}\b", discourse_context.lower()):
                    if dt not in meta_stopwords and dt not in generic_signpost_stopwords:
                        discourse_tokens.add(dt)
                        d_lem = cls._surface_lemma(dt)
                        if d_lem and len(d_lem) >= 3 and d_lem not in meta_stopwords and d_lem not in generic_signpost_stopwords:
                            discourse_tokens.add(d_lem)
                        for ss in cls.get_wordnet_supersenses(d_lem or dt, pos="n"):
                            if "." in ss:
                                dom = ss.split(".", 1)[1].lower()
                                if len(dom) >= 3 and dom not in ("tops", "act", "state", "object", "attribute", "cognition", "change", "stative", "person", "group", "communication", "relation", "possession", "phenomenon", "quantity", "time", "location"):
                                    discourse_tokens.add(dom)
            if quote:
                try:
                    nlp = cls.get_spacy()
                    doc_proj = nlp(quote)
                    for tok_c in doc_proj:
                        tc_low = tok_c.text.lower()
                        tc_lem = tok_c.lemma_.lower()
                        if tc_low in target_inflections or tc_lem in target_inflections:
                            continue
                        if tok_c.pos_ not in ("NOUN", "PROPN", "VERB", "ADJ"):
                            continue
                        if tc_lem in meta_stopwords or tc_lem in generic_signpost_stopwords or len(tc_lem) < 3:
                            continue
                        # 1. WordNet supersense domain labels (e.g. noun.plant -> plant, noun.animal -> animal)
                        wn_p = "n" if tok_c.pos_ in ("NOUN", "PROPN") else ("v" if tok_c.pos_ == "VERB" else None)
                        if wn_p:
                            for ss in cls.get_wordnet_supersenses(tc_lem, pos=wn_p):
                                if "." in ss:
                                    dom = ss.split(".", 1)[1].lower()
                                    if len(dom) >= 3 and dom not in ("tops", "act", "state", "object", "attribute", "cognition", "change", "stative", "person", "group", "communication", "relation", "possession", "phenomenon", "quantity", "time", "location"):
                                        second_order_domain_tokens.add(dom)
                        # 2. Primary PedLex sense signpost & taxonomic/bracketed domain of content collocate (e.g. 'wild' -> [plants/animals], 'crossbreed' -> 'breed of plant or animal')
                        col_entry = cls.get_pedlex_entry(tc_lem) or cls.get_pedlex_entry(tc_low)
                        if col_entry:
                            for cs in (col_entry.get("senses") or [])[:2]:
                                csp = (cs.get("signpost") or "").lower()
                                for st in re.findall(r"\b[a-z]{3,}\b", csp):
                                    if st not in meta_stopwords and st not in generic_signpost_stopwords:
                                        second_order_domain_tokens.add(st)
                                        s_lem = cls._surface_lemma(st)
                                        if s_lem and len(s_lem) >= 3 and s_lem not in meta_stopwords and s_lem not in generic_signpost_stopwords:
                                            second_order_domain_tokens.add(s_lem)
                                cdef = (cs.get("definition") or "").lower()
                                # Extract bracketed domain/subject labels (e.g. '[plants/animals]', '[biology]') or taxonomic domain nouns ('plant or animal')
                                for brk in re.findall(r"\[([^\]]+)\]", cdef):
                                    for dt in re.findall(r"\b[a-z]{3,}\b", brk):
                                        if dt not in meta_stopwords and dt not in generic_signpost_stopwords:
                                            second_order_domain_tokens.add(dt)
                                            d_lem = cls._surface_lemma(dt)
                                            if d_lem and len(d_lem) >= 3 and d_lem not in meta_stopwords and d_lem not in generic_signpost_stopwords:
                                                second_order_domain_tokens.add(d_lem)
                                for tax_m in re.findall(r"\b(?:type|kind|breed|variety|species|form|group|class)\s+of\s+([a-z,\s]+?)(?:\bthat\b|\bwhich\b|\bwith\b|\bin\b|\bor\b|$)", cdef):
                                    for dt in re.findall(r"\b[a-z]{3,}\b", tax_m):
                                        if dt not in meta_stopwords and dt not in generic_signpost_stopwords:
                                            second_order_domain_tokens.add(dt)
                                            d_lem = cls._surface_lemma(dt)
                                            if d_lem and len(d_lem) >= 3 and d_lem not in meta_stopwords and d_lem not in generic_signpost_stopwords:
                                                second_order_domain_tokens.add(d_lem)
                                for bio_m in re.findall(r"\b(plant|plants|animal|animals|crop|crops|disease|diseases|organism|organisms|bacteria|virus|computer|computers|electronic|screen|screens|device|devices|phone|telephone|mobile|digital|internet|software|wireless)\b", cdef):
                                    second_order_domain_tokens.add(bio_m)
                                    b_lem = cls._surface_lemma(bio_m)
                                    if b_lem:
                                        second_order_domain_tokens.add(b_lem)
                except Exception:
                    pass

        # Map phrasal verb senses to their mandatory phrasal particles/words (e.g. 'pay back' -> ['back'])
        _pv_slot_stop = {"something", "somebody", "someone", "sth", "sb", "<->", "\u2194", "oneself", "one's", "etc"}
        pv_sense_map: Dict[int, List[str]] = {}
        for pv in entry.get("phrasal_verbs") or []:
            phr = re.sub(r"[\(\)\[\]]", " ", pv.get("phrase", ""))
            p_words = [
                w for w in phr.lower().split()
                if w != target_w_clean and not cls.form_matches(w, target_w_clean) and w not in _pv_slot_stop
            ]
            if p_words:
                for pvs in pv.get("senses") or []:
                    p_defn = pvs.get("definition")
                    for s_idx, s_cand in enumerate(senses):
                        if (
                            s_cand.get("definition") == p_defn
                            and not s_cand.get("sensenum")
                            and (s_cand.get("homograph_num") in (None, 0) or s_cand.get("from_phrasal_verb"))
                        ):
                            pv_sense_map[s_idx] = p_words

        return WsdContext(
            entry=entry,
            target_pos=target_pos,
            target_p=target_p,
            definition=definition,
            quote=quote,
            frame_evidence=frame_evidence,
            stopwords=stopwords,
            q_tokens=q_tokens,
            d_tokens=d_tokens,
            q_low_all=q_low_all,
            q_word_tokens=q_word_tokens,
            q_seq=q_seq,
            block_poses=block_poses,
            target_w_clean=target_w_clean,
            headword_tokens=headword_tokens,
            target_inflections=target_inflections,
            has_pron_obj=has_pron_obj,
            target_tok_morph_number=target_tok_morph_number,
            target_tok_is_predicative=target_tok_is_predicative,
            target_tok_is_attributive=target_tok_is_attributive,
            target_v_has_dobj=target_v_has_dobj,
            target_v_is_bare_active_intrans=target_v_is_bare_active_intrans,
            doc_q=doc_q,
            meta_stopwords=meta_stopwords,
            cue_scores=cue_scores,
            generic_signpost_stopwords=generic_signpost_stopwords,
            second_order_domain_tokens=second_order_domain_tokens,
            discourse_tokens=discourse_tokens,
            pv_sense_map=pv_sense_map,
            senses=senses,
            discourse_context=discourse_context,
            local_q_tokens=local_q_tokens,
        )

    @classmethod
    def _lock_sense(
        cls,
        entry: Dict[str, Any],
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        frame_evidence: Optional[str] = None,
        target_pos: Optional[str] = None,
        return_confidence: bool = False,
        return_scores: bool = False,
        discourse_context: Optional[str] = None,
    ):
        """Determines the authoritative sense index in PedLex entry aligned to quote and definition.
        Calculates content-word overlap across examples, patterns, definitions, signposts,
        and enforces grammar/syntactic frame agreement.

        With return_confidence=True the result is (index, score, margin, confidence) instead of
        the index alone, so every existing caller keeps the single-value contract it expects.
        """
        senses = entry.get("senses", [])
        if not senses:
            return None

        # this gate used to be `target_p not in s_pos`, a substring test
        # against single-letter codes. Both sides are canonical codes compared for
        # equality now, so a noun query can no longer admit an adverb sense.
        target_p = cls._pos_code(target_pos)

        # Exact / clean definition override: if definition was passed directly from PedLex extraction
        # Pass 1: exact definition equality across all POS-compatible senses
        # Pass 2: prefix match fallback if no exact equality match exists
        if definition:
            clean_def = re.sub(r"\[.*?\]", "", definition).strip().lower()
            if clean_def:
                for idx, s in enumerate(senses):
                    if target_p and not cls._sense_matches_pos(s, target_p):
                        continue
                    s_defn_clean = re.sub(r"\[.*?\]", "", s.get("definition") or "").strip().lower()
                    if s_defn_clean and s_defn_clean == clean_def:
                        if return_scores:
                            return (idx, 100.0, 100.0, "high", [(idx, 100.0)])
                        if return_confidence:
                            return (idx, 100.0, 100.0, "high")
                        return idx
                for idx, s in enumerate(senses):
                    if target_p and not cls._sense_matches_pos(s, target_p):
                        continue
                    s_defn_clean = re.sub(r"\[.*?\]", "", s.get("definition") or "").strip().lower()
                    if s_defn_clean and (s_defn_clean.startswith(clean_def) or clean_def.startswith(s_defn_clean)):
                        if return_scores:
                            return (idx, 100.0, 100.0, "high", [(idx, 100.0)])
                        if return_confidence:
                            return (idx, 100.0, 100.0, "high")
                        return idx

        ctx = cls._build_wsd_context(
            entry=entry,
            target_pos=target_pos,
            target_p=target_p,
            definition=definition,
            quote=quote,
            frame_evidence=frame_evidence,
            discourse_context=discourse_context,
            senses=senses,
        )

        best_idx = None
        best_score = -9999.0
        second_score = -9999.0
        eligible_senses = 0
        score_trace: List[Tuple[int, float]] = []

        for i, s in enumerate(senses):
            s_pos = cls._pos_code(s.get("pos"))
            if target_p and not cls._sense_matches_pos(s, target_p):
                continue
            eligible_senses += 1
            score = cls._score_single_sense(s, i, ctx)
            score_trace.append((i, score))

            if score > best_score:
                second_score = best_score
                best_score = score
                best_idx = i
            elif score > second_score:
                second_score = score

        if cls.SENSE_DEBUG and score_trace:
            for i, sc in sorted(score_trace, key=lambda r: -r[1])[:8]:
                print(
                    f"[sense-lock] {ctx.target_w_clean} pos={target_pos} quote={quote!r} "
                    f"sense {i} score {sc:.3f} :: {(senses[i].get('definition') or '')[:70]}"
                )

        if best_idx is not None:
            idx = best_idx
        elif target_p:
            # Check if any sense matches target_pos
            idx = next((i for i, s in enumerate(senses) if cls._sense_matches_pos(s, target_p)), None)
            if idx is None:
                # If entry is an entity or has only 1 sense carrying no POS label, fall back to sense 0
                if len(senses) == 1 and (entry.get("kind") == "entity" or not senses[0].get("pos")):
                    idx = 0
        else:
            idx = 0 if senses else None
        score = best_score if best_idx is not None else 0.0
        margin = (
            (best_score - second_score)
            if best_idx is not None and second_score > -9999
            else 0.0
        )
        # Nothing to disambiguate: a single sense, or a POS gate that matched exactly one
        # sense, is as certain as the entry itself, so the floors do not apply. A gate that
        # matched nothing across several senses is a guess and stays under the floors.
        unambiguous = eligible_senses == 1 or (eligible_senses == 0 and len(senses) == 1)
        confident = unambiguous or (
            score >= cls.SENSE_CONFIDENCE_FLOOR and margin >= cls.SENSE_MARGIN_FLOOR
        )
        if return_scores:
            return idx, score, margin, ("high" if confident else "low"), score_trace
        if not return_confidence:
            return idx
        return idx, score, margin, ("high" if confident else "low")

    @classmethod
    def sense_confidence(
        cls,
        entry: Dict[str, Any],
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        target_pos: Optional[str] = None,
    ) -> Dict[str, Any]:
        """B1: how firmly the quote and the curriculum definition pin down which sense this item
        teaches. 'low' means the sense was reached by ordering, not by evidence, and the caller
        must not ship that definition as authoritative.
        """
        senses = (entry or {}).get("senses") or []
        if not senses:
            return {
                "index": None,
                "score": 0.0,
                "margin": 0.0,
                "confidence": "low",
                "eligible_senses": 0,
            }
        target_p = cls._pos_code(target_pos)
        eligible = [
            s for s in senses if not (target_p and not cls._sense_matches_pos(s, target_p))
        ]
        idx, score, margin, confidence = cls._lock_sense(
            entry,
            definition=definition,
            quote=quote,
            target_pos=target_pos,
            return_confidence=True,
        )
        return {
            "index": idx,
            "score": score,
            "margin": margin,
            "confidence": confidence,
            "eligible_senses": len(eligible),
        }


__all__ = ["WsdLockMixin"]
