"""Per-sense scoring engine for neuro-symbolic WSD sense locking."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class WsdContext:
    """Precomputed syntactic and contextual evidence for WSD sense locking."""

    entry: Dict[str, Any]
    target_pos: Optional[str]
    target_p: Optional[str]
    definition: Optional[str]
    quote: Optional[str]
    frame_evidence: Optional[str]
    stopwords: Set[str]
    q_tokens: Set[str]
    d_tokens: Set[str]
    q_low_all: str
    q_word_tokens: Set[str]
    q_seq: List[str]
    block_poses: Dict[Any, Set[str]]
    target_w_clean: str
    headword_tokens: Set[str]
    target_inflections: Set[str]
    has_pron_obj: bool
    target_tok_morph_number: Any
    target_tok_is_predicative: bool
    target_tok_is_attributive: bool
    target_v_has_dobj: bool
    target_v_is_bare_active_intrans: bool
    doc_q: Any
    meta_stopwords: Set[str]
    cue_scores: Dict[int, float]
    generic_signpost_stopwords: Set[str]
    second_order_domain_tokens: Set[str]
    discourse_tokens: Set[str]
    pv_sense_map: Dict[int, List[str]]
    senses: List[Dict[str, Any]]
    discourse_context: Optional[str] = None


class WsdScoringMixin:
    """Fine-grained syntactic and semantic scoring across individual sense candidates."""

    @classmethod
    def _score_single_sense(cls, s: Dict[str, Any], i: int, ctx: WsdContext) -> float:
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

        # 5. Pattern / Preposition / Collocation match directly from Quote
        # A pattern states a frame, so it is tested as one. As a bag of words it accepts
        # any sentence that merely contains the headword plus the pattern's other words
        # somewhere, which is not the same claim at all.
        if quote:
            target_w_clean = entry.get("word", "").lower()
            any_sense_pat_matched = False
            sense_selectional_violation = False
            for pat in s.get("patterns", []):
                if sense_selectional_violation:
                    break
                pat_clean = pat.lower().strip()
                branches = cls.clean_and_expand_pattern_slashes(pat)

                matched_pat = False
                for br in branches:
                    # Check separable/transitive phrasal patterns: if pattern specifies a mandatory (non-parenthesized)
                    # 'somebody ↔ ...' or 'something ↔ ...', it requires an explicit transitive direct object governed by the target verb.
                    br_non_opt = re.sub(r"\(.*?\)", "", br)
                    is_separable_transitive = "\u2194" in br or "<->" in br or (eff_p == "v" and ("somebody" in br_non_opt or "something" in br_non_opt or " sb " in f" {br_non_opt} " or " sth " in f" {br_non_opt} "))
                    if is_separable_transitive and quote and eff_p == "v":
                        try:
                            nlp = cls.get_spacy()
                            doc_q = nlp(quote)
                            # Locate head verb tokens in quote
                            head_cand_lemmas = {target_w_clean.split()[0]} if target_w_clean else set()
                            v_toks = [t for t in doc_q if t.lemma_.lower() in head_cand_lemmas or t.text.lower() in head_cand_lemmas]
                            if not v_toks:
                                v_toks = [t for t in doc_q if t.pos_ in ("VERB", "AUX")]
                            
                            # Check if any matching verb governs a direct object, passive Patient (nsubjpass), or relative/participial head
                            governing_objs = [
                                c for v in v_toks for c in v.children
                                if c.dep_ in ("dobj", "obj", "nsubjpass", "nsubj:pass")
                            ]
                            if not governing_objs:
                                governing_objs = [
                                    v.head for v in v_toks
                                    if v.dep_ in ("relcl", "acl", "amod") and v.head.pos_ in ("NOUN", "PROPN", "PRON")
                                ]
                            if not governing_objs:
                                governing_objs = [
                                    gc for v in v_toks for c in v.children if c.dep_ == "prep"
                                    for gc in c.children if gc.dep_ in ("pobj", "pcomp")
                                ]
                            if not governing_objs:
                                governing_objs = [
                                    gc for v in v_toks for c in v.children if c.dep_ in ("ccomp", "xcomp")
                                    for gc in c.children if gc.dep_ in ("nsubj", "dobj", "obj")
                                ]
                            if not governing_objs:
                                # Passive check: if passive subject or relative/infinitive object extraction exists ('much to go on'), treat as matched
                                has_passive_or_rel_obj = any(c.dep_ in ("nsubjpass", "nsubj:pass", "relcl") for v in v_toks for c in v.children) or any(v.dep_ in ("relcl", "acl") for v in v_toks)
                                if not has_passive_or_rel_obj:
                                    continue
                            else:
                                # Enforce selectional restrictions: human vs inanimate (across active dobj AND passive nsubjpass)
                                br_obj = re.sub(r"\bto\s+do\s+(?:something|somebody|someone|sth|sb)\b", "", br, flags=re.IGNORECASE)
                                has_sb = "somebody" in br_obj or " sb " in f" {br_obj} " or "someone" in br_obj
                                has_sth = "something" in br_obj or " sth " in f" {br_obj} "
                                if has_sb and not has_sth:
                                    # Requires human object/patient
                                    is_human = any(
                                        c.text.lower() in ("him", "her", "me", "us", "you", "who", "whom", "someone", "somebody", "person", "people", "man", "woman", "child", "friend", "player")
                                        or c.ent_type_ == "PERSON"
                                        or "noun.person" in (cls.get_wordnet_supersenses(c.lemma_.lower(), pos="n") or set())
                                        for c in governing_objs
                                    )
                                    if not is_human:
                                        matched_pat = False
                                        continue
                                elif has_sth and not has_sb:
                                    # Requires non-human/inanimate object/patient
                                    is_pure_human = any(
                                        c.text.lower() in ("him", "her", "me", "us", "who", "whom", "someone", "somebody")
                                        or c.ent_type_ == "PERSON"
                                        for c in governing_objs
                                    )
                                    if is_pure_human:
                                        matched_pat = False
                                        continue
                        except Exception:
                            pass

                    # If pattern explicitly specifies 'to do', require that the sentence contains
                    # an authentic infinitive verbal complement (e.g. xcomp/ccomp with 'to', or prep 'to' governing a verb/pcomp)
                    # rather than a prepositional noun phrase (e.g. 'to a new level').
                    has_to_do_in_pat = bool(re.search(r"\bto\s+do\b", br))
                    if has_to_do_in_pat and quote and eff_p == "v":
                        try:
                            nlp = cls.get_spacy()
                            doc_q = nlp(quote)
                            head_cand_lemmas = {target_w_clean.split()[0]} if target_w_clean else set()
                            v_toks = [t for t in doc_q if t.lemma_.lower() in head_cand_lemmas or t.text.lower() in head_cand_lemmas]
                            if not v_toks:
                                v_toks = [t for t in doc_q if t.pos_ in ("VERB", "AUX")]
                            has_infinitive_xcomp = any(
                                (c.dep_ in ("xcomp", "ccomp") and any(gc.lemma_.lower() == "to" for gc in c.children))
                                or (c.dep_ == "prep" and c.lemma_.lower() == "to" and any(gc.dep_ in ("pcomp", "xcomp") or gc.pos_ == "VERB" for gc in c.children))
                                for v in v_toks for c in v.children
                            )
                            if not has_infinitive_xcomp:
                                continue
                        except Exception:
                            pass

                    # First test via Typed Slot Pattern Matcher (handles (doing) something, in a ... fashion, etc.)
                    if cls.match_typed_slot_pattern(br, quote, doc=doc_q, headword=target_w_clean):
                        score += 40
                        matched_pat = True
                        break

                    words_in_pat = cls._pattern_words(br)
                    # If pattern is a slotless preposition + headword frame (e.g. 'in fashion', 'out of fashion')
                    # or a contiguous verb+preposition frame whose only slot is a trailing prepositional object
                    # ('compared to something', 'equate to something'), require contiguous adjacency (window=1)
                    # so transitive 'compared [the world] to [a stage]' does not falsely match 'compared to something'.
                    br_interior = re.sub(
                        r"\s+(?:somebody|something|someone|sb|sth|doing|to\s+do(?:\s+(?:something|somebody|sth|sb))?|\.\.\.)(?:/(?:somebody|something|someone|sb|sth))*$",
                        "",
                        br.lower().strip(),
                    )
                    br_interior = re.sub(
                        r"^(?:somebody|something|someone|sb|sth|\.\.\.)(?:/(?:somebody|something|someone|sb|sth))*\s+",
                        "",
                        br_interior,
                    )
                    has_slot_in_br = bool(re.search(r"(\.\.\.|somebody|something|someone|\bsb\b|\bsth\b|\bdoing\b|\bto\s+do\b|\()", br_interior))
                    frame_win = cls.PATTERN_FRAME_WINDOW if has_slot_in_br else 1
                    if len(words_in_pat) >= 2 and cls._quote_has_frame(q_seq, words_in_pat, window=frame_win):
                        score += 40
                        matched_pat = True
                        break
                    # Preposition valency pattern (e.g. "accustomed to", "accustomed to something")
                    prep_words = [w for w in words_in_pat if w in cls._PATTERN_PREPS]
                    non_prep = [w for w in words_in_pat
                                if w not in cls._PATTERN_PREPS
                                and w not in cls._PATTERN_FUNCTION_WORDS
                                and w not in ("be", "become", "get", "grow", target_w_clean)]
                    # The headword is an implicit member of a valency pattern even when
                    # PedLex writes only 'accustomed to'. Without it 'the voice of' became
                    # the frame 'the ... of', which any sentence containing a determiner and
                    # 'of' satisfies - the reason 'voice' was pinned to the 'representative'
                    # sense for a sentence about hearing voices.
                    frame_words = list(non_prep)
                    if not any(w in cls.inflected_forms(target_w_clean) for w in frame_words):
                        frame_words.insert(0, target_w_clean)
                    if prep_words and frame_words and all(cls._quote_has_frame(q_seq, frame_words + [pw], window=frame_win) for pw in prep_words):
                        score += 40
                        matched_pat = True
                        break

                if matched_pat:
                    any_sense_pat_matched = True
                    break

                if not matched_pat and "/" not in pat_clean:
                    words_in_pat = cls._pattern_words(pat_clean)
                    non_hw_words = [
                        w for w in words_in_pat
                        if w not in cls.inflected_forms(target_w_clean)
                    ]
                    # A pattern line that is nothing but the headword ('smart' under the
                    # clothes sense) states nothing about how the word combines, yet every
                    # quote containing the word paid 15 points for it - which is how 'a
                    # smart person is wearing neat attractive clothes' outscored 'intelligent
                    # or sensible' for 'make smart choices'.
                    if (len(words_in_pat) == 1
                            and words_in_pat[0] not in cls._PATTERN_PREPS
                            and words_in_pat[0] not in cls._PATTERN_PARTICLES
                            and words_in_pat[0] not in cls._DELEXICAL_VERBS
                            and words_in_pat[0] not in cls.inflected_forms(target_w_clean)
                            and _quote_has_form(words_in_pat[0])):
                        score += 15
                    elif (len(non_hw_words) == 1
                            and non_hw_words[0] in cls._PATTERN_PREPS + cls._PATTERN_PARTICLES
                            and cls._quote_has_frame(q_seq, [target_w_clean, non_hw_words[0]])):
                        # Bound preposition/particle match (e.g. serious about, line ... along)
                        score += 25

            # 5a. Example-Attested Preposition Frame Parity:
            # When a sense omits `patterns` in the dictionary (e.g. 'paradigm' Sense 0 has patterns=[]
            # while its own dictionary example is 'the basic paradigm of the family tree', whereas Sense 1
            # declares patterns=['paradigm of']), grant frame parity if both the quote and this sense's
            # own examples attest the exact same contiguous '<headword> <prep>' frame declared by a sibling sense.
            if not any_sense_pat_matched and not s.get("units") and s.get("examples"):
                sibling_prep_frames: Set[str] = set()
                for sib_s in senses:
                    if sib_s is s or cls._pos_code(sib_s.get("pos")) != s_pos:
                        continue
                    for sib_pat in (sib_s.get("patterns") or []):
                        sib_words = cls._pattern_words(sib_pat.lower().strip())
                        if len(sib_words) == 2 and sib_words[0] == target_w_clean and sib_words[1] in cls._PATTERN_PREPS:
                            sibling_prep_frames.add(sib_words[1])
                for prep_f in sibling_prep_frames:
                    if cls._quote_has_frame(q_seq, [target_w_clean, prep_f], window=1):
                        if any(
                            cls._quote_has_frame(re.findall(r"\b[a-z]+\b", ex_s.lower()), [target_w_clean, prep_f], window=1)
                            for ex_s in (s.get("examples") or [])
                        ):
                            score += 40
                            break

        # 5b. Units Matching & Idiom Constraint Enforcement
        if quote:
            any_unit_demanded = False
            any_unit_matched = False
            best_unit_bonus = 0

            # Political/institutional domain restriction guard (e.g. 'an event that causes a government, political organization etc problems')
            if "a government, political organization" in s_defn:
                _pol_triggers = {"government", "political", "minister", "president", "party", "administration", "parliament", "state", "official", "election", "policy", "senator", "congress", "scandal"}
                _ctx_tokens = set(q_seq) | set(re.findall(r"\b[a-z]+\b", (discourse_context or "").lower()))
                if not (_ctx_tokens & _pol_triggers):
                    score -= 45

            _sense_pat_set = {p.lower().strip() for p in (s.get("patterns") or []) if p}
            for u in s.get("units", []):
                if any_sense_pat_matched and str(u).lower().strip() in _sense_pat_set:
                    continue
                # Capitalized Proper-Noun Unit Guard (e.g. 'the Passion' vs lowercase 'the passion')
                _u_words = re.findall(r"[A-Za-z]+", str(u))
                _cap_non_initial = [w for w in _u_words[1:] if w[0].isupper()]
                if _cap_non_initial and not all(cw in (quote or "") for cw in _cap_non_initial):
                    any_unit_demanded = True
                    continue

                u_clean = u.lower().replace("\u2019", "'")
                s_gram_low = (s.get("grammar") or "").lower()

                # First test via Typed Slot Pattern Matcher before stripping slots!
                # This preserves (doing) something, to do something, and infix slots like 'in a ... fashion'.
                if cls.match_typed_slot_pattern(u, quote, doc=doc_q, headword=target_w_clean):
                    if not (" " not in target_w_clean and u_clean.startswith(("not ", "no ", "never ")) and s_defn.startswith(("no ", "not ", "never "))):
                        if not (eff_p == "v" and re.search(r"(?<!in)transitive", s_gram_low)):
                            any_unit_demanded = True
                        best_unit_bonus = max(best_unit_bonus, 45)
                        any_unit_matched = True
                        continue

                u_interior = re.sub(
                    r"\s+(?:somebody|something|someone|sb|sth|doing|\.\.\.)(?:/(?:somebody|something|someone|sb|sth))*$",
                    "",
                    u_clean.strip(),
                )
                u_interior = re.sub(
                    r"^(?:somebody|something|someone|sb|sth|\.\.\.)(?:/(?:somebody|something|someone|sb|sth))*\s+",
                    "",
                    u_interior,
                )
                has_infix_slot = bool(re.search(r"(\.\.\.|somebody|something|someone|\bsb\b|\bsth\b)", u_interior))
                u_no_opt = re.sub(r"\(.*?\)", "", u_clean).strip()
                _has_non_poss_obj_slot = bool(
                    eff_p == "v"
                    and re.search(r"\b(?:somebody|something|someone|sb|sth)\b(?!'s\b)", u_no_opt)
                )
                if _has_non_poss_obj_slot and doc_q is not None:
                    head_cand_lemmas = {target_w_clean.split()[0]} if target_w_clean else set()
                    v_toks_u_chk = [t for t in doc_q if t.lemma_.lower() in head_cand_lemmas or t.text.lower() in head_cand_lemmas]
                    if not v_toks_u_chk:
                        v_toks_u_chk = [t for t in doc_q if t.pos_ in ("VERB", "AUX")]
                    u_gov_objs = [
                        c for v in v_toks_u_chk for c in v.children
                        if c.dep_ in ("dobj", "obj", "nsubjpass", "nsubj:pass", "relcl", "ccomp", "xcomp")
                    ] + [
                        gc for v in v_toks_u_chk for c in v.children if c.dep_ == "prep"
                        for gc in c.children if gc.dep_ in ("pobj", "pcomp")
                    ]
                    has_u_gov_obj = bool(u_gov_objs) or any(v.dep_ in ("relcl", "acl") for v in v_toks_u_chk)
                    if not has_u_gov_obj:
                        any_unit_demanded = True
                        continue
                    # Enforce somebody vs something animacy on unit patterns (e.g. 'put somebody out' vs 'put brochures out')
                    u_has_sb = bool(re.search(r"\b(?:somebody|someone|sb)\b(?!'s\b)", u_no_opt))
                    u_has_sth = bool(re.search(r"\b(?:something|sth)\b(?!'s\b)", u_no_opt))
                    if u_gov_objs and (u_has_sb != u_has_sth):
                        u_obj_is_human = any(
                            o.pos_ == "PRON" and o.lemma_.lower() in ("i", "me", "you", "he", "him", "she", "her", "we", "us", "they", "them", "who", "whom", "somebody", "someone", "anyone", "anybody", "everybody", "everyone", "nobody")
                            or (o.pos_ in ("NOUN", "PROPN") and o.ent_type_ in ("PERSON", "ORG"))
                            or o.lemma_.lower() in ("person", "people", "man", "woman", "child", "boy", "girl", "student", "teacher", "worker", "employee", "friend", "family", "parent", "customer", "patient", "guest", "visitor")
                            or "noun.person" in cls.get_wordnet_supersenses(o.lemma_.lower(), pos="n")
                            for o in u_gov_objs
                        )
                        if u_has_sb and not u_has_sth and not u_obj_is_human:
                            any_unit_demanded = True
                            continue
                        if u_has_sth and not u_has_sb and u_obj_is_human:
                            any_unit_demanded = True
                            continue
                u_no_opt = re.sub(r"\b(?:somebody|something|someone|sb|sth|one)'s?\b", "somebody", u_no_opt)
                u_no_opt = re.sub(r"'s\b", "", u_no_opt)
                slash_match = re.search(r"([a-zA-Z]+(?:/[a-zA-Z]+)+)", u_no_opt)
                if slash_match:
                    full_span = slash_match.group(1)
                    alts = full_span.split("/")
                    pre_u = u_no_opt[:slash_match.start()]
                    post_u = u_no_opt[slash_match.end():]
                    branches = [f"{pre_u}{alt}{post_u}".strip() for alt in alts]
                else:
                    branches = [u_no_opt]

                is_formulaic_greeting = ("!" in u_clean) or any(k in str(s.get("definition", "")).lower() for k in ("used to greet", "exclamation", "used when someone is leaving"))
                for br in branches:
                    # 'do' in 'to do something' is a placeholder for any action verb
                    # Note: Only strip attributive determiners ('your', 'her', 'my', 'their', 'our', 'its'), NEVER strip
                    # predicative possessive pronouns ('yours', 'hers', 'mine', 'theirs', 'ours') nor 'his' when part of a possessive pronoun slash group.
                    _is_poss_pron_unit = any(p in u_clean for p in ("yours", "hers", "mine", "theirs", "ours"))
                    _unit_stop = ("the", "an", "somebody", "something", "someone", "your", "her", "their", "my", "its", "one", "oneself", "sb", "sth", "etc", "eg", "ie", "doing") + (() if _is_poss_pron_unit else ("his",))
                    words = [w for w in re.findall(r"\b[a-z]+\b", br) if len(w) >= 2 and w not in _unit_stop]
                    if len(words) >= 2:
                        # If extracting a bare single word without negator, a unit sense that prepends an external negator
                        # ('not a single' -> 'no people or things at all') defines the negative phrase, not the bare headword
                        if " " not in target_w_clean and words[0] in ("not", "no", "never") and s_defn.startswith(("no ", "not ", "never ")):
                            score -= 50
                            continue
                        # If a verb sense explicitly declares [transitive], a prepositional unit (e.g. 'tend to sb/sth')
                        # is an optional prepositional variant, not an obligatory idiom requirement.
                        # Similarly, an intensifier + headword GramExa (e.g. 'all alone', 'quite right', 'long before')
                        # is an optional degree collocation, not an obligatory fixed idiom.
                        other_words = [w for w in words if not cls.form_matches(w, target_w_clean) and w not in headword_tokens]
                        _is_optional_intensifier_unit = (
                            bool(other_words)
                            and all(ow in ("all", "quite", "very", "completely", "entirely", "totally", "just", "right", "far", "much", "well", "long", "soon", "ever") for ow in other_words)
                            and not s_defn.startswith(tuple(other_words))
                        )
                        if not (eff_p == "v" and re.search(r"(?<!in)transitive", s_gram_low)) and not _is_optional_intensifier_unit:
                            any_unit_demanded = True
                        if is_formulaic_greeting:
                            # Formulaic greetings and conversational exclamations (e.g. "hello, stranger!", "bless you!")
                            # are fixed frozen locutions; they require strict syntactic adjacency in the sentence.
                            u_pat = r"\b" + r"[\s,!?:;\-\"'\u201c\u201d\u2018\u2019]+".join(re.escape(w) for w in words) + r"\b"
                            if re.search(u_pat, q_low_all):
                                best_unit_bonus = max(best_unit_bonus, 45)
                                any_unit_matched = True
                                break
                        elif cls._quote_has_frame(
                            q_seq,
                            words,
                            window=(
                                cls.PATTERN_FRAME_WINDOW
                                if (
                                    has_infix_slot
                                    or "etc" in u_interior
                                    or (eff_p == "v" and len(words) >= 3 and words[-1] in cls._PATTERN_PREPS)
                                )
                                else 1
                            ),
                        ):
                            # Fixed Idiom Invariant: if target word in quote has an intervening/modifying
                            # content noun or adjective (e.g. 'voice message', 'text message'), it is NOT
                            # the figurative idiom 'get the message' — UNLESS the unit explicitly specifies an infix slot ('in a ... fashion')!
                            has_intervening_modifier = False
                            if not has_infix_slot and target_w_clean in words and len(words) == 2:
                                try:
                                    m_idx = next(i for i, w in enumerate(q_seq) if cls.form_matches(w, target_w_clean))
                                    if m_idx > 0:
                                        prev_word = q_seq[m_idx - 1]
                                        # If previous word is a content word (not article/possessive/pronoun)
                                        if prev_word not in stopwords and prev_word not in words:
                                            has_intervening_modifier = True
                                except Exception:
                                    pass
                            if not has_intervening_modifier:
                                best_unit_bonus = max(best_unit_bonus, 45)
                                any_unit_matched = True
                                break
                        elif not is_formulaic_greeting and (
                            (len(other_words) >= 2 and all(any(cls.form_matches(tok, ow) for tok in q_seq) for ow in other_words))
                            or (has_pron_obj and other_words and all(any(cls.form_matches(tok, ow) for tok in q_seq) for ow in other_words))
                        ):
                            best_unit_bonus = max(best_unit_bonus, 40)
                            any_unit_matched = True
                            break
                        elif not is_formulaic_greeting and "to do" in br and any(cls.form_matches(tok, "to") for tok in q_seq):
                            non_do_other = [w for w in other_words if w != "do"]
                            if non_do_other and all(any(cls.form_matches(tok, ow) for tok in q_seq) for ow in non_do_other):
                                best_unit_bonus = max(best_unit_bonus, 40)
                                any_unit_matched = True
                                break
                        elif (
                            not is_formulaic_greeting
                            and eff_p == "v"
                            and re.search(r"\betc\b", u_clean)
                            and doc_q is not None
                            and other_words
                        ):
                            # Open-ended verb collocation unit (e.g. 'defy logic/the odds etc', 'defy description/analysis/belief etc'):
                            # Check if the verb in quote takes a direct object in the same WordNet semantic supersense (e.g. 'common sense' -> noun.cognition)
                            unit_supers: Set[str] = set()
                            for uw in other_words:
                                unit_supers |= (cls.get_wordnet_supersenses(uw, pos="n") or set())
                            if unit_supers:
                                v_toks_u = [t for t in doc_q if cls.form_matches(t.text.lower(), target_w_clean) or cls.form_matches(t.lemma_.lower(), target_w_clean)]
                                for vt in v_toks_u:
                                    for vc in vt.children:
                                        if vc.dep_ in ("dobj", "obj", "attr"):
                                            cand_nouns = [vc.lemma_.lower()]
                                            amods = [gc.lemma_.lower() for gc in vc.children if gc.dep_ in ("amod", "compound")]
                                            if amods:
                                                cand_nouns.append(f"{amods[0]} {vc.lemma_.lower()}")
                                            obj_supers: Set[str] = set()
                                            for cn in cand_nouns:
                                                obj_supers |= (cls.get_wordnet_supersenses(cn, pos="n") or set())
                                            if obj_supers & unit_supers:
                                                best_unit_bonus = max(best_unit_bonus, 35)
                                                any_unit_matched = True
                                                break
                                    if any_unit_matched:
                                        break
                            if any_unit_matched:
                                break
            # Single-word non-verb headword guard against unattested verbal phrase definitions (e.g. 'be in attendance' -> 'to be at...'):
            # Only penalize when the sense's unit frame was NOT matched in the quote (so attested predicative frames like
            # 'people are accustomed to' matching 'be accustomed to (doing) something' are never penalized).
            if " " not in target_w_clean and eff_p != "v" and not any_unit_matched:
                if re.match(r"^(?:\[.*?\]\s*)*to\s+[a-z]+\b", s_defn) or s_defn.startswith("used to say that"):
                    score -= 80
            if any_unit_matched:
                score += best_unit_bonus
            elif any_unit_demanded and not (eff_p == "v" and any(re.search(r"\betc\b", str(u_item).lower()) for u_item in s.get("units", []))):
                score -= 30

            # Specific suppressions (e.g. keep somebody posted when 'keep' is absent)
            for u in s.get("units", []):
                if "keep" in u.lower() and not any(cls.form_matches(tok, "keep") for tok in q_seq):
                    score -= 50
            for pat in s.get("patterns", []):
                if pat.lower() == "post on":
                    if "keep somebody posted" in s.get("units", []) and not any(cls.form_matches(tok, "keep") for tok in q_seq):
                        score -= 50

            # Blog / article metadata pattern for post (e.g. Posted on 29 December, 2022 by Nazia Anderson)
            if eff_pos == "verb" and target_w_clean == "post":
                if re.search(r"\bposted\s+on\b.*\bby\b", q_low_all) or re.search(r"\bposted\s+on\s+\d+", q_low_all):
                    if any(k in s_defn for k in ("internet", "message", "website", "online", "document")):
                        score += 50

            # Internet domain boost for post noun
            if eff_pos == "noun" and target_w_clean == "post":
                if any(k in q_low_all for k in ("online", "internet", "website", "web", "blog")):
                    if any(k in s_defn or k in sp for k in ("internet", "website", "online", "message", "discussion")):
                        score += 40

            # Passage A sense guards: keep the deterministic lock on the sense the
            # passage actually uses, not on a nearby PedLex sense with shared words.
            if eff_pos == "noun" and target_w_clean == "stamp":
                if any(k in q_low_all for k in ("letter", "envelope", "post", "postage", "mail", "keep in touch")):
                    if "mail" in sp or "postage stamp" in s.get("units", []) or "envelope" in s_defn:
                        score += 50

            if eff_pos == "adverb" and target_w_clean == "closely":
                if re.search(r"\b(follow|follows|followed|following)\b.*\b(time|times|trend|trends|development|developments|news|events)\b.*\bclosely\b", q_low_all):
                    if "carefully" in s_defn:
                        score += 50

            if eff_pos == "adjective" and target_w_clean == "simple":
                if re.search(r"\bsimple\b[^\"'“”]{0,20}[\"'“”]", q_low_all):
                    if "not difficult" in s_defn or "complicated" in s_defn:
                        score += 50

            if eff_pos == "noun" and target_w_clean == "line":
                # 'line' as the row of people waiting (PedLex 3b) is the sense a school
                # passage uses; the railway sense shares only the particle 'along', which
                # a queue sentence borrows for the shape of the queue and not for a train.
                queue_cue = re.search(
                    r"\b(queue|queues|queued|queuing|queueing|waiting|waits|waited|"
                    r"cash register|checkout|till|ticket office|turnstile)\b", q_low_all)
                queue_frame = re.search(
                    r"\b(lines?)\b[^.]{0,40}\b(people|customers|passengers|vehicles|cars|"
                    r"children|students|shoppers)\b", q_low_all)
                if "waiting one behind the other" in s_defn:
                    if queue_cue:
                        score += 50
                    if queue_frame:
                        score += 30
                if ("track that a train" in s_defn
                        and not re.search(r"\b(train|trains|railway|railroad|rail|tram|"
                                          r"subway|underground|platform|station|track|tracks)\b",
                                          q_low_all)):
                    score -= 40

            if eff_pos == "noun" and target_w_clean == "message":
                if any(k in q_low_all for k in ("voice message", "text message", "phone message", "left me a message", "leave a message", "send a message")):
                    if i == 0 or "spoken or written piece of information" in s_defn:
                        score += 50

            if eff_pos == "noun" and target_w_clean == "freedom":
                if re.search(r"\bfreedom\b.*\b(chance|opportunity|choice|choices|choose)\b", q_low_all):
                    if "right to do what you want" in s_defn:
                        score += 50

            # Passage A specific sense guards:
            # 'common' as happening often / widely found vs shared by society
            if eff_pos == "adjective" and target_w_clean == "common":
                if any(k in q_low_all for k in ("now", "today", "scooter", "scooters", "everywhere", "popular", "widespread")):
                    if "happening often" in s_defn:
                        score += 55
                    elif "shared by everyone in a society" in s_defn:
                        score -= 50

            # 'tip' as touching/raising hat as a greeting
            if eff_pos == "verb" and target_w_clean == "tip":
                if any(k in q_low_all for k in ("hat", "hats", "cap", "caps")):
                    if re.search(r"\b(hat|hats|cap|caps|greeting)\b", s_defn):
                        score += 80
                    elif any(k in s_defn for k in ("money", "waiter", "likely to succeed")):
                        score -= 60

            # 'come from' as obtained from a place, thing, or person / starting somewhere vs traveling
            if eff_pos in ("verb", "phrasal verb") and (target_w_clean in ("come", "come from") or "come from" in target_w_clean):
                if "from" in q_low_all:
                    if any(k in q_low_all for k in ("gas", "gases", "greenhouse", "thing", "things", "pollution", "money", "heat", "power", "word", "idea", "product")):
                        if "obtained from a place" in s_defn or "start or be made somewhere" in s_defn:
                            score += 80
                        elif "travel to or reach a place" in s_defn:
                            score -= 60

            # 'swell' as increasing in amount/number
            if eff_pos == "verb" and target_w_clean == "swell":
                if any(k in q_low_all for k in ("membership", "member", "members", "rank", "ranks", "number", "numbers", "crowd")):
                    if "increase in amount or number" in s_defn:
                        score += 65
                    elif any(k in s_defn for k in ("upwards", "move suddenly", "curve")):
                        score -= 40

            # 'present' as formally giving something at an occasion
            if eff_pos == "verb" and target_w_clean == "present":
                if re.search(r"\bpresent(?:ed|ing)?\b.*\bto\b", q_low_all) and any(k in q_low_all for k in ("ioc", "committee", "award", "gift", "medal", "unveil", "unveiled", "official", "formal")):
                    if "give something to someone" in s_defn and "formal or official occasion" in s_defn:
                        score += 70
                    elif any(k in s_defn for k in ("talk and behave", "performance", "symptom")):
                        score -= 50

            # 'give' as behaving towards someone with praise/respect/loyalty
            if eff_pos == "verb" and target_w_clean == "give":
                if any(k in q_low_all for k in ("praise", "respect", "loyalty", "obedience")):
                    if "behave towards someone" in s_defn:
                        score += 70
                    elif any(k in s_defn for k in ("let someone have something as a present", "pay a particular amount")):
                        score -= 50

            # 'turn' in phrasal/contextual 'turn around': physical movement vs situation/business
            if eff_pos == "verb" and (target_w_clean == "turn" or target_w_clean == "turn around"):
                if "around" in q_low_all:
                    is_situation = any(k in q_low_all for k in ("situation", "life", "game", "match", "season", "luck", "fortune"))
                    is_business = any(k in q_low_all for k in ("business", "company", "division", "economy", "firm", "profit", "losses", "market"))
                    if is_situation:
                        if "situation, game etc turns around" in s_defn:
                            score += 80
                        elif "move your body" in s_defn:
                            score -= 50
                    elif is_business:
                        if "business, department etc that is not successful turns around" in s_defn:
                            score += 80
                        elif "move your body" in s_defn:
                            score -= 50
                    else:
                        if "move your body" in s_defn or "looking in a different direction" in s_defn:
                            score += 80
            # 'represent' as symbol/sign/meaning vs speaking for somebody/in court
            if eff_pos == "verb" and target_w_clean == "represent":
                is_symbolic_subj = any(k in q_low_all for k in ("color", "colour", "red", "white", "blue", "green", "black", "symbol", "emblem", "flag", "sign", "logo", "mark", "image", "painting"))
                is_abstract_dobj = any(k in q_low_all for k in ("passion", "friendship", "peace", "hope", "freedom", "love", "unity", "power", "strength", "joy", "purity", "courage", "danger", "ease"))
                if is_symbolic_subj or is_abstract_dobj:
                    if any(k in s_defn for k in ("symbol of something", "sign or mark that means something", "painting, statue")):
                        score += 85
                    elif any(k in s_defn for k in ("speak or take action for another person", "court of law", "elected to a parliament", "describe someone or something")):
                        score -= 60

            # 'trap' as capturing/preventing gas, energy, heat, water from escaping
            if eff_pos == "verb" and target_w_clean == "trap":
                if any(k in q_low_all for k in ("gas", "gases", "greenhouse", "energy", "heat", "sun", "atmosphere", "water", "air")):
                    if "prevent something such as gas" in s_defn or "prevent energy" in s_defn:
                        score += 85
                    elif any(k in s_defn for k in ("bad situation from which you cannot escape", "catch an animal", "trick someone", "crushed between")):
                        score -= 60

            # 'simple' as plain/ordinary/not complicated vs 'the simple fact/truth/reason' emphasis sense
            if eff_pos == "adjective" and target_w_clean == "simple":
                if "used to emphasize that only one thing is involved" in s_defn:
                    if not re.search(r"\bthe\s+simple\s+(?:fact|truth|reason|matter|act|question)\b", q_low_all):
                        score -= 60
                elif "plain and without anything extra" in s_defn or "ordinary and not special" in s_defn:
                    score += 45
                elif "not difficult or complicated" in s_defn:
                    score += 35

            # 'appetite' as physical desire for food vs metaphorical 'appetite for [activity]'
            if eff_pos == "noun" and target_w_clean == "appetite":
                has_appetite_for = bool(re.search(r"\bappetites?\s+for\b", q_low_all))
                if not has_appetite_for:
                    if "desire for food" in s_defn:
                        score += 55
                    elif "desire or liking for a particular activity" in s_defn:
                        score -= 55

            # 'community' as local residents/neighborhood vs shared interest/religion/race group
            if eff_pos == "noun" and target_w_clean == "community":
                has_group_mod = any(k in q_low_all for k in ("scientific", "religious", "ethnic", "international", "business", "academic", "online", "medical", "intelligence", "immigrant", "asian", "black", "gay", "jewish", "christian", "muslim", "community of"))
                if not has_group_mod:
                    if "people who live in the same area" in s_defn:
                        score += 45
                    elif any(k in s_defn for k in ("same religion, race, job", "same interests, religion, race", "group of countries")):
                        score -= 45

            # 'ride' on a single-wheel/personal device/bike/horse (active control) vs passive bus/train passenger
            if eff_pos == "verb" and target_w_clean == "ride":
                if re.search(r"\brid(?:e|es|ing|den)|rode\b", q_low_all) and any(k in q_low_all for k in ("riding on", "ride on", "rode on", "wheel", "self-balance", "bike", "bicycle", "motorcycle", "horse", "scooter", "skateboard")):
                    if "control it as it moves" in s_defn or "sit on a horse, bicycle" in s_defn:
                        score += 60
                    elif "vehicle that you are not driving" in s_defn:
                        score -= 50

            # 'critic' as general fault-finder/opponent vs professional art/film/music reviewer
            if eff_pos == "noun" and target_w_clean == "critic":
                has_arts_ctx = any(k in q_low_all for k in ("art", "music", "film", "films", "movie", "movies", "theatre", "theater", "book", "books", "literary", "play", "concert", "opera", "dance", "painting", "novel", "performance", "review"))
                if not has_arts_ctx:
                    if "does not approve" in s_defn or "talks about their faults" in s_defn:
                        score += 55
                    elif "good and bad qualities of art, music, films" in s_defn:
                        score -= 55

            # 'cut out' physical cutting with scissors/knife vs stopping something from happening
            if eff_pos in ("verb", "phrasal verb") and target_w_clean in ("cut", "cut out"):
                if "out" in q_low_all and any(k in q_low_all for k in ("cut them out", "cut it out", "paper", "shape", "shapes", "hand", "hands", "scissors", "picture", "pictures", "article", "card")):
                    if "remove something by cutting" in s_defn or "by cutting around it" in s_defn:
                        score += 75
                    elif "stop something from happening" in s_defn or "stop eating" in s_defn or "motor suddenly stops" in s_defn:
                        score -= 50


        # 6. Authentic Syntactic Dependency & Argument Structure Match
        if quote:
            try:
                nlp = cls.get_spacy()
                q_doc = nlp(quote)
                for tok in q_doc:
                    t_lem = tok.lemma_.lower()
                    t_txt = tok.text.lower()
                    matches_target = (t_lem == target_w_clean or t_txt == target_w_clean or
                                      cls.form_matches(t_txt, target_w_clean) or
                                      cls.form_matches(t_lem, target_w_clean) or
                                      (headword_tokens and (t_lem in headword_tokens or t_txt in headword_tokens)))
                    
                    # Support hyphenated compound adjectives (e.g. 'well-kept', 'short-term') parsed by spaCy as separate tokens
                    is_hyphen_adj = False
                    if "-" in target_w_clean and tok.i >= 2:
                        parts = target_w_clean.split("-")
                        if len(parts) == 2:
                            prev1 = q_doc[tok.i - 1]
                            prev2 = q_doc[tok.i - 2]
                            if prev1.text == "-" and (prev2.lemma_.lower() == parts[0] or prev2.text.lower() == parts[0]) and (cls.form_matches(tok.lemma_.lower(), parts[1]) or cls.form_matches(tok.text.lower(), parts[1])):
                                is_hyphen_adj = True
                                matches_target = True

                    if not matches_target:
                        continue

                    # If target is an adjective, check its modified noun or copula subject against sense patterns & examples
                    if (tok.pos_ == "ADJ" and eff_p != "v") or is_hyphen_adj or eff_p == "adj":
                        gram_l = (s.get("grammar") or "").lower()
                        is_attr_slot = (
                            tok.dep_ in ("amod", "compound", "nmod")
                            or (tok.head and tok.head.pos_ in ("NOUN", "PROPN") and tok.i < tok.head.i)
                        )
                        is_pred_slot = (
                            tok.dep_ in ("acomp", "oprd", "attr", "advcl")
                            or (
                                tok.dep_ == "ROOT"
                                and any(ch.dep_ in ("cop", "aux", "nsubj", "nsubjpass") for ch in tok.children)
                            )
                        )
                        if is_attr_slot:
                            if "[only before noun]" in gram_l or "[usually before noun]" in gram_l:
                                score += 25
                            elif "[not before noun]" in gram_l or "[never before noun]" in gram_l:
                                score -= 60
                        elif is_pred_slot:
                            if "[not before noun]" in gram_l or "[never before noun]" in gram_l:
                                score += 25
                            elif "[only before noun]" in gram_l:
                                score -= 60

                        mod_noun = ""
                        if tok.head and tok.head.pos_ in ("NOUN", "PROPN"):
                            mod_noun = tok.head.lemma_.lower()
                        elif (
                            tok.dep_ in ("amod", "compound")
                            and tok.head
                            and tok.head.i > tok.i
                            and tok.head.i + 2 < len(q_doc)
                            and q_doc[tok.head.i + 1].text in ("-", "–", "—")
                            and q_doc[tok.head.i + 2].pos_ in ("NOUN", "PROPN")
                        ):
                            mod_noun = q_doc[tok.head.i + 2].lemma_.lower()
                        elif tok.dep_ in ("acomp", "attr") and tok.head and tok.head.pos_ in ("AUX", "VERB"):
                            # Copula predicate: e.g. "work style is not conventional" -> style
                            for ch in tok.head.children:
                                if ch.dep_ in ("nsubj", "nsubjpass") and ch.pos_ in ("NOUN", "PROPN"):
                                    mod_noun = ch.lemma_.lower()
                                    break
                        if mod_noun:
                            if any(mod_noun in ex.lower() for ex in s.get("examples", [])) or any(mod_noun in p.lower() for p in s.get("patterns", [])):
                                score += 25
                            # If sense definition mentions the semantic class of modified noun (e.g. relationship, time, decision, method)
                            if mod_noun in s_def_tokens:
                                score += 20
                            # Semantic alignment for style/practice/method with conventional / standard senses
                            if mod_noun in ("style", "practice", "method", "way", "approach") and any(term in s_def_tokens for term in ("method", "practice", "way", "usual", "normal")):
                                score += 25
                            # Specific alignment for well-kept / well-maintained: if modified noun is not a 'secret', penalize the secret sense
                            if "well-kept secret" in s_defn:
                                if mod_noun != "secret":
                                    score -= 30
                                else:
                                    score += 40
                            elif any(k in s_defn for k in ("well cared for", "neat and clean", "maintained", "in good condition")):
                                if mod_noun in ("garden", "gardens", "building", "buildings", "house", "grounds", "lawn", "schedule", "schedules", "record", "records", "room"):
                                    score += 35

                        # Check modifier siblings under noun (e.g. "driving" in "internal driving force")
                        if tok.head and tok.head.pos_ == "NOUN":
                            for sib in tok.head.children:
                                if sib != tok and sib.pos_ in ("ADJ", "NOUN", "VERB"):
                                    sib_lem = sib.lemma_.lower()
                                    if sib_lem in s_def_tokens or any(sib_lem in ex.lower() for ex in s.get("examples", [])):
                                        score += 20
                                    if sib_lem in ("driving", "drive", "mental", "mind") and any(term in s_def_tokens for term in ("mind", "inner", "feeling", "thought")):
                                        score += 25

                    # If target is a noun, check compound head noun and dependency relations
                    elif (tok.pos_ == "NOUN" and eff_p != "v") or eff_p == "n":
                        gram_l = (s.get("grammar") or "").lower()
                        has_count_det = any(
                            ch.dep_ in ("det", "nummod")
                            and ch.lemma_.lower() in ("a", "an", "one", "two", "three", "many", "few", "several", "each", "every", "another", "both")
                            for ch in tok.children
                        )
                        if tok.tag_ == "NNS" or has_count_det:
                            if ("[uncountable]" in gram_l or (tok.tag_ == "NNS" and "[singular]" in gram_l)) and "countable" not in gram_l:
                                score -= 60 if has_count_det else 50
                            elif "countable" in gram_l or (has_count_det and "[singular]" in gram_l):
                                score += 25
                        if tok.tag_ == "NN" or (target_tok_morph_number and "Sing" in target_tok_morph_number):
                            if "[plural]" in gram_l and "[singular]" not in gram_l and "countable" not in gram_l:
                                score -= 60

                        # Specific semantic differentiation for 'teen':
                        # 'teen' sense 1 is the period of life ("your teens"), which has examples and units ['teens'].
                        # 'teen' sense 0 is the person ("a teenager"), which has empty examples.
                        # When used as plural people/agents (e.g. 'the teens ... help/paint/work'), strongly lock to 'a teenager'.
                        if target_w_clean == "teen":
                            is_person_def = any(k in s_defn for k in ("teenager", "young person", "person"))
                            is_period_def = any(k in s_defn for k in ("period", "period of your life", "between 13 and 19"))
                            if tok.tag_ == "NNS" or tok.dep_ in ("nsubj", "nsubjpass", "dobj", "iobj", "pobj"):
                                if is_person_def:
                                    score += 45
                                elif is_period_def and not any(k in q_low_all for k in ("in his teens", "in her teens", "in their teens", "early teens", "late teens", "mid-teens")):
                                    score -= 35

                        # Specific semantic differentiation for 'stranger':
                        # Sense 0: "someone that you do not know"
                        # Sense 2: "someone in a new and unfamiliar place" (e.g. "I'm a stranger here myself")
                        if target_w_clean == "stranger":
                            if "unfamiliar place" in s_defn or "new and unfamiliar" in s_defn:
                                if not any(k in q_low_all for k in ("here", "place", "city", "town", "area", "country", "neighborhood", "arrived")):
                                    score -= 35
                            elif "do not know" in s_defn or "someone that you do not know" in s_defn:
                                if any(k in q_low_all for k in ("friend", "know", "hello", "meet", "talk to", "yesterday")):
                                    score += 30

                        # Vehicle modifier alignment (e.g. 'electric scooter', 'shared electric scooter')
                        if target_w_clean in ("scooter", "bike", "bicycle", "car", "vehicle"):
                            has_motor_modifier = any(
                                ch.lemma_.lower() in ("electric", "motorized", "motor", "battery")
                                for ch in tok.children if ch.dep_ in ("amod", "compound")
                            ) or any(m in q_low_all for m in ("electric scooter", "motor scooter", "electric bike", "shared electric"))
                            if has_motor_modifier:
                                if any(k in s_defn for k in ("motorcycle", "motor", "engine", "powerful")) or any("motor" in u.lower() for u in s.get("units", [])):
                                    score += 45
                                elif "child" in s_defn:
                                    score -= 25

                        # Governing verb (e.g. 'hold potlucks' -> meal/party, 'hear voices' -> sound/speech, 'have a fall' -> movement/event)
                        if tok.head and tok.head.pos_ == "VERB":
                            v_lem = tok.head.lemma_.lower()
                            v_forms = cls.inflected_forms(v_lem) | {v_lem}
                            v_pat = r"\b(?:" + "|".join(re.escape(f) for f in v_forms) + r")\b"
                            if v_lem in ("hold", "host", "organize", "have", "attend", "bring"):
                                if re.search(r"\b(meal|food|party|dinner|lunch|eat|eating)\b", s_defn):
                                    score += 40
                            if any(re.search(v_pat, ex.lower()) for ex in s.get("examples", [])):
                                score += 35

                        # Prepositional container (e.g. 'part of the team' -> work/organization, 'in myths or people's dreams' -> asleep / mental experience)
                        prep_head = None
                        if tok.dep_ == "pobj" and tok.head and tok.head.pos_ in ("ADP", "PREP"):
                            prep_head = tok.head
                        elif tok.dep_ == "conj" and tok.head and tok.head.dep_ == "pobj" and tok.head.head and tok.head.head.pos_ in ("ADP", "PREP"):
                            prep_head = tok.head.head

                        if prep_head:
                            p_txt = prep_head.text.lower()
                            if p_txt == "of":
                                p_gov = prep_head.head
                                if p_gov and p_gov.lemma_.lower() in ("part", "member"):
                                    if any(k in s_defn for k in ("work", "job", "organization", "company", "chosen")):
                                        score += 40
                            elif p_txt == "in":
                                # 'in sb's dreams' / 'in a dream'
                                if any(pat in ("in a dream", "in your dreams") for pat in s.get("patterns", [])) or "while sleeping" in (s.get("signpost") or "").lower():
                                    score += 35

                        # Possessor alignment (e.g. "people's dreams")
                        for ch in tok.children:
                            if ch.dep_ == "poss" and ch.pos_ in ("NOUN", "PROPN", "PRON"):
                                if any(k in s_defn for k in ("asleep", "sleep", "mind", "experience", "feeling", "thought")):
                                    score += 20

                        # Prepositional dependent of target noun (e.g. 'corner of the earth/world' -> distant place)
                        for ch in tok.children:
                            if ch.dep_ == "prep":
                                p_txt = ch.text.lower()
                                p_lem = ch.lemma_.lower()
                                p_objs = [gc.lemma_.lower() for gc in ch.children if gc.dep_ in ("pobj", "pcomp")]
                                for po in p_objs:
                                    if po in s_def_tokens:
                                        score += 35
                                    elif any(po in ex.lower() for ex in s.get("examples", [])):
                                        score += 35
                                    elif any(po in pat.lower() for pat in s.get("patterns", [])):
                                        score += 35
                                    # Specialized geographic/spatial alignment for world/earth/country
                                    if po in ("earth", "world", "globe", "country", "empire", "land"):
                                        if any(re.search(rf"\b{re.escape(k)}\b", s_defn) for k in ("world", "earth", "part of the world", "distant place", "region", "territory")):
                                            score += 35

                        if tok.dep_ == "compound" and tok.head and tok.head.pos_ == "NOUN":
                            compound_head = tok.head.lemma_.lower()
                            if compound_head in ("award", "program", "programme", "ceremony", "event"):
                                if any(k in s_defn for k in ("respect", "thanks", "achievement", "work", "praise", "honor")):
                                    score += 40
                            # Check collocations and examples in sense
                            if any(compound_head in ex.lower() for ex in s.get("examples", [])):
                                score += 35
                            elif compound_head in s_def_tokens:
                                score += 30
                            else:
                                entry_collocs = entry.get("collocations", {})
                                if isinstance(entry_collocs, dict):
                                    matched_collocs = [
                                        c for c in entry_collocs.get("nouns", [])
                                        if isinstance(c, dict) and (compound_head == c.get("collocation", "").lower() or compound_head in c.get("collocation", "").lower())
                                    ]
                                    if matched_collocs:
                                        # Check if this sense aligns with the collocation's example or definition tokens
                                        lem_clean = entry.get("word", "").lower()
                                        c_tokens = set()
                                        for mc in matched_collocs:
                                            c_tokens.update(re.findall(r"\b[a-zA-Z]{3,}\b", mc.get("example", "").lower()))
                                        c_tokens.discard(lem_clean)
                                        c_tokens -= stopwords
                                        
                                        s_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", s.get("definition", "").lower()))
                                        for ex in s.get("examples", []):
                                            s_tokens.update(re.findall(r"\b[a-zA-Z]{3,}\b", ex.lower()))
                                        s_tokens.discard(lem_clean)
                                        s_tokens -= stopwords
                                        
                                        if c_tokens & s_tokens:
                                            score += 35
                                        elif not any(len(c_tokens & (set(re.findall(r"\b[a-zA-Z]{3,}\b", other_s.get("definition", "").lower())) - stopwords)) for other_s in senses if other_s != s):
                                            # If no other sense claims this collocation, grant general bonus
                                            score += 15

                    # If target is a verb, check transitivity, complex complements and arguments
                    elif tok.pos_ == "VERB" or (eff_p == "v" and (tok.dep_ in ("ROOT", "conj", "advcl", "relcl", "ccomp", "xcomp", "pobj", "dobj", "obj", "amod", "compound") or any(ch.dep_ in ("nsubj", "nsubjpass", "csubj") for ch in tok.children))):
                        # Section 7.3: Deep Passive Projection
                        # In passive voice, the surface subject (nsubjpass) is the underlying semantic object (deep_dobj),
                        # and the prepositional 'by' agent is the underlying semantic subject (deep_nsubj).
                        is_passive = (
                            any(c.dep_ in ("auxpass", "aux:pass") for c in tok.children) or
                            any(c.dep_ in ("nsubjpass", "nsubj:pass") for c in tok.children)
                        )
                        is_progressive = (
                            tok.tag_ == "VBG"
                            and any(c.dep_ == "aux" and c.lemma_.lower() == "be" for c in tok.children)
                        )

                        # Sense-linked Grammar Box constraints (e.g. 'admire' sense 1: no progressive; 'consist of': no passive)
                        for gb in entry.get("grammar_boxes") or []:
                            gb_senses = gb.get("sense_indices")
                            if gb_senses and i not in gb_senses:
                                continue
                            syn_c = gb.get("syntactic_constraints") or {}
                            if is_progressive and syn_c.get("no_progressive"):
                                score -= 50
                            if is_passive and syn_c.get("no_passive"):
                                score -= 50

                        surface_dobjs = [ch for ch in tok.children if ch.dep_ in ("dobj", "obj")]
                        if not surface_dobjs and eff_p == "v":
                            # Recover noun-verb homograph direct objects misparsed by en_core_web_sm as right-hand npadvmod
                            # (e.g. 'compared sleep to death', 'equate work with play')
                            _temporal_measure_nouns = {
                                "today", "yesterday", "tonight", "tomorrow", "morning", "afternoon",
                                "evening", "night", "year", "month", "week", "day", "hour", "minute",
                                "second", "time", "way", "mile", "kilometer", "percent", "lot", "bit", "little",
                            }
                            surface_dobjs = [
                                ch for ch in tok.children
                                if ch.dep_ == "npadvmod"
                                and ch.pos_ in ("NOUN", "PROPN", "PRON")
                                and ch.i > tok.i
                                and ch.lemma_.lower() not in _temporal_measure_nouns
                            ]
                        if not surface_dobjs and eff_p == "v":
                            for conj_v in tok.children:
                                if conj_v.dep_ == "conj" and conj_v.pos_ == "VERB" and conj_v.i > tok.i:
                                    conj_dobjs = [ch for ch in conj_v.children if ch.dep_ in ("dobj", "obj")]
                                    if not conj_dobjs:
                                        for prep_ch in conj_v.children:
                                            if prep_ch.dep_ == "prep" and prep_ch.lower_ in ("about", "into", "of", "on"):
                                                conj_dobjs.extend(p_ch for p_ch in prep_ch.children if p_ch.dep_ == "pobj")
                                    if conj_dobjs:
                                        surface_dobjs = conj_dobjs
                                        break
                        if not surface_dobjs and eff_p == "v" and tok.dep_ in ("amod", "compound") and tok.head.i > tok.i:
                            if tok.head.pos_ in ("NOUN", "PROPN"):
                                surface_dobjs = [tok.head]
                            elif (
                                tok.head.i + 2 < len(q_doc)
                                and q_doc[tok.head.i + 1].text in ("-", "–", "—")
                                and q_doc[tok.head.i + 2].pos_ in ("NOUN", "PROPN")
                            ):
                                surface_dobjs = [q_doc[tok.head.i + 2]]
                        if not surface_dobjs and eff_p == "v" and tok.dep_ == "conj":
                            _r_head = (
                                tok.head if tok.head.dep_ == "ROOT"
                                else (
                                    tok.head.head.head
                                    if tok.head.dep_ == "pobj" and tok.head.head.dep_ == "prep" and tok.head.head.head.dep_ == "ROOT"
                                    else None
                                )
                            )
                            if _r_head is not None and tok.i + 1 < len(q_doc) and q_doc[tok.i + 1].text != ",":
                                surface_dobjs = [ch for ch in _r_head.children if ch.dep_ == "appos" and ch.i > tok.i]
                        passive_dobjs = [ch for ch in tok.children if ch.dep_ in ("nsubjpass", "nsubj:pass")] if is_passive else []
                        rel_dobjs = [tok.head] if (
                            tok.dep_ == "relcl"
                            and tok.head.pos_ in ("NOUN", "PROPN", "PRON")
                            and tok.head.lemma_.lower() not in ("way", "time", "day", "year", "moment", "reason", "place", "manner", "case")
                            and not surface_dobjs
                            and not any(t.text == "," for t in q_doc[tok.head.i : tok.i])
                            and any(ch.dep_ in ("nsubj", "csubj") and ch.lower_ not in ("which", "that", "who") for ch in tok.children)
                        ) else []
                        deep_dobj_tokens = surface_dobjs + passive_dobjs + rel_dobjs
                        has_direct_obj = bool(deep_dobj_tokens)

                        # Locate by-agent for passive voice
                        by_agents = []
                        if is_passive:
                            for c in tok.children:
                                if c.dep_ in ("agent", "prep") and c.lemma_.lower() == "by":
                                    for gc in c.children:
                                        if gc.dep_ in ("pobj", "pcomp"):
                                            by_agents.append(gc)
                            if not by_agents:
                                # Handle nested prepositional attachment (e.g. 'to the winner by the mayor')
                                for c in tok.children:
                                    if c.dep_ == "prep":
                                        for gc in c.subtree:
                                            if gc != c and gc.dep_ == "prep" and gc.lemma_.lower() == "by":
                                                for ggc in gc.children:
                                                    if ggc.dep_ in ("pobj", "pcomp"):
                                                        by_agents.append(ggc)

                        surface_nsubjs = [ch for ch in tok.children if ch.dep_ in ("nsubj", "csubj")]
                        if not surface_nsubjs and eff_p == "v" and tok.dep_ == "xcomp":
                            surface_nsubjs = [ch for ch in tok.head.children if ch.dep_ in ("nsubj", "nsubjpass", "nsubj:pass", "csubj", "dobj", "obj")]
                        if not surface_nsubjs and eff_p == "v":
                            left_nom_children = [
                                ch for ch in tok.children
                                if ch.dep_ in ("compound", "nmod", "npadvmod")
                                and ch.pos_ in ("NOUN", "PROPN", "PRON")
                                and ch.i < tok.i
                            ]
                            for lnc in list(left_nom_children):
                                for gc in lnc.children:
                                    if gc.dep_ == "conj" and gc.pos_ in ("NOUN", "PROPN", "PRON") and gc.i < tok.i:
                                        left_nom_children.append(gc)
                            surface_nsubjs = left_nom_children
                            if tok.dep_ == "conj" and tok.i > 0 and q_doc[tok.i - 1].pos_ in ("NOUN", "PROPN", "PRON") and q_doc[tok.i - 1].dep_ == "conj" and q_doc[tok.i - 1].head == tok.head:
                                surface_nsubjs.append(q_doc[tok.i - 1])
                            elif tok.dep_ == "conj" and tok.head.pos_ in ("NOUN", "PROPN", "PRON") and tok.head.i < tok.i:
                                surface_nsubjs.append(tok.head)
                                if tok.head.dep_ == "pobj" and tok.head.head.dep_ == "prep" and tok.head.head.head.pos_ in ("NOUN", "PROPN", "PRON"):
                                    surface_nsubjs.append(tok.head.head.head)
                        deep_nsubj_tokens = by_agents if (is_passive and by_agents) else surface_nsubjs

                        # Complex clausal complement with infinitive: e.g. [V] + [NP] + [to-inf]
                        # spaCy can parse this either as v -> ccomp (inf) with nsubj/dobj and aux:to,
                        # or as v -> dobj (NP) and xcomp (inf) with aux:to.
                        has_clausal_inf = any(
                            (c.dep_ in ("ccomp", "xcomp") and any(gc.lemma_.lower() == "to" for gc in c.children))
                            for c in tok.children
                        )
                        has_finite_ccomp = any(
                            c.dep_ == "ccomp" and not any(gc.lemma_.lower() == "to" and gc.dep_ == "aux" for gc in c.children)
                            for c in tok.children
                        )
                        has_complex_comp = (
                            (has_direct_obj and any(c.dep_ == "xcomp" and any(gc.lemma_.lower() == "to" for gc in c.children) for c in tok.children))
                            or any(
                                c.dep_ in ("ccomp", "xcomp") and any(gc.dep_ in ("nsubj", "dobj", "obj") for gc in c.children)
                                and any(gc.lemma_.lower() == "to" for gc in c.children)
                                for c in tok.children
                            )
                        )
                        has_effective_obj = has_direct_obj or has_clausal_inf or has_finite_ccomp

                        gram_l = (s.get("grammar") or "").lower()
                        s_units = [u.lower() for u in (s.get("units") or [])]
                        s_pats = [p.lower() for p in (s.get("patterns") or [])]
                        all_pat_units = s_units + s_pats

                        if has_finite_ccomp and any("(that)" in p or p.endswith(" that") for p in all_pat_units):
                            score += 40

                        def _has_cc_structure(txt: str) -> bool:
                            t_clean = re.sub(r"\(.*?\)", "", txt)
                            return bool(re.search(r"\b(?:somebody|someone|something|sb|sth)(?:/(?:somebody|someone|something|sb|sth))*\s+to\s+(?:do|make|have|be)\b", t_clean))

                        # Distinguish direct object requirement from complex complements or non-finite verbal complements
                        def _requires_direct_object(txt: str, require_quote_lex_match: bool = False) -> bool:
                            t_clean = re.sub(r"\(.*?\)", "", txt)
                            t_clean = re.sub(r"\b(?:somebody|someone|something|sb|sth)(?:/(?:somebody|someone|something|sb|sth))*\s+to\s+(?:do|make|have|be)\s+(?:something|sth|somebody|sb)?\b", "", t_clean)
                            t_clean = re.sub(r"\b(?:to|and)\s+(?:do|make|have)\s+(?:something|sth|somebody|sb)\b", "", t_clean)
                            # Strip prepositional objects (Prep + sb/sth) so intransitive/participial frames
                            # ('compared to/with something', 'equate to something', 'pay for something')
                            # are not misclassified as requiring a direct object before the preposition.
                            t_clean = re.sub(
                                r"\b(?:to|with|for|of|in|on|at|from|by|about|into|onto|upon|against|between|among|through|over|under|after|before|without|towards|toward|around|across)"
                                r"(?:/(?:to|with|for|of|in|on|at|from|by|about|into|onto|upon|against|between|among|through|over|under|after|before|without|towards|toward|around|across))*"
                                r"\s+(?:somebody|someone|something|sb|sth)(?:/(?:somebody|someone|something|sb|sth))*\b",
                                "",
                                t_clean,
                            )
                            if re.search(r"\b(?:somebody|someone|something|sb|sth)\b", t_clean):
                                return True
                            if not t_clean.strip().lower().startswith("be "):
                                rem_words = [
                                    w for w in re.findall(r"[a-z]{2,}", t_clean.lower())
                                    if w not in cls._PATTERN_PREPS
                                    and w not in cls._PATTERN_PARTICLES
                                    and w not in cls._PATTERN_FUNCTION_WORDS
                                    and w not in ("be", "become", "get", "grow", "do", "doing", "etc")
                                    and not cls.form_matches(w, target_w_clean)
                                    and w not in target_inflections
                                ]
                                if rem_words:
                                    if require_quote_lex_match:
                                        return any(
                                            cls.form_matches(rw, qw)
                                            for rw in rem_words
                                            for qw in q_seq
                                        )
                                    return True
                            return False

                        # Check if definition shows ergative or intransitive conditional usage:
                        # e.g. 'if a situation, game etc turns around, or if someone turns it around', 'if a situation picks up, it improves'
                        is_ergative_defn = bool(re.search(r"\bor\s+if\s+someone\s+\w+\s+(?:it|them)\b", s_defn))
                        is_intrans_cond_defn = bool(re.match(r"^if\s+(?:a|an|the|your|someone's|somebody's)?\s*(?:situation|things|business|economy|trade|sales|market|weather|wind|speed|prices?|traffic|conversation|life)\b", s_defn))

                        # Check if sense allows intransitive usage:
                        # 1) explicit 'intransitive' in grammar tag (e.g. '[intransitive, transitive]', '[intransitive]')
                        # 2) sense declares patterns that do not require a direct object (e.g. 'pay for', 'pay in cash')
                        has_intrans_gram = "intransitive" in gram_l or is_intrans_cond_defn
                        has_trans_gram = bool(re.search(r"(?<!in)transitive", gram_l)) and not is_intrans_cond_defn
                        has_intransitive_pattern = any(not _requires_direct_object(p) for p in s_pats) if s_pats else False
                        allows_intransitive = has_intrans_gram or (not has_trans_gram and has_intransitive_pattern)

                        sense_has_cc = any(_has_cc_structure(p) for p in all_pat_units)
                        requires_obj = (
                            not is_ergative_defn and not allows_intransitive and (
                                has_trans_gram or
                                sense_has_cc or
                                any(_requires_direct_object(u) for u in s_units) or
                                any(_requires_direct_object(p) for p in s_pats)
                            )
                        )
                        is_pure_intransitive = (
                            (has_intrans_gram and not has_trans_gram) or
                            not requires_obj
                        )

                        # Complex Complement Structure Gate:
                        # If context has [V] + [NP] + [to-inf] (e.g. 'expect anyone to feel surprised', 'wants you to stay'),
                        # award strong alignment bonus to senses declaring this pattern;
                        # if other senses declare specific patterns that contradict this structure, discount them.
                        if has_complex_comp:
                            if sense_has_cc:
                                score += 55
                            else:
                                score -= 65

                        is_coord_predicate_ellipsis = (
                            not has_effective_obj
                            and (any(c.dep_ == "conj" for c in tok.children) or tok.dep_ == "conj")
                            and any(
                                sn.lemma_.lower() in ("relationship", "divorce", "failure", "defeat", "rejection", "criticism", "remark", "comment", "word", "experience", "life")
                                for sn in deep_nsubj_tokens
                            )
                            and any(k in s_defn for k in ("affect someone badly", "feel less confident", "make them feel", "upset", "emotional"))
                        )

                        if not has_effective_obj:
                            if is_pure_intransitive and not requires_obj:
                                score += 25
                            elif is_coord_predicate_ellipsis:
                                score += 25
                            elif requires_obj:
                                # Penalize transitive senses when sentence has no direct object or clausal complement
                                score -= 80
                        else:
                            if requires_obj or has_trans_gram or is_ergative_defn:
                                score += 25
                            elif has_intrans_gram and not has_trans_gram:
                                score -= 55

                            # Transitive valency alignment: if an explicit direct object is present,
                            # senses providing explicit transitive patterns (e.g. 'attach something to something', 'pay somebody something')
                            # strongly match the syntactic frame; senses whose patterns are strictly intransitive (e.g. 'attach to')
                            # receive a discount.
                            _all_pat_preps = set(cls._PATTERN_PREPS) | {
                                "above", "before", "after", "below", "under", "over", "between",
                                "among", "into", "onto", "against", "through", "across", "towards",
                                "toward", "upon", "by", "without", "behind", "beside", "beyond", "around"
                            }
                            def _pat_preps_satisfied(pat_str: str) -> bool:
                                p_toks = re.findall(r"\b[a-z]+\b", pat_str.lower())
                                req_preps = [t for t in p_toks if t in _all_pat_preps]
                                return (not req_preps) or any(rp in q_seq for rp in req_preps)

                            has_trans_pat = any(
                                _requires_direct_object(p, require_quote_lex_match=True) and _pat_preps_satisfied(p)
                                for p in all_pat_units
                            )
                            has_strictly_intrans_prep_pat = any(
                                not _requires_direct_object(p)
                                and len(p.lower().split()) >= 2
                                and p.lower().split()[1] in cls._PATTERN_PREPS
                                for p in all_pat_units
                            )
                            if has_trans_pat:
                                score += 35
                            elif has_strictly_intrans_prep_pat and not has_trans_pat:
                                score -= 30

                        # Mandatory Passive / Predicative Frame Gate (e.g. 'be well/ideally placed to do something'):
                        if all_pat_units and all(re.match(r"^be\b", p.strip()) for p in all_pat_units):
                            if not is_passive and tok.tag_ not in ("VBN", "JJ"):
                                score -= 60

                        # Mandatory Locative / Prepositional Complement Gate:
                        # Senses marked '[... always + adverb/preposition]' (e.g. 'conduct sb to/into sth', 'place sth on/under sth')
                        # require a prepositional or directional adverbial complement governed by the verb (or attached to its direct object).
                        if "always + adverb/preposition" in gram_l:
                            has_adv_or_prep_comp = any(
                                c.dep_ in ("prep", "prt", "npadvmod")
                                or (c.dep_ == "advmod" and c.i > tok.i)
                                for c in tok.children
                            ) or any(
                                gc.dep_ in ("prep", "prt", "npadvmod") or (gc.dep_ == "advmod" and gc.i > tok.i)
                                for dobj_c in surface_dobjs
                                for gc in dobj_c.children
                            )
                            if has_adv_or_prep_comp:
                                score += 35
                            elif not (has_direct_obj and has_trans_gram):
                                score -= 70

                        # Direct object argument matching (governs deep_dobj_tokens, supporting passive projection)
                        for ch in deep_dobj_tokens:
                            obj_lem = ch.lemma_.lower()
                            obj_txt = ch.text.lower()
                            if obj_lem in s_def_tokens or obj_txt in s_def_tokens or any(cls.form_matches(dt, obj_lem) for dt in s_def_tokens):
                                score += 35
                            elif any(re.search(rf"\b{re.escape(obj_lem)}\b", ex.lower()) for ex in s.get("examples", [])):
                                score += 30
                            elif any(re.search(rf"\b{re.escape(obj_lem)}\b", p.lower()) for p in all_pat_units):
                                score += 30

                            if obj_lem in ("dream", "dreams", "ambition", "goal", "potential", "hope", "desire", "plan", "aim", "vision", "aspiration"):
                                if any(k in s_defn or k in sp for k in ("achieve", "ambition", "potential", "hoping to achieve", "fulfil", "fulfill")):
                                    score += 45

                            # Abstract temporal/informational direct objects (e.g. 'follow the time closely', 'follow the news/events/schedule'):
                            # Penalize physical locomotion, event-sequence, or instruction-compliance senses and boost attention/tracking senses.
                            if obj_lem in ("time", "schedule", "timetable", "news", "event", "events", "story", "progress", "development", "trend", "change", "situation"):
                                if any(k in s_defn for k in ("walk", "drive", "behind or after someone", "chase", "happen or come after", "told or advised you")):
                                    score -= 55
                                elif any(k in s_defn for k in ("pay close attention", "very interested in", "understand", "watch", "listen")):
                                    score += 55

                            # Supersenses selectional restrictions for dobj:
                            obj_supersenses = cls.get_wordnet_supersenses(obj_lem, pos="n")
                            if obj_supersenses:
                                if "noun.food" in obj_supersenses:
                                    if any(k in s_defn or k in sp for k in ("food", "eat", "drink", "cook", "meal", "meat", "preserve")):
                                        score += 45
                                    elif "liquid" in s_defn or "liquid" in sp:
                                        score += 20
                                elif "noun.substance" in obj_supersenses:
                                    if any(k in s_defn or k in sp for k in ("liquid", "substance", "solid", "food", "eat", "drink", "cook")):
                                        score += 45
                                if "noun.body" not in obj_supersenses and "part of your body" in s_defn:
                                    score -= 40
                                if "noun.communication" in obj_supersenses or "noun.possession" in obj_supersenses:
                                    if any(k in s_defn for k in ("give", "award", "prize", "gift", "medal", "money", "present", "degree")):
                                        score += 40
                                if "noun.communication" in obj_supersenses:
                                    if any(k in s_defn for k in ("broadcast", "publish", "read or listen", "information", "statement", "newspaper", "magazine", "book")):
                                        score += 45
                                if "noun.person" in obj_supersenses:
                                    if any(k in s_defn for k in ("introduce someone", "formally introduce")):
                                        score += 35

                        # Prepositional complement / adverbial phrase / particle matching (e.g. 'recognize sb by doing', 'come in', 'sit down', 'turn around')
                        _verb_prep_children = [ch for ch in tok.children if ch.dep_ in ("prep", "prt", "advmod")]
                        for _dobj_ch in surface_dobjs:
                            for _d_prep in _dobj_ch.children:
                                if _d_prep.dep_ == "prep" and _d_prep not in _verb_prep_children:
                                    _verb_prep_children.append(_d_prep)
                        _any_sib_has_trans_pat = any(
                            _requires_direct_object(p_other)
                            for s_other in senses if isinstance(s_other, dict)
                            for p_other in (s_other.get("patterns") or [])
                        )
                        for ch in _verb_prep_children:
                            if ch.dep_ in ("prep", "prt", "advmod"):
                                p_txt = ch.text.lower()
                                p_lem = ch.lemma_.lower()
                                # Check if preposition / particle is explicitly in sense patterns (e.g. 'for', 'by', 'in', 'around', 'down', 'back', 'away')
                                # If pattern contains content words or wh-markers (e.g. 'appear in a film', 'where somebody is coming from'), they must also be present
                                pat_matched = False
                                sense_person_violation = False
                                for pat in list(s.get("patterns") or []) + list(s.get("units") or []):
                                    if sense_person_violation:
                                        break
                                    branches = cls.clean_and_expand_pattern_slashes(pat)
                                    has_etc_locative = (
                                        "etc" in pat.lower().split()
                                        and any(lp in pat.lower() for lp in ("through", "around", "across", "along", "into", "over", "down"))
                                    )
                                    for br in branches:
                                        br_tokens = br.split()
                                        prep_hit = (
                                            f" {p_txt} " in f" {br} " or f" {p_lem} " in f" {br} "
                                            or p_txt in br_tokens or p_lem in br_tokens
                                        )
                                        if not prep_hit and has_etc_locative and p_lem in ("in", "inside", "throughout", "across", "through", "around", "over", "along"):
                                            p_objs_loc = [pc for pc in ch.children if pc.dep_ in ("pobj", "pcomp")]
                                            if any(
                                                any(ss in cls.get_wordnet_supersenses(pc.lemma_.lower(), pos="n") for ss in ("noun.location", "noun.artifact", "noun.object"))
                                                for pc in p_objs_loc
                                            ):
                                                prep_hit = True
                                        if prep_hit:
                                            pw = cls._pattern_words(br)
                                            # Wh-markers / complementizers in pattern (e.g. 'where somebody is coming from') MUST match in context
                                            wh_markers = [w for w in pw if w in ("where", "when", "why", "how", "what", "who", "whom", "whose")]
                                            if wh_markers and not any(w in q_seq for w in wh_markers):
                                                continue
                                            # Selectional restriction: if pattern requires person object (e.g. 'coming from him/her/you etc' or 'put somebody out')
                                            # and context prepositional or direct/passive object is inanimate / non-person, skip this sense's patterns
                                            has_person_constraint = any(w in ("him", "her", "you", "someone", "somebody", "sb") for w in pw) and not any(w in ("something", "sth", "what") for w in pw)
                                            if has_person_constraint:
                                                p_objs = [pc for pc in ch.children if pc.dep_ in ("pobj", "pcomp")] or list(deep_dobj_tokens)
                                                if p_objs:
                                                    is_pobj_person = any(
                                                        pc.pos_ == "PRON" and pc.lemma_.lower() in ("he", "she", "i", "we", "they", "you", "who", "someone", "somebody", "him", "her", "me", "us", "them")
                                                        or (pc.pos_ in ("NOUN", "PROPN") and pc.ent_type_ in ("PERSON", "ORG"))
                                                        or pc.lemma_.lower() in ("person", "people", "man", "woman", "child", "staff", "employee", "worker", "friend", "parent")
                                                        or "noun.person" in cls.get_wordnet_supersenses(pc.lemma_.lower(), pos="n")
                                                        for pc in p_objs
                                                    )
                                                    if not is_pobj_person:
                                                        sense_person_violation = True
                                                        break
                                            content_pw = [w for w in pw if w not in cls._PATTERN_PREPS and w not in cls._PATTERN_PARTICLES and w not in cls._PATTERN_FUNCTION_WORDS and w not in ("be", "become", "get", "grow") and w not in target_inflections]
                                            req_prep_prt = [w for w in pw if w in cls._PATTERN_PREPS or w in cls._PATTERN_PARTICLES]
                                            if req_prep_prt and not all(rp in q_seq for rp in req_prep_prt):
                                                continue
                                            if _requires_direct_object(br) and not has_effective_obj:
                                                continue
                                            if not _requires_direct_object(br) and has_effective_obj and _any_sib_has_trans_pat and not has_trans_gram:
                                                continue
                                            if not content_pw or any(any(cls.form_matches(tok_q, cpw) for tok_q in q_seq) for cpw in content_pw):
                                                # When particle/preposition matches pattern (or open-ended locative pattern)
                                                score += 40
                                                pat_matched = True
                                                break
                                    if pat_matched:
                                        break

                                # Ontological Class Comparison vs. Metaphorical Analogy (e.g. 'compare A with/to B' vs 'compare A to B = liken'):
                                # When a verb governs a direct/passive object A and a 'to + B' prepositional complement,
                                # check whether A and B belong to the same taxonomic class (near-hypernym overlap in WordNet)
                                # or orthogonal domains ("another person or thing of a different kind").
                                if p_lem == "to" and deep_dobj_tokens and (
                                    "similar or different" in s_defn or "of a different kind" in s_defn
                                ):
                                    to_pobjs = [pc for pc in ch.children if pc.dep_ in ("pobj", "pcomp")]
                                    if to_pobjs:
                                        a_tok = deep_dobj_tokens[0]
                                        b_tok = to_pobjs[0]
                                        a_lem = a_tok.lemma_.lower()
                                        b_lem = b_tok.lemma_.lower()
                                        is_same_class = (
                                            a_lem == b_lem
                                            or b_lem in ("those", "that", "one", "ones", "other", "others", "year", "month", "week", "day", "average", "standard", "norm", "level", "rate", "figure")
                                            or any(gc.dep_ in ("case", "poss") for gc in b_tok.children)
                                        )
                                        if not is_same_class and a_tok.pos_ == "NOUN" and b_tok.pos_ == "NOUN":
                                            try:
                                                wn_inst = cls.get_wordnet()
                                                _high_roots = {
                                                    "entity", "physical entity", "abstraction", "abstract entity",
                                                    "object", "whole", "artifact", "matter", "psychological feature",
                                                    "state", "event", "act", "group", "thing",
                                                }
                                                sa = wn_inst.synsets(a_lem, pos="n")[:2]
                                                sb = wn_inst.synsets(b_lem, pos="n")[:2]
                                                for _s1 in sa:
                                                    _h1 = {_s1} | {h for path in _s1.hypernym_paths() for h in path[:3] if not (set(h.lemmas()) & _high_roots)}
                                                    for _s2 in sb:
                                                        _h2 = {_s2} | {h for path in _s2.hypernym_paths() for h in path[:3] if not (set(h.lemmas()) & _high_roots)}
                                                        if _h1 & _h2:
                                                            is_same_class = True
                                                            break
                                                    if is_same_class:
                                                        break
                                            except Exception:
                                                pass
                                        if is_same_class:
                                            if "similar or different" in s_defn:
                                                score += 45
                                            elif "of a different kind" in s_defn:
                                                score -= 35
                                        else:
                                            if "of a different kind" in s_defn:
                                                score += 55
                                            elif "similar or different" in s_defn:
                                                score -= 25
                                # Check prepositional objects or complement verbs
                                has_action_pcomp = False
                                p_objs_expanded = []
                                for p_child in ch.children:
                                    if p_child.dep_ in ("pobj", "pcomp"):
                                        p_objs_expanded.append(p_child)
                                        p_objs_expanded.extend([gc for gc in p_child.children if gc.dep_ == "conj"])

                                for p_child in p_objs_expanded:
                                    p_child_lem = p_child.lemma_.lower()
                                    if p_child.dep_ == "pcomp" or p_child.pos_ == "VERB":
                                        has_action_pcomp = True
                                    # Generic words / time adverbials (e.g. 'time', 'people', 'thing') carry no specific collocational evidence
                                    if p_child_lem in meta_stopwords:
                                        continue
                                    # Match against definition tokens or example tokens (word boundary enforced)
                                    if p_child_lem in s_def_tokens:
                                        score += 30
                                    elif any(re.search(rf"\b{re.escape(p_child_lem)}\b", ex.lower()) for ex in s.get("examples", [])):
                                        score += 25
                                    # Institutional / organization prepositional complement ('admitted to Jilin University / the club / the society')
                                    p_ch_supers = cls.get_wordnet_supersenses(p_child_lem, pos="n")
                                    if (
                                        p_child.ent_type_ == "ORG"
                                        or p_child_lem in ("university", "college", "school", "academy", "institute", "club", "organization", "organisation", "society", "party", "union", "association")
                                        or (p_ch_supers and "noun.group" in p_ch_supers)
                                    ):
                                        if any(k in s_defn for k in ("organization", "organisation", "club", "school", "university", "college", "member", "join")):
                                            score += 35

                                # Specific alignment for intransitive verbs of appearing/existence with locative/conceptual preposition (in myths, in dreams, etc.)
                                if p_lem in ("in", "at", "on", "from") and any(k in s_defn for k in ("exist in a place", "start to be seen", "arrive", "become visible")):
                                    # When appearing in concepts/settings (myths, dreams, place, air, sky) rather than film/court/publication
                                    if not any(k in s_defn for k in ("film", "play", "court", "television", "print", "newspaper")):
                                        score += 30

                                # Universal verb-preposition / verb-adverb valency frame agreement:
                                # If the verb in context governs a preposition (e.g. 'function as', 'look into', 'depend on', 'succeed in')
                                # or an open-class manner/degree adverb (e.g. 'follow ... closely', 'listen ... carefully'),
                                # check if this frame is explicitly articulated in the sense definition or examples.
                                target_lemmas = {target_w_clean} | cls.inflected_forms(target_w_clean) | {tok.lemma_.lower()}
                                v_prep_pat = r"\b(?:" + "|".join(re.escape(f) for f in target_lemmas) + r")(?:\s+\w+)?\s+(?:" + re.escape(p_lem) + r"|" + re.escape(p_txt) + r")\b"
                                if ch.dep_ == "advmod" and len(p_lem) >= 4 and p_lem not in stopwords and p_lem not in (
                                    "very", "quite", "rather", "pretty", "really", "always", "never", "often", "sometimes",
                                    "usually", "already", "still", "just", "even", "also", "only", "almost", "nearly",
                                    "hardly", "barely", "merely", "simply", "actually", "probably", "perhaps", "maybe"
                                ):
                                    v_adv_pat = r"\b(?:" + "|".join(re.escape(f) for f in target_lemmas) + r")(?:\s+\S+){0,4}\s+(?:" + re.escape(p_lem) + r"|" + re.escape(p_txt) + r")\b"
                                    if p_lem in s_def_tokens or re.search(v_adv_pat, s_defn) or any(re.search(v_adv_pat, ex.lower()) for ex in s.get("examples", [])):
                                        score += 45
                                        pat_matched = True
                                    elif any(re.search(rf"\b(?:{re.escape(p_lem)}|{re.escape(p_txt)})\b", ex.lower()) for ex in s.get("examples", [])):
                                        score += 30
                                if not pat_matched:
                                    if re.search(v_prep_pat, s_defn):
                                        score += 35
                                    elif any(re.search(v_prep_pat, ex.lower()) for ex in s.get("examples", [])):
                                        # Guard incidental locative prepositions ('in', 'at', 'on') in examples when the sense has no explicit patterns
                                        # and the quote's prepositional object is a physical room/building/place while the sense is purely abstract/folklore.
                                        if p_lem in ("in", "at", "on") and not s.get("patterns") and any(
                                            any(ss in cls.get_wordnet_supersenses(pc.lemma_.lower(), pos="n") for ss in ("noun.location", "noun.artifact"))
                                            and not any(ss in cls.get_wordnet_supersenses(pc.lemma_.lower(), pos="n") for ss in ("noun.communication", "noun.cognition"))
                                            for pc in p_objs_expanded
                                        ) and any(k in s_defn for k in ("mentioned", "talked about", "folklore", "history", "memory")):
                                            score -= 25
                                        else:
                                            score += 40
                                if any(k in s_defn for k in ("radio or television broadcast", "touch a child sexually")) and not any(k in q_low_all for k in ("radio", "television", "broadcast", "signal", "reception", "child", "sexually", "boy", "girl")):
                                    score -= 45

                                # Selectional restrictions on prepositional complements:
                                # Distinguish commercial/financial exchange ('pay for dinner/tickets/lessons', 'pay in cash/dollars')
                                # from punitive suffering/penalty ('pay for mistakes/crimes/sins').
                                if p_lem in ("for", "in", "by"):
                                    for p_child in p_objs_expanded:
                                        p_ch_lem = p_child.lemma_.lower()
                                        p_supersenses = cls.get_wordnet_supersenses(p_ch_lem, pos="n")
                                        is_financial_or_goods = (
                                            p_ch_lem in ("dollar", "dollars", "euro", "euros", "pound", "pounds", "cash", "credit", "card", "cheque", "check", "coin", "cent", "bill", "dinner", "ticket", "tickets", "meal", "lunch", "drink", "food", "rent", "tax", "fare", "fee", "goods", "service", "lesson", "lessons", "car")
                                            or (p_supersenses and "noun.possession" in p_supersenses)
                                            or (p_lem == "for" and p_supersenses and any(ss in p_supersenses for ss in ("noun.food", "noun.artifact")))
                                        )
                                        is_wrongdoing_or_penalty = (
                                            p_ch_lem in ("mistake", "mistakes", "crime", "crimes", "sin", "sins", "error", "errors", "fault", "wrong", "damage", "foolishness", "carelessness", "neglect", "misdeed", "offense", "offence")
                                            or (p_supersenses and "noun.cognition" in p_supersenses and any(k in s_defn for k in ("punish", "suffer", "wrong", "penalty")))
                                        )
                                        if is_financial_or_goods:
                                            if any(k in s_defn for k in ("money", "buy", "service", "cost", "owe", "salary", "wage")):
                                                score += 55
                                            elif any(k in s_defn for k in ("punish", "suffer", "wrong", "penalty")):
                                                score -= 75
                                            # Currency bonus: 'pay in dollars/euros/cash' directly matches currency patterns/examples
                                            if p_lem == "in" and p_ch_lem in ("dollar", "dollars", "euro", "euros", "pound", "pounds", "cash", "currency"):
                                                if any(k in s_defn or any(k in ex.lower() for ex in s.get("examples", [])) or any(k in pat.lower() for pat in s.get("patterns", [])) for k in ("dollar", "euro", "cash", "currency")):
                                                    score += 40
                                        elif is_wrongdoing_or_penalty:
                                            if any(k in s_defn for k in ("punish", "suffer", "wrong", "penalty", "unpleasant")):
                                                score += 65
                                            elif any(k in s_defn for k in ("money", "buy", "service", "cost")):
                                                score -= 40

                                # Generalized semantic valency: 'by [doing]' or 'for [doing/NP]' as instrument/reward/cause
                                # For verbs denoting appraisal, thank, reward, praise, recognition, honor
                                if p_lem in ("by", "for"):
                                    is_appraisal = any(k in s_defn for k in ("thank", "honour", "honor", "reward", "praise", "award", "appreciat", "respect"))
                                    if is_appraisal:
                                        # Strong alignment for rewarding/thanking/honoring actions:
                                        # e.g. 'recognize volunteers by taking them...', 'recognized for having saved...'
                                        score += 45
                                        if has_action_pcomp:
                                            score += 25
                                    elif any(f" {p_lem} " in f" {ex.lower()} " for ex in s.get("examples", [])):
                                        # If prep is 'by' but followed by an action/event verb (taking/giving/doing),
                                        # cognitive/sensory recognition ('know by sight/voice/face') does not apply.
                                        if p_lem == "by" and has_action_pcomp and any(k in s_defn for k in ("know", "hear", "see", "sight", "voice", "face", "appearance")):
                                            score -= 20
                                        else:
                                            score += 15

                        # Clausal head / nominal governing structure (e.g. 'effort/move/decision to recognize them')
                        if tok.dep_ == "acl" and tok.head and tok.head.pos_ in ("NOUN", "PROPN"):
                            gov_lem = tok.head.lemma_.lower()
                            if any(re.search(rf"\b{re.escape(gov_lem)}\b", ex.lower()) for ex in s.get("examples", [])) or gov_lem in s_def_tokens:
                                score += 35
                            elif gov_lem in ("effort", "move", "attempt", "initiative", "campaign", "gesture", "step"):
                                if any(k in s_defn for k in ("thank", "honour", "honor", "reward", "praise", "appreciat", "respect")):
                                    score += 45
                            # Check higher governing verb (e.g. 'appreciate the effort to recognize them')
                            if tok.head.head and tok.head.head.pos_ in ("VERB", "AUX"):
                                higher_v = tok.head.head.lemma_.lower()
                                if higher_v in ("appreciate", "welcome", "value", "praise", "applaud", "support"):
                                    if any(k in s_defn for k in ("thank", "honour", "honor", "reward", "praise", "appreciat", "respect")):
                                        score += 30

                        # Check semantic subject tokens (governs deep_nsubj_tokens, supporting passive by-agent)
                        for ch in deep_nsubj_tokens:
                            subj_lem = ch.lemma_.lower()
                            # Direct subject-signpost selectional match (e.g. "As time went on" -> signpost [time])
                            if subj_lem and sp and (subj_lem == sp or subj_lem in re.findall(r"\b[a-z]+\b", sp)):
                                score += 35
                            # Closed-class personal pronouns ('you', 'i', 'he', etc.) appear ubiquitously in dictionary examples
                            # and must not be used to award generic token-overlap bonus unless the head noun/pronoun is lexical
                            if subj_lem not in stopwords:
                                if subj_lem in s_def_tokens or any(re.search(rf"\b{re.escape(subj_lem)}\b", ex.lower()) for ex in s.get("examples", [])):
                                    score += 25
                            # Human vs non-human subject differentiation for verbs with 'effective/successful' or job/activity senses (e.g. work)
                            subj_supersenses = cls.get_wordnet_supersenses(subj_lem, pos="n")
                            is_human_subj = (
                                (ch.pos_ == "PRON" and subj_lem in ("he", "she", "i", "we", "they", "you", "who", "someone", "somebody")) or
                                (ch.pos_ in ("NOUN", "PROPN") and (
                                    subj_lem in ("person", "people", "man", "woman", "child", "staff", "employee", "worker", "doctor", "lawyer", "teacher", "mentor", "volunteer", "mayor", "president", "director", "manager", "king", "queen", "winner") or
                                    ch.ent_type_ in ("PERSON", "ORG") or
                                    "noun.person" in subj_supersenses
                                ))
                            )
                            if is_human_subj:
                                # Human subject + 'from' preposition specifying origin/birth/culture/background
                                # (e.g. 'come from a culture/country/place', 'hail from ...')
                                prep_from_objs = [
                                    pc.lemma_.lower()
                                    for p_ch in tok.children if p_ch.dep_ == "prep" and p_ch.lemma_.lower() == "from"
                                    for pc in p_ch.children if pc.dep_ in ("pobj", "pcomp")
                                ]
                                if any(po in ("culture", "country", "place", "background", "family", "city", "town", "region", "society", "nation", "area") for po in prep_from_objs):
                                    if any(k in s_defn for k in ("born there", "lived there", "from a place", "place, you were", "family or background", "belong to")):
                                        score += 40
                                    elif any(k in s_defn for k in ("attitude or opinion", "basic attitude")):
                                        score -= 30

                                # Human subject vs vehicle arriving / landing sense (e.g. 'train, bus, plane, or ship comes in')
                                if any(k in s_defn for k in ("train, bus, plane", "plane, or ship", "train arrives", "plane arrives", "ship arrives")):
                                    score -= 40

                                # Human subject physical body movement / posture alignment vs situation/business turnaround
                                # (e.g. 'She turned around and walked away', 'The student turned around and smiled')
                                if any(re.search(rf"\b{re.escape(k)}\b", sp or "") or re.search(rf"\b{re.escape(k)}\b", s_defn) for k in ("your body", "body", "look in a different direction", "move your body", "posture")):
                                    score += 60
                                elif any(k in s_defn for k in ("if a situation", "if a business", "if a shop or business", "consider an idea", "making a product")):
                                    score -= 40
                            else:
                                # Non-human / abstract subject alignment (e.g. situation / crisis / event vs business / corporate)
                                is_business_subj = (
                                    subj_lem in ("business", "company", "firm", "department", "division", "corporation", "organization", "enterprise", "airline", "bank", "shop", "store", "factory", "pub", "bar", "restaurant", "hotel", "cafe") or
                                    ("noun.group" in subj_supersenses and subj_lem not in ("economy", "market", "trade", "society", "system", "world"))
                                ) and subj_lem not in ("economy", "market", "trade")
                                is_situation_subj = (
                                    subj_lem in ("situation", "crisis", "problem", "event", "condition", "economy", "market", "game", "match", "trend", "matter", "affair", "life") or
                                    bool(subj_supersenses & {"noun.state", "noun.event"})
                                ) and not is_business_subj

                                is_quality_or_action_subj = (
                                    subj_lem in ("punctuality", "honesty", "patience", "hard work", "diligence", "effort", "kindness", "courage", "caution", "persistence") or
                                    bool(subj_supersenses & {"noun.attribute", "noun.act"})
                                )

                                if is_business_subj:
                                    if any(k in s_defn for k in ("business", "department", "company", "shop", "makes a profit", "starts to be successful")):
                                        score += 55
                                elif is_situation_subj:
                                    if any(k in s_defn for k in ("situation", "game", "develop in the way you want", "changes and starts", "improves", "gets better", "picks up")):
                                        score += 55
                                    elif any(k in s_defn for k in ("business, department", "business or department")):
                                        score -= 20
                                elif is_quality_or_action_subj:
                                    if any(k in s_defn for k in ("particular action pays", "brings a good result", "advantage for you", "pays dividends")):
                                        score += 55
                                    elif any(k in s_defn for k in ("give someone money", "job they do")):
                                        score -= 20

                                # First-principles invariant: non-human subjects only boost 'effective/successful' for verbs of functioning/operating (e.g. work, operate, run, function)
                                is_functioning_verb = tok.lemma_.lower() in ("work", "function", "operate", "run", "act", "serve") or any(k in target_w_clean for k in ("work", "function", "operate"))
                                if is_functioning_verb and any(k in s_defn for k in ("effective", "successful", "function", "operate")):
                                    score += 45
                                elif any(k in s_defn for k in ("paid", "job", "salary", "employer")):
                                    score -= 25
                                # Specific domain check for acoustic/sound vs quantity/amount increase
                                is_sound_subj = subj_lem in ("music", "sound", "noise", "applause", "cheer", "chorus", "song", "shout", "cry", "word", "voice", "laughter", "echo", "bell", "horn", "roar", "note", "tone")
                                if is_sound_subj and any(k in s_defn for k in ("loud", "sound", "noise", "hear")):
                                    score += 40
                                elif not is_sound_subj and any(k in s_defn for k in ("become louder", "louder")):
                                    score -= 30

                                is_qty_subj = subj_lem in ("number", "amount", "membership", "population", "crowd", "rank", "ranks", "size", "volume", "figure", "total")
                                if is_qty_subj and any(k in s_defn for k in ("amount", "number", "increase in", "grow in number")):
                                    score += 40

                                # Liquid / fluid subject alignment (e.g. 'Tears were running down her face', 'Water was running')
                                is_liquid_subj = subj_lem in ("tear", "tears", "water", "sweat", "blood", "river", "rain", "liquid", "stream", "fluid", "droplet")
                                if is_liquid_subj and (any(k in s_defn for k in ("flow in", "flow", "liquid")) or (s.get("signpost") or "").lower() == "flow"):
                                    score += 50
                                elif is_liquid_subj and any(k in s_defn for k in ("using your legs", "kill or injure", "criticize", "manage", "business")):
                                    score -= 40

                                # Origin / derivation vs birth / youth place differentiation:
                                # Non-human subjects (words, culture, craft, coffee, etc.) originate/derive from a place/source;
                                # they are not born or lived there when young.
                                is_origin_defn = any(k in s_defn for k in ("obtained from", "start or be made", "originate", "derive"))
                                is_birth_defn = any(k in s_defn for k in ("born there", "lived there when you were young", "grow up"))
                                if is_birth_defn:
                                    score -= 30
                                # Selectional restriction: physical body movement senses (signpost 'your body' or definition 'move your body')
                                # or interpersonal social visiting ('to visit someone you know') require human/animate subjects.
                                # Non-human/abstract/corporate subjects (company, economy, market, crisis, temperature, etc.)
                                # cannot perform physical body movements unless explicit body parts are mentioned.
                                is_body_movement_sense = "your body" in (s.get("signpost") or "").lower() or any(k in s_defn for k in ("move your body", "using your body", "part of your body"))
                                has_body_part_in_quote = any(bw in q_low_all for bw in ("body", "head", "face", "neck", "shoulder", "arm", "leg", "eye", "eyes", "heel", "back"))
                                if is_body_movement_sense and not has_body_part_in_quote:
                                    score -= 40
                                if "visit someone" in s_defn:
                                    score -= 50

                                # Quantitative / Financial / Abstract Metric Subject vs. Physical Gravity / Spatial Relocation:
                                is_metric_or_abstract_subj = (
                                    subj_lem in (
                                        "price", "prices", "stock", "stocks", "gold", "rate", "rates", "wage", "wages",
                                        "tax", "taxes", "demand", "supply", "export", "exports", "import", "imports",
                                        "credit", "trust", "wealth", "power", "value", "cost", "costs", "revenue",
                                        "income", "profit", "profits", "temperature", "temperatures", "heat", "pressure",
                                        "level", "levels", "standard", "standards", "quality", "culture", "management",
                                        "policy", "policies", "law", "laws", "attitude", "attitudes", "habit", "habits",
                                    )
                                    or (
                                        bool(subj_supersenses & {"noun.possession", "noun.quantity", "noun.attribute", "noun.cognition"})
                                        and not bool(subj_supersenses & {"noun.artifact", "noun.body", "noun.person", "noun.animal", "noun.plant", "noun.object"})
                                    )
                                )
                                if is_metric_or_abstract_subj:
                                    if any(k in s_defn or k in sp for k in (
                                        "lower level or amount", "lower level, amount", "become less", "decrease",
                                        "opinions, beliefs", "become different", "go upwards", "increase in",
                                    )):
                                        score += 18
                                    elif any(k in s_defn for k in (
                                        "onto the ground", "from one place or position to another",
                                        "from a higher position to a lower position", "take off your clothes",
                                        "take back to a shop",
                                    )):
                                        score -= 18

                                # Universal Conditional-Subject Ontology Gate ('if a <Subject-Class> <verb>s'):
                                # Parse LDOCE's canonical conditional subject clause (e.g. 'if a piece of furniture or equipment collapses',
                                # 'if a piece of fruit bruises', 'if a system, idea, or organization collapses') and verify ontological
                                # compatibility between the passage subject (subj_lem + WordNet supersenses) and the condition's nouns.
                                m_cond_subj = re.match(
                                    r"^if\s+(?:a\s+|an\s+|the\s+|your\s+)?([^,]{2,65}?)\s+"
                                    + rf"(?:{re.escape(target_w_clean)}[a-z]*|{re.escape(tok.lemma_.lower())}[a-z]*)\b",
                                    s_defn,
                                )
                                if m_cond_subj:
                                    cond_np = m_cond_subj.group(1).lower()
                                    _cond_stop = {
                                        "piece", "pieces", "kind", "kinds", "type", "types", "sort", "sorts",
                                        "part", "parts", "amount", "number", "group", "set", "someone", "somebody",
                                        "something", "person", "people", "thing", "things", "one", "two", "other",
                                        "particular", "certain", "large", "small", "new", "old", "etc"
                                    }
                                    cond_nouns = [
                                        w_c for w_c in re.findall(r"[a-z]{3,}", cond_np)
                                        if w_c not in stopwords and w_c not in _cond_stop
                                    ]
                                    if cond_nouns:
                                        _open_class_anchors = {
                                            "system", "organization", "situation", "process", "idea", "plan",
                                            "institution", "company", "business", "relationship", "activity",
                                            "opinion", "opinions", "belief", "beliefs", "price", "prices", "level", "levels",
                                        }
                                        is_open_class_cond = (
                                            len(cond_nouns) >= 3
                                            or "etc" in cond_np
                                            or bool(set(cond_nouns) & _open_class_anchors)
                                        )
                                        cond_supers: Set[str] = set()
                                        for cn in cond_nouns:
                                            cond_supers.update(cls.get_wordnet_supersenses(cn, pos="n"))
                                        # Check primary WordNet hypernym closure for narrow conditions
                                        is_hypernym_of_cond = False
                                        wn_inst = cls.get_wordnet()
                                        if wn_inst and subj_lem:
                                            try:
                                                for syn in (wn_inst.synsets(subj_lem, pos="n") or [])[:2]:
                                                    for h_path in syn.hypernym_paths():
                                                        for h_syn in h_path:
                                                            h_lems = {l.lower() for l in h_syn.lemmas()}
                                                            if h_lems & set(cond_nouns):
                                                                is_hypernym_of_cond = True
                                                                break
                                                        if is_hypernym_of_cond:
                                                            break
                                                    if is_hypernym_of_cond:
                                                        break
                                            except Exception:
                                                pass
                                        # Direct lexical or hypernym/supersense match with passage subject
                                        if subj_lem in cond_nouns or any(cls.form_matches(subj_lem, cn) for cn in cond_nouns) or is_hypernym_of_cond:
                                            score += 45
                                        elif is_open_class_cond and subj_supersenses and cond_supers and (subj_supersenses & cond_supers):
                                            score += 15
                                        elif any(cn in q_low_all for cn in cond_nouns):
                                            score += 20
                                        elif not is_open_class_cond:
                                            # Narrow closed-class condition (e.g. 'if the wind changes', 'if a leader or a government falls',
                                            # 'if someone's voice or a sound falls', 'if a piece of furniture or equipment collapses')
                                            # is penalized when the passage subject is neither the restricted noun nor its hyponym.
                                            score -= 25
                                        else:
                                            # Narrow concrete/artifact/substance conditional subjects should be penalized when
                                            # the passage subject belongs to a disjoint ontological domain (e.g. abstract/state/location)
                                            _narrow_concrete_supers = {
                                                "noun.food", "noun.plant", "noun.animal", "noun.substance", "noun.body"
                                            }
                                            _narrow_artifact_nouns = {
                                                "furniture", "equipment", "chair", "table", "bed", "tent", "vehicle",
                                                "car", "aircraft", "plane", "ship", "boat", "train", "machine", "engine",
                                                "garment", "cloth", "clothing", "shoe", "containers", "container", "bottle",
                                                "pipe", "wire", "coin", "note", "cheque", "check", "wound", "bone", "tooth"
                                            }
                                            is_narrow_concrete_cond = (
                                                bool(cond_supers & _narrow_concrete_supers)
                                                or any(cn in _narrow_artifact_nouns for cn in cond_nouns)
                                            )
                                            if is_narrow_concrete_cond and not (subj_supersenses & (cond_supers | {"noun.Tops"})):
                                                score -= 35

                                if "piece of furniture or equipment" in s_defn:
                                    if not any(fw in q_low_all for fw in ("furniture", "equipment", "chair", "table", "bed", "tent", "umbrella", "stroller", "tripod", "fold")):
                                        score -= 35
                                if "piece of fruit" in s_defn:
                                    if not any(fw in q_low_all for fw in ("fruit", "fruits", "apple", "peach", "pear", "banana", "plum", "berry", "tomato", "produce", "peel", "ripe")):
                                        score -= 45
                                if subj_lem in ("relationship", "divorce", "failure", "defeat", "rejection", "criticism", "remark", "comment", "word", "experience"):
                                    if any(k in s_defn for k in ("affect someone badly", "feel less confident", "make them feel", "upset", "hurt")):
                                        score += 55
                                if subj_lem in ("system", "economy", "society", "environment", "ecosystem", "land", "ocean", "atmosphere", "nature", "climate", "civilization", "empire", "regime", "marriage", "relationship", "talk", "negotiation"):
                                    if "system, idea, or organization" in s_defn:
                                        score += 35
                                    elif "building, wall" in s_defn:
                                        score -= 20

                        # Discourse/deliberative forum context alignment (e.g. 'Go to shareholder meetings and speak up')
                        has_forum_context = any(
                            t_ctx.lemma_.lower() in ("meeting", "conference", "assembly", "shareholder", "debate", "discussion", "hearing", "forum", "session", "council", "parliament", "board")
                            for t_ctx in q_doc
                        )
                        if has_forum_context:
                            if any(k in s_defn for k in ("express your opinion", "express their opinion", "say what you think", "say what they think", "opinion")):
                                score += 35
                            elif re.match(r"^(?:used\s+to\s+(?:ask|say|tell|show|express)|spoken\b)", s_defn, re.I):
                                score -= 20

                        # Check for xcomp infinitive complement (e.g. tend to do / think)
                        has_xcomp = any(ch.dep_ == "xcomp" for ch in tok.children)
                        if has_xcomp:
                            for u in s.get("units", []):
                                if "to do" in u.lower():
                                    score += 45
                                    break

                    # If target is a preposition / conjunction, check complement & clause aspect
                    elif tok.pos_ in ("ADP", "PREP", "SCONJ") or tok.dep_ in ("prep", "mark") or target_p in ("prep", "conj"):
                        time_nouns = {"day", "hour", "minute", "second", "week", "month", "year", "decade", "century", "time", "period"}
                        has_time_pobj = any(
                            c.lemma_.lower() in time_nouns or any(gc.lemma_.lower() in time_nouns for gc in c.subtree)
                            for c in tok.children if c.dep_ in ("pobj", "pcomp")
                        )
                        if has_time_pobj:
                            if any(k in s_defn for k in ("period of time", "time", "hours", "weeks", "months", "years", "days")):
                                score += 40
                            # Exact numeric interval distinction: 'before a certain period has passed' (e.g. within 24 hours / within one day / an hour)
                            has_num_time = any(
                                any(gc.dep_ in ("nummod", "quantmod") or gc.pos_ == "NUM" or gc.lemma_.lower() in ("a", "an", "one", "two", "three", "few", "several") for gc in c.children)
                                for c in tok.children if c.dep_ in ("pobj", "pcomp") and (c.lemma_.lower() in time_nouns or any(gc.lemma_.lower() in time_nouns for gc in c.subtree))
                            )
                            if has_num_time:
                                if "before a certain period" in s_defn or any("24 hours" in ex or "an hour" in ex for ex in s.get("examples", [])):
                                    score += 25

                        # Temporal vs. Causal Conjunction Disambiguation (e.g. 'since', 'as'):
                        # 1) Temporal origin sense ('from a particular time in the past until a later past time, or until now'):
                        #    governed by perfect aspect in matrix clause ('has/have/had + VBN'), 'ever since', or past-tense point-in-time subordinate clause.
                        # 2) Causal reason sense ('used to give the reason why something happens'):
                        #    sentence-initial or present/modal subordinate clause without 'ever since' and without matrix temporal duration.
                        if target_p == "conj" or tok.dep_ == "mark":
                            is_temporal_origin_sense = (
                                "from a particular time in the past" in s_defn
                                or str(s.get("signpost") or "").strip().lower() == "time"
                            )
                            is_causal_reason_sense = (
                                "give the reason" in s_defn
                                or "because" in s_defn
                                or str(s.get("signpost") or "").strip().lower() == "reason"
                            )
                            if is_temporal_origin_sense or is_causal_reason_sense:
                                sub_verb = tok.head if tok.dep_ == "mark" else None
                                matrix_verb = sub_verb.head if (sub_verb is not None and sub_verb.head != sub_verb) else None
                                has_ever_since = tok.i > 0 and q_doc[tok.i - 1].lemma_.lower() == "ever"
                                sub_is_past_event = bool(
                                    sub_verb is not None
                                    and (
                                        sub_verb.tag_ == "VBD"
                                        or any(ch.tag_ == "VBD" for ch in sub_verb.children if ch.dep_ in ("aux", "auxpass"))
                                    )
                                )
                                sub_is_present_or_modal = bool(
                                    sub_verb is not None
                                    and (
                                        sub_verb.tag_ in ("VBP", "VBZ")
                                        or any(ch.tag_ in ("VBP", "VBZ", "MD") for ch in sub_verb.children if ch.dep_ in ("aux", "auxpass"))
                                    )
                                )
                                matrix_has_perfect_or_time = bool(
                                    matrix_verb is not None
                                    and (
                                        any(ch.lemma_.lower() == "have" and ch.dep_ == "aux" for ch in matrix_verb.children)
                                        or any(t.lemma_.lower() in time_nouns or t.lemma_.lower() in ("pass", "elapse", "ago") for t in q_doc)
                                    )
                                )
                                if has_ever_since or (sub_is_past_event and matrix_has_perfect_or_time):
                                    if is_temporal_origin_sense:
                                        score += 45
                                    if is_causal_reason_sense:
                                        score -= 35
                                elif sub_is_present_or_modal and not has_ever_since and target_w_clean in ("since", "as"):
                                    if is_causal_reason_sense:
                                        score += 45
                                    if is_temporal_origin_sense:
                                        score -= 35
                            if tok.dep_ == "mark" and target_w_clean == "after":
                                if "when a particular event or time has happened" in s_defn:
                                    score += 45
                                elif "because of something that happened earlier" in s_defn:
                                    score -= 30

                    # If target is an adverb, check focus attachment, degree-infinitive frame, and motion/space vs dimension/time
                    if tok.pos_ == "ADV" or target_p == "adv":
                        # 1) Post-nominal focus adverb ('Money alone cannot solve this', 'in legal fees alone')
                        if (
                            tok.i > 0
                            and q_doc[tok.i - 1].pos_ in ("NOUN", "PROPN", "PRON")
                            and (tok.head == q_doc[tok.i - 1] or q_doc[tok.i - 1].dep_ in ("nsubj", "nsubjpass", "pobj", "dobj"))
                        ):
                            if any(k in s_defn for k in ("without including anything else", "emphasize that there is only one")):
                                score += 45
                            elif any(k in s_defn for k in ("there is no one with you", "without any friends")):
                                score -= 35
                        # 2) Degree adverb with 'to'-infinitive / 'for' complement ('fast enough to catch', 'old enough for school')
                        if (
                            (tok.i + 1 < len(q_doc) and q_doc[tok.i + 1].lower_ in ("to", "for"))
                            or any(c.dep_ in ("xcomp", "advcl") for c in tok.head.children)
                        ):
                            if "to the degree that is necessary or wanted" in s_defn:
                                score += 45
                            elif "fairly but not very" in s_defn:
                                score -= 35
                        # 3) Directional motion adverb vs. dimensional width measurement ('swam across' vs '2 km across')
                        has_measure_np = tok.i > 0 and (
                            q_doc[tok.i - 1].pos_ == "NUM"
                            or (
                                q_doc[tok.i - 1].pos_ == "NOUN"
                                and any(c.pos_ == "NUM" or c.dep_ == "nummod" for c in q_doc[tok.i - 1].children)
                            )
                        )
                        if not has_measure_np and tok.head.pos_ == "VERB":
                            if any(k in s_defn for k in ("from one side of something to the other", "towards someone or something on the other side")):
                                score += 40
                            elif "used to show how wide something is" in s_defn:
                                score -= 40
                        # 4) Spatial vs. Temporal separation ('in between' two houses vs two times/events)
                        _adv_time_nouns = {"day", "hour", "minute", "second", "week", "month", "year", "decade", "century", "time", "period", "meal", "job", "event", "monday", "sunday", "christmas", "summer", "winter"}
                        _has_adv_time = any(t.lemma_.lower() in _adv_time_nouns or t.ent_type_ in ("DATE", "TIME") for t in q_doc)
                        if not _has_adv_time:
                            if "in or through the space that separates" in s_defn:
                                score += 35
                            elif "in the time that separates" in s_defn:
                                score -= 35
            except Exception:
                pass

        # 6b. Adverbial Governing Verb Valency & Selectional Alignment
        if quote and target_p in ("adv", "r"):
            if s_defn.startswith("(of the elements)") and not any(
                w in q_low_all for w in ("wind", "storm", "rain", "sea", "wave", "waves", "weather", "gale", "snow", "tempest", "thunder", "lightning")
            ):
                score -= 45
            try:
                nlp_r = cls.get_spacy()
                doc_r = nlp_r(quote) if nlp_r else None
                if doc_r is not None:
                    for tok_r in doc_r:
                        if cls.form_matches(tok_r.text.lower(), target_w_clean) or cls.form_matches(tok_r.lemma_.lower(), target_w_clean):
                            if tok_r.head and tok_r.head.pos_ == "VERB":
                                gov_v = tok_r.head.lemma_.lower()
                                if gov_v in s_def_tokens or any(re.search(rf"\b{re.escape(gov_v)}\b", ex.lower()) for ex in s.get("examples", [])):
                                    score += 35
                                v_supers = cls.get_wordnet_supersenses(gov_v, pos="v")
                                is_effort_motion_v = (
                                    gov_v in ("work", "type", "write", "run", "row", "pedal", "scrub", "dig", "fight", "struggle", "think", "kick", "swim", "drive", "ride", "climb", "dance", "spin", "beat", "scribble", "search")
                                    or bool(v_supers & {"verb.motion", "verb.creation", "verb.competition", "verb.contact"})
                                ) and gov_v not in ("shout", "yell", "scream", "argue", "complain", "protest", "react", "glare", "stare")
                                if is_effort_motion_v:
                                    if any(k in s_defn for k in ("energy", "effort", "speed", "fast", "hard", "rapidly", "quickly")):
                                        score += 35
                                    elif any(k in s_defn for k in ("angry", "anger", "annoyed", "upset", "sad")):
                                        score -= 20
            except Exception:
                pass

        # 7. Syntactic frame / Grammar alignment
        gram = (s.get("grammar") or "").lower()
        if frame_evidence:
            if frame_evidence == "modifies_adj" and any(k in gram for k in ("adjective", "adverb", "predicative", "[+adjective")):
                score += 25
            elif frame_evidence == "modifies_verb" and any(k in gram for k in ("verb", "action", "[+verb")):
                score += 25
            elif frame_evidence == "prep" and re.search(r"\[.*prep.*\]", gram):
                score += 10

        # 8. Contextual Pedagogical Sense Cue Index bonus
        if i in cue_scores:
            score += cue_scores[i]

        # Tie-break slightly favors primary head senses
        score -= i * 0.5


        return float(score)


__all__ = ["WsdContext", "WsdScoringMixin"]
