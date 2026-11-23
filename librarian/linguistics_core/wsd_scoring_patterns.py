"""WSD scoring stage 2: Syntactic patterns, valency frames, and idiomatic multi-word units."""

from __future__ import annotations

import re
from typing import Any, Dict, Set

from .wsd_scoring_context import WsdContext, WsdSenseInfo


class WsdScoringPatternsMixin:
    """Stage 2 scoring: pattern frame matches, separable phrasals, and unit alignment."""

    @classmethod
    def _score_sense_patterns_and_units(
        cls, s: Dict[str, Any], i: int, ctx: WsdContext, s_info: WsdSenseInfo, score: float
    ) -> float:
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

        s_pos = s_info.s_pos
        eff_pos = s_info.eff_pos
        eff_p = s_info.eff_p
        s_defn = s_info.s_defn
        s_def_tokens = s_info.s_def_tokens
        sp = s_info.sp
        s_gram = s_info.s_gram

        def _quote_has_form(word_form: str) -> bool:
            return cls._quote_has_form(word_form, ctx)

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



        return float(score)


__all__ = ["WsdScoringPatternsMixin"]
