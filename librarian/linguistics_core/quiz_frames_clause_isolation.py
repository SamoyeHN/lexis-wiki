"""
Collocation half-slicing and SLA discourse clause isolation gates.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


class QuizFramesClauseIsolationMixin:
    @classmethod
    def collocation_half_slice(
        cls,
        phrase: str,
        sentence: str,
        pos: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        SLA Lexical Approach: Collocation-Slot Half-Slice Cloze (搭配槽位半切片完形).
        Decomposes multi-word collocations, phrasal verbs, and prepositional frames into:
          - A tested single-word slot (adjective collocator, light verb, phrasal particle, noun slot)
          - Preserved collocational anchor in the sentence stem (e.g. '____ praise', 'is ____ with')
          - Pure single-word distractors targeting learner L1 negative transfer & collocational clash.
        Guarantees zero whole-chunk masking, zero length disparity, and maximum psychometric discrimination.
        """
        if not phrase or not sentence:
            return None

        clean_p = re.sub(r"[\(\[][^\)\]]+[\)\]]", "", phrase).strip()
        p_words = [w.lower() for w in clean_p.split() if w]
        if len(p_words) < 2:
            return None

        nlp = cls.get_spacy()
        doc = nlp(sentence)
        s_lower = sentence.lower()

        _DEGREE_ADJ_DISTRACTORS = {
            "high": ["tall", "deep", "heavy"],
            "heavy": ["strong", "thick", "deep"],
            "deep": ["heavy", "high", "thick"],
            "strong": ["heavy", "deep", "powerful"],
            "sharp": ["steep", "high", "bitter"],
            "broad": ["wide", "large", "thick"],
            "profound": ["intense", "severe", "deep"],
            "great": ["tall", "high", "heavy"],
        }
        _LIGHT_VERB_DISTRACTORS = {
            "make": ["take", "give", "do"],
            "made": ["taken", "given", "done"],
            "makes": ["takes", "gives", "does"],
            "making": ["taking", "giving", "doing"],
            "take": ["make", "play", "have"],
            "took": ["made", "played", "had"],
            "takes": ["makes", "plays", "has"],
            "taking": ["making", "playing", "having"],
            "give": ["make", "take", "do"],
            "gave": ["made", "took", "did"],
            "gives": ["makes", "takes", "does"],
            "giving": ["making", "taking", "doing"],
            "pay": ["give", "make", "lend"],
            "paid": ["given", "made", "lent"],
            "cast": ["throw", "shed", "put"],
            "shed": ["cast", "throw", "shine"],
            "strike": ["keep", "make", "hold"],
            "struck": ["kept", "made", "held"],
        }
        _PREP_NOUN_DISTRACTORS = {
            "ease": ["rest", "peace", "comfort"],
            "advance": ["progress", "forward", "ahead"],
            "purpose": ["reason", "intent", "aim"],
            "risk": ["danger", "threat", "hazard"],
            "hand": ["reach", "touch", "arm"],
            "vain": ["waste", "loss", "void"],
            "fact": ["truth", "reality", "deed"],
            "peace": ["ease", "rest", "calm"],
            "turn": ["time", "round", "shift"],
            "accident": ["mishap", "misfortune", "casualty"],
        }
        _PARTICIPLE_DISTRACTORS = {
            "filled": ["loaded", "crowded", "packed"],
            "based": ["founded", "grounded", "modeled"],
            "equipped": ["furnished", "fitted", "armed"],
            "associated": ["connected", "linked", "related"],
            "proud": ["fond", "aware", "jealous"],
            "short": ["free", "devoid", "clear"],
            "full": ["short", "empty", "bare"],
        }

        # 1. Copular / Passive + Participle/Adjective + Preposition (e.g. 'be filled with', 'be new to', 'be proud of')
        m_cop = re.search(
            r"(?:\b(?:is|was|are|were|be|been|being)|'(?:re|s|m))\s+([a-zA-Z'-]+)\s+([a-zA-Z]+)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_cop and len(p_words) >= 2:
            cand_adj = m_cop.group(1).lower()
            cand_prep = m_cop.group(2).lower()
            adj_ent = cls.get_ldoce_entry(cand_adj)
            all_p = [p.lower() for p in ((adj_ent or {}).get("all_poses", []) or [])]
            is_valid_adj_participle = (
                cand_adj in _PARTICIPLE_DISTRACTORS
                or (adj_ent and ("adjective" in (adj_ent.get("pos") or "").lower() or "adjective" in all_p))
                or bool(cls._wordnet_pos(cand_adj) & {"a", "s"})
                or (cand_adj.endswith(("ed", "ing", "en")) and len(cand_adj) > 4)
            ) and cand_adj not in ("worry", "care", "think", "talk", "hear", "know", "speak", "tell", "ask", "look")
            if is_valid_adj_participle and cand_adj in p_words and (cand_prep in p_words or cand_prep in ("with", "of", "on", "in", "by", "for", "about", "to")):
                v_tok_str = m_cop.group(1)
                v_start = s_lower.find(v_tok_str.lower(), m_cop.start())
                v_end = v_start + len(v_tok_str)
                stem = sentence[:v_start] + "____" + sentence[v_end:]
                target_tok = sentence[v_start:v_end]
                dists = _PARTICIPLE_DISTRACTORS.get(v_tok_str.lower())
                if not dists:
                    dists = cls.generate_zero_collision_distractors(target_tok, pos="a", count=6, context_sentence=sentence)
                    dists = [d for d in dists if d.lower() != target_tok.lower()][:3]
                if len(dists) < 3:
                    dists = [d for d in ["old", "experienced", "early"] if d != target_tok.lower()][:3]
                return {
                    "target_slot": target_tok,
                    "base_headword": phrase,
                    "stem": stem,
                    "slot_type": "participle_collocator",
                    "cloze_slot_frame": f"... is/was/are ____ {m_cop.group(2)}",
                    "distractors": dists,
                    "explanation": f"In English, \"{clean_p}\" is the fixed collocation. \"{dists[0]}\", \"{dists[1]}\", or \"{dists[2]}\" cannot collocate with \"{m_cop.group(2)}\" in this context.",
                }

        # 2. Adjective + Noun Collocation (e.g. 'high praise', 'heavy rain', 'deep trouble')
        m_adj = re.search(
            r"\b(high|tall|deep|heavy|strong|big|sharp|broad|wide|steep|rich|profound|great)\s+([a-z]+)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_adj and (p_words[0] in _DEGREE_ADJ_DISTRACTORS or p_words[-1] == m_adj.group(2).lower()):
            adj_tok_str = m_adj.group(1)
            adj_start = s_lower.find(adj_tok_str.lower(), m_adj.start())
            adj_end = adj_start + len(adj_tok_str)
            stem = sentence[:adj_start] + "____" + sentence[adj_end:]
            target_tok = sentence[adj_start:adj_end]
            dists = _DEGREE_ADJ_DISTRACTORS.get(adj_tok_str.lower(), ["tall", "deep", "heavy"])
            noun_str = m_adj.group(2)
            return {
                "target_slot": target_tok,
                "base_headword": phrase,
                "stem": stem,
                "slot_type": "collocator_adj",
                "cloze_slot_frame": f"... ____ {noun_str}",
                "distractors": dists,
                "explanation": f"In English, \"{target_tok.lower()} {noun_str}\" is the standard idiomatic collocation. Adjectives like \"{dists[0]}\", \"{dists[1]}\", or \"{dists[2]}\" cannot collocate with \"{noun_str}\".",
            }

        # 3. Prepositional Phrase / Set Phrase (e.g. 'at ease', 'in advance', 'on purpose', 'by accident')
        _PREP_SET = {"at", "in", "on", "by", "for", "under", "with"}
        if p_words[0] in _PREP_SET and len(p_words) == 2:
            p0, p1 = p_words[0], p_words[1]
            m_pp = re.search(rf"\b{re.escape(p0)}\s+{re.escape(p1)}\b", sentence, re.IGNORECASE)
            if m_pp:
                noun_start = s_lower.find(p1, m_pp.start())
                noun_end = noun_start + len(p1)
                stem = sentence[:noun_start] + "____" + sentence[noun_end:]
                target_tok = sentence[noun_start:noun_end]
                dists = list(_PREP_NOUN_DISTRACTORS.get(p1, []))
                if len(dists) < 3:
                    # Dynamically harvest Oxford near-synonyms with PedLex zero-collocation check
                    ox_entries = cls.get_oxford_thesaurus_entries(p1)
                    ox_syns = []
                    phrase_index = cls._get_phrase_index()
                    for ent in ox_entries:
                        if ent.get("pos") in ("noun", ""):
                            for s in ent.get("senses", []):
                                ox_syns.extend(s.get("core_synonyms", []))
                                ox_syns.extend(s.get("synonyms", []))
                    for syn in ox_syns:
                        s_low = syn.lower().strip()
                        if (
                            s_low != p1
                            and len(s_low.split()) == 1
                            and s_low.isalpha()
                            and s_low not in dists
                            and s_low not in ("chance", "mistake")
                            and f"{p0} {s_low}" not in phrase_index
                            and not cls.double_key_collision(p1, s_low, pos="n")[0]
                        ):
                            dists.append(s_low)
                            if len(dists) >= 3:
                                break
                if len(dists) < 3:
                    dists = _PREP_NOUN_DISTRACTORS.get(p1, ["rest", "peace", "comfort"])
                return {
                    "target_slot": target_tok,
                    "base_headword": phrase,
                    "stem": stem,
                    "slot_type": "fixed_noun_slot",
                    "cloze_slot_frame": f"... {p0} ____",
                    "distractors": dists[:3],
                    "explanation": f"\"{clean_p}\" is a fixed prepositional set phrase. Nouns like \"{dists[0]}\", \"{dists[1]}\", or \"{dists[2]}\" do not form this standard idiom.",
                }

        # 4. Nominal Genitive Idioms (e.g. 'peace of mind', 'state of mind', 'matter of course', 'line of duty')
        m_ng = re.search(
            r"\b([a-zA-Z]+)\s+of\s+([a-zA-Z]+)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_ng and (" of " in clean_p.lower()):
            ng_n1 = m_ng.group(1).lower()
            ng_n2 = m_ng.group(2).lower()
            if ng_n1 in p_words and ng_n2 in p_words:
                n1_start = s_lower.find(ng_n1, m_ng.start())
                n1_end = n1_start + len(ng_n1)
                stem = sentence[:n1_start] + "____" + sentence[n1_end:]
                target_tok = sentence[n1_start:n1_end]
                phrase_index = cls._get_phrase_index()

                # Dynamic Oxford near-synonym extraction with PedLex zero-collocation filter
                ox_syns: List[str] = []
                for ent in cls.get_oxford_thesaurus_entries(ng_n1):
                    if ent.get("pos") in ("noun", ""):
                        for s in ent.get("senses", []):
                            ox_syns.extend(s.get("core_synonyms", []))
                            ox_syns.extend(s.get("synonyms", []))

                ng_dists: List[str] = []
                for syn in ox_syns:
                    s_low = syn.lower().strip()
                    if (
                        s_low != ng_n1
                        and len(s_low.split()) == 1
                        and s_low.isalpha()
                        and s_low not in ng_dists
                        and f"{s_low} of {ng_n2}" not in phrase_index
                        and not cls.is_attested_phrase(f"{s_low} of {ng_n2}")
                    ):
                        ng_dists.append(s_low)
                        if len(ng_dists) >= 3:
                            break

                if len(ng_dists) < 3:
                    fb_n = cls.generate_zero_collision_distractors(target_tok.lower(), pos="n", count=6, context_sentence=sentence)
                    for fn in fb_n:
                        if (
                            fn != ng_n1
                            and fn not in ng_dists
                            and f"{fn} of {ng_n2}" not in phrase_index
                        ):
                            ng_dists.append(fn)
                            if len(ng_dists) >= 3:
                                break

                if len(ng_dists) >= 3:
                    return {
                        "target_slot": target_tok,
                        "base_headword": phrase,
                        "stem": stem,
                        "slot_type": "nominal_genitive_noun",
                        "cloze_slot_frame": f"... ____ of {ng_n2}",
                        "distractors": ng_dists[:3],
                        "explanation": f"\"{clean_p}\" is a fixed idiomatic collocation. Nouns like \"{ng_dists[0]}\", \"{ng_dists[1]}\", or \"{ng_dists[2]}\" do not form this standard idiom with \"of {ng_n2}\".",
                    }

        # 5. Complex Prepositions (e.g. 'in terms of', 'on behalf of', 'with regard to', 'in light of', 'in view of')
        m_cp = re.search(
            r"\b(in|on|at|by|with|under)\s+([a-zA-Z]+)\s+(of|with|to|for|from|as)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_cp and len(p_words) >= 3 and p_words[0] in ("in", "on", "at", "by", "with", "under"):
            cp_p1 = m_cp.group(1).lower()
            cp_noun = m_cp.group(2).lower()
            cp_p2 = m_cp.group(3).lower()
            if cp_p1 == p_words[0] and cp_noun in p_words and cp_p2 == p_words[-1]:
                n_start = s_lower.find(cp_noun, m_cp.start())
                n_end = n_start + len(cp_noun)
                stem = sentence[:n_start] + "____" + sentence[n_end:]
                target_tok = sentence[n_start:n_end]
                phrase_index = cls._get_phrase_index()
                peer_medial = [
                    k.split()[1] for k in phrase_index
                    if len(k.split()) == 3
                    and k.split()[0] == cp_p1
                    and k.split()[2] == cp_p2
                    and k.split()[1] != cp_noun
                    and k.split()[1].isalpha()
                ]
                valid_cp_dists: List[str] = []
                for pm in peer_medial:
                    cand_full = f"{cp_p1} {pm} {cp_p2}"
                    if not cls.double_key_collision(clean_p.lower(), cand_full, pos="phrase")[0]:
                        if pm not in valid_cp_dists:
                            valid_cp_dists.append(pm)
                            if len(valid_cp_dists) >= 3:
                                break
                if len(valid_cp_dists) >= 3:
                    return {
                        "target_slot": target_tok,
                        "base_headword": phrase,
                        "stem": stem,
                        "slot_type": "complex_prep_noun",
                        "cloze_slot_frame": f"... {cp_p1} ____ {cp_p2}",
                        "distractors": valid_cp_dists[:3],
                        "explanation": f"\"{clean_p}\" is the fixed complex preposition. While \"{valid_cp_dists[0]}\", \"{valid_cp_dists[1]}\", or \"{valid_cp_dists[2]}\" appear in other fixed phrases, only \"{target_tok.lower()}\" fits the meaning required here.",
                    }

        # 6. Bare Verbal Idioms (e.g. 'give rise to', 'catch sight of', 'shed light on', 'cast doubt on', 'pay tribute to')
        m_bvi = re.search(
            r"\b(give|gave|gives|giving|catch|caught|catches|catching|shed|cast|lose|lost|loses|losing|pay|paid|pays|paying|take|took|takes|taking|set|keep|kept|keeps|keeping)\s+(rise|birth|way|place|sight|glimpse|light|doubt|track|attention|tribute|heed|touch|ear|vent|voice)\s+(to|of|on|in|with)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_bvi and len(p_words) >= 3:
            v_tok = m_bvi.group(1).lower()
            bare_n = m_bvi.group(2).lower()
            p_prep = m_bvi.group(3).lower()
            if bare_n in p_words and p_prep == p_words[-1]:
                bn_start = s_lower.find(bare_n, m_bvi.start())
                bn_end = bn_start + len(bare_n)
                stem = sentence[:bn_start] + "____" + sentence[bn_end:]
                target_tok = sentence[bn_start:bn_end]

                doc_v = nlp(v_tok)
                v_lemma = doc_v[0].lemma_.lower() if doc_v else v_tok
                phrase_index = cls._get_phrase_index()
                peer_bare = [
                    k.split()[1] for k in phrase_index
                    if len(k.split()) == 3
                    and k.split()[0] == v_lemma
                    and k.split()[2] == p_prep
                    and k.split()[1] != bare_n
                    and k.split()[1].isalpha()
                    and k.split()[1] not in ("in", "on", "up", "out", "off", "away", "over", "down", "it", "so", "as")
                ]
                curated_bare_map = {
                    "rise": ["birth", "way", "place"],
                    "sight": ["glimpse", "wind", "track"],
                    "light": ["doubt", "shadow", "gloom"],
                    "doubt": ["light", "blame", "aspersions"],
                    "tribute": ["attention", "heed", "respect"],
                    "track": ["sight", "count", "hold"],
                    "touch": ["contact", "reach", "track"],
                }
                bvi_dists = list(curated_bare_map.get(bare_n, []))
                for pb in peer_bare:
                    if pb not in bvi_dists and pb != bare_n:
                        bvi_dists.append(pb)
                        if len(bvi_dists) >= 3:
                            break
                if len(bvi_dists) >= 3:
                    return {
                        "target_slot": target_tok,
                        "base_headword": phrase,
                        "stem": stem,
                        "slot_type": "bare_verbal_idiom_noun",
                        "cloze_slot_frame": f"... {v_tok} ____ {p_prep}",
                        "distractors": bvi_dists[:3],
                        "explanation": f"\"{clean_p}\" is a fixed verbal idiom. Nouns like \"{bvi_dists[0]}\", \"{bvi_dists[1]}\", or \"{bvi_dists[2]}\" form different idioms that do not fit the meaning here.",
                    }

        # 6b. Tripartite Phrasal-Prepositional Verbs (e.g. 'keep in touch with', 'stay in touch with')
        m_tpp = re.search(
            r"\b(keep|kept|keeps|keeping|stay|stayed|stays|staying|get|got|gets|getting|be|was|were|been|being)\s+in\s+touch\s+with\b",
            sentence,
            re.IGNORECASE,
        )
        if m_tpp and "in touch with" in clean_p.lower():
            v_tok = m_tpp.group(1)
            t_start = s_lower.find("touch", m_tpp.start())
            t_end = t_start + len("touch")
            stem = sentence[:t_start] + "____" + sentence[t_end:]
            target_tok = sentence[t_start:t_end]
            tpp_dists = ["contact", "reach", "communication"]
            return {
                "target_slot": target_tok,
                "base_headword": phrase,
                "stem": stem,
                "slot_type": "phrasal_prep_verb_noun",
                "cloze_slot_frame": f"... {v_tok.lower()} in ____ with",
                "distractors": tpp_dists,
                "explanation": f"\"{clean_p}\" is a fixed phrasal-prepositional idiom. Near-synonyms like \"{tpp_dists[0]}\", \"{tpp_dists[1]}\", or \"{tpp_dists[2]}\" do not complete this fixed construction.",
            }

        # Determine any trailing preposition from the phrase
        trailing_prep = ""
        for p_tok in reversed(p_words):
            if p_tok.lower() in ("to", "on", "in", "of", "with", "for", "between", "against", "into"):
                trailing_prep = f" {p_tok.lower()}"
                break

        # 7. Light-Verb Collocations (e.g. 'make a commitment to', 'commitment made to')
        m_lv_act = re.search(
            r"\b(make|made|makes|making|take|took|takes|taking|give|gave|gives|giving|do|did|does|doing|pay|paid|cast|strike|struck)\s+(?:a|an|the)?\s+(?:[a-zA-Z]+\s+)?(commitment|promise|decision|effort|contribution|tribute|balance|doubt)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_lv_act and (m_lv_act.group(1).lower() in p_words or m_lv_act.group(2).lower() in p_words):
            v_tok_str = m_lv_act.group(1)
            v_start = s_lower.find(v_tok_str.lower(), m_lv_act.start())
            v_end = v_start + len(v_tok_str)
            stem = sentence[:v_start] + "____" + sentence[v_end:]
            target_tok = sentence[v_start:v_end]
            dists = _LIGHT_VERB_DISTRACTORS.get(v_tok_str.lower(), ["take", "give", "do"])
            noun_word = m_lv_act.group(2)
            det_str = "a/the " if noun_word.lower() in ("commitment", "promise", "decision", "effort", "contribution", "balance") else ""
            frame_str = f"... ____ {det_str}{noun_word}{trailing_prep}".strip()
            return {
                "target_slot": target_tok,
                "base_headword": phrase,
                "stem": stem,
                "slot_type": "light_verb",
                "cloze_slot_frame": frame_str,
                "distractors": dists,
                "explanation": f"In English, \"{v_tok_str.lower()}\" collocates with \"{noun_word}\". Verbs like \"{dists[0]}\", \"{dists[1]}\", or \"{dists[2]}\" are unidiomatic in this collocation.",
            }

        m_lv_pass = re.search(
            r"\b(commitment|promise|decision|effort|contribution|tribute|balance|doubt|praise)\s+(made|taken|given|done|paid|cast|struck)\s+(to|for|with|on)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_lv_pass and (m_lv_pass.group(2).lower() in p_words or m_lv_pass.group(1).lower() in p_words):
            v_tok_str = m_lv_pass.group(2)
            v_start = s_lower.find(v_tok_str.lower(), m_lv_pass.start())
            v_end = v_start + len(v_tok_str)
            stem = sentence[:v_start] + "____" + sentence[v_end:]
            target_tok = sentence[v_start:v_end]
            dists = _LIGHT_VERB_DISTRACTORS.get(v_tok_str.lower(), ["taken", "given", "done"])
            noun_word = m_lv_pass.group(1)
            p_prep = f" {m_lv_pass.group(3)}" if m_lv_pass.group(3) else trailing_prep
            det_str = "a/the " if noun_word.lower() in ("commitment", "promise", "decision", "effort", "contribution", "balance") else ""
            # Canonical collocation order puts verb before noun (e.g. '... ____ a/the commitment to')
            frame_str = f"... ____ {det_str}{noun_word}{p_prep}".strip()
            return {
                "target_slot": target_tok,
                "base_headword": phrase,
                "stem": stem,
                "slot_type": "light_verb",
                "cloze_slot_frame": frame_str,
                "distractors": dists,
                "explanation": f"In English, \"{v_tok_str.lower()}\" collocates with \"{noun_word}\". Verbs like \"{dists[0]}\", \"{dists[1]}\", or \"{dists[2]}\" cannot collocate with this noun.",
            }

        # 8. Two-Word Phrasal Verbs (e.g. 'shut down', 'get ahead', 'break up', 'call off')
        _PARTICLES_AND_PREPS = {
            "in", "off", "up", "down", "away", "back", "over", "into", "through",
            "on", "to", "for", "out", "with", "from", "around", "about",
            "by", "at", "under", "between", "against", "after", "before",
            "along", "across", "behind", "beyond", "aside", "apart", "ahead"
        }
        if len(p_words) == 2 and p_words[1] in _PARTICLES_AND_PREPS:
            v_base, part_word = p_words[0], p_words[1]
            m_pv = re.search(
                rf"\b([a-zA-Z]+)\s+({re.escape(part_word)})\b",
                sentence,
                re.IGNORECASE,
            )
            if m_pv:
                cand_v = m_pv.group(1).lower()
                doc_cand = nlp(cand_v)
                cand_lemma = doc_cand[0].lemma_.lower() if doc_cand else cand_v
                if cand_lemma == v_base or cand_v == v_base:
                    p_start = s_lower.find(part_word, m_pv.start())
                    p_end = p_start + len(part_word)
                    stem = sentence[:p_start] + "____" + sentence[p_end:]
                    target_tok = sentence[p_start:p_end]
                    _CONTRAST_PARTICLES = ["down", "up", "off", "away", "out", "in", "back", "over", "ahead", "behind"]
                    pv_dists = [p for p in _CONTRAST_PARTICLES if p != part_word][:3]
                    return {
                        "target_slot": target_tok,
                        "base_headword": phrase,
                        "stem": stem,
                        "slot_type": "phrasal_verb_particle",
                        "cloze_slot_frame": f"... {m_pv.group(1)} ____",
                        "distractors": pv_dists,
                        "explanation": f"In English, \"{m_pv.group(1).lower()} {part_word}\" is the correct phrasal verb here. Particles like \"{pv_dists[0]}\", \"{pv_dists[1]}\", or \"{pv_dists[2]}\" create different meanings or unidiomatic combinations.",
                    }

        # 9. Guarded 2-Word Compound Noun / Modifier Half-Slice
        if len(p_words) == 2:
            w0, w1 = p_words[0], p_words[1]
            m_2w = re.search(rf"\b({re.escape(w0)})\s+({re.escape(w1)})\b", sentence, re.IGNORECASE)
            if m_2w:
                phrase_index = cls._get_phrase_index()
                # Check if w1 is a head noun with attested peer modifiers in PedLex (e.g. 'fighter plane', 'cargo plane')
                attested_mod_peers = [
                    k.split()[0] for k in phrase_index
                    if len(k.split()) == 2
                    and k.split()[1] == w1
                    and k.split()[0] != w0
                    and k.split()[0].isalpha()
                    and k.split()[0] not in _PARTICLES_AND_PREPS
                ]
                if len(attested_mod_peers) >= 3:
                    w0_tok = m_2w.group(1)
                    w0_start = s_lower.find(w0_tok.lower(), m_2w.start())
                    w0_end = w0_start + len(w0_tok)
                    stem = sentence[:w0_start] + "____" + sentence[w0_end:]
                    target_tok = sentence[w0_start:w0_end]
                    return {
                        "target_slot": target_tok,
                        "base_headword": phrase,
                        "stem": stem,
                        "slot_type": "compound_noun_modifier",
                        "cloze_slot_frame": f"... ____ {m_2w.group(2)}",
                        "distractors": attested_mod_peers[:3],
                        "explanation": f"In English, \"{target_tok.lower()} {m_2w.group(2)}\" is the standard compound noun. Modifiers like \"{attested_mod_peers[0]}\", \"{attested_mod_peers[1]}\", or \"{attested_mod_peers[2]}\" form different terms.",
                    }
                # If no attested modifiers exist, DO NOT force an ungrammatical half-slice!
                # Returning None delegates whole-chunk assessment to generate_phrase_distractors,
                # ensuring high-quality symmetrical peer phrases (e.g. passenger plane -> passenger train, passenger ship).
                return None

        return None

    @classmethod
    def isolate_cloze_clause(cls, stem: str) -> str:
        """
        SLA Discourse Unit Isolation Gate:
        Splits compound/complex sentences across coordinate and subordinate clause boundaries
        to isolate the self-contained clause containing the blank '____'.
        Eliminates cross-item leakage (e.g. other vocabulary targets appearing elsewhere in the same sentence)
        and guarantees tight, natural single-sentence cloze test stems.
        """
        if not stem or "____" not in stem:
            return stem

        # Major coordinating & subordinating boundaries (lookbehind must be fixed-width in Python 3.14)
        clauses = re.split(
            r"(?<=,)\s+(?=(?:and|but|or|so|saying|standing|the\s+[A-Z]\w+|when|while|after|before|as|although|because|since|if)\b)|(?<=;)\s+",
            stem
        )
        if len(clauses) <= 1:
            return stem

        for cl in clauses:
            if "____" in cl:
                cl_clean = cl.strip().rstrip(",").strip()
                # Clean leading coordinating conjunctions
                cl_clean = re.sub(r"^(?:and|but|so|or)\s+", "", cl_clean, flags=re.IGNORECASE).strip()
                # A clause starting with '____' lacks an overt subject (coordinate predicate); do not isolate
                if cl_clean.startswith("____"):
                    return stem
                # Prepositional/adverbial fragment guard: introductory adjuncts must not sever the main clause
                first_word = cl_clean.split()[0].lower() if cl_clean.split() else ""
                if first_word in ("by", "in", "at", "on", "with", "without", "from", "to", "for", "through", "despite"):
                    return stem
                if len(cl_clean.split()) >= 4:
                    nlp = cls.get_spacy()
                    doc = nlp(cl_clean)
                    has_finite_verb = any(t.pos_ in ("VERB", "AUX") and t.tag_ in ("VBD", "VBZ", "VBP", "MD", "VB") for t in doc)
                    if not has_finite_verb:
                        return stem
                    if cl_clean and cl_clean[0].islower():
                        cl_clean = cl_clean[0].upper() + cl_clean[1:]
                    if not cl_clean.endswith((".", "!", "?")):
                        cl_clean += "."
                    return cl_clean

        return stem


