"""Single-word vocabulary distractor generation and zero-collision orchestrator."""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics.vocab_distractors")


class DistractorVocabMixin:

    @classmethod
    def generate_vocab_distractors(
        cls,
        target_word: str,
        pos: str = "noun",
        context_anchor: str = None,
        anchor_type: str = "collocation",
        target_count: int = 3,
        exclude_words: Optional[Set[str]] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        return_metadata: bool = False
    ) -> Union[List[str], Tuple[List[str], Dict[str, str]]]:
        """
        Synthesizes high-discrimination, collision-free distractors for a target vocabulary word,
        grounded in the target's authentic curriculum definition (Sense-Specific Primacy).
        
        Pipeline:
        1. Definition-Locked WSD (Word Sense Disambiguation):
           Matches the target word's curriculum definition against WordNet synsets (Lesk overlap)
           to lock in the exact pedagogical sense, preventing out-of-domain troponym leakage.
        2. Sense-Locked Antonym Extraction (Polarity Discrimination):
           Extracts genuine antonyms from the locked synset/senses. Antonyms share valency and
           collocation environments but create logical/truth-value contradictions in context.
        3. WordNet Semantic Candidates: Retrieves direct synset synonyms (Tier 1 for nouns/adj),
           hyponyms/troponyms (Tier 2), and coordinate terms (Tier 3) under the matched synset.
        4. Lexicon Filter (CEFR / Oxford 20k): Restricts candidates to words present in the Oxford
           Collocations Dictionary, ensuring natural, curriculum-appropriate words and eliminating obscure terms.
        5. Oxford Collision Clearance (Zero Double-Key Guarantee):
           - All synset synonyms across WordNet are strictly banned as distractors for verbs.
           - Checks context_anchor collocations to guarantee absolute single-fit validity.
           - Sense-locked antonyms are whitelisted from anchor collision clearance since they are intentionally plausible.
        6. Syntactic / Morphological Parallelism: Restricts to single words matching the target's POS.
        """
        clean_target = target_word.strip().lower()

        # Closed Grammatical Paradigms Gate:
        # If target word is an indexed function word (connective, clausal conjunction, preposition),
        # retrieve psychometrically calibrated distractors directly from closed paradigms.
        if cls.is_function_word(clean_target):
            f_dists, f_meta = cls.get_function_word_distractors(
                clean_target,
                target_count=target_count,
                exclude_words=exclude_words
            )
            if len(f_dists) >= target_count:
                if return_metadata:
                    return f_dists, f_meta
                return f_dists

        wn = cls.get_wordnet()
        ocd = cls.get_oxford_collocations()

        # Map POS to WordNet tag ('n', 'v', 'a', 'r')
        if pos.startswith("adv") or pos == "r":
            wn_pos = "r"
        elif pos.startswith("v"):
            wn_pos = "v"
        elif pos.startswith("n"):
            wn_pos = "n"
        elif pos.startswith("adj") or pos == "a":
            wn_pos = "a"
        else:
            wn_pos = None

        words = []
        if wn:
            if wn_pos == "a":
                # In WordNet, adjectives are split into head adjectives ('a') and satellite adjectives ('s')
                words = wn.words(clean_target, pos="a") + wn.words(clean_target, pos="s")
            elif wn_pos:
                words = wn.words(clean_target, pos=wn_pos)
            else:
                words = wn.words(clean_target)

            # Morphological base lemma fallback (e.g. 'possessions' -> 'possession')
            # and hyphenated/compound variant lookup (e.g. 'non-stop' -> 'nonstop')
            # ensures inflectional and compound variants access the full semantic taxonomy
            if not words:
                alt_forms = []
                if "-" in clean_target:
                    alt_forms.extend([clean_target.replace("-", ""), clean_target.replace("-", " "), clean_target.replace("-", "_")])
                if " " in clean_target:
                    alt_forms.extend([clean_target.replace(" ", "-"), clean_target.replace(" ", "_")])
                for alt in alt_forms:
                    if wn_pos == "a":
                        words = wn.words(alt, pos="a") + wn.words(alt, pos="s")
                    elif wn_pos:
                        words = wn.words(alt, pos=wn_pos)
                    else:
                        words = wn.words(alt)
                    if words:
                        break

            if not words and ("-" in clean_target or " " in clean_target):
                # Compound head word fallback: In English compounds (e.g. 'self-repair', 'decision-making'),
                # the syntactic and semantic head is usually the second/last component ('repair', 'making').
                comp_parts = re.split(r"[\-\s]+", clean_target)
                if len(comp_parts) >= 2 and comp_parts[-1]:
                    head_part = comp_parts[-1]
                    if wn_pos == "a":
                        words = wn.words(head_part, pos="a") + wn.words(head_part, pos="s")
                    elif wn_pos:
                        words = wn.words(head_part, pos=wn_pos)
                    else:
                        words = wn.words(head_part)

            if not words:
                nlp = cls.get_spacy()
                t_lemma = nlp(clean_target)[0].lemma_.lower()
                if t_lemma != clean_target:
                    if wn_pos == "a":
                        words = wn.words(t_lemma, pos="a") + wn.words(t_lemma, pos="s")
                    elif wn_pos:
                        words = wn.words(t_lemma, pos=wn_pos)
                    else:
                        words = wn.words(t_lemma)

        if words:
            oewn_words = [w for w in words if getattr(w, "id", "").startswith("oewn-")]
            if oewn_words:
                words = oewn_words

        exclude_words = set(exclude_words or set())

        def _pos_ok(lemma: str) -> bool:
            """Keep hyponym/coordinate candidates on the target's own part of speech."""
            if not wn_pos:
                return True
            try:
                if wn:
                    if wn_pos == "a":
                        if wn.words(lemma, pos="a") or wn.words(lemma, pos="s"):
                            return True
                    elif wn.words(lemma, pos=wn_pos):
                        return True
                c_ent = cls.get_ldoce_entry(lemma)
                if c_ent:
                    p_code = {"n": "n", "v": "v", "a": "adj", "r": "adv"}.get(wn_pos, "")
                    return any(cls._pos_code(p) == p_code for p in (c_ent.get("all_poses") or [c_ent.get("pos") or ""]))
                return False
            except Exception:
                return True

        def _is_primarily_adj_or_verb(lemma: str) -> bool:
            """Filter out collective nominals or words that are primarily verbs/adjectives."""
            try:
                cand_ent = cls.get_ldoce_entry(lemma)
                if cand_ent:
                    prim_pos = (cand_ent.get("pos") or "").lower()
                    if prim_pos in ("verb", "adjective", "adverb"):
                        return True
                if wn:
                    pos_list = [w.pos for w in wn.words(lemma)]
                    if any(p in ("a", "s") for p in pos_list):
                        if "n" not in pos_list or len([p for p in pos_list if p in ("a", "s")]) >= len([p for p in pos_list if p == "n"]):
                            return True
                    if "v" in pos_list and "n" not in pos_list:
                        return True
            except Exception:
                pass
            return False

        def _is_primarily_noun_or_adj_for_verb(lemma: str) -> bool:
            """Filter out nominal/adjectival homographs (e.g. 'pitch', 'gear', 'wire', 'square', 'tame', 'shoehorn')
            whose primary POS in PedLex is noun/adjective rather than verb."""
            try:
                cand_ent = cls.get_ldoce_entry(lemma)
                if cand_ent:
                    prim_pos = (cand_ent.get("pos") or "").lower()
                    if prim_pos in ("noun", "adjective", "adverb"):
                        return True
                if wn:
                    syn_v = len(wn.synsets(lemma, "v"))
                    syn_na = len(wn.synsets(lemma, "n")) + len(wn.synsets(lemma, "a")) + len(wn.synsets(lemma, "s"))
                    if syn_na > syn_v * 2 and syn_v <= 2:
                        return True
            except Exception:
                pass
            return False

        # Collect global synonyms across ALL synsets of target to permanently ban them
        nlp = cls.get_spacy()
        target_lemma = nlp(clean_target)[0].lemma_.lower()
        global_synonyms: Set[str] = {clean_target, target_lemma}
        exclude_words.add(clean_target)
        exclude_words.add(target_lemma)
        if "-" in clean_target or " " in clean_target:
            for part in re.split(r"[\-\s]+", clean_target):
                if len(part) >= 2:
                    exclude_words.add(part.lower())
                    global_synonyms.add(part.lower())
        for w in words:
            for s in w.synsets():
                for sw in s.words():
                    global_synonyms.add(sw.lemma().lower())
                if wn_pos == "v":
                    for vg_syn in s.get_related("verb_group"):
                        for vgw in vg_syn.words():
                            vg_lem = vgw.lemma().lower()
                            if " " not in vg_lem and "_" not in vg_lem:
                                global_synonyms.add(vg_lem)

        # Integrate PedLex native + Oxford Thesaurus of English synonyms & antonyms
        ped_ox_syns, ped_ox_ants = cls.get_pedlex_and_oxford_synonyms_antonyms(
            clean_target, pos=pos, definition=definition, quote=quote
        )
        strict_ped_ox_syns = (
            cls.get_strict_pedlex_and_oxford_synonyms(
                clean_target, pos=pos, definition=definition, quote=quote
            )
            if wn_pos == "n"
            else set()
        )
        for psyn in ped_ox_syns:
            if " " not in psyn:
                global_synonyms.add(psyn)
            # If bare-gloss synonym (e.g. 'luncheon' -> 'lunch'), also include its synonyms (e.g. 'dinner', 'banquet')
            if definition and psyn == re.sub(r"^\[.*?\]\s*", "", str(definition).strip().lower()):
                sub_syns, _ = cls.get_pedlex_and_oxford_synonyms_antonyms(psyn, pos=pos)
                for ss in sub_syns:
                    if " " not in ss:
                        global_synonyms.add(ss)

        # 1. Definition-Locked WSD: Rank synsets by token overlap with curriculum definition & hypernym chain
        target_synsets = []
        if definition:
            nlp = cls.get_spacy()
            def_clean = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", definition or "").strip()
            quote_clean = quote or ""
            def_doc = nlp(def_clean.lower())
            quote_doc = nlp(quote_clean.lower())
            self_forms = {clean_target, target_lemma} | {t.lemma_.lower() for t in nlp(clean_target)}
            _generic_degree_mods = {"extreme", "extremely", "complete", "completely", "total", "totally"}
            def_tokens = (({t.lemma_.lower() for t in def_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS) - self_forms) - _generic_degree_mods
            quote_tokens = ({t.lemma_.lower() for t in quote_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS) - self_forms

            scored_synsets = []
            seen_syn_ids = set()
            def_lower_stripped = def_clean.lower()
            pedlex_units: Set[str] = set()
            pedlex_signpost = ""
            pedlex_matched_gram = ""
            if wn_pos == "n":
                _ld_ent = cls.get_ldoce_entry(clean_target) or {}
                _ld_senses = [
                    sn for sn in (_ld_ent.get("senses") or [])
                    if not sn.get("pos") or cls._pos_code(sn.get("pos")) == "n"
                ] or (_ld_ent.get("senses") or [])
                _matched_sn = None
                for _sn in _ld_senses:
                    _sn_def = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", str(_sn.get("definition") or "")).strip().lower()
                    if _sn_def and (_sn_def == def_lower_stripped or _sn_def in def_lower_stripped or def_lower_stripped in _sn_def):
                        _matched_sn = _sn
                        break
                if _matched_sn is None and _ld_senses:
                    _matched_sn = _ld_senses[0]
                if _matched_sn:
                    for _u in (_matched_sn.get("units") or []):
                        if _u and isinstance(_u, str):
                            pedlex_units.add(_u.strip().lower())
                    pedlex_signpost = str(_matched_sn.get("signpost") or "").strip().lower()
                    pedlex_matched_gram = str(_matched_sn.get("grammar") or "").strip().lower()
            _material_genus_words: Set[str] = set()
            if wn_pos == "n":
                _mat_m = re.match(
                    r"^a\s+(?:small\s+|thin\s+|flat\s+|printed\s+|folded\s+|narrow\s+|short\s+|long\s+|round\s+|square\s+|hard\s+|soft\s+|stiff\s+)*piece\s+of\s+(paper|card|cardboard|metal|plastic|cloth|wood|glass|leather|rubber|stone)\s+(?:that|which|used|with|for|on|in|to|showing)\b",
                    def_lower_stripped,
                )
                if _mat_m:
                    _material_genus_words.add(_mat_m.group(1).lower())
            is_human_def = (
                any(h in def_tokens for h in ("people", "person", "organization", "employee", "staff", "student", "worker", "member", "social", "human", "group"))
                or def_lower_stripped.startswith(("a person", "someone ", "a man", "a woman", "people who"))
            )
            is_bio_sample_def = any(b in def_tokens for b in ("blood", "urine", "tissue", "medical", "scientific", "plant", "animal", "mineral", "bacteria", "cell", "organ"))
            is_military_def = any(m in def_tokens for m in ("military", "army", "soldier", "war", "battle", "weapon", "navy"))
            is_linguistic_def = any(
                lw in def_tokens
                for lw in ("word", "language", "speech", "grammar", "sentence", "letter", "writing", "expression", "phrase", "term", "meaning", "text", "voice", "sound", "verb", "noun", "pronounce", "syllable")
            )
            is_concrete_substance_or_sum_def = (
                wn_pos == "n"
                and def_lower_stripped.startswith((
                    "a layer of", "a piece of", "a substance", "a mass of",
                    "a small amount of", "an amount of", "a sum of", "a part of the cost"
                ))
            )

            for w in words:
                for s in w.synsets():
                    if s.id in seen_syn_ids:
                        continue
                    seen_syn_ids.add(s.id)

                    s_def_doc = nlp(s.definition().lower())
                    s_def_lemmas = ({t.lemma_.lower() for t in s_def_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS) - self_forms
                    s_peer_text = " ".join(sw.lemma().replace("-", " ").replace("_", " ") for sw in s.words())
                    s_direct_doc = nlp((s.definition() + " " + " ".join(s.examples()) + " " + s_peer_text).lower())
                    s_direct_lemmas = ({t.lemma_.lower() for t in s_direct_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS) - self_forms

                    chain_texts = []
                    cur = s
                    depth = 0
                    climbed_into_substance = False
                    while cur.hypernyms() and depth < 4:
                        cur = cur.hypernyms()[0]
                        if cur.lexfile() == "noun.substance":
                            climbed_into_substance = True
                        chain_texts.append(cur.definition())
                        for hw in cur.words():
                            chain_texts.append(hw.lemma())
                        depth += 1
                    if s.pos in ("a", "s"):
                        for adj_rel in s.get_related("similar") + s.get_related("also") + s.get_related("attribute"):
                            chain_texts.append(adj_rel.definition())
                            for hw in adj_rel.words():
                                chain_texts.append(hw.lemma().replace("-", " ").replace("_", " "))
                    chain_doc = nlp(" ".join(chain_texts).lower())
                    chain_lemmas = ({t.lemma_.lower() for t in chain_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS) - self_forms
                    if _material_genus_words and climbed_into_substance:
                        chain_lemmas = chain_lemmas - _material_genus_words

                    # Semantic Domain Guardrails:
                    # Penalize animal group or biological/substance/body-sample senses if target definition is human/social/person
                    penalty = 0
                    s_lf = s.lexfile() or ""
                    s_def_low = s.definition().lower()
                    chain_str_low = " ".join(chain_texts).lower()
                    if is_human_def and not is_bio_sample_def:
                        if (
                            "animal" in chain_lemmas
                            or "animal group" in chain_str_low
                            or s_lf in ("noun.body", "noun.animal", "noun.plant", "noun.substance")
                            or any(bio_w in s_def_low for bio_w in ("tissue", "blood", "urine", "biological", "plant", "mineral"))
                        ):
                            penalty += 15

                    # Penalize military/war senses if definition is not military/war
                    if not is_military_def and ("military" in s_def_low or "military" in chain_str_low):
                        penalty += 8

                    # Penalize metalinguistic/word-level senses (e.g. 'loanword: a word borrowed from another language')
                    # when the pedagogical definition is not about language/words/grammar
                    if not is_linguistic_def and (
                        s_def_low.startswith(("a word ", "an expression ", "a linguistic ", "the grammatical "))
                        or (s_lf == "noun.communication" and ("word" in s_direct_lemmas or "language" in s_direct_lemmas))
                    ):
                        penalty += 12

                    # Penalize abstract process/action synsets (e.g. 'deposition: the natural process of laying down a deposit')
                    # when the locked pedagogical definition denotes a concrete physical layer, substance, or monetary sum
                    if is_concrete_substance_or_sum_def and (
                        s_lf in ("noun.process", "noun.act")
                        or s_def_low.startswith(("the natural process of", "the act of", "the process of"))
                    ):
                        penalty += 10

                    # Prevent generic 'a ... place' location definitions (e.g. 'oasis: a peaceful or pleasant place...')
                    # from falsely locking onto a `noun.artifact` building/shelter synset solely via the meta-words 'place'/'provide'
                    if (
                        wn_pos == "n"
                        and "place" in def_tokens
                        and not (def_tokens & {"building", "structure", "shelter", "house", "room", "hut", "wall", "roof"})
                    ):
                        if s_lf == "noun.artifact":
                            s_direct_lemmas = s_direct_lemmas - {"place", "provide", "offer", "serve"}
                            chain_lemmas = chain_lemmas - {"place", "provide", "offer", "serve"}
                            penalty += 8
                        elif s_lf == "noun.location":
                            penalty -= 8

                    # Prevent extent/scope/range definitions (e.g. 'breadth: the fact of including many different things...')
                    # from locking onto `noun.cognition` intelligence/mental-capacity synsets when no mental terms are present
                    if (
                        wn_pos == "n"
                        and s_lf == "noun.cognition"
                        and bool((s_direct_lemmas | chain_lemmas) & {"intelligence", "understand", "mental", "capacity", "mind", "thought"})
                        and not (def_tokens & {"intelligence", "mind", "mental", "understand", "understanding", "knowledge", "thought", "wise", "wisdom", "clever"})
                    ):
                        penalty += 8

                    def_ov = len(def_tokens.intersection(s_direct_lemmas)) * 6 + len(def_tokens.intersection(chain_lemmas)) * 3
                    quote_ov = (
                        len(quote_tokens.intersection(s_def_lemmas)) * 2 +
                        len((quote_tokens - def_tokens).intersection(chain_lemmas)) * 1
                    )
                    if def_ov > 0:
                        quote_ov += len((quote_tokens - def_tokens).intersection(s_direct_lemmas - s_def_lemmas)) * 1
                    unit_signpost_bonus = 0
                    s_lemmas_low = {sw.lemma().lower() for sw in s.words()}
                    if pedlex_units and (s_lemmas_low & pedlex_units):
                        unit_signpost_bonus += 14
                    if pedlex_signpost and (pedlex_signpost in s_direct_lemmas or pedlex_signpost in chain_lemmas):
                        unit_signpost_bonus += 6
                    if wn_pos == "n" and strict_ped_ox_syns:
                        ox_syn_peers = (s_lemmas_low - self_forms) & strict_ped_ox_syns
                        if ox_syn_peers:
                            unit_signpost_bonus += min(12, len(ox_syn_peers) * 5)
                        ox_direct_peers = (s_direct_lemmas - self_forms) & strict_ped_ox_syns
                        if ox_direct_peers:
                            unit_signpost_bonus += min(6, len(ox_direct_peers) * 2)
                    score = def_ov + quote_ov + unit_signpost_bonus - penalty
                    scored_synsets.append((score, s))

            scored_synsets.sort(key=lambda x: x[0], reverse=True)
            if scored_synsets:
                best_ov = scored_synsets[0][0]
                if best_ov > 0:
                    target_synsets = [s for ov, s in scored_synsets if ov >= max(1, best_ov - 1)]
                else:
                    target_synsets = [scored_synsets[0][1]]
        if not target_synsets:
            target_synsets = [s for w in words for s in w.synsets()]
            # When no definition is provided for a polysemous noun, exclude informal/figurative
            # `noun.person` synsets (e.g. 'fossil' = old-fashioned person) if the primary PedLex sense
            # is non-human and non-person synsets exist.
            if not definition and wn_pos == "n" and len(target_synsets) > 1:
                _non_person_syns = [s for s in target_synsets if s.lexfile() != "noun.person"]
                if _non_person_syns and len(_non_person_syns) < len(target_synsets):
                    _t_ent_np = cls.get_ldoce_entry(clean_target) or {}
                    _t_sns_np = [
                        sn for sn in (_t_ent_np.get("senses") or [])
                        if not sn.get("pos") or cls._pos_code(sn.get("pos")) == "n"
                    ] or (_t_ent_np.get("senses") or [])
                    _first_def_np = str((_t_sns_np[0].get("definition") if _t_sns_np else "") or "").lower()
                    _first_reg_np = str((_t_sns_np[0].get("register") if _t_sns_np else "") or "").lower()
                    if not (
                        "informal" in _first_reg_np
                        or _first_def_np.startswith(("someone ", "a person", "a man", "a woman", "an old "))
                    ):
                        target_synsets = _non_person_syns

        # Deduplicate target_synsets preserving order
        dedup_synsets = []
        seen_syn_ids = set()
        for s in target_synsets:
            if s.id not in seen_syn_ids:
                seen_syn_ids.add(s.id)
                dedup_synsets.append(s)
        target_synsets = dedup_synsets

        # Bridge WordNet single-synset blind spot on concrete [plural] '-ics' nouns
        # (e.g. 'electronics' = "electronic equipment" [plural], 'ceramics' = "pots, plates etc made of clay" [plural],
        # 'hydraulics' = "parts of a machine or vehicle..." [plural], where WordNet only has the `noun.cognition`
        # academic-discipline synset "the branch of physics/science...").
        _matched_gram_for_ics = locals().get("pedlex_matched_gram", "")
        _acad_discipline_cues = {
            "study", "science", "scientific", "branch", "subject", "field",
            "theory", "art", "principle", "method", "philosophy", "politics", "academic",
        }
        _concrete_plural_cues = {
            "equipment", "device", "machine", "machinery", "part", "system", "wire",
            "electric", "electronic", "pot", "plate", "clay", "dish", "substance",
            "material", "liquid", "water", "pipe", "vehicle", "house", "room",
        }
        _bridged_concrete_plural_ics = False
        if (
            wn_pos == "n"
            and bool(definition)
            and "plural" in _matched_gram_for_ics
            and "uncountable" not in _matched_gram_for_ics
            and target_synsets
            and all(s.lexfile() == "noun.cognition" for s in target_synsets)
            and not (def_tokens & _acad_discipline_cues)
            and bool(def_tokens & _concrete_plural_cues)
        ):
            _bridged_concrete_plural_ics = True
            # Block superordinate/overlapping device/appliance/equipment terms from becoming double-keys
            if def_tokens & {"equipment", "electronic", "electric", "device", "machine", "machinery", "wire"}:
                global_synonyms.update({
                    "appliance", "device", "machine", "machinery", "equipment",
                    "apparatus", "hardware", "gadget", "instrument", "mechanism",
                })
            _bridge_seeds = []
            for _b_lem in ("electronic_equipment", "clothing", "conveyance", "ceramic"):
                for _bs in wn.synsets(_b_lem, "n")[:1]:
                    if _bs.lexfile() == "noun.artifact":
                        _bridge_seeds.append(_bs)
            if _bridge_seeds:
                target_synsets = _bridge_seeds

        # For satellite/head adjectives ('a'/'s'), track same-pole similar synset lemmas
        # so direct cluster synonyms (e.g. 'heavy' for 'burdensome', 'uncommon' for 'rare', 'unlikely' for 'improbable')
        # are never misclassified as antonyms or unanchored distractors.
        same_pole_adj_similars: Set[str] = set()
        for s in target_synsets:
            if wn_pos == "a" and s.pos in ("a", "s"):
                for head in s.get_related("similar"):
                    for hw in head.words():
                        hw_lem = hw.lemma().lower()
                        same_pole_adj_similars.add(hw_lem)
                        if s.pos == "s":
                            global_synonyms.add(hw_lem)
        if wn_pos == "a":
            for w in words:
                for s in w.synsets():
                    for sw in s.words():
                        same_pole_adj_similars.add(sw.lemma().lower())
                    for head in s.get_related("similar"):
                        for hw in head.words():
                            same_pole_adj_similars.add(hw.lemma().lower())

        # 2. Extract Sense-Locked Antonyms from PedLex/Oxford Thesaurus + matched WordNet synsets
        sense_antonyms: List[str] = []
        raw_antonyms: List[Tuple[str, int]] = []

        t_ent_reg = cls.get_ldoce_entry(clean_target) or {}
        t_regs_all = " ".join(str(s.get("register") or "") for s in (t_ent_reg.get("senses") or [])).lower()
        target_is_informal = any(r in t_regs_all for r in ("informal", "slang", "taboo")) or (
            bool(definition) and any(r in definition.lower() for r in ("[informal]", "[slang]", "informal", "slang"))
        )
        target_lfs_set = {s.lexfile() for s in target_synsets if s.lexfile()}
        # Determine target noun countability & physical-object status from PedLex locked sense
        _t_gram_low = locals().get("pedlex_matched_gram", "")
        if wn_pos == "n" and not _t_gram_low:
            _t_ent_g = cls.get_ldoce_entry(clean_target) or {}
            _t_sns_g = [
                sn for sn in (_t_ent_g.get("senses") or [])
                if not sn.get("pos") or cls._pos_code(sn.get("pos")) == "n"
            ] or (_t_ent_g.get("senses") or [])
            if _t_sns_g:
                _t_gram_low = str(_t_sns_g[0].get("grammar") or "").strip().lower()
        target_strictly_uncountable = (
            wn_pos == "n"
            and (
                ("uncountable" in _t_gram_low and "countable" not in re.sub(r"\buncountable\b", "", _t_gram_low))
                or cls._is_primarily_uncountable_noun(clean_target)
            )
            and not _bridged_concrete_plural_ics
            and "plural" not in _t_gram_low
        )
        target_strictly_countable = (
            wn_pos == "n"
            and not target_strictly_uncountable
            and (
                cls._is_primarily_countable_noun(clean_target)
                or ("countable" in re.sub(r"\buncountable\b", "", _t_gram_low) and "uncountable" not in _t_gram_low)
                or _bridged_concrete_plural_ics
            )
        )
        _CONCRETE_PHYSICAL_LFS = {
            "noun.substance", "noun.artifact", "noun.plant", "noun.animal",
            "noun.food", "noun.object", "noun.body",
        }
        is_concrete_physical_noun = wn_pos == "n" and (
            bool(locals().get("_material_genus_words"))
            or (bool(target_lfs_set) and target_lfs_set.issubset(_CONCRETE_PHYSICAL_LFS))
        )

        def _is_informal_or_slang_distractor(cand_w: str) -> bool:
            if cls._is_taboo_or_proper_noun(cand_w, wn_pos=wn_pos):
                return True
            if target_is_informal:
                return False
            c_ent = cls.get_ldoce_entry(cand_w)
            if not c_ent:
                return False
            base_cand = str(c_ent.get("word") or cand_w).strip().lower()
            c_senses = c_ent.get("senses") or []
            p_code = {"n": "n", "v": "v", "a": "adj", "r": "adv"}.get(wn_pos or "n", "n")
            pos_senses = [s for s in c_senses if not s.get("pos") or cls._pos_code(s.get("pos")) == p_code] or c_senses
            if not pos_senses:
                return False
            bad_regs = ("informal", "slang", "taboo", "old-fashioned", "old use", "biblical")
            # Case A: Every matching-POS sense (or the primary sense of a monosemous/derived word) is informal/slang
            if all(any(r in str(s.get("register") or "").lower() for r in bad_regs) for s in pos_senses):
                return True
            # Case A2: Inflected/plural surface lemma (e.g. 'clams' -> base 'clam') where every sense attesting the plural/surface form is informal/slang
            if cand_w != base_cand:
                surf_senses = [
                    s for s in pos_senses
                    if cand_w in " ".join(str(u.get("phrase") if isinstance(u, dict) else u) for u in (s.get("units") or [])).lower().split()
                    or "[plural]" in str(s.get("grammar") or "").lower()
                ]
                if surf_senses and all(any(r in str(s.get("register") or "").lower() for r in bad_regs) for s in surf_senses):
                    return True
            # Case B: Polysemous word whose non-informal senses belong to an unrelated concrete domain (e.g. 'clam' sense 1
            # is a sea creature/shellfish, while only its [informal] sense 2 means 'dollar')
            has_inf_sense = any(any(r in str(s.get("register") or "").lower() for r in ("informal", "slang", "taboo")) for s in pos_senses)
            if has_inf_sense and wn_pos == "n" and target_lfs_set:
                c_wn_syns = wn.synsets(base_cand, "n") or wn.synsets(cand_w, "n")
                if c_wn_syns:
                    primary_lf = c_wn_syns[0].lexfile()
                    if primary_lf in ("noun.animal", "noun.plant", "noun.food", "noun.body") and primary_lf not in target_lfs_set:
                        return True
            return False

        def _evaluate_antonym(al: str, priority_bonus: int = 0):
            al = al.strip().lower()
            if (
                al not in global_synonyms
                and al not in same_pole_adj_similars
                and al != clean_target
                and "_" not in al
                and " " not in al
                and "-" not in al
                and cls.is_exact_pedlex_headword(al)
                and _pos_ok(al)
                and al not in exclude_words
                and not _is_informal_or_slang_distractor(al)
            ):
                if cls.is_cefr_compliant_distractor(al, clean_target):
                    cefr_penalty = 0
                else:
                    t_lvl_ant = cls._lexicon_profile(clean_target)[2] or "B1"
                    al_lvl_ant = cls._lexicon_profile(al)[2] or "C2"
                    text_ceil_rank = cls.CEFR_ORDER.get(cls.cefr_ceiling(t_lvl_ant, mode="text"), 4)
                    if cls.CEFR_ORDER.get(al_lvl_ant, 6) > text_ceil_rank:
                        return
                    cefr_penalty = 4
                al_ent = cls.get_ldoce_entry(al)
                if al_ent:
                    al_primary = (al_ent.get("pos") or "").lower()
                    al_poses = [p.lower() for p in al_ent.get("all_poses", [al_primary])]
                    if wn_pos == "n" and not any("noun" in p for p in al_poses):
                        return
                    if wn_pos == "v" and not any("verb" in p for p in al_poses):
                        return
                    if wn_pos == "a" and not any("adj" in p for p in al_poses):
                        return
                    if wn_pos == "r" and not any("adv" in p for p in al_poses):
                        return
                    if al_ent.get("kind") != "derived" and not any(
                        s.get("examples") or s.get("pattern_bundles") or s.get("patterns")
                        for s in (al_ent.get("senses") or [])
                    ):
                        is_morph_neg = any(
                            al.startswith(p) and al[len(p):] == clean_target
                            for p in ("un", "in", "im", "ir", "il", "non", "dis")
                        )
                        if not is_morph_neg:
                            return
                        cefr_penalty += 2
                al_senses = (al_ent or {}).get("senses") or []
                al_pos_senses = [
                    s for s in al_senses
                    if not wn_pos or not s.get("pos") or cls._pos_code(s.get("pos")) == {"n": "n", "v": "v", "a": "adj", "r": "adv"}.get(wn_pos, "")
                ] or al_senses
                if is_concrete_physical_noun:
                    al_wn = wn.synsets(al, "n")
                    if (
                        al.endswith(("tion", "sion", "ness", "ment", "ity", "usal", "ance", "ence"))
                        or (al_wn and al_wn[0].lexfile() in ("noun.act", "noun.attribute", "noun.state", "noun.cognition", "noun.feeling", "noun.communication"))
                    ):
                        return
                _bad_ant_regs = ("old use", "old-fashioned", "biblical", "literary", "informal", "slang")
                if al_pos_senses and (
                    any(r in str(al_pos_senses[0].get("register") or "").lower() for r in _bad_ant_regs)
                    or all(any(r in str(s.get("register") or "").lower() for r in _bad_ant_regs) for s in al_pos_senses)
                ):
                    cefr_penalty += 3
                if al in ocd:
                    prio = 0 - priority_bonus + cefr_penalty
                elif cls.get_word_cefr(al) in ("A1", "A2", "B1", "B2", "C1") or any(al.startswith(p) and al[len(p):] in ocd for p in ("un", "in", "im", "ir", "il", "non", "dis")):
                    prio = 1 - priority_bonus + cefr_penalty
                elif wn_pos in ("a", "r"):
                    prio = 2 - priority_bonus + cefr_penalty
                else:
                    return
                raw_antonyms.append((al, prio))

        # Priority 0: Authoritative PedLex OPP + Oxford Thesaurus antonyms + symmetrical prefix duals
        for ant_cand in ped_ox_ants:
            _evaluate_antonym(ant_cand, priority_bonus=3)
        for p1, p2 in (
            ("on", "off"), ("off", "on"),
            ("in", "out"), ("out", "in"),
            ("up", "down"), ("down", "up"),
            ("over", "under"), ("under", "over"),
            ("pre", "post"), ("post", "pre"),
        ):
            if clean_target.startswith(p1) and len(clean_target) - len(p1) >= 4:
                dual_cand = p2 + clean_target[len(p1):]
                if cls.is_exact_pedlex_headword(dual_cand):
                    _evaluate_antonym(dual_cand, priority_bonus=3)

        for s in target_synsets:
            # Direct & Synset-Peer sense antonyms (SnappyWords/WordVis synset-peer bridging)
            for sense in s.senses():
                is_self_sense = sense.word().lemma().lower() in (clean_target, target_lemma)
                base_bonus = 2 if is_self_sense else 1
                for ant_sense in sense.get_related("antonym"):
                    ant_lemma = ant_sense.word().lemma().lower()
                    _evaluate_antonym(ant_lemma, priority_bonus=base_bonus)
                    # Opposite satellites in bipolar cluster (1 hop via 'similar' only; never 'also' which jumps back across poles)
                    for opp_sim in ant_sense.synset().get_related("similar"):
                        for ow in opp_sim.words():
                            ow_lemma = ow.lemma().lower()
                            _evaluate_antonym(ow_lemma, priority_bonus=max(0, base_bonus - 1))
            # If adjective satellite ('s'), trace to head synset in bipolar cluster
            if wn_pos == "a" and s.pos == "s":
                for head in s.get_related("similar"):
                    for hs in head.senses():
                        for ant_sense in hs.get_related("antonym"):
                            ant_lemma = ant_sense.word().lemma().lower()
                            _evaluate_antonym(ant_lemma, priority_bonus=2)
                            for opp_sim in ant_sense.synset().get_related("similar"):
                                for ow in opp_sim.words():
                                    ow_lemma = ow.lemma().lower()
                                    _evaluate_antonym(ow_lemma, priority_bonus=1)
            # Cross-POS Pertainym / Derivation Antonym Projection (e.g. reluctantly -> reluctant <-> eager -> eagerly,
            # scarcity -> scarce <-> abundant -> abundance)
            _CONCRETE_NOUN_LEXFILES = {
                "noun.substance", "noun.artifact", "noun.plant", "noun.animal",
                "noun.food", "noun.object", "noun.body", "noun.location",
            }
            for sense in s.senses():
                if sense.word().lemma().lower() not in (clean_target, target_lemma):
                    continue
                if wn_pos == "n" and s.lexfile() in _CONCRETE_NOUN_LEXFILES:
                    continue
                for base_sense in sense.get_related("pertainym") + sense.get_related("derivation"):
                    b_syn = base_sense.synset()
                    b_ant_senses = list(base_sense.get_related("antonym"))
                    if b_syn.pos == "s":
                        for b_head in b_syn.get_related("similar"):
                            for b_hs in b_head.senses():
                                b_ant_senses.extend(b_hs.get_related("antonym"))
                    for b_ant in b_ant_senses:
                        for proj in b_ant.get_related("derivation") + b_ant.get_related("pertainym"):
                            p_pos = proj.synset().pos
                            if wn_pos == "n" and proj.synset().lexfile() != s.lexfile():
                                continue
                            if (wn_pos == "a" and p_pos in ("a", "s")) or (wn_pos and p_pos == wn_pos):
                                _evaluate_antonym(proj.word().lemma().lower(), priority_bonus=1)
                        if wn_pos == "r" and b_ant.synset().pos in ("a", "s"):
                            adj_lem = b_ant.word().lemma().lower()
                            ly_cands = [adj_lem + "ly"]
                            if adj_lem.endswith("y") and len(adj_lem) > 2:
                                ly_cands.append(adj_lem[:-1] + "ily")
                            elif adj_lem.endswith("le") and len(adj_lem) > 2:
                                ly_cands.append(adj_lem[:-1] + "y")
                            elif adj_lem.endswith("ic"):
                                ly_cands.append(adj_lem + "ally")
                            for ly_c in ly_cands:
                                if cls.is_exact_pedlex_headword(ly_c):
                                    _evaluate_antonym(ly_c, priority_bonus=1)

        raw_antonyms.sort(key=lambda x: x[1])
        for al, _ in raw_antonyms:
            if al not in sense_antonyms:
                sense_antonyms.append(al)

        # PedLex 6th Edition Thesaurus, Grammar Box Comparison Peers & Activator Integration (Tier 0: Primary Pedagogical Lexical Source)
        tier0_ldoce_thesaurus: List[str] = []
        seen = {clean_target}
        seen.update(sense_antonyms)

        thes_items = list(cls.get_ldoce_thesaurus(clean_target))
        t_gram_prof = cls.get_word_gram_profile(clean_target, pos=pos)
        for peer_w in t_gram_prof.get("comparison_peers", set()):
            thes_items.append({
                "word": peer_w,
                "form": "word",
                "distinction": definition or "",
                "grammar": "",
                "register": "",
                "concept": "",
                "is_grambox_peer": True,
            })

        if thes_items:
            # Rank items by content-word definition and quote overlap + syntactic slot clash bonus
            _thes_stopwords = {
                "the", "and", "for", "that", "with", "from", "into", "about", "over", "under", "between", "through",
                "something", "someone", "somebody", "anything", "anyone", "anybody", "everything", "everyone",
                "person", "people", "thing", "things", "way", "time", "part", "place", "kind", "type", "sort", "form",
                "act", "action", "state", "quality", "fact", "process", "example", "used", "using", "use", "make",
                "making", "made", "have", "having", "had", "give", "giving", "take", "taking", "get", "getting",
                "want", "wanting", "feel", "feeling", "show", "showing", "cause", "causing", "become", "becoming",
                "able", "very", "more", "most", "much", "many", "some", "other", "another", "such", "same", "different",
                "good", "bad", "high", "low", "large", "small", "long", "short", "new", "old", "young", "particular",
                "especially", "usually", "often", "always", "never", "when", "where", "which", "what", "who", "whom",
                "whose", "while", "because", "although", "though", "whether", "without", "within", "during", "before",
                "after", "above", "below", "around", "among", "against", "towards", "toward", "upon", "onto", "they",
                "them", "their", "theirs", "themselves", "you", "your", "yours", "yourself", "our", "ours", "his",
                "her", "hers", "its", "this", "that", "these", "those", "there", "here", "will", "would", "could",
                "should", "might", "must", "can", "may", "not", "only", "also", "even", "still", "just", "well",
                "than", "then", "too", "out", "off", "down", "back", "away", "one", "ones", "two", "all", "both",
                "each", "every", "either", "neither", "none", "own", "being", "been", "were", "was", "are", "does", "did",
            }
            q_words = {w for w in re.findall(r"\b[a-zA-Z]{3,}\b", (quote or "").lower()) if w not in _thes_stopwords}
            d_words = {w for w in re.findall(r"\b[a-zA-Z]{3,}\b", (definition or "").lower()) if w not in _thes_stopwords}
            content_words = d_words | q_words

            # Map each concept cluster to the target word's own distinction in that cluster (if present)
            # so polysemous targets (e.g. 'compulsion' sense 1 'strong desire' vs sense 2 'forcing/obligation')
            # do not pull Activator/Thesaurus clusters belonging to a different sense.
            cluster_self_distinction: Dict[str, Set[str]] = {}
            target_ortho_set = set(cls.orthographic_variants(clean_target)) if hasattr(cls, "orthographic_variants") else {clean_target}
            for it in thes_items:
                it_w = str(it.get("word", "")).strip().lower()
                it_concept = str(it.get("concept", "")).strip().lower()
                it_cluster_key = it_concept or (f"sec_{it.get('section_index')}" if it.get("section_index") is not None else "")
                if it_w in target_ortho_set and it_cluster_key:
                    self_toks = {
                        w for w in re.findall(r"\b[a-zA-Z]{3,}\b", (str(it.get("distinction", "")) + " " + it_concept).lower())
                        if w not in _thes_stopwords and w not in target_ortho_set
                    }
                    cluster_self_distinction[it_cluster_key] = self_toks

            scored_thes_items = []
            for it in thes_items:
                it_concept = str(it.get("concept", "")).strip().lower()
                it_cluster_key = it_concept or (f"sec_{it.get('section_index')}" if it.get("section_index") is not None else "")
                it_dist = str(it.get("distinction", "")).strip().lower()
                it_reg = str(it.get("register") or "").lower()
                if not target_is_informal and any(
                    r in it_reg for r in ("informal", "slang", "taboo", "old-fashioned", "old use", "biblical")
                ):
                    continue
                # POS-frame guard on Language Activator clusters:
                # Reject verb concept clusters ('to put something somewhere') when target is a noun/adjective,
                # and reject noun concept clusters ('a layer of...') when target is a verb.
                if wn_pos != "v" and (it_concept.startswith("to ") or it_dist.startswith("to ")):
                    continue
                if wn_pos == "v" and it_concept and not it_concept.startswith("to ") and it_dist.startswith(("a ", "an ", "the ")):
                    continue
                if d_words and it_cluster_key in cluster_self_distinction and not it.get("is_grambox_peer"):
                    if not (cluster_self_distinction[it_cluster_key] & (d_words | q_words | ped_ox_syns)):
                        continue
                t_text = (it.get("word", "") + " " + it_dist + " " + it_concept).lower()
                t_toks = {w for w in re.findall(r"\b[a-zA-Z]{3,}\b", t_text) if w not in _thes_stopwords}
                overlap_score = len(t_toks & d_words) * 3 + len(t_toks & q_words)

                raw_cand_head = str(it.get("word", "")).strip().lower().split("/")[0].split()[0] if it.get("word") else ""
                has_clash = False
                if raw_cand_head:
                    has_clash, _ = cls.evaluate_syntactic_slot_clash(
                        clean_target, raw_cand_head,
                        anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )
                    if has_clash:
                        overlap_score += 15

                # If definition/quote provided, filter out Activator/Thesaurus clusters with zero content-word overlap against the target sense
                # UNLESS the candidate has a verified syntactic slot clash or is a comparison peer
                if content_words and overlap_score == 0 and not it.get("is_grambox_peer") and not has_clash:
                    continue

                # Boost true Thesaurus distinction records slightly over broad Activator clusters
                if not it.get("concept"):
                    overlap_score += 1
                if it.get("is_grambox_peer"):
                    overlap_score += 5
                scored_thes_items.append((overlap_score, it))
            scored_thes_items.sort(key=lambda x: x[0], reverse=True)

            for _, it in scored_thes_items:
                raw_w = it.get("word", "").strip().lower()
                # Skip instructional template phrases containing 'etc'
                if "etc" in raw_w:
                    continue
                # If target is a single word and the entry is a concrete multi-word phrase
                # (e.g. 'breach of protocol', 'defy convention', 'a matter of convention'),
                # skip it to prevent dismantling collocations into false synonyms or action heads.
                if " " not in clean_target and it.get("form") == "phrase":
                    if not any(ph in raw_w.split() for ph in ("sb", "sth", "someone", "something", "oneself")):
                        continue
                # If phrase contains spaces and slashes (e.g. 'the Department of Science/English/Trade'),
                # it is a template frame when the first alternative itself contains a space (unless verb phrase);
                # if the first alternative is a single word (e.g. 'compete/take part'), keep single-word alternatives.
                if "/" in raw_w and wn_pos != "v":
                    parts_check = [p.strip() for p in raw_w.split("/") if p.strip()]
                    if parts_check and " " in parts_check[0]:
                        continue
                chunks = [ch.strip() for ch in raw_w.split("/")] if "/" in raw_w else [raw_w]
                for chunk in chunks:
                    if not chunk:
                        continue
                    parts = chunk.split()
                    cand = None
                    if len(parts) == 1:
                        cand = parts[0]
                    elif len(parts) >= 2:
                        # Extract verb head from verb phrases ONLY if followed by standard placeholders (e.g. 'perform sth')
                        # or bound prepositions (e.g. 'derive from', 'stem from', 'result from')
                        # Do NOT extract verb head from phrasal verbs or idiomatic phrases (e.g. 'come forward', 'let me')
                        if wn_pos == "v":
                            _light_verbs = {"make", "get", "have", "take", "give", "do", "put", "set", "keep", "let", "go", "come", "be"}
                            if (
                                len(parts) == 2
                                and parts[1] in ("sb", "sth", "someone", "something", "oneself")
                                and parts[0].lower() not in _light_verbs
                            ):
                                cand = parts[0]
                            elif (
                                len(parts) == 2
                                and parts[1] in cls._STANDARD_PREPOSITIONS
                                and parts[0].lower() not in _light_verbs
                            ):
                                cand = parts[0]
                            else:
                                continue
                        # Extract noun head from noun phrases (e.g. 'branch of', 'facilities for', 'member of')
                        elif wn_pos == "n" and parts[1] in ("of", "for", "to", "in", "on", "with", "as", "sb", "sth"):
                            if len(parts) >= 3 and parts[-1] not in ("sb", "sth", "someone", "something", "oneself", "etc"):
                                continue
                            p_doc = nlp(parts[0])
                            if p_doc and p_doc[0].pos_ in ("NOUN", "PROPN"):
                                cand = parts[0]
                    if not cand or not cand.isalpha() or len(cand) < 3:
                        continue

                    if cls.is_exact_pedlex_headword(cand):
                        picked = cand
                    else:
                        cand_doc = nlp(cand)
                        cand_lemma = cand_doc[0].lemma_.lower()
                        picked = cand_lemma if cls.is_exact_pedlex_headword(cand_lemma) else cand
                    has_picked_clash, _ = cls.evaluate_syntactic_slot_clash(
                        clean_target, picked,
                        anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )
                    # Differentiated PedLex Thesaurus peers (e.g. 'loan' vs 'mortgage'/'overdraft' under anchor 'of')
                    # are curated contrastive peers: allow direct synset synonyms ONLY when a bound prepositional
                    # anchor or syntactic slot clash strictly rules them out, and never when the candidate literally
                    # defines the target word in its dictionary definition (e.g. 'interested' in 'intrigued').
                    is_anchor_discriminated_peer = bool(
                        context_anchor
                        and anchor_type == "prep"
                        and picked not in d_words
                        and cls.lemma_of(picked) not in d_words
                        and (
                            wn_pos != "n"
                            or (
                                it.get("section_index") is not None
                                and context_anchor.lower() != "into"
                                and clean_target not in cls.get_strict_pedlex_and_oxford_synonyms(picked, pos="noun")
                                and cls.lemma_of(clean_target) not in cls.get_strict_pedlex_and_oxford_synonyms(picked, pos="noun")
                            )
                        )
                        and not cls.double_key_collision(
                            clean_target, picked,
                            anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                        )[0]
                    )

                    if (
                        picked
                        and picked != clean_target
                        and picked not in seen
                        and (picked not in global_synonyms or has_picked_clash or is_anchor_discriminated_peer)
                        and not cls.are_same_word_family(picked, clean_target)
                        and not any(cls.are_same_word_family(picked, existing) for existing in tier0_ldoce_thesaurus)
                        and not cls._is_collocational_partner(clean_target, picked)
                        and _pos_ok(picked)
                        and cls.is_cefr_compliant_distractor(picked, clean_target)
                        and not _is_informal_or_slang_distractor(picked)
                    ):
                        cand_entry = cls.get_ldoce_entry(picked)
                        if cand_entry:
                            c_primary = (cand_entry.get("pos") or "").lower()
                            c_poses = [p.lower() for p in cand_entry.get("all_poses", [c_primary])]
                            c_senses = cand_entry.get("senses") or []
                            if wn_pos == "n":
                                if not any("noun" in p for p in c_poses):
                                    continue
                                # Reject if primary PedLex POS is verb or adjective unless it has an attested noun sense
                                if c_primary in ("verb", "adjective", "adverb") and not any(s.get("pos") == "noun" for s in c_senses):
                                    continue
                            elif wn_pos == "v":
                                if not any("verb" in p for p in c_poses):
                                    continue
                                if c_primary in ("noun", "adjective", "adverb") and not any(s.get("pos") == "verb" for s in c_senses):
                                    continue
                            elif wn_pos == "a":
                                if not any("adj" in p for p in c_poses):
                                    continue
                                if not cls.distractor_occupies_slot(picked, "adj"):
                                    continue
                                # Reject words whose adjective senses are strictly [only before noun] when target is not attributive-only
                                adj_senses = [s for s in c_senses if s.get("pos") == "adjective"]
                                if adj_senses and all("[only before noun]" in str(s.get("grammar") or "") for s in adj_senses):
                                    t_entry = cls.get_ldoce_entry(clean_target)
                                    t_grams = [str(s.get("grammar") or "") for s in ((t_entry or {}).get("senses") or [])]
                                    if not (t_grams and all("[only before noun]" in g for g in t_grams)):
                                        continue
                        seen.add(picked)
                        tier0_ldoce_thesaurus.append(picked)
                        # Cluster Throttling: pick at most 1 distinct distractor per Thesaurus/Activator entry
                        break
                if len(tier0_ldoce_thesaurus) >= target_count * 2:
                    break

        tier1_synonyms: List[str] = []
        tier2_satellites: List[str] = []
        tier3_attributes: List[str] = []
        tier2_hyponyms: List[str] = []
        tier3_coordinates: List[str] = []
        coord_hyper_overlap: Dict[str, int] = {}
        coord_domain_cues: Set[str] = set()

        for s in target_synsets:
            # Tier 1: Synonyms in synset (Only for nouns/adjectives; strictly banned for verbs)
            if wn_pos != "v":
                for sw in s.words():
                    lemma = sw.lemma().lower()
                    if (
                        lemma not in seen
                        and " " not in lemma
                        and "_" not in lemma
                        and "-" not in lemma
                        and (cls.is_exact_pedlex_headword(lemma) if cls._ensure_ldoce_conn() else lemma in ocd)
                        and cls.is_cefr_compliant_distractor(lemma, clean_target)
                        and not _is_informal_or_slang_distractor(lemma)
                    ):
                        if wn_pos == "n" and (_is_primarily_adj_or_verb(lemma) or not cls.is_exact_pedlex_headword(lemma)):
                            continue
                        if lemma in global_synonyms:
                            continue
                        seen.add(lemma)
                        tier1_synonyms.append(lemma)

            if wn_pos == "a":
                # For Adjectives: WordNet organizes concepts into Bipolar Clusters (Satellite Synset Harvester)
                # Tier 2: Satellite synsets (fine-grained descriptive neighbors in the bipolar cluster)
                sat_synsets = []
                if s.pos == "a":
                    sat_synsets.extend(s.get_related("also") + s.get_related("similar"))
                elif s.pos == "s":
                    for head in s.get_related("similar"):
                        sat_synsets.extend(head.get_related("also") + head.get_related("similar"))

                for sim in sat_synsets:
                    for sw in sim.words():
                        lemma = sw.lemma().lower()
                        if (
                            lemma not in seen
                            and (
                                lemma not in global_synonyms
                                or cls.evaluate_syntactic_slot_clash(
                                    clean_target, lemma, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                                )[0]
                            )
                            and " " not in lemma
                            and "_" not in lemma
                            and "-" not in lemma
                            and (cls.is_exact_pedlex_headword(lemma) if cls._ensure_ldoce_conn() else lemma in ocd)
                            and _pos_ok(lemma)
                            and cls.is_cefr_compliant_distractor(lemma, clean_target)
                            and not _is_informal_or_slang_distractor(lemma)
                        ):
                            seen.add(lemma)
                            tier2_satellites.append(lemma)

                # Tier 3: Attribute coordinates (adjectives sharing the same abstract attribute, e.g. speed, scope, magnitude)
                attrs = s.get_related("attribute")
                if s.pos == "s":
                    for head in s.get_related("similar"):
                        attrs.extend(head.get_related("attribute"))
                for attr in attrs:
                    for att_adj in attr.get_related("attribute"):
                        for aw in att_adj.words():
                            lemma = aw.lemma().lower()
                            if (
                                lemma not in seen
                                and lemma not in global_synonyms
                                and " " not in lemma
                                and "_" not in lemma
                                and "-" not in lemma
                                and (cls.is_exact_pedlex_headword(lemma) if cls._ensure_ldoce_conn() else lemma in ocd)
                                and _pos_ok(lemma)
                                and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                and not _is_informal_or_slang_distractor(lemma)
                            ):
                                seen.add(lemma)
                                tier3_attributes.append(lemma)
                        for sim in att_adj.get_related("similar"):
                            for sw in sim.words():
                                lemma = sw.lemma().lower()
                                if (
                                    lemma not in seen
                                    and lemma not in global_synonyms
                                    and " " not in lemma
                                    and "_" not in lemma
                                    and "-" not in lemma
                                    and (cls.is_exact_pedlex_headword(lemma) if cls._ensure_ldoce_conn() else lemma in ocd)
                                    and _pos_ok(lemma)
                                    and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                    and not _is_informal_or_slang_distractor(lemma)
                                ):
                                    seen.add(lemma)
                                    tier3_attributes.append(lemma)
                # Tier 3b: Relational Adjective Pertainym / Derivation Graph Traversal
                # Covers both denominal pertainyms (`adj.pert` like `doctoral`, `medical`, `musical`)
                # and deverbal/institutional relational adjectives (`regulatory`, `compensatory`, `descriptive`, `advisory`)
                # whose WordNet `derivation` links point to verbs (`pos == "v"`) or LDOCE `word_family` nouns/verbs.
                if len(tier2_satellites) + len(tier3_attributes) < target_count * 2:
                    pert_base_syns = []
                    pert_base_lemmas = []
                    for sn in s.senses():
                        for rel_sn in sn.get_related("pertainym") + sn.get_related("derivation"):
                            try:
                                rs = rel_sn.synset()
                                if rs and rs.pos in ("n", "v") and rs not in pert_base_syns:
                                    pert_base_syns.append(rs)
                                r_lem = rel_sn.word().lemma().lower()
                                r_pos = rs.pos if rs else None
                                if r_lem and r_pos in ("n", "v") and (r_lem, r_pos) not in pert_base_lemmas:
                                    pert_base_lemmas.append((r_lem, r_pos))
                            except Exception:
                                pass
                    for _b_lem, _b_pos in pert_base_lemmas:
                        if _b_pos == "v":
                            for _bs in wn.synsets(_b_lem, "v")[:3]:
                                if _bs not in pert_base_syns:
                                    pert_base_syns.append(_bs)
                    if not pert_base_syns:
                        _t_ent_wf = cls.get_ldoce_entry(clean_target) or {}
                        _wf_map = _t_ent_wf.get("word_family") or {}
                        for _wf_w in list(_wf_map.get("noun") or []) + list(_wf_map.get("verb") or []):
                            _wf_clean = str(_wf_w or "").strip().lower()
                            if _wf_clean and _wf_clean != clean_target:
                                for _rs in wn.synsets(_wf_clean, "n")[:2] + wn.synsets(_wf_clean, "v")[:2]:
                                    if _rs not in pert_base_syns:
                                        pert_base_syns.append(_rs)
                    sister_base_syns = []
                    for pns in pert_base_syns:
                        for hyp in pns.hypernyms():
                            if hyp.id in ("oewn-05844071-n", "oewn-00002137-n"):
                                continue
                            if hyp not in sister_base_syns:
                                sister_base_syns.append(hyp)
                            for coh in hyp.hyponyms()[:25]:
                                if coh not in sister_base_syns:
                                    sister_base_syns.append(coh)
                    _tier3b_cands: List[str] = []
                    for sns in sister_base_syns:
                        for n_sn in sns.senses():
                            for drv_sn in n_sn.get_related("derivation") + n_sn.get_related("pertainym"):
                                try:
                                    ds = drv_sn.synset()
                                    if not ds or ds.pos not in ("a", "s"):
                                        continue
                                    lemma = drv_sn.word().lemma().lower()
                                    if (
                                        lemma not in seen
                                        and lemma not in global_synonyms
                                        and not cls.are_same_word_family(lemma, clean_target)
                                        and " " not in lemma
                                        and "_" not in lemma
                                        and "-" not in lemma
                                        and (cls.is_exact_pedlex_headword(lemma) if cls._ensure_ldoce_conn() else lemma in ocd)
                                        and _pos_ok(lemma)
                                        and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                        and not _is_informal_or_slang_distractor(lemma)
                                    ):
                                        _c_ent_3b = cls.get_ldoce_entry(lemma) or {}
                                        if (_c_ent_3b.get("pos") or "").lower() in ("noun", "verb", "adverb"):
                                            continue
                                        seen.add(lemma)
                                        _tier3b_cands.append(lemma)
                                except Exception:
                                    pass
                    if _tier3b_cands:
                        _rel_sfx = ("ory", "ary", "ive", "al", "ic", "ar", "ular", "ental")
                        _deverbal_sfx = ("ory", "ary", "ive")
                        _t_has_rel_sfx = clean_target.endswith(_rel_sfx)
                        _t_is_deverbal_sfx = clean_target.endswith(_deverbal_sfx)
                        _tier3b_cands.sort(
                            key=lambda w: (
                                0 if cls._common_suffix_len(clean_target, w) >= 3 else 1,
                                0 if (_t_is_deverbal_sfx and w.endswith(_deverbal_sfx)) else 1,
                                0 if (_t_has_rel_sfx and w.endswith(_rel_sfx)) else 1,
                                0
                                if (
                                    w in ocd
                                    or int((cls.get_ldoce_entry(w) or {}).get("core_level") or 0) > 0
                                    or cls._lexicon_profile(w)[2] in ("A2", "B1", "B2")
                                )
                                else 1,
                            )
                        )
                        tier3_attributes.extend(_tier3b_cands)
            else:
                # Tier 2b/3r: Adverb Pertainym & Base-Adjective Bridge (`wn_pos == "r"`)
                # WordNet adverb synsets have 0 hypernyms/hyponyms; they connect to their base adjective
                # via `pertainym`/`derivation` (or regular `-ly` morphology). Traverse the base adjective's
                # antonyms and attribute/similar coordinates and project them back into attested `-ly` adverbs.
                if wn_pos == "r":
                    def _adj_to_adv_forms(adj_lem: str) -> List[str]:
                        a_l = adj_lem.strip().lower()
                        if not a_l or not a_l.isalpha():
                            return []
                        cands = [a_l + "ly"]
                        if a_l.endswith("y") and len(a_l) > 2:
                            cands.append(a_l[:-1] + "ily")
                        if a_l.endswith("le") and len(a_l) > 3:
                            cands.append(a_l[:-2] + "ly")
                        if a_l.endswith("ic"):
                            cands.append(a_l + "ally")
                        if a_l.endswith("ll"):
                            cands.append(a_l + "y")
                        return list(dict.fromkeys(cands))

                    base_adj_syns = []
                    for sn in s.senses():
                        for rel_sn in sn.get_related("pertainym") + sn.get_related("derivation"):
                            try:
                                rs = rel_sn.synset()
                                if rs and rs.pos in ("a", "s") and rs not in base_adj_syns:
                                    base_adj_syns.append(rs)
                            except Exception:
                                pass
                    if not base_adj_syns and clean_target.endswith("ly") and len(clean_target) > 4:
                        stem_cands = [clean_target[:-2], clean_target[:-2] + "e"]
                        if clean_target.endswith("ily"):
                            stem_cands.append(clean_target[:-3] + "y")
                        if clean_target.endswith("ally"):
                            stem_cands.append(clean_target[:-4])
                        for st_c in stem_cands:
                            for as_c in wn.synsets(st_c):
                                if as_c.pos in ("a", "s") and as_c not in base_adj_syns:
                                    base_adj_syns.append(as_c)
                            if base_adj_syns:
                                break
                    for b_adj in base_adj_syns[:3]:
                        # 1. Base-adjective antonyms projected to `-ly` adverbs -> `tier1_antonyms`
                        adj_ant_syns = list(b_adj.get_related("antonym"))
                        for b_sn in b_adj.senses():
                            for a_sn in b_sn.get_related("antonym"):
                                try:
                                    if a_sn.synset() and a_sn.synset() not in adj_ant_syns:
                                        adj_ant_syns.append(a_sn.synset())
                                except Exception:
                                    pass
                        if b_adj.pos == "s":
                            for h_adj in b_adj.get_related("similar"):
                                for h_sn in h_adj.senses():
                                    for a_sn in h_sn.get_related("antonym"):
                                        try:
                                            if a_sn.synset() and a_sn.synset() not in adj_ant_syns:
                                                adj_ant_syns.append(a_sn.synset())
                                        except Exception:
                                            pass
                        for ant_s in adj_ant_syns:
                            for aw in ant_s.words():
                                for adv_f in _adj_to_adv_forms(aw.lemma()):
                                    if (
                                        adv_f not in seen
                                        and adv_f not in global_synonyms
                                        and not cls.are_same_word_family(adv_f, clean_target)
                                        and (cls.is_exact_pedlex_headword(adv_f) or bool(wn.synsets(adv_f, "r")))
                                        and _pos_ok(adv_f)
                                        and cls.is_cefr_compliant_distractor(adv_f, clean_target)
                                        and not _is_informal_or_slang_distractor(adv_f)
                                    ):
                                        seen.add(adv_f)
                                        tier3_coordinates.append(adv_f)
                            for ant_sim in ant_s.get_related("similar")[:6]:
                                for asw in ant_sim.words():
                                    for adv_f in _adj_to_adv_forms(asw.lemma()):
                                        if (
                                            adv_f not in seen
                                            and adv_f not in global_synonyms
                                            and not cls.are_same_word_family(adv_f, clean_target)
                                            and (cls.is_exact_pedlex_headword(adv_f) or bool(wn.synsets(adv_f, "r")))
                                            and _pos_ok(adv_f)
                                            and cls.is_cefr_compliant_distractor(adv_f, clean_target)
                                            and not _is_informal_or_slang_distractor(adv_f)
                                        ):
                                            seen.add(adv_f)
                                            tier3_coordinates.append(adv_f)
                        # 2. Attribute coordinates projected to `-ly` adverbs -> `tier3_coordinates`
                        adj_attrs = list(b_adj.get_related("attribute"))
                        if b_adj.pos == "s":
                            for h_adj in b_adj.get_related("similar"):
                                adj_attrs.extend(h_adj.get_related("attribute"))
                        for attr_n in adj_attrs:
                            for att_adj in attr_n.get_related("attribute"):
                                if att_adj.id == b_adj.id:
                                    continue
                                for aw in att_adj.words():
                                    for adv_f in _adj_to_adv_forms(aw.lemma()):
                                        if (
                                            adv_f not in seen
                                            and adv_f not in global_synonyms
                                            and not cls.are_same_word_family(adv_f, clean_target)
                                            and (cls.is_exact_pedlex_headword(adv_f) or bool(wn.synsets(adv_f, "r")))
                                            and _pos_ok(adv_f)
                                            and cls.is_cefr_compliant_distractor(adv_f, clean_target)
                                            and not _is_informal_or_slang_distractor(adv_f)
                                        ):
                                            seen.add(adv_f)
                                            tier3_coordinates.append(adv_f)

                # Tier 2: Hyponyms (more specific concepts / troponyms)
                # Only harvest hyponyms from the locked target_synsets (already represented by `s in target_synsets`),
                # and for nouns skip hyponyms to prevent synonym-pile double-keys with semantic_fields.
                if wn_pos == "v":
                    for tok in re.findall(r"\b[a-z]{4,}\b", (s.definition() or "").lower()):
                        if tok not in cls._PARONYM_STOP_DEF_TOKS:
                            coord_domain_cues.add(tok)
                    for hypo in s.hyponyms():
                        for hw in hypo.words():
                            lemma = hw.lemma().lower()
                            if (
                                lemma not in seen
                                and lemma not in global_synonyms
                                and " " not in lemma
                                and "_" not in lemma
                                and "-" not in lemma
                                and cls.is_exact_pedlex_headword(lemma)
                                and not _is_primarily_noun_or_adj_for_verb(lemma)
                                and _pos_ok(lemma)
                                and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                and not _is_informal_or_slang_distractor(lemma)
                            ):
                                seen.add(lemma)
                                tier2_hyponyms.append(lemma)
                    # SnappyWords/WordVis Verb Script Relations: entails & causes (same-event-script actions, never synonyms)
                    for rel_syn in s.get_related("entails") + s.get_related("causes"):
                        for rw in rel_syn.words():
                            lemma = rw.lemma().lower()
                            if (
                                lemma not in seen
                                and lemma not in global_synonyms
                                and " " not in lemma
                                and "_" not in lemma
                                and "-" not in lemma
                                and cls.is_exact_pedlex_headword(lemma)
                                and not _is_primarily_noun_or_adj_for_verb(lemma)
                                and _pos_ok(lemma)
                                and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                and not _is_informal_or_slang_distractor(lemma)
                                and lemma not in ("say", "tell", "speak", "talk", "whisper", "shout", "reply", "state", "declare", "be", "have", "do", "make", "get")
                            ):
                                seen.add(lemma)
                                coord_hyper_overlap[lemma] = max(coord_hyper_overlap.get(lemma, 0), 2)
                                tier3_coordinates.append(lemma)
                # Helper to verify that a polysemous noun coordinate's top PedLex senses align with the WordNet sister branch & countability
                def _is_aligned_noun_coord(cand_lem: str, sis_syn: Optional[Any] = None, hyper_syn: Optional[Any] = None) -> bool:
                    if cls._is_collocational_partner(clean_target, cand_lem):
                        return False
                    c_ent_n = cls.get_ldoce_entry(cand_lem) or {}
                    c_senses_n = [
                        sn for sn in (c_ent_n.get("senses") or [])
                        if not sn.get("pos") or cls._pos_code(sn.get("pos")) == "n"
                    ] or (c_ent_n.get("senses") or [])
                    if not c_senses_n:
                        return True
                    # Countability & Number Symmetry Gate for noun coordinates:
                    # 1. Reject countable artifact homonyms (e.g. 'diaper', 'sheet', 'till', 'sliver', 'nap', 'duck')
                    #    whose top 2 PedLex noun senses do not attest [uncountable] when target is strictly uncountable.
                    # 2. Symmetric: Reject strictly [uncountable] mass nouns (e.g. 'adrenalin', 'afterbirth')
                    #    when target is strictly [countable].
                    # 3. Reject pluralia tantum [plural] nouns (e.g. 'banns') when clean_target is singular.
                    if target_strictly_uncountable:
                        top2_grams = " ".join(str(sn.get("grammar") or "").lower() for sn in c_senses_n[:2])
                        if "uncountable" not in top2_grams:
                            return False
                    elif target_strictly_countable:
                        top2_grams = " ".join(str(sn.get("grammar") or "").lower() for sn in c_senses_n[:2])
                        if "uncountable" in top2_grams and "countable" not in re.sub(r"\buncountable\b", "", top2_grams):
                            return False
                    if not clean_target.endswith("s") and cand_lem.endswith("s"):
                        g0_low = str(c_senses_n[0].get("grammar") or "").lower()
                        if "plural" in g0_low and "countable" not in re.sub(r"\buncountable\b", "", g0_low):
                            return False
                    c_wn_n = [cs for cs in wn.synsets(cand_lem, "n") if cs.lexfile()] or wn.synsets(cand_lem, "n")
                    if c_wn_n:
                        if s.lexfile() == "noun.person" and c_wn_n[0].lexfile() != "noun.person":
                            return False
                        if s.lexfile() and s.lexfile() != "noun.person" and c_wn_n[0].lexfile() == "noun.person":
                            return False
                        # Reject biological/anatomical coordinates (e.g. 'palate' under 'surface') when target is non-biological
                        _bio_lfs = ("noun.body", "noun.animal", "noun.plant")
                        if (
                            s.lexfile()
                            and s.lexfile() not in _bio_lfs
                            and not locals().get("is_bio_sample_def")
                            and (
                                c_wn_n[0].lexfile() in _bio_lfs
                                or (sis_syn is not None and sis_syn.lexfile() in _bio_lfs)
                            )
                        ):
                            return False
                        # Reject metonymic building/container/artifact coordinates (e.g. 'room', 'suite', 'house', 'hall', 'floor')
                        # whose primary WordNet sense is concrete physical/location when target is a `noun.group` collective noun.
                        if s.lexfile() == "noun.group" and c_wn_n[0].lexfile() in (
                            "noun.artifact", "noun.location", "noun.body",
                            "noun.substance", "noun.object", "noun.food", "noun.plant",
                        ):
                            return False
                    # Monosemous in PedLex and sharing primary WordNet lexfile -> aligned
                    # (Exclude broad `noun.person` so kinship/age terms like 'aunt'/'baby' must share domain cues with occupation targets.)
                    if (
                        len(c_senses_n) == 1
                        and s.lexfile()
                        and s.lexfile() != "noun.person"
                        and c_wn_n
                        and c_wn_n[0].lexfile() == s.lexfile()
                        and (sis_syn is None or not sis_syn.lexfile() or sis_syn.lexfile() == s.lexfile())
                    ):
                        return True
                    if sis_syn is None or hyper_syn is None:
                        return True
                    _meta_stop = {
                        "someone", "somebody", "anyone", "anybody", "something", "anything",
                        "person", "people", "thing", "things", "place", "time", "part", "type",
                        "kind", "form", "amount", "number", "small", "large", "used", "make",
                        "made", "making", "taken", "order", "particular", "usually", "often",
                    }
                    # Check if top-2 PedLex senses share domain tokens (with plural -s normalization) with sister synset / hypernym / target definition
                    prim_txt = re.sub(
                        r"\bthe\s+earth\b",
                        "",
                        " ".join(
                            " ".join([
                                str(sn.get("definition") or ""),
                                str(sn.get("signpost") or ""),
                                " ".join(sn.get("units") or []),
                            ])
                            for sn in c_senses_n[:2]
                        ).lower(),
                    )
                    raw_prim_toks = {
                        tok for tok in re.findall(r"\b[a-z]{3,}\b", prim_txt)
                        if tok not in cls._PARONYM_STOP_DEF_TOKS and tok not in cls._ANCHOR_STOPWORDS and tok not in _meta_stop and tok != cand_lem
                    }
                    prim_toks = set(raw_prim_toks) | {
                        tok[:-1] for tok in raw_prim_toks if tok.endswith("s") and len(tok) > 3
                    }
                    branch_txt = " ".join([
                        str(sis_syn.definition() or ""),
                        " ".join(sis_syn.examples() or []),
                        str(hyper_syn.definition() or ""),
                        " ".join(hw.lemma() for hw in hyper_syn.words()),
                        " ".join(ghw.lemma() for gh in hyper_syn.hypernyms() for ghw in gh.words()),
                    ]).lower()
                    raw_branch_toks = {
                        tok for tok in re.findall(r"\b[a-z]{3,}\b", branch_txt)
                        if tok not in cls._PARONYM_STOP_DEF_TOKS and tok not in cls._ANCHOR_STOPWORDS and tok not in _meta_stop and tok != cand_lem
                    } | ((coord_domain_cues - _meta_stop) - {cand_lem})
                    if definition:
                        raw_branch_toks |= ((def_tokens - _meta_stop) - {cand_lem})
                    branch_toks = set(raw_branch_toks) | {
                        tok[:-1] for tok in raw_branch_toks if tok.endswith("s") and len(tok) > 3
                    }
                    return bool(prim_toks & branch_toks)

                # Tier 3: Coordinate terms (sisters under same hypernym - Troponyms for verbs)
                for hyper in s.hypernyms():
                    # Allow up to 200 hyponyms for specific mid/deep-level noun hypernyms (min_depth >= 4, e.g. container=166, fabric=176, metallic element=78)
                    h_depth = hyper.min_depth() if hasattr(hyper, "min_depth") else 0
                    if wn_pos == "n" and h_depth >= 4:
                        max_hypos = 200
                    else:
                        max_hypos = 85 if (len(s.hypernyms()) == 1 and not tier3_coordinates) else 50
                    if len(hyper.hyponyms()) > max_hypos:
                        continue
                    for hw in hyper.words():
                        for tok in re.findall(r"\b[a-z]{3,}\b", hw.lemma().lower()):
                            if tok not in cls._PARONYM_STOP_DEF_TOKS:
                                coord_domain_cues.add(tok)
                    for tok in re.findall(r"\b[a-z]{4,}\b", (hyper.definition() or "").lower()):
                        if tok not in cls._PARONYM_STOP_DEF_TOKS:
                            coord_domain_cues.add(tok)
                    for sis_c in hyper.hyponyms():
                        for sw_c in sis_c.words():
                            lem_c = sw_c.lemma().lower()
                            if lem_c.isalpha() and len(lem_c) >= 4:
                                coord_domain_cues.add(lem_c)
                    h_ov = 0
                    if definition:
                        h_txt = (hyper.definition() + " " + " ".join(hw.lemma() for hw in hyper.words())).lower()
                        h_ov = sum(1 for dt in def_tokens if dt in h_txt) * 2 + 1
                    sis_pool = list(hyper.hyponyms())
                    for sis in list(sis_pool):
                        # Unwrap multi-word taxonomic intermediate nodes (e.g. 'natural fiber', 'precious metal')
                        if (
                            wn_pos == "n"
                            and sis.id != s.id
                            and not any(" " not in sw.lemma() and "_" not in sw.lemma() and "-" not in sw.lemma() for sw in sis.words())
                            and 0 < len(sis.hyponyms()) <= 25
                        ):
                            sis_pool.extend(sis.hyponyms())
                    for sis in sis_pool:
                        if sis.id == s.id:
                            continue
                        for sw in sis.words():
                            raw_lem = sw.lemma()
                            if wn_pos == "n" and (not raw_lem.islower() or len(raw_lem) < 3):
                                continue
                            lemma = raw_lem.lower()
                            if (
                                lemma not in seen
                                and lemma not in global_synonyms
                                and " " not in lemma
                                and "_" not in lemma
                                and "-" not in lemma
                                and (lemma in ocd or bool(cls.get_ldoce_entry(lemma)))
                                and _pos_ok(lemma)
                                and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                and not _is_informal_or_slang_distractor(lemma)
                            ):
                                if wn_pos == "n" and (_is_primarily_adj_or_verb(lemma) or not cls.is_exact_pedlex_headword(lemma)):
                                    continue
                                if wn_pos == "n" and s.lexfile() == "noun.person":
                                    c_syns_p = wn.synsets(lemma, "n")
                                    if not c_syns_p or c_syns_p[0].lexfile() != "noun.person":
                                        continue
                                if wn_pos == "n" and not _is_aligned_noun_coord(lemma, sis, hyper):
                                    continue
                                if wn_pos == "v" and (_is_primarily_noun_or_adj_for_verb(lemma) or not cls.is_exact_pedlex_headword(lemma)):
                                    continue
                                # Verbs: Block speech/reporting verbs with clausal or double-object valency that break transitive substitution
                                if wn_pos == "v" and lemma in ("say", "tell", "speak", "talk", "whisper", "shout", "reply", "state", "declare"):
                                    continue
                                seen.add(lemma)
                                coord_hyper_overlap[lemma] = max(coord_hyper_overlap.get(lemma, 0), h_ov)
                                tier3_coordinates.append(lemma)
                # Tier 3.2: Co-Meronyms (fellow parts/members under the same holonym, e.g. petal -> sepal/stamen/pistil, roof -> wall/floor)
                if wn_pos == "n" and len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count * 2:
                    holo_pool = list(s.holonyms()) if hasattr(s, "holonyms") else (s.get_related("holonym_part") + s.get_related("holonym_member"))
                    for holo in holo_pool:
                        mero_list = list(holo.meronyms()) if hasattr(holo, "meronyms") else (holo.get_related("meronym_part") + holo.get_related("meronym_member"))
                        if not (1 < len(mero_list) <= 35):
                            continue
                        for hw in holo.words():
                            for tok in re.findall(r"\b[a-z]{3,}\b", hw.lemma().lower()):
                                if tok not in cls._PARONYM_STOP_DEF_TOKS:
                                    coord_domain_cues.add(tok)
                        for tok in re.findall(r"\b[a-z]{4,}\b", (holo.definition() or "").lower()):
                            if tok not in cls._PARONYM_STOP_DEF_TOKS:
                                coord_domain_cues.add(tok)
                        for co_mero in mero_list:
                            if co_mero.id == s.id:
                                continue
                            for mw in co_mero.words():
                                lemma = mw.lemma().lower()
                                if (
                                    lemma not in seen
                                    and lemma not in global_synonyms
                                    and " " not in lemma
                                    and "_" not in lemma
                                    and "-" not in lemma
                                    and cls.is_exact_pedlex_headword(lemma)
                                    and not _is_primarily_adj_or_verb(lemma)
                                    and _pos_ok(lemma)
                                    and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                    and not _is_informal_or_slang_distractor(lemma)
                                ):
                                    seen.add(lemma)
                                    coord_hyper_overlap[lemma] = max(coord_hyper_overlap.get(lemma, 0), 1)
                                    tier3_coordinates.append(lemma)
                # Tier 3.5: Grand-hypernym coordinate terms (Grand-sisters under same grandparent)
                # First Defense: Extends candidate pool for nouns when immediate coordinates are sparse,
                # strictly constrained to the same WordNet lexfile and non-mega-bucket grandparents.
                if wn_pos == "n" and len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count * 2:
                    target_lf = s.lexfile()
                    for hyper in s.hypernyms():
                        # If direct coordinates are sparse (< target_count) and the direct hypernym is a specific
                        # concrete domain node (depth >= 5, e.g. ['soil', 'dirt'] for 'clay'), include non-synonym hypernym lemmas
                        if (
                            len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count
                            and hasattr(hyper, "min_depth")
                            and hyper.min_depth() >= 5
                        ):
                            for hw in hyper.words():
                                h_lem = hw.lemma().lower()
                                if (
                                    h_lem.isalpha()
                                    and len(h_lem) >= 3
                                    and h_lem not in seen
                                    and h_lem not in global_synonyms
                                    and cls.is_exact_pedlex_headword(h_lem)
                                    and not _is_primarily_adj_or_verb(h_lem)
                                    and _pos_ok(h_lem)
                                    and cls.is_cefr_compliant_distractor(h_lem, clean_target)
                                    and not _is_informal_or_slang_distractor(h_lem)
                                    and _is_aligned_noun_coord(h_lem, hyper, hyper)
                                ):
                                    seen.add(h_lem)
                                    coord_hyper_overlap[h_lem] = max(coord_hyper_overlap.get(h_lem, 0), 1)
                                    tier3_coordinates.append(h_lem)
                        for g_hyper in hyper.hypernyms():
                            # Block upper-ontology root grand-hypernyms (depth <= 3 for substances so 'fuel' at depth 4 is kept; depth <= 4 otherwise)
                            min_g_depth = 3 if target_lf == "noun.substance" else 4
                            if hasattr(g_hyper, "min_depth") and g_hyper.min_depth() <= min_g_depth:
                                continue
                            max_g_hypos = 65 if not tier3_coordinates else 35
                            if len(g_hyper.hyponyms()) > max_g_hypos:
                                continue
                            if (
                                len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count
                                and hasattr(g_hyper, "min_depth")
                                and g_hyper.min_depth() >= 5
                                and (not target_lf or g_hyper.lexfile() == target_lf)
                            ):
                                for ghw in g_hyper.words():
                                    gh_lem = ghw.lemma().lower()
                                    if (
                                        gh_lem.isalpha()
                                        and len(gh_lem) >= 3
                                        and gh_lem not in seen
                                        and gh_lem not in global_synonyms
                                        and cls.is_exact_pedlex_headword(gh_lem)
                                        and not _is_primarily_adj_or_verb(gh_lem)
                                        and _pos_ok(gh_lem)
                                        and cls.is_cefr_compliant_distractor(gh_lem, clean_target)
                                        and not _is_informal_or_slang_distractor(gh_lem)
                                        and _is_aligned_noun_coord(gh_lem, g_hyper, g_hyper)
                                    ):
                                        seen.add(gh_lem)
                                        coord_hyper_overlap[gh_lem] = max(coord_hyper_overlap.get(gh_lem, 0), 1)
                                        tier3_coordinates.append(gh_lem)
                            g_sis_pool = list(g_hyper.hyponyms())
                            for g_sis in list(g_sis_pool):
                                # Unwrap multi-word taxonomic intermediate nodes (e.g. 'paper money' -> 'bill', 'animal fiber' -> 'silk', 'wool')
                                if (
                                    g_sis.id != hyper.id
                                    and not any(" " not in sw.lemma() and "_" not in sw.lemma() and "-" not in sw.lemma() for sw in g_sis.words())
                                    and 0 < len(g_sis.hyponyms()) <= 25
                                ):
                                    g_sis_pool.extend(g_sis.hyponyms())
                            for sis in g_sis_pool:
                                if target_lf and sis.lexfile() != target_lf:
                                    continue
                                for sw in sis.words():
                                    raw_lem = sw.lemma()
                                    if not raw_lem.islower() or len(raw_lem) < 3:
                                        continue
                                    lemma = raw_lem.lower()
                                    if (
                                        lemma not in seen
                                        and lemma not in global_synonyms
                                        and " " not in lemma
                                        and "_" not in lemma
                                        and "-" not in lemma
                                        and (lemma in ocd or bool(cls.get_ldoce_entry(lemma)))
                                        and _pos_ok(lemma)
                                        and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                        and not _is_informal_or_slang_distractor(lemma)
                                    ):
                                        if _is_primarily_adj_or_verb(lemma) or not cls.is_exact_pedlex_headword(lemma):
                                            continue
                                        if s.lexfile() == "noun.person":
                                            c_syns_p = wn.synsets(lemma, "n")
                                            if not c_syns_p or c_syns_p[0].lexfile() != "noun.person":
                                                continue
                                        if not _is_aligned_noun_coord(lemma, sis, g_hyper):
                                            continue
                                        seen.add(lemma)
                                        coord_hyper_overlap.setdefault(lemma, 0)
                                        tier3_coordinates.append(lemma)
                # Tier 3.2b: Aristotelian Genus-Noun Taxonomic Bridge
                # When a noun's WordNet hypernym branch is sparse (< target_count) and its dictionary definition
                # follows the Aristotelian "a/an [modifier]* <genus_noun> in/of/about/..." structure (e.g., obituary ->
                # "an article in a newspaper..."), harvest direct single-word co-hyponyms of the genus noun in the same lexfile.
                if wn_pos == "n" and definition and len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count:
                    _m_genus = re.match(
                        r"^(?:(?:\[[^\]]+\]\s*)+)?(?:a|an|the)\s+"
                        r"(?:(?:small|large|short|long|brief|formal|official|public|special|particular|printed|written|spoken|musical|medical|scientific|legal|political|religious|traditional|modern|simple|basic)\s+)*"
                        r"([a-z]{4,})\s+(?:in|of|about|for|that|which|who|where|on|from|to|with|by|used)\b",
                        definition.strip().lower(),
                    )
                    if _m_genus:
                        _g_noun = _m_genus.group(1)
                        if _g_noun not in cls._PARONYM_STOP_DEF_TOKS and _g_noun not in cls._ANCHOR_STOPWORDS and _g_noun != clean_target:
                            _g_cands = []
                            for _g_idx, _g_syn in enumerate(wn.synsets(_g_noun, "n")):
                                if not _g_syn.lexfile():
                                    continue
                                if target_lfs_set and _g_syn.lexfile() not in target_lfs_set:
                                    continue
                                if hasattr(_g_syn, "min_depth") and _g_syn.min_depth() < 4:
                                    continue
                                _g_hypos = _g_syn.hyponyms()
                                if not (0 < len(_g_hypos) <= 50):
                                    continue
                                _g_def_toks = set(re.findall(r"\b[a-z]{4,}\b", (_g_syn.definition() or "").lower()))
                                _g_ov = len(_g_def_toks & def_tokens)
                                _g_cands.append((-_g_ov, _g_idx, _g_syn))
                            _g_cands.sort(key=lambda x: (x[0], x[1]))
                            for _, _, _g_syn in _g_cands[:1]:
                                for _gsis in _g_syn.hyponyms():
                                    if _gsis.id == s.id or (target_lfs_set and _gsis.lexfile() not in target_lfs_set):
                                        continue
                                    for _gsw in _gsis.words():
                                        _raw_glem = _gsw.lemma()
                                        if not _raw_glem.islower() or len(_raw_glem) < 3:
                                            continue
                                        _glem = _raw_glem.lower()
                                        if (
                                            _glem not in seen
                                            and _glem not in global_synonyms
                                            and " " not in _glem
                                            and "_" not in _glem
                                            and "-" not in _glem
                                            and cls.is_exact_pedlex_headword(_glem)
                                            and not _is_primarily_adj_or_verb(_glem)
                                            and _pos_ok(_glem)
                                            and cls.is_cefr_compliant_distractor(_glem, clean_target)
                                            and not _is_informal_or_slang_distractor(_glem)
                                            and _is_aligned_noun_coord(_glem, _gsis, _g_syn)
                                        ):
                                            seen.add(_glem)
                                            coord_hyper_overlap[_glem] = max(coord_hyper_overlap.get(_glem, 0), 1)
                                            tier3_coordinates.append(_glem)
                # Tier 3.3: PedLex `ldoce_sense_cue_index` Scenario/Material Co-Hyponym Tier
                # When a concrete/scenario noun sits under a singleton or sparse WordNet branch (e.g. 'stamp' under 'postage stamp',
                # 'ticket' under 'commercial document'), query `ldoce_sense_cue_index` for primary-sense nouns sharing >= 2 domain cues.
                _is_concrete_t = wn_pos == "n" and s.lexfile() in (
                    "noun.artifact", "noun.substance", "noun.communication",
                    "noun.possession", "noun.plant", "noun.food", "noun.object",
                )
                if (
                    (_is_concrete_t and len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count * 2)
                    or (wn_pos == "n" and len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count)
                ):
                    conn_cue = cls._ldoce_conn if cls._ensure_ldoce_conn() else None
                    if conn_cue is not None:
                        _cue_stop = {
                            "piece", "small", "large", "something", "someone", "used", "make", "made", "making",
                            "type", "kind", "form", "part", "place", "time", "person", "people", "thing",
                            "things", "usually", "often", "especially", "particular", "official", "amount",
                            "number", "group", "system", "area", "side", "front", "back", "surface", "edge",
                            "keep", "put", "take", "give", "show", "shows", "have", "having", "with", "from",
                            "into", "onto", "over", "under", "between", "through", "before", "after", "when",
                            "where", "which", "that", "their", "your", "other", "another", "many", "more",
                            "most", "some", "such", "very", "quite", "rather", "able", "good", "well", "high",
                            "long", "short", "thin", "thick", "flat", "round", "hard", "soft", "dark", "light",
                            "white", "black", "brown", "yellow", "green", "blue", "colour", "color", "shaped",
                            "you", "and", "the", "for", "out", "can", "has", "been", "are", "was", "were",
                            "his", "her", "its", "our", "who", "what", "how", "all", "any", "one", "two",
                            "not", "but", "second", "first", "last", "week", "year", "day", "open", "like", "got",
                            "about", "just", "recently", "already", "still", "even", "also", "only", "always", "never",
                            "taken", "tested", "examined", "order",
                        }
                        _t_cues: List[str] = []
                        if definition:
                            for _dt in def_tokens:
                                if _dt not in _cue_stop and len(_dt) >= 3 and _dt != clean_target:
                                    _t_cues.append(_dt)
                            if pedlex_signpost and pedlex_signpost not in _cue_stop:
                                _t_cues.append(pedlex_signpost)
                            for _pu in pedlex_units:
                                for _ptok in re.findall(r"\b[a-z]{3,}\b", _pu):
                                    if _ptok not in _cue_stop and _ptok != clean_target:
                                        _t_cues.append(_ptok)
                        for _wt in re.findall(r"\b[a-z]{4,}\b", (s.definition() or "").lower()):
                            if _wt not in _cue_stop and _wt != clean_target:
                                _t_cues.append(_wt)
                                coord_domain_cues.add(_wt)
                        try:
                            for ( _cw, ) in conn_cue.execute(
                                "SELECT cue_word FROM ldoce_sense_cue_index WHERE host_word = ? AND sense_idx = 0",
                                (clean_target,),
                            ).fetchall():
                                if _cw not in _cue_stop and len(_cw) >= 3 and _cw != clean_target:
                                    _t_cues.append(_cw)
                                    coord_domain_cues.add(_cw)
                        except Exception:
                            pass
                        _t_cues = list(dict.fromkeys(_t_cues))
                        if len(_t_cues) >= 2:
                            try:
                                _q_sql = (
                                    f"SELECT host_word, COUNT(DISTINCT cue_word) as cnt, SUM(weight) as wsum "
                                    f"FROM ldoce_sense_cue_index "
                                    f"WHERE sense_idx = 0 AND cue_word IN ({','.join(['?'] * len(_t_cues))}) "
                                    f"AND host_word != ? AND host_word NOT LIKE '% %' AND host_word NOT LIKE '%-%' "
                                    f"GROUP BY host_word HAVING cnt >= 2 "
                                    f"ORDER BY cnt DESC, wsum DESC LIMIT 35"
                                )
                                _cue_rows = conn_cue.execute(_q_sql, _t_cues + [clean_target]).fetchall()
                            except Exception:
                                _cue_rows = []
                            _allowed_cue_lfs = set(target_lfs_set)
                            if _allowed_cue_lfs & {"noun.communication", "noun.artifact", "noun.possession", "noun.group"}:
                                _allowed_cue_lfs |= {"noun.communication", "noun.artifact", "noun.possession"}
                            if _allowed_cue_lfs & {"noun.state", "noun.feeling", "noun.attribute", "noun.act", "noun.event"}:
                                _allowed_cue_lfs |= {"noun.state", "noun.feeling", "noun.attribute", "noun.act", "noun.event", "noun.cognition"}
                            _wn_coords_enough = len(tier3_coordinates) >= target_count
                            for _hw, _cnt, _wsum in _cue_rows:
                                _lem = str(_hw or "").strip().lower()
                                if (
                                    not _lem.isalpha()
                                    or len(_lem) < 3
                                    or _lem in seen
                                    or _lem in global_synonyms
                                    or (_is_concrete_t and _lem.endswith(("ing", "tion", "sion", "ment", "ness", "ity")))
                                    or not cls.is_exact_pedlex_headword(_lem)
                                    or _is_primarily_adj_or_verb(_lem)
                                    or not _pos_ok(_lem)
                                    or not cls.is_cefr_compliant_distractor(_lem, clean_target)
                                    or _is_informal_or_slang_distractor(_lem)
                                    or not _is_aligned_noun_coord(_lem)
                                ):
                                    continue
                                _h_ent = cls.get_ldoce_entry(_lem) or {}
                                if cls._pos_code(_h_ent.get("pos") or "") != "n":
                                    continue
                                if (
                                    _lem not in ocd
                                    and int(_h_ent.get("core_level") or 0) == 0
                                    and not any(sn.get("examples") for sn in (_h_ent.get("senses") or []))
                                ):
                                    continue
                                _c_syns_cue = wn.synsets(_lem, "n")
                                if s.lexfile() == "noun.person":
                                    if not _c_syns_cue or _c_syns_cue[0].lexfile() != "noun.person":
                                        continue
                                elif _c_syns_cue and _c_syns_cue[0].lexfile() == "noun.person":
                                    continue
                                if _allowed_cue_lfs and (
                                    not _c_syns_cue or _c_syns_cue[0].lexfile() not in _allowed_cue_lfs
                                ):
                                    continue
                                seen.add(_lem)
                                if _wn_coords_enough:
                                    coord_hyper_overlap.setdefault(_lem, 0)
                                else:
                                    coord_hyper_overlap[_lem] = max(coord_hyper_overlap.get(_lem, 0), int(_cnt))
                                tier3_coordinates.append(_lem)

        # Tier 3.35: Pustejovsky Qualia Scenario Lifecycle Verb Bridge (for sparse WordNet verb branches like `enroll`)
        if (
            wn_pos == "v"
            and definition
            and len(tier0_ldoce_thesaurus) + len(tier2_hyponyms) + len(tier3_coordinates) < target_count * 2
        ):
            _qualia_peers = cls._get_verb_qualia_scenario_peers(
                clean_target,
                definition=definition,
                quote=quote,
                context_anchor=context_anchor,
                anchor_type=anchor_type,
            )
            for _qv, _qinfo in sorted(_qualia_peers.items(), key=lambda kv: (-kv[1]["score"], kv[0])):
                if (
                    _qv in seen
                    or not _qv.isalpha()
                    or not cls.is_exact_pedlex_headword(_qv)
                    or _is_primarily_noun_or_adj_for_verb(_qv)
                    or not _pos_ok(_qv)
                    or not cls.is_cefr_compliant_distractor(_qv, clean_target)
                    or _is_informal_or_slang_distractor(_qv)
                ):
                    continue
                if _qv in global_synonyms:
                    _has_slot_clash, _ = cls.evaluate_syntactic_slot_clash(
                        clean_target,
                        _qv,
                        anchor=context_anchor,
                        anchor_type=anchor_type,
                        pos=pos,
                        quote=quote,
                    )
                    if not _has_slot_clash:
                        continue
                seen.add(_qv)
                coord_hyper_overlap[_qv] = max(coord_hyper_overlap.get(_qv, 0), int(_qinfo["score"]))
                tier3_coordinates.append(_qv)

        # Tier 3.4: Orthogonal-Domain Collective Noun Peer Pool (`_COLLECTIVE_NOUN_ONTOLOGY`)
        # For collective nouns (human role collectives, animal collectives, vessel/batch collectives),
        # ban co-synonyms in the same subgroup and inject orthogonal subgroup heads from the same macro class.
        _is_collective_noun_target = False
        if wn_pos == "n":
            _coll_macro: Optional[str] = None
            _coll_own_sub: Tuple[str, ...] = ()
            for _m_cat, _sub_list in cls._COLLECTIVE_NOUN_ONTOLOGY.items():
                for _s_grp in _sub_list:
                    if clean_target in _s_grp:
                        _coll_macro = _m_cat
                        _coll_own_sub = _s_grp
                        break
                if _coll_macro:
                    break
            if _coll_macro:
                _def_low_coll = (definition or "").strip().lower()
                _non_coll_prefixes = (
                    "a natural ability", "a long thick stick", "a hard cover", "a flat piece",
                    "a small paper", "a large bag", "a thin flat", "a narrow piece", "a mould", "a mold",
                )
                _coll_def_cues = {
                    "people", "person", "workers", "employees", "teachers", "actors", "performers",
                    "musicians", "singers", "sailors", "officers", "soldiers", "customers", "members",
                    "priests", "leaders", "group", "crowd", "organization", "department", "university",
                    "college", "school", "ship", "ships", "plane", "aircraft", "play", "film", "movie",
                    "theatre", "theater", "court", "judge", "decide", "birds", "sheep", "goats", "animals",
                    "cows", "cattle", "wolves", "dogs", "insects", "bees", "ants", "fish", "chickens",
                    "hens", "vehicles", "boats", "things", "items", "objects", "flowers", "keys",
                }
                if _def_low_coll:
                    _def_coll_toks = set(re.findall(r"\b[a-z]{3,}\b", _def_low_coll))
                    _is_collective_noun_target = bool(_def_coll_toks & _coll_def_cues) and not _def_low_coll.startswith(_non_coll_prefixes)
                else:
                    _is_collective_noun_target = bool(not target_synsets or target_synsets[0].lexfile() == "noun.group")
                if _is_collective_noun_target:
                    global_synonyms.update(_coll_own_sub)
                    # Remove orthogonal collective-noun subgroup members that Oxford Writer's Thesaurus
                    # or broad secondary WordNet synsets (e.g. `staff`/`faculty`) lumped into `global_synonyms`.
                    for _other_grp in cls._COLLECTIVE_NOUN_ONTOLOGY[_coll_macro]:
                        if _other_grp == _coll_own_sub:
                            continue
                        for _orth_w in _other_grp:
                            global_synonyms.discard(_orth_w)
                    tier1_synonyms = [w for w in tier1_synonyms if w not in global_synonyms]
                    tier3_coordinates = [w for w in tier3_coordinates if w not in global_synonyms]
                    for _grp_rank, _other_grp in enumerate(cls._COLLECTIVE_NOUN_ONTOLOGY[_coll_macro]):
                        if _other_grp == _coll_own_sub:
                            continue
                        for _peer_w in _other_grp:
                            if (
                                _peer_w != clean_target
                                and _peer_w not in global_synonyms
                                and _peer_w not in exclude_words
                                and cls.is_cefr_compliant_distractor(_peer_w, clean_target)
                            ):
                                if _peer_w not in seen:
                                    seen.add(_peer_w)
                                    tier3_coordinates.append(_peer_w)
                                coord_hyper_overlap[_peer_w] = max(coord_hyper_overlap.get(_peer_w, 0), 35 - _grp_rank)
                                break

        # Sort tier3_coordinates by hypernym domain overlap, primary-sense domain alignment, PedLex richness, and core B1/B2 CEFR proximity
        # (cap target reference rank at B2=4 so C2-rated targets like 'slump' or 'motel' do not prefer obscure C2 words over B1/B2 peers)
        if tier2_hyponyms or tier3_coordinates or wn_pos == "n":
            t_lvl_str = cls._lexicon_profile(clean_target)[2] or "B1"
            t_rank = min(cls.CEFR_ORDER.get(t_lvl_str, 3), 4)
            analyzer = cls.get_cefr_analyzer()
            t_float = analyzer.get_average_word_level_float(clean_target) if hasattr(analyzer, "get_average_word_level_float") else float(t_rank)
            if t_float is None:
                t_float = float(t_rank)
            p_code_coord = {"n": "n", "v": "v", "a": "adj", "r": "adv"}.get(wn_pos or "n", "n")

            def _coord_sort_key(cand_w: str) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
                c_ent = cls.get_ldoce_entry(cand_w) or {}
                c_senses = c_ent.get("senses") or []
                pos_senses = [sn for sn in c_senses if not sn.get("pos") or cls._pos_code(sn.get("pos")) == p_code_coord] or c_senses
                has_ex = any(sn.get("examples") for sn in pos_senses)
                countability_mismatch = 0
                if wn_pos == "n" and pos_senses:
                    c_g0 = str(pos_senses[0].get("grammar") or "").lower()
                    if target_strictly_countable and "uncountable" in c_g0 and "countable" not in re.sub(r"\buncountable\b", "", c_g0):
                        countability_mismatch = 1
                    elif target_strictly_uncountable and "uncountable" not in c_g0:
                        countability_mismatch = 1
                prim_aligned = False
                if pos_senses and coord_domain_cues:
                    prim_def = str(pos_senses[0].get("definition") or "").lower()
                    prim_toks = set(re.findall(r"\b[a-z]{3,}\b", prim_def)) - {cand_w}
                    if prim_toks & (coord_domain_cues - {cand_w}) and len(prim_toks) >= 2:
                        prim_aligned = True
                if not prim_aligned and not pos_senses and target_lfs_set:
                    c_wn = wn.synsets(cand_w, wn_pos)
                    if c_wn and c_wn[0].lexfile() in target_lfs_set:
                        prim_aligned = True
                c_lvl_raw = cls._lexicon_profile(cand_w)[2]
                c_lvl = c_lvl_raw or "C1"
                c_rank = cls.CEFR_ORDER.get(c_lvl, 5)
                c_float = analyzer.get_average_word_level_float(cand_w) if hasattr(analyzer, "get_average_word_level_float") else float(c_rank)
                if c_float is None:
                    c_float = float(c_rank)
                diff_overshoot = 1 if (c_float > t_float + 0.6) else 0
                has_ldoce_core = 0 if int(c_ent.get("core_level") or 0) > 0 else 1
                is_non_core = 0 if (cand_w in ocd or has_ldoce_core == 0 or (c_lvl_raw in ("A1", "A2", "B1", "B2") and diff_overshoot == 0)) else 1
                gerund_mismatch = 1 if (wn_pos == "n" and cand_w.endswith("ing") and not clean_target.endswith("ing")) else 0
                return (
                    gerund_mismatch,
                    countability_mismatch,
                    diff_overshoot,
                    has_ldoce_core,
                    is_non_core,
                    -coord_hyper_overlap.get(cand_w, 0),
                    0 if has_ex else 1,
                    0 if prim_aligned else 1,
                    abs(c_rank - t_rank),
                    1 if c_rank >= 6 else 0,
                )
            if tier2_hyponyms:
                tier2_hyponyms.sort(key=_coord_sort_key)
            if tier3_coordinates:
                tier3_coordinates.sort(key=_coord_sort_key)

        # Candidate prioritization: PedLex Thesaurus & Activators are primary (Tier 0); WordNet is secondary/fallback
        if wn_pos == "v":
            antonym_candidates = [
                ant for ant in sense_antonyms
                if ant != clean_target and cls.is_cefr_compliant_distractor(ant, clean_target)
            ]
            candidates = tier0_ldoce_thesaurus + tier2_hyponyms + tier3_coordinates + antonym_candidates
        elif wn_pos == "a":
            # For adjectives: retain satellite_synonyms first to satisfy existing fine-grained satellite tests,
            # then inject PedLex thesaurus peers, compliant antonyms, followed by attribute coordinates.
            antonym_candidates = [
                ant for ant in sense_antonyms
                if ant != clean_target and cls.is_cefr_compliant_distractor(ant, clean_target)
            ]
            candidates = tier2_satellites + tier0_ldoce_thesaurus + antonym_candidates + tier3_attributes + tier1_synonyms
            # Preposition valency cloze (): prioritize academic adjectives governing DIFFERENT prepositions
            if anchor_type == "prep" and context_anchor:
                clean_anchor = context_anchor.strip().lower()
                valency_candidates = []
                for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items():
                    if p != clean_anchor:
                        for a in adjs:
                            if a != clean_target and a not in exclude_words and cls.is_cefr_compliant_distractor(a, clean_target):
                                valency_candidates.append(a)
                # Prioritize local semantic peers (satellites, PedLex thesaurus) for domain relevance,
                # then complement with academic adjectives governing distinct prepositions
                candidates = tier2_satellites + tier0_ldoce_thesaurus + antonym_candidates + valency_candidates + tier3_attributes + tier1_synonyms
        elif wn_pos == "n":
            extra_noun_ants = [
                a for a in sense_antonyms[1:]
                if cls.is_cefr_compliant_distractor(a, clean_target) and not _is_informal_or_slang_distractor(a)
            ]
            if _is_collective_noun_target:
                candidates = tier3_coordinates + tier0_ldoce_thesaurus + tier2_hyponyms + tier1_synonyms + extra_noun_ants
            else:
                candidates = tier0_ldoce_thesaurus + tier1_synonyms + tier2_hyponyms + tier3_coordinates + extra_noun_ants
        else:
            candidates = tier0_ldoce_thesaurus + tier1_synonyms + tier2_hyponyms + tier3_coordinates

        # 3. Oxford Collision Clearance (Anti Double-Key Gate - Schemes 3 & 4)
        # Exempt candidates that have a verified syntactic slot clash with target's slot
        # or are anchor-discriminated Tier 0 PedLex Thesaurus peers from global_synonyms ban.
        tier0_set = set(tier0_ldoce_thesaurus)
        forbidden_words = {
            gw for gw in global_synonyms
            if gw not in tier0_set
            and not cls.evaluate_syntactic_slot_clash(
                clean_target, gw, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
            )[0]
        } | {clean_target}

        # 3b. Unified Eight Construction Families Subcategorization Gate (`pedlex_cxn_index`):
        # When the target word in context instantiates one of the 8 lexico-syntactic construction families,
        # prioritize subcategorization-contrasting distractors (`contrast_groups \ ban_groups`) and ban
        # same-construction synonyms/antonyms (`ban_groups`) that also fit the grammatical slot.
        _locked_s_idx_cxn: Optional[int] = None
        _is_concrete_physical_noun_sense = bool(
            wn_pos == "n"
            and (
                any(
                    s.lexfile() in (
                        "noun.artifact", "noun.substance", "noun.object",
                        "noun.food", "noun.plant", "noun.animal", "noun.body",
                    )
                    for s in target_synsets
                )
                or (
                    bool(definition)
                    and bool(
                        set(re.findall(r"\b[a-z]{3,}\b", definition.lower()))
                        & {
                            "machine", "tool", "equipment", "device", "instrument",
                            "appliance", "vehicle", "piece", "object", "substance",
                            "clothing", "garment", "container", "hardware",
                        }
                    )
                )
            )
        )
        if definition or quote:
            _t_ent_cxn = cls.get_ldoce_entry(clean_target)
            if _t_ent_cxn:
                _senses_cxn = _t_ent_cxn.get("senses") or []
                if definition:
                    _def_norm_cxn = definition.strip().lower()
                    for _si, _sn in enumerate(_senses_cxn):
                        if str(_sn.get("definition") or "").strip().lower() == _def_norm_cxn:
                            _locked_s_idx_cxn = _si
                            break
                if _locked_s_idx_cxn is None:
                    _locked_s_idx_cxn = cls._lock_sense(
                        _t_ent_cxn, quote=quote or definition, target_pos=pos
                    )
        if _is_concrete_physical_noun_sense:
            cxn_cands, active_cxn = [], None
        else:
            cxn_cands, active_cxn = cls.get_construction_contrast_distractors(
                clean_target,
                pos=pos,
                sense_idx=_locked_s_idx_cxn,
                quote=quote,
                context_anchor=context_anchor,
                anchor_type=anchor_type,
                target_count=max(target_count * 2, 6),
                exclude_words=exclude_words,
                require_context_match=True,
            )
        cxn_cand_set: Set[str] = set()
        if active_cxn:
            cxn_banned = cls.get_cxn_banned_words(active_cxn.get("ban_groups")) | set(active_cxn.get("ban_groups") or [])
            forbidden_words |= cxn_banned
            sense_antonyms = [ant for ant in sense_antonyms if ant not in cxn_banned]
            if cxn_cands:
                cxn_cand_set = set(cxn_cands)
                cxn_excl = cxn_banned | (exclude_words or set()) | {clean_target}
                candidates = [
                    c for c in cxn_cands
                    if c not in cxn_excl and cls.is_cefr_compliant_distractor(c, clean_target)
                ] + [
                    c for c in candidates if c not in cxn_cand_set
                ]
        is_fronted_inv_slot = cls.is_fronted_inversion_context(
            quote=quote, anchor=context_anchor, anchor_type=anchor_type, target=clean_target
        )
        if context_anchor:
            clean_anchor = context_anchor.strip().lower()
            if (wn_pos in ("a", "n") or not wn_pos) and anchor_type == "prep":
                # Preposition Valency Gate - reject candidates that also govern the target preposition
                for cand in candidates:
                    cand_preps = cls.get_governed_prepositions(cand, pos=pos)
                    if clean_anchor in cand_preps or cls.lookup_ldoce_phrase(f"{cand} {clean_anchor}"):
                        forbidden_words.add(cand)
            elif wn_pos == "a" and anchor_type == "modified_noun":
                # Modified Noun Collocation Clash - reject legal adjectives that also collocate with the noun
                anchor_entry = ocd.get(clean_anchor, {})
                for a in anchor_entry.get("adj", []):
                    tok = a.split()[0].lower()
                    if tok not in sense_antonyms:
                        forbidden_words.add(tok)
            else:
                anchor_entry = ocd.get(clean_anchor, {})
                # If target is verb, anchor is noun: check verb_before and verb_after
                for v in anchor_entry.get("verb_before", []) + anchor_entry.get("verb_after", []):
                    forbidden_words.add(v.split()[0].lower())
                # If target is adj, anchor is noun: check adj
                for a in anchor_entry.get("adj", []):
                    tok = a.split()[0].lower()
                    if tok not in sense_antonyms:
                        forbidden_words.add(tok)
                # If target is noun, anchor is verb (verb-object frame): ban nouns that also collocate with that verb
                # (When anchor_type == 'adj', e.g. 'economic slump', noun items use definition-grounded sentence stems,
                # so non-synonym coordinates like 'prosperity' must not be banned merely for sharing 'economic'.)
                if not (wn_pos == "n" and anchor_type == "adj"):
                    for n in anchor_entry.get("colloc_nouns", []) + anchor_entry.get("noun_after", []):
                        forbidden_words.add(n.split()[0].lower())

        safe_distractors: List[str] = []
        distractor_metadata: Dict[str, str] = {}

        # Slot 1: Sense-Locked Antonym (if available and not superseded by a Construction Family contrast)
        # Antonyms are whitelisted from context_anchor collision clearance because their
        # collocational plausibility is deliberate, to be eliminated by sentence polarity/logic.
        # However, when a Construction Family subcategorization contrast (`cxn_cand_set`) is active,
        # subcategorization-contrasting candidates take primary slots so syntactic frame contrast is tested.
        max_antonyms = 0 if cxn_cand_set else (1 if (wn_pos != "a" or len(tier2_satellites) >= 2) else 2)
        for ant in sense_antonyms:
            if len([s for s in safe_distractors if distractor_metadata.get(s) == "antonym"]) >= max_antonyms:
                break
            if ant not in safe_distractors and ant not in exclude_words and len(ant) >= 2:
                if wn_pos == "v" and anchor_type == "prep" and cls.double_key_collision(
                    clean_target, ant, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                )[0]:
                    continue
                if cls.is_cefr_compliant_distractor(ant, clean_target):
                    safe_distractors.append(ant)
                    distractor_metadata[ant] = "antonym"

        # Slots 2 & 3: Candidates from PedLex Thesaurus & Activator / Coordinate / Troponym / Synonym trees
        near_set_for_pile, ants_for_pile = cls.semantic_fields(clean_target, pos=pos)
        if _is_collective_noun_target:
            near_set_for_pile = {
                w for w in near_set_for_pile
                if not cls._are_orthogonal_collective_nouns(clean_target, w)
            }
        strict_target_syns = cls.get_strict_pedlex_and_oxford_synonyms(
            clean_target, pos=pos, definition=definition, quote=quote
        )
        wn_same_synset_words: Set[str] = set()
        if target_synsets:
            for s in target_synsets:
                for w in s.words():
                    wn_same_synset_words.add(w.lemma().lower())
        near_only_for_pile = (
            (near_set_for_pile | tier0_set) & (strict_target_syns | wn_same_synset_words)
        ) - ants_for_pile - set(sense_antonyms)
        for cand in candidates:
            if cand in safe_distractors:
                continue
            if not cls.is_cefr_compliant_distractor(cand, clean_target):
                continue
            if _is_informal_or_slang_distractor(cand):
                continue
            has_cand_clash, cand_clash_r = cls.evaluate_syntactic_slot_clash(
                clean_target, cand, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
            )
            # Fronted Negative Inversion Arena Gate:
            # In an inverted clause (Aux + Subj), ban all non-inverting adverbs (*often, *steadily, *completely)
            # so multiple-choice distractors test SLA correlative/aspectual nuances rather than syntax giveaways.
            if is_fronted_inv_slot and cand_clash_r == "non_inverting_candidate_in_fronted_inversion_slot":
                continue
            # Synonym-Pile Gate: at most 1 non-antonym distractor may sit inside near_only_for_pile (strict near-synonyms of target)
            if not has_cand_clash and cand in near_only_for_pile and cand not in cxn_cand_set and sum(1 for s in safe_distractors if s in near_only_for_pile and distractor_metadata.get(s) != "antonym") >= 1:
                continue
            # Countability Symmetry Gate for Nouns:
            # - For strictly uncountable noun targets: allow at most 1 countable foil into safe_distractors.
            # - For strictly countable noun targets: allow at most 1 uncountable foil into safe_distractors.
            if wn_pos == "n":
                if target_strictly_uncountable and cls._is_primarily_countable_noun(cand):
                    if sum(1 for s in safe_distractors if cls._is_primarily_countable_noun(s)) >= 1:
                        continue
                elif target_strictly_countable and cls._is_primarily_uncountable_noun(cand):
                    if sum(1 for s in safe_distractors if cls._is_primarily_uncountable_noun(s)) >= 1:
                        continue
            # Attributive position check: if modifying a noun, reject predicative-only adjectives
            if anchor_type == "modified_noun" and cand in cls._PREDICATIVE_ONLY_ADJS:
                continue
            # Allomorph / Word Family Guard: prevent morphological double-keys (e.g. resiliency vs resilience)
            if (
                cls.are_same_word_family(cand, clean_target)
                and not (wn_pos == "n" and cls.is_cross_ontology_derivational_noun_pair(cand, clean_target))
            ) or any(cls.are_same_word_family(cand, s) for s in safe_distractors):
                continue
            # Cross-distractor mutual synonym exclusion: avoid multiple distractors being mutual synonyms
            # (For nouns, if non-antonym coordinates are exhausted and we are filling from `sense_antonyms`,
            # allow at most 1 extra antonym peer so we don't fall back to random orthographic rhymes.)
            cand_syns = cls.get_synonyms(cand)
            is_noun_fallback_ant = (
                wn_pos == "n"
                and cand in sense_antonyms
                and sum(1 for s in safe_distractors if distractor_metadata.get(s) == "antonym") < 2
            )
            if not is_noun_fallback_ant and cand not in cxn_cand_set and any(
                (s in cand_syns or cand in cls.get_synonyms(s))
                and not cls._are_orthogonal_collective_nouns(cand, s)
                for s in safe_distractors
            ):
                continue

            # Deterministic Double-Key & Pattern Exclusion Gate:
            # Rejects candidates that share identical syntactic patterns or collocations (double-key collision)
            if cand not in cxn_cand_set or cand in near_set_for_pile:
                is_dk, dk_reason = cls.double_key_collision(
                    clean_target, cand,
                    anchor=context_anchor,
                    anchor_type=anchor_type,
                    pos=pos,
                    quote=quote
                )
                if is_dk or (dk_reason == "contrast" and cand not in sense_antonyms):
                    continue

            if (cand in cxn_cand_set or cand not in forbidden_words) and cand not in exclude_words and len(cand) >= 2:
                safe_distractors.append(cand)
                if cand in cxn_cand_set:
                    distractor_metadata[cand] = "cxn_subcat_contrast"
                elif cand in sense_antonyms:
                    distractor_metadata[cand] = "antonym"
                elif wn_pos == "n" and target_strictly_uncountable and cls._is_primarily_countable_noun(cand):
                    distractor_metadata[cand] = "countability_contrast_countable"
                elif wn_pos == "n" and target_strictly_countable and cls._is_primarily_uncountable_noun(cand):
                    distractor_metadata[cand] = "countability_contrast_uncountable"
                elif cand in tier0_ldoce_thesaurus:
                    distractor_metadata[cand] = "ldoce_thesaurus"
                elif anchor_type == "prep" and has_cand_clash and cand_clash_r == "unattested_prepositional_government_clash":
                    distractor_metadata[cand] = "preposition_valency"
                elif wn_pos == "a":
                    if anchor_type == "prep" and any(cand in adjs for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items() if p != context_anchor):
                        distractor_metadata[cand] = "preposition_valency"
                    elif cand in tier2_satellites:
                        distractor_metadata[cand] = "satellite_synonym"
                    elif cand in tier3_attributes:
                        distractor_metadata[cand] = "attribute_coordinate"
                    else:
                        distractor_metadata[cand] = "synonym"
                else:
                    if cand in tier2_hyponyms:
                        distractor_metadata[cand] = "troponym"
                    elif cand in tier3_coordinates:
                        distractor_metadata[cand] = "coordinate"
                    else:
                        distractor_metadata[cand] = "synonym"
                if len(safe_distractors) >= target_count:
                    break

        # Fallback from Academic Semantic Families if candidates pool is sparse (e.g. specialized adjectives)
        if len(safe_distractors) < target_count and wn_pos == "a":
            for fam in cls._ACADEMIC_ADJ_FAMILIES:
                if clean_target in fam:
                    for a in fam:
                        if a != clean_target and a not in forbidden_words and a not in exclude_words and a not in safe_distractors:
                            if anchor_type == "modified_noun" and a in cls._PREDICATIVE_ONLY_ADJS:
                                continue
                            safe_distractors.append(a)
                            distractor_metadata[a] = "academic_family"
                            if len(safe_distractors) >= target_count:
                                break
                    break

        # Fallback from Academic Semantic Families if candidates pool is sparse (e.g. specialized verbs)
        if len(safe_distractors) < target_count and wn_pos == "v":
            academic_verb_families = [
                ["transmit", "distribute", "circulate", "relay", "deliver", "publish", "release"],
                ["instruct", "direct", "guide", "train", "advise", "command", "brief", "inform"],
                ["acquire", "obtain", "gain", "attain", "secure", "gather", "accumulate", "possess"],
                ["challenge", "assess", "evaluate", "test", "question", "examine", "review", "probe"],
                ["establish", "maintain", "sustain", "create", "develop", "foster", "initiate", "generate"]
            ]
            for fam in academic_verb_families:
                if clean_target in fam:
                    for v in fam:
                        if v != clean_target and v not in forbidden_words and v not in exclude_words and v not in safe_distractors:
                            safe_distractors.append(v)
                            distractor_metadata[v] = "academic_family"
                            if len(safe_distractors) >= target_count:
                                break
                    break

        # For concrete/substance/person nouns, if candidate pool is still sparse (< target_count),
        # harvest domain-aligned same-lexfile peers BEFORE orthographic paronyms so concrete substances (e.g. 'clay')
        # get genuine substance peers ('sand', 'dirt', 'dust') rather than orthographic rhymes ('clue'),
        # and person nouns (e.g. 'scientist') get 2 person coordinates + 1 morphological peer ("2 + 1").
        _is_person_target = wn_pos == "n" and bool(target_synsets) and all(
            s.lexfile() == "noun.person" for s in target_synsets if s.lexfile()
        )
        _pre_fill_cap = (target_count - 1) if (_is_person_target and not is_concrete_physical_noun) else target_count
        if len(safe_distractors) < _pre_fill_cap and wn_pos == "n":
            target_syns_pre = [s for s in target_synsets if s.lexfile()]
            target_lfs_pre = [s.lexfile() for s in target_syns_pre if s.lexfile()]
            if target_lfs_pre:
                if not cls._ocd_lexfile_cache:
                    for w in ocd.keys():
                        if (
                            len(w) >= 3
                            and w.isalpha()
                            and not re.fullmatch(r"(.)\1+", w)
                            and cls.is_exact_pedlex_headword(w)
                            and (cls.word_zipf(w) >= 3.0 or w in ocd)
                        ):
                            w_syns = [ws for ws in wn.synsets(w, "n") if ws.lexfile()]
                            if w_syns:
                                lf = w_syns[0].lexfile()
                                if lf not in cls._ocd_lexfile_cache:
                                    cls._ocd_lexfile_cache[lf] = []
                                cls._ocd_lexfile_cache[lf].append(w)
                _ref_syn = target_syns_pre[0]
                _ref_hyp = _ref_syn.hypernyms()[0] if _ref_syn.hypernyms() else _ref_syn
                for t_lf in target_lfs_pre:
                    raw_lf_pool = [
                        fb for fb in cls._ocd_lexfile_cache.get(t_lf, [])
                        if (
                            fb != clean_target
                            and fb not in forbidden_words
                            and fb not in exclude_words
                            and fb not in global_synonyms
                            and fb not in safe_distractors
                            and not _is_primarily_adj_or_verb(fb)
                            and not _is_informal_or_slang_distractor(fb)
                            and _is_aligned_noun_coord(fb, _ref_syn, _ref_hyp)
                            and cls.is_cefr_compliant_distractor(fb, clean_target)
                        )
                    ]
                    if "_coord_sort_key" in locals():
                        raw_lf_pool.sort(key=_coord_sort_key)
                    for fb in raw_lf_pool:
                        if (
                            not cls.are_same_word_family(fb, clean_target)
                            and not any(cls.are_same_word_family(fb, s) for s in safe_distractors)
                            and not cls.double_key_collision(
                                clean_target, fb, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                            )[0]
                        ):
                            safe_distractors.append(fb)
                            distractor_metadata[fb] = "coordinate"
                            if len(safe_distractors) >= _pre_fill_cap:
                                break
                    if len(safe_distractors) >= _pre_fill_cap:
                        break

        # 2-hop WordNet Semantic-Field Harvest for sparse verbs & adjectives before corpus fallback:
        # (1) Verbs: grandparent hypernym cousins ('admire' -> 'appreciate'/'value'/'consider', 'trap' -> 'release'/'waste'/'remove')
        # (2) Adjectives: similar_to -> head-antonym & its satellites ('renewable' -> 'exhaustible', 'habitable' -> 'unlivable')
        if len(safe_distractors) < target_count and wn_pos in ("v", "a") and target_synsets:
            two_hop_cands: List[Tuple[str, str]] = []
            two_hop_seen = set(safe_distractors) | set(exclude_words) | {clean_target}
            if wn_pos == "v":
                for _s in target_synsets[:2]:
                    for _h in _s.hypernyms():
                        for _gp in _h.hypernyms():
                            for _uncle in _gp.hyponyms():
                                for _lem in _uncle.lemmas():
                                    _lw = _lem.lower()
                                    if (
                                        _lw not in two_hop_seen
                                        and _lw.isalpha()
                                        and cls.is_exact_pedlex_headword(_lw)
                                        and cls.is_cefr_compliant_distractor(_lw, clean_target)
                                    ):
                                        two_hop_seen.add(_lw)
                                        two_hop_cands.append((_lw, "coordinate"))
                                        if len(two_hop_cands) >= 30:
                                            break
                                if len(two_hop_cands) >= 30:
                                    break
                            if len(two_hop_cands) >= 30:
                                break
                        if len(two_hop_cands) >= 30:
                            break
                    if len(two_hop_cands) >= 30:
                        break
            elif wn_pos == "a":
                for _s in target_synsets[:3]:
                    for _sim in _s.relations().get("similar", []):
                        for _sn in _sim.senses():
                            for _ant_sn in _sn.relations().get("antonym", []):
                                _aw = _ant_sn.word().lemma().lower()
                                if (
                                    _aw not in two_hop_seen
                                    and _aw.isalpha()
                                    and cls.is_exact_pedlex_headword(_aw)
                                    and cls.is_cefr_compliant_distractor(_aw, clean_target)
                                ):
                                    two_hop_seen.add(_aw)
                                    two_hop_cands.append((_aw, "antonym"))
                                for _ant_sim in _ant_sn.synset().relations().get("similar", []):
                                    for _as_lem in _ant_sim.lemmas():
                                        _asw = _as_lem.lower()
                                        if (
                                            _asw not in two_hop_seen
                                            and _asw.isalpha()
                                            and cls.is_exact_pedlex_headword(_asw)
                                            and cls.is_cefr_compliant_distractor(_asw, clean_target)
                                        ):
                                            two_hop_seen.add(_asw)
                                            two_hop_cands.append((_asw, "attribute_coordinate"))
            if "_coord_sort_key" in locals() and two_hop_cands:
                two_hop_cands.sort(key=lambda pair: _coord_sort_key(pair[0]))
            for _cw, _cmech in two_hop_cands:
                if len(safe_distractors) >= target_count:
                    break
                if (
                    _cw not in forbidden_words
                    and not cls.are_same_word_family(_cw, clean_target)
                    and not any(cls.are_same_word_family(_cw, s) for s in safe_distractors)
                    and not cls.double_key_collision(
                        clean_target, _cw, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )[0]
                    and not cls.evaluate_syntactic_slot_clash(
                        clean_target, _cw, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )[0]
                ):
                    safe_distractors.append(_cw)
                    distractor_metadata[_cw] = _cmech

        # Collocational head-noun adjective harvest for sparse adjectives before corpus_fallback:
        # e.g. '... of ____ land' -> ocd['land']['adj'] ('industrial', 'dry', 'native', 'public')
        #      '... of ____ energy' -> ocd['energy']['adj'] ('nuclear', 'atomic', 'alternative')
        if len(safe_distractors) < target_count and wn_pos == "a" and context_anchor:
            _anchor_low = str(context_anchor).strip().lower()
            _m_adj_head = re.search(r"_{2,}\s+([a-z]{3,})\b", _anchor_low)
            _head_n = _m_adj_head.group(1) if _m_adj_head else (
                _anchor_low if (anchor_type == "modified_noun" and _anchor_low.isalpha() and len(_anchor_low) >= 3) else ""
            )
            if _head_n:
                _head_ocd = cls.get_oxford_collocations(_head_n) or {}
                _raw_adj_cands: List[str] = []
                _seen_adj = set(safe_distractors) | set(exclude_words) | set(forbidden_words) | set(global_synonyms) | {clean_target, _head_n}
                for _entry in _head_ocd.get("adj", []):
                    for _tok in str(_entry).replace("/", " ").split():
                        _tok = _tok.strip().lower()
                        if (
                            _tok.isalpha()
                            and len(_tok) >= 3
                            and _tok not in _seen_adj
                            and cls.is_exact_pedlex_headword(_tok)
                            and any(ws.pos in ("a", "s") for ws in wn.synsets(_tok))
                            and cls.is_cefr_compliant_distractor(_tok, clean_target)
                        ):
                            _tok_ent = cls.get_ldoce_entry(_tok) or {}
                            _tok_senses = _tok_ent.get("senses") or []
                            if _tok_senses and any(cls._pos_code(str(sn.get("pos") or "")) == "adj" for sn in _tok_senses[:2]):
                                _seen_adj.add(_tok)
                                _raw_adj_cands.append(_tok)
                if "_coord_sort_key" in locals() and _raw_adj_cands:
                    _raw_adj_cands.sort(key=_coord_sort_key)
                for _aw in _raw_adj_cands:
                    if len(safe_distractors) >= target_count:
                        break
                    if (
                        not cls.are_same_word_family(_aw, clean_target)
                        and not any(cls.are_same_word_family(_aw, s) for s in safe_distractors)
                        and not cls.double_key_collision(
                            clean_target, _aw, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                        )[0]
                        and not cls.evaluate_syntactic_slot_clash(
                            clean_target, _aw, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                        )[0]
                    ):
                        safe_distractors.append(_aw)
                        distractor_metadata[_aw] = "collocation"

        # Fallback for Focus / Degree / Approximation / Discourse / Temporal Adverbs before Orthographic Paronym Tier
        # (so focus/degree/discourse/temporal adverbs like 'merely', 'virtually', 'barely', 'similarly', 'nowadays'
        # get semantic/functional contrasts rather than orthographic rhymes or generic manner adverbs)
        if len(safe_distractors) < target_count and wn_pos == "r" and not is_fronted_inv_slot:
            _adv_contrast_map = [
                (
                    {"merely", "solely", "purely", "simply", "exclusively", "just"},
                    ["entirely", "completely", "thoroughly", "utterly", "fully"],
                ),
                (
                    {"virtually", "practically", "nearly", "almost", "essentially", "effectively"},
                    ["barely", "scarcely", "hardly", "rarely", "seldom"],
                ),
                (
                    {"barely", "scarcely", "hardly", "narrowly"},
                    ["entirely", "completely", "thoroughly", "fully", "totally"],
                ),
                (
                    {"similarly", "likewise", "equally", "correspondingly"},
                    ["conversely", "otherwise", "instead", "meanwhile", "nevertheless"],
                ),
                (
                    {"conversely", "contrarily", "alternatively", "otherwise", "instead"},
                    ["similarly", "likewise", "moreover", "furthermore", "meanwhile"],
                ),
                (
                    {"however", "nevertheless", "nonetheless", "still", "yet"},
                    ["therefore", "consequently", "moreover", "furthermore", "similarly"],
                ),
                (
                    {"therefore", "consequently", "thus", "hence", "accordingly"},
                    ["however", "nevertheless", "otherwise", "meanwhile", "instead"],
                ),
                (
                    {"moreover", "furthermore", "besides", "additionally"},
                    ["however", "otherwise", "instead", "conversely", "nevertheless"],
                ),
                (
                    {"nowadays", "currently", "presently", "today"},
                    ["previously", "formerly", "rarely", "seldom", "eventually"],
                ),
                (
                    {"previously", "formerly", "originally", "initially", "earlier"},
                    ["currently", "nowadays", "eventually", "finally", "ultimately"],
                ),
                (
                    {"recently", "lately", "newly"},
                    ["previously", "formerly", "rarely", "eventually", "originally"],
                ),
                (
                    {"eventually", "ultimately", "finally"},
                    ["initially", "originally", "previously", "rarely", "seldom"],
                ),
                (
                    {"thereafter", "subsequently", "afterwards", "afterward"},
                    ["previously", "formerly", "initially", "originally", "meanwhile"],
                ),
                (
                    {"predominantly", "chiefly", "mainly", "primarily", "largely", "mostly"},
                    ["partly", "partially", "barely", "scarcely", "rarely", "equally"],
                ),
                (
                    {"decidedly", "undoubtedly", "unquestionably", "indisputably", "undeniably", "markedly"},
                    ["barely", "scarcely", "slightly", "faintly", "hardly", "marginally"],
                ),
                (
                    {"somehow", "anyhow"},
                    ["nowhere", "never", "seldom", "rarely", "barely", "scarcely"],
                ),
                (
                    {"vaguely", "faintly", "dimly", "obscurely"},
                    ["clearly", "vividly", "distinctly", "precisely", "explicitly", "sharply"],
                ),
                (
                    {"stubbornly", "obstinately", "defiantly", "doggedly"},
                    ["willingly", "readily", "gladly", "eagerly", "calmly", "gently"],
                ),
                (
                    {"conveniently", "smoothly", "effortlessly", "handily"},
                    ["awkwardly", "clumsily", "painfully", "barely", "reluctantly"],
                ),
                (
                    {"responsibly", "carefully", "cautiously", "prudently", "sensibly"},
                    ["recklessly", "carelessly", "rashly", "blindly", "foolishly"],
                ),
                (
                    {"passionately", "fiercely", "fervently", "intensely", "enthusiastically"},
                    ["calmly", "coldly", "indifferently", "mildly", "casually", "reluctantly"],
                ),
                (
                    {"happily", "cheerfully", "joyfully", "gladly", "merrily"},
                    ["sadly", "bitterly", "gloomily", "reluctantly", "anxiously", "angrily"],
                ),
                (
                    {"online", "offline"},
                    ["abroad", "overseas", "indoors", "outdoors", "locally", "remotely"],
                ),
                (
                    {"abroad", "overseas"},
                    ["indoors", "outdoors", "online", "locally", "remotely", "nearby"],
                ),
                (
                    {"indoors", "outdoors", "upstairs", "downstairs", "underground"},
                    ["abroad", "overseas", "online", "locally", "remotely", "nearby"],
                ),
            ]
            for _src_set, _contrast_list in _adv_contrast_map:
                if clean_target in _src_set:
                    for _adv_c in _contrast_list:
                        if (
                            _adv_c != clean_target
                            and _adv_c not in forbidden_words
                            and _adv_c not in exclude_words
                            and _adv_c not in safe_distractors
                            and not cls.are_same_word_family(_adv_c, clean_target)
                            and not any(cls.are_same_word_family(_adv_c, s) for s in safe_distractors)
                        ):
                            safe_distractors.append(_adv_c)
                            distractor_metadata[_adv_c] = "adv_semantic_contrast"
                            if len(safe_distractors) >= target_count:
                                break
                    break

        # Orthographic / Paronym Distractor Tier (with Cognate Near-Synonym Guard):
        # 1. If candidate pool is sparse (len(safe_distractors) < target_count), fill remaining slots with
        #    verified orthographic paronyms / orthogonal-domain root peers.
        # 2. If all 3 slots were filled, blend 1 paronym into Slot 3 when:
        #    - Slot 3 is a generic coordinate/troponym/attribute_coordinate or a 3rd antonym and a sharp
        #      paronym (score >= 45, e.g. adapt/adopt, complement/compliment, simulate/stimulate,
        #      industrial/industrious, considerate/considerable, respectful/respective) exists, OR
        #    - All 3 slots are generic coordinates and a clean orthogonal-domain root peer (score >= 28,
        #      e.g. specimen -> spectrum/spectacle/speculation) exists.
        if wn_pos in ("n", "v", "a", "r") and not is_fronted_inv_slot:
            need_paronym_fill = len(safe_distractors) < target_count
            pure_generic_coords = (
                len(safe_distractors) >= target_count
                and all(
                    distractor_metadata.get(d) in ("coordinate", "troponym", "attribute_coordinate")
                    for d in safe_distractors
                )
            )
            slot3_replaceable = (
                len(safe_distractors) >= target_count
                and (
                    (
                        distractor_metadata.get(safe_distractors[-1]) in (
                            "coordinate", "troponym", "attribute_coordinate", "ldoce_thesaurus",
                            *(("synonym",) if wn_pos == "n" else ()),
                        )
                        and (
                            wn_pos in ("n", "v")
                            or all(
                                distractor_metadata.get(d) not in ("antonym", "ldoce_thesaurus")
                                for d in safe_distractors[:-1]
                            )
                        )
                    )
                    or sum(1 for d in safe_distractors if distractor_metadata.get(d) == "antonym") >= 2
                )
            )
            if need_paronym_fill or pure_generic_coords or slot3_replaceable:
                paronym_cands = cls.get_orthographic_paronym_distractors(
                    clean_target,
                    pos=wn_pos,
                    target_count=target_count,
                    exclude_words=(exclude_words | forbidden_words | set(safe_distractors)),
                    context_anchor=context_anchor,
                    anchor_type=anchor_type,
                    quote=quote,
                    return_scores=True,
                )
                if need_paronym_fill:
                    for p_word, _p_score in paronym_cands:
                        if not (
                            len(clean_target) >= 5
                            and (
                                cls._common_prefix_len(clean_target, p_word) >= 3
                                or cls._common_suffix_len(clean_target, p_word) >= (3 if wn_pos == "n" else 4)
                                or _p_score >= 45
                            )
                        ):
                            continue
                        if wn_pos == "n":
                            if target_strictly_uncountable and cls._is_primarily_countable_noun(p_word):
                                if sum(1 for s in safe_distractors if cls._is_primarily_countable_noun(s)) >= 1:
                                    continue
                            elif target_strictly_countable and cls._is_primarily_uncountable_noun(p_word):
                                if sum(1 for s in safe_distractors if cls._is_primarily_uncountable_noun(s)) >= 1:
                                    continue
                        if p_word not in safe_distractors:
                            safe_distractors.append(p_word)
                            distractor_metadata[p_word] = "orthographic_paronym"
                            if len(safe_distractors) >= target_count:
                                break
                elif paronym_cands:
                    for best_p_word, best_p_score in paronym_cands:
                        # "2 + 1" Standard Distractor Blending:
                        # - For nouns (`wn_pos == 'n'`), Slot 3 is populated by a high-discrimination morphological/phonetic peer when:
                        #   (a) Multi-syllable root-sharing paronym (`len >= 7` and `pref >= 4`, e.g. specimen/spectrum, emergence/emergency, scientist/science),
                        #   (b) Stressed `-ion` (`-tion`/`-sion`) rhyme peer (`recession` -> `concession`/`succession`/`obsession`, `collision` -> `precision`/`coalition`),
                        #   (c) Academic suffix family peer (`-ics`, `-ist`, `-ism`, `-ship`, `-ware`, `-type`, `-ology`, e.g. `electronics`/`physics`, `scientist`/`physicist`, `optimism`/`criticism`, `hardware`/`software`),
                        #   (d) Rhyme-containing noun peer (`min_len >= 5` and suffix containment, e.g. `venue` -> `revenue`/`avenue`),
                        #   while never popping concrete coordinates for short everyday words (shopper, heat, cricket, clue, coat).
                        if wn_pos == "n":
                            _acad_sfx = ("ics", "ist", "ism", "ship", "ware", "type", "ology", "archy", "cracy")
                            is_true_orthographic_peer = (
                                (len(clean_target) >= 7 and cls._common_prefix_len(clean_target, best_p_word) >= 4)
                                or (
                                    clean_target.endswith(("tion", "sion"))
                                    and best_p_word.endswith(("tion", "sion"))
                                    and cls._common_suffix_len(clean_target, best_p_word) >= 4
                                    and cls._phonetic_rhyme_tail(clean_target) == cls._phonetic_rhyme_tail(best_p_word)
                                )
                                or (
                                    clean_target.endswith(("ance", "ence", "ment"))
                                    and best_p_word.endswith(("ance", "ence", "ment"))
                                    and cls._common_suffix_len(clean_target, best_p_word) >= 5
                                    and cls._phonetic_rhyme_tail(clean_target) == cls._phonetic_rhyme_tail(best_p_word)
                                )
                                or any(clean_target.endswith(sf) and best_p_word.endswith(sf) for sf in _acad_sfx)
                                or (
                                    min(len(clean_target), len(best_p_word)) >= 5
                                    and abs(len(clean_target) - len(best_p_word)) <= 3
                                    and (best_p_word.endswith(clean_target) or clean_target.endswith(best_p_word))
                                )
                            )
                        else:
                            is_true_orthographic_peer = (
                                clean_target[0] == best_p_word[0]
                                or cls._common_suffix_len(clean_target, best_p_word) >= 4
                            )
                        can_replace_coord = is_true_orthographic_peer and (
                            best_p_score >= 45
                            or (
                                pure_generic_coords
                                and best_p_score >= 28
                                and cls._common_prefix_len(clean_target, best_p_word) >= 4
                            )
                        )
                        replace_idx = -1
                        if wn_pos == "n" and can_replace_coord:
                            if target_strictly_uncountable and cls._is_primarily_countable_noun(best_p_word):
                                c_indices = [idx for idx, s in enumerate(safe_distractors) if cls._is_primarily_countable_noun(s)]
                                if len(c_indices) >= 1:
                                    replace_idx = c_indices[-1]
                                else:
                                    can_replace_coord = False
                            elif target_strictly_countable and cls._is_primarily_uncountable_noun(best_p_word):
                                u_indices = [idx for idx, s in enumerate(safe_distractors) if cls._is_primarily_uncountable_noun(s)]
                                if len(u_indices) >= 1:
                                    replace_idx = u_indices[-1]
                                else:
                                    can_replace_coord = False
                        if can_replace_coord and best_p_word not in safe_distractors:
                            popped = safe_distractors.pop(replace_idx)
                            distractor_metadata.pop(popped, None)
                            safe_distractors.append(best_p_word)
                            distractor_metadata[best_p_word] = "orthographic_paronym"
                            break

        # Tier 3.9: Noun Countability-Contrast Safe Distractor Slot ("countability_contrast_uncountable")
        # For countable singular/plural or [plural] nouns in `_NOUN_COUNTABILITY_CONTRAST_MAP`, inject at most 1
        # semantically related pure-[uncountable] peer (e.g. device/devices -> equipment/hardware; suggestion -> advice;
        # breakthrough -> progress; belongings -> luggage; finding -> information/evidence) into Slot 3 so the
        # countable/plural determiner frame deterministically rules out the uncountable distractor with 0 double-key risk.
        if wn_pos == "n" and not clean_target.endswith("ics"):
            t_gram_prof_n = cls.get_word_gram_profile(clean_target, pos="noun")
            is_countable_or_plural_noun_target = bool(
                target_strictly_countable
                or t_gram_prof_n.get("pure_countable")
                or t_gram_prof_n.get("pure_plural")
                or (
                    clean_target.endswith("s")
                    and not target_strictly_uncountable
                    and not t_gram_prof_n.get("pure_uncountable")
                )
            )
            if is_countable_or_plural_noun_target:
                t_lem_n = cls.lemma_of(clean_target)
                thes_uncountable_peers: List[str] = []
                t_entry_n = cls.get_ldoce_entry(t_lem_n) or cls.get_ldoce_entry(clean_target)
                if t_entry_n:
                    n_senses = [
                        s for s in (t_entry_n.get("senses") or [])
                        if "noun" in (s.get("pos") or t_entry_n.get("pos") or "").lower()
                    ]
                    is_single_noun_sense = len(n_senses) <= 1
                    d_words_n = {
                        w for w in re.findall(r"\b[a-zA-Z]{3,}\b", (definition or "").lower())
                        if w not in {"the", "and", "for", "with", "from", "that", "this", "something", "someone", "somebody"}
                    }
                    q_words_n = {
                        w for w in re.findall(r"\b[a-zA-Z]{3,}\b", (quote or "").lower())
                        if w not in {"the", "and", "for", "with", "from", "that", "this", "something", "someone", "somebody"}
                    }
                    thes_list_n = t_entry_n.get("thesaurus") or []
                    self_verified_clusters: Set[Any] = set()
                    for th_it in thes_list_n:
                        th_w = str(th_it.get("word") or "").strip().lower()
                        if th_w in (clean_target, t_lem_n):
                            th_dist_toks = set(re.findall(r"\b[a-zA-Z]{3,}\b", str(th_it.get("distinction") or "").lower()))
                            if is_single_noun_sense or not d_words_n or (th_dist_toks & (d_words_n | q_words_n)):
                                self_verified_clusters.add(th_it.get("section_index"))
                    for th_it in thes_list_n:
                        th_w = str(th_it.get("word") or "").strip().lower()
                        th_g = str(th_it.get("grammar") or "").strip().lower()
                        if (
                            th_w
                            and th_w not in (clean_target, t_lem_n)
                            and " " not in th_w
                            and "/" not in th_w
                            and "uncountable" in th_g
                            and not re.search(r"\bcountable\b", th_g)
                            and (is_single_noun_sense or th_it.get("section_index") in self_verified_clusters)
                        ):
                            if th_w not in thes_uncountable_peers:
                                thes_uncountable_peers.append(th_w)

                # If a sense-aligned uncountable peer (or pure_uncountable noun) was already selected in an earlier tier
                # (e.g. Tier 0 ldoce_thesaurus 'credit' for 'loan'), promote its metadata so we don't inject a 2nd uncountable foil.
                for ex_d in safe_distractors:
                    if (
                        ex_d in thes_uncountable_peers
                        or cls.get_word_gram_profile(ex_d, pos="noun").get("pure_uncountable")
                    ):
                        distractor_metadata[ex_d] = "countability_contrast_uncountable"
                        break

            if is_countable_or_plural_noun_target and not any(
                v == "countability_contrast_uncountable" for v in distractor_metadata.values()
            ):
                t_lem_n = cls.lemma_of(clean_target)
                contrast_pool: List[str] = list(thes_uncountable_peers)
                for key_w in (clean_target, t_lem_n):
                    for u_cand in cls._NOUN_COUNTABILITY_CONTRAST_MAP.get(key_w, []):
                        if u_cand not in contrast_pool:
                            contrast_pool.append(u_cand)
                for u_cand in list(t_gram_prof_n.get("comparison_peers") or []):
                    u_clean = str(u_cand or "").strip().lower()
                    if (
                        u_clean
                        and u_clean not in contrast_pool
                        and " " not in u_clean
                        and "_" not in u_clean
                        and "-" not in u_clean
                        and cls.get_word_gram_profile(u_clean, pos="noun").get("pure_uncountable")
                    ):
                        contrast_pool.append(u_clean)
                # For singular countable targets, prefer uncountable foils sharing the target's vowel/consonant initial
                # so indefinite article frames ('... repay a ____') are never forced into mixed-initial rewrites.
                if not clean_target.endswith("s") and clean_target[:1].isalpha():
                    t_is_vowel = clean_target[0].lower() in "aeiou"
                    contrast_pool.sort(key=lambda w: 0 if (bool(w) and (w[0].lower() in "aeiou") == t_is_vowel) else 1)
                picked_uncountable: Optional[str] = None
                for u_word in contrast_pool:
                    if (
                        u_word != clean_target
                        and u_word != t_lem_n
                        and u_word not in safe_distractors
                        and u_word not in exclude_words
                        and not cls.are_same_word_family(u_word, clean_target)
                        and cls.is_exact_pedlex_headword(u_word)
                        and (
                            u_word in thes_uncountable_peers
                            or cls.get_word_gram_profile(u_word, pos="noun").get("pure_uncountable")
                        )
                        and cls.is_cefr_compliant_distractor(u_word, clean_target)
                    ):
                        picked_uncountable = u_word
                        break
                if picked_uncountable:
                    if len(safe_distractors) < target_count:
                        safe_distractors.append(picked_uncountable)
                        distractor_metadata[picked_uncountable] = "countability_contrast_uncountable"
                    elif len(safe_distractors) >= target_count:
                        last_d = safe_distractors[-1]
                        if distractor_metadata.get(last_d) in (
                            "coordinate", "troponym", "attribute_coordinate",
                            "synonym", "ldoce_thesaurus", "dynamic_domain_fallback",
                        ):
                            safe_distractors.pop()
                            distractor_metadata.pop(last_d, None)
                            safe_distractors.append(picked_uncountable)
                            distractor_metadata[picked_uncountable] = "countability_contrast_uncountable"

            # Part B: Symmetrical Countability Contrast for Strictly Uncountable Noun Targets
            # Inject at most 1 semantically related countable foil (e.g. advice -> suggestion/tip;
            # equipment -> tool/device; information -> fact/detail; furniture -> chair/desk;
            # luggage -> belongings/item; progress -> breakthrough/advance) into safe_distractors.
            is_strictly_uncountable_noun_target = bool(
                target_strictly_uncountable
                or t_gram_prof_n.get("pure_uncountable")
            )
            if is_strictly_uncountable_noun_target:
                t_lem_n = cls.lemma_of(clean_target)
                rev_map = cls.get_noun_countability_contrast_map(reverse=True)
                curated_contrast_cands: List[str] = []
                for key_w in (clean_target, t_lem_n):
                    for c_cand in rev_map.get(key_w, []):
                        if c_cand not in curated_contrast_cands:
                            curated_contrast_cands.append(c_cand)

                # 1. Promote or normalize existing countable distractor metadata (protecting paronyms and antonyms)
                curated_in_distractors = [d for d in safe_distractors if d in curated_contrast_cands]
                if curated_in_distractors:
                    distractor_metadata[curated_in_distractors[0]] = "countability_contrast_countable"
                elif not curated_contrast_cands:
                    c_distractors = [
                        d for d in safe_distractors
                        if cls._is_primarily_countable_noun(d)
                        and distractor_metadata.get(d) not in ("orthographic_paronym", "antonym")
                    ]
                    if c_distractors:
                        distractor_metadata[c_distractors[0]] = "countability_contrast_countable"

                # Prune excess countable distractors beyond the 1-foil quota
                c_distractors = [
                    d for d in safe_distractors
                    if cls._is_primarily_countable_noun(d)
                    and distractor_metadata.get(d) not in ("orthographic_paronym", "antonym")
                ]
                if len(c_distractors) > 1:
                    keep_c = curated_in_distractors[0] if curated_in_distractors else c_distractors[0]
                    for extra_c in list(c_distractors):
                        if extra_c != keep_c and extra_c in safe_distractors:
                            safe_distractors.remove(extra_c)
                            distractor_metadata.pop(extra_c, None)

                # 2. If no countable foil is present yet, or a curated foil is available and missing, inject at most 1 from reverse countability contrast map or thesaurus
                has_curated_contrast = any(d in curated_contrast_cands for d in safe_distractors)
                need_curated_injection = bool(curated_contrast_cands and not has_curated_contrast)
                if not any(v == "countability_contrast_countable" for v in distractor_metadata.values()) or need_curated_injection:
                    contrast_pool_c: List[str] = list(curated_contrast_cands)

                    # Also inspect LDOCE thesaurus for countable peers
                    t_entry_n = cls.get_ldoce_entry(t_lem_n) or cls.get_ldoce_entry(clean_target)
                    if t_entry_n:
                        thes_list_n = t_entry_n.get("thesaurus") or []
                        for th_it in thes_list_n:
                            th_w = str(th_it.get("word") or "").strip().lower()
                            th_g = str(th_it.get("grammar") or "").strip().lower()
                            if (
                                th_w
                                and th_w not in (clean_target, t_lem_n)
                                and " " not in th_w
                                and "/" not in th_w
                                and "countable" in th_g
                                and not re.search(r"\buncountable\b", th_g)
                                and th_w not in contrast_pool_c
                            ):
                                contrast_pool_c.append(th_w)

                    picked_countable: Optional[str] = None
                    for c_word in contrast_pool_c:
                        if (
                            c_word != clean_target
                            and c_word != t_lem_n
                            and c_word not in safe_distractors
                            and c_word not in exclude_words
                            and not cls.are_same_word_family(c_word, clean_target)
                            and cls.is_exact_pedlex_headword(c_word)
                            and cls._is_primarily_countable_noun(c_word)
                            and cls.is_cefr_compliant_distractor(c_word, clean_target)
                        ):
                            picked_countable = c_word
                            break

                    if picked_countable:
                        # If safe_distractors has an un-curated countable distractor, replace it
                        uncurated_c = [
                            d for d in safe_distractors
                            if cls._is_primarily_countable_noun(d)
                            and d not in curated_contrast_cands
                            and distractor_metadata.get(d) not in ("orthographic_paronym", "antonym")
                        ]
                        if uncurated_c:
                            rep_d = uncurated_c[0]
                            rep_idx = safe_distractors.index(rep_d)
                            safe_distractors[rep_idx] = picked_countable
                            distractor_metadata.pop(rep_d, None)
                            distractor_metadata[picked_countable] = "countability_contrast_countable"
                        elif len(safe_distractors) < target_count:
                            safe_distractors.append(picked_countable)
                            distractor_metadata[picked_countable] = "countability_contrast_countable"
                        elif len(safe_distractors) >= target_count:
                            last_d = safe_distractors[-1]
                            if distractor_metadata.get(last_d) in (
                                "coordinate", "troponym", "attribute_coordinate",
                                "synonym", "ldoce_thesaurus", "dynamic_domain_fallback",
                            ):
                                safe_distractors.pop()
                                distractor_metadata.pop(last_d, None)
                                safe_distractors.append(picked_countable)
                                distractor_metadata[picked_countable] = "countability_contrast_countable"

        # Second Defense: Dynamic domain-aligned fallback using target WordNet lexfile
        # When candidates pool is still sparse, extract homogeneous fallback words from the target's
        # sense-locked WordNet lexfile (e.g. noun.cognition, noun.group) to prevent Domain/Category Mismatch.
        if len(safe_distractors) < target_count and wn_pos == "n":
            target_syns = [s for s in target_synsets if s.lexfile()] or [s for s in wn.synsets(clean_target, "n") if s.lexfile()]
            target_lfs = [s.lexfile() for s in target_syns if s.lexfile()]
            if target_lfs:
                # Lazy-build or retrieve lexfile-indexed OCD nouns
                if not cls._ocd_lexfile_cache:
                    for w in ocd.keys():
                        if (
                            len(w) >= 3
                            and w.isalpha()
                            and not re.fullmatch(r"(.)\1+", w)
                            and cls.is_exact_pedlex_headword(w)
                            and (cls.word_zipf(w) >= 3.0 or w in ocd)
                        ):
                            w_syns = [ws for ws in wn.synsets(w, "n") if ws.lexfile()]
                            if w_syns:
                                lf = w_syns[0].lexfile()
                                if lf not in cls._ocd_lexfile_cache:
                                    cls._ocd_lexfile_cache[lf] = []
                                cls._ocd_lexfile_cache[lf].append(w)
                for t_lf in target_lfs:
                    raw_fb_pool = [
                        fb for fb in cls._ocd_lexfile_cache.get(t_lf, [])
                        if (
                            fb != clean_target
                            and fb not in forbidden_words
                            and fb not in exclude_words
                            and fb not in global_synonyms
                            and fb not in safe_distractors
                            and not _is_primarily_adj_or_verb(fb)
                            and not _is_informal_or_slang_distractor(fb)
                            and cls.is_cefr_compliant_distractor(fb, clean_target)
                        )
                    ]
                    if "_coord_sort_key" in locals():
                        raw_fb_pool.sort(key=_coord_sort_key)
                    else:
                        raw_fb_pool.sort(key=lambda x: -cls._zipf_freq(x))
                    for fb in raw_fb_pool:
                        if target_strictly_uncountable:
                            if not cls._is_primarily_uncountable_noun(fb):
                                _fb_ent = cls.get_ldoce_entry(fb) or {}
                                _fb_sns = [
                                    sn for sn in (_fb_ent.get("senses") or [])
                                    if not sn.get("pos") or cls._pos_code(sn.get("pos")) == "n"
                                ] or (_fb_ent.get("senses") or [])
                                if not _fb_sns or "uncountable" not in " ".join(str(sn.get("grammar") or "").lower() for sn in _fb_sns[:2]):
                                    continue
                        if (
                            not cls.are_same_word_family(fb, clean_target)
                            and not any(cls.are_same_word_family(fb, s) for s in safe_distractors)
                            and not cls.double_key_collision(
                                clean_target, fb, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                            )[0]
                        ):
                            safe_distractors.append(fb)
                            distractor_metadata[fb] = "dynamic_domain_fallback"
                            if len(safe_distractors) >= target_count:
                                break
                    if len(safe_distractors) >= target_count:
                        break

            if len(safe_distractors) < target_count and target_strictly_uncountable:
                core_unc_pool = [
                    "guidance", "counsel", "feedback", "information", "equipment", "hardware",
                    "machinery", "progress", "research", "evidence", "assistance", "funding",
                    "legislation", "pollution", "luggage", "baggage", "furniture", "property",
                ]
                for unc_cand in (list(near_set_for_pile) + core_unc_pool):
                    if (
                        unc_cand != clean_target
                        and unc_cand not in safe_distractors
                        and unc_cand not in forbidden_words
                        and unc_cand not in exclude_words
                        and cls._is_primarily_uncountable_noun(unc_cand)
                        and not cls.are_same_word_family(unc_cand, clean_target)
                        and not any(cls.are_same_word_family(unc_cand, s) for s in safe_distractors)
                        and cls.is_cefr_compliant_distractor(unc_cand, clean_target)
                        and not cls.double_key_collision(
                            clean_target, unc_cand, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                        )[0]
                    ):
                        safe_distractors.append(unc_cand)
                        distractor_metadata[unc_cand] = "uncountable_domain_peer"
                        if len(safe_distractors) >= target_count:
                            break

        # Second Defense for Adjectives (`wn_pos == "a"`): Morphological & Syntactic-Slot Aligned Peers
        # Prevents participial `-ing` adjectives (e.g. `rewarding`) or relational `adj.pert` adjectives (e.g. `doctoral`)
        # from falling back to generic `['simple', 'common', 'similar']`.
        if len(safe_distractors) < target_count and wn_pos == "a":
            _is_pert_adj = any(s.lexfile() == "adj.pert" for s in target_synsets if s.lexfile()) or bool(
                cls.get_word_gram_profile(clean_target, pos="adj").get("only_before_noun")
            )
            if clean_target.endswith("ing"):
                _adj_morph_pool = [
                    "demanding", "challenging", "exhausting", "frustrating", "promising",
                    "convincing", "appealing", "astonishing", "overwhelming", "confusing",
                    "alarming", "disturbing", "encouraging", "refreshing",
                ]
            elif _is_pert_adj or clean_target.endswith(("al", "ic", "ar", "ory", "ary", "ive")):
                _ory_ive_pool = [
                    "advisory", "compulsory", "mandatory", "voluntary", "legislative", "executive",
                    "administrative", "judicial", "disciplinary", "consultative", "investigative",
                    "corrective", "preventive", "selective",
                ]
                _al_ic_pool = [
                    "academic", "medical", "surgical", "clinical", "legal", "musical",
                    "vocational", "clerical", "dental", "naval", "royal", "rural",
                    "urban", "civil", "regional", "domestic", "internal", "external",
                ]
                if clean_target.endswith(("ory", "ive")):
                    _adj_morph_pool = _ory_ive_pool + [w for w in _al_ic_pool if w not in _ory_ive_pool]
                else:
                    _adj_morph_pool = _al_ic_pool + [w for w in _ory_ive_pool if w not in _al_ic_pool]
            elif clean_target.endswith("ed"):
                _adj_morph_pool = [
                    "Exhausted", "confused", "frustrated", "convinced", "ashamed",
                    "relieved", "pleased", "disappointed", "crowded", "isolated",
                ]
                _adj_morph_pool = [w.lower() for w in _adj_morph_pool]
            else:
                _adj_morph_pool = []
            for _m_adj in _adj_morph_pool:
                if (
                    _m_adj != clean_target
                    and _m_adj not in forbidden_words
                    and _m_adj not in exclude_words
                    and _m_adj not in global_synonyms
                    and _m_adj not in safe_distractors
                    and not cls.are_same_word_family(_m_adj, clean_target)
                    and not any(cls.are_same_word_family(_m_adj, s) for s in safe_distractors)
                    and cls.is_cefr_compliant_distractor(_m_adj, clean_target)
                    and not cls.evaluate_syntactic_slot_clash(
                        clean_target, _m_adj, anchor=context_anchor, anchor_type=anchor_type, pos="adj", quote=quote
                    )[0]
                    and not cls.double_key_collision(
                        clean_target, _m_adj, anchor=context_anchor, anchor_type=anchor_type, pos="adj", quote=quote
                    )[0]
                ):
                    safe_distractors.append(_m_adj)
                    distractor_metadata[_m_adj] = "morphological_peer"
                    if len(safe_distractors) >= target_count:
                        break

        # Negative Inversion Fallback: strictly select peer inversion triggers
        if len(safe_distractors) < target_count and is_fronted_inv_slot and clean_target in cls._NEGATIVE_INVERSION_ADVERBS:
            for inv_peer in ("seldom", "hardly", "scarcely", "barely", "little", "no sooner", "rarely"):
                if (
                    inv_peer != clean_target
                    and inv_peer not in safe_distractors
                    and inv_peer not in forbidden_words
                    and inv_peer not in exclude_words
                    and not cls.double_key_collision(
                        clean_target, inv_peer, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )[0]
                ):
                    clash, clash_reason = cls.evaluate_syntactic_slot_clash(
                        clean_target, inv_peer, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )
                    if clash and clash_reason == "non_inverting_candidate_in_fronted_inversion_slot":
                        continue
                    safe_distractors.append(inv_peer)
                    distractor_metadata[inv_peer] = "fronted_inv_peer"
                    if len(safe_distractors) >= target_count:
                        break

        # Universal Fallback if candidates pool is still sparse: select stratified core corpus items
        if len(safe_distractors) < target_count and not is_fronted_inv_slot:
            needed = target_count - len(safe_distractors)
            excl = set(forbidden_words) | set(exclude_words) | set(safe_distractors) | {clean_target}
            pref_vowel: Optional[bool] = None
            if wn_pos == "n" and not clean_target.endswith("s") and clean_target[:1].isalpha():
                pref_vowel = clean_target[0].lower() in "aeiou"
            fb_cands = cls.get_corpus_fallback_candidates(
                target_word=clean_target,
                pos=wn_pos or pos or "noun",
                target_count=needed,
                exclude_words=excl,
                prefer_initial_vowel=pref_vowel,
            )
            for fb in fb_cands:
                safe_distractors.append(fb)
                distractor_metadata[fb] = "corpus_fallback"
                if len(safe_distractors) >= target_count:
                    break

        final_distractors = safe_distractors[:target_count]
        if return_metadata:
            return final_distractors, {k: distractor_metadata.get(k, "distractor") for k in final_distractors}
        return final_distractors

    @classmethod
    def generate_zero_collision_distractors(
        cls,
        target_word: str,
        dependent_prep: Optional[str] = None,
        pos: str = "v",
        count: int = 3,
        context_sentence: Optional[str] = None
    ) -> List[str]:
        """
        Pre-computes distractors mathematically guaranteed to avoid synonym collisions with target.
        When dependent_prep is specified (e.g. 'on' for 'rely'), prioritizes candidates that
        do NOT govern that preposition or clash with it.
        Uses context_sentence and MiniLM disambiguation to anchor to the pedagogical synset
        rather than blind synsets[0].
        """
        wn = cls.get_wordnet()
        target_lower = target_word.lower()
        target_synonyms = cls.get_synonyms(target_lower)
        target_synonyms.add(target_lower)

        antonyms = cls.get_antonyms(target_lower)

        candidates: List[str] = []

        # Antonyms make outstanding, collision-free distractors
        for ant in antonyms:
            if ant not in target_synonyms and " " not in ant and cls.is_cefr_compliant_distractor(ant, target_lower):
                candidates.append(ant)

        # 2. Taxonomy sibling / hypernym search
        target_synsets = cls.get_wordnet_synsets(target_lower, pos=pos)
        if target_synsets:
            primary_synset = cls.disambiguate_wordnet_synset(
                target_lower, pos=pos, context_sentence=context_sentence, candidate_synsets=target_synsets
            ) or target_synsets[0]
            # Get hypernyms (parent categories)
            hypernyms = primary_synset.get_related("hypernym")
            for hyper in hypernyms:
                # Get siblings (hyponyms of the hypernym)
                for sibling in hyper.get_related("hyponym"):
                    for w in sibling.words():
                        lemma = w.lemma().replace("_", " ").lower()
                        if (
                            lemma not in target_synonyms
                            and " " not in lemma
                            and lemma not in candidates
                            and cls.is_cefr_compliant_distractor(lemma, target_lower)
                        ):
                            if pos == "n" and (cls._is_primarily_adj_or_verb(lemma) if hasattr(cls, "_is_primarily_adj_or_verb") else False):
                                continue
                            candidates.append(lemma)

        # 3. Deduplicate and return requested count
        final_distractors = candidates[:count]

        # Fallback if synset taxonomy was sparse
        if len(final_distractors) < count:
            needed = count - len(final_distractors)
            excl = set(target_synonyms) | set(final_distractors) | {target_lower}
            pref_vowel: Optional[bool] = None
            if pos == "n" and not target_lower.endswith("s") and target_lower[:1].isalpha():
                pref_vowel = target_lower[0].lower() in "aeiou"
            fb_cands = cls.get_corpus_fallback_candidates(
                target_word=target_lower,
                pos=pos,
                target_count=needed,
                exclude_words=excl,
                prefer_initial_vowel=pref_vowel,
            )
            for fb in fb_cands:
                if fb not in target_synonyms and fb not in final_distractors:
                    final_distractors.append(fb)
                if len(final_distractors) >= count:
                    break

        return final_distractors
