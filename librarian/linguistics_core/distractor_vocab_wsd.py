"""Word Sense Disambiguation and Antonym extraction for vocabulary distractors."""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics.vocab_distractors")


class DistractorVocabWsdMixin:

    _AUDITORY_PREDICATES = {
        "hear", "listen", "overhear", "sound", "ring", "whisper", "shout", "cry",
        "loud", "faint", "soft", "harsh", "quiet", "noisy", "deafening", "audible",
        "earshot", "angry", "screaming", "yelling", "speak", "spoke", "spoken", "utter",
    }

    @classmethod
    def _pos_ok(cls, lemma: str, wn_pos: Optional[str] = None) -> bool:
        """Keep hyponym/coordinate candidates on the target's own part of speech."""
        if not wn_pos:
            return True
        try:
            wn = cls.get_wordnet()
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

    @classmethod
    def _is_primarily_adj_or_verb(cls, lemma: str) -> bool:
        """Filter out collective nominals or words that are primarily verbs/adjectives."""
        try:
            cand_ent = cls.get_ldoce_entry(lemma)
            if cand_ent:
                prim_pos = (cand_ent.get("pos") or "").lower()
                all_senses = cand_ent.get("senses") or []
                noun_senses = [s for s in all_senses if cls._pos_code(s.get("pos") or prim_pos) == "n"]
                if prim_pos == "noun" or len(noun_senses) >= 2:
                    return False
                if prim_pos in ("verb", "adjective", "adverb") and len(noun_senses) == 0:
                    return True
            wn = cls.get_wordnet()
            if wn:
                pos_list = [w.pos for w in wn.words(lemma)]
                syns = wn.synsets(lemma)
                syn_n = len([s for s in syns if s.pos == "n"])
                syn_a = len([s for s in syns if s.pos in ("a", "s")])
                syn_v = len([s for s in syns if s.pos == "v"])
                if syn_a > syn_n * 2 and syn_n <= 1:
                    return True
                if syn_v > syn_n * 3 and syn_n <= 1:
                    return True
                if "v" in pos_list and "n" not in pos_list:
                    return True
                if any(p in ("a", "s") for p in pos_list) and "n" not in pos_list:
                    return True
            return False
        except Exception:
            pass
        return False

    @classmethod
    def _is_primarily_noun_or_adj_for_verb(cls, lemma: str) -> bool:
        """Filter out nominal/adjectival homographs (e.g. 'pitch', 'gear', 'wire', 'square', 'tame', 'shoehorn')
        whose primary POS in PedLex is noun/adjective rather than verb."""
        try:
            cand_ent = cls.get_ldoce_entry(lemma)
            if cand_ent:
                prim_pos = (cand_ent.get("pos") or "").lower()
                if prim_pos in ("noun", "adjective", "adverb"):
                    return True
            wn = cls.get_wordnet()
            if wn:
                syn_v = len(wn.synsets(lemma, "v"))
                syn_na = len(wn.synsets(lemma, "n")) + len(wn.synsets(lemma, "a")) + len(wn.synsets(lemma, "s"))
                if syn_na > syn_v * 2 and syn_v <= 2:
                    return True
            return False
        except Exception:
            pass
        return False

    @classmethod
    def _is_informal_or_slang_distractor(
        cls,
        cand_w: str,
        wn_pos: Optional[str] = "n",
        target_is_informal: bool = False,
        target_lfs_set: Optional[Set[str]] = None,
    ) -> bool:
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
            wn = cls.get_wordnet()
            if wn:
                c_wn_syns = wn.synsets(base_cand, "n") or wn.synsets(cand_w, "n")
                if c_wn_syns:
                    primary_lf = c_wn_syns[0].lexfile()
                    if primary_lf in ("noun.animal", "noun.plant", "noun.food", "noun.body") and primary_lf not in target_lfs_set:
                        return True
        return False

    @classmethod
    def _prepare_vocab_wsd_and_antonyms(
        cls,
        clean_target: str,
        pos: str = "noun",
        context_anchor: Optional[str] = None,
        anchor_type: str = "collocation",
        target_count: int = 3,
        exclude_words: Optional[Set[str]] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes Phase 1: WSD, definition locking, and sense-locked antonym extraction."""
        wn = cls.get_wordnet()

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

        _pos_ok = lambda lem: cls._pos_ok(lem, wn_pos=wn_pos)
        _is_primarily_adj_or_verb = cls._is_primarily_adj_or_verb
        _is_primarily_noun_or_adj_for_verb = cls._is_primarily_noun_or_adj_for_verb
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
        def_tokens: Set[str] = set()
        pedlex_signpost = ""
        pedlex_units: Set[str] = set()
        if definition:
            nlp = cls.get_spacy()
            def_clean = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", definition or "").strip()
            quote_clean = quote or ""
            def_doc = nlp(def_clean.lower())
            quote_doc = nlp(quote_clean.lower())
            self_forms = {clean_target, target_lemma} | {t.lemma_.lower() for t in nlp(clean_target)}
            _generic_degree_mods = {"extreme", "extremely", "complete", "completely", "total", "totally", "want", "wants", "wanted"}
            def_tokens = (({t.lemma_.lower() for t in def_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS) - self_forms) - _generic_degree_mods
            quote_tokens = ({t.lemma_.lower() for t in quote_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS) - self_forms
            anchor_low = (context_anchor or "").lower().strip()
            quote_low = (quote or "").lower()
            has_auditory_context = (
                anchor_low in cls._AUDITORY_PREDICATES
                or any(ap in quote_tokens for ap in cls._AUDITORY_PREDICATES)
                or any(re.search(rf"\b{re.escape(ap)}\b", quote_low) for ap in ("hear", "listen", "overhear", "loud", "faint", "angry", "whisper", "shout", "sound", "speak", "voice"))
            )

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

                    imm_hyper = s.hypernyms()[0] if s.hypernyms() else None
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
                        s_def_low.startswith(("a word ", "an expression ", "a linguistic ", "the grammatical ", "a term used in ", "a term in "))
                        or "phonetics" in s_def_low
                        or "phonology" in s_def_low
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

                    # Auditory Context Weighting:
                    if has_auditory_context and wn_pos == "n":
                        is_auditory_synset = (
                            s_lf in ("noun.communication", "noun.sound", "noun.event")
                            or any(w in s_def_low for w in ("sound", "vocal", "auditory", "voice", "speech", "utterance", "noise"))
                        )
                        is_abstract_faculty_synset = (
                            s_lf in ("noun.attribute", "noun.cognition", "noun.relation")
                            or "ability to" in s_def_low
                            or "faculty of" in s_def_low
                        )
                        if is_auditory_synset and not is_abstract_faculty_synset:
                            penalty -= 15
                        elif is_abstract_faculty_synset:
                            penalty += 15

                    # Ontological Hypernym Penalty:
                    # Penalize synsets whose immediate hypernym is an ultra-abstract ontological container
                    # (e.g. 'condition, status', 'state', 'attribute') rather than a specific taxonomy
                    if imm_hyper and hasattr(imm_hyper, "id") and imm_hyper.id in getattr(cls, "_ABSTRACT_ONTOLOGY_HYPERNYMS", set()):
                        penalty += 8

                    # Musical / Melody Domain Mismatch Penalty:
                    # When context and pedagogical definition do not pertain to music/singing,
                    # heavily penalize musical/instrument/polyphonic voice synsets (e.g. voice = polyphonic part).
                    is_music_context = bool((def_tokens | quote_tokens) & {"music", "musical", "song", "sing", "singing", "melody", "tune", "instrument", "choir", "opera"})
                    if not is_music_context and ("music" in chain_str_low or "melody" in s_def_low or "musical" in s_def_low or "polyphonic" in s_def_low):
                        penalty += 14

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
                    pedlex_signpost_lemma = nlp(pedlex_signpost)[0].lemma_.lower() if pedlex_signpost else ""
                    if pedlex_signpost and (
                        pedlex_signpost in s_direct_lemmas
                        or pedlex_signpost_lemma in s_direct_lemmas
                        or pedlex_signpost in chain_lemmas
                        or pedlex_signpost_lemma in chain_lemmas
                    ):
                        unit_signpost_bonus += 12
                    if wn_pos == "n" and strict_ped_ox_syns:
                        ox_syn_peers = (s_lemmas_low - self_forms) & strict_ped_ox_syns
                        if ox_syn_peers:
                            unit_signpost_bonus += min(12, len(ox_syn_peers) * 5)
                        ox_direct_peers = (s_direct_lemmas - self_forms) & strict_ped_ox_syns
                        if ox_direct_peers:
                            unit_signpost_bonus += min(6, len(ox_direct_peers) * 2)
                        ox_chain_peers = (chain_lemmas - self_forms) & strict_ped_ox_syns
                        if ox_chain_peers:
                            unit_signpost_bonus += min(6, len(ox_chain_peers) * 2)
                    score = def_ov + quote_ov + unit_signpost_bonus - penalty
                    scored_synsets.append((score, s))

            scored_synsets.sort(key=lambda x: x[0], reverse=True)
            if scored_synsets:
                best_ov = scored_synsets[0][0]
                if best_ov > 0:
                    target_synsets = [s for ov, s in scored_synsets if ov >= max(1, best_ov - 3)]
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

        _is_informal_or_slang_distractor = lambda w: cls._is_informal_or_slang_distractor(
            w, wn_pos=wn_pos, target_is_informal=target_is_informal, target_lfs_set=target_lfs_set
        )

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
                if cls.is_exact_pedlex_headword(al):
                    prio = 0 - priority_bonus + cefr_penalty
                elif cls.get_word_cefr(al) in ("A1", "A2", "B1", "B2", "C1") or any(al.startswith(p) and cls.is_exact_pedlex_headword(al[len(p):]) for p in ("un", "in", "im", "ir", "il", "non", "dis")):
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

        return {
            "clean_target": clean_target,
            "pos": pos,
            "wn_pos": wn_pos,
            "context_anchor": context_anchor,
            "anchor_type": anchor_type,
            "target_count": target_count,
            "exclude_words": exclude_words,
            "definition": definition,
            "quote": quote,
            "wn": wn,
            "nlp": nlp,
            "target_synsets": target_synsets,
            "global_synonyms": global_synonyms,
            "ped_ox_syns": ped_ox_syns,
            "def_tokens": def_tokens,
            "target_lfs_set": target_lfs_set,
            "target_strictly_countable": target_strictly_countable,
            "target_strictly_uncountable": target_strictly_uncountable,
            "is_concrete_physical_noun": is_concrete_physical_noun,
            "target_is_informal": target_is_informal,
            "pedlex_signpost": pedlex_signpost,
            "pedlex_units": pedlex_units,
            "sense_antonyms": sense_antonyms,
            "has_auditory_context": locals().get("has_auditory_context", False),
        }
