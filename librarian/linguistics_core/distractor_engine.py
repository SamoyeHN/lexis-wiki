"""
Single-word, multi-word phrase, and orthographic paronym distractor generation, syntactic <gram> slot clash gating, and double-key screening.
"""

"""
Linguistic Engine for Lexis Wiki.
Integrates spaCy (computational dependency syntax) and WordNet (lexical relations)
to achieve deterministic sentence indexing, macro grammar domain classification,
boundary-accurate phrase extraction, canonical lemmatization, and zero-collision distractors.
"""

import datetime
import hashlib
import json
import os
import re
import sqlite3
import threading
import zlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union



class DistractorEngineMixin:

    # Orthogonal-Domain Collective Noun Peer Pool:
    # Collective nouns (WordNet `noun.group`) have no antonyms and often sit under fragmented or
    # metonymic WordNet hypernyms (`force/personnel`, `social unit`, `gathering` with `room`/`suite`).
    # Grouping them into three macro ontological classes (`human_role`, `animal`, `vessel_batch`)
    # where each inner tuple represents a synonym subgroup guarantees:
    # 1. Intra-subgroup synonyms (e.g. `staff`/`personnel`/`workforce`, `jury`/`panel`) are mutually banned.
    # 2. Orthogonal subgroup heads within the same macro class serve as high-discrimination Tier 3 coordinates.
    _COLLECTIVE_NOUN_ONTOLOGY: Dict[str, List[Tuple[str, ...]]] = {
        "human_role": [
            ("staff", "personnel", "workforce"),
            ("crew",),
            ("cast", "troupe"),
            ("faculty",),
            ("jury", "panel", "tribunal"),
            ("audience", "crowd", "congregation", "throng", "mob"),
            ("committee", "council", "board", "commission", "assembly", "delegation"),
            ("choir", "chorus", "orchestra", "band"),
            ("clientele",),
            ("clergy",),
            ("police", "squad", "troop", "regiment", "contingent"),
            ("gentry", "aristocracy", "elite"),
        ],
        "animal": [
            ("flock",),
            ("herd", "cattle", "livestock"),
            ("pack",),
            ("swarm", "colony"),
            ("shoal",),
            ("poultry", "brood"),
        ],
        "vessel_batch": [
            ("fleet", "convoy", "armada"),
            ("batch", "consignment"),
            ("bundle", "bunch", "cluster", "array"),
            ("cargo", "freight"),
        ],
    }

    @classmethod
    def _are_orthogonal_collective_nouns(cls, w1: str, w2: str) -> bool:
        """True when `w1` and `w2` belong to DIFFERENT synonym subgroups in `_COLLECTIVE_NOUN_ONTOLOGY`
        (e.g., `staff` vs `crew`/`cast`/`jury`/`faculty`, or `flock` vs `herd`/`pack`/`swarm`).
        Oxford Writer's Thesaurus groups co-hyponym collective nouns in the same thesaurus bucket,
        whereas different subgroups denote mutually exclusive roles/species and are prime coordinates."""
        a = (w1 or "").strip().lower()
        b = (w2 or "").strip().lower()
        if not a or not b or a == b:
            return False
        for sub_list in cls._COLLECTIVE_NOUN_ONTOLOGY.values():
            grp_a = next((i for i, grp in enumerate(sub_list) if a in grp), None)
            grp_b = next((i for i, grp in enumerate(sub_list) if b in grp), None)
            if grp_a is not None and grp_b is not None:
                return grp_a != grp_b
        return False

    @classmethod
    def _is_taboo_or_proper_noun(cls, cand_w: str, wn_pos: Optional[str] = None) -> bool:
        """Unified safety gate blocking capitalized proper nouns/demonyms (e.g. Polack, Islam)
        and taboo/offensive/vulgar/derogatory/trademark entries across PedLex and WordNet."""
        cw = (cand_w or "").strip().lower()
        if not cw:
            return True
        _classroom_sensitive = {
            "nudity", "mistress", "prostitute", "prostitution", "harlot",
            "concubine", "brothel", "adultery", "erotic", "porn", "pornography",
            "rape", "rapist", "incest", "orgasm", "molest", "molester",
            "pedophile", "paedophile", "pimp", "whore", "slut", "hooker",
            "fetish", "penis", "vagina", "anus", "masturbate", "masturbation",
            "sodomy", "bestiality", "fornication", "genitals", "genitalia",
            "testicle", "scrotum", "clitoris", "erection", "ejaculate", "ejaculation",
        }
        if cw in _classroom_sensitive:
            return True
        p_code = {"n": "n", "v": "v", "a": "adj", "r": "adv", "s": "adj"}.get(wn_pos or "", "")
        e_cand = cls.get_ldoce_entry(cw)
        if e_cand:
            homographs = e_cand.get("homographs") or []
            if homographs and all(
                isinstance(h.get("headword"), str) and h["headword"] and h["headword"][0].isupper()
                for h in homographs
            ):
                return True
            senses = e_cand.get("senses") or []
            pos_senses = [s for s in senses if not p_code or not s.get("pos") or cls._pos_code(s.get("pos")) == p_code] or senses
            _hard_bad_regs = ("taboo", "offensive", "vulgar", "derogatory", "trademark")
            _hard_bad_defs = (
                "offensive word", "do not use this word", "very offensive",
                "used as an insult", "disparaging", "ethnic slur",
            )
            for s in pos_senses:
                reg_str = str(s.get("register") or "").lower()
                defn_str = str(s.get("definition") or "").lower()
                if any(tok in reg_str for tok in _hard_bad_regs):
                    return True
                if any(cue in defn_str for cue in _hard_bad_defs):
                    return True
        try:
            wn = cls.get_wordnet()
            if wn_pos in ("a", "s"):
                c_words = wn.words(cw, pos="a") + wn.words(cw, pos="s")
            elif wn_pos in ("n", "v", "r"):
                c_words = wn.words(cw, pos=wn_pos)
            else:
                c_words = wn.words(cw)
            c_syns = [s for w in c_words for s in w.synsets()]
            if c_syns and all(
                all(lemma and lemma[0].isupper() for lemma in s.lemmas())
                for s in c_syns if s.lemmas()
            ):
                return True
        except Exception:
            pass
        return False

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

        # For satellite adjectives ('s'), add head synset lemmas (e.g. 'heavy' for 'burdensome')
        # directly to global_synonyms so the head adjective is never picked as a distractor
        for s in target_synsets:
            if wn_pos == "a" and s.pos == "s":
                for head in s.get_related("similar"):
                    for hw in head.words():
                        global_synonyms.add(hw.lemma().lower())

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
            and "uncountable" in _t_gram_low
            and "countable" not in re.sub(r"\buncountable\b", "", _t_gram_low)
        )
        target_strictly_countable = (
            wn_pos == "n"
            and (
                (
                    "countable" in re.sub(r"\buncountable\b", "", _t_gram_low)
                    and "uncountable" not in _t_gram_low
                )
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
                    # Opposite satellites in bipolar cluster (1 hop only; no 3rd-hop antonym-of-satellite)
                    for opp_sim in ant_sense.synset().get_related("similar") + ant_sense.synset().get_related("also"):
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
                            for opp_sim in ant_sense.synset().get_related("similar") + ant_sense.synset().get_related("also"):
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
            for it in thes_items:
                it_w = str(it.get("word", "")).strip().lower()
                it_concept = str(it.get("concept", "")).strip().lower()
                it_cluster_key = it_concept or (f"sec_{it.get('section_index')}" if it.get("section_index") is not None else "")
                if it_w == clean_target and it_cluster_key:
                    self_toks = {
                        w for w in re.findall(r"\b[a-zA-Z]{3,}\b", (str(it.get("distinction", "")) + " " + it_concept).lower())
                        if w not in _thes_stopwords and w != clean_target
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
                # If definition/quote provided, filter out Activator/Thesaurus clusters with zero content-word overlap against the target sense
                if content_words and overlap_score == 0 and not it.get("is_grambox_peer"):
                    continue
                # Boost true Thesaurus distinction records slightly over broad Activator clusters
                if not it.get("concept"):
                    overlap_score += 1
                if it.get("is_grambox_peer"):
                    overlap_score += 5
                raw_cand_head = str(it.get("word", "")).strip().lower().split("/")[0].split()[0] if it.get("word") else ""
                if raw_cand_head:
                    has_clash, _ = cls.evaluate_syntactic_slot_clash(
                        clean_target, raw_cand_head,
                        anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )
                    if has_clash:
                        overlap_score += 15
                scored_thes_items.append((overlap_score, it))
            scored_thes_items.sort(key=lambda x: x[0], reverse=True)

            for _, it in scored_thes_items:
                raw_w = it.get("word", "").strip().lower()
                # Skip instructional template phrases containing 'etc'
                if "etc" in raw_w:
                    continue
                # If phrase contains spaces and slashes (e.g. 'the Department of Science/English/Trade'),
                # it is a template frame, not an atomic candidate list; skip to prevent fragmentary distractor leaks
                if "/" in raw_w:
                    parts_check = [p.strip() for p in raw_w.split("/")]
                    if any(" " in p for p in parts_check):
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
                        # Do NOT extract verb head from phrasal verbs or idiomatic phrases (e.g. 'come forward', 'let me')
                        if wn_pos == "v":
                            _light_verbs = {"make", "get", "have", "take", "give", "do", "put", "set", "keep", "let", "go", "come", "be"}
                            if (
                                len(parts) == 2
                                and parts[1] in ("sb", "sth", "someone", "something", "oneself")
                                and parts[0].lower() not in _light_verbs
                            ):
                                cand = parts[0]
                            else:
                                continue
                        # Extract noun head from noun phrases (e.g. 'branch of', 'facilities for', 'member of')
                        elif wn_pos == "n" and parts[1] in ("of", "for", "to", "in", "on", "with", "as", "sb", "sth"):
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
                        and _pos_ok(picked)
                        and cls.is_cefr_compliant_distractor(picked, clean_target)
                        and not _is_informal_or_slang_distractor(picked)
                    ):
                        cand_entry = cls.get_ldoce_entry(picked)
                        if cand_entry:
                            c_primary = (cand_entry.get("pos") or "").lower()
                            c_poses = [p.lower() for p in cand_entry.get("all_poses", [c_primary])]
                            if wn_pos == "n":
                                if not any("noun" in p for p in c_poses):
                                    continue
                                # Reject if primary PedLex POS is verb or adjective and spacy also detects non-noun
                                if c_primary in ("verb", "adjective", "adverb"):
                                    continue
                            elif wn_pos == "v":
                                if not any("verb" in p for p in c_poses):
                                    continue
                                if c_primary in ("noun", "adjective", "adverb"):
                                    continue
                            elif wn_pos == "a" and not any("adj" in p for p in c_poses):
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
            else:
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
            p_code_coord = {"n": "n", "v": "v", "a": "adj", "r": "adv"}.get(wn_pos or "n", "n")

            def _coord_sort_key(cand_w: str) -> Tuple[int, int, int, int, int, int, int, int, int]:
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
                has_ldoce_core = 0 if int(c_ent.get("core_level") or 0) > 0 else 1
                is_non_core = 0 if (cand_w in ocd or has_ldoce_core == 0 or c_lvl_raw in ("A1", "A2", "B1", "B2", "C1")) else 1
                gerund_mismatch = 1 if (wn_pos == "n" and cand_w.endswith("ing") and not clean_target.endswith("ing")) else 0
                return (
                    gerund_mismatch,
                    countability_mismatch,
                    is_non_core,
                    -coord_hyper_overlap.get(cand_w, 0),
                    0 if has_ex else 1,
                    0 if prim_aligned else 1,
                    has_ldoce_core,
                    abs(c_rank - t_rank),
                    1 if c_rank >= 6 else 0,
                )
            if tier2_hyponyms:
                tier2_hyponyms.sort(key=_coord_sort_key)
            if tier3_coordinates:
                tier3_coordinates.sort(key=_coord_sort_key)

        # Candidate prioritization: PedLex Thesaurus & Activators are primary (Tier 0); WordNet is secondary/fallback
        if wn_pos == "v":
            candidates = tier0_ldoce_thesaurus + [c for c in (tier2_hyponyms + tier3_coordinates) if c not in global_synonyms]
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
                candidates = tier0_ldoce_thesaurus + tier3_coordinates + tier2_hyponyms + tier1_synonyms + extra_noun_ants
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
            cxn_banned = cls.get_cxn_banned_words(active_cxn.get("ban_groups"))
            forbidden_words |= cxn_banned
            sense_antonyms = [ant for ant in sense_antonyms if ant not in cxn_banned]
            if cxn_cands:
                cxn_cand_set = set(cxn_cands)
                candidates = [c for c in cxn_cands if c not in forbidden_words] + [
                    c for c in candidates if c not in cxn_cand_set
                ]
        if context_anchor:
            clean_anchor = context_anchor.strip().lower()
            if wn_pos == "a" and anchor_type == "prep":
                # Preposition Valency Gate - reject candidates that also govern the target preposition
                for cand in candidates:
                    cand_preps = ocd.get(cand, {}).get("prep", [])
                    if clean_anchor in cand_preps or cand in cls._ADJ_PREP_VALENCY_MAP.get(clean_anchor, []):
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
                if ant == sense_antonyms[0] or cls.is_cefr_compliant_distractor(ant, clean_target):
                    safe_distractors.append(ant)
                    distractor_metadata[ant] = "antonym"

        # Slots 2 & 3: Candidates from PedLex Thesaurus & Activator / Coordinate / Troponym / Synonym trees
        near_set_for_pile, ants_for_pile = cls.semantic_fields(clean_target, pos=pos)
        if _is_collective_noun_target:
            near_set_for_pile = {
                w for w in near_set_for_pile
                if not cls._are_orthogonal_collective_nouns(clean_target, w)
            }
        near_only_for_pile = (near_set_for_pile | tier0_set) - ants_for_pile - set(sense_antonyms)
        for cand in candidates:
            if cand in safe_distractors:
                continue
            if _is_informal_or_slang_distractor(cand):
                continue
            has_cand_clash, _ = cls.evaluate_syntactic_slot_clash(
                clean_target, cand, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
            )
            # Synonym-Pile Gate: at most 1 non-antonym distractor may sit inside `near_set` (unless syntactically clashing)
            if not has_cand_clash and cand in near_only_for_pile and sum(1 for s in safe_distractors if s in near_only_for_pile and distractor_metadata.get(s) != "antonym") >= 1:
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
            if not is_noun_fallback_ant and any(
                (s in cand_syns or cand in cls.get_synonyms(s))
                and not cls._are_orthogonal_collective_nouns(cand, s)
                for s in safe_distractors
            ):
                continue

            # Deterministic Double-Key & Pattern Exclusion Gate:
            # Rejects candidates that share identical syntactic patterns or collocations (double-key collision)
            if cand not in cxn_cand_set:
                is_dk, dk_reason = cls.double_key_collision(
                    clean_target, cand,
                    anchor=context_anchor,
                    anchor_type=anchor_type,
                    pos=pos,
                    quote=quote
                )
                if is_dk or (dk_reason == "contrast" and cand not in sense_antonyms):
                    continue

            if cand not in forbidden_words and cand not in exclude_words and len(cand) >= 2:
                safe_distractors.append(cand)
                if cand in cxn_cand_set:
                    distractor_metadata[cand] = "cxn_subcat_contrast"
                elif cand in sense_antonyms:
                    distractor_metadata[cand] = "antonym"
                elif cand in tier0_ldoce_thesaurus:
                    distractor_metadata[cand] = "ldoce_thesaurus"
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
        if len(safe_distractors) < _pre_fill_cap and wn_pos == "n" and (is_concrete_physical_noun or _is_person_target):
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

        # Orthographic / Paronym Distractor Tier (with Cognate Near-Synonym Guard):
        # 1. If candidate pool is sparse (len(safe_distractors) < target_count), fill remaining slots with
        #    verified orthographic paronyms / orthogonal-domain root peers.
        # 2. If all 3 slots were filled, blend 1 paronym into Slot 3 when:
        #    - Slot 3 is a generic coordinate/troponym/attribute_coordinate or a 3rd antonym and a sharp
        #      paronym (score >= 45, e.g. adapt/adopt, complement/compliment, simulate/stimulate,
        #      industrial/industrious, considerate/considerable, respectful/respective) exists, OR
        #    - All 3 slots are generic coordinates and a clean orthogonal-domain root peer (score >= 28,
        #      e.g. specimen -> spectrum/spectacle/speculation) exists.
        if wn_pos in ("n", "v", "a", "r"):
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
                            wn_pos == "n"
                            or all(
                                distractor_metadata.get(d) not in ("antonym", "ldoce_thesaurus")
                                for d in safe_distractors
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
                        if wn_pos == "n" and not (
                            len(clean_target) >= 5
                            and (
                                cls._common_prefix_len(clean_target, p_word) >= 3
                                or cls._common_suffix_len(clean_target, p_word) >= 3
                                or _p_score >= 45
                            )
                        ):
                            continue
                        if p_word not in safe_distractors:
                            safe_distractors.append(p_word)
                            distractor_metadata[p_word] = "orthographic_paronym"
                            if len(safe_distractors) >= target_count:
                                break
                elif paronym_cands:
                    best_p_word, best_p_score = paronym_cands[0]
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
                    if can_replace_coord and best_p_word not in safe_distractors:
                        popped = safe_distractors.pop()
                        distractor_metadata.pop(popped, None)
                        safe_distractors.append(best_p_word)
                        distractor_metadata[best_p_word] = "orthographic_paronym"

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
                            _fb_ent = cls.get_ldoce_entry(fb) or {}
                            _fb_sns = [
                                sn for sn in (_fb_ent.get("senses") or [])
                                if not sn.get("pos") or cls._pos_code(sn.get("pos")) == "n"
                            ] or (_fb_ent.get("senses") or [])
                            if _fb_sns and "uncountable" not in " ".join(str(sn.get("grammar") or "").lower() for sn in _fb_sns[:2]):
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

        # Universal Fallback if candidates pool is still sparse: select standard homogeneous items from OCD
        if len(safe_distractors) < target_count:
            t_lvl_str = cls.get_word_cefr(clean_target) or "B2"
            t_rank = cls.CEFR_ORDER.get(t_lvl_str, 4)
            if t_rank <= 3:
                fallback_pool = {
                    "v": ["choose", "accept", "decide", "expect", "follow", "notice", "explain", "remain", "manage", "allow"],
                    "n": ["choice", "reason", "matter", "situation", "effort", "problem", "action", "change", "moment", "condition"],
                    "a": ["simple", "common", "certain", "different", "similar", "natural", "clear", "direct", "actual", "special"],
                    "r": ["deeply", "strictly", "tightly", "widely", "carefully", "clearly", "readily", "sharply", "directly", "steadily"]
                }.get(wn_pos or "n", ["choice", "reason", "matter"])
            else:
                fallback_pool = {
                    "v": ["maintain", "establish", "determine", "evaluate", "indicate", "demonstrate", "direct", "guide", "sustain"],
                    "n": ["aspect", "factor", "process", "measure", "element", "context", "matter", "institution", "framework"],
                    "a": ["crucial", "essential", "primary", "initial", "direct", "specific", "constant", "fundamental"],
                    "r": ["deeply", "strictly", "tightly", "widely", "carefully", "clearly", "readily", "sharply", "directly", "steadily"]
                }.get(wn_pos or "n", ["factor", "element", "process"])
            for fb in fallback_pool:
                if (
                    fb != clean_target
                    and fb not in forbidden_words
                    and fb not in exclude_words
                    and fb not in safe_distractors
                    and not cls.are_same_word_family(fb, clean_target)
                    and not any(cls.are_same_word_family(fb, s) for s in safe_distractors)
                    and cls.is_cefr_compliant_distractor(fb, clean_target)
                ):
                    safe_distractors.append(fb)
                    distractor_metadata[fb] = "corpus_fallback"
                if len(safe_distractors) >= target_count:
                    break

        final_distractors = safe_distractors[:target_count]
        if return_metadata:
            return final_distractors, {k: distractor_metadata.get(k, "distractor") for k in final_distractors}
        return final_distractors

    # -------------------------------------------------------------------------
    # Orthographic / Paronym Distractor Tier (with Cognate Near-Synonym Guard)
    # -------------------------------------------------------------------------
    _PARONYM_STOP_DEF_TOKS = frozenset({
        "someone", "something", "person", "people", "thing", "things", "make", "made",
        "have", "having", "used", "very", "more", "most", "much", "many", "good", "bad",
        "way", "time", "place", "part", "group", "kind", "type", "form", "state", "act",
        "process", "quality", "relating", "connected", "particular", "certain",
        "able", "likely", "full", "without", "especially", "usually", "often", "always",
        "from", "that", "with", "into", "this", "these", "those", "which", "what", "when",
        "where", "while", "whose", "whom", "your", "their", "other", "another", "some",
        "such", "each", "every", "both", "either", "neither", "only", "also", "even",
        "just", "still", "already", "about", "above", "after", "again", "against", "along",
        "among", "around", "because", "before", "behind", "below", "beneath", "beside",
        "between", "beyond", "during", "except", "inside", "instead", "outside", "through",
        "toward", "towards", "under", "until", "upon", "within", "amount",
        "piece", "taken", "take", "taking", "example", "single", "small", "large", "great",
        "high", "long", "short", "order", "achieve", "produce", "produced", "show", "shown",
        "seen", "notice", "laugh", "help", "try", "trying", "know", "knowing", "think",
        "thinking", "ideas", "idea", "mind", "real", "containing", "interesting",
        "work", "works", "working", "hard", "feeling", "feelings", "information",
        "action", "activity",
    })

    @classmethod
    def _ensure_orthographic_pos_index(cls) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """Lazily build O(1) POS-bucketed first/second character index over PedLex (or OCD/WordNet fallback)."""
        if cls._orthographic_pos_index is not None:
            return cls._orthographic_pos_index

        pos_vocab: Dict[str, Set[str]] = {"n": set(), "v": set(), "a": set(), "r": set()}
        pos_map = {
            "noun": "n", "verb": "v", "adjective": "a", "adverb": "r",
            "n": "n", "v": "v", "adj": "a", "adv": "r", "s": "a",
        }

        cls.get_ldoce_entry("test")
        conn = cls._ldoce_conn
        if conn is not None:
            try:
                rows = conn.execute("SELECT word, pos, pos_all FROM ldoce WHERE kind='article'").fetchall()
                for w, p, p_all in rows:
                    if not w or not w.isalpha() or len(w) < 4:
                        continue
                    wl = w.lower()
                    if wl in ("enrol", "fulfil", "instil", "distil", "skilful", "wilful"):
                        continue
                    raw_ps = [
                        x.strip().strip('"\'[]').lower()
                        for x in re.split(r"[,|]", p_all or p or "")
                        if x.strip().strip('"\'[]')
                    ]
                    for rp in raw_ps:
                        code = pos_map.get(rp)
                        if code:
                            if code == "v" and wl in ("grown", "known", "shown", "thrown", "blown", "flown", "drawn", "worn", "torn", "sworn", "born", "borne", "fallen", "taken", "given", "driven", "written", "ridden", "risen", "chosen", "spoken", "broken", "frozen", "stolen", "woven"):
                                continue
                            pos_vocab[code].add(wl)
            except Exception:
                pass

        # Open-source / fallback enrichment from OCD + WordNet if PedLex is disabled or sparse
        if not any(pos_vocab.values()):
            wn = cls.get_wordnet()
            ocd = cls.get_oxford_collocations()
            pool = set(ocd.keys())
            for w in pool:
                if not w or not w.isalpha() or len(w) < 4:
                    continue
                wl = w.lower()
                for p_code in ("n", "v", "a", "r"):
                    try:
                        syns = wn.synsets(wl, "a") + wn.synsets(wl, "s") if p_code == "a" else wn.synsets(wl, p_code)
                        if syns:
                            pos_vocab[p_code].add(wl)
                    except Exception:
                        pass
        else:
            # Ensure pluralia tantum / -ics academic disciplines in OCD (e.g. 'mechanics', 'robotics')
            # that live inside singular PedLex headword articles are indexed under 'n'.
            for w in cls.get_oxford_collocations().keys():
                if w and w.isalpha() and len(w) >= 6 and w.lower().endswith("ics"):
                    pos_vocab["n"].add(w.lower())

        by_first: Dict[str, Dict[str, List[str]]] = {k: {} for k in pos_vocab}
        by_second: Dict[str, Dict[str, List[str]]] = {k: {} for k in pos_vocab}
        by_skel: Dict[str, Dict[str, List[str]]] = {k: {} for k in pos_vocab}
        by_rhyme: Dict[str, Dict[str, List[str]]] = {k: {} for k in pos_vocab}
        by_suffix: Dict[str, Dict[str, List[str]]] = {k: {} for k in pos_vocab}
        _noun_sfx_keys = ("ics", "ist", "ism", "ment", "ance", "ence", "ity", "ship", "ware", "type", "ology", "archy", "cracy")
        for k, words in pos_vocab.items():
            for w in sorted(words):
                by_first[k].setdefault(w[0], []).append(w)
                if len(w) >= 2:
                    by_second[k].setdefault(w[1], []).append(w)
                skel = cls._phonetic_consonant_skeleton(w)
                if len(skel) >= 3:
                    by_skel[k].setdefault(skel, []).append(w)
                rhyme = cls._phonetic_rhyme_tail(w)
                if len(rhyme) >= 3:
                    by_rhyme[k].setdefault(rhyme, []).append(w)
                if k == "n":
                    for sfx in _noun_sfx_keys:
                        if w.endswith(sfx) and len(w) >= len(sfx) + 2:
                            by_suffix[k].setdefault(sfx, []).append(w)
                    if len(w) >= 6:
                        by_suffix[k].setdefault(w[-4:], []).append(w)

        cls._orthographic_pos_index = {
            "by_first": by_first,
            "by_second": by_second,
            "by_skel": by_skel,
            "by_rhyme": by_rhyme,
            "by_suffix": by_suffix,
        }
        return cls._orthographic_pos_index

    @staticmethod
    def _edit_distance_bounded(s1: str, s2: str) -> int:
        """Fast bounded Levenshtein distance (returns 99 if distance > 2)."""
        if abs(len(s1) - len(s2)) > 2:
            return 99
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            min_row = dp[0]
            for j in range(1, n + 1):
                temp = dp[j]
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = temp
                if dp[j] < min_row:
                    min_row = dp[j]
            if min_row > 2:
                return 99
        return dp[n]

    @staticmethod
    def _common_prefix_len(s1: str, s2: str) -> int:
        c = 0
        for a, b in zip(s1, s2):
            if a == b:
                c += 1
            else:
                break
        return c

    @staticmethod
    def _common_suffix_len(s1: str, s2: str) -> int:
        c = 0
        for a, b in zip(reversed(s1), reversed(s2)):
            if a == b:
                c += 1
            else:
                break
        return c

    @staticmethod
    def _phonetic_consonant_skeleton(word: str) -> str:
        """SnappyWords / Datamuse 'Consonant Match' (`rel_cns`) + Double-Metaphone G2P normalization.
        Collapses vowel differences and English orthographic-phonetic equivalences (ph->f, c(e/i/y)->s,
        c->k, qu->kw, x->ks, silent gh, twin consonants) into a canonical consonant skeleton."""
        w = (word or "").strip().lower()
        if not w:
            return ""
        w = re.sub(r"^kn", "n", w)
        w = re.sub(r"^wr", "r", w)
        w = re.sub(r"^ps", "s", w)
        w = re.sub(r"^wh", "w", w)
        w = w.replace("ph", "f").replace("qu", "kw").replace("x", "ks").replace("ck", "k").replace("dg", "j")
        w = re.sub(r"c(?=[eiy])", "s", w)
        w = w.replace("c", "k")
        w = re.sub(r"gh(?=[aeiouy]|$)", "", w)
        w = re.sub(r"[aeiouy]+", "", w)
        out: List[str] = []
        for ch in w:
            if not out or out[-1] != ch:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _phonetic_rhyme_tail(word: str) -> str:
        """SnappyWords / Datamuse 'Rhyme Coda' (`rel_rhy`) G2P normalization from the last stressed/penultimate
        vowel cluster to the end of the word, normalizing -ceed/-cede/-sede -> -seed, etc."""
        w = (word or "").strip().lower()
        if len(w) < 3:
            return w
        for old_end, new_end in (
            ("ceed", "seed"), ("cede", "seed"), ("sede", "seed"),
            ("ceive", "seev"), ("lieve", "leev"),
            ("ight", "ite"), ("ought", "awt"), ("aught", "awt"),
            ("ssion", "shun"), ("scion", "shun"),
            ("tion", "shun"), ("sion", "shun"), ("cian", "shun"),
            ("cial", "shul"), ("tial", "shul"),
            ("ious", "us"), ("eous", "us"), ("ous", "us"),
        ):
            if w.endswith(old_end):
                w = w[:-len(old_end)] + new_end
                break
        m = list(re.finditer(r"[aeiouy]+[^aeiouy]*", w))
        if not m:
            return w[-3:]
        if len(m) >= 2 and w.endswith("e") and len(m[-1].group(0)) <= 2:
            tail = w[m[-2].start():]
        elif len(m) >= 2 and len(w[m[-1].start():]) <= 3:
            tail = w[m[-2].start():]
        else:
            tail = w[m[-1].start():]
        tail = re.sub(r"c(?=[eiy])", "s", tail)
        tail = tail.replace("c", "k").replace("ph", "f")
        return tail

    @staticmethod
    def _estimate_syllables(word: str) -> int:
        """Fast deterministic syllable count estimator matching Datamuse `sl` syllable alignment."""
        w = (word or "").strip().lower()
        if not w:
            return 0
        if w.endswith("e") and not w.endswith(("le", "ee", "ie", "oe", "ue")) and len(w) > 3:
            w = w[:-1]
        groups = re.findall(r"[aeiouy]+", w)
        return max(1, len(groups))

    @classmethod
    def _get_def_content_words(cls, word: str, wn_pos: str) -> Set[str]:
        toks: Set[str] = set()
        e = cls.get_ldoce_entry(word)
        p_code = {"n": "n", "v": "v", "a": "adj", "r": "adv"}.get(wn_pos, "n")
        if e:
            for s in e.get("senses") or []:
                if not s.get("pos") or cls._pos_code(s.get("pos")) == p_code:
                    d_txt = str(s.get("definition") or "").lower()
                    for w in re.findall(r"\b[a-z]{4,}\b", d_txt):
                        if w not in cls._PARONYM_STOP_DEF_TOKS:
                            toks.add(w)
                            if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) >= 5:
                                toks.add(w[:-1])
        wn = cls.get_wordnet()
        w_objs = wn.words(word, pos="a") + wn.words(word, pos="s") if wn_pos == "a" else wn.words(word, pos=wn_pos)
        for w_obj in w_objs:
            for syn in w_obj.synsets():
                d_txt = str(syn.definition() or "").lower()
                for w in re.findall(r"\b[a-z]{4,}\b", d_txt):
                    if w not in cls._PARONYM_STOP_DEF_TOKS:
                        toks.add(w)
                        if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) >= 5:
                            toks.add(w[:-1])
        return toks

    @classmethod
    def _get_derivational_bases(cls, word: str) -> Set[str]:
        """Return 2-hop WordNet derivational / pertainym base lemmas (cached)."""
        w_low = (word or "").strip().lower()
        if w_low in cls._derivational_base_cache:
            return cls._derivational_base_cache[w_low]
        wn = cls.get_wordnet()
        bases: Set[str] = set()
        try:
            for w_obj in wn.words(w_low):
                for syn in w_obj.synsets():
                    for sense in syn.senses():
                        if sense.word().lemma().lower() == w_low:
                            for rel in sense.get_related("derivation") + sense.get_related("pertainym"):
                                bases.add(rel.word().lemma().lower())
            hop2 = set(bases)
            for b in list(bases):
                for w_obj in wn.words(b):
                    for syn in w_obj.synsets():
                        for sense in syn.senses():
                            if sense.word().lemma().lower() == b:
                                for rel in sense.get_related("derivation") + sense.get_related("pertainym"):
                                    hop2.add(rel.word().lemma().lower())
        except Exception:
            hop2 = bases
        cls._derivational_base_cache[w_low] = hop2
        return hop2

    @classmethod
    def is_cross_ontology_derivational_noun_pair(cls, w1: str, w2: str) -> bool:
        """True when two root-sharing nouns belong to mutually exclusive ontological categories:
        (a) `-ance`/`-ence` abstract process vs. `-ancy`/`-ency` crisis/office/agency (`emergence` vs `emergency`), OR
        (b) `[+HUMAN]` Person/Agent (`noun.person`: `scientist`, `physicist`, `optimist`, `journalist`, `economist`)
            vs. `[-ANIMATE]` Discipline/Ideology/Field (`science`, `physics`, `optimism`, `journalism`, `economics`)."""
        a = (w1 or "").strip().lower()
        b = (w2 or "").strip().lower()
        if not a or not b or a == b or " " in a or " " in b:
            return False
        if cls._common_prefix_len(a, b) < 5:
            return False
        if (a.endswith(("ance", "ence")) and b.endswith(("ancy", "ency"))) or (
            b.endswith(("ance", "ence")) and a.endswith(("ancy", "ency"))
        ):
            wn_chk = cls.get_wordnet()
            a_ids = {s.id for w in wn_chk.words(a, pos="n") for s in w.synsets()}
            b_ids = {s.id for w in wn_chk.words(b, pos="n") for s in w.synsets()}
            return bool(a_ids and b_ids and not (a_ids & b_ids))
        if (
            (a.endswith(("ist", "ian", "ic")) or b.endswith(("ist", "ian", "ic")))
            and a.endswith(("ics", "ism", "ce", "cy", "y", "ist", "ian"))
            and b.endswith(("ics", "ism", "ce", "cy", "y", "ist", "ian"))
            and a[-3:] != b[-3:]
        ):
            wn_chk = cls.get_wordnet()
            a_syns = [s for w in wn_chk.words(a, pos="n") for s in w.synsets()]
            b_syns = [s for w in wn_chk.words(b, pos="n") for s in w.synsets()]
            a_person = bool(a_syns) and all(s.lexfile() == "noun.person" for s in a_syns if s.lexfile())
            b_person = bool(b_syns) and all(s.lexfile() == "noun.person" for s in b_syns if s.lexfile())
            a_non_person = bool(a_syns) and all(s.lexfile() not in ("noun.person", "noun.group") for s in a_syns if s.lexfile())
            b_non_person = bool(b_syns) and all(s.lexfile() not in ("noun.person", "noun.group") for s in b_syns if s.lexfile())
            if (a_person and b_non_person) or (b_person and a_non_person):
                return True
        return False

    @classmethod
    def is_safe_orthographic_paronym(
        cls,
        target_word: str,
        candidate_word: str,
        pos: str = "n",
        context_anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        quote: Optional[str] = None,
    ) -> bool:
        """Cognate Near-Synonym Guard: verifies that `candidate_word` is a semantically disjoint
        orthographic paronym or orthogonal-domain root peer of `target_word`, blocking root-sharing
        near-synonyms (e.g. historic/historical, classic/classical, economic/economical,
        continual/continuous, specimen/species, activism/activity)."""
        target = (target_word or "").strip().lower()
        cand = (candidate_word or "").strip().lower()
        if not target or not cand or target == cand:
            return False

        pos_clean = (pos or "n").strip().lower()
        wn_pos = "a" if pos_clean in ("a", "adj", "adjective", "s") else (
            "v" if pos_clean in ("v", "verb") else (
                "r" if pos_clean in ("r", "adv", "adverb") else "n"
            )
        )

        # 1. Reject inflectional / participial / negation prefix derivatives & unaligned productive-prefix compounds
        for pfx in ("un", "in", "im", "ir", "il", "non", "dis", "mis", "re", "over", "under"):
            if cand == pfx + target or target == pfx + cand:
                return False
        for c_pfx in (
            "under", "over", "inter", "trans", "super", "counter", "multi", "ultra",
            "extra", "micro", "macro", "cross", "out", "sub", "mis", "dis", "pre",
            "post", "anti", "auto", "semi", "fore", "back", "down",
        ):
            if target.startswith(c_pfx) and cand.startswith(c_pfx):
                t_st = target[len(c_pfx):]
                c_st = cand[len(c_pfx):]
                if len(t_st) >= 3 and len(c_st) >= 3:
                    # Require the stems after the shared productive prefix to also be orthographic look-alikes
                    # (e.g. allow 'overreach'/'overreact' where 'reach'/'react' have edit distance 1,
                    #  but block 'underpin'/'underlie' where 'pin'/'lie' only share the prefix 'under-').
                    if not (
                        (t_st[:2] == c_st[:2] or t_st[-2:] == c_st[-2:])
                        and cls._edit_distance_bounded(t_st, c_st) <= 2
                    ):
                        return False
        if wn_pos == "a" and cand.endswith("ed") and not target.endswith("ed"):
            return False
        if wn_pos == "n" and cand.endswith("ing") and not target.endswith("ing"):
            return False

        is_cross_ontology_deriv_noun = (
            wn_pos == "n" and cls.is_cross_ontology_derivational_noun_pair(target, cand)
        )
        if (
            wn_pos == "n"
            and cand.endswith(("ism", "ist"))
            and not target.endswith(("ism", "ist"))
            and not is_cross_ontology_deriv_noun
        ):
            return False

        # 2. Cognate Near-Synonym Guard:
        # Block -ic / -ical pairs (historic/historical, classic/classical, economic/economical, electric/electrical, comic/comical)
        if (target.endswith("ic") and cand == target + "al") or (cand.endswith("ic") and target == cand + "al"):
            return False
        # Block British/American or variant abstract noun suffixes on identical stem (specialty / speciality / specialism)
        if (
            cls._common_prefix_len(target, cand) >= 6
            and target.endswith(("ty", "ity", "ism", "ness", "ment", "tion", "sion"))
            and cand.endswith(("ty", "ity", "ism", "ness", "ment", "tion", "sion"))
        ):
            return False

        if cls.are_same_word_family(target, cand) and not is_cross_ontology_deriv_noun:
            return False

        # 3. Primary POS, Proper-Noun & Register/Taboo Verification in PedLex (when available):
        if cls._is_taboo_or_proper_noun(cand, wn_pos=wn_pos):
            return False
        p_code = {"n": "n", "v": "v", "a": "adj", "r": "adv"}.get(wn_pos, "n")
        e_cand = cls.get_ldoce_entry(cand)
        if e_cand:
            senses = e_cand.get("senses") or []
            if (
                not any(s.get("examples") for s in senses)
                and not any((e_cand.get("collocations") or {}).values())
                and not e_cand.get("thesaurus")
                and not e_cand.get("language_activator")
                and not cls.get_oxford_thesaurus_entries(cand)
            ):
                return False
            pos_senses = [s for s in senses if cls._pos_code(s.get("pos") or "") == p_code] or senses
            _blocked_reg_tokens = (
                "taboo", "offensive", "vulgar", "derogatory", "slang",
                "informal", "old-fashioned", "old use", "biblical", "trademark",
            )
            if pos_senses and all(
                any(tok in str(s.get("register") or "").lower() for tok in _blocked_reg_tokens)
                for s in pos_senses
            ):
                return False

            raw_e_pos = str(e_cand.get("pos") or "").split("|")[0].strip()
            first_sense_pos = str(senses[0].get("pos") or "").strip() if senses else ""
            cand_prim_pos = cls._pos_code(first_sense_pos or raw_e_pos)
            if cand_prim_pos and cand_prim_pos != p_code:
                # Only allow secondary POS when edit distance == 1 AND shared initial + len >= 6 (classic homophone/paronym like complement/compliment)
                strict_pos_senses = [s for s in senses if cls._pos_code(s.get("pos") or "") == p_code]
                if not strict_pos_senses or cls._edit_distance_bounded(target, cand) > 1 or target[0] != cand[0] or len(target) < 6:
                    return False

        # 4. Double-Key Collision Check (includes context_anchor & quote frame checks)
        is_dk, _ = cls.double_key_collision(
            target, cand, anchor=context_anchor, anchor_type=anchor_type, pos=wn_pos, quote=quote
        )
        if is_dk:
            return False

        # 5. WordNet Synset / Similar / Also / Hypernym Disjointness
        wn = cls.get_wordnet()
        t_words = wn.words(target, pos="a") + wn.words(target, pos="s") if wn_pos == "a" else wn.words(target, pos=wn_pos)
        c_words = wn.words(cand, pos="a") + wn.words(cand, pos="s") if wn_pos == "a" else wn.words(cand, pos=wn_pos)
        t_syns = [s for w in t_words for s in w.synsets()]
        c_syns = [s for w in c_words for s in w.synsets()]
        if not t_syns or not c_syns:
            return False
        # Reject candidates that exist only as capitalized proper nouns/demonyms in WordNet
        if all(
            all(lemma and lemma[0].isupper() for lemma in s.lemmas())
            for s in c_syns if s.lemmas()
        ):
            return False
        # For nouns, block jarring concrete-object vs. abstract-state clashes
        # (e.g., blocking 'leopard' [animal] for 'jeopardy' [abstract danger] and 'breast' [body] for 'breadth' [abstract quality],
        # while still permitting long root-sharing morphological paronyms like 'specimen' [len 8] vs. 'speculation').
        if wn_pos == "n":
            _abs_lfs = {
                "noun.state", "noun.attribute", "noun.cognition", "noun.feeling",
                "noun.communication", "noun.act", "noun.event", "noun.phenomenon",
                "noun.time", "noun.relation",
            }
            _bio_conc_lfs = {
                "noun.animal", "noun.plant", "noun.food", "noun.body", "noun.substance",
            }
            _all_conc_lfs = _bio_conc_lfs | {"noun.artifact", "noun.object"}
            t_lfs_all = [s.lexfile() for s in t_syns if s.lexfile()]
            c_prim_lf = c_syns[0].lexfile() if c_syns[0].lexfile() else ""
            if t_lfs_all and c_prim_lf:
                if (all(lf in _abs_lfs for lf in t_lfs_all) and c_prim_lf in _bio_conc_lfs) or (
                    all(lf in _bio_conc_lfs for lf in t_lfs_all) and c_prim_lf in _abs_lfs
                ):
                    return False
                if cls._common_prefix_len(target, cand) < 4 or len(target) < 8:
                    if (all(lf in _abs_lfs for lf in t_lfs_all) and c_prim_lf in _all_conc_lfs) or (
                        all(lf in _all_conc_lfs for lf in t_lfs_all) and c_prim_lf in _abs_lfs
                    ):
                        return False
        t_ids = {s.id for s in t_syns}
        c_ids = {s.id for s in c_syns}
        if t_ids & c_ids:
            return False

        t_ext = set(t_ids)
        for s in t_syns:
            for r in s.get_related("similar") + s.get_related("also"):
                t_ext.add(r.id)
            for h in s.hypernyms():
                if len(h.hyponyms()) <= 50:
                    t_ext.add(h.id)
        c_ext = set(c_ids)
        for s in c_syns:
            for r in s.get_related("similar") + s.get_related("also"):
                c_ext.add(r.id)
            for h in s.hypernyms():
                if len(h.hyponyms()) <= 50:
                    c_ext.add(h.id)
        if t_ext & c_ext:
            return False

        # 6. Root-Sharing Cognate & Compound Head-Sharing Semantic Overlap Check:
        # If target and cand share a 4+ char prefix (same Latin/Greek root, e.g. activism/activity, specimen/species, continual/continuous)
        # OR share a 5+ char compound head suffix (excluding derivational suffixes, e.g. shareholder/stakeholder, ecological/biological),
        # block if they share content words across their PedLex+WN definitions OR share 2-hop derivational bases (for nouns/verbs).
        pref = cls._common_prefix_len(target, cand)
        suff = cls._common_suffix_len(target, cand)
        is_deriv_suffix_only = target.endswith((
            "itude", "ation", "ition", "tion", "sion", "ment", "ness", "ship",
            "ance", "ence", "ics", "ist", "ism", "ity", "ure",
        ))
        if not is_cross_ontology_deriv_noun and (pref >= 4 or (suff >= 5 and not is_deriv_suffix_only)):
            t_defs = cls._get_def_content_words(target, wn_pos)
            c_defs = cls._get_def_content_words(cand, wn_pos)
            if t_defs & c_defs:
                return False
            if cand in t_defs or target in c_defs:
                return False
            if wn_pos in ("n", "v"):
                if cls._get_derivational_bases(target) & cls._get_derivational_bases(cand):
                    return False

        return True

    @classmethod
    def get_orthographic_paronym_distractors(
        cls,
        target_word: str,
        pos: str = "n",
        target_count: int = 3,
        exclude_words: Optional[Set[str]] = None,
        context_anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        quote: Optional[str] = None,
        return_scores: bool = False,
    ) -> Union[List[str], List[Tuple[str, int]]]:
        """Mine psychometrically sharp orthographic look-alikes (`形近词`) and orthogonal-domain
        root paronyms (`同根异义词`) guarded by `is_safe_orthographic_paronym`."""
        tw = (target_word or "").strip().lower()
        if not tw or len(tw) < 4 or not tw.isalpha():
            return []

        pos_clean = (pos or "n").strip().lower()
        wn_pos = "a" if pos_clean in ("a", "adj", "adjective", "s") else (
            "v" if pos_clean in ("v", "verb") else (
                "r" if pos_clean in ("r", "adv", "adverb") else "n"
            )
        )
        excl = {w.lower() for w in (exclude_words or set())} | {tw}

        idx = cls._ensure_orthographic_pos_index()
        by_first = idx["by_first"].get(wn_pos, {})
        by_second = idx["by_second"].get(wn_pos, {})
        by_skel = idx.get("by_skel", {}).get(wn_pos, {})
        by_rhyme = idx.get("by_rhyme", {}).get(wn_pos, {})
        by_suffix = idx.get("by_suffix", {}).get(wn_pos, {})

        tw_skel = cls._phonetic_consonant_skeleton(tw)
        tw_rhyme = cls._phonetic_rhyme_tail(tw)
        tw_syl = cls._estimate_syllables(tw)

        raw_pool = set(by_first.get(tw[0], [])) | set(by_second.get(tw[1], []))
        if len(tw_skel) >= 3:
            raw_pool |= set(by_skel.get(tw_skel, []))
        if len(tw_rhyme) >= 3:
            raw_pool |= set(by_rhyme.get(tw_rhyme, []))
        _acad_sfx = ("ics", "ist", "ism", "ship", "ware", "type", "ology", "archy", "cracy")
        if wn_pos == "n":
            for sfx in _acad_sfx:
                if tw.endswith(sfx) and len(tw) >= len(sfx) + 2:
                    raw_pool |= set(by_suffix.get(sfx, []))
            if len(tw) >= 6:
                raw_pool |= set(by_suffix.get(tw[-4:], []))

        scored: List[Tuple[int, str]] = []
        for w in raw_pool:
            is_acad_sfx_pair = wn_pos == "n" and any(tw.endswith(sf) and w.endswith(sf) for sf in _acad_sfx)
            max_len_diff = 4 if is_acad_sfx_pair else 3
            if w in excl or abs(len(w) - len(tw)) > max_len_diff:
                continue
            if anchor_type == "modified_noun" and wn_pos == "a" and w in cls._PREDICATIVE_ONLY_ADJS:
                continue
            pref = cls._common_prefix_len(tw, w)
            suff = cls._common_suffix_len(tw, w)
            ed = cls._edit_distance_bounded(tw, w)
            w_skel = cls._phonetic_consonant_skeleton(w)
            w_rhyme = cls._phonetic_rhyme_tail(w)
            w_syl = cls._estimate_syllables(w)
            skel_match = bool(tw_skel and w_skel == tw_skel and abs(len(w) - len(tw)) <= 2)
            rhyme_containment = bool(
                wn_pos == "n"
                and min(len(tw), len(w)) >= 5
                and abs(len(w) - len(tw)) <= 3
                and (w.endswith(tw) or tw.endswith(w))
                and w_rhyme == tw_rhyme
            )
            deriv_rhyme_match = bool(
                wn_pos == "n"
                and w_rhyme == tw_rhyme
                and abs(w_syl - tw_syl) <= 1
                and (
                    (tw.endswith(("tion", "sion")) and w.endswith(("tion", "sion")) and suff >= 4)
                    or (tw.endswith(("ance", "ence", "ment")) and w.endswith(("ance", "ence", "ment")) and suff >= 5)
                )
            )
            acad_sfx_match = bool(
                is_acad_sfx_pair
                and abs(w_syl - tw_syl) <= 2
            )
            rhyme_match = bool(
                (len(tw_rhyme) >= 3 and w_rhyme == tw_rhyme and ed <= 2)
                or rhyme_containment
                or deriv_rhyme_match
            )
            if (
                ed <= 2
                or skel_match
                or rhyme_match
                or acad_sfx_match
                or (pref >= 4 and len(tw) >= 6 and abs(len(w) - len(tw)) <= 3)
                or (pref >= 3 and suff >= 2)
            ):
                score = 0
                if ed == 1:
                    score += 45
                elif ed == 2:
                    score += 26
                score += pref * 4 + suff * 2 - (0 if acad_sfx_match else abs(len(w) - len(tw)) * 2)
                if skel_match:
                    if ed <= 2:
                        score += 18
                    elif w_syl == tw_syl:
                        score += 14
                if rhyme_match and w_syl == tw_syl:
                    score += 12
                if deriv_rhyme_match:
                    score += 36 if suff >= 6 else (32 if suff >= 5 else 26)
                    if suff >= 6:
                        score += 24
                elif rhyme_containment:
                    score += 30
                if acad_sfx_match:
                    score += 44 + (4 if abs(w_syl - tw_syl) <= 1 else 0)
                if deriv_rhyme_match or acad_sfx_match:
                    w_ldoce = cls.get_ldoce_entry(w) or {}
                    if int(w_ldoce.get("core_level") or 0) > 0:
                        score += 16
                if wn_pos == "a" and pref >= 6:
                    score += 18
                if wn_pos == "n" and pref >= 4:
                    if not w.endswith(("ist", "or", "er", "ian", "ty", "ity", "ism")) or (
                        tw.endswith(("ist", "ian", "ty", "ity", "ism", "ics")) or pref >= 5
                    ):
                        score += 14
                    if pref >= 5 and cls.is_cross_ontology_derivational_noun_pair(tw, w):
                        score += 52
                scored.append((score, w))

        scored.sort(key=lambda x: (-x[0], x[1]))
        # Lazily verify CEFR, orthographic paronym safety, and deduplicate mutual word-family members
        selected: List[Tuple[str, int]] = []
        for sc, w in scored:
            if not cls.is_cefr_compliant_distractor(w, tw):
                continue
            if not cls.is_safe_orthographic_paronym(
                tw, w, pos=wn_pos, context_anchor=context_anchor, anchor_type=anchor_type, quote=quote
            ):
                continue
            if any(cls.are_same_word_family(w, prev_w) for prev_w, _ in selected):
                continue
            selected.append((w, sc))
            if len(selected) >= target_count:
                break

        if return_scores:
            return selected
        return [w for w, _ in selected]

    # -------------------------------------------------------------------------
    # Double-Key Clearance (Frame-Aware Distractor Audit)
    # -------------------------------------------------------------------------
    # Empirically validated 2026-09-25 against Book_3_Unit_4_Section_A:
    #   * Prep items  -> double-key = distractor shares the target's bound
    #     preposition (OCD) AND sits in the target's semantic neighborhood.
    #     (Shared frame alone is NOT sufficient: break-with is valid but not a
    #     substitute for coincide-with. overlap-with and liberty-from ARE.)
    #   * Slot items  -> double-key = distractor is a near-synonym / troponym
    #     / hyponym of the target (same semantic slot). Sense antonyms are
    #     deliberately whitelisted: they are contrast distractors eliminated by
    #     sentence polarity, not double-keys.
    #     Coordinate-term (sister-synset) expansion is deliberately EXCLUDED —
    #     it over-matches (break ~ coincide) and yields false double-keys.
    _semantic_fields_cache: Dict[Tuple[str, str], Tuple[Set[str], Set[str]]] = {}

    @classmethod
    def semantic_fields(cls, word: str, pos: Optional[str] = None) -> Tuple[Set[str], Set[str]]:
        """Returns (near_set, antonyms) for a lemma, WordNet-derived.

        near_set  : same-synset words, bipolar similar/also (2 hops),
                    hypernym / entailment / hyponym lemmas.
        antonyms  : sense-locked antonyms (+ their satellite cluster).
        Deterministic; never raises."""
        target = (word or "").strip().lower()
        pos_clean = (pos or "").strip().lower()
        cache_key = (target, pos_clean)
        cached = cls._semantic_fields_cache.get(cache_key)
        if cached is not None:
            return cached

        wn = cls.get_wordnet()
        near: Set[str] = set()
        antonyms: Set[str] = set()

        def add(syn) -> None:
            for w in syn.words():
                near.add(w.lemma().lower())

        if pos_clean in ("noun", "n"):
            pos_tags = ("n",)
        elif pos_clean in ("verb", "v"):
            pos_tags = ("v",)
        elif pos_clean in ("adjective", "adj", "a", "s"):
            pos_tags = ("a", "s")
        elif pos_clean in ("adverb", "adv", "r"):
            pos_tags = ("r",)
        else:
            pos_tags = ("a", "s", "v", "n", "r")

        seeds = []
        for p_tag in pos_tags:
            try:
                seeds += wn.synsets(target, p_tag)
            except Exception:
                pass
        if not seeds and pos_tags != ("a", "s", "v", "n", "r"):
            for p_tag in ("a", "s", "v", "n", "r"):
                try:
                    seeds += wn.synsets(target, p_tag)
                except Exception:
                    pass

        for s in seeds:
            add(s)
            for rel in ("similar", "also"):
                for t in s.get_related(rel):
                    add(t)
                    if t.pos in ("a", "s"):
                        for rel2 in ("similar", "also"):
                            for u in t.get_related(rel2):
                                add(u)
            for rel in ("hypernym", "entailment", "hyponym"):
                for t in s.get_related(rel):
                    add(t)
            # Sense-locked antonyms (same cluster for adjectives)
            for sense in s.senses():
                for ant_sense in sense.get_related("antonym"):
                    antonyms.add(ant_sense.word().lemma().lower())
                    for rel in ("similar", "also"):
                        for t in ant_sense.synset().get_related(rel):
                            for w in t.words():
                                antonyms.add(w.lemma().lower())

        ped_ox_syns, ped_ox_ants = cls.get_pedlex_and_oxford_synonyms_antonyms(target, pos=pos)
        for psyn in ped_ox_syns:
            if " " not in psyn:
                near.add(psyn)
        for pant in ped_ox_ants:
            if " " not in pant:
                antonyms.add(pant)

        near.discard(target)
        antonyms.discard(target)
        res = (near, antonyms)
        cls._semantic_fields_cache[cache_key] = res
        return res

    GENERIC_COLLOCATION_MODIFIERS: Set[str] = {
        "big", "bigger", "biggest", "large", "larger", "largest",
        "huge", "enormous", "massive", "great", "greater", "greatest",
        "small", "smaller", "smallest", "tiny", "little",
        "good", "better", "best", "bad", "worse", "worst", "fine",
        "new", "old", "strong", "weak", "high", "higher", "highest", "low", "lower", "lowest",
        "deep", "different", "certain", "various", "several", "many", "much",
        "more", "most", "less", "least", "real", "true", "full", "complete",
        "major", "main", "key", "top", "leading", "primary", "single", "only", "first", "last"
    }
    GENERIC_COLLOCATION_VERBS: Set[str] = {
        "have", "has", "had", "make", "made", "get", "got", "give", "gave",
        "take", "took", "do", "did", "done", "see", "saw", "find", "found", "use", "used"
    }
    COLLECTIVE_GROUP_NOUNS: Set[str] = {
        "team", "company", "crew", "committee", "board", "panel", "unit",
        "staff", "service", "force", "group", "crowd", "squad", "cast", "faculty"
    }

    @classmethod
    def get_syntactic_pattern_slots(cls, word_or_entry: Union[str, Dict[str, Any]], sense: Optional[Dict[str, Any]] = None) -> Set[str]:
        """Extracts normalized syntactic complement pattern slots for a headword from PedLex 6th Edition.
        
        Returned slots can include:
          - 'to_inf'     : infinitive complement ('to do sth')
          - 'gerund'     : gerund complement ('doing sth') without bound prepositions
          - 'sb_to_do'   : complex transitive with infinitive ('somebody to do sth')
          - 'that_clause': clausal complement ('that ...' or '(that)')
          - 'wh_clause'  : indirect question / wh-clause ('who/what/whether etc')
        """
        if isinstance(word_or_entry, dict):
            entry = word_or_entry
        else:
            entry = cls.get_ldoce_entry(str(word_or_entry).strip().lower())
        if not entry:
            return set()

        target_senses = [sense] if sense else entry.get("senses", [])
        slots: Set[str] = set()
        for s in target_senses:
            for p in s.get("patterns", []):
                pl = p.lower()
                if "to do" in pl:
                    if any(x in pl for x in ("sb to do", "somebody to do", "someone to do")):
                        slots.add("sb_to_do")
                    else:
                        slots.add("to_inf")
                if "doing" in pl:
                    # Normalize away parentheses around '(doing)' and slash-grouped prepositions so
                    # prepositional gerunds like 'decide against/in favour of (doing) something' are excluded
                    pl_norm = re.sub(r"[\(\)]", "", pl)
                    if not re.search(
                        r"\b(?:of|in|for|from|against|about|with|upon|on|at|to|into|by|over|between|through|without|after|before|despite)(?:/[a-z\s]+)*\s+(?:[^\s]+\s+)?doing\b",
                        pl_norm,
                    ):
                        slots.add("gerund")
                if "(that)" in pl or "that" in pl:
                    if not pl.startswith("the ") and not pl.startswith("in the "):
                        slots.add("that_clause")
                if any(x in pl for x in ("wh-", "how/where", "who/what", "whether")):
                    slots.add("wh_clause")
            # Example-based inference fallback if patterns field is silent
            for ex in s.get("examples", []):
                ex_l = ex.lower()
                w = entry.get("word", "").lower()
                if f"{w} to " in ex_l:
                    slots.add("to_inf")
                elif re.search(rf"\b{re.escape(w)}\s+[a-z]+ing\b", ex_l):
                    slots.add("gerund")
        for gb in entry.get("grammar_boxes") or []:
            syn_c = gb.get("syntactic_constraints") or {}
            for af in syn_c.get("allowed_frames") or []:
                if af in ("to_inf", "gerund", "that_clause", "sb_to_do", "wh_clause"):
                    slots.add(af)
        return slots

    _NOUN_COUNTABILITY_CONTRAST_MAP: Dict[str, List[str]] = {
        "device": ["equipment", "hardware", "machinery"],
        "devices": ["equipment", "hardware", "machinery"],
        "appliance": ["equipment", "hardware", "machinery"],
        "appliances": ["equipment", "hardware", "machinery"],
        "instrument": ["equipment", "apparatus", "machinery"],
        "gadget": ["equipment", "hardware"],
        "tool": ["equipment", "hardware", "machinery"],
        "machine": ["machinery", "equipment", "hardware"],
        "facility": ["equipment", "infrastructure", "funding"],
        "facilities": ["equipment", "infrastructure", "funding"],
        "suggestion": ["advice", "guidance", "counsel"],
        "proposal": ["advice", "guidance", "legislation"],
        "recommendation": ["advice", "guidance", "feedback"],
        "tip": ["advice", "guidance", "information"],
        "breakthrough": ["progress", "headway", "research"],
        "advance": ["progress", "headway"],
        "stride": ["progress", "headway"],
        "achievement": ["progress", "success"],
        "belongings": ["luggage", "baggage", "property"],
        "valuables": ["wealth", "jewelry", "luggage"],
        "possession": ["property", "wealth", "luggage"],
        "possessions": ["property", "wealth", "luggage"],
        "clothes": ["clothing", "apparel", "footwear"],
        "garment": ["clothing", "apparel"],
        "outfit": ["clothing", "apparel"],
        "costume": ["clothing", "apparel"],
        "finding": ["information", "evidence", "research"],
        "findings": ["information", "evidence", "research"],
        "fact": ["information", "evidence", "knowledge"],
        "detail": ["information", "evidence"],
        "clue": ["evidence", "information"],
        "statistic": ["information", "data", "research"],
        "study": ["research", "evidence", "literature"],
        "experiment": ["research", "testing"],
        "law": ["legislation", "regulation"],
        "statute": ["legislation", "law"],
        "rule": ["legislation", "guidance"],
        "loan": ["credit", "funding", "finance"],
        "grant": ["funding", "finance", "aid"],
        "subsidy": ["funding", "finance", "assistance"],
        "donation": ["funding", "charity", "aid"],
        "resource": ["funding", "equipment", "wealth"],
        "resources": ["funding", "equipment", "wealth"],
        "comment": ["feedback", "criticism", "advice"],
        "remark": ["feedback", "commentary"],
        "remedy": ["assistance", "relief", "treatment"],
        "measure": ["legislation", "action", "assistance"],
        "emission": ["pollution", "contamination"],
        "emissions": ["pollution", "contamination"],
        "pollutant": ["pollution", "contamination", "waste"],
        "contaminant": ["pollution", "contamination", "waste"],
        "chair": ["furniture", "seating"],
        "table": ["furniture", "equipment"],
        "desk": ["furniture", "equipment"],
        "cabinet": ["furniture", "storage"],
        "vehicle": ["traffic", "transportation", "transport"],
        "poem": ["poetry", "literature", "verse"],
        "view": ["scenery", "landscape"],
        "task": ["homework", "work", "labor"],
        "assignment": ["homework", "work", "research"],
        "chore": ["housework", "work", "labor"],
        "job": ["employment", "work", "labor"],
        "product": ["merchandise", "produce", "goods"],
        "item": ["merchandise", "equipment", "baggage"],
        "commodity": ["merchandise", "commerce", "trade"],
        "permit": ["permission", "authorization", "consent"],
        "license": ["permission", "authorization"],
        "program": ["software", "programming", "funding"],
        "application": ["software", "technology"],
    }

    @classmethod
    def get_word_gram_profile(cls, word: str, pos: Optional[str] = None) -> Dict[str, Any]:
        """Extracts a unified grammatical constraint profile for `word` from `pedlex.db` (`senses`, `thesaurus`, and `grammar_boxes`)."""
        w_clean = (word or "").strip().lower()
        profile: Dict[str, Any] = {
            "only_before_noun": False,
            "not_before_noun": w_clean in cls._PREDICATIVE_ONLY_ADJS,
            "pure_uncountable": False,
            "pure_countable": False,
            "pure_plural": False,
            "pure_intransitive": False,
            "no_passive": False,
            "no_progressive": False,
            "forbidden_frames": set(),
            "allowed_frames": set(),
            "comparison_peers": set(),
        }
        if not w_clean:
            return profile
        entry = cls.get_ldoce_entry(w_clean)
        if not entry:
            return profile

        pos_norm = (pos or "").strip().lower()
        wn_pos = "a" if pos_norm.startswith("a") or pos_norm == "j" else (
            "n" if pos_norm.startswith("n") else (
                "v" if pos_norm.startswith("v") else ""
            )
        )

        senses = entry.get("senses") or []
        matching_senses = []
        for s in senses:
            sp = (s.get("pos") or entry.get("pos") or "").lower()
            if wn_pos == "a" and "adj" in sp:
                matching_senses.append(s)
            elif wn_pos == "n" and "noun" in sp:
                matching_senses.append(s)
            elif wn_pos == "v" and "verb" in sp:
                matching_senses.append(s)
            elif not wn_pos:
                matching_senses.append(s)
        if not matching_senses:
            matching_senses = senses

        raw_sense_grams = [str(s.get("grammar") or "").strip().lower() for s in matching_senses]
        non_empty_sense_grams = [g for g in raw_sense_grams if g]
        # A word-level constraint holds across the POS when EVERY matching sense has a grammar tag
        # satisfying the constraint (e.g., entryhead <gram> propagated to all senses), OR when no
        # sense carries a <gram> tag and the self-referential THESAURUS exponent supplies it
        # (e.g., 'added', 'additional'). If a polysemous word has 1 restricted sense and other
        # unrestricted senses (grammar == ""), it is NOT globally restricted across all senses.
        if matching_senses and len(non_empty_sense_grams) == len(matching_senses):
            grams = list(non_empty_sense_grams)
        elif not non_empty_sense_grams:
            grams = [
                str(th["grammar"]).strip().lower()
                for th in (entry.get("thesaurus") or [])
                if (th.get("word") or "").strip().lower() == w_clean and th.get("grammar")
            ]
        else:
            grams = []

        if grams:
            if all("only before noun" in g for g in grams):
                profile["only_before_noun"] = True
            if all("not before noun" in g or "never before noun" in g for g in grams):
                # Guard against PedLex sense grammar pollution by embedded sub-phrases
                # (e.g., `out of this world [not before noun]` under `incredible`, which has attested noun collocations)
                has_noun_collocs = bool((entry.get("collocations") or {}).get("nouns"))
                if not has_noun_collocs or w_clean in cls._PREDICATIVE_ONLY_ADJS:
                    profile["not_before_noun"] = True
            if (not wn_pos or wn_pos == "n") and all(
                "uncountable" in g and not re.search(r"\bcountable\b", g)
                for g in grams
            ):
                profile["pure_uncountable"] = True
            if (not wn_pos or wn_pos == "v") and all(
                "intransitive" in g and not re.search(r"\btransitive\b", g)
                for g in grams
            ):
                profile["pure_intransitive"] = True

        if not wn_pos or wn_pos == "n":
            if (
                not w_clean.endswith("s")
                and raw_sense_grams
                and "uncountable" in raw_sense_grams[0]
                and not any(re.search(r"\bcountable\b", g) for g in raw_sense_grams)
                and not any("plural" in g for g in raw_sense_grams)
            ):
                profile["pure_uncountable"] = True
            if (
                raw_sense_grams
                and bool(re.search(r"\bcountable\b", raw_sense_grams[0]))
                and not any("uncountable" in g for g in raw_sense_grams)
            ):
                profile["pure_countable"] = True
            if (
                raw_sense_grams
                and "plural" in raw_sense_grams[0]
                and not any("uncountable" in g for g in raw_sense_grams)
            ):
                profile["pure_plural"] = True

        for gb in entry.get("grammar_boxes") or []:
            for p in gb.get("comparison_peers") or []:
                p_clean = str(p).strip().lower()
                if p_clean and p_clean != w_clean and " " not in p_clean:
                    profile["comparison_peers"].add(p_clean)
            syn_c = gb.get("syntactic_constraints") or {}
            # Only apply no_passive / no_progressive globally when box is not restricted to a subset of senses
            gb_senses = gb.get("sense_indices") or []
            if not gb_senses or len(gb_senses) >= len(matching_senses):
                if syn_c.get("no_passive"):
                    profile["no_passive"] = True
                if syn_c.get("no_progressive"):
                    profile["no_progressive"] = True
            for ff in syn_c.get("forbidden_frames") or []:
                profile["forbidden_frames"].add(ff)
            for af in syn_c.get("allowed_frames") or []:
                profile["allowed_frames"].add(af)

        return profile

    @classmethod
    def evaluate_syntactic_slot_clash(
        cls,
        target: str,
        candidate: str,
        anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        pos: Optional[str] = None,
        quote: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Determines whether `candidate` is grammatically barred from occupying `target`'s syntactic slot in `quote`/`anchor_type`.
        
        Returns `(has_clash, reason)` where `has_clash == True` guarantees that `candidate` is syntactically ruled out
        (making it a high-discrimination single-fit distractor and immune to synonym double-key collisions).
        """
        t = (target or "").strip().lower()
        c = (candidate or "").strip().lower()
        if not t or not c or t == c:
            return False, ""

        c_prof = cls.get_word_gram_profile(c, pos=pos)
        t_prof = cls.get_word_gram_profile(t, pos=pos)
        # Also check if `t`'s own THESAURUS box explicitly constrains `c` under the shared synonym sense
        # (e.g. `extra` THESAURUS box labels `spare [only before noun]`, `complete` THESAURUS box labels `done [not before noun]`)
        t_entry = cls.get_ldoce_entry(t)
        if t_entry:
            t_self_countable = any(
                (th.get("word") or "").strip().lower() == t
                and "countable" in str(th.get("grammar") or "").lower()
                and "uncountable" not in str(th.get("grammar") or "").lower()
                for th in (t_entry.get("thesaurus") or [])
            )
            for th in t_entry.get("thesaurus") or []:
                if (th.get("word") or "").strip().lower() == c and th.get("grammar"):
                    th_g = str(th["grammar"]).lower()
                    if "only before noun" in th_g:
                        c_prof["only_before_noun"] = True
                    if "not before noun" in th_g or "never before noun" in th_g:
                        c_prof["not_before_noun"] = True
                    if "uncountable" in th_g and not re.search(r"\bcountable\b", th_g):
                        c_prof["pure_uncountable"] = True
                        if t_self_countable:
                            t_prof["pure_countable"] = True
        pos_norm = (pos or "").strip().lower()
        wn_pos = "a" if pos_norm.startswith("a") or pos_norm == "j" else (
            "n" if pos_norm.startswith("n") else (
                "v" if pos_norm.startswith("v") else ""
            )
        )

        # Inspect quote via spaCy if provided
        is_attr_adj_slot = anchor_type == "modified_noun"
        is_pred_adj_slot = False
        has_count_det_or_plural = False
        is_passive_verb_slot = False
        has_direct_obj_slot = anchor_type in ("object", "sb_to_do")
        active_verb_frame = anchor_type if anchor_type in ("to_inf", "gerund", "that_clause", "sb_to_do") else None

        if quote:
            nlp = cls.get_spacy()
            q_doc = nlp(quote)
            for tok in q_doc:
                if cls.form_matches(tok.lemma_.lower(), t) or cls.form_matches(tok.text.lower(), t):
                    if wn_pos == "a" or tok.pos_ == "ADJ":
                        if tok.dep_ in ("amod", "compound", "nmod") or (tok.head and tok.head.pos_ in ("NOUN", "PROPN") and tok.i < tok.head.i):
                            is_attr_adj_slot = True
                        elif tok.dep_ in ("acomp", "oprd", "attr", "advcl") or (
                            tok.dep_ == "ROOT" and any(ch.dep_ in ("cop", "aux", "nsubj", "nsubjpass") for ch in tok.children)
                        ):
                            is_pred_adj_slot = True
                    elif wn_pos == "n" or tok.pos_ == "NOUN":
                        if tok.tag_ == "NNS" or any(
                            ch.dep_ in ("det", "nummod")
                            and ch.lemma_.lower() in ("a", "an", "one", "two", "three", "many", "few", "several", "each", "every", "another", "both")
                            for ch in tok.children
                        ):
                            has_count_det_or_plural = True
                    elif wn_pos == "v" or tok.pos_ == "VERB":
                        if any(ch.dep_ in ("auxpass", "aux:pass", "nsubjpass", "nsubj:pass") for ch in tok.children):
                            is_passive_verb_slot = True
                        if any(ch.dep_ in ("dobj", "obj") for ch in tok.children):
                            has_direct_obj_slot = True
                        if not active_verb_frame:
                            if any(ch.dep_ in ("dobj", "obj") for ch in tok.children) and any(
                                ch.dep_ == "xcomp" and any(gc.lemma_.lower() == "to" for gc in ch.children)
                                for ch in tok.children
                            ):
                                active_verb_frame = "sb_to_do"
                            elif any(
                                ch.dep_ in ("ccomp", "xcomp")
                                and any(gc.dep_ in ("nsubj", "dobj", "obj") for gc in ch.children)
                                and any(gc.lemma_.lower() == "to" for gc in ch.children)
                                for ch in tok.children
                            ):
                                active_verb_frame = "sb_to_do"
                    break

        # 1. Adjective positional slot clash ([only before noun] vs predicative; [not before noun] vs attributive)
        if wn_pos == "a" or (not wn_pos and (c_prof["only_before_noun"] or c_prof["not_before_noun"])):
            if is_pred_adj_slot and c_prof["only_before_noun"] and not t_prof["only_before_noun"]:
                return True, "adj_only_before_noun_in_predicative_slot"
            if is_attr_adj_slot and c_prof["not_before_noun"] and not t_prof["not_before_noun"]:
                return True, "adj_not_before_noun_in_attributive_slot"

        # 2. Noun countability clash (pure [uncountable] in countable determiner/plural slot)
        if (wn_pos == "n" or not wn_pos) and (
            has_count_det_or_plural
            or t_prof.get("pure_countable")
            or t_prof.get("pure_plural")
            or (t.endswith("s") and not t.endswith("ics") and not t_prof.get("pure_uncountable"))
        ):
            if c_prof["pure_uncountable"] and not t_prof["pure_uncountable"]:
                return True, "uncountable_noun_in_countable_slot"

        # 3. Verb transitivity / passive / forbidden complement frame clash
        if wn_pos == "v" or not wn_pos:
            if is_passive_verb_slot and (c_prof["no_passive"] or c_prof["pure_intransitive"]):
                return True, "no_passive_or_intransitive_in_passive_slot"
            if has_direct_obj_slot and c_prof["pure_intransitive"] and not t_prof["pure_intransitive"]:
                return True, "intransitive_verb_in_transitive_object_slot"
            if active_verb_frame and (
                active_verb_frame in c_prof["forbidden_frames"]
                or (active_verb_frame == "sb_to_do" and "V_sb_to_do" in c_prof["forbidden_frames"])
                or (active_verb_frame == "to_inf" and "V_to_do" in c_prof["forbidden_frames"])
            ):
                return True, f"forbidden_frame_{active_verb_frame}"

        return False, ""

    @classmethod
    def double_key_collision(cls, target: str, distractor: str,
                             anchor: Optional[str] = None,
                             anchor_type: Optional[str] = None,
                             pos: Optional[str] = None,
                             quote: Optional[str] = None) -> Tuple[bool, str]:
        """Deterministic double-key detector. Returns (is_double_key, kind).

        kind is one of:
          "double_key_prep"        : shared bound preposition + semantic neighbor
          "double_key_slot"        : same semantic slot (near-synonym/coordinate/troponym)
          "double_key_collocation" : shared collocation with anchor
          "frame_only"             : shared bound preposition or syntactic frame, semantically distinct
                                     (NOT a double-key; needs a specific exclusion clue)
          "none"                   : clear
        """
        t = (target or "").strip().lower()
        d = (distractor or "").strip().lower()
        if not t or not d or t == d:
            return False, "none"
        if cls.are_same_word_family(t, d) and not cls.is_cross_ontology_derivational_noun_pair(t, d):
            return True, "double_key_family"

        t_lem = cls.lemma_of(t) if " " not in t else t
        d_lem = cls.lemma_of(d) if " " not in d else d

        # Same-Head Dual-Preposition Double-Key Guard:
        # Multi-word phrases sharing the exact same content headword and differing only in a relational
        # preposition (e.g. 'equate to' vs 'equate with', 'compare to' vs 'compare with', 'free of' vs 'free from',
        # 'obsessed with' vs 'obsessed by', 'integrate into' vs 'integrate with', 'hinge on' vs 'hinge upon')
        # are interchangeable or co-attested in standard English when no syntactic complement contrast exists.
        t_words = t.split()
        d_words = d.split()
        if len(t_words) == 2 and len(d_words) == 2 and t_words[0] == d_words[0] and t_words[1] != d_words[1]:
            _rel_preps = {
                "to", "with", "into", "from", "of", "upon", "on", "for", "at",
                "by", "about", "as", "against", "between", "among", "towards", "toward", "in", "over",
            }
            if t_words[1] in _rel_preps and d_words[1] in _rel_preps:
                _adv_particles = {"up", "down", "out", "off", "away", "back", "along", "ahead", "apart", "aside", "around", "through", "together"}
                if t_words[1] not in _adv_particles and d_words[1] not in _adv_particles:
                    return True, "double_key_prep"

        # Construction Family Subcategorization Gate (`pedlex_cxn_index`):
        # When `anchor_type` or `quote` locks one of the 8 construction families, any candidate in
        # `ban_groups` (sharing the target's subcategorization frame in the same semantic/syntactic class)
        # is a constructional double-key, whereas verified `contrast_groups` candidates are single-fit safe.
        if anchor_type in (
            "it_extrapos", "gerund", "to_inf", "sb_to_do",
            "shell_to_do", "shell_of_doing", "shell_that",
            "correl_pair", "fronted_inv", "npi", "mandative_subj", "lv_triad",
        ):
            cxn_cands, active_cxn = cls.get_construction_contrast_distractors(
                t_lem,
                pos=pos or "noun",
                quote=quote,
                context_anchor=anchor,
                anchor_type=anchor_type,
                target_count=12,
                require_context_match=True,
            )
            if active_cxn:
                cxn_banned = cls.get_cxn_banned_words(active_cxn.get("ban_groups"))
                if d in cxn_banned or d_lem in cxn_banned:
                    return True, "double_key_pattern"
                if d in cxn_cands or d_lem in cxn_cands:
                    return False, "none"

        # Syntactic Disjointness Exemption Gate (<GRAM> / THESAURUS / grammar_boxes):
        # If distractor `d` has a hard grammatical clash with the syntactic slot of `t` in `quote`/`anchor_type`
        # (e.g., [only before noun] in predicative slot, [not before noun] in attributive slot,
        # pure [uncountable] after 'a/an/many', or no_passive/intransitive in passive/object slot),
        # `d` is 100% syntactically eliminated and cannot be a double key!
        has_slot_clash, clash_reason = cls.evaluate_syntactic_slot_clash(
            t, d, anchor=anchor, anchor_type=anchor_type, pos=pos, quote=quote
        )
        if has_slot_clash:
            if clash_reason.startswith("forbidden_frame_"):
                return False, "none"
            return False, "frame_only"

        near, antonyms = cls.semantic_fields(t, pos=pos)
        if t_lem != t:
            near_lem, ant_lem = cls.semantic_fields(t_lem, pos=pos)
            near |= near_lem
            antonyms |= ant_lem
        ped_ox_syns, ped_ox_ants = cls.get_pedlex_and_oxford_synonyms_antonyms(t, pos=pos)
        for sw in ped_ox_syns:
            if " " not in sw and sw not in (t, t_lem):
                near.add(sw)
        for aw in ped_ox_ants:
            if " " not in aw and aw not in (t, t_lem):
                antonyms.add(aw)
        # Also include PedLex thesaurus words and grammar_boxes comparison_peers in near semantic field
        for q_t in ({t, t_lem}):
            for item in cls.get_ldoce_thesaurus(q_t):
                raw_w = item.get("word", "").strip().lower()
                if not raw_w or raw_w == t:
                    continue
                # Parse slash-separated tokens (e.g. 'longing/yearning' -> 'longing', 'yearning')
                for chunk in raw_w.split("/"):
                    chunk = chunk.strip()
                    if chunk and " " not in chunk and chunk not in (t, t_lem):
                        near.add(chunk)
        t_prof = cls.get_word_gram_profile(t, pos=pos)
        for peer in t_prof.get("comparison_peers", set()):
            if peer not in (t, t_lem):
                near.add(peer)

        # Orthogonal Collective Noun Exemption:
        # Distinct subgroups in `_COLLECTIVE_NOUN_ONTOLOGY` (e.g. `staff` vs `crew`/`cast`/`jury`,
        # `flock` vs `herd`/`pack`/`swarm`) are mutually exclusive role/species coordinates, not synonyms,
        # even when Oxford Writer's Thesaurus groups them under a broad collective-noun heading.
        if (not pos or str(pos).lower().startswith("n")) and cls._are_orthogonal_collective_nouns(t_lem, d_lem):
            near.discard(d)
            near.discard(d_lem)

        # Compound Head-Sharing Co-Hyponym Gate (e.g., shareholder vs stakeholder, policymaker vs lawmaker):
        # If both nouns share a >= 5-char compound suffix head (excluding general derivational suffixes)
        # and share definition content words, treat them as near-synonyms so they cannot co-occur under generic modifiers.
        _deriv_noun_suffixes = (
            "itude", "ation", "ition", "ession", "ussion", "asion", "ision", "osion", "usion",
            "tion", "sion", "ment", "ness", "ship", "ance", "ence", "ancy", "ency", "ics", "ist", "ism", "ity", "ure",
        )
        if (not pos or str(pos).lower().startswith("n")) and " " not in t_lem and " " not in d_lem:
            if cls._common_suffix_len(t_lem, d_lem) >= 5 and not t_lem.endswith(_deriv_noun_suffixes):
                if cls._get_def_content_words(t_lem, "n") & cls._get_def_content_words(d_lem, "n"):
                    near.add(d)
                    near.add(d_lem)

        # Collective / Group Noun Double-Key Gate:
        # If both target and distractor denote collections or groups of people/staff
        # and frame is partitive/of-complement ('a ____ of experts'), any other group noun is a double key!
        if (t in cls.COLLECTIVE_GROUP_NOUNS or t_lem in cls.COLLECTIVE_GROUP_NOUNS) and (d in cls.COLLECTIVE_GROUP_NOUNS or d_lem in cls.COLLECTIVE_GROUP_NOUNS):
            if anchor in ("of", "part of", "member of", "on", "in") or anchor_type in ("prep", "partitive"):
                return True, "double_key_slot"

        # Adverb syntactic modification frame discrimination
        if anchor_type in ("modifies_adj", "modifies_verb"):
            d_entry = cls.get_ldoce_entry(d) or cls.get_ldoce_entry(d_lem)
            if d_entry:
                d_senses = d_entry.get("senses", [])
                has_adj_mod = any(
                    any(k in (s.get("grammar") or "").lower() for k in ("adjective", "adverb", "[+adjective"))
                    for s in d_senses
                )
                has_verb_mod = any(
                    any(k in (s.get("grammar") or "").lower() for k in ("verb", "[+verb"))
                    or not s.get("grammar")  # default manner adverb
                    for s in d_senses
                )
                if anchor_type == "modifies_adj" and not has_adj_mod:
                    return False, "frame_only"
                if anchor_type == "modifies_verb" and not has_verb_mod:
                    return False, "frame_only"

        # Verb Syntactic Pattern Exclusion Gate (Single-Fit Distractor Guarantee):
        # If target governs a specific clausal/non-finite complement pattern ('to_inf', 'gerund', 'that_clause', 'sb_to_do')
        # and distractor is completely incompatible with that complement pattern, the distractor is
        # strictly single-fit discriminated (100% immune to double-key collision).
        # Conversely, if distractor shares the identical complement pattern and is in near semantic field,
        # it is a dangerous double-key collision!
        # NOTE: Only trigger when testing complement patterns (not when an explicit bound preposition or noun collocation anchor is being tested).
        if anchor_type in ("to_inf", "gerund", "that_clause", "sb_to_do") or (not anchor and pos in ("v", "verb")):
            t_slots = cls.get_syntactic_pattern_slots(t) | cls.get_syntactic_pattern_slots(t_lem)
            d_slots = cls.get_syntactic_pattern_slots(d) | cls.get_syntactic_pattern_slots(d_lem)
            d_prof = cls.get_word_gram_profile(d, pos=pos)
            d_slots -= d_prof.get("forbidden_frames", set())
            # If anchor_type directly specifies the target complement slot:
            active_slot = anchor_type if anchor_type in ("to_inf", "gerund", "that_clause", "sb_to_do") else None
            if not active_slot:
                # Find intersective slots declared by target
                common_slots = t_slots & {"to_inf", "gerund", "that_clause", "sb_to_do"}
                if len(common_slots) == 1:
                    active_slot = next(iter(common_slots))

            if active_slot and (active_slot in t_slots or anchor_type == active_slot):
                if active_slot not in d_slots or active_slot in d_prof.get("forbidden_frames", set()):
                    # Distractor cannot occupy target's syntactic complement slot -> single-fit discriminated!
                    return False, "none"
                elif d in near or d_lem in near or (cls.semantic_fields(t_lem, pos="v")[0] & cls.semantic_fields(d_lem, pos="v")[0]):
                    # Both share the non-finite/clausal complement slot and are semantic neighbors -> double-key collision!
                    return True, "double_key_pattern"

        t_preps_raw = cls.get_oxford_collocations(t, pos=pos).get("prep", []) + cls.get_oxford_collocations(t_lem, pos=pos).get("prep", [])
        t_preps = [w.strip().lower() for p in t_preps_raw for w in p.split("/") if w.strip()]
        for lp in list(cls.get_ldoce_preps(t, pos=pos)) + list(cls.get_ldoce_preps(t_lem, pos=pos)):
            if lp not in t_preps:
                t_preps.append(lp)
        shared = bool(anchor) and anchor.lower() in t_preps

        if anchor_type == "prep" and anchor and shared:
            # Productive Source/Agent Preposition Guard (`from` / `by`):
            # For nouns, `from [source/organization]` and `by [agent]` are productive theta-role adjuncts
            # that attach to any noun denoting support, funding, approval, assistance, or communication
            # (e.g. `sponsorship` vs `endorsement` / `patronage` / `backing` / `funding` / `subsidy` from corporate partners),
            # even when a dictionary only lists `of` as the object-complement preposition for the distractor.
            if (
                pos
                and str(pos).lower().startswith("n")
                and anchor.lower() in ("from", "by")
                and d not in antonyms
                and d_lem not in antonyms
            ):
                _productive_source_domains = {
                    "support", "help", "assistance", "aid", "money", "fund", "funding", "financial",
                    "approval", "permission", "agreement", "payment", "gift", "donation", "advice",
                    "guidance", "message", "statement", "praise", "criticism", "protection", "benefit",
                }
                t_def_w = cls._get_def_content_words(t_lem, "n") | {t_lem}
                d_def_w = cls._get_def_content_words(d_lem, "n") | {d_lem}
                if (t_def_w & _productive_source_domains) and (d_def_w & _productive_source_domains):
                    return True, "double_key_prep"
            d_preps_raw = cls.get_oxford_collocations(d, pos=pos or "noun").get("prep", []) + cls.get_oxford_collocations(d_lem, pos=pos or "noun").get("prep", [])
            d_preps = [w.strip().lower() for p in d_preps_raw for w in p.split("/") if w.strip()]
            for lp in list(cls.get_ldoce_preps(d, pos=pos)) + list(cls.get_ldoce_preps(d_lem, pos=pos)):
                if lp not in d_preps:
                    d_preps.append(lp)
            if anchor.lower() in d_preps:
                if d in near or d_lem in near:
                    return True, "double_key_prep"
                return False, "frame_only"
            if (d in near or d_lem in near) and d not in antonyms and d_lem not in antonyms:
                # For nouns (`pos` starting with 'n'), a near-synonym in `near` is ONLY allowed under a
                # prepositional anchor when it is a curated contrastive peer in LDOCE's `thesaurus` box
                # (e.g. `loan` vs `overdraft` under `of`) and NOT a reciprocal synonym in `oxford_thesaurus`
                # (e.g. `threat` vs `hazard`, `recession` vs `downturn`, `oath` vs `commitment`, `shortage` vs `absence`).
                if pos and str(pos).lower().startswith("n"):
                    if anchor.lower() in ("into", "from", "by"):
                        return True, "double_key_prep"
                    d_strict_n_syns = cls.get_strict_pedlex_and_oxford_synonyms(d, pos="noun") | cls.get_strict_pedlex_and_oxford_synonyms(d_lem, pos="noun")
                    if t in d_strict_n_syns or t_lem in d_strict_n_syns:
                        return True, "double_key_prep"
                    t_ent = cls.get_ldoce_entry(t) or cls.get_ldoce_entry(t_lem) or {}
                    t_thes_words: Set[str] = set()
                    for th_it in (t_ent.get("thesaurus") or []):
                        if isinstance(th_it, dict):
                            cw = str(th_it.get("word") or "").strip().lower()
                            if cw:
                                t_thes_words.add(cw)
                                t_thes_words.add(cw.split()[0])
                    if d not in t_thes_words and d_lem not in t_thes_words:
                        return True, "double_key_prep"
                # A near-synonym that either lacks competing prepositional valency in PedLex/OCD
                # or, when the target verb itself can take a direct object (`V + [sth] + prep`, e.g.,
                # `convert/adapt [sth] into [sth]`), is also a transitive verb cannot be safely
                # discriminated by preposition alone. Conversely, when the target verb is strictly
                # intransitive (`listen to`), the slot is `____ + prep` with no intervening NP, so a
                # transitive verb lacking that preposition (`hear`, `watch`, `notice`) is barred (`*notice to`).
                if pos and str(pos).lower().startswith("v"):
                    if cls._verb_takes_object(t_lem):
                        if not d_preps or cls._verb_takes_object(d_lem):
                            return True, "double_key_prep"
                    elif not d_preps and not cls._verb_takes_object(d_lem):
                        return True, "double_key_prep"
                elif not pos and not d_preps:
                    return True, "double_key_prep"
            # Candidate does NOT govern the anchor preposition -> single-fit discriminated!
            return False, "none"

        if d in near or d_lem in near:
            if d in antonyms or d_lem in antonyms:
                # Deliberate contrast distractor (eliminated by polarity/logic),
                # whitelisted by design in the distractor generator.
                return False, "contrast"
            # If an explicit collocational anchor is present (e.g. adjective modifier or verb)
            # and candidate d does NOT share that collocation in OCD/PedLex, candidate d is ruled out
            # by the anchor slot itself and is therefore a valid, single-fit distractor.
            # EXCEPTION: If anchor is a generic adjective or verb (e.g. huge, massive, new, good, have),
            # OR if `quote` is provided and the purported `anchor` does not actually appear in `quote`,
            # absence in OCD does NOT discriminate near-synonyms! Near-synonyms remain double keys.
            if anchor and anchor_type in ("adj", "verb", "object", "modified_noun", "verb_subject", "modifies_adj", "modifies_verb"):
                a_clean = anchor.lower()
                is_generic_anchor = (
                    (anchor_type in ("adj", "modified_noun", "object") and a_clean in cls.GENERIC_COLLOCATION_MODIFIERS) or
                    (anchor_type in ("verb", "verb_subject") and a_clean in cls.GENERIC_COLLOCATION_VERBS)
                )
                if is_generic_anchor:
                    return True, "double_key_slot"
                if quote and len(a_clean) >= 3:
                    q_low = quote.lower()
                    a_lem = cls.lemma_of(a_clean) if " " not in a_clean else a_clean
                    if a_clean not in q_low and a_lem not in q_low:
                        return True, "double_key_slot"

                d_entry = cls.get_oxford_collocations(d, pos=pos or "noun")
                shared_colloc = False
                if anchor_type == "adj":
                    shared_colloc = a_clean in [a.split()[0].lower() for a in d_entry.get("adj", [])]
                    if not shared_colloc:
                        d_ldoce = cls.get_ldoce_entry(d)
                        if d_ldoce:
                            for c_item in d_ldoce.get("collocations", {}).get("adjectives", []):
                                if a_clean in c_item.get("collocation", "").lower():
                                    shared_colloc = True
                                    break
                elif anchor_type == "verb":
                    shared_colloc = a_clean in [v.split()[0].lower() for v in d_entry.get("verb_before", [])]
                    if not shared_colloc:
                        d_ldoce = cls.get_ldoce_entry(d)
                        if d_ldoce:
                            for c_item in d_ldoce.get("collocations", {}).get("verbs", []):
                                if a_clean in c_item.get("collocation", "").lower():
                                    shared_colloc = True
                                    break
                elif anchor_type in ("modified_noun", "object"):
                    shared_colloc = a_clean in [n.split()[0].lower() for n in d_entry.get("colloc_nouns", []) + d_entry.get("noun_after", [])]
                    if not shared_colloc:
                        d_ldoce = cls.get_ldoce_entry(d)
                        if d_ldoce:
                            for c_item in d_ldoce.get("collocations", {}).get("nouns", []):
                                if a_clean in c_item.get("collocation", "").lower():
                                    shared_colloc = True
                                    break
                    if not shared_colloc and anchor_type == "object":
                        a_entry = cls.get_oxford_collocations(a_clean, pos="noun")
                        if d in [v.split()[0].lower() for v in a_entry.get("verb_before", [])] or d_lem in [v.split()[0].lower() for v in a_entry.get("verb_before", [])]:
                            shared_colloc = True
                        if not shared_colloc:
                            a_ldoce = cls.get_ldoce_entry(a_clean)
                            if a_ldoce:
                                for v_item in a_ldoce.get("collocations", {}).get("verbs", []):
                                    v_col = v_item.get("collocation", "").lower()
                                    if re.search(rf"\b{re.escape(d)}\b|\b{re.escape(d_lem)}\b", v_col):
                                        shared_colloc = True
                                        break
                    if not shared_colloc and anchor_type == "modified_noun":
                        a_entry = cls.get_oxford_collocations(a_clean, pos="noun")
                        if d in [adj_w.split()[0].lower() for adj_w in a_entry.get("adj", [])] or d_lem in [adj_w.split()[0].lower() for adj_w in a_entry.get("adj", [])]:
                            shared_colloc = True
                        if not shared_colloc:
                            a_ldoce = cls.get_ldoce_entry(a_clean)
                            if a_ldoce:
                                for adj_item in a_ldoce.get("collocations", {}).get("adjectives", []):
                                    adj_col = (adj_item.get("collocation") or adj_item.get("phrase") or "").lower()
                                    if re.search(rf"\b{re.escape(d)}\b|\b{re.escape(d_lem)}\b", adj_col):
                                        shared_colloc = True
                                        break
                        if not shared_colloc:
                            if cls.lookup_ldoce_phrase(f"{d} {a_clean}") or cls.get_ldoce_entry(f"{d} {a_clean}") or (d_lem != d and (cls.lookup_ldoce_phrase(f"{d_lem} {a_clean}") or cls.get_ldoce_entry(f"{d_lem} {a_clean}"))):
                                shared_colloc = True
                        if not shared_colloc and d_ldoce:
                            for s in d_ldoce.get("senses", []):
                                for ex in s.get("examples", []):
                                    if re.search(rf"\b(?:{re.escape(d)}|{re.escape(d_lem)})\s+{re.escape(a_clean)}s?\b", str(ex).lower()):
                                        shared_colloc = True
                                        break
                                if shared_colloc:
                                    break
                elif anchor_type in ("modifies_adj", "modifies_verb"):
                    # Check if distractor adverb explicitly collocates with this anchor in PedLex
                    d_ldoce = cls.get_ldoce_entry(d)
                    if d_ldoce:
                        for s in d_ldoce.get("senses", []):
                            for ex in s.get("examples", []):
                                if re.search(rf"\b{re.escape(a_clean)}\b", ex.lower()):
                                    shared_colloc = True
                                    break
                            if shared_colloc:
                                break
                if not shared_colloc:
                    return False, "none"
            return True, "double_key_slot"

        # Collocational Double-Key Gate: If distractor directly collocates with the anchor in the identical slot
        if anchor:
            a_clean = anchor.lower()
            if (pos == "v" or pos == "verb") and anchor_type in ("object", "collocation"):
                d_entry = cls.get_oxford_collocations(d, pos="verb")
                if a_clean in [n.split()[0].lower() for n in d_entry.get("colloc_nouns", [])]:
                    return True, "double_key_collocation"
                a_entry = cls.get_oxford_collocations(a_clean, pos="noun")
                if d in [v.split()[0].lower() for v in a_entry.get("verb_before", [])]:
                    return True, "double_key_collocation"
                a_ldoce = cls.get_ldoce_entry(a_clean)
                if a_ldoce:
                    for v_item in a_ldoce.get("collocations", {}).get("verbs", []):
                        if d in v_item.get("collocation", "").lower():
                            return True, "double_key_collocation"
            elif (pos == "a" or pos == "adj") and anchor_type == "modified_noun":
                a_entry = cls.get_oxford_collocations(a_clean, pos="noun")
                if d in [adj.split()[0].lower() for adj in a_entry.get("adj", [])] and d not in antonyms:
                    return True, "double_key_collocation"
                a_ldoce = cls.get_ldoce_entry(a_clean)
                if a_ldoce:
                    for a_item in a_ldoce.get("collocations", {}).get("adjectives", []):
                        if d in a_item.get("collocation", "").lower() and d not in antonyms:
                            return True, "double_key_collocation"
            elif (pos == "n" or pos == "noun") and anchor_type in ("verb", "verb_subject"):
                d_entry = cls.get_oxford_collocations(d, pos="noun")
                if a_clean in [v.split()[0].lower() for v in d_entry.get("verb_before", [])] and d not in antonyms:
                    return True, "double_key_collocation"
                d_ldoce = cls.get_ldoce_entry(d)
                if d_ldoce:
                    for v_item in d_ldoce.get("collocations", {}).get("verbs", []):
                        if a_clean in v_item.get("collocation", "").lower() and d not in antonyms:
                            return True, "double_key_collocation"

        return False, "none"

    _PREDICATIVE_ONLY_ADJS = {
        "asleep", "afraid", "alive", "alone", "ashamed", "aware", "awake", "content", "unable", "prone", "glad"
    }

    _ADJ_PREP_VALENCY_MAP = {
        "to": ["vulnerable", "immune", "conducive", "susceptible", "allergic", "attentive", "adjacent", "subordinate", "accessible"],
        "of": ["aware", "capable", "characteristic", "deprived", "indicative", "reminiscent", "composed", "devoid", "conscious"],
        "for": ["eligible", "notorious", "responsible", "suitable", "renowned", "adequate", "prepared", "qualified"],
        "with": ["consistent", "compatible", "acquainted", "associated", "compliant", "congruent", "comparable"],
        "from": ["distinct", "exempt", "absent", "derived", "detached", "remote"],
        "in": ["proficient", "inherent", "rich", "experienced", "versed", "lacking"],
        "on": ["reliant", "dependent", "contingent", "intent", "keen"]
    }

    _ACADEMIC_ADJ_FAMILIES = [
        ["comprehensive", "extensive", "thorough", "limited", "exhaustive", "restricted", "fragmented"],
        ["available", "accessible", "obtainable", "unavailable", "inaccessible", "restricted", "exclusive"],
        ["stable", "volatile", "constant", "variable", "consistent", "fluctuating", "fragile"],
        ["crucial", "fundamental", "marginal", "negligible", "vital", "trivial", "secondary"],
        ["vulnerable", "susceptible", "immune", "resistant", "resilient", "defenseless"],
        ["feasible", "viable", "impractical", "attainable", "unrealistic", "plausible"],
        ["explicit", "ambiguous", "apparent", "obscure", "evident", "vague", "distinct"],
        ["substantial", "considerable", "minimal", "negligible", "significant", "modest"],
        ["coherent", "consistent", "incompatible", "contradictory", "harmonious", "conflicting"],
    ]

    # Academic Register Gate: Filter out slang, informal, archaic, or fantasy/fairy-tale words
    # that WordNet taxonomy occasionally harbors (e.g. 'dwarf', 'beast' for 'individual', or 'swell', 'cracking' for 'positive').
    _REGISTER_BANNED_WORDS = {
        # Archaic / fantasy / fairy-tale nouns
        "dwarf", "beast", "fauna", "flora", "benthos", "heterotroph", "amphidiploid", "diploid",
        "ogre", "giant", "goblin", "witch", "wizard", "fairy", "elf", "gnome", "troll",
        # Colloquial slang / archaic positive adjectives
        "swell", "cracking", "bully", "smashing", "peachy", "dandy", "nifty", "corking", "groovy", "slap-up", "bang-up"
    }

    _ANCHOR_STOPWORDS = {
        "a", "an", "the", "of", "to", "in", "on", "for", "with", "and", "or", "at",
        "by", "from", "into", "onto", "out", "off", "up", "down", "as", "be", "is",
        "are", "was", "were", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "shall", "should", "can", "could", "may", "might",
        "must", "that", "this", "these", "those", "it", "its", "he", "she", "we",
        "they", "you", "i", "me", "my", "your", "his", "her", "their", "our",
        "each", "every", "some", "any", "all", "both", "few", "sth", "sb",
        # Degree adverbs, quantifiers and determiners can never serve as
        # collocational anchors (they carry no lexical collocation signal;
        # e.g. 'more' in 'more hours' must not be harvested for 'autonomy').
        "more", "most", "less", "least", "many", "much", "very", "quite",
        "rather", "such", "own", "other", "another", "same", "different",
        "particular", "certain", "various", "several", "enough", "only",
        "just", "even", "also", "well", "still", "yet", "already",
        # Empty / generic head nouns carry virtually zero lexical collocational value
        # (e.g. 'thing', 'stuff', 'item' in 'positive thing' or 'important matter').
        # Banning them prevents trivial or unacademic frames like 'positive thing'.
        "thing", "things", "stuff", "person", "people", "way", "ways", "item", "items",
        # Indefinite pronouns and quantifier nouns are placeholders, not collocations:
        # 'offer someone something' and 'lots of goodies' say nothing about how the
        # target actually collocates, so they must never become anchors.
        "something", "anything", "everything", "nothing", "someone", "somebody",
        "anyone", "anybody", "everyone", "everybody", "nobody", "one", "ones",
        "lot", "lots", "bit", "bits", "kind", "kinds", "sort", "sorts",
        "piece", "pieces", "pair", "pairs", "number", "amount", "rest"
    }

    # Fixed prepositional adjuncts / set phrases that are interjections or
    # discourse markers, NEVER valency complements of a governing headword.
    # When a (prep, pobj) dependency pair hits this blacklist it is skipped,
    # so 'correlate for example' can no longer be mistaken for a frame.
    _ADJUNCT_PREP_PAIRS = {
        ("for", "example"), ("for", "instance"), ("in", "fact"), ("in", "general"),
        ("at", "first"), ("on", "average"), ("in", "particular"), ("in", "summary"),
        ("in", "total"), ("in", "brief"), ("in", "short"), ("in", "part")
    }

    # When a verb has NO Oxford Collocations entry (or an empty 'prep' valency
    # list), only these typical academic complement prepositions may be
    # trusted from syntactic guessing; anything else degrades to anchor=None
    # (neutral template) instead of fabricating a frame (e.g. 'degrade in a
    # society' from a fronted adjunct).
    _TYPICAL_COMPLEMENT_PREPS = {"into", "to", "from"}

    _PHRASE_INDEX_CACHE: Optional[Dict[str, str]] = None
    _PHRASE_SYNONYMS_CACHE: Optional[Dict[str, Set[str]]] = None

    _TABOO_PHRASE_TOKENS: Set[str] = {
        "arse", "ass", "asshole", "bastard", "bitch", "bloody", "bollocks",
        "bugger", "bum", "crap", "cunt", "damn", "dick", "fag", "fart", "fuck", "fucking",
        "motherfucker", "nigger", "piss", "prick", "pussy", "shit", "shitty",
        "slut", "tit", "twat", "wanker", "whore", "doss"
    }

    _TABOO_MULTIWORD_PHRASES: Set[str] = {
        "go down on", "get it on", "get laid", "knock up", "jerk off", "jack off",
        "toss off", "make out with", "sleep with", "sleep around", "fool around with",
        "come on to", "have it off", "do it with", "turn on", "screw around",
    }

    @classmethod
    def is_inappropriate_phrase(cls, phrase: str) -> bool:
        """
        Determines whether a candidate phrase contains taboo, vulgar, offensive,
        or coarse slang tokens unsuited for pedagogical curriculum assessments.
        """
        tokens = [t.lower() for t in re.findall(r'[a-zA-Z]+', phrase)]
        if not tokens:
            return False
        if any(t in cls._TABOO_PHRASE_TOKENS for t in tokens):
            return True
        norm_p = " ".join(tokens)
        if norm_p in cls._TABOO_MULTIWORD_PHRASES:
            return True
        return False

    @classmethod
    def _get_phrase_synonyms(cls) -> Dict[str, Set[str]]:
        """
        Loads and memoizes synonym groups generated by slash-alternations in PedLex raw phrases
        (e.g., 'be/keep/stay etc in touch (with something)' -> {'be in touch with', 'keep in touch with', 'stay in touch with'}).
        """
        if cls._PHRASE_SYNONYMS_CACHE is not None:
            return cls._PHRASE_SYNONYMS_CACHE
        if cls._ensure_in_memory_phrase_index() and cls._PHRASE_SYNONYMS_CACHE is not None:
            return cls._PHRASE_SYNONYMS_CACHE
        cls._PHRASE_SYNONYMS_CACHE = {}
        return cls._PHRASE_SYNONYMS_CACHE

    @classmethod
    def _get_phrase_index(cls) -> Dict[str, str]:
        """Loads and memoizes {phrase_key: tier} from ldoce_phrase_index in the PedLex database."""
        if cls._PHRASE_INDEX_CACHE is not None:
            return cls._PHRASE_INDEX_CACHE
        if cls._ensure_in_memory_phrase_index() and cls._PHRASE_INDEX_CACHE is not None:
            return cls._PHRASE_INDEX_CACHE
        cls._PHRASE_INDEX_CACHE = {}
        return cls._PHRASE_INDEX_CACHE

    @classmethod
    def generate_phrase_distractors(
        cls,
        phrase: str,
        pos: str = "phrasal_verb",
        count: int = 3,
        exclude_words: Optional[Set[str]] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
    ) -> List[str]:
        """
        Synthesizes high-discrimination, authentic distractors for multi-word phrasal verbs, idioms, and collocations
        strictly grounded in PedLex 6th Edition's ldoce_phrase_index and lexical evidence.
        
        Symmetric structural families:
          1. Head/Tail Sharing (e.g. 'peace of mind' -> 'state of mind', 'frame of mind', 'turn of mind';
                                    'keep in touch with' -> 'stay in touch with', 'get in touch with')
          2. Light-Verb & Lexical Frames (e.g. 'have a try' -> 'have a go', 'have a heart', 'have a care')
          3. Binomial / Parallel Match (e.g. 'pros and cons' -> 'bits and pieces', 'back and forth', 'give and take')
          4. 2-Word Phrasal Verbs (Particle contrast & Core verb contrast via phrasal_verb index)
        """
        phrase_clean = phrase.strip().lower()
        words = phrase_clean.split()
        exclude = set(exclude_words or [])
        distractors: List[str] = []
        n_words = len(words)

        phrase_index = cls._get_phrase_index()
        phrase_synonyms = cls._get_phrase_synonyms()
        synonym_set = phrase_synonyms.get(phrase_clean, set())

        _NON_VERB_PHRASE_HEADS = {
            "a", "an", "the", "no", "not", "all", "any", "some", "every", "each", "both",
            "either", "neither", "much", "many", "more", "most", "less", "least", "few",
            "little", "other", "another", "such", "what", "which", "whose", "this", "that",
            "these", "those", "my", "your", "his", "her", "its", "our", "their", "one",
            "in", "on", "at", "by", "for", "with", "from", "to", "of", "out", "up", "down",
            "over", "under", "into", "onto", "upon", "about", "above", "below", "between",
            "among", "through", "during", "before", "after", "since", "until", "without",
            "within", "against", "along", "around", "behind", "beside", "beyond", "near",
            "off", "oward", "towards", "as", "if", "so", "than", "though", "while", "when"
        }
        head_w = words[0] if words else ""
        _PARTICLES_AND_PREPS = {
            "in", "off", "up", "down", "away", "back", "over", "into", "through",
            "on", "to", "for", "out", "with", "from", "around", "about", "of",
            "by", "at", "under", "between", "against", "after", "before",
            "without", "within", "along", "across", "behind", "beyond", "upon",
            "onto", "towards", "toward", "aside", "apart", "ahead", "aback", "forth"
        }
        _VERB_NP_HEADS = {
            "take", "make", "have", "do", "give", "get", "keep", "lose", "catch",
            "pay", "save", "bear", "hold", "set", "play", "run", "put", "bring",
            "turn", "go", "come", "fall", "break", "cut", "draw", "drop", "find",
            "form", "gain", "lay", "lead", "leave", "meet", "pass", "pull", "push",
            "raise", "reach", "see", "send", "shake", "show", "spend", "spread",
            "stand", "start", "stay", "strike", "tell", "throw", "touch", "use",
            "waste", "win", "work", "be", "become", "feel", "seem", "look", "sound",
            "remain", "grow", "prove", "smell", "taste", "change", "join", "cross"
        }
        t_entry_top = cls.get_ldoce_entry(phrase_clean) if n_words == 2 else None
        t_pos_top = str((t_entry_top or {}).get("pos", "")).strip().lower()
        pos_norm = str(pos or "").strip().lower()
        _ADV_OR_ADJ_COMPLEMENTS = {
            "better", "worse", "best", "worst", "well", "badly", "hard", "fast",
            "far", "more", "less", "long", "short", "high", "low", "early", "late",
            "alone", "wrong", "right", "free", "clear", "open", "close", "sure", "certain",
        }
        is_compound_noun = (
            n_words == 2
            and words[0] not in _PARTICLES_AND_PREPS
            and words[0] not in _NON_VERB_PHRASE_HEADS
            and words[1] not in _PARTICLES_AND_PREPS
            and words[1] not in _ADV_OR_ADJ_COMPLEMENTS
            and str((cls.get_ldoce_entry(words[1]) or {}).get("pos") or "").strip().lower() not in ("adverb", "adjective")
            and pos_norm not in ("verb", "v", "phrasal verb", "phrasal_verb", "adj", "adjective", "adv", "adverb")
            and (
                t_pos_top == "noun"
                or pos_norm in ("noun", "n")
                or (
                    words[0] not in _VERB_NP_HEADS
                    and bool(cls._wordnet_pos(words[1]) & {"n"})
                    and (
                        not bool(cls._wordnet_pos(words[0]) & {"v"})
                        or bool(cls._wordnet_pos(words[0]) & {"n", "a", "s"})
                    )
                )
            )
        )
        target_head_is_verb = (
            bool(head_w)
            and not is_compound_noun
            and head_w not in _NON_VERB_PHRASE_HEADS
            and bool(cls._wordnet_pos(head_w) & {"v"})
        )
        target_head_is_prep = head_w in {
            "in", "on", "at", "by", "for", "with", "from", "to", "of", "out", "up", "down",
            "over", "under", "into", "onto", "upon", "about", "above", "below", "between",
            "among", "through", "during", "before", "after", "since", "until", "without",
            "within", "against", "along", "around", "behind", "beside", "beyond"
        }

        _PARTICLE_OR_DET_MIDDLES = {
            "a", "an", "the", "my", "your", "his", "her", "its", "our", "their", "some", "any", "no",
            "it", "them", "him", "me", "us", "you", "this", "that", "these", "those", "all", "what",
            "up", "down", "in", "out", "on", "off", "away", "back", "over", "under", "around", "along",
            "through", "into", "onto", "to", "for", "with", "from", "by", "at", "of", "about", "as",
            "got", "been", "done", "finished", "heard", "taken", "caught", "lost", "stuck", "lying", "too"
        }
        target_is_v_noun_prep = (
            n_words == 3
            and target_head_is_verb
            and words[1] not in _PARTICLE_OR_DET_MIDDLES
            and bool(cls._wordnet_pos(words[1]) & {"n"})
        )
        target_head_is_det = head_w in ("a", "an", "the")

        def _norm_poss(s: str) -> str:
            return re.sub(
                r"\b(?:one's|your|my|his|her|its|our|their|sb's|somebody's|someone's)\b",
                "<POSS>",
                s.lower(),
            )

        target_norm_poss = _norm_poss(phrase_clean)

        _DEFINING_GENUS_VERBS = {
            "be", "have", "do", "make", "get", "give", "take", "use", "say", "think",
            "feel", "know", "show", "cause", "keep", "put", "go", "come", "move",
            "turn", "start", "stop", "try", "want", "need", "help", "allow", "let",
            "mean", "seem", "become", "act", "happen"
        }

        def _get_phrase_def_text(p_str: str) -> str:
            chunks: List[str] = []
            e_direct = cls.get_ldoce_entry(p_str)
            if e_direct:
                for s in e_direct.get("senses", []):
                    chunks.append(str(s.get("definition") or ""))
            p_tokens = p_str.split()
            if p_tokens:
                e_head = cls.get_ldoce_entry(p_tokens[0])
                if e_head:
                    for s in e_head.get("senses", []):
                        phw = str(s.get("phrasal_headword") or "").lower()
                        pats = " ".join(str(pt) for pt in (s.get("patterns") or [])).lower()
                        if p_str in phw or p_str in pats:
                            chunks.append(str(s.get("definition") or ""))
                    for pv in e_head.get("phrasal_verbs", []) or []:
                        if isinstance(pv, dict):
                            pv_name = str(pv.get("phrase") or pv.get("headword") or "").lower()
                            if p_str in pv_name:
                                for pvs in pv.get("senses", []) or []:
                                    if isinstance(pvs, dict):
                                        chunks.append(str(pvs.get("definition") or ""))
                    for ph in e_head.get("phrases", []) or []:
                        if isinstance(ph, dict):
                            ph_name = str(ph.get("phrase") or ph.get("headword") or "").lower()
                            if p_str in ph_name:
                                for phs in ph.get("senses", []) or []:
                                    if isinstance(phs, dict):
                                        chunks.append(str(phs.get("definition") or ""))
            return " ; ".join(chunks).lower()

        target_phrase_def_text = _get_phrase_def_text(phrase_clean) if n_words == 2 else ""

        def ships(cand: str) -> bool:
            if (cand == phrase_clean or _norm_poss(cand) == target_norm_poss
                    or cand in exclude or cand in synonym_set
                    or "." in cand or "[" in cand or "]" in cand or "(" in cand or ")" in cand or "/" in cand
                    or cls.is_inappropriate_phrase(cand)
                    or not (cand in phrase_index or cls.is_attested_phrase(cand))):
                return False
            c_words = cand.split()
            if not c_words:
                return False
            c_head = c_words[0]
            if is_compound_noun:
                if len(c_words) == 2 and (c_words[1] in _PARTICLES_AND_PREPS or c_head in _VERB_NP_HEADS):
                    return False
                if not (cls._wordnet_pos(c_words[-1]) & {"n"}):
                    return False
            elif target_head_is_verb:
                # Active lexical/light verb phrases (e.g. 'make the most of', 'take pleasure in', 'play a part', 'fare better')
                # must only pair with active verb-headed phrases, never determiner/noun/prep fragments
                # ('no to speak of', 'a bad judge of'), nominal compounds ('walk-on part', 'fare dodger'),
                # or mismatched inflectional forms ('getting better' vs base 'fare better').
                if c_head in _NON_VERB_PHRASE_HEADS or not (cls._wordnet_pos(c_head) & {"v"}):
                    return False
                if (c_head.endswith("ing") and not head_w.endswith("ing")) or (head_w.endswith("ing") and not c_head.endswith("ing")):
                    return False
                c_ent_full = cls.get_ldoce_entry(cand) or cls.get_ldoce_entry(cand.replace(" ", "-"))
                if c_ent_full and str(c_ent_full.get("pos") or "").strip().lower() == "noun":
                    return False
                if len(c_words) >= 2:
                    c_ent_pre = cls.get_ldoce_entry(f"{c_words[0]}-{c_words[1]}")
                    if c_ent_pre and str(c_ent_pre.get("pos") or "").strip().lower() in ("adjective", "noun"):
                        return False
                ch_entry = cls.get_ldoce_entry(c_head)
                ch_primary_pos = str((ch_entry or {}).get("pos") or "").lower()
                if any(bad_p in ch_primary_pos for bad_p in ("interjection", "exclamation", "adjective", "adverb", "preposition", "determiner")):
                    return False
                if n_words == 2 and len(c_words) == 2 and c_words[1] in _PARTICLES_AND_PREPS:
                    _ditrans_only_heads = {
                        "attribute", "equip", "provide", "confine", "devote", "dedicate",
                        "expose", "subject", "owe", "accuse", "deprive", "remind",
                        "convince", "inform", "warn", "assure", "rob", "rid", "cure",
                    }
                    if c_head in _ditrans_only_heads and head_w not in _ditrans_only_heads:
                        return False
                if c_head == "be" and head_w != "be":
                    return False
                if target_is_v_noun_prep and len(c_words) == 3:
                    # Preserve Verb + Bare-Noun + Prep structural symmetry: reject phrasal verb + prep
                    # ('bring out in', 'come on in', 'take up with') or Verb + Det + Noun ('take a chance')
                    if c_words[1] in _PARTICLE_OR_DET_MIDDLES or not (cls._wordnet_pos(c_words[1]) & {"n"}):
                        return False
                if n_words == 3 and words[1] in ("a", "an", "the"):
                    # Preserve Verb + Det + Noun structural symmetry (e.g. 'play a part'):
                    # reject phrasal-verb nominalizations ('walk on part') and synonymous light-verb variants ('have a part').
                    if len(c_words) != 3 or c_words[1] not in ("a", "an", "the", "one's", "your", "his", "her", "their", "some", "no"):
                        return False
                    if c_words[1:] == words[1:] and head_w in ("play", "have", "take", "make", "do", "give") and c_head in ("play", "have", "take", "make", "do", "give"):
                        return False
            elif target_head_is_det:
                if c_head not in ("a", "an", "the"):
                    return False
            elif target_head_is_prep:
                if c_head not in _PARTICLES_AND_PREPS:
                    return False
                if n_words == 3 and words[1] in ("a", "an", "the"):
                    if len(c_words) != 3 or c_words[1] not in ("a", "an", "the"):
                        return False
                if n_words == 4 and words[1] in ("a", "an", "the"):
                    if len(c_words) != 4 or c_words[1] not in ("a", "an", "the"):
                        return False
                    if words[-1] not in _PARTICLES_AND_PREPS and c_words[-1] == words[-1]:
                        return False

            # Same-head dual-preposition double-key guard (e.g. 'equate to' vs 'equate with', 'compare to' vs 'compare with')
            if n_words == 2 and len(c_words) == 2 and c_head == head_w and c_words[1] != words[1]:
                if cls.double_key_collision(phrase_clean, cand, pos="phrase", quote=quote)[0]:
                    return False

            # Same-preposition / same-tail 2-word phrase double-key guard (e.g. 'care about' vs 'worry about')
            if n_words == 2 and len(c_words) == 2 and c_words[1] == words[1] and c_head != head_w:
                head_pos = "verb" if target_head_is_verb else ("adj" if words[0] in cls._wordnet_pos(words[0]) and "a" in cls._wordnet_pos(words[0]) else None)
                if cls.double_key_collision(head_w, c_head, anchor=words[1], anchor_type="prep", pos=head_pos)[0]:
                    return False
                cand_def_text = _get_phrase_def_text(cand)
                c_forms = cls.inflected_forms(c_head) | {c_head}
                t_forms = cls.inflected_forms(head_w) | {head_w}
                if target_phrase_def_text:
                    for cf in c_forms:
                        if re.search(rf"\b{re.escape(cf)}\s+{re.escape(words[1])}\b", target_phrase_def_text):
                            return False
                        if c_head not in _DEFINING_GENUS_VERBS and re.search(rf"\b{re.escape(cf)}\b", target_phrase_def_text):
                            return False
                if cand_def_text:
                    for tf in t_forms:
                        if re.search(rf"\b{re.escape(tf)}\s+{re.escape(words[1])}\b", cand_def_text):
                            return False
                        if head_w not in _DEFINING_GENUS_VERBS and re.search(rf"\b{re.escape(tf)}\b", cand_def_text):
                            return False
            return True

        tier_rank = {tier: index for index, tier in enumerate(cls._PHRASE_TIERS)}

        def strength(cand: str) -> int:
            """PedLex's own phrasal-verb block beats a PHRASES item, which beats a pattern."""
            idx_tier = phrase_index.get(cand)
            if idx_tier == "phrasal_verb":
                return 0
            hits = cls.ldoce_phrase_hits(cand)
            if hits:
                return min((tier_rank.get(tier, len(tier_rank)) for tier, _text in hits),
                           default=len(tier_rank))
            tier = "phrase" if idx_tier == "sense_unit" else (idx_tier or "example")
            return tier_rank.get(tier, len(tier_rank))

        # -------------------------------------------------------------
        # 0a. Unified Construction Families Phrase Contrast (`pedlex_cxn_index`)
        # (Article Minimal-Pair PPs, Fronted Negative Inversion Prepositional Phrases, Correlative Triggers)
        # -------------------------------------------------------------
        if phrase_clean in cls._CORREL_FIRST_ARMS:
            correl_cands, _ = cls.get_construction_contrast_distractors(
                phrase_clean, pos="phrase", target_count=count, exclude_words=exclude
            )
            if len(correl_cands) >= count:
                return correl_cands[:count]
            distractors.extend(correl_cands)
        else:
            cxn_p_cands, cxn_p_spec = cls.get_construction_contrast_distractors(
                phrase_clean, pos="phrase", target_count=count * 2, exclude_words=exclude
            )
            if cxn_p_spec and cxn_p_spec.get("subcat_group") in (
                "PP_BARE_ARTICLE", "PP_DEF_ARTICLE", "PHR_FRONTED_INV", "CORREL_PAIR"
            ):
                subcat_g = cxn_p_spec.get("subcat_group")
                filtered_cxn_p: List[str] = []
                # For PP_BARE_ARTICLE <-> PP_DEF_ARTICLE, place the exact article minimal-pair counterpart first
                if subcat_g == "PP_BARE_ARTICLE" and n_words >= 3:
                    counterpart = f"{words[0]} the {' '.join(words[1:])}"
                    if counterpart not in exclude and counterpart != phrase_clean:
                        filtered_cxn_p.append(counterpart)
                elif subcat_g == "PP_DEF_ARTICLE" and n_words >= 4 and words[1] == "the":
                    counterpart = f"{words[0]} {' '.join(words[2:])}"
                    if counterpart not in exclude and counterpart != phrase_clean:
                        filtered_cxn_p.append(counterpart)
                for cp in cxn_p_cands:
                    if cp in filtered_cxn_p or cp in exclude or cp == phrase_clean:
                        continue
                    if subcat_g in ("PP_BARE_ARTICLE", "PP_DEF_ARTICLE") and words[0] in ("in", "out", "on", "at", "by", "under", "for"):
                        if cp.split()[0] not in ("in", "out", "on", "at", "by", "under", "for"):
                            continue
                    filtered_cxn_p.append(cp)
                if len(filtered_cxn_p) >= count:
                    return filtered_cxn_p[:count]
                distractors.extend(filtered_cxn_p)

        # -------------------------------------------------------------
        # 0. Declarative Collocation & Prepositional Frames
        # (Verb + Prep, Adjective + Prep, Noun + Prep, 3/4-Part Prepositional Frames)
        # -------------------------------------------------------------
        try:
            from librarian.expression_patterns import (
                COLLOCATION_VERB_PREPS,
                ADJ_DEPENDENT_PREPS,
                NOUN_DEPENDENT_PREPS,
                FRAME_PREPOSITION_NOUNS,
            )
        except ImportError:
            COLLOCATION_VERB_PREPS = {}
            ADJ_DEPENDENT_PREPS = {}
            NOUN_DEPENDENT_PREPS = {}
            FRAME_PREPOSITION_NOUNS = set()

        if n_words == 2 and words[0] in COLLOCATION_VERB_PREPS and words[1] == COLLOCATION_VERB_PREPS[words[0]]:
            verb, prep = words[0], words[1]
            verb_cands: List[str] = []
            # Priority 1: Same preposition contrast (e.g. 'worry about' -> 'care about', 'complain about'; 'derive from' -> 'benefit from', 'suffer from', 'differ from')
            for v, p in COLLOCATION_VERB_PREPS.items():
                if p == prep and v != verb:
                    c = f"{v} {prep}"
                    if ships(c) and c not in verb_cands:
                        verb_cands.append(c)
            if len(verb_cands) >= count:
                return verb_cands[:count]
            distractors.extend(verb_cands)

        elif n_words == 2 and words[0] in ADJ_DEPENDENT_PREPS:
            adj, prep = words[0], words[1]
            adj_cands: List[str] = []
            # Priority 1: Same preposition contrast (e.g. 'aware of' -> 'conscious of', 'capable of', 'tired of')
            for a, p in ADJ_DEPENDENT_PREPS.items():
                if p == prep and a != adj:
                    c = f"{a} {prep}"
                    if ships(c) and c not in adj_cands:
                        adj_cands.append(c)
            # Priority 2: Cross-preposition adjective collocations
            for a, p in ADJ_DEPENDENT_PREPS.items():
                if a != adj and p != prep:
                    c = f"{a} {p}"
                    if ships(c) and c not in adj_cands:
                        adj_cands.append(c)
            if len(adj_cands) >= count:
                return adj_cands[:count]
            distractors.extend(adj_cands)

        elif n_words == 2 and words[0] in NOUN_DEPENDENT_PREPS:
            noun, prep = words[0], words[1]
            noun_cands: List[str] = []
            # Priority 1: Same preposition noun collocations (e.g. 'access to' -> 'approach to', 'attitude to', 'solution to')
            for n, p in NOUN_DEPENDENT_PREPS.items():
                if p == prep and n != noun:
                    c = f"{n} {prep}"
                    if ships(c) and c not in noun_cands:
                        noun_cands.append(c)
            # Priority 2: Cross-preposition noun collocations
            for n, p in NOUN_DEPENDENT_PREPS.items():
                if n != noun and p != prep:
                    c = f"{n} {p}"
                    if ships(c) and c not in noun_cands:
                        noun_cands.append(c)
            if len(noun_cands) >= count:
                return noun_cands[:count]
            distractors.extend(noun_cands)

        elif (n_words in (3, 4) and any(w in FRAME_PREPOSITION_NOUNS for w in words)
              and words[0] in ("in", "at", "on", "by", "for", "with", "under", "out")):
            frame_cands: List[str] = []
            canonical_frames = [
                "in terms of", "in front of", "in light of", "in view of", "in spite of",
                "in addition to", "in response to", "on behalf of", "in charge of",
                "in need of", "in search of", "in favor of", "in order to",
                # 4-word prepositional frames (e.g. 'at the expense of')
                "at the expense of", "for the sake of", "in the event of", "at the mercy of",
                "at the hands of", "for the good of", "by means of"
            ]
            # Prioritize frames sharing the initial preposition or final preposition
            for f in canonical_frames:
                if f != phrase_clean and len(f.split()) == n_words:
                    fw = f.split()
                    if fw[0] == words[0] or fw[-1] == words[-1]:
                        if ships(f) and f not in frame_cands:
                            frame_cands.append(f)
            for f in canonical_frames:
                if f != phrase_clean and len(f.split()) == n_words and ships(f) and f not in frame_cands:
                    frame_cands.append(f)
            if len(frame_cands) >= count:
                return frame_cands[:count]
            distractors.extend(frame_cands)

        # -------------------------------------------------------------
        # Quantifier / Partitive frame (e.g. 'a piece of', 'a bit of', 'a slice of')
        # -------------------------------------------------------------
        elif n_words == 3 and words[0] in ("a", "an") and words[2] == "of":
            head_n = words[1]
            q_siblings = [
                "piece", "bit", "slice", "sheet", "block", "drop",
                "pair", "pack", "stroke", "portion", "set", "share"
            ]
            quant_cands: List[str] = []
            for qs in q_siblings:
                if qs != head_n:
                    art = "an" if qs[0] in "aeiou" else "a"
                    cand_frame = f"{art} {qs} of"
                    if ships(cand_frame) and cand_frame not in quant_cands:
                        quant_cands.append(cand_frame)
            if len(quant_cands) >= count:
                return quant_cands[:count]
            distractors.extend(quant_cands)

        # -------------------------------------------------------------
        # 0b. Two-word Compound Nouns (e.g. 'jet lag', 'culture shock', 'rush hour', 'passenger plane')
        # -------------------------------------------------------------
        if n_words == 2 and not distractors and is_compound_noun:
            t_entry = t_entry_top
            t_gram = " ".join(str(s.get("grammar", "")) for s in (t_entry or {}).get("senses", [])).lower()
            t_is_uncountable = "uncountable" in t_gram and "countable" not in t_gram.replace("uncountable", "")
            comp_cands: List[str] = []

            # Priority 1: Slash-expanded and head/modifier-sharing 2-word nominal collocations in phrase_index
            # (e.g., 'passenger train/plane/ship' -> 'passenger train', 'passenger ship'; '* plane' -> 'fighter plane')
            _PARTICIPLE_SUFFIX_STOP = {"formed", "laid", "made", "improved", "born", "built", "grown", "found", "known"}
            same_mod_cands: List[str] = []
            same_head_cands: List[str] = []
            for k in phrase_index:
                kw = k.split()
                if len(kw) != 2:
                    continue
                # Expand slash-grouped 2-word collocations (e.g. 'passenger train/plane/ship')
                if "/" in kw[1] and "/" not in kw[0] and kw[0] == words[0]:
                    w1_syns = cls.get_synonyms(words[1])
                    for alt_n in kw[1].split("/"):
                        alt_n = alt_n.strip()
                        if alt_n and alt_n != words[1] and alt_n.isalpha() and alt_n not in w1_syns:
                            cand_alt = f"{kw[0]} {alt_n}"
                            if cand_alt not in exclude and cand_alt not in synonym_set and not cls.is_inappropriate_phrase(cand_alt):
                                if cand_alt not in same_mod_cands:
                                    same_mod_cands.append(cand_alt)
                    continue
                if "/" in k:
                    continue
                if kw[1] in _PARTICLES_AND_PREPS or kw[0] in _PARTICLES_AND_PREPS or kw[0] in _NON_VERB_PHRASE_HEADS or kw[0] in _VERB_NP_HEADS:
                    continue
                if kw[0] == words[0] and kw[1] != words[1]:
                    if kw[1] in _PARTICIPLE_SUFFIX_STOP or kw[1].endswith("ed"):
                        continue
                    w1_pos = cls._wordnet_pos(kw[1])
                    if ("n" in w1_pos) and not cls.double_key_collision(words[1], kw[1], pos="noun")[0]:
                        if k not in exclude and k not in synonym_set and not cls.is_inappropriate_phrase(k) and k not in same_mod_cands:
                            same_mod_cands.append(k)
                elif kw[1] == words[1] and kw[0] != words[0]:
                    w0_pos = cls._wordnet_pos(kw[0])
                    if (w0_pos & {"n", "a", "s"}) and not cls.double_key_collision(words[0], kw[0])[0]:
                        if k not in exclude and k not in synonym_set and not cls.is_inappropriate_phrase(k) and k not in same_head_cands:
                            same_head_cands.append(k)

            same_mod_cands.sort(
                key=lambda c: (
                    0 if c not in phrase_index else 1,
                    cls.CEFR_ORDER.get(cls.get_word_cefr(c.split()[1]) or "B2", 4),
                    strength(c),
                    c,
                )
            )
            same_head_cands.sort(key=lambda c: (strength(c), c))
            for sc in same_mod_cands + same_head_cands:
                if sc not in comp_cands:
                    comp_cands.append(sc)
                if len(comp_cands) >= count:
                    return comp_cands[:count]

            # Priority 2: Canonical countable/uncountable 2-word compound nouns
            if t_is_uncountable:
                canonical_compounds = [
                    "culture shock", "stage fright", "peer pressure", "food poisoning",
                    "blood pressure", "common sense", "junk food", "global warming",
                    "climate change", "heart failure", "sleep deprivation", "air conditioning"
                ]
            else:
                canonical_compounds = [
                    "rush hour", "heart attack", "side effect", "turning point",
                    "traffic jam", "brain drain", "wake-up call", "blind spot",
                    "comfort zone", "safety net", "generation gap", "role model"
                ]
            for cc in canonical_compounds:
                if cc != phrase_clean and cc not in exclude and cc not in synonym_set and not cls.is_inappropriate_phrase(cc):
                    if cc in phrase_index or cls.get_ldoce_entry(cc) is not None or cls.is_attested_phrase(cc):
                        if cc not in comp_cands:
                            comp_cands.append(cc)
            if len(comp_cands) >= count:
                return comp_cands[:count]
            distractors.extend(comp_cands)

        # -------------------------------------------------------------
        # 0c. Two-word Prepositional State Phrases (e.g. 'at ease', 'at risk', 'on guard', 'in danger')
        # -------------------------------------------------------------
        if n_words == 2 and not distractors and target_head_is_prep and words[1] not in _PARTICLES_AND_PREPS:
            canonical_prep_states = [
                "on guard", "at risk", "in danger", "on duty", "under pressure",
                "by chance", "on purpose", "in vain", "at fault", "in charge",
                "on board", "at peace"
            ]
            prep_state_cands: List[str] = []
            for ps in canonical_prep_states:
                ps_w = ps.split()
                if ps != phrase_clean and ps not in exclude and ps not in synonym_set and len(ps_w) == 2:
                    if not cls.double_key_collision(words[1], ps_w[1], pos="noun")[0]:
                        if ps in phrase_index or cls.is_attested_phrase(ps):
                            prep_state_cands.append(ps)
            if len(prep_state_cands) >= count:
                return prep_state_cands[:count]
            distractors.extend(prep_state_cands)

        # -------------------------------------------------------------
        # 1. Two-word phrase (e.g. 'send out', 'tend to', 'cut out', 'work for')
        # -------------------------------------------------------------
        if n_words == 2 and not distractors:
            v, p = words[0], words[1]
            diff_part: List[str] = []
            _adverbial_particles = ["in", "off", "up", "down", "away", "back", "over", "through", "on", "out", "around", "along", "ahead", "apart", "aside", "together"]
            _relational_preps = {"to", "with", "into", "from", "of", "upon", "for", "at", "by", "about", "as", "against", "between", "among", "towards", "toward"}
            particles = ["in", "off", "up", "down", "away", "back", "over", "into", "through", "on", "to", "for", "out", "with", "from", "around", "about"]
            p_is_particle = p in particles
            p_is_adverbial_particle = p in _adverbial_particles and p not in _relational_preps
            v_is_verb = bool(cls._wordnet_pos(v) & {"v"})

            if p_is_particle:
                # Only vary the particle on the same verb when `p` and `part` are genuine adverbial particles
                # ('break up' -> 'break down'), NEVER when `p` or `part` is a relational preposition
                # ('equate to' -x-> 'equate with', 'compare to' -x-> 'compare with', 'integrate into' -x-> 'integrate with').
                if p_is_adverbial_particle:
                    for part in _adverbial_particles:
                        if part != p:
                            cand = f"{v} {part}"
                            if ships(cand) and cand not in diff_part:
                                diff_part.append(cand)

                    # PedLex's own phrasal-verb list for this verb is the strongest pool there is
                    for cand in cls.get_ldoce_phrasal_verbs(v):
                        parts = cand.split()
                        if len(parts) == 2 and parts[1] != p and parts[1] in _adverbial_particles and ships(cand) and cand not in diff_part:
                            diff_part.insert(0, cand)

                same_part: List[str] = []
                common_verbs = [
                    "point", "carry", "turn", "figure", "bring", "stand", "make", "find",
                    "set", "take", "come", "go", "get", "give", "hold", "look", "run",
                    "keep", "leave", "work", "fall", "call", "pass", "lead", "move", "break"
                ]
                for cv in common_verbs:
                    if cv != v:
                        cand = f"{cv} {p}"
                        if ships(cand) and cand not in diff_part and cand not in same_part:
                            same_part.append(cand)

                # Query phrase_index for 2-word phrasal verbs ending with particle `p`
                p_cands = [
                    k for k in phrase_index
                    if " " in k and len(k.split()) == 2 and k.split()[1] == p and k.split()[0] != v
                ]
                p_cands.sort(key=lambda k: (0 if phrase_index.get(k) == "phrasal_verb" else 1, k))
                tier0_count = sum(1 for c in same_part if strength(c) == 0)
                for k in p_cands:
                    if tier0_count >= count:
                        break
                    cand_head = k.split()[0]
                    # Enforce head syntactic concord: if target head is a verb, cand head must be a primary verb
                    # (not an exclamation/interjection/adjective greeting frame like 'welcome to')
                    if v_is_verb:
                        if not (cls._wordnet_pos(cand_head) & {"v"}):
                            continue
                        ch_entry = cls.get_ldoce_entry(cand_head)
                        ch_primary_pos = str((ch_entry or {}).get("pos") or "").lower()
                        if any(bad_p in ch_primary_pos for bad_p in ("interjection", "exclamation", "adjective", "adverb", "preposition", "determiner")):
                            continue
                    if k not in diff_part and k not in same_part and ships(k):
                        same_part.append(k)
                        if strength(k) == 0:
                            tier0_count += 1
                            if tier0_count >= count:
                                break

                diff_part.sort(key=lambda c: (strength(c), c))
                same_part.sort(key=lambda c: (strength(c), c))

                # Balanced selection: combine particle variation and verb variation
                if diff_part and same_part:
                    combined = [diff_part[0], same_part[0]]
                    if len(diff_part) > 1:
                        combined.append(diff_part[1])
                    elif len(same_part) > 1:
                        combined.append(same_part[1])
                    for item in diff_part[2:] + same_part[2:]:
                        if len(combined) >= count:
                            break
                        if item not in combined:
                            combined.append(item)
                    distractors = combined
                elif diff_part:
                    distractors = diff_part
                elif same_part:
                    distractors = same_part
            else:
                # Verb + Complement/Adjective/Adverb Collocation (e.g. 'make possible', 'fare better', 'go wrong')
                p_is_adj = bool(cls._wordnet_pos(p) & {"a", "s"})
                p_is_adv = p in _ADV_OR_ADJ_COMPLEMENTS or bool(cls._wordnet_pos(p) & {"r"})
                diff_comp: List[str] = []
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) == 2 and kw[0] == v and kw[1] != p and kw[1] not in particles:
                        if p_is_adv and str((cls.get_ldoce_entry(kw[1]) or {}).get("pos") or "").strip().lower() == "noun":
                            continue
                        if p_is_adj:
                            if not (cls._wordnet_pos(kw[1]) & {"a", "s"}):
                                continue
                        if ships(k) and k not in diff_comp:
                            diff_comp.append(k)
                if p_is_adv and len(diff_comp) < count:
                    for k in phrase_index:
                        kw = k.split()
                        if len(kw) == 2 and kw[1] == p and kw[0] != v:
                            if v == "fare" and kw[0] in ("do", "get"):
                                continue
                            if ships(k) and k not in diff_comp:
                                diff_comp.append(k)
                    for _adv_col in ("know better", "think better", "go wrong", "fall short", "come first", "work hard", "hold fast", "stand firm"):
                        if _adv_col != phrase_clean and ships(_adv_col) and _adv_col not in diff_comp:
                            diff_comp.append(_adv_col)
                if diff_comp:
                    diff_comp.sort(key=lambda c: (strength(c), c))
                    distractors = diff_comp[:count]

        # -------------------------------------------------------------
        # 2. Multi-word collocations & phrases (3+ words)
        # -------------------------------------------------------------
        elif n_words >= 3 and not distractors:
            cands_found: List[str] = []

            # Strategy A: Binomial Symmetric Match (e.g. 'pros and cons', 'up and down')
            if n_words == 3 and words[1] in ("and", "or"):
                op = words[1]
                popular_binomials = [
                    "bits and pieces", "back and forth", "give and take",
                    "checks and balances", "ups and downs", "black and white",
                    "born and bred", "by and large", "again and again"
                ]
                for k in popular_binomials:
                    if ships(k) and k not in cands_found:
                        cands_found.append(k)
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) == 3 and kw[1] == op and ships(k) and k not in cands_found:
                        cands_found.append(k)

            # Strategy A2: Possessive Idiom Structural Symmetry (`one's` / `your` at matching index)
            if "one's" in words and len(cands_found) < count:
                poss_idx = words.index("one's")
                _RELATIONAL_PREPS = {
                    "to", "on", "of", "from", "into", "for", "with", "at", "against", "about", "upon", "over"
                }
                _ADV_PARTICLES = {
                    "up", "down", "out", "off", "away", "back", "in", "along", "around", "through", "aside", "apart"
                }
                _UNDIGNIFIED_POSS_IDIOMS = {
                    _norm_poss(s)
                    for s in (
                        "pick your nose", "pick your teeth", "blow your nose", "get off your ass",
                        "get off your butt", "contemplate your navel", "lose your virginity",
                        "be your woman", "cry into your beer", "wet your pants", "soil your hands",
                        "have to your credit", "look to your laurels",
                    )
                }
                _POSS_SYNONYM_CLUSTERS = [
                    {_norm_poss(s) for s in ("do your best", "do your utmost", "do your part", "do your bit", "try your best")},
                    {_norm_poss(s) for s in ("turn your attention to", "turn your mind to", "direct your attention to", "give your attention to")},
                    {_norm_poss(s) for s in ("come to your aid", "come to your rescue", "come to your assistance", "go to your aid")},
                    {_norm_poss(s) for s in ("come into your mind", "cross your mind", "enter your mind", "pop into your head", "come into your head")},
                    {_norm_poss(s) for s in ("catch your eye", "catch your attention", "attract your attention", "get your attention")},
                    {_norm_poss(s) for s in ("keep your word", "keep your promise")},
                ]
                _target_poss_syns: Set[str] = set()
                for _p_grp in _POSS_SYNONYM_CLUSTERS:
                    if target_norm_poss in _p_grp:
                        _target_poss_syns |= _p_grp
                poss_cands: List[str] = []
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) != n_words or kw[poss_idx] not in ("your", "one's"):
                        continue
                    if target_head_is_det and kw[0] not in ("a", "an"):
                        continue
                    if not target_head_is_det and kw[0] in ("a", "an", "the"):
                        continue
                    norm_k = _norm_poss(k)
                    if norm_k in _UNDIGNIFIED_POSS_IDIOMS or norm_k in _target_poss_syns:
                        continue
                    if words[-1] in _RELATIONAL_PREPS:
                        if kw[-1] not in _RELATIONAL_PREPS:
                            continue
                    elif words[-1] in _ADV_PARTICLES:
                        if kw[-1] not in _ADV_PARTICLES:
                            continue
                    else:
                        if kw[-1] in (_RELATIONAL_PREPS | _ADV_PARTICLES):
                            continue
                        # Reject truncated 3-word prefixes of 4-word phrasal idioms (e.g. 'keep your hands [off]', 'let your guard [down]')
                        if phrase_index.get(k) == "phrasal_verb" or (
                            phrase_index.get(k) not in ("sense_unit", "phrase_row")
                            and any(f"{k} {p}" in phrase_index for p in (_RELATIONAL_PREPS | _ADV_PARTICLES))
                        ):
                            continue
                    if ships(k):
                        poss_cands.append(k)
                if poss_cands:
                    poss_cands.sort(
                        key=lambda c: (
                            0 if phrase_index.get(c) in ("sense_unit", "phrase_row") else 1,
                            0 if c.split()[0] == words[0] else (1 if (poss_idx >= 1 and c.split()[poss_idx - 1] == words[poss_idx - 1]) else 2),
                            strength(c),
                            c,
                        )
                    )
                    seen_poss_pref = {" ".join(words[:-1])}
                    seen_poss_tail = {" ".join(words[1:])}
                    for pc in poss_cands:
                        pc_w = pc.split()
                        p_pref = " ".join(pc_w[:-1])
                        p_tail = " ".join(pc_w[1:])
                        if p_pref in seen_poss_pref or p_tail in seen_poss_tail:
                            continue
                        seen_poss_pref.add(p_pref)
                        seen_poss_tail.add(p_tail)
                        cands_found.append(pc)
                        if len(cands_found) >= count:
                            break

            # Strategy B: Common Suffix / Tail Match (e.g. 'peace of mind' -> '... of mind')
            # Guard: When words[0] is a light/support verb (take, find, have, get, make, give, keep, feel, etc.)
            # and tl == n_words - 1 (matching the exact noun+prep tail, e.g. '... pleasure in' for 'take pleasure in'
            # -> 'find pleasure in'), swapping only the light verb creates synonymous collocation double-keys!
            light_support_verbs = {
                "take", "find", "have", "get", "make", "give", "keep", "hold", "bear",
                "pay", "show", "feel", "put", "set", "bring", "come", "go", "lose", "catch"
            }
            is_v_it_adj_target = (
                n_words == 3
                and words[0] in ("make", "find", "keep", "leave")
                and words[1] == "it"
                and bool(cls._wordnet_pos(words[2]) & {"a", "s"})
            )
            if (not cands_found or len(cands_found) < count) and not is_v_it_adj_target:
                tail_lengths = [2] if target_is_v_noun_prep else ([2, 1] if n_words >= 3 else [1])
                for tl in tail_lengths:
                    if words[0] in light_support_verbs and tl >= n_words - 1:
                        continue
                    tail_phrase = " " + " ".join(words[-tl:])
                    norm_tail_phrase = " " + _norm_poss(" ".join(words[-tl:]))
                    # Look for exact length matches with the same suffix (including possessive-normalized suffix)
                    tail_cands = []
                    for k in phrase_index:
                        kw = k.split()
                        if (
                            len(kw) == n_words
                            and (k.endswith(tail_phrase) or _norm_poss(k).endswith(norm_tail_phrase))
                            and ships(k)
                            and k not in cands_found
                        ):
                            tail_cands.append(k)
                    if tail_cands:
                        # Prioritize by authentic lexical strength
                        tail_cands.sort(key=lambda c: (strength(c), c))
                        cands_found.extend(tail_cands)
                        if len(cands_found) >= count:
                            break

            # Strategy C: Head-Bound Prepositional Transitive Match
            # (e.g. 'keep in touch with' -> 'keep a lid on', 'keep close tabs on', 'keep an eye on')
            if len(cands_found) < count:
                particles_set = {"with", "to", "for", "from", "on", "in", "at", "into", "against", "out of", "about", "of"}
                if words[-1] in particles_set:
                    head_cands = []
                    for k in phrase_index:
                        kw = k.split()
                        if len(kw) == n_words and kw[0] == words[0] and kw[-1] in particles_set and ships(k) and k not in cands_found:
                            head_cands.append(k)
                    if head_cands:
                        head_cands.sort(key=lambda c: (strength(c), c))
                        cands_found.extend(head_cands)

            # Strategy D: Light-Verb & Construct Prefix Match (e.g. 'have a try', 'take a look', 'make a move', 'make it possible')
            if len(cands_found) < count:
                prefix_2 = " ".join(words[:2]) + " "
                prefix_cands = []
                is_v_it_adj = (
                    n_words == 3
                    and words[0] in ("make", "find", "keep", "leave")
                    and words[1] == "it"
                    and bool(cls._wordnet_pos(words[2]) & {"a", "s"})
                )
                inf_taking_adjs = {
                    "possible", "impossible", "easy", "easier", "hard", "harder",
                    "difficult", "simple", "simpler", "safe", "safer", "necessary", "essential",
                }
                if is_v_it_adj:
                    # For extraposition '(make|find|keep|leave) it <adj>' (e.g. 'make it possible ... to do [sth]', 'find it hard ... to do [sth]'),
                    # prioritize attested '<verb> it <adj/participle>' constructions that take that-clauses or NP complements
                    # rather than '(for sb) to do sth' to guarantee zero double-key collision.
                    if words[0] == "make":
                        curated_v_it = ["make it clear", "make it known", "make it certain", "make it plain", "make it good"]
                    elif words[0] == "find":
                        curated_v_it = ["find it strange", "find it odd", "find it clear", "find it funny", "find it weird"]
                    else:
                        curated_v_it = [f"{words[0]} it clear", f"{words[0]} it plain", f"{words[0]} it secret", f"{words[0]} it quiet"]
                    for ph in curated_v_it:
                        if ph != phrase_clean and ph not in exclude and ph not in synonym_set and ph not in cands_found and ph not in prefix_cands:
                            if words[0] != "make" or ships(ph):
                                prefix_cands.append(ph)
                popular_have_a = ["have a go", "have a care", "have a heart", "have a think", "have a word", "have a ball", "have a breather"]
                for ph in popular_have_a:
                    if prefix_2.strip() == "have a" and ships(ph) and ph not in cands_found and ph not in prefix_cands:
                        prefix_cands.append(ph)
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) == n_words and k.startswith(prefix_2) and ships(k) and k not in cands_found and k not in prefix_cands:
                        if is_v_it_adj:
                            if kw[2] in cls._PHRASE_PARTICLE_WORDS or kw[2] in {"snappy"}:
                                continue
                            if words[2] in inf_taking_adjs and kw[2] in inf_taking_adjs:
                                continue
                        prefix_cands.append(k)
                if prefix_cands:
                    if not is_v_it_adj:
                        prefix_cands.sort(key=lambda c: (strength(c), c))
                    cands_found.extend(prefix_cands)

            # Strategy E: Structural Sister Frames (Prepositional / Nominal frames)
            if len(cands_found) < count:
                if "end of" in phrase_clean:
                    for c in ["at the beginning of", "in the middle of", "for the rest of", "at the start of"]:
                        if ships(c) and c not in cands_found:
                            cands_found.append(c)
                elif "piece of" in phrase_clean:
                    for c in ["a series of", "a range of", "a couple of", "a matter of", "a variety of"]:
                        if ships(c) and c not in cands_found:
                            cands_found.append(c)
                elif "thanks to" in phrase_clean:
                    for c in ["show respect to", "pay tribute to", "give credit to", "express gratitude to"]:
                        if ships(c) and c not in cands_found:
                            cands_found.append(c)
                elif "type of" in phrase_clean:
                    for c in ["any form of", "any sort of", "any kind of", "any part of"]:
                        if ships(c) and c not in cands_found:
                            cands_found.append(c)
                elif target_head_is_prep and n_words == 3 and words[1] in ("a", "an"):
                    for c in ["at a time", "in a hurry", "on a diet", "at a loss", "in a row", "on a roll", "at a glance", "in a mess", "on a whim"]:
                        if ships(c) and c not in cands_found:
                            cands_found.append(c)
                elif target_head_is_prep and n_words == 4 and words[1] in ("a", "an"):
                    for c in ["on a regular basis", "at a rapid pace", "on a tight budget", "in a big way", "on a daily basis", "to a certain extent", "at a steady rate"]:
                        if ships(c) and c not in cands_found:
                            cands_found.append(c)

            # Strategy F: Same-Length Syntactic Frame Matching (Same connector/prep at same index)
            if len(cands_found) < count:
                seen_f_prefixes = {" ".join(c.split()[:-1]) for c in cands_found if len(c.split()) >= 3}
                for idx, w in enumerate(words):
                    if w in ("of", "in", "to", "for", "with", "on", "at", "from", "over", "under", "into", "through"):
                        for k in phrase_index:
                            kw = k.split()
                            if len(kw) == n_words and kw[idx] == w and ships(k) and k not in cands_found:
                                k_pref = " ".join(kw[:-1]) if len(kw) >= 3 else k
                                if k_pref in seen_f_prefixes:
                                    continue
                                seen_f_prefixes.add(k_pref)
                                cands_found.append(k)
                                if len(cands_found) >= count + 6:
                                    break
                        if len(cands_found) >= count + 3:
                            break

            distractors = cands_found

        # -------------------------------------------------------------
        # 3. General Fallback if pool is still sparse
        # -------------------------------------------------------------
        if len(distractors) < count:
            if n_words == 2 and not is_compound_noun:
                for fallback_cand in ["carry out", "take over", "bring about", "set up", "look into", "come across", "give in"]:
                    if ships(fallback_cand) and fallback_cand not in distractors:
                        distractors.append(fallback_cand)
                    if len(distractors) >= count:
                        break
            elif n_words >= 3:
                seen_fb_prefixes = {" ".join(c.split()[:-1]) for c in distractors if len(c.split()) >= 3}
                # Fallback to authentic multi-word phrases of matching length from phrase_index
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) == n_words and ships(k) and k not in distractors:
                        k_pref = " ".join(kw[:-1]) if len(kw) >= 3 else k
                        if k_pref in seen_fb_prefixes:
                            continue
                        seen_fb_prefixes.add(k_pref)
                        distractors.append(k)
                    if len(distractors) >= count:
                        break

        res = distractors[:count]
        if "one's" in words:
            res = [re.sub(r"\byour\b", "one's", d) for d in res]
        res = [
            re.sub(r"\ba\s+(?!(?:uni|use|usu|urol|utens|eul|eup|eur|one)\w*)([aeiou][a-z]+)\b", r"an \1", d)
            for d in res
        ]
        return res

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
                return tok.tag_.upper()
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
                    re_parts.append(r"(?P<poss>\b(?:my|your|his|her|its|our|their|one's|[a-z]+'s)\b)")
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
                if not has_singular_head:
                    for opt in base_options:
                        ow = opt.strip().lower().split()
                        if poss_idx + 1 < len(ow):
                            noun_after = ow[poss_idx + 1]
                            if not noun_after.endswith("s") or noun_after.endswith("ss"):
                                has_singular_head = True
                                break
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
            return new_desc, reconciled_opts, new_target, tag

        if pos_low != "verb" or inflection_desc != "base form":
            return inflection_desc, base_options, target_word, "VB"
        tag = cls._example_target_tag(authentic_example, target_word)
        if tag not in ("VBD", "VBZ", "VBG", "VBN"):
            return inflection_desc, base_options, target_word, "VB"
        inflected: List[str] = []
        for opt in base_options:
            form = cls._inflect_verb(opt, tag)
            if not form or not cls._morph_form_valid(opt, form, tag):
                return inflection_desc, base_options, target_word, "VB"  # unsafe to rewrite
            inflected.append(form)
        new_target = inflected[target_index] if 0 <= target_index < len(inflected) else target_word
        # A1: an inflected target that is not an attested English form must never ship,
        # even when spaCy was willing to lemmatize it back to the base verb.
        if not cls.is_attested_form(new_target):
            return inflection_desc, base_options, target_word, "VB"

        # Lexical POS Boundary Guard: reject inflection ONLY if the inflected form is an independent
        # dictionary headword entry (`word == new_target`) with a non-verb POS (e.g. annoy[verb] -> annoyed[adjective entry]).
        # Note: get_ldoce_entry('pledged') falls back to the lemma entry 'pledge' (homograph 1 = noun),
        # whose `word` is 'pledge', NOT 'pledged' — that is a lemmatized fallback, not a dedicated participial-adjective entry!
        _inflected_entry = cls.get_ldoce_entry(new_target)
        if (
            _inflected_entry
            and _inflected_entry.get("kind") == "article"
            and (_inflected_entry.get("word") or "").strip().lower() == new_target.lower()
        ):
            _inflected_pos = (_inflected_entry.get("pos") or "").lower()
            if _inflected_pos and _inflected_pos not in ("verb", "v"):
                return inflection_desc, base_options, target_word, "VB"

        return cls._VERB_FORM_LABELS.get(tag, "base form"), inflected, new_target, tag

    # -------------------------------------------------------------------------
    # 4. WordNet Distractor Assembly & Zero-Double-Key Guarantee
    # -------------------------------------------------------------------------
    @classmethod
    def get_synonyms(cls, word: str) -> Set[str]:
        """Returns all synonymous lemma strings for the word across all synsets."""
        wn = cls.get_wordnet()
        synonyms = set()
        for syn in wn.synsets(word.lower()):
            for w in syn.words():
                lemma = w.lemma().replace("_", " ").lower()
                if lemma != word.lower():
                    synonyms.add(lemma)
        return synonyms

    @classmethod
    def get_antonyms(
        cls,
        word: str,
        pos: Optional[str] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
    ) -> Set[str]:
        """Returns direct and indirect (synset-peer, satellite-head, and cross-POS pertainym/derivation)
        antonyms from WordNet sense relations, filtered by headword and taboo/proper-noun safety gates."""
        wn = cls.get_wordnet()
        target = (word or "").strip().lower()
        if not target:
            return set()
        wn_pos = None
        p_single = None
        if pos:
            pos_low = pos.lower()
            if "adj" in pos_low or pos_low in ("a", "s"):
                wn_pos = ("a", "s")
                p_single = "a"
            elif "noun" in pos_low or pos_low == "n":
                wn_pos = ("n",)
                p_single = "n"
            elif "verb" in pos_low or pos_low == "v":
                wn_pos = ("v",)
                p_single = "v"
            elif "adv" in pos_low or pos_low == "r":
                wn_pos = ("r",)
                p_single = "r"

        candidate_synsets = []
        seen_syn_ids = set()
        for w in wn.words(target):
            if w.lemma().lower() != target:
                continue
            for syn in w.synsets():
                if wn_pos and syn.pos not in wn_pos:
                    continue
                if syn.id not in seen_syn_ids:
                    seen_syn_ids.add(syn.id)
                    candidate_synsets.append(syn)

        if (definition or quote) and len(candidate_synsets) > 1:
            nlp = cls.get_spacy()
            ctx_doc = nlp(((definition or "") + " " + (quote or "")).lower())
            ctx_toks = {t.lemma_.lower() for t in ctx_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - {target}
            if ctx_toks:
                scored_syns = []
                for syn in candidate_synsets:
                    s_doc = nlp((syn.definition() + " " + " ".join(syn.examples())).lower())
                    s_toks = {t.lemma_.lower() for t in s_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3}
                    scored_syns.append((len(ctx_toks & s_toks), syn))
                scored_syns.sort(key=lambda x: x[0], reverse=True)
                if scored_syns[0][0] > 0:
                    best_ov = scored_syns[0][0]
                    candidate_synsets = [syn for ov, syn in scored_syns if ov >= max(1, best_ov - 1)]

        antonyms: Set[str] = set()

        def _add_ant(lem: str):
            al = lem.replace("_", " ").strip().lower()
            if (
                al
                and al != target
                and " " not in al
                and not cls._is_taboo_or_proper_noun(al, wn_pos=p_single)
                and cls.is_exact_pedlex_headword(al)
            ):
                antonyms.add(al)

        for syn in candidate_synsets:
            # 1. Direct and synset-peer sense antonyms
            for sense in syn.senses():
                for ant_sense in sense.get_related("antonym"):
                    _add_ant(ant_sense.word().lemma())
            # 2. Adjective satellite ('s') -> head ('a') -> antonym
            if syn.pos == "s":
                for head in syn.get_related("similar"):
                    for hs in head.senses():
                        for ant_sense in hs.get_related("antonym"):
                            _add_ant(ant_sense.word().lemma())
            # 3. Cross-POS pertainym / derivation antonym projection
            _CONCRETE_NOUN_LEXFILES_ANT = {
                "noun.substance", "noun.artifact", "noun.plant", "noun.animal",
                "noun.food", "noun.object", "noun.body", "noun.location",
            }
            for sense in syn.senses():
                if sense.word().lemma().lower() != target:
                    continue
                if syn.pos == "n" and syn.lexfile() in _CONCRETE_NOUN_LEXFILES_ANT:
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
                            if syn.pos == "n" and proj.synset().lexfile() != syn.lexfile():
                                continue
                            if not wn_pos or proj.synset().pos in wn_pos:
                                _add_ant(proj.word().lemma())
                        if syn.pos == "r" and b_ant.synset().pos in ("a", "s"):
                            adj_lem = b_ant.word().lemma().lower()
                            for ly_c in (
                                [adj_lem + "ly"]
                                + ([adj_lem[:-1] + "ily"] if adj_lem.endswith("y") and len(adj_lem) > 2 else [])
                                + ([adj_lem[:-1] + "y"] if adj_lem.endswith("le") and len(adj_lem) > 2 else [])
                                + ([adj_lem + "ally"] if adj_lem.endswith("ic") else [])
                            ):
                                _add_ant(ly_c)
        return antonyms

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
            t_lvl_str = cls.get_word_cefr(target_lower) or "B2"
            t_rank = cls.CEFR_ORDER.get(t_lvl_str, 4)
            if t_rank <= 3:
                # Elementary / Intermediate Fallbacks (A1 - B1 core words)
                common_fallbacks = {
                    "v": ["choose", "accept", "decide", "expect", "follow", "notice", "explain", "remain"],
                    "n": ["choice", "reason", "matter", "situation", "effort", "problem", "action", "change"],
                    "a": ["simple", "common", "certain", "different", "similar", "natural", "clear", "direct"],
                    "r": ["deeply", "strictly", "tightly", "widely", "carefully", "clearly", "readily"]
                }
            else:
                # Advanced Fallbacks (B2 - C1 academic words)
                common_fallbacks = {
                    "v": ["assume", "indicate", "require", "maintain", "assess"],
                    "n": ["aspect", "factor", "criterion", "approach", "phenomenon"],
                    "a": ["apparent", "consistent", "variable", "significant", "distinct"],
                    "r": ["deeply", "strictly", "tightly", "widely", "carefully", "clearly", "readily"]
                }
            default_fb = common_fallbacks.get(pos, common_fallbacks["n" if pos == "n" else ("a" if pos == "a" else ("r" if pos == "r" else "v"))])
            for fb in default_fb:
                if fb not in target_synonyms and fb not in final_distractors:
                    final_distractors.append(fb)
                if len(final_distractors) == count:
                    break

        return final_distractors

    _CORREL_FIRST_ARMS: Dict[str, List[str]] = {
        "not only": ["neither", "either", "no sooner", "hardly", "whether", "scarcely"],
        "neither": ["either", "not only", "no sooner", "hardly", "whether", "both"],
        "either": ["neither", "not only", "no sooner", "hardly", "whether", "both"],
        "no sooner": ["hardly", "scarcely", "not only", "neither", "either", "whether"],
        "hardly": ["no sooner", "not only", "neither", "either", "whether"],
        "scarcely": ["no sooner", "not only", "neither", "either", "whether"],
        "whether": ["either", "neither", "not only", "no sooner", "hardly"],
        "would rather": ["had better", "would prefer", "not only", "neither", "either"],
    }

    _LV_TRIAD_NOUN_CONTRASTS: Dict[str, Tuple[List[str], Set[str]]] = {
        "attention": (["impact", "role", "advantage", "use", "place", "charge"], {"attention", "heed", "tribute", "homage", "respect"}),
        "impact": (["attention", "role", "advantage", "use", "place", "charge"], {"impact", "effect", "influence", "bearing"}),
        "effect": (["attention", "role", "advantage", "use", "place", "charge"], {"impact", "effect", "influence", "bearing"}),
        "role": (["attention", "impact", "advantage", "use", "charge", "care"], {"role", "part"}),
        "part": (["attention", "impact", "advantage", "use", "charge", "care"], {"role", "part"}),
        "advantage": (["attention", "impact", "role", "place", "effort", "decision"], {"advantage", "use", "care", "charge", "notice", "account"}),
        "use": (["attention", "impact", "role", "place", "effort", "decision"], {"advantage", "use", "sense", "fun", "room", "way"}),
    }

    @classmethod
    def get_cxn_banned_words(cls, ban_groups: Optional[List[str]]) -> Set[str]:
        """Return all words belonging to `ban_groups` in `pedlex_cxn_index`."""
        if not ban_groups or not cls._ensure_ldoce_conn() or not cls._ldoce_table_exists("pedlex_cxn_index"):
            return set()
        clean_groups = [str(g).strip() for g in ban_groups if g]
        if not clean_groups:
            return set()
        placeholders = ",".join("?" for _ in clean_groups)
        try:
            cur = cls._ldoce_conn.execute(
                f"SELECT DISTINCT word FROM pedlex_cxn_index WHERE subcat_group IN ({placeholders})",
                tuple(clean_groups),
            )
            return {str(r[0]).strip().lower() for r in cur.fetchall() if r and r[0]}
        except Exception:
            return set()

    @classmethod
    def _match_cxn_spec_to_context(
        cls,
        w_clean: str,
        specs: List[Dict[str, Any]],
        quote: Optional[str],
        context_anchor: Optional[str],
        anchor_type: Optional[str],
        require_context_match: bool,
    ) -> Optional[Dict[str, Any]]:
        """Match a candidate `pedlex_cxn_index` spec against the active syntactic context."""
        if not specs:
            return None
        q_low = (quote or "").lower()
        anc_low = str(context_anchor or "").strip().lower()
        atype = str(anchor_type or "").strip().lower()

        for sp in specs:
            subcat = str(sp.get("subcat_group") or "")
            prep_a = str(sp.get("prep_anchor") or "").strip().lower()
            if not prep_a and subcat.startswith("V_OBJ_PREP_"):
                prep_a = subcat.split("V_OBJ_PREP_", 1)[1].strip().lower()
            # 1. Explicit constructional anchor_type match
            if atype == "it_extrapos" and subcat in ("ADJ_IT_EXTRAPOS", "ADJ_TOUGH", "ADJ_TOUGH_BOTH", "ADJ_MANDATIVE_SUBJ"):
                return sp
            if atype == "gerund" and subcat == "V_GERUND_ONLY":
                return sp
            if atype == "to_inf" and subcat == "V_INF_ONLY":
                return sp
            if atype == "sb_to_do" and subcat == "V_OBJ_TO_DO":
                return sp
            if atype == "shell_to_do" and subcat == "N_SHELL_TO_DO":
                return sp
            if atype == "shell_of_doing" and subcat == "N_SHELL_OF_DOING":
                return sp
            if atype == "shell_that" and subcat == "N_SHELL_THAT_ONLY":
                return sp
            if atype == "correl_pair" and subcat == "CORREL_PAIR":
                return sp
            if atype == "fronted_inv" and subcat in ("ADV_FRONTED_INV", "PHR_FRONTED_INV"):
                return sp
            if atype == "npi" and subcat == "NPI_ITEM":
                return sp
            if atype == "mandative_subj" and subcat in ("V_MANDATIVE_SUBJ", "ADJ_MANDATIVE_SUBJ", "N_MANDATIVE_SUBJ"):
                return sp
            if atype == "lv_triad" and (subcat.startswith("LV_TRIAD_") or subcat == "LV_DEVERBAL_TRIAD"):
                return sp

            # 2. Multi-word phrase entry (sense_idx == -1)
            if " " in w_clean and int(sp.get("sense_idx", -1)) == -1:
                return sp

            # 3. Trivalent Verb + Object + Preposition (+ NP / -ing)
            if (subcat.startswith("V_OBJ_PREP_") or subcat in ("V_OBJ_FROM_ING", "V_OBJ_INTO_ING")) and prep_a:
                if (atype == "prep" and anc_low == prep_a) or (
                    q_low and re.search(rf"\b{re.escape(prep_a)}\s+(?:[a-z]+\s+)?[a-z]+", q_low)
                ):
                    return sp

            # 4. Quote-based syntactic frame match
            if q_low:
                if subcat in ("ADJ_IT_EXTRAPOS", "ADJ_TOUGH", "ADJ_TOUGH_BOTH") and re.search(
                    rf"\bit\s+(?:is|was|seems|appears|becomes|gets|'s|would\s+be|will\s+be|can\s+be|could\s+be)\s+(?:not\s+|very\s+|quite\s+|extremely\s+|particularly\s+|especially\s+)?{re.escape(w_clean)}\b[^.?!;]*\bto\s+[a-z]+\b",
                    q_low,
                ):
                    return sp
                if (
                    subcat == "V_GERUND_ONLY"
                    and atype not in ("prep", "object", "verb_subject")
                    and re.search(rf"\b{re.escape(w_clean)}(?:s|es|ed|d|ing)?\s+(?:not\s+)?[a-z]+ing\b", q_low)
                ):
                    return sp
                if (
                    subcat == "V_INF_ONLY"
                    and atype not in ("prep", "object", "verb_subject")
                    and re.search(rf"\b{re.escape(w_clean)}(?:s|es|ed|d|ing)?\s+(?:not\s+)?to\s+[a-z]+\b", q_low)
                ):
                    return sp
                if subcat == "N_SHELL_TO_DO" and w_clean not in (
                    "device", "machine", "tool", "instrument", "appliance",
                    "vehicle", "ticket", "room", "camera", "sensor", "weapon",
                ):
                    # Must be followed by `to + base verb (VB)`, never `to + ProperNoun/Determiner/NP` (e.g. "a train ticket to London")
                    _m_to = re.search(rf"\b{re.escape(w_clean)}(?:s|es|ies)?\s+to\s+([A-Za-z]+)\b", quote or "")
                    if _m_to:
                        _after_to = _m_to.group(1)
                        _after_to_low = _after_to.lower()
                        _np_starters = {
                            "the", "a", "an", "my", "your", "his", "her", "its", "our", "their",
                            "this", "that", "these", "those", "some", "any", "each", "every", "all",
                            "both", "another", "other", "many", "most", "several", "few", "no",
                            "whom", "which", "what", "everyone", "everybody", "someone", "somebody",
                            "anyone", "anybody", "nobody", "nothing", "everything", "something",
                            "him", "them", "us", "me", "you", "one", "ones",
                        }
                        if not _after_to[0].isupper() and _after_to_low not in _np_starters:
                            _to_ent = cls.get_ldoce_entry(_after_to_low) or {}
                            _to_senses = _to_ent.get("senses") or []
                            _has_verb_sense = (
                                cls._pos_code(_to_ent.get("pos") or "") == "v"
                                or any(cls._pos_code(sn.get("pos") or "") == "v" for sn in _to_senses)
                                or _after_to_low in ("be", "have", "do", "go", "get", "make", "take", "see", "find", "give", "keep")
                            )
                            if _has_verb_sense:
                                return sp
                if subcat == "N_SHELL_OF_DOING" and (
                    (atype == "prep" and anc_low == "of" and re.search(r"\bof\s+(?:[a-z]+\s+)?[a-z]+ing\b", q_low))
                    or re.search(rf"\b{re.escape(w_clean)}\s+of\s+(?:[a-z]+\s+)?[a-z]+ing\b", q_low)
                ):
                    return sp
                if subcat == "N_SHELL_THAT_ONLY" and re.search(rf"\b{re.escape(w_clean)}\s+that\s+[a-z]+\b", q_low):
                    return sp
                if subcat in ("ADV_FRONTED_INV", "PHR_FRONTED_INV") and re.search(
                    rf"(?:^|[.;!?]\s*|\b(?:and|but)\s+){re.escape(w_clean)}\s+(?:do|does|did|have|has|had|am|is|are|was|were|can|could|will|would|shall|should|may|might|must)\b",
                    q_low,
                ):
                    return sp
                if subcat == "NPI_ITEM" and re.search(
                    r"\b(?:not|n't|never|hardly|scarcely|barely|seldom|rarely|little|few|no|neither|nor|without|doubt|deny|refused?|failed?|unlikely|impossible)\b",
                    q_low,
                ):
                    return sp
                if subcat in ("V_MANDATIVE_SUBJ", "ADJ_MANDATIVE_SUBJ") and re.search(
                    rf"\b{re.escape(w_clean)}(?:s|es|ed|d|ing)?\s+that\s+(?:[a-z]+\s+){{1,4}}(?:should\s+[a-z]+|be\b|not\s+[a-z]+)",
                    q_low,
                ):
                    return sp

        if not require_context_match:
            if q_low and len(specs) > 1:
                for sp in specs:
                    prep_a = str(sp.get("prep_anchor") or "").strip().lower()
                    if prep_a and re.search(rf"\b{re.escape(prep_a)}\b", q_low):
                        return sp
            return specs[0]
        return None

    @classmethod
    def get_construction_contrast_distractors(
        cls,
        target_word: str,
        pos: str = "noun",
        sense_idx: Optional[int] = None,
        quote: Optional[str] = None,
        context_anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        target_count: int = 3,
        exclude_words: Optional[Set[str]] = None,
        require_context_match: bool = False,
    ) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """Query `pedlex_cxn_index` for construction-family contrasting distractors (`contrast_groups \\ ban_groups`).
        Returns `(distractors, matched_cxn_spec)`."""
        w_clean = (target_word or "").strip().lower()
        if not w_clean:
            return [], None
        excl = {w.strip().lower() for w in (exclude_words or set()) if w} | {w_clean}

        # Special handler 1: Decomposed Family 2 Correlative First Arms ('not only', 'neither', 'no sooner', etc.)
        if w_clean in cls._CORREL_FIRST_ARMS and (not require_context_match or anchor_type == "correl_pair"):
            cands = [c for c in cls._CORREL_FIRST_ARMS[w_clean] if c not in excl]
            synth_spec = {
                "word": w_clean,
                "pos": "phrase",
                "family_id": 2,
                "subcat_group": "CORREL_PAIR",
                "canonical_blank": w_clean,
                "contrast_groups": ["CORREL_MISMATCH"],
                "ban_groups": ["CORREL_SYNONYM"],
            }
            return cands[:target_count], synth_spec

        # Special handler 2: Decomposed Family 6 Light-Verb Triad Head Nouns ('attention', 'impact', 'role', etc.)
        if anchor_type == "lv_triad" and w_clean in cls._LV_TRIAD_NOUN_CONTRASTS:
            contrast_nouns, banned_nouns = cls._LV_TRIAD_NOUN_CONTRASTS[w_clean]
            cands = [c for c in contrast_nouns if c not in excl and c not in banned_nouns]
            synth_spec = {
                "word": w_clean,
                "pos": "noun",
                "family_id": 6,
                "subcat_group": "LV_TRIAD_NOUN",
                "canonical_blank": w_clean,
                "contrast_groups": ["LV_TRIAD_CONTRAST"],
                "ban_groups": [],
            }
            return cands[:target_count], synth_spec

        # Special handler 3: Family 1A Preparatory Object Extraposition ('make it easy/possible/clear/a rule')
        if w_clean.startswith("make it "):
            _fam1a_ease_group = {
                "make it easy", "make it easier", "make it simple",
                "make it possible", "make it impossible",
                "make it hard", "make it difficult", "make it tough",
            }
            _fam1a_foils = [
                "make it a rule",
                "make it a point",
                "make it a habit",
                "make it clear",
                "make it plain",
                "make it known",
            ]
            if w_clean not in _fam1a_ease_group:
                _fam1a_foils = ["make it possible", "make it easy", "make it hard"] + _fam1a_foils
            cands = [
                c for c in _fam1a_foils
                if c != w_clean and c not in excl and (w_clean not in _fam1a_ease_group or c not in _fam1a_ease_group)
            ]
            synth_spec = {
                "word": w_clean,
                "pos": "phrase",
                "family_id": 1,
                "subcat_group": "MAKE_IT_EXTRAPOS",
                "canonical_blank": w_clean,
                "contrast_groups": ["MAKE_IT_COMP_CONTRAST"],
                "ban_groups": sorted(_fam1a_ease_group) if w_clean in _fam1a_ease_group else [],
            }
            return cands[:target_count], synth_spec

        specs = cls.get_word_cxn_specs(w_clean, sense_idx=sense_idx)
        if not specs:
            return [], None

        # Enforce target POS isolation so noun/verb homographs (e.g. 'stamp' noun vs 'stamp' verb)
        # never leak cross-POS subcategorization specs.
        p_code_cxn = cls._pos_code(pos)
        if " " not in w_clean and p_code_cxn in ("n", "v", "adj", "adv"):
            specs = [
                sp for sp in specs
                if not sp.get("pos") or cls._pos_code(str(sp.get("pos"))) == p_code_cxn
            ]
            if not specs:
                return [], None

        chosen_spec = cls._match_cxn_spec_to_context(
            w_clean=w_clean,
            specs=specs,
            quote=quote,
            context_anchor=context_anchor,
            anchor_type=anchor_type,
            require_context_match=require_context_match,
        )
        if not chosen_spec:
            return [], None

        contrast_groups = list(chosen_spec.get("contrast_groups") or [])
        ban_groups = list(chosen_spec.get("ban_groups") or [])
        if not contrast_groups:
            return [], chosen_spec

        raw_cands = cls.query_cxn_contrast_distractors(
            contrast_groups=contrast_groups,
            ban_groups=ban_groups,
            pos=chosen_spec.get("pos") or None,
            exclude_words=excl,
            limit=max(target_count * 6, 20),
        )
        # Filter by CEFR compliance and deterministic CRC32 ordering so selection is stable and varied
        valid_cands = [
            c for c in raw_cands
            if c != w_clean
            and c not in excl
            and (" " in c or cls.is_cefr_compliant_distractor(c, w_clean))
        ]
        if len(valid_cands) < target_count:
            valid_cands = [c for c in raw_cands if c != w_clean and c not in excl]

        valid_cands.sort(key=lambda c: zlib.crc32(f"{w_clean}:{c}".encode("utf-8")) & 0xFFFFFFFF)
        return valid_cands[:target_count], chosen_spec
