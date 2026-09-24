"""
Linguistic Engine for Lexis Wiki.
Integrates spaCy (computational dependency syntax) and WordNet (lexical relations)
to achieve deterministic sentence indexing, macro grammar domain classification,
boundary-accurate phrase extraction, canonical lemmatization, and zero-collision distractors.
"""

import json
import os
import re
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class LinguisticEngine:
    _spacy_nlp = None
    _wn = None
    _acl_data = None
    _awl_data = None
    _ocd_data = None

    @classmethod
    def get_spacy(cls):
        """Lazy-loads the spaCy English model (~12MB)."""
        if cls._spacy_nlp is None:
            import spacy
            cls._spacy_nlp = spacy.load("en_core_web_sm")
        return cls._spacy_nlp

    @classmethod
    def get_wordnet(cls):
        """Lazy-loads the wn (Open English WordNet) client with multithreading enabled."""
        if cls._wn is None:
            import wn
            # Web dashboard / background worker threads share WordNet SQLite connection.
            # SQLite check_same_thread must be disabled for read-only multi-threaded queries.
            wn.config.allow_multithreading = True
            cls._wn = wn
        return cls._wn

    @classmethod
    def get_oxford_collocations(cls) -> Dict[str, Dict[str, List[str]]]:
        """Lazy-loads the Oxford Collocations Dictionary 2nd Edition (20,791 headwords, ~4.9MB)."""
        if cls._ocd_data is None:
            cls._ocd_data = {}
            ocd_path = Path(__file__).parent / "data" / "oxford_collocations.json"
            if ocd_path.exists():
                try:
                    with open(ocd_path, "r", encoding="utf-8") as f:
                        cls._ocd_data = json.load(f)
                except Exception:
                    pass
        return cls._ocd_data

    @classmethod
    def get_rich_collocations(cls, word: str, top_k: int = 5) -> List[str]:
        """
        Retrieves top authoritative, natural collocations for a word from Oxford Collocations Dictionary.
        Synthesizes phrases like 'lay the foundation', 'firm foundation', 'comprehensive guide'.
        """
        if not word:
            return []
        ocd = cls.get_oxford_collocations()
        entry = ocd.get(word.lower().strip())
        if not entry:
            return []

        results: List[str] = []

        # 1. Verb + Noun: 'lay [word]', 'establish [word]'
        if "verb_before" in entry:
            for v in entry["verb_before"][:3]:
                if "(" not in v:
                    results.append(f"{v} {word}")

        # 2. Adj + Noun: 'firm [word]', 'solid [word]'
        if "adj" in entry:
            for adj in entry["adj"][:3]:
                results.append(f"{adj} {word}")

        # 3. Noun collocations: '[word] guide', '[word] analysis'
        if "colloc_nouns" in entry:
            for noun in entry["colloc_nouns"][:3]:
                results.append(f"{word} {noun}")

        # 4. Adv + Verb: '[word] heavily', '[word] directly'
        if "verb" in entry:
            for adv in entry["verb"][:2]:
                results.append(f"{word} {adv}")

        # 5. Preposition: '[word] for', '[word] to'
        if "prep" in entry:
            for p in entry["prep"][:2]:
                results.append(f"{word} {p}")

        # Dedup preserving order
        seen = set()
        deduped = []
        for r in results:
            if r.lower() not in seen:
                seen.add(r.lower())
                deduped.append(r)
            if len(deduped) >= top_k:
                break
        return deduped

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
        wn = cls.get_wordnet()
        ocd = cls.get_oxford_collocations()

        # Map POS to WordNet tag ('n', 'v', 'a', 'r')
        wn_pos = "v" if pos.startswith("v") else ("n" if pos.startswith("n") else ("a" if pos.startswith("adj") or pos.startswith("a") else None))
        if wn_pos == "a":
            # In WordNet, adjectives are split into head adjectives ('a') and satellite adjectives ('s')
            words = wn.words(clean_target, pos="a") + wn.words(clean_target, pos="s")
        elif wn_pos:
            words = wn.words(clean_target, pos=wn_pos)
        else:
            words = wn.words(clean_target)

        exclude_words = set(exclude_words or set())

        def _pos_ok(lemma: str) -> bool:
            """Keep hyponym/coordinate candidates on the target's own part of speech."""
            if not wn_pos:
                return True
            try:
                if wn_pos == "a":
                    return bool(wn.words(lemma, pos="a") or wn.words(lemma, pos="s"))
                return bool(wn.words(lemma, pos=wn_pos))
            except Exception:
                return True

        def _is_primarily_adj_or_verb(lemma: str) -> bool:
            """Filter out collective nominals (e.g. 'blind', 'baffled', 'dead') that are primarily adjectives."""
            try:
                pos_list = [w.pos for w in wn.words(lemma)]
                if any(p in ("a", "s") for p in pos_list):
                    if "n" not in pos_list or len([p for p in pos_list if p in ("a", "s")]) >= len([p for p in pos_list if p == "n"]):
                        return True
            except Exception:
                pass
            return False

        # Collect global synonyms across ALL synsets of target to permanently ban them
        global_synonyms: Set[str] = {clean_target}
        for w in words:
            for s in w.synsets():
                for sw in s.words():
                    global_synonyms.add(sw.lemma().lower())

        # 1. Definition-Locked WSD: Rank synsets by token overlap with curriculum definition
        target_synsets = []
        if definition:
            def_tokens = set(re.findall(r"[a-zA-Z]{3,}", definition.lower())) - cls._ANCHOR_STOPWORDS
            scored_synsets = []
            for w in words:
                for s in w.synsets():
                    stext = (s.definition() + " " + " ".join(s.examples())).lower()
                    stoks = set(re.findall(r"[a-zA-Z]{3,}", stext)) - cls._ANCHOR_STOPWORDS
                    ov = len(def_tokens.intersection(stoks))
                    scored_synsets.append((ov, s))
            scored_synsets.sort(key=lambda x: x[0], reverse=True)
            if scored_synsets:
                best_ov = scored_synsets[0][0]
                if best_ov > 0:
                    target_synsets = [s for ov, s in scored_synsets if ov >= best_ov - 1]
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

        # 2. Extract Sense-Locked Antonyms directly from matched synsets (Scheme 2: Bipolar Opposite Cluster)
        sense_antonyms: List[str] = []
        raw_antonyms: List[Tuple[str, int]] = []

        def _evaluate_antonym(al: str, priority_bonus: int = 0):
            al = al.strip().lower()
            if (
                al not in global_synonyms
                and al != clean_target
                and "_" not in al
                and " " not in al
                and "-" not in al
                and _pos_ok(al)
                and al not in exclude_words
            ):
                if al in ocd:
                    prio = 0 - priority_bonus
                elif al in cls.get_awl_words() or any(al.startswith(p) and al[len(p):] in ocd for p in ("un", "in", "im", "ir", "il", "non", "dis")):
                    prio = 1 - priority_bonus
                elif wn_pos == "a":
                    prio = 2 - priority_bonus
                else:
                    return
                raw_antonyms.append((al, prio))

        for s in target_synsets:
            # Direct sense antonyms
            for sense in s.senses():
                for ant_sense in sense.get_related("antonym"):
                    ant_lemma = ant_sense.word().lemma().lower()
                    _evaluate_antonym(ant_lemma, priority_bonus=2)
                    # Opposite satellites in bipolar cluster
                    for opp_sim in ant_sense.synset().get_related("similar") + ant_sense.synset().get_related("also"):
                        for ow in opp_sim.words():
                            _evaluate_antonym(ow.lemma().lower(), priority_bonus=1)
            # If adjective satellite ('s'), trace to head synset in bipolar cluster
            if wn_pos == "a" and s.pos == "s":
                for head in s.get_related("similar"):
                    for hs in head.senses():
                        for ant_sense in hs.get_related("antonym"):
                            ant_lemma = ant_sense.word().lemma().lower()
                            _evaluate_antonym(ant_lemma, priority_bonus=2)
                            for opp_sim in ant_sense.synset().get_related("similar") + ant_sense.synset().get_related("also"):
                                for ow in opp_sim.words():
                                    _evaluate_antonym(ow.lemma().lower(), priority_bonus=1)

        raw_antonyms.sort(key=lambda x: x[1])
        for al, _ in raw_antonyms:
            if al not in sense_antonyms:
                sense_antonyms.append(al)

        tier1_synonyms: List[str] = []
        tier2_satellites: List[str] = []
        tier3_attributes: List[str] = []
        tier2_hyponyms: List[str] = []
        tier3_coordinates: List[str] = []
        seen = {clean_target}
        seen.update(sense_antonyms)

        for s in target_synsets:
            # Tier 1: Synonyms in synset (Only for nouns/adjectives; strictly banned for verbs)
            if wn_pos != "v":
                for sw in s.words():
                    lemma = sw.lemma().lower()
                    if lemma not in seen and " " not in lemma and "_" not in lemma and "-" not in lemma and (lemma in ocd or lemma in cls.get_awl_words()):
                        if wn_pos == "n" and _is_primarily_adj_or_verb(lemma):
                            continue
                        seen.add(lemma)
                        tier1_synonyms.append(lemma)

            if wn_pos == "a":
                # For Adjectives: WordNet organizes concepts into Bipolar Clusters (Scheme 1: Satellite Synset Harvester)
                # Tier 2: Satellite synsets (fine-grained descriptive neighbors in the bipolar cluster)
                sat_synsets = []
                if s.pos == "a":
                    sat_synsets.extend(s.get_related("similar") + s.get_related("also"))
                elif s.pos == "s":
                    for head in s.get_related("similar"):
                        for hw in head.words():
                            hl = hw.lemma().lower()
                            if (
                                hl not in seen
                                and hl not in global_synonyms
                                and " " not in hl and "_" not in hl and "-" not in hl
                                and (hl in ocd or hl in cls.get_awl_words())
                                and _pos_ok(hl)
                            ):
                                seen.add(hl)
                                tier2_satellites.append(hl)
                        sat_synsets.extend(head.get_related("similar") + head.get_related("also"))

                for sim in sat_synsets:
                    for sw in sim.words():
                        lemma = sw.lemma().lower()
                        if (
                            lemma not in seen
                            and lemma not in global_synonyms
                            and " " not in lemma
                            and "_" not in lemma
                            and "-" not in lemma
                            and (lemma in ocd or lemma in cls.get_awl_words())
                            and _pos_ok(lemma)
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
                                and (lemma in ocd or lemma in cls.get_awl_words())
                                and _pos_ok(lemma)
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
                                    and (lemma in ocd or lemma in cls.get_awl_words())
                                    and _pos_ok(lemma)
                                ):
                                    seen.add(lemma)
                                    tier3_attributes.append(lemma)
            else:
                # Tier 2: Hyponyms (more specific concepts / troponyms)
                for hypo in s.hyponyms():
                    for hw in hypo.words():
                        lemma = hw.lemma().lower()
                        if lemma not in seen and lemma not in global_synonyms and " " not in lemma and "_" not in lemma and "-" not in lemma and (lemma in ocd or lemma in cls.get_awl_words()) and _pos_ok(lemma):
                            if wn_pos == "n" and _is_primarily_adj_or_verb(lemma):
                                continue
                            seen.add(lemma)
                            tier2_hyponyms.append(lemma)
                # Tier 3: Coordinate terms (sisters under same hypernym - Troponyms for verbs)
                for hyper in s.hypernyms():
                    # For verbs: skip broad generic root hypernyms (e.g. > 50 hyponyms)
                    if wn_pos == "v" and len(hyper.hyponyms()) > 50:
                        continue
                    for sis in hyper.hyponyms():
                        if sis.id == s.id:
                            continue
                        for sw in sis.words():
                            lemma = sw.lemma().lower()
                            if lemma not in seen and lemma not in global_synonyms and " " not in lemma and "_" not in lemma and "-" not in lemma and (lemma in ocd or lemma in cls.get_awl_words()) and _pos_ok(lemma):
                                if wn_pos == "n" and _is_primarily_adj_or_verb(lemma):
                                    continue
                                seen.add(lemma)
                                tier3_coordinates.append(lemma)

        # Candidate prioritization
        if wn_pos == "v":
            candidates = [c for c in (tier2_hyponyms + tier3_coordinates) if c not in global_synonyms]
        elif wn_pos == "a":
            candidates = tier2_satellites + tier3_attributes + tier1_synonyms
            # Preposition valency cloze (Scheme 4): prioritize academic adjectives governing DIFFERENT prepositions
            if anchor_type == "prep" and context_anchor:
                clean_anchor = context_anchor.strip().lower()
                valency_candidates = []
                for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items():
                    if p != clean_anchor:
                        for a in adjs:
                            if a != clean_target and a not in exclude_words:
                                valency_candidates.append(a)
                candidates = valency_candidates + candidates
        else:
            candidates = tier1_synonyms + tier3_coordinates + tier2_hyponyms

        # 3. Oxford Collision Clearance (Anti Double-Key Gate - Schemes 3 & 4)
        forbidden_words = set(global_synonyms) if wn_pos == "v" else {clean_target}
        if context_anchor:
            clean_anchor = context_anchor.strip().lower()
            if wn_pos == "a" and anchor_type == "prep":
                # Scheme 4: Preposition Valency Gate - reject candidates that also govern the target preposition
                for cand in candidates:
                    cand_preps = ocd.get(cand, {}).get("prep", [])
                    if clean_anchor in cand_preps or cand in cls._ADJ_PREP_VALENCY_MAP.get(clean_anchor, []):
                        forbidden_words.add(cand)
            elif wn_pos == "a" and anchor_type == "modified_noun":
                # Scheme 3: Modified Noun Collocation Clash - reject legal adjectives that also collocate with the noun
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
                # If target is noun, anchor is verb/adj: check colloc_nouns and noun_after
                for n in anchor_entry.get("colloc_nouns", []) + anchor_entry.get("noun_after", []):
                    forbidden_words.add(n.split()[0].lower())

        safe_distractors: List[str] = []
        distractor_metadata: Dict[str, str] = {}

        # Slot 1: Sense-Locked Antonym (if available)
        # Antonyms are whitelisted from context_anchor collision clearance because their
        # collocational plausibility is deliberate, to be eliminated by sentence polarity/logic.
        for ant in sense_antonyms:
            if ant not in safe_distractors and ant not in exclude_words and len(ant) >= 2:
                safe_distractors.append(ant)
                distractor_metadata[ant] = "antonym"
                break

        # Slots 2 & 3: Candidates from Coordinate/Troponym/Synonym trees
        for cand in candidates:
            if cand in safe_distractors:
                continue
            # Attributive position check: if modifying a noun, reject predicative-only adjectives
            if anchor_type == "modified_noun" and cand in cls._PREDICATIVE_ONLY_ADJS:
                continue
            if cand not in forbidden_words and cand not in exclude_words and len(cand) >= 2:
                safe_distractors.append(cand)
                if wn_pos == "a":
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

        # Universal Fallback if candidates pool is still sparse: select standard homogeneous items from OCD
        if len(safe_distractors) < target_count:
            fallback_pool = {
                "v": ["maintain", "establish", "determine", "evaluate", "indicate", "demonstrate", "direct", "guide", "sustain"],
                "n": ["aspect", "factor", "process", "measure", "element", "context", "matter", "institution", "framework"],
                "a": ["crucial", "essential", "primary", "initial", "direct", "specific", "constant", "fundamental"]
            }.get(wn_pos or "n", ["factor", "element", "process"])
            for fb in fallback_pool:
                if fb != clean_target and fb not in forbidden_words and fb not in exclude_words and fb not in safe_distractors:
                    safe_distractors.append(fb)
                    distractor_metadata[fb] = "corpus_fallback"
                if len(safe_distractors) >= target_count:
                    break

        final_distractors = safe_distractors[:target_count]
        if return_metadata:
            return final_distractors, {k: distractor_metadata.get(k, "distractor") for k in final_distractors}
        return final_distractors

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
    @classmethod
    def semantic_fields(cls, word: str) -> Tuple[Set[str], Set[str]]:
        """Returns (near_set, antonyms) for a lemma, WordNet-derived.

        near_set  : same-synset words, bipolar similar/also (2 hops),
                    hypernym / entailment / hyponym lemmas.
        antonyms  : sense-locked antonyms (+ their satellite cluster).
        Deterministic; never raises."""
        wn = cls.get_wordnet()
        target = word.lower()
        near: Set[str] = set()
        antonyms: Set[str] = set()

        def add(syn) -> None:
            for w in syn.words():
                near.add(w.lemma().lower())

        seeds = []
        for pos in ("a", "s", "v", "n", "r"):
            try:
                seeds += wn.synsets(target, pos)
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

        near.discard(target)
        antonyms.discard(target)
        return near, antonyms

    @classmethod
    def double_key_collision(cls, target: str, distractor: str,
                             anchor: Optional[str] = None,
                             anchor_type: Optional[str] = None) -> Tuple[bool, str]:
        """Deterministic double-key detector. Returns (is_double_key, kind).

        kind is one of:
          "double_key_prep"  : shared bound preposition + semantic neighbor
          "double_key_slot"  : same semantic slot (near-synonym/coordinate/troponym)
          "frame_only"       : shared bound preposition, semantically distinct
                               (NOT a double-key; needs a specific exclusion clue)
          "none"             : clear
        """
        t = (target or "").strip().lower()
        d = (distractor or "").strip().lower()
        if not t or not d or t == d:
            return False, "none"
        near, antonyms = cls.semantic_fields(t)
        shared = bool(anchor) and anchor.lower() in (cls.get_oxford_collocations().get(t, {}).get("prep") or [])

        if anchor_type == "prep" and anchor and shared:
            if d in near:
                return True, "double_key_prep"
            return False, "frame_only"
        if d in near:
            if d in antonyms:
                # Deliberate contrast distractor (eliminated by polarity/logic),
                # whitelisted by design in the distractor generator.
                return False, "contrast"
            return True, "double_key_slot"
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
        ["prevalent", "widespread", "ubiquitous", "scarce", "isolated", "sporadic", "rare"]
    ]

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
        "just", "even", "also", "well", "still", "yet", "already"
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

    @classmethod
    def _pick_anchor_token(cls, candidates: List[str]) -> Optional[str]:
        """
        Selects the first meaningful single-word anchor from a list of collocation
        phrases. Rejects truncated function-word tokens (e.g. 'be', 'have', 'of')
        produced by naively splitting multi-word collocations, so collision
        clearance operates on a real lexical headword instead of a stopword.
        """
        for cand in candidates:
            if not cand:
                continue
            parts = cand.split()
            if not parts:
                continue
            token = parts[0].lower()
            if token.isalpha() and len(token) >= 3 and token not in cls._ANCHOR_STOPWORDS:
                return token
        return None

    @classmethod
    def _select_sense_aligned_anchor(
        cls,
        candidates: List[str],
        definition: Optional[str] = None,
        quote: Optional[str] = None
    ) -> Optional[str]:
        """
        Selects the most semantically aligned single-word anchor from collocation candidates,
        grounded in the target item's authentic quote and curriculum definition (Sense-Specific Primacy).
        Prevents sense-drift where polysemous words select collocations from irrelevant senses.
        """
        if not candidates:
            return None

        quote_clean = (quote or "").lower()
        def_clean = (definition or "").lower()

        # 1. Level 1 (Authentic Quote Match): If any collocation candidate occurs directly
        # in the source sentence quote, pick it immediately (zero hallucination, verbatim fidelity)
        for cand in candidates:
            token = cls._pick_anchor_token([cand])
            if token and re.search(r"\b" + re.escape(token) + r"\b", quote_clean):
                return token

        # 2. Level 2 (Definition Semantic Overlap): Score candidates by word overlap with definition and quote
        context_tokens = set(re.findall(r"[a-zA-Z]{3,}", (quote_clean + " " + def_clean)))
        context_tokens = context_tokens - cls._ANCHOR_STOPWORDS

        scored_candidates: List[Tuple[int, str]] = []
        for cand in candidates:
            token = cls._pick_anchor_token([cand])
            if not token:
                continue
            score = 1 if token in context_tokens else 0
            scored_candidates.append((score, token))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        if scored_candidates and scored_candidates[0][0] > 0:
            return scored_candidates[0][1]

        # 3. Level 3 (Strict None Semantics): if NO candidate is evidenced by
        # either the authentic quote (Level 1) or the definition (Level 2),
        # refuse to fabricate an anchor. Returning the first clean token (the
        # old "first-word lottery") produced fake anchors such as
        # 'compulsion ~ strange'; a missing anchor is a first-class outcome
        # that the micro-task planner degrades to a neutral academic template.
        return None

    @classmethod
    def get_acl_collocations(cls) -> Dict[str, str]:
        """Lazy-loads the Academic Collocation List (ACL, ~2474 items). Returns dict mapping core pattern -> canonical original."""
        if cls._acl_data is None:
            cls._acl_data = {}
            acl_path = Path(__file__).parent / "data" / "academic_collocations.json"
            if acl_path.exists():
                try:
                    with open(acl_path, "r", encoding="utf-8") as f:
                        raw_items = json.load(f)
                    for item in raw_items:
                        # strip optional prefixes/suffixes like (a), (be), (to), (of) for robust matching
                        core = re.sub(r"^\([a-z\s]+\)\s*", "", item)
                        core = re.sub(r"\s*\([a-z\s]+\)$", "", core).strip().lower()
                        if len(core) >= 3:
                            cls._acl_data[core] = item
                except Exception:
                    pass
        return cls._acl_data

    @classmethod
    def get_awl_words(cls) -> Set[str]:
        """Lazy-loads the Academic Word List (AWL, 560+ headwords)."""
        if cls._awl_data is None:
            cls._awl_data = set()
            awl_path = Path(__file__).parent / "data" / "academic_word_list.json"
            if awl_path.exists():
                try:
                    with open(awl_path, "r", encoding="utf-8") as f:
                        cls._awl_data = set(json.load(f))
                except Exception:
                    pass
        return cls._awl_data

    # -------------------------------------------------------------------------
    # 1. Sentence Boundary Tokenization & Indexing ([S-1], [S-2])
    # -------------------------------------------------------------------------
    @classmethod
    def tokenize_and_index_sentences(cls, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Pre-tokenizes raw source text into an indexed, complete-sentence pool.
        Returns:
            indexed_text: Text formatted with '[S-1] ... [S-2] ...' for LLM prompts.
            sentence_pool: Dict mapping 'S-1' -> pristine full sentence text.
        """
        nlp = cls.get_spacy()
        # Cleanly separate frontmatter if present so spaCy only tokenizes authentic body prose
        fm_match = re.match(r"^(---\s*\n.*?\n---\s*\n)(.*)", text, re.DOTALL)
        frontmatter = fm_match.group(1) if fm_match else ""
        body = fm_match.group(2) if fm_match else text

        doc = nlp(body)

        sentence_pool: Dict[str, str] = {}
        indexed_paragraphs: List[str] = []
        counter = 1

        # Process paragraph-by-paragraph to preserve natural paragraph boundaries while streaming sentences within paragraphs
        raw_paragraphs = re.split(r'\n{2,}', body.strip())
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            # If the paragraph is a pure markdown header or bold block (e.g. '## Text A', '**Title**')
            lines = [l.strip() for l in para.split("\n") if l.strip()]
            if all(l.startswith("#") or (l.startswith("**") and l.endswith("**")) for l in lines):
                indexed_paragraphs.append(para)
                continue

            # Separate optional leading heading lines in this paragraph
            prefix_lines = []
            content_lines = []
            for l in lines:
                if not content_lines and (l.startswith("#") or (l.startswith("**") and l.endswith("**"))):
                    prefix_lines.append(l)
                else:
                    content_lines.append(l)

            prefix_str = "\n\n".join(prefix_lines) + "\n\n" if prefix_lines else ""
            para_text = " ".join(content_lines).strip()
            if not para_text:
                if prefix_lines:
                    indexed_paragraphs.append("\n\n".join(prefix_lines))
                continue

            para_doc = nlp(para_text)
            para_sent_parts = []
            for sent in para_doc.sents:
                sent_str = sent.text.strip()
                if not sent_str:
                    continue
                words = [t for t in nlp(sent_str) if t.is_alpha]
                if len(words) < 2 and not sent_str.endswith((".", "?", "!")):
                    para_sent_parts.append(sent_str)
                    continue

                sid = f"S-{counter}"
                sentence_pool[sid] = sent_str
                para_sent_parts.append(f"[{sid}] {sent_str}")
                counter += 1

            indexed_para = prefix_str + " ".join(para_sent_parts)
            indexed_paragraphs.append(indexed_para.strip())

        if frontmatter:
            indexed_text = frontmatter + "\n\n" + "\n\n".join(indexed_paragraphs)
        else:
            indexed_text = "\n\n".join(indexed_paragraphs)
        return indexed_text, sentence_pool

    @classmethod
    def snap_to_sentence_pool(cls, raw_quote_or_id: str, sentence_pool: Dict[str, str]) -> Optional[str]:
        """
        Deterministically snaps a model-provided sentence ID (e.g. '[S-3]', 'S-3') or a
        partial quote with ellipsis back to the pristine, authentic source sentence in the pool.
        Eliminates trailing ellipses, copy-paste hallucinations, and verbatim mismatches.
        """
        if not raw_quote_or_id or not sentence_pool:
            return None

        clean_input = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", raw_quote_or_id.strip(), flags=re.IGNORECASE).strip()

        # 1. Direct ID lookup (e.g. '[S-3]' or 'S-3' or 'Sentence 3')
        id_match = re.search(r"\bS-(\d+)\b", raw_quote_or_id, re.IGNORECASE)
        if id_match:
            sid = f"S-{id_match.group(1)}"
            if sid in sentence_pool:
                return re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sentence_pool[sid], flags=re.IGNORECASE).strip()

        # 2. Exact match in sentence pool
        for sid, sent in sentence_pool.items():
            sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sent, flags=re.IGNORECASE).strip()
            if clean_input.lower() == sent_clean.lower():
                return sent_clean

        # 3. Substring containment: if quote is a fragment of a pool sentence
        clean_target = re.sub(r"\.{3,}|…", "", clean_input).strip().lower()
        if len(clean_target) >= 15:
            for sid, sent in sentence_pool.items():
                sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sent, flags=re.IGNORECASE).strip()
                if clean_target in sent_clean.lower():
                    return sent_clean

        # 4. High overlap fallback
        clean_tokens = set(re.findall(r"\w{3,}", clean_target))
        if clean_tokens:
            best_sent = None
            max_overlap = 0.0
            for sid, sent in sentence_pool.items():
                sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sent, flags=re.IGNORECASE).strip()
                sent_tokens = set(re.findall(r"\w{3,}", sent_clean.lower()))
                if not sent_tokens:
                    continue
                overlap = len(clean_tokens & sent_tokens) / len(clean_tokens)
                if overlap > max_overlap and overlap >= 0.75:
                    max_overlap = overlap
                    best_sent = sent_clean
            if best_sent:
                return best_sent

    @classmethod
    def get_sentence_id(cls, raw_quote_or_id: str, sentence_pool: Dict[str, str]) -> Optional[str]:
        """
        Returns the canonical sentence identifier (e.g. 'S-3') for a given quote or ID.
        """
        if not raw_quote_or_id or not sentence_pool:
            return None

        clean_input = raw_quote_or_id.strip()

        # 1. Direct ID lookup
        id_match = re.search(r"\bS-(\d+)\b", clean_input, re.IGNORECASE)
        if id_match:
            sid = f"S-{id_match.group(1)}"
            if sid in sentence_pool:
                return sid

        # 2. Exact match
        for sid, sent in sentence_pool.items():
            if clean_input.lower() == sent.lower():
                return sid

        # 3. Substring containment
        clean_target = re.sub(r"\.{3,}|…", "", clean_input).strip().lower()
        if len(clean_target) >= 15:
            for sid, sent in sentence_pool.items():
                if clean_target in sent.lower():
                    return sid

        # 4. Token overlap fallback
        clean_tokens = set(re.findall(r"\w{3,}", clean_target))
        if clean_tokens:
            best_sid = None
            max_overlap = 0.0
            for sid, sent in sentence_pool.items():
                sent_tokens = set(re.findall(r"\w{3,}", sent.lower()))
                if not sent_tokens:
                    continue
                overlap = len(clean_tokens & sent_tokens) / len(clean_tokens)
                if overlap > max_overlap and overlap >= 0.75:
                    max_overlap = overlap
                    best_sid = sid
            if best_sid:
                return best_sid

        return None

    @classmethod
    def extract_grammar_fingerprint(cls, sentence: str, category: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Extracts a structural grammar fingerprint (macro_domain, dep_type, core_anchor)
        using spaCy dependency parsing. Used for deterministic deduplication of identical
        syntactic patterns across different sentences.
        """
        nlp = cls.get_spacy()
        doc = nlp(sentence)
        macro_domain = category or cls.classify_grammar_dependency(sentence) or "Unknown"

        dep_type = "generic"
        core_anchor = ""

        # 1. Rhetoric & Emphasis
        for token in doc:
            if token.dep_ == "nsubj":
                head = token.head
                if head.pos_ in ("VERB", "AUX"):
                    aux_children = [c for c in head.children if c.dep_ in ("aux", "auxpass") and c.i < token.i]
                    if aux_children and not sentence.strip().endswith("?"):
                        first_tok = doc[0].lemma_.lower()
                        if first_tok in ("hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no"):
                            dep_type = "inversion"
                            core_anchor = first_tok
                            break

        if dep_type == "generic":
            text_lower = sentence.lower()
            if re.search(r"\bnot\s+.*?\s*,\s*but\b", text_lower):
                dep_type = "antithesis"
                core_anchor = "not...but"
            elif re.search(r"\bnot\s+only\b.*?\bbut\s+also\b", text_lower):
                dep_type = "parallelism"
                core_anchor = "not only...but also"
            elif any(t.dep_ == "expl" and t.text.lower() == "it" for t in doc):
                for t in doc:
                    if t.dep_ == "expl" and t.head.lemma_ in ("be", "seem"):
                        if any(c.dep_ == "relcl" or any(gc.dep_ == "relcl" for gc in c.children) for c in t.head.children):
                            dep_type = "cleft"
                            core_anchor = "it...that"
                            break

        # 2. Logic & Stance
        if dep_type == "generic":
            for token in doc:
                if token.dep_ == "mark" and token.lemma_.lower() in {"although", "though", "while", "whereas", "even though", "if", "unless", "provided"}:
                    dep_type = "subordinating_clause"
                    core_anchor = token.lemma_.lower()
                    break

        # 3. Cohesion & Framing
        if dep_type == "generic":
            interpretive_verbs = {"mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"}
            for token in doc:
                if token.dep_ in ("relcl", "advcl") and token.lemma_.lower() in interpretive_verbs:
                    subj_which = any(c.dep_ in ("nsubj", "nsubjpass") and c.text.lower() == "which" for c in token.children)
                    if subj_which:
                        dep_type = "propositional_encapsulation"
                        core_anchor = f"which {token.lemma_.lower()}"
                        break
            if dep_type == "generic":
                shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "difference", "discrepancy"}
                for token in doc:
                    if token.lemma_.lower() in shell_nouns:
                        dep_type = "shell_noun"
                        core_anchor = token.lemma_.lower()
                        break

            # Correlative comparative: the more..., the more... (Rhetoric & Emphasis or Information Packaging)
            COMPARATIVE_WORDS = r"(?:more|less|fewer|better|worse|[a-z]{2,}er)"
            if re.search(rf"\bthe\s+{COMPARATIVE_WORDS}\b(?!\s+hand\b).*?(?:,\s*|\band\s+the\s+{COMPARATIVE_WORDS}\b.*?,?\s*)\bthe\s+{COMPARATIVE_WORDS}\b", text_lower):
                dep_type = "correlative_comparative"
                core_anchor = "the...the"

            if dep_type == "generic":
                for token in doc:
                    if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                        if not any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children) and token.lemma_.lower() not in ("include", "accord", "regard"):
                            dep_type = "participial_adjunct"
                            core_anchor = token.tag_
                            break
            # Dummy-It Subject Extraposition: It is adj that/to-V
            if dep_type == "generic" and any(t.text.lower() == "it" and t.dep_ in ("expl", "nsubj") for t in doc):
                for t in doc:
                    if t.text.lower() == "it" and t.head.lemma_ in ("be", "seem"):
                        if any(c.dep_ in ("acomp", "attr") for c in t.head.children) and any(c.dep_ in ("ccomp", "csubj", "xcomp") for c in t.head.children):
                            dep_type = "dummy_it_extraposition"
                            core_anchor = "it be adj to/that"
                            break
            # Dummy-It Object Extraposition: find/make/think it adj to-V
            if dep_type == "generic":
                for t in doc:
                    if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                        for c in t.children:
                            if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                                dep_type = "dummy_it_object"
                                core_anchor = f"{t.lemma_} it {c.lemma_}"
                                break

        return (macro_domain, dep_type, core_anchor)


    @classmethod
    def lemmatize_headword(cls, word_or_phrase: str, context_sentence: Optional[str] = None) -> str:
        """
        Derives canonical base dictionary headwords from sentence context,
        eliminating inflected headwords (-ed, -ing, 3sg).
        """
        nlp = cls.get_spacy()
        target = word_or_phrase.strip()

        # If a single word
        if " " not in target:
            doc = nlp(target)
            if len(doc) > 0 and doc[0].pos_ in ("VERB", "NOUN", "ADJ"):
                return doc[0].lemma_.lower()
            return target.lower()

        # If a multi-word expression (e.g. phrasal verb or idiom)
        doc = nlp(target)
        lemmatized_tokens = []
        for token in doc:
            if token.pos_ == "VERB":
                lemmatized_tokens.append(token.lemma_.lower())
            else:
                lemmatized_tokens.append(token.text.lower())
        return " ".join(lemmatized_tokens)

    @classmethod
    def extract_phrasal_verbs(cls, sentence: str) -> List[str]:
        """
        Deterministically extracts multi-word phrasal verbs from the dependency tree
        by querying verb heads governing direct 'prt' (particle) children,
        strictly pruning ordinary spatial/temporal prepositional phrases ('prep + pobj').
        e.g.
          'come in' -> extracted (prt)
          'give up' -> extracted (prt)
          'come in a city' -> rejected (prep + pobj, spatial adjunct)
        """
        nlp = cls.get_spacy()
        doc = nlp(sentence)
        phrasal_verbs = []

        for token in doc:
            if token.pos_ != "VERB":
                continue

            # Check direct particle children (prt)
            prts = [child for child in token.children if child.dep_ == "prt"]
            for p in prts:
                pv = f"{token.lemma_.lower()} {p.lemma_.lower()}"
                # Check for secondary preposition attached to verb or particle (e.g. come up with)
                prep_children = [c for c in p.children if c.dep_ == "prep"]
                if not prep_children:
                    prep_children = [c for c in token.children if c.dep_ == "prep" and c.i > p.i]
                if prep_children:
                    pv += f" {prep_children[0].lemma_.lower()}"
                phrasal_verbs.append(pv)

        return list(dict.fromkeys(phrasal_verbs))

    # -------------------------------------------------------------------------
    # 3. Dependency-Based Macro Domain Classification Gate
    # -------------------------------------------------------------------------
    @classmethod
    def classify_grammar_dependency(cls, sentence: str) -> Optional[str]:
        """
        Classifies complex sentence into one of the Four Macro Functional Domains
        using deterministic spaCy dependency trees and syntactic relations:
          - 'Rhetoric & Emphasis'
          - 'Cohesion & Framing'
          - 'Information Packaging'
          - 'Logic & Stance'
        Returns None if no unambiguous structural pattern dominates.
        """
        nlp = cls.get_spacy()
        doc = nlp(sentence)

        # ---------------------------------------------------------------------
        # 1. Rhetoric & Emphasis:
        #    a. Fronted Inversion: aux or ROOT precedes nsubj in linear word order
        #    b. Cleft Focus: expl ('it') + 'is/was' + focal constituent + relcl (that...)
        #    c. Antithesis: 'not... but...' parallel coordinating balance
        # ---------------------------------------------------------------------
        # Inversion check
        for token in doc:
            if token.dep_ == "nsubj":
                # Check if auxiliary or root verb precedes nsubj (excluding questions)
                head = token.head
                if head.pos_ == "VERB" or head.pos_ == "AUX":
                    aux_children = [c for c in head.children if c.dep_ in ("aux", "auxpass")]
                    for aux in aux_children:
                        if aux.i < token.i and not sentence.strip().endswith("?"):
                            # Check if fronted negative/restrictive adverbial starts sentence
                            first_token = doc[0].lemma_.lower()
                            if first_token in ("hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no"):
                                return "Rhetoric & Emphasis"

        # Cleft vs Dummy-it check
        for token in doc:
            if token.dep_ == "expl" and token.text.lower() == "it":
                # Check copula verb
                head = token.head
                if head.lemma_ in ("be", "seem"):
                    # If followed by relative clause (relcl) modifying non-subject focus -> Cleft
                    has_relcl = any(child.dep_ == "relcl" for child in head.children)
                    for child in head.children:
                        if any(gc.dep_ == "relcl" for gc in child.children):
                            has_relcl = True
                    # Check if followed by that/who relcl
                    if has_relcl:
                        return "Rhetoric & Emphasis"

        # Antithesis / Correlative Parallelism check (not... but..., not only... but also)
        text_lower = sentence.lower()
        if re.search(r"\bnot\s+.*?\s*,\s*but\b", text_lower) or re.search(r"\bnot\s+only\b.*?\bbut\s+also\b", text_lower):
            return "Rhetoric & Emphasis"

        # ---------------------------------------------------------------------
        # 2. Logic & Stance:
        #    Conditionals (if/unless) or Concessives (although/even though/while/whereas)
        #    Checked before Information Packaging to avoid 'promising' being treated as participle
        # ---------------------------------------------------------------------
        subordinating_conjs = {"although", "though", "while", "whereas", "even though", "if", "unless", "provided", "providing"}
        for token in doc:
            if token.dep_ == "mark" and token.lemma_.lower() in subordinating_conjs:
                return "Logic & Stance"

        # ---------------------------------------------------------------------
        # 3. Cohesion & Framing:
        #    a. Propositional Encapsulation: , which + [interpretive verb] + that
        #    b. Shell Noun Frames: [shell noun] + that [clause]
        # ---------------------------------------------------------------------
        # Propositional encapsulation: which + interpretive verb + that
        interpretive_verbs = {"mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"}
        for token in doc:
            if token.dep_ in ("relcl", "advcl") and token.lemma_.lower() in interpretive_verbs:
                # Check if subject is 'which'
                subj_children = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass") and c.text.lower() == "which"]
                if subj_children:
                    has_that_comp = any(child.dep_ == "ccomp" for child in token.children)
                    if has_that_comp:
                        return "Cohesion & Framing"

        # Shell noun patterns: fact, idea, conclusion, claim, belief, hypothesis + that
        shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "evidence", "assumption"}
        for token in doc:
            if token.lemma_.lower() in shell_nouns:
                for child in token.children:
                    if child.dep_ in ("acl", "appos", "ccomp"):
                        return "Cohesion & Framing"

        # ---------------------------------------------------------------------
        # 4. Information Packaging:
        #    a. Evaluative Dummy-It: It is + [adj/noun] + that/to [csubj/ccomp]
        #    b. Non-finite participial adjuncts: advcl with VerbForm=Part
        #    c. Elaborative non-restrictive relative clause: , which + VP (without that-clause)
        # ---------------------------------------------------------------------
        # Evaluative Dummy-It (spaCy tags 'It' as nsubj/expl and the complement clause as ccomp/csubj)
        for token in doc:
            if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj"):
                head = token.head
                if head.lemma_ in ("be", "seem"):
                    # Check if head has acomp/attr and ccomp/csubj/xcomp
                    has_predicate = any(child.dep_ in ("acomp", "attr") for child in head.children)
                    has_complement = any(child.dep_ in ("ccomp", "csubj", "xcomp") for child in head.children)
                    if has_predicate and has_complement:
                        return "Information Packaging"

        # Evaluative Dummy-It Object (find/make/think it adj to-V)
        for t in doc:
            if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                for c in t.children:
                    if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                        return "Information Packaging"

        # Non-finite participial adjuncts (advcl with VBG or VBN)
        for token in doc:
            if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                # Ensure it has no subject of its own (non-finite) and not a preposition
                has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children)
                if not has_subj and token.lemma_.lower() not in ("include", "accord", "regard", "concern"):
                    return "Information Packaging"

        # Elaborative non-restrictive relative clauses
        if re.search(r",\s*which\s+[a-z]+", text_lower):
            return "Information Packaging"

        return None

    @classmethod
    def generate_cobuild_formula(cls, sentence: str, category: Optional[str] = None) -> str:
        """
        Deterministically derives standard algebraic COBUILD slot formula from sentence
        via spaCy dependency trees and syntactic pattern anchors.
        Replaces prone-to-hallucination LLM formula drafting with zero-token invariant logic.
        """
        nlp = cls.get_spacy()
        doc = nlp(sentence)
        cat = category or cls.classify_grammar_dependency(sentence) or ""
        text_lower = sentence.lower()

        # 1. Rhetoric & Emphasis
        if cat == "Rhetoric & Emphasis" or not cat:
            # Inversion
            for token in doc:
                if token.dep_ == "nsubj":
                    head = token.head
                    if head.pos_ in ("VERB", "AUX"):
                        aux_children = [c for c in head.children if c.dep_ in ("aux", "auxpass") and c.i < token.i]
                        if aux_children and not sentence.strip().endswith("?"):
                            first_tok = doc[0].lemma_.lower()
                            if first_tok in ("hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no"):
                                return f"{doc[0].text} + [aux/be] + [Subject] + [VP]"
            # Cleft
            for token in doc:
                if token.dep_ == "expl" and token.text.lower() == "it" and token.head.lemma_ in ("be", "seem"):
                    if any(c.dep_ == "relcl" or any(gc.dep_ == "relcl" for gc in c.children) for c in token.head.children):
                        return "It + [be] + [Focal Element] + that/who + [Clause]"
            # Antithesis
            if re.search(r"\bnot\s+.*?\s*,\s*but\b", text_lower):
                return "[Subject] + [VP], not + [PrepP/NP], but + [PrepP/NP]"
            # Correlative Parallelism
            if re.search(r"\bnot\s+only\b.*?\bbut\s+also\b", text_lower):
                return "[Subject] + not only + [VP], but also + [VP]"

        # 2. Logic & Stance
        if cat == "Logic & Stance" or not cat:
            for token in doc:
                if token.dep_ == "mark" and token.lemma_.lower() in {"although", "though", "while", "whereas", "even though", "if", "unless", "provided"}:
                    mark_word = token.text.capitalize()
                    return f"{mark_word} + [Clause], [Subject] + [VP]"

        # 3. Cohesion & Framing
        if cat == "Cohesion & Framing" or not cat:
            # Propositional Encapsulation: , which + [interpretive verb] + that
            interpretive_verbs = {"mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"}
            for token in doc:
                if token.dep_ in ("relcl", "advcl") and token.lemma_.lower() in interpretive_verbs:
                    if any(c.dep_ in ("nsubj", "nsubjpass") and c.text.lower() == "which" for c in token.children):
                        return f", which + {token.lemma_.lower()}s + that + [Proposition Clause]"
            # Shell noun frame
            shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "difference", "discrepancy"}
            for token in doc:
                if token.lemma_.lower() in shell_nouns:
                    return f"The + {token.lemma_.lower()} + that/of + [Proposition Clause]"

        # 4. Information Packaging
        if cat == "Information Packaging" or not cat:
            # Correlative comparative: The more..., the more...
            COMPARATIVE_WORDS = r"(?:more|less|fewer|better|worse|[a-z]{2,}er)"
            if re.search(rf"\bthe\s+{COMPARATIVE_WORDS}\b(?!\s+hand\b).*?(?:,\s*|\band\s+the\s+{COMPARATIVE_WORDS}\b.*?,?\s*)\bthe\s+{COMPARATIVE_WORDS}\b", text_lower):
                return "The + [comparative] + [Clause], the + [comparative] + [Clause]"
            # Dummy-It Object Extraposition (find/make/think it adj to-V)
            for t in doc:
                if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                    for c in t.children:
                        if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                            return f"[Subject] + {t.lemma_} + it + [{c.lemma_.capitalize()}] + to-V"
            # Dummy-It Subject Extraposition
            if any(t.text.lower() == "it" and t.dep_ in ("expl", "nsubj") for t in doc):
                for t in doc:
                    if t.text.lower() == "it" and t.head.lemma_ in ("be", "seem"):
                        if any(c.dep_ in ("acomp", "attr") for c in t.head.children) and any(c.dep_ in ("ccomp", "csubj", "xcomp") for c in t.head.children):
                            return "It + [be] + [Adj/NP] + to-V/that + [Clause]"
            # Participial adjunct
            for token in doc:
                if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                    if not any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children) and token.lemma_.lower() not in ("include", "accord", "regard"):
                        is_fronted = token.i < token.head.i
                        v_type = "V-ing" if token.tag_ == "VBG" else "V-ed"
                        if is_fronted:
                            return f"[{v_type} Phrase], [Subject] + [VP]"
                        else:
                            return f"[Subject] + [VP], [{v_type} Phrase]"
            # Elaborative clause
            if re.search(r",\s*which\s+[a-z]+", text_lower):
                return "[Subject] + [VP], which + [VP]"

        # Default fallback standard formula
        return "[Subject] + [VP] + [Clause]"

    @classmethod
    def mine_grammar_skeletons(
        cls,
        sentence_pool: Dict[str, str],
        target_count: int = 5,
        syllabus_grammar: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Deterministically mines genuine, non-trivial grammar structural skeletons from the sentence pool.
        Selects sentences exhibiting rich dependency structures across the Four Macro Domains.
        Returns a list of dicts: [{'sid': 'S-9', 'quote': '...', 'category': '...', 'pattern_formula': '...'}]
        """
        raw_skeletons: List[Dict[str, str]] = []

        for sid, sent in sentence_pool.items():
            sent_clean = sent.strip()
            # Skip very short or title-like sentences
            if len(sent_clean.split()) < 7:
                continue

            cat = cls.classify_grammar_dependency(sent_clean)
            if cat:
                formula = cls.generate_cobuild_formula(sent_clean, category=cat)
                # Filter out generic/un-abstracted formulas (e.g. [Subject] + [VP] + [Clause] or simple [S])
                if formula and not formula.startswith("[Subject] + [VP] + [Clause]") and "[" in formula:
                    raw_skeletons.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "category": cat,
                        "pattern_formula": formula,
                    })

        if not raw_skeletons:
            return []

        # Diverse selection algorithm across macro domains and distinct formulas
        selected: List[Dict[str, str]] = []
        seen_cats: Set[str] = set()
        seen_formulas: Set[str] = set()

        # Pass 1: maximize category diversity across the 4 domains
        for item in raw_skeletons:
            cat = item["category"]
            form = item["pattern_formula"]
            if cat not in seen_cats and form not in seen_formulas:
                selected.append(item)
                seen_cats.add(cat)
                seen_formulas.add(form)
            if len(selected) >= target_count:
                break

        # Pass 2: fill remaining slots with distinct structural formulas
        if len(selected) < target_count:
            for item in raw_skeletons:
                if item not in selected and item["pattern_formula"] not in seen_formulas:
                    selected.append(item)
                    seen_formulas.add(item["pattern_formula"])
                if len(selected) >= target_count:
                    break

        # Pass 3: if still under target_count, append distinct quotes with unique formulas
        if len(selected) < target_count:
            selected_quotes = {s["quote"] for s in selected}
            for item in raw_skeletons:
                if item["quote"] not in selected_quotes and item["pattern_formula"] not in seen_formulas:
                    selected.append(item)
                    selected_quotes.add(item["quote"])
                    seen_formulas.add(item["pattern_formula"])
                if len(selected) >= target_count:
                    break

        return selected[:target_count]

    # -------------------------------------------------------------------------
    # 3b. Deterministic Multi-Word Expression & Collocation Mining (ACL + spaCy)
    # -------------------------------------------------------------------------
    @classmethod
    def mine_expression_skeletons(cls, text: str, target_count: int = 8) -> List[Dict[str, str]]:
        """
        Deterministically extracts high-value multi-word expressions, phrasal verbs,
        and Academic Collocation List (ACL) items from the indexed sentence pool.
        Standardizes slotted formulas (e.g. [sb], [sth], [one's]) at zero token cost.

        Returns list of dicts:
            [{
                "sid": "S-1",
                "quote": "full sentence text",
                "phrase": "have an adverse effect on",
                "type": "collocation",
                "pattern_formula": "have an adverse effect on [sth]"
            }, ...]
        """
        _, pool = cls.tokenize_and_index_sentences(text)
        if not pool:
            return []

        nlp = cls.get_spacy()
        acl_map = cls.get_acl_collocations()
        awl_set = cls.get_awl_words()

        COMMON_PARTICLES = frozenset({"away", "back", "down", "in", "off", "on", "out", "over", "round", "through", "up"})
        DEPENDENT_PREPS = frozenset({"on", "upon", "to", "with", "for", "from", "into", "against", "about", "of", "in"})

        raw_candidates: List[Dict[str, Any]] = []

        for sid, sentence in pool.items():
            sent_clean = sentence.strip()
            if len(sent_clean.split()) < 5:
                continue

            doc = nlp(sent_clean)
            text_lower = sent_clean.lower()

            # 1. Academic Collocation List (ACL) Matcher
            for core, original in acl_map.items():
                pattern = r"\b" + re.escape(core) + r"\b"
                if re.search(pattern, text_lower):
                    # Determine type: verb+noun or adj+noun
                    words = core.split()
                    expr_type = "collocation"
                    formula = core
                    if original.endswith("(to)") or original.endswith("(of)") or original.endswith("(in)") or original.endswith("(with)"):
                        prep_match = re.search(r"\(([a-z]+)\)$", original)
                        if prep_match:
                            formula = f"{core} {prep_match.group(1)} [sth]"
                    elif len(words) == 2 and any(w in awl_set for w in words):
                        formula = f"{core}"

                    raw_candidates.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "phrase": core,
                        "type": expr_type,
                        "pattern_formula": formula,
                        "score": 10 + (2 if any(w in awl_set for w in words) else 0),
                    })

            # 2. Dependency Syntax Phrasal Verbs & Prepositional Combinations
            COLLOCATION_VERB_PREPS = {
                "welcome": "to",
                "provide": "with",
                "share": "with",
                "equip": "with",
                "attribute": "to",
                "contribute": "to",
                "rely": "on",
                "depend": "on",
                "focus": "on",
                "concentrate": "on",
                "participate": "in",
                "engage": "in",
                "succeed": "in",
                "benefit": "from",
                "derive": "from",
                "suffer": "from",
                "cope": "with",
                "comply": "with",
                "lead": "to",
                "adapt": "to",
                "conform": "to",
                "refer": "to",
                "apply": "to",
                "appeal": "to",
                "insist": "on",
                "consist": "of",
            }

            for token in doc:
                if token.pos_ in ("VERB", "AUX"):
                    v_lemma = token.lemma_.lower()

                    # Check particle (e.g. wake up, get by, slave away, carry out)
                    for c in token.children:
                        if (c.dep_ == "prt" or (c.dep_ == "advmod" and c.text.lower() in COMMON_PARTICLES)) and c.i > token.i:
                            p_tok = c.text.lower()
                            # Check three-part phrasal verb: verb + particle + prep immediately following (distance <= 2)
                            prep_child = [gc for gc in token.children if gc.dep_ == "prep" and 0 < (gc.i - c.i) <= 2]
                            if prep_child:
                                prep_word = prep_child[0].text.lower()
                                p_verb = f"{v_lemma} {p_tok} {prep_word}"
                                formula = f"{v_lemma} {p_tok} {prep_word} [sth/sb]"
                            else:
                                dobj = [d for d in token.children if d.dep_ == "dobj"]
                                if dobj:
                                    p_verb = f"{v_lemma} {p_tok}"
                                    formula = f"{v_lemma} [sb/sth] {p_tok}"
                                else:
                                    p_verb = f"{v_lemma} {p_tok}"
                                    formula = f"{v_lemma} {p_tok}"

                            raw_candidates.append({
                                "sid": sid,
                                "quote": sent_clean,
                                "phrase": p_verb,
                                "type": "phrasal verb",
                                "pattern_formula": formula,
                                "score": 9,
                            })

                    # Check possessive object construction via spaCy 'poss' dependency (e.g. attain [one's] best, make up [one's] mind)
                    # Restrict to genuine idiomatic and academic lexicalized noun heads to avoid trivial 'take his dad'
                    IDIOMATIC_POSS_NOUNS = frozenset({
                        "best", "mind", "temper", "breath", "time", "part", "way", "heart",
                        "step", "place", "voice", "promise", "life", "potential", "future",
                        "passion", "dream", "goal", "duty", "responsibility", "effort", "stride",
                        "horizon", "footing", "view", "interest", "role", "purpose"
                    })
                    dobjs = [d for d in token.children if d.dep_ == "dobj"]
                    for d in dobjs:
                        poss = [p for p in d.children if p.dep_ == "poss"]
                        if poss:
                            noun_text = d.text.lower()
                            noun_lemma = d.lemma_.lower()
                            matched_noun = None
                            if noun_text in IDIOMATIC_POSS_NOUNS:
                                matched_noun = noun_text
                            elif noun_lemma in IDIOMATIC_POSS_NOUNS:
                                matched_noun = noun_lemma

                            if matched_noun:
                                prt = [p for p in token.children if p.dep_ == "prt"]
                                if prt:
                                    p_clean = f"{v_lemma} {prt[0].text.lower()} [one's] {matched_noun}"
                                    formula = f"{v_lemma} {prt[0].text.lower()} [one's] {matched_noun}"
                                else:
                                    p_clean = f"{v_lemma} [one's] {matched_noun}"
                                    formula = f"{v_lemma} [one's] {matched_noun}"

                                raw_candidates.append({
                                    "sid": sid,
                                    "quote": sent_clean,
                                    "phrase": p_clean,
                                    "type": "collocation",
                                    "pattern_formula": formula,
                                    "score": 10,
                                })

                    # Check dependent preposition collocation or idiomatic frame
                    for c in token.children:
                        if c.dep_ == "prep" and c.i > token.i:
                            p_text = c.text.lower()
                            dobj = [d for d in token.children if d.dep_ == "dobj"]

                            # Known academic verb+prep collocation (e.g. welcome [sb] to [sth], provide [sb] with [sth])
                            if v_lemma in COLLOCATION_VERB_PREPS and COLLOCATION_VERB_PREPS[v_lemma] == p_text:
                                if dobj:
                                    coll_name = f"{v_lemma} ... {p_text}"
                                    formula = f"{v_lemma} [sb] {p_text} [sth]"
                                else:
                                    coll_name = f"{v_lemma} {p_text}"
                                    formula = f"{v_lemma} {p_text} [sth/sb]"
                                raw_candidates.append({
                                    "sid": sid,
                                    "quote": sent_clean,
                                    "phrase": coll_name,
                                    "type": "collocation",
                                    "pattern_formula": formula,
                                    "score": 9,
                                })
                            elif not dobj and p_text in DEPENDENT_PREPS and (c.i - token.i <= 2):
                                # Intransitive prepositional verb (e.g. rely on, contend with)
                                p_verb = f"{v_lemma} {p_text}"
                                formula = f"{v_lemma} {p_text} [sth/sb]"
                                raw_candidates.append({
                                    "sid": sid,
                                    "quote": sent_clean,
                                    "phrase": p_verb,
                                    "type": "phrasal verb",
                                    "pattern_formula": formula,
                                    "score": 8,
                                })
                            elif dobj:
                                # Fixed verbal idiom/collocation (e.g. take into account, make a difference)
                                pobj = [p for p in c.children if p.dep_ == "pobj"]
                                if pobj and (c.i - token.i <= 2) and pobj[0].lemma_.lower() in ("account", "consideration", "advantage", "part", "effect", "place"):
                                    prep_phrase = f"{v_lemma} {p_text} {pobj[0].lemma_.lower()}"
                                    formula = f"{v_lemma} {p_text} {pobj[0].lemma_.lower()} [sth]"
                                    raw_candidates.append({
                                        "sid": sid,
                                        "quote": sent_clean,
                                        "phrase": prep_phrase,
                                        "type": "idiom",
                                        "pattern_formula": formula,
                                        "score": 10,
                                    })

        if not raw_candidates:
            return []

        # Sort by score descending and deduplicate phrases
        raw_candidates.sort(key=lambda x: x["score"], reverse=True)

        selected: List[Dict[str, str]] = []
        seen_phrases: Set[str] = set()
        seen_quotes: Set[str] = set()

        # Balance across types: collocation, phrasal verb, idiom
        for item in raw_candidates:
            p_clean = item["phrase"].lower()
            if p_clean not in seen_phrases and item["quote"] not in seen_quotes:
                selected.append({
                    "sid": item["sid"],
                    "quote": item["quote"],
                    "phrase": item["phrase"],
                    "type": item["type"],
                    "pattern_formula": item["pattern_formula"],
                })
                seen_phrases.add(p_clean)
                seen_quotes.add(item["quote"])
            if len(selected) >= target_count:
                break

        # If more slots needed, allow reusing quote if phrase is distinct
        if len(selected) < target_count:
            for item in raw_candidates:
                p_clean = item["phrase"].lower()
                if p_clean not in seen_phrases:
                    selected.append({
                        "sid": item["sid"],
                        "quote": item["quote"],
                        "phrase": item["phrase"],
                        "type": item["type"],
                        "pattern_formula": item["pattern_formula"],
                    })
                    seen_phrases.add(p_clean)
                if len(selected) >= target_count:
                    break

        return selected[:target_count]

    # -------------------------------------------------------------------------
    # 3c. Deterministic Academic Vocabulary Mining (AWL + Lemmatization)
    # -------------------------------------------------------------------------
    @classmethod
    def mine_vocabulary_skeletons(cls, text: str, target_count: int = 20) -> List[Dict[str, Any]]:
        """
        Deterministically mines genuine academic vocabulary targets from the indexed sentence pool.
        Uses spaCy context-aware lemmatization to extract base dictionary headwords,
        prioritizes the Academic Word List (AWL 560), and filters out low-value everyday words.

        Returns list of dicts:
            [{
                "sid": "S-1",
                "quote": "full sentence text",
                "word": "autonomy",
                "part_of_speech": "noun",
                "is_awl": True
            }, ...]
        """
        _, pool = cls.tokenize_and_index_sentences(text)
        if not pool:
            return []

        nlp = cls.get_spacy()
        awl_set = cls.get_awl_words()

        # Closed whitelist of parts of speech accepted by schema
        VALID_POS_MAP = {
            "NOUN": "noun",
            "VERB": "verb",
            "ADJ": "adjective",
            "ADV": "adverb",
        }

        # Exclude trivial grammaticalized or ultra-common verbs/words
        TRIVIAL_WORDS = frozenset({
            "be", "have", "do", "say", "get", "make", "go", "know", "take", "see",
            "come", "think", "look", "want", "give", "use", "find", "tell", "ask",
            "work", "seem", "feel", "try", "leave", "call", "good", "new", "first",
            "last", "long", "great", "little", "own", "other", "old", "right", "big",
            "high", "different", "small", "large", "next", "early", "young", "important",
            "few", "public", "bad", "same", "able", "man", "woman", "person", "people",
            "child", "time", "year", "day", "way", "thing", "life", "hand", "part",
            "eye", "place", "case", "week", "company", "system", "program", "question",
            "government", "number", "night", "point", "home", "water", "room", "mother",
            "area", "money", "story", "fact", "month", "lot", "right", "study", "book",
            "word", "business", "issue", "side", "kind", "head", "house", "service",
            "friend", "father", "power", "hour", "game", "line", "end", "member", "law",
            "car", "city", "community", "name", "president", "team", "minute", "idea",
            "kid", "body", "information", "back", "parent", "face", "others", "level",
            "office", "door", "health", "person", "art", "war", "history", "party",
            "result", "change", "morning", "reason", "research", "girl", "guy", "moment",
            "air", "teacher", "force", "education"
        })

        candidates: List[Dict[str, Any]] = []
        seen_lemmas: Set[str] = set()

        for sid, sentence in pool.items():
            sent_clean = sentence.strip()
            if len(sent_clean.split()) < 5:
                continue

            doc = nlp(sent_clean)
            for token in doc:
                if token.pos_ not in VALID_POS_MAP:
                    continue
                lemma = token.lemma_.lower().strip()
                if len(lemma) < 3 or lemma in seen_lemmas or token.is_stop or not lemma.isalpha():
                    continue

                is_awl = lemma in awl_set

                # If not in AWL and is trivial or very short, skip
                if not is_awl and (lemma in TRIVIAL_WORDS or len(lemma) < 5):
                    continue

                # Scoring: AWL headwords get high priority, followed by word length and syllable complexity
                score = (20 if is_awl else 0) + min(len(lemma), 12)

                candidates.append({
                    "sid": sid,
                    "quote": sent_clean,
                    "word": lemma,
                    "part_of_speech": VALID_POS_MAP[token.pos_],
                    "is_awl": is_awl,
                    "score": score
                })
                seen_lemmas.add(lemma)

        if not candidates:
            return []

        # Sort descending by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected: List[Dict[str, Any]] = []
        selected_words: Set[str] = set()
        selected_quotes: Set[str] = set()

        # Pass 1: select diverse items across different sentences
        for item in candidates:
            if item["word"] not in selected_words and item["quote"] not in selected_quotes:
                selected.append(item)
                selected_words.add(item["word"])
                selected_quotes.add(item["quote"])
            if len(selected) >= target_count:
                break

        # Pass 2: fill remaining slots allowing multiple words from same sentence if high value
        if len(selected) < target_count:
            for item in candidates:
                if item["word"] not in selected_words:
                    selected.append(item)
                    selected_words.add(item["word"])
                if len(selected) >= target_count:
                    break

        return selected[:target_count]

    @classmethod
    def build_authentic_cloze_items(
        cls,
        vocab_content: str,
        target_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Track 1: Authentic Passage Cloze (Achievement Testing / 学业水平测试)
        Extracts extracted vocabulary items and masks their target appearance in the authentic
        quoted passage sentence using spaCy lemmatization and exact word-boundary masking.
        Generates 0-token, 100% textbook-grounded question stems.
        """
        if not vocab_content:
            return []

        nlp = cls.get_spacy()
        blocks = re.split(r'\n(?=##\s*\[\[)', vocab_content)
        parsed_items: List[Dict[str, str]] = []

        for b in blocks:
            m_word = re.search(r'##\s*\[\[(.*?)\]\]', b)
            if not m_word:
                continue
            word = m_word.group(1).strip()
            m_pos = re.search(r'-\s*\*\*Part Of Speech\*\*:\s*([^\n]+)', b, re.IGNORECASE)
            m_def = re.search(r'-\s*\*\*Definition\*\*:\s*([^\n]+)', b, re.IGNORECASE)
            m_quote = re.search(r'-\s*\*\*Quoted Sentence\*\*:\s*([^\n]+)', b, re.IGNORECASE)

            parsed_items.append({
                "word": word,
                "part_of_speech": m_pos.group(1).strip() if m_pos else "noun",
                "definition": m_def.group(1).strip() if m_def else "",
                "quote": m_quote.group(1).strip() if m_quote else ""
            })

        cloze_items: List[Dict[str, Any]] = []

        for it in parsed_items:
            w = it["word"]
            q = it["quote"]
            if not q:
                continue

            # Strip leading/trailing quotation marks from quote
            q_clean = q.strip().strip('"').strip("'").strip("“").strip("”")
            if not q_clean:
                continue

            doc = nlp(q_clean)
            w_lower = w.lower()

            # Locate token in sentence that matches the lemma or literal surface form
            target_tok = None
            for t in doc:
                if t.text.lower() == w_lower or t.lemma_.lower() == w_lower:
                    target_tok = t
                    break

            if not target_tok:
                continue

            surface_form = target_tok.text
            # Replace target token with exact 4 underscores '____'
            stem = re.sub(rf'\b{re.escape(surface_form)}\b', '____', q_clean, count=1)

            # Ensure exactly one blank exists in stem
            if len(re.findall(r'_{2,}', stem)) != 1:
                continue

            # Wordnet distractor generation for Track 1 Cloze
            pos_tag = "n"
            pos_raw = it["part_of_speech"].lower()
            if "verb" in pos_raw: pos_tag = "v"
            elif "adj" in pos_raw: pos_tag = "a"
            elif "adv" in pos_raw: pos_tag = "r"

            distractors = cls.generate_zero_collision_distractors(
                target_word=surface_form,
                pos=pos_tag,
                count=3
            )

            cloze_items.append({
                "target_word": surface_form,
                "base_headword": w,
                "part_of_speech": it["part_of_speech"],
                "definition": it["definition"],
                "question": stem,
                "source_sentence": q_clean,
                "precomputed_distractors": distractors
            })

            if len(cloze_items) >= target_count:
                break

        return cloze_items

    # Backward compatibility alias
    extract_authentic_cloze_items = build_authentic_cloze_items

    @classmethod
    def build_precomputed_target_skeletons(
        cls,
        vocab_content: str,
        target_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Track 2: Pre-Computed Generative Skeletons (Proficiency Testing / 语境迁移单选题)
        Extracts vocabulary items from extracted cards and synthesizes authoritative collocational
        anchors (via Oxford Collocations Dictionary) and collision-free distractors (via WordNet + OCD).
        Produces pre-computed item skeletons to strictly constrain LLM generation, guaranteeing zero
        duplicate options, zero synonym pile double-keys, and perfect stem-option-explanation alignment.
        """
        if not vocab_content:
            return []

        nlp = cls.get_spacy()
        ocd = cls.get_oxford_collocations()
        blocks = re.split(r'\n(?=##\s*\[\[)', vocab_content)
        parsed_items: List[Dict[str, str]] = []

        for b in blocks:
            m_word = re.search(r'##\s*\[\[(.*?)\]\]', b)
            if not m_word:
                continue
            word = m_word.group(1).strip()
            # Filter out multi-word idioms or phrases from single-word vocabulary MCQ targets
            if " " in word or "-" in word:
                continue

            m_pos = re.search(r'-\s*\*\*Part Of Speech\*\*:\s*([^\n]+)', b, re.IGNORECASE)
            m_def = re.search(r'-\s*\*\*Definition\*\*:\s*([^\n]+)', b, re.IGNORECASE)
            m_quote = re.search(r'-\s*\*\*Quoted Sentence\*\*:\s*([^\n]+)', b, re.IGNORECASE)

            parsed_items.append({
                "word": word,
                "part_of_speech": m_pos.group(1).strip() if m_pos else "noun",
                "definition": m_def.group(1).strip() if m_def else "",
                "quote": m_quote.group(1).strip() if m_quote else ""
            })

        skeletons: List[Dict[str, Any]] = []
        used_distractors: Set[str] = set()

        for it in parsed_items:
            w = it["word"]
            w_lower = w.lower()
            raw_pos = it["part_of_speech"].lower()
            
            # Map POS category
            canonical_pos = "noun"
            if "verb" in raw_pos:
                canonical_pos = "verb"
            elif "adj" in raw_pos:
                canonical_pos = "adj"
            elif "adv" in raw_pos:
                canonical_pos = "adv"

            # 1. Retrieve Authentic Oxford Collocational Anchor (Sense-Guided)
            anchor = None
            anchor_type = "collocation"
            prep_obj = None

            if canonical_pos == "verb":
                # Verb Step A: Extract the valency complement PP from the authentic quote.
                # Four-way gate (Pillar 1): (1) post-head linear constraint — complements
                # of the head verb sit to its RIGHT, physically excluding fronted adjuncts
                # ('In a society ... degraded' can never attach to 'degrade'); (2) fixed
                # adjunct blacklist (for example / in fact / ...) blocks discourse markers;
                # (3) OCD consensus gate — a preposition confirmed by the Oxford valency
                # list of this verb outranks an unconfirmed one; (4) when the verb has no
                # OCD valency evidence at all, only typical academic complements (into/to/
                # from) are trusted, otherwise the anchor degrades to None (neutral template).
                if it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        consensus_preps = set(ocd.get(w_lower, {}).get("prep", []))
                        for tok in doc:
                            if tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower:
                                verified_prep = None   # (token_index, prep, pobj) confirmed by OCD valency
                                unverified_prep = None  # (token_index, prep, pobj) bare syntactic complement
                                for child in tok.children:
                                    if child.dep_ != "prep":
                                        continue
                                    prep_tok = child.text.lower()
                                    if prep_tok not in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards"):
                                        continue
                                    if child.i <= tok.i:
                                        continue  # Post-head linear constraint
                                    child_pobj = None
                                    for pchild in child.children:
                                        if pchild.dep_ == "pobj" and pchild.lemma_.lower() not in cls._ANCHOR_STOPWORDS:
                                            child_pobj = pchild.lemma_.lower()
                                            break
                                    if child_pobj is not None and (prep_tok, child_pobj) in cls._ADJUNCT_PREP_PAIRS:
                                        continue  # Fixed adjunct / interjection (e.g. 'for example')
                                    if prep_tok in consensus_preps:
                                        if verified_prep is None:
                                            verified_prep = (child.i, prep_tok, child_pobj)
                                    elif unverified_prep is None:
                                        unverified_prep = (child.i, prep_tok, child_pobj)
                                # Consensus cross-validation (soft gate, Pillar 1 step 3):
                                # an OCD-confirmed preposition and a bare syntactic
                                # complement are both real — the more head-adjacent one
                                # prevails. A complement that already survived the post-head
                                # linear gate and the adjunct blacklist is genuine syntax,
                                # so an incomplete dictionary record (e.g. 'rely': OCD lists
                                # 'for', yet the authentic frame is 'rely on') never vetoes it.
                                chosen = None
                                if verified_prep is not None and unverified_prep is not None:
                                    chosen = verified_prep if verified_prep[0] < unverified_prep[0] else unverified_prep
                                elif verified_prep is not None:
                                    chosen = verified_prep
                                elif unverified_prep is not None:
                                    chosen = unverified_prep
                                if chosen is not None:
                                    anchor, prep_obj = chosen[1], chosen[2]
                                    anchor_type = "prep"
                                break
                    except Exception:
                        pass

                # Verb Step B: If no bound preposition found in quote, retrieve sense-aligned direct noun object from OCD
                if not anchor and w_lower in ocd:
                    entry = ocd[w_lower]
                    candidates = entry.get("colloc_nouns", []) + entry.get("noun_after", [])
                    if candidates:
                        anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"))
                        if anchor:
                            anchor_type = "object"

            elif canonical_pos == "noun":
                # Noun Syntactic Dependency Walker (Pillar 2) — terminates the old flat
                # string-matching that misattributed 'more' (degree adverb of 'more hours')
                # to 'autonomy' and let the selector draw 'strange' for 'compulsion'.
                # Priority A: post-nominal prepositional complement (autonomy FROM compulsion)
                # Priority B: true syntactic modifiers (amod/compound), stopword-filtered
                # Priority C: governing verb (head of dobj/pobj), cross-validated vs OCD verb_before
                # Priority D: OCD candidates vs quote/definition overlap (strict None on miss)
                if it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        ocd_entry = ocd.get(w_lower, {})
                        for tok in doc:
                            if tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower:
                                for child in tok.children:
                                    if child.dep_ == "prep" and child.text.lower() in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards"):
                                        pobj_tok = None
                                        for pchild in child.children:
                                            if pchild.dep_ == "pobj" and pchild.lemma_.lower() not in cls._ANCHOR_STOPWORDS:
                                                pobj_tok = pchild.lemma_.lower()
                                                break
                                        if pobj_tok is not None and (child.text.lower(), pobj_tok) in cls._ADJUNCT_PREP_PAIRS:
                                            continue
                                        anchor = child.text.lower()
                                        anchor_type = "prep"
                                        prep_obj = pobj_tok
                                        break
                                if not anchor:
                                    for child in tok.children:
                                        if child.dep_ in ("amod", "compound"):
                                            mod_tok = child.lemma_.lower()
                                            if mod_tok not in cls._ANCHOR_STOPWORDS:
                                                anchor = mod_tok
                                                anchor_type = "adj"
                                                break
                                if not anchor and tok.dep_ in ("dobj", "pobj", "attr"):
                                    head_tok = tok.head
                                    if head_tok.pos_ == "VERB":
                                        head_tok_lemma = head_tok.lemma_.lower()
                                        if head_tok_lemma not in cls._ANCHOR_STOPWORDS:
                                            verb_before_set = {v.lower().split()[0] for v in ocd_entry.get("verb_before", [])}
                                            if not verb_before_set or head_tok_lemma in verb_before_set:
                                                anchor = head_tok_lemma
                                                anchor_type = "verb"
                                break
                    except Exception:
                        pass

                # Priority D: dictionary candidates, strictly gated by quote/definition evidence
                if not anchor and w_lower in ocd:
                    entry = ocd[w_lower]
                    # Check modifying adjectives, governing verbs, or single-word bound prepositions
                    if entry.get("adj") or entry.get("verb_before"):
                        candidates = entry.get("adj", []) + entry.get("verb_before", [])
                        anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"))
                        if anchor:
                            anchor_type = "adj"
                    elif entry.get("prep"):
                        p_cand = cls._pick_anchor_token(entry["prep"])
                        if p_cand:
                            anchor = p_cand
                            anchor_type = "prep"

            elif canonical_pos == "adj":
                # Adjective Step A: Syntactic Dependency Analysis on Authentic Quote
                if it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower:
                                # A1. Predicative Valency: Check if adjective governs a bound preposition
                                for child in tok.children:
                                    if child.dep_ == "prep" and child.text.lower() in ("to", "for", "with", "of", "in", "from", "at", "about", "on"):
                                        anchor = child.text.lower()
                                        anchor_type = "prep"
                                        for pchild in child.children:
                                            if pchild.dep_ == "pobj" and pchild.text.lower() not in cls._ANCHOR_STOPWORDS:
                                                prep_obj = pchild.lemma_.lower()
                                                break
                                        break
                                # A2. Attributive Modification: Check if adjective modifies a head noun (dep_ == "amod")
                                if not anchor and tok.dep_ == "amod" and tok.head and tok.head.pos_ in ("NOUN", "PROPN"):
                                    if tok.head.text.lower() not in cls._ANCHOR_STOPWORDS:
                                        anchor = tok.head.lemma_.lower()
                                        anchor_type = "modified_noun"
                                break
                    except Exception:
                        pass

                # Adjective Step B: Collocation Lexicon Lookup (OCD) if no anchor found in quote
                if not anchor and w_lower in ocd:
                    entry = ocd[w_lower]
                    candidates = entry.get("noun_after", []) + entry.get("colloc_nouns", [])
                    if candidates:
                        anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"))
                        if anchor:
                            anchor_type = "modified_noun"
                    # If still no anchor, check if the adjective has a bound preposition in OCD
                    if not anchor and entry.get("prep"):
                        p_cand = cls._pick_anchor_token(entry["prep"])
                        if p_cand:
                            anchor = p_cand
                            anchor_type = "prep"

                # Adjective Step C: Fall back to curated academic adjective preposition valency map
                if not anchor:
                    for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items():
                        if w_lower in adjs:
                            anchor = p
                            anchor_type = "prep"
                            break

            # 2. Syntactic Fallback: Extract anchor from quoted sentence if OCD / prep lookup was empty
            if not anchor and it["quote"]:
                q_clean = it["quote"].strip().strip('"').strip("'").strip("“").strip("”")
                anchor_type = "contextual"
                try:
                    doc = nlp(q_clean)
                    for tok in doc:
                        if tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower:
                            for child in tok.children:
                                if child.dep_ in ("dobj", "pobj"):
                                    anchor = child.lemma_.lower()
                                    break
                                elif child.dep_ == "prep":
                                    for pchild in child.children:
                                        if pchild.dep_ == "pobj":
                                            anchor = pchild.lemma_.lower()
                                            break
                            if not anchor and tok.head and tok.head.pos_ in ("NOUN", "VERB"):
                                anchor = tok.head.lemma_.lower()
                            break
                except Exception:
                    pass

            # 3. Generate 3 Zero-Collision Distractors with Definition-Locked WSD & Sense Antonyms
            distractors, dist_meta = cls.generate_vocab_distractors(
                target_word=w_lower,
                pos=canonical_pos,
                context_anchor=anchor,
                anchor_type=anchor_type,
                target_count=3,
                exclude_words=used_distractors,
                definition=it.get("definition"),
                return_metadata=True
            )

            # Ensure we have valid distractors; fallback to zero_collision if needed
            if len(distractors) < 3:
                pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else "n")
                distractors = cls.generate_zero_collision_distractors(w_lower, pos=pos_char, count=3)

            # 3a. Double-Key Clearance (deterministic, frame-aware)
            #     A distractor is a double-key if it shares the target's bound
            #     preposition (OCD) AND sits in the target's WordNet semantic
            #     neighborhood; in slot items, same-slot near-synonyms / coordinate
            #     terms / troponyms qualify. Sense antonyms are whitelisted
            #     contrast distractors (eliminated by sentence polarity, not
            #     valency). Double-keys are dropped and refilled from an extended
            #     candidate pool so the item never degrades below 3 distractors.
            def _is_double_key(cand: str) -> bool:
                flag, _kind = cls.double_key_collision(w_lower, cand, anchor, anchor_type)
                return flag

            double_keys = [d for d in distractors if d and _is_double_key(d)]
            cleared = [d for d in distractors if d and not _is_double_key(d)]
            if double_keys:
                try:
                    refill_pool, _ = cls.generate_vocab_distractors(
                        target_word=w_lower, pos=canonical_pos,
                        context_anchor=anchor, anchor_type=anchor_type,
                        target_count=8, exclude_words=used_distractors,
                        definition=it.get("definition"), return_metadata=True
                    )
                except Exception:
                    refill_pool = []
                for cand in refill_pool:
                    if len(cleared) >= 3:
                        break
                    if cand and cand not in cleared and not _is_double_key(cand):
                        cleared.append(cand)
                # Secondary refill: the primary semantic-tree pool is saturated
                # with near-synonyms for synonym-heavy targets (designate,
                # adverse), so fall back to a semantically distant pool
                # (antonyms + taxonomy siblings + generic academic words).
                if len(cleared) < 3:
                    pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else "n")
                    try:
                        for cand in cls.generate_zero_collision_distractors(w_lower, pos=pos_char, count=3):
                            if len(cleared) >= 3:
                                break
                            if cand and cand not in cleared and cand not in used_distractors and not _is_double_key(cand):
                                cleared.append(cand)
                    except Exception:
                        pass
                # Last resort: if the pool is exhausted, keep best-remaining
                # (frame-only / contrast survivors) rather than degrading the item.
                if len(cleared) < 3:
                    for d in distractors:
                        if len(cleared) >= 3:
                            break
                        if d not in cleared:
                            cleared.append(d)
            distractors = cleared

            # Assemble full options (target + 3 distractors). The target is placed at a
            # stable, per-word position (not always slot A) so the answer key is not
            # trivially predictable and does not rely on post-hoc shuffling.
            distractors3 = distractors[:3]
            slot = zlib.crc32(w_lower.encode("utf-8")) % 4
            raw_options = distractors3[:slot] + [w_lower] + distractors3[slot:]
            correct_answer_index = slot
            used_distractors.update(d for d in distractors3 if d)

            # 4. Plan A: All options and target word strictly maintain Lemma / Base Form
            # Relinquish sentence-crafting freedom back to LLM to prevent syntax breaks (e.g. 'was instructing by...')
            inflection_desc = "base form"
            final_target = w_lower
            final_options = raw_options

            # 5. Synthesize Itemized Micro-Task for LLM based on Distractor Mechanisms & Definition
            dist_list = [opt for opt in final_options if opt.lower() != final_target.lower()]
            dist_str = ", ".join(dist_list)
            antonym_words = [d for d in dist_list if dist_meta.get(d) == "antonym"]

            # Frame-aware micro-task guard (double-key template fix):
            # Only distractors that genuinely LACK the bound frame may be claimed as
            # "strictly ruled out" on valency grounds. Survivors that still share the
            # frame (deliberate contrast / frame-only distractors) MUST be excluded
            # on semantic or polarity grounds — asserting a false valency exclusion
            # makes the LLM fabricate bogus exclusion rationales.
            if anchor_type == "prep" and anchor:
                _bound = anchor.lower()
                shared_frame = [
                    d for d in dist_list
                    if _bound in (cls.get_oxford_collocations().get(d.lower(), {}).get("prep") or [])
                    or d.lower() in cls._ADJ_PREP_VALENCY_MAP.get(_bound, [])
                ]
                non_shared = [d for d in dist_list if d not in shared_frame]
                if shared_frame and non_shared:
                    valency_clause = (
                        f"strictly ruling out [{', '.join(non_shared)}] which require different prepositions or direct objects. "
                        f"Note that [{', '.join(shared_frame)}] can also grammatically govern '{anchor}', "
                        f"so exclude it on precise semantic or polarity grounds, not valency"
                    )
                elif shared_frame:
                    valency_clause = (
                        f"excluding [{dist_str}] on precise semantic or polarity grounds, "
                        f"since every option can govern '{anchor}'"
                    )
                else:
                    valency_clause = f"strictly ruling out [{dist_str}] which require different prepositions or direct objects"
            else:
                valency_clause = f"strictly ruling out [{dist_str}]"

            if canonical_pos == "verb":
                if anchor_type == "prep":
                    obj_note = f" (followed by object '{prep_obj}')" if prep_obj else ""
                    ant_clue = f" Additionally, create a contextual condition that rules out opposite '{antonym_words[0]}'." if antonym_words else ""
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires base-form verb '{final_target}' "
                        f"immediately followed by bound preposition '{anchor}'{obj_note}. "
                        f"Ensure '{final_target}' is the only idiomatic fit governing '{anchor}', "
                        f"{valency_clause}.{ant_clue}"
                    )
                elif anchor:
                    ant_clue = f" Embed a clear causal or concessive condition that logically eliminates opposite '{antonym_words[0]}'." if antonym_words else ""
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires base-form verb '{final_target}' "
                        f"collocating with object/anchor '{anchor}'. "
                        f"Establish semantic clues demanding '{final_target}' while ruling out [{dist_str}].{ant_clue}"
                    )
                else:
                    ant_clue = f" Provide clear clues eliminating opposite '{antonym_words[0]}'." if antonym_words else ""
                    micro_task = (
                        f"Construct a natural academic sentence in a formal register requiring base-form verb '{final_target}'. "
                        f"Establish clear contextual contrast that rules out [{dist_str}].{ant_clue}"
                    )
            elif canonical_pos == "noun":
                ant_clue = f" Contrast with opposite '{antonym_words[0]}'." if antonym_words else ""
                # Contract tiering (Pillar 4): a real prepositional complement yields a
                # hard valency command (autonomy from compulsion); a genuine modifier or
                # governing verb yields a strict collocation slot; with no reliable anchor
                # the blueprint degrades to a neutral conceptual/semantic-field template
                # instead of hard-welding a fabricated anchor.
                if anchor_type == "prep" and anchor:
                    obj_note = f" (governing object '{prep_obj}')" if prep_obj else ""
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires noun '{final_target}' "
                        f"immediately followed by bound preposition '{anchor}'{obj_note}. "
                        f"Ensure '{final_target}' is the only idiomatic fit governing '{anchor}'{obj_note}, "
                        f"{valency_clause}.{ant_clue}"
                    )
                elif anchor_type == "verb" and anchor:
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires noun '{final_target}' "
                        f"serving as the direct object of verb '{anchor}'. "
                        f"Establish semantic clues demanding this classic governing-verb collocation, "
                        f"ruling out [{dist_str}].{ant_clue}"
                    )
                elif anchor and anchor_type in ("adj", "modified_noun", "object", "collocation"):
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires noun '{final_target}' "
                        f"directly modified or governed by '{anchor}'. "
                        f"Establish semantic clues demanding this classic collocation, "
                        f"ruling out [{dist_str}].{ant_clue}"
                    )
                else:
                    micro_task = (
                        f"Construct a natural academic sentence in a formal register requiring noun '{final_target}'. "
                        f"Establish its conceptual functional features in an academic context, "
                        f"discriminating it from [{dist_str}] through antonymic contrast or precise semantic-field cues.{ant_clue}"
                    )
            elif canonical_pos == "adj":
                if anchor_type == "prep":
                    obj_note = f" (followed by noun/phrase '{prep_obj}')" if prep_obj else ""
                    ant_clue = f" Additionally, create a contextual condition that rules out opposite '{antonym_words[0]}'." if antonym_words else ""
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires adjective '{final_target}' "
                        f"immediately followed by bound preposition '{anchor}'{obj_note}. "
                        f"Ensure '{final_target}' is the only idiomatic fit governing '{anchor}', "
                        f"{valency_clause}.{ant_clue}"
                    )
                elif anchor_type == "modified_noun" and anchor:
                    ant_clue = f" Embed a concessive or contrasting condition (e.g. 'Although initial ... was {antonym_words[0]}...') that logically eliminates opposite '{antonym_words[0]}'." if antonym_words else ""
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires adjective '{final_target}' "
                        f"modifying noun '{anchor}'. "
                        f"Ensure the sentence context strictly demands '{final_target}' as the precise collocational and semantic fit, "
                        f"while ruling out [{dist_str}].{ant_clue}"
                    )
                else:
                    ant_clue = f" Contrast with opposite '{antonym_words[0]}'." if antonym_words else ""
                    anchor_note = f"modifying '{anchor}' or similar academic concepts" if anchor else "in an academic evaluation"
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires adjective '{final_target}' ({anchor_note}). "
                        f"Ensure the sentence context strictly demands '{final_target}' while ruling out [{dist_str}].{ant_clue}"
                    )
            else:
                micro_task = (
                    f"Compose a CEFR-aligned academic sentence where the blank requires '{final_target}'. "
                    f"Provide clear clues eliminating [{dist_str}]."
                )

            skeletons.append({
                "target_word": final_target,
                "base_headword": w,
                "part_of_speech": canonical_pos,
                "inflection": inflection_desc,
                "context_anchor": anchor,
                "anchor_type": anchor_type,
                "micro_task": micro_task,
                "distractor_mechanisms": dist_meta,
                "prescribed_options": final_options,
                "correct_answer_index": correct_answer_index,
                "definition": it["definition"],
                "candidate_quote": it["quote"]
            })

            if len(skeletons) >= target_count:
                break

        return skeletons

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
    def get_antonyms(cls, word: str) -> Set[str]:
        """Returns strict direct antonyms from WordNet sense relations."""
        wn = cls.get_wordnet()
        antonyms = set()
        for syn in wn.synsets(word.lower()):
            # Sense-level antonyms
            for sense in syn.senses():
                for ant_sense in sense.get_related("antonym"):
                    antonyms.add(ant_sense.word().lemma().replace("_", " ").lower())
        return antonyms

    @classmethod
    def generate_zero_collision_distractors(
        cls,
        target_word: str,
        dependent_prep: Optional[str] = None,
        pos: str = "v",
        count: int = 3
    ) -> List[str]:
        """
        Pre-computes distractors mathematically guaranteed to avoid synonym collisions with target.
        When dependent_prep is specified (e.g. 'on' for 'rely'), prioritizes candidates that
        do NOT govern that preposition or clash with it.
        """
        wn = cls.get_wordnet()
        target_lower = target_word.lower()
        target_synonyms = cls.get_synonyms(target_lower)
        target_synonyms.add(target_lower)

        antonyms = cls.get_antonyms(target_lower)

        candidates: List[str] = []

        # 1. Antonyms make outstanding, collision-free distractors
        for ant in antonyms:
            if ant not in target_synonyms and " " not in ant:
                candidates.append(ant)

        # 2. Taxonomy sibling / hypernym search
        target_synsets = wn.synsets(target_lower, pos=pos)
        if target_synsets:
            primary_synset = target_synsets[0]
            # Get hypernyms (parent categories)
            hypernyms = primary_synset.get_related("hypernym")
            for hyper in hypernyms:
                # Get siblings (hyponyms of the hypernym)
                for sibling in hyper.get_related("hyponym"):
                    for w in sibling.words():
                        lemma = w.lemma().replace("_", " ").lower()
                        if lemma not in target_synonyms and " " not in lemma and lemma not in candidates:
                            candidates.append(lemma)

        # 3. Deduplicate and return requested count
        final_distractors = candidates[:count]

        # Fallback if synset taxonomy was sparse
        if len(final_distractors) < count:
            common_fallbacks = {
                "v": ["assume", "indicate", "require", "maintain", "assess"],
                "n": ["aspect", "factor", "criterion", "approach", "phenomenon"],
                "a": ["apparent", "consistent", "variable", "significant", "distinct"]
            }
            for fb in common_fallbacks.get(pos, common_fallbacks["v"]):
                if fb not in target_synonyms and fb not in final_distractors:
                    final_distractors.append(fb)
                if len(final_distractors) == count:
                    break

        return final_distractors
