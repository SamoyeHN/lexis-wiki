"""Orthographic paronym generation, morpho-phonetic distance, and semantic fields."""

from __future__ import annotations

import math
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class DistractorParonymsMixin:

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

        # Open-source / fallback enrichment from PedLex + WordNet if PedLex table is disabled or sparse
        if not any(pos_vocab.values()):
            wn = cls.get_wordnet()
            pool = set(cls.get_pedlex_headwords())
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
            # Ensure pluralia tantum / -ics academic disciplines in PedLex (e.g. 'mechanics', 'robotics')
            # that live inside singular PedLex headword articles are indexed under 'n'.
            for w in cls.get_pedlex_headwords():
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
        """Same-family derivational noun pairs (e.g. emergence/emergency, optimism/optimist,
        anarchism/anarchist, colonialism/colonialist, politician/politics, scientist/science)
        are blocked under the Same-Word-Family Guard to prevent morphological root giveaways."""
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
        if cand in (target + "s", target + "es") or target in (cand + "s", cand + "es"):
            return False
        if wn_pos == "n" and min(len(target), len(cand)) >= 6 and (cand.startswith(target) or target.startswith(cand)):
            return False
        if wn_pos == "r" and target.endswith("ly") and cand.endswith("ly"):
            if target[0] != cand[0] or cls._edit_distance_bounded(target[:-2], cand[:-2]) > 1:
                return False
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

        tw_is_artifact_ism = bool(
            wn_pos == "n"
            and tw.endswith("ism")
            and any(s.lexfile() in ("noun.artifact", "noun.object") for s in cls.get_wordnet_synsets(tw, pos="n"))
        )
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
                and (not tw_is_artifact_ism or suff >= 4)
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
                if wn_pos == "v" and pref >= 3 and tw.endswith("ate") and w.endswith("ate"):
                    score += 30
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

        if pos_clean.startswith("n"):
            pos_tags = ("n",)
        elif pos_clean.startswith("v"):
            pos_tags = ("v",)
        elif pos_clean.startswith(("a", "j", "s")) and not pos_clean.startswith("adv"):
            pos_tags = ("a", "s")
        elif pos_clean.startswith(("r", "adv")):
            pos_tags = ("r",)
        else:
            pos_tags = ("a", "s", "v", "n", "r")

        target_ortho_vars = list(cls.orthographic_variants(target)) if hasattr(cls, "orthographic_variants") else [target]
        seeds = []
        seen_seed_ids: Set[str] = set()
        for q_t in target_ortho_vars:
            for p_tag in pos_tags:
                try:
                    for ss in wn.synsets(q_t, p_tag):
                        if ss.id not in seen_seed_ids:
                            seen_seed_ids.add(ss.id)
                            seeds.append(ss)
                except Exception:
                    pass
        if not seeds and pos_tags != ("a", "s", "v", "n", "r"):
            for q_t in target_ortho_vars:
                for p_tag in ("a", "s", "v", "n", "r"):
                    try:
                        for ss in wn.synsets(q_t, p_tag):
                            if ss.id not in seen_seed_ids:
                                seen_seed_ids.add(ss.id)
                                seeds.append(ss)
                    except Exception:
                        pass

        direct_same_pole: Set[str] = set()
        for s in seeds:
            add(s)
            for sw in s.words():
                direct_same_pole.add(sw.lemma().lower())
            for rel in ("similar", "also"):
                for t in s.get_related(rel):
                    add(t)
                    if rel == "similar":
                        for tw in t.words():
                            direct_same_pole.add(tw.lemma().lower())
                    if t.pos in ("a", "s"):
                        for rel2 in ("similar", "also"):
                            for u in t.get_related(rel2):
                                add(u)
            for rel in ("hypernym", "entailment", "hyponym"):
                for t in s.get_related(rel):
                    add(t)
            # Sense-locked antonyms (same cluster for adjectives; only 'similar' on ant_sense, never 'also')
            for sense in s.senses():
                for ant_sense in sense.get_related("antonym"):
                    antonyms.add(ant_sense.word().lemma().lower())
                    for t in ant_sense.synset().get_related("similar"):
                        for w in t.words():
                            antonyms.add(w.lemma().lower())
            if s.pos == "s":
                for head in s.get_related("similar"):
                    for hs in head.senses():
                        for ant_sense in hs.get_related("antonym"):
                            antonyms.add(ant_sense.word().lemma().lower())
                            for t in ant_sense.synset().get_related("similar"):
                                for w in t.words():
                                    antonyms.add(w.lemma().lower())

        ped_ox_syns, ped_ox_ants = cls.get_pedlex_and_oxford_synonyms_antonyms(target, pos=pos)
        for psyn in ped_ox_syns:
            near.add(psyn)
        for pant in ped_ox_ants:
            antonyms.add(pant)
        for g_ant in cls.get_antonyms(target, pos=pos):
            antonyms.add(g_ant)

        antonyms -= direct_same_pole
        for ov in target_ortho_vars:
            near.discard(ov)
            antonyms.discard(ov)
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
