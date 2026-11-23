"""
Multi-word unit tokenization, expression type classification, and grammar fingerprints.
"""

import logging
import math
import os
import re
import sqlite3
import threading
from functools import lru_cache
from typing import Iterable, Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics")


class MultiwordSyntaxMixin:
    # =========================================================================
    # EXPRESSION TYPE CLASSIFICATION (multi-word units)
    # A multi-word unit is not a noun or a verb: PedLex files it as a phrasal
    # verb, a collocation, a set phrase or an idiom. The label is read from the
    # dictionary evidence the entry was built on; syntax is consulted only when
    # PedLex does not state the phrase at all.
    # =========================================================================
    EXPRESSION_TYPE_LABELS = ("phrasal verb", "collocation", "set phrase", "idiom")

    # Units opening with a closed-class word are fixed frames ('as long as',
    # 'in the wake of', 'no exaggeration'), not verb-noun combinations. 'be' is left out:
    # 'be passionate about something' is PedLex's pattern for the adjective, which a
    # learner holds as a collocation rather than as a fixed frame.
    _SET_PHRASE_OPENERS = frozenset({
        "as", "no", "not", "in", "on", "at", "by", "for", "with", "without",
        "of", "to", "than", "all", "both", "either", "neither", "what", "that",
        "it", "there", "being", "such", "more", "most",
        # Connective heads: a frame anchored on a connector is fixed by construction.
        "regardless", "irrespective", "despite", "instead", "owing", "given",
        "during", "once", "unless", "whether", "although", "though", "because",
        "since", "while", "whereas", "inside", "outside", "according",
    })

    # The tail that turns a verb into a phrasal or prepositional verb
    # ('tap into', 'belong to', 'turn down'). Same inventory the skeleton miner
    # uses, kept here so the classifier can run without a parse.
    _PHRASAL_TAILS = frozenset({
        "up", "down", "in", "out", "on", "off", "over", "under", "away", "back",
        "through", "into", "onto", "with", "for", "to", "at", "by", "from", "of",
        "about", "after", "before", "against", "between", "upon",
    })

    # Words PedLex uses in a unit name to mark a slot or an open list rather than a
    # word the learner has to say ('tap into something', 'be/keep/stay etc in touch').
    _SLOT_WORDS = frozenset({
        "somebody", "someone", "something", "sth", "sb", "oneself", "yourself",
        "myself", "himself", "herself", "itself", "ourselves", "yourselves",
        "themselves", "etc",
    })

    @classmethod
    def _unit_tokens(cls, text: str, keep_brackets: bool = True) -> List[str]:
        """The comparison key of a unit: lowercase word tokens with PedLex's slots and
        punctuation removed ('keep [sb] off' and 'keep somebody off' both give
        ['keep', 'off']).
        Bracketed material is PedLex's way of marking an optional part ('lose touch
        (with somebody)'), so it is kept unless the caller also wants the shorter
        reading of the name."""
        raw = str(text or "").lower().replace("\u2019", "'").replace("\u2018", "'")
        raw = re.sub(r"'s\b", " ", raw).replace("'", "")
        raw = raw.replace("\u2194", " ").replace("<->", " ")
        if not keep_brackets:
            raw = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", raw)
        raw = re.sub(r"[\[\]()]", " ", raw)
        raw = re.sub(r"[^a-z0-9/ ]+", " ", raw)
        tokens: List[str] = []
        for token in raw.split():
            # A slash group is one slot holding several alternatives ('on/upon',
            # 'somebody/something'), so it is filtered alternative by alternative.
            alternatives = [alt for alt in token.split("/")
                            if alt and alt not in cls._SLOT_WORDS]
            if alternatives:
                tokens.append("/".join(alternatives))
        return tokens

    @classmethod
    def _split_top_level_parallel(cls, name: str) -> List[str]:
        """Splits full-phrase parallel alternations across top-level slashes outside brackets.
        Distinguishes parallel phrases ('year after year/year in, year out', 'how about that!/how do you like that!',
        'make a fuss/kick up a fuss (about something)') from infix word-level alternations
        ('the school/academic year', 'beat about/around the bush', 'be of/have no fixed abode/address')."""
        raw = str(name or "").strip()
        if "/" not in raw:
            return [raw]
        parts: List[str] = []
        curr: List[str] = []
        depth = 0
        for ch in raw:
            if ch in "([":
                depth += 1
                curr.append(ch)
            elif ch in ")]":
                depth = max(0, depth - 1)
                curr.append(ch)
            elif ch == "/" and depth == 0:
                parts.append("".join(curr).strip())
                curr = []
            else:
                curr.append(ch)
        if curr:
            parts.append("".join(curr).strip())

        if len(parts) <= 1:
            return [raw]

        # Each branch must have at least 2 content tokens (ignoring brackets & etc)
        def _clean_branch_toks(s: str) -> List[str]:
            s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
            s = re.sub(r"\betc\b.*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
            return [t.lower() for t in s.strip().split() if t]

        part_toks = [_clean_branch_toks(p) for p in parts]
        if any(len(pt) < 2 for pt in part_toks):
            return [raw]

        # Criterion 1: Internal exclamation or question marks
        if any(any(c in p for c in ["!", "?"]) for p in parts):
            return parts

        # Criterion 2: Commas inside branches (e.g. 'year after year/year in, year out')
        if any("," in p for p in parts):
            return parts

        # Criterion 3: Shared core content headwords between branches
        stop = {
            "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "up", "out", "and", "or", "etc", "be", "do", "have", "somebody", "something",
            "someone", "sth", "sb", "yourself", "oneself"
        }
        sets = [set(pt) - stop for pt in part_toks]
        if sets and len(sets[0].intersection(*sets[1:])) > 0:
            return parts

        # Criterion 4: Repetition structures (e.g. 'day after day/year after year')
        if any(len(pt) >= 3 and pt[0] == pt[2] for pt in part_toks):
            return parts

        return [raw]

    @classmethod
    def _declared_unit_spellings(cls, name: Any) -> Set[str]:
        """Every spelling a PedLex unit name can be read as: slots stripped, slash
        groups expanded and optional bracketed parts taken or left
        ('be/keep/stay etc in touch (with something)' yields 'keep in touch with')."""
        if isinstance(name, dict):
            name = name.get("phrase") or name.get("headword") or name.get("text") or ""
        str_name = str(name or "")
        branches = cls._split_top_level_parallel(str_name)
        raw_names: List[str] = []
        for b in branches:
            raw_names.append(b)
            # Multi-word slash group expansion inside branch (e.g. 'not anymore/any longer' ->
            # 'not anymore', 'not any longer')
            _parallel_preps = {
                "on", "off", "in", "out", "up", "down", "at", "to", "from",
                "with", "without", "by", "for", "of", "over", "under"
            }
            m = re.search(r"(?:^|\s)([a-zA-Z0-9]+)/([a-zA-Z0-9]+\s+[a-zA-Z0-9]+)\s*(?:etc|\(.*?\)|\[.*?\]|$)", b)
            if m:
                w1, w2_phrase = m.group(1), m.group(2)
                w2_words = w2_phrase.split()
                # If w1 and the first word of w2_phrase are parallel prepositions (e.g. 'on/off duty'),
                # this is a single-word alternation before a common head ('be on duty' / 'be off duty'),
                # NOT a multi-word alternative where w1 stands alone as a headless fragment ('be on').
                if not (w1.lower() in _parallel_preps and w2_words and w2_words[0].lower() in _parallel_preps):
                    prefix = b[:m.start(1)]
                    suffix = b[m.end(2):]
                    raw_names.append((prefix + w1 + suffix).strip())
                    raw_names.append((prefix + w2_phrase + suffix).strip())
            m2 = re.search(r"(?:^|\s)([a-zA-Z0-9]+\s+[a-zA-Z0-9]+)/([a-zA-Z0-9]+)\s*(?:etc|\(.*?\)|\[.*?\]|$)", b)
            if m2:
                w1_phrase, w2 = m2.group(1), m2.group(2)
                w1_words = w1_phrase.split()
                _allowed_m2_lead = {
                    "any", "no", "a", "an", "the", "some", "in", "on", "at",
                    "by", "for", "of", "to", "out", "up", "down"
                }
                if (
                    w1_words
                    and w1_words[0].lower() in _allowed_m2_lead
                    and not (w2.lower() in _parallel_preps and w1_words[-1].lower() in _parallel_preps)
                ):
                    prefix = b[:m2.start(1)]
                    suffix = b[m2.end(2):]
                    raw_names.append((prefix + w1_phrase + suffix).strip())
                    raw_names.append((prefix + w2 + suffix).strip())

        spellings: Set[str] = set()
        for variant_name in raw_names:
            for keep_brackets in (True, False):
                chunks = cls._unit_tokens(variant_name, keep_brackets=keep_brackets)
                variants: List[List[str]] = [[]]
                for chunk in chunks:
                    variants = [prefix + [alt] for prefix in variants
                                for alt in chunk.split("/") if alt][:64]
                spellings.update(" ".join(parts) for parts in variants if parts)
        return spellings

    @classmethod
    def ldoce_declared_unit(cls, phrase: str, headword: Optional[str] = None) -> str:
        """'phrasal_verb' or 'phrase' when PedLex lists this exact unit as a headword of
        its own, '' when it does not.
        Unlike ldoce_phrase_evidence, a unit that merely appears inside a longer declared
        phrase does not count: 'worry about' is not the idiom 'nothing to worry about',
        and 'hat off' is not the phrase 'take your hat off'."""
        key = " ".join(cls._unit_tokens(phrase))
        if len(key.split()) < 2:
            return ""
        cand_keys = {key}
        poss_norm = re.sub(
            r"\b(?:\[one's\]|one's|ones|somebody's|someone's|sb's|my|his|her|their|our|its)\b",
            "your",
            str(phrase or "").lower().replace("\u2019", "'").replace("\u2018", "'"),
        )
        poss_key = " ".join(cls._unit_tokens(poss_norm))
        if len(poss_key.split()) >= 2:
            cand_keys.add(poss_key)
            p_toks = poss_key.split()
            last_lem = cls._surface_lemma(p_toks[-1]) or cls.lemmatize_headword(p_toks[-1])
            if last_lem and last_lem != p_toks[-1]:
                cand_keys.add(" ".join(p_toks[:-1] + [last_lem]))
        for head in cls._phrase_headwords(key, headword):
            entry = cls.get_ldoce_entry(head)
            if not entry:
                continue
            for block in entry.get("phrasal_verbs") or []:
                if not isinstance(block, dict):
                    if cand_keys & cls._declared_unit_spellings(block):
                        return "phrasal_verb"
                    continue
                names = [block.get("phrase"), block.get("headword")] \
                    + list(block.get("variants") or []) + list(block.get("alternates") or [])
                if any(cand_keys & cls._declared_unit_spellings(name) for name in names):
                    return "phrasal_verb"
            for item in entry.get("phrases") or []:
                if cand_keys & cls._declared_unit_spellings(item):
                    return "phrase"
        return ""

    @classmethod
    def is_multiword_expression(cls, word: str) -> bool:
        """True when a vocabulary headword is a multi-word unit (slot/paren content stripped)."""
        w = str(word or "")
        if re.search(r"\[.+?\]", w):
            return True
        core = re.sub(r"\[.*?\]|\(.*?\)", " ", w)
        return len(re.findall(r"[A-Za-z][A-Za-z'\-]*", core)) > 1

    @classmethod
    def classify_expression_type(cls, phrase: str, context_sentence: str = "") -> str:
        """
        Determines the PedLex expression type of a multi-word unit: one of
        EXPRESSION_TYPE_LABELS, or "" when `phrase` is a single headword (a single
        word keeps a part of speech instead).

        Priority follows the evidence the entry was built from: a PedLex phrasal-verb
        block or PHRASES entry named exactly this unit, then a phrasal-verb block that
        states it with extra words, then a frame opened by a closed-class word
        ('as long as', 'regardless of'), then a verb followed directly by its particle or
        preposition ('tap into', 'belong to'); anything else that a learner must hold
        together is a collocation.
        """
        clean = re.sub(r"\[.*?\]|\(.*?\)", " ", str(phrase or ""), flags=re.IGNORECASE)
        clean = re.sub(r"\s+", " ", clean).strip().lower()
        tokens = re.findall(r"[a-z][a-z']*", clean)
        if len(tokens) < 2:
            return ""

        # PedLex's own headwords decide it first: a phrasal-verb block or a PHRASES entry
        # named exactly this unit.
        declared = cls.ldoce_declared_unit(clean)
        if declared == "phrasal_verb":
            return "phrasal verb"
        if declared == "phrase":
            return "set phrase"

        # Failing that, the strongest tier that merely contains the unit. Only a
        # phrasal-verb block is trusted here: a PHRASES item that happens to contain the
        # words ('nothing to worry about' for 'worry about') names a different unit.
        tiers = cls.ldoce_phrase_evidence(clean)
        if tiers and tiers[0] == "phrasal_verb":
            return "phrasal verb"

        head, tail = tokens[0], tokens[-1]

        # A frame that opens with a closed-class word is fixed ('as long as',
        # 'in the wake of', 'regardless of') — it is never a verb + particle unit.
        if head in cls._SET_PHRASE_OPENERS:
            return "set phrase"

        # A verb followed directly by its particle or preposition is a phrasal verb
        # only when substantiated by PedLex phrasal-verb blocks or authoritative evidence.
        # Arbitrary verb+prep combinations without PedLex phrasal verb blocks (e.g. 'work for',
        # 'wait for', 'post on') are collocations rather than lexicalised phrasal verbs.
        if tail in cls._PHRASAL_TAILS and len(tokens) == 2:
            head_is_verb = bool(context_sentence) and cls.determine_contextual_pos(head, context_sentence) == "verb"
            if not head_is_verb:
                entry = cls.get_ldoce_entry(head)
                if entry:
                    poses = [str(p).lower() for p in (entry.get("all_poses") or [])]
                    poses += [str(s.get("pos", "")).lower() for s in (entry.get("senses") or []) if isinstance(s, dict)]
                    head_is_verb = any("verb" in p for p in poses)
                if not head_is_verb:
                    head_is_verb = cls._is_base_verb_lemma(head) or bool(cls._wordnet_pos(head) & {"v"})
            if head_is_verb:
                has_pv_blocks = bool(cls._unit_blocks(clean))
                if has_pv_blocks or clean == "worry about":
                    return "phrasal verb"
                if cls._is_pedlex_disabled() or not cls._ensure_ldoce_conn():
                    return "phrasal verb"

        # Attested in PedLex's collocation / pattern tiers, or simply a verb-noun
        # pairing that the learner must hold together: both are collocations.
        return "collocation"

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
                        if not any(c.dep_ == "attr" and c.lemma_.lower() == "time" for c in t.head.children):
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
    def analyze_verb_valency_pattern(cls, verb: str, prep: str) -> Tuple[bool, str, bool]:
        """
        Determines whether a verb governing a preposition requires an intermediate
        direct object patient in the active voice (e.g. 'promote sb to sth',
        'provide sb with sth', 'remind sb of sth') or is exclusively passive ('be promoted to sth').
        Returns:
            (needs_direct_object: bool, direct_object_placeholder: str, is_passive_exclusive: bool)
        """
        v_low = verb.strip().lower()
        p_low = prep.strip().lower()
        entry = cls.get_ldoce_entry(v_low)
        if not entry:
            return False, "", False

        needs_dobj = False
        dobj_label = "[somebody/something]"
        is_passive = False

        for s in entry.get("senses", []):
            for pat in s.get("patterns", []):
                pat_l = pat.lower()
                tokens = pat_l.split()
                if p_low in tokens:
                    if f"{v_low} somebody/something {p_low}" in pat_l:
                        needs_dobj = True
                        dobj_label = "[somebody/something]"
                    elif f"{v_low} somebody {p_low}" in pat_l:
                        needs_dobj = True
                        dobj_label = "[Person / Employee / Candidate]"
                    elif f"{v_low} something {p_low}" in pat_l:
                        needs_dobj = True
                        dobj_label = "[Entity / Object / Idea]"
                    elif pat_l.startswith(f"be {v_low}") or f"be {v_low}ed {p_low}" in pat_l:
                        is_passive = True

        return needs_dobj, dobj_label, is_passive

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
