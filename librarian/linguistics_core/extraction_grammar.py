"""
Deterministic COBUILD syntactic grammar classification, formula generation, skeleton mining, and extraction.
"""

import datetime
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import zlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics")


class ExtractionGrammarMixin:

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
        sentence_norm = sentence.replace('\u2019', "'").replace('\u2018', "'")
        nlp = cls.get_spacy()
        doc = nlp(sentence_norm)

        # Declarative DependencyMatcher Engine Pass (High-Precision DSL)
        try:
            from librarian.grammar_patterns import GrammarPatternEngine
            gp_engine = GrammarPatternEngine.get_engine(nlp)
            matched_patterns = gp_engine.match_sentence(doc)
            if matched_patterns:
                return matched_patterns[0]["category"]
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # 1. Rhetoric & Emphasis:
        #    a. Fronted Inversion: aux or ROOT precedes nsubj in linear word order
        #    b. Cleft Focus: expl ('it') + 'is/was' + focal constituent + relcl/that...
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
            if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj"):
                # Check copula verb
                head = token.head
                if head.lemma_ in ("be", "seem"):
                    if any(c.dep_ == "attr" and c.lemma_.lower() == "time" for c in head.children):
                        continue
                    # 1. Relcl under focal constituent with relative pronoun
                    has_relcl = False
                    for child in head.children:
                        if child.dep_ == "relcl":
                            if any(gc.text.lower() in ("that", "who", "whom", "which", "whose") for gc in child.children):
                                has_relcl = True
                        for gc in child.children:
                            if gc.dep_ == "relcl":
                                if any(ggc.text.lower() in ("that", "who", "whom", "which", "whose") for ggc in gc.children):
                                    has_relcl = True
                    if has_relcl:
                        return "Rhetoric & Emphasis"

                    # 2. Prepositional or adverbial cleft (e.g. It's in December that we tend...)
                    for child in head.children:
                        if child.dep_ in ("ccomp", "advcl"):
                            has_marker = any(
                                gc.dep_ in ("mark", "nsubj", "advmod", "dobj", "pobj")
                                and gc.text.lower() in ("that", "who", "whom", "where", "when")
                                for gc in child.children
                            )
                            if has_marker:
                                # Ensure focal element is not an evaluative adjective or extraposition noun (dummy-it)
                                _EXTRAPOSITION_NOUNS = {
                                    "surprise", "wonder", "secret", "doubt", "coincidence", "accident",
                                    "pity", "shame", "mystery", "fact", "truth", "news", "use", "good",
                                    "point", "time", "question", "matter"
                                }
                                has_eval_adj = any(
                                    (c.dep_ == "acomp" and c.pos_ == "ADJ")
                                    or (c.dep_ == "attr" and c.lemma_.lower() in _EXTRAPOSITION_NOUNS)
                                    for c in head.children
                                )
                                if not has_eval_adj:
                                    return "Rhetoric & Emphasis"

        # Antithesis / Correlative Parallelism check (not... but..., not only... but also)
        text_lower = sentence_norm.lower()
        if re.search(r"\bnot\s+only\b.*?\bbut\s+also\b", text_lower):
            return "Rhetoric & Emphasis"

        # Resultative Degree Constructions (so + Adj/Adv + that / such + NP + that)
        if re.search(r"\bso\s+[a-z-]{3,}(?:\s+[a-z-]{3,})?\s+that\b", text_lower) or re.search(r"\bsuch\s+(?:a\s+|an\s+)?[a-z-]{3,}(?:\s+[a-z-]{3,}){0,2}\s+that\b", text_lower):
            return "Rhetoric & Emphasis"

        # ---------------------------------------------------------------------
        # 2. Logic & Stance:
        #    Conditionals (if/unless/as long as/provided that) or Concessives (although/even though/while/whereas/in spite of)
        #    Adversative transitions (however/nevertheless/nonetheless)
        #    Dual-stance coordinate contrast (..., but [S] also...)
        # ---------------------------------------------------------------------
        # Multi-word subordinating connectives
        if re.search(r"\b(as long as|so long as|provided that|providing that|in case|on condition that)\b", text_lower):
            return "Logic & Stance"

        subordinating_conjs = {"although", "though", "while", "whereas", "even though", "if", "unless", "provided", "providing"}
        for token in doc:
            if token.dep_ == "mark" and token.lemma_.lower() in subordinating_conjs:
                if token.lemma_.lower() == "if" and token.head.dep_ != "advcl":
                    continue
                # Subjectless 'If/While/Though/Although + V-ing/V-ed/Adj' is a reduced non-finite adjunct, not a finite clause
                if not any(c.dep_ in ("nsubj", "nsubjpass", "expl", "csubj", "csubjpass") for c in token.head.children):
                    continue
                return "Logic & Stance"

        # Stance / Adversative transition (However, / Nevertheless, ...)
        if re.match(r"^(however|nevertheless|nonetheless|conversely|in contrast)\b", text_lower):
            return "Logic & Stance"

        # Stance / Contrastive coordinate balance (..., but [S] also...)
        if re.search(r"\b,\s*but\s+[a-z]+\s+(also|still|yet)\b", text_lower):
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
                    if child.dep_ in ("acl", "appos", "ccomp") and any(gc.dep_ == "mark" and gc.lower_ in ("that", "whether") for gc in child.children):
                        return "Cohesion & Framing"

        # ---------------------------------------------------------------------
        # 4. Information Packaging:
        #    a. Evaluative Dummy-It: It is + [adj/noun] + that/to [csubj/ccomp]
        #    b. Dense Object Complement: make/find/render/keep + Object + Adj
        #    c. Non-finite participial adjuncts: advcl with VerbForm=Part
        #    d. Non-finite subject nominalization (Gerund / Infinitive phrase as Subject)
        #    e. Correlative comparative: The more..., the more...
        #    f. Dense Prepositional Frame: Instead of / By / Thanks to + [V-ing/NP]
        #    g. Relative clauses: which/who/whom/whose + VP
        #    h. Elaborative non-restrictive relative clause: , which + VP
        # ---------------------------------------------------------------------
        # Correlative comparative: The more..., the more...
        COMPARATIVE_WORDS = r"(?:more|less|fewer|better|worse|[a-z]{2,}er)"
        if re.search(rf"\bthe\s+{COMPARATIVE_WORDS}\b(?!\s+hand\b).*?(?:,\s*|\band\s+the\s+{COMPARATIVE_WORDS}\b.*?,?\s*)\bthe\s+{COMPARATIVE_WORDS}\b", text_lower):
            return "Information Packaging"

        # Dense Prepositional Frame (Instead of, By doing, Due to, Thanks to + V-ing/NP, [Subject] + [VP])
        if re.match(r"^(instead of|by|through|despite|in spite of|thanks to|due to|owing to)\s+[a-z0-9\s-]+?,", text_lower):
            return "Information Packaging"

        # Non-finite subject nominalization (Gerund or Infinitive as subject)
        # Handles direct sentence subject or embedded clause subject (e.g. That is why [being on wheels] means...)
        for token in doc:
            if token.tag_ == "VBG" and token.dep_ in ("nsubj", "nsubjpass", "csubj"):
                return "Information Packaging"
            if token.tag_ == "VB" and token.dep_ == "csubj":
                return "Information Packaging"
            # Embedded gerund subject in why/that/wh-clauses where spaCy might label it advcl or pobj
            if token.tag_ == "VBG" and token.dep_ in ("advcl", "csubj") and token.head.pos_ in ("VERB", "AUX"):
                # Check if this VBG introduces a gerund clause following 'why' or copula
                has_wh = any(c.dep_ == "advmod" and c.lemma_.lower() in ("why", "how", "what", "where") for c in token.children)
                if has_wh:
                    return "Information Packaging"
                head_subjs = [c for c in token.head.children if c.dep_ in ("nsubj", "nsubjpass")]
                if not head_subjs and token.i < token.head.i:
                    return "Information Packaging"

        # Evaluative Dummy-It (spaCy tags 'It' as nsubj/expl and the complement clause as ccomp/csubj)
        _AMBIENT_PREDS = {
            "daylight", "dark", "night", "morning", "afternoon", "evening", "noon", "midnight",
            "dawn", "dusk", "winter", "summer", "spring", "autumn", "fall", "cold", "hot", "warm",
            "cool", "sunny", "rainy", "snowy", "windy", "cloudy", "late", "early", "o'clock"
        }
        for token in doc:
            if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj"):
                head = token.head
                if head.lemma_ in ("be", "seem") and not any(c.dep_ == "advmod" and c.lower_ in ("when", "where") for c in head.children):
                    eval_children = [
                        child for child in head.children
                        if child.dep_ in ("acomp", "attr") and child.lemma_.lower() not in _AMBIENT_PREDS
                    ]
                    comp_children = [
                        child for child in head.children
                        if child.dep_ in ("ccomp", "csubj", "xcomp")
                        and child.tag_ not in ("VBG", "VBN")
                        and not any(gc.dep_ == "advmod" and gc.lower_ in ("when", "where") for gc in child.children)
                    ]
                    if eval_children and comp_children and comp_children[0].i > eval_children[0].i:
                        return "Information Packaging"

        # Evaluative Dummy-It Object (find/make/think it adj to-V)
        for t in doc:
            if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                for c in t.children:
                    if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                        return "Information Packaging"

        # Dense Object Complement / Causative (make/find/render/keep + Object + Adj/Complement)
        for t in doc:
            if t.lemma_ in ("make", "find", "render", "keep") and t.pos_ in ("VERB", "AUX"):
                for c in t.children:
                    if c.dep_ in ("ccomp", "oprd") and c.pos_ == "ADJ":
                        return "Information Packaging"

        # Non-finite participial adjuncts (advcl with VBG or VBN)
        _OBJ_VING_VERBS = {"have", "find", "spend", "waste", "get", "catch", "keep", "leave", "start", "stop", "prevent", "busy"}
        for token in doc:
            if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                # Exclude 'have trouble/difficulty/time V-ing' and 'spend/waste time V-ing' complement frames
                if token.tag_ == "VBG" and token.head.lemma_.lower() in _OBJ_VING_VERBS and token.i > token.head.i:
                    if not any(t.text == "," for t in doc[token.head.i : token.i]):
                        continue
                # Ensure it has no subject of its own (non-finite) and not a preposition
                has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children)
                if not has_subj and token.lemma_.lower() not in ("include", "accord", "regard", "concern"):
                    return "Information Packaging"

        # Relative clauses (which, that, who, whom, whose)
        for token in doc:
            if token.dep_ == "relcl":
                if any(c.text.lower() in ("which", "that", "who", "whom", "whose") for c in token.children):
                    return "Information Packaging"

        # Passive voice (agentless or agentive passive foregrounding patient)
        _UNACCUSATIVE_STATE_VERBS = {"go", "come", "fall", "rise", "arrive", "depart", "vanish", "disappear", "die", "remain", "stay"}
        for token in doc:
            if token.dep_ in ("nsubjpass", "auxpass"):
                head = token.head
                if head.pos_ in ("VERB", "AUX") and head.lemma_.lower() not in _UNACCUSATIVE_STATE_VERBS:
                    return "Information Packaging"

        return None

    @classmethod
    def generate_cobuild_formula(cls, sentence: str, category: Optional[str] = None) -> str:
        """
        Deterministically derives standard algebraic COBUILD slot formula from sentence
        via spaCy dependency trees and syntactic pattern anchors.
        Replaces prone-to-hallucination LLM formula drafting with zero-token invariant logic.
        """
        sentence_norm = sentence.replace('\u2019', "'").replace('\u2018', "'")
        nlp = cls.get_spacy()
        doc = nlp(sentence_norm)
        cat = category or cls.classify_grammar_dependency(sentence_norm) or ""
        text_lower = sentence_norm.lower()

        # Declarative DependencyMatcher Engine Pass (High-Precision DSL)
        try:
            from librarian.grammar_patterns import GrammarPatternEngine
            gp_engine = GrammarPatternEngine.get_engine(nlp)
            matched_patterns = gp_engine.match_sentence(doc)
            if matched_patterns:
                if cat:
                    matched_cand = next((m for m in matched_patterns if m["category"].lower() == cat.lower() or cat.lower() in m["name"].lower() or cat.lower() in m["pattern_id"].lower()), None)
                else:
                    matched_cand = matched_patterns[0]
                if matched_cand and matched_cand.get("formula"):
                    return matched_cand["formula"]
        except Exception:
            pass

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
            _EXTRAPOSITION_NOUNS = {
                "surprise", "wonder", "secret", "doubt", "coincidence", "accident",
                "pity", "shame", "mystery", "fact", "truth", "news", "use", "good",
                "point", "time", "question", "matter"
            }
            for token in doc:
                if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj") and token.head.lemma_ in ("be", "seem"):
                    if any(
                        (c.dep_ == "acomp" and c.pos_ == "ADJ")
                        or (c.dep_ == "attr" and c.lemma_.lower() in _EXTRAPOSITION_NOUNS)
                        for c in token.head.children
                    ):
                        continue
                    # Check relcl
                    has_relcl = any(
                        (c.dep_ == "relcl" and any(gc.text.lower() in ("that", "who", "whom", "which", "whose") for gc in c.children))
                        or any(gc.dep_ == "relcl" and any(ggc.text.lower() in ("that", "who", "whom", "which", "whose") for ggc in gc.children) for gc in c.children)
                        for c in token.head.children
                    )
                    # Check prepositional or adverbial cleft
                    has_cleft_marker = any(
                        c.dep_ in ("ccomp", "advcl")
                        and any(gc.dep_ in ("mark", "nsubj", "advmod", "dobj", "pobj") and gc.text.lower() in ("that", "who", "whom", "where", "when") for gc in c.children)
                        for c in token.head.children
                    )
                    if has_relcl or has_cleft_marker:
                        return "It + [be] + [Focal Element] + that/who + [Clause]"
            # Correlative Parallelism (with or without Fronted Negative Inversion)
            if re.search(r"\bnot\s+only\b.*?\bbut\s+(?:also\b|[a-z]+\s+also\b)", text_lower):
                if re.search(r"(?:^|[,;—–-]\s*)not\s+only\s+(?:can|could|will|would|shall|should|may|might|must|do|does|did|have|has|had|am|is|are|was|were)\b", text_lower):
                    return "Not only + [Aux] + [Subject] + [VP], but + [Subject] + [also] + [VP]"
                return "[Subject] + not only + [VP], but also + [VP]"
            # Resultative so/such ... that
            if re.search(r"\bsuch\s+(?:a\s+|an\s+)?[a-z-]{3,}(?:\s+[a-z-]{3,}){0,2}\s+that\b", text_lower):
                return "such + [NP] + that + [Result Clause]"
            if re.search(r"\bso\s+[a-z-]{3,}(?:\s+[a-z-]{3,})?\s+that\b", text_lower):
                return "so + [Adj/Adv] + that + [Result Clause]"

        # 2. Logic & Stance
        if cat == "Logic & Stance" or not cat:
            # Multi-word conditional & concessive connectives
            m_conn = re.search(r"\b(as long as|so long as|provided that|providing that|in case|on condition that|even though|even if)\b", text_lower)
            if m_conn:
                conn_name = m_conn.group(1).capitalize()
                return f"{conn_name} + [Clause], [Subject] + [VP]"
            for token in doc:
                if token.dep_ == "mark" and token.lemma_.lower() in {"although", "though", "while", "whereas", "even though", "if", "unless", "provided"}:
                    if token.lemma_.lower() == "if" and token.head.dep_ != "advcl":
                        continue
                    if not any(c.dep_ in ("nsubj", "nsubjpass", "expl", "csubj", "csubjpass") for c in token.head.children):
                        continue
                    prev_w = doc[token.i - 1].lower_ if token.i > 0 else ""
                    if prev_w == "even" and token.lower_ in ("if", "though"):
                        mark_word = f"Even {token.lower_}"
                    else:
                        mark_word = token.text.capitalize()
                    return f"{mark_word} + [Clause], [Subject] + [VP]"
            m_trans = re.match(r"^(however|nevertheless|nonetheless|conversely|in contrast)\b", text_lower)
            if m_trans:
                return f"{m_trans.group(1).capitalize()}, [Subject] + [Modal/VP]"
            if re.search(r"\b,\s*but\s+[a-z]+\s+(also|still|yet)\b", text_lower):
                return "[Subject] + [VP], but [Subject] + [also] + [VP]"

        # 3. Cohesion & Framing
        if cat == "Cohesion & Framing" or not cat:
            # Propositional Encapsulation: , which + [interpretive verb] + that
            interpretive_verbs = {"mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"}
            for token in doc:
                if token.dep_ in ("relcl", "advcl") and token.lemma_.lower() in interpretive_verbs:
                    if any(c.dep_ in ("nsubj", "nsubjpass") and c.text.lower() == "which" for c in token.children):
                        return f", which + {token.lemma_.lower()}s + that + [Proposition Clause]"
            # Shell noun frame
            shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "evidence", "assumption"}
            for token in doc:
                if token.lemma_.lower() in shell_nouns:
                    if any(
                        c.dep_ in ("acl", "appos", "ccomp")
                        and any(gc.dep_ == "mark" and gc.lower_ in ("that", "whether") for gc in c.children)
                        for c in token.children
                    ):
                        return f"The + {token.lemma_.lower()} + that/of + [Proposition Clause]"

        # 4. Information Packaging
        if cat == "Information Packaging" or not cat:
            # Correlative comparative: The more..., the more...
            COMPARATIVE_WORDS = r"(?:more|less|fewer|better|worse|[a-z]{2,}er)"
            if re.search(rf"\bthe\s+{COMPARATIVE_WORDS}\b(?!\s+hand\b).*?(?:,\s*|\band\s+the\s+{COMPARATIVE_WORDS}\b.*?,?\s*)\bthe\s+{COMPARATIVE_WORDS}\b", text_lower):
                return "The + [comparative] + [Clause], the + [comparative] + [Clause]"
            # Dense Prepositional Frame (Instead of, By doing, Thanks to...)
            m_prep = re.match(r"^(instead of|by|through|despite|in spite of|thanks to|due to|owing to)\b", text_lower)
            if m_prep:
                raw_prep = m_prep.group(1).lower()
                if raw_prep != "by" or re.match(r"^by\s+(?:not\s+)?[a-z]+ing\b", text_lower):
                    prep_head = m_prep.group(1).capitalize()
                    return f"{prep_head} + [V-ing/NP], [Subject] + [VP]"
            # Non-finite Subject Nominalization (Gerund / Infinitive phrase as Subject)
            # Detects direct sentence subject or embedded predicate clause subject (e.g. That is why [being on wheels] means...)
            for token in doc:
                is_gerund_subj = (token.tag_ == "VBG" and token.dep_ in ("nsubj", "nsubjpass", "csubj"))
                is_infinitive_subj = (token.tag_ == "VB" and token.dep_ == "csubj")
                is_embedded_gerund = (
                    token.tag_ == "VBG" and token.dep_ in ("advcl", "csubj")
                    and (
                        any(c.dep_ == "advmod" and c.lemma_.lower() in ("why", "how", "what", "where") for c in token.children)
                        or (
                            token.head.pos_ in ("VERB", "AUX")
                            and not any(c.dep_ in ("nsubj", "nsubjpass") for c in token.head.children)
                            and token.i < token.head.i
                        )
                    )
                )
                if is_gerund_subj or is_embedded_gerund:
                    if re.search(r"\b(that|this)\s+(?:is|’s|'s)\s+why\b", text_lower):
                        return "[Subject] + is why + [V-ing / Gerund Phrase] + [VP]"
                    return "[V-ing / Gerund Phrase] + [Predicate Verb] + [Complement/Object]"
                if is_infinitive_subj:
                    return "To + [Infinitive Phrase] + [Predicate Verb] + [Complement/Object]"
            # Dummy-It Object Extraposition (find/make/think it adj to-V)
            for t in doc:
                if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                    for c in t.children:
                        if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                            return f"[Subject] + {t.lemma_} + it + [{c.lemma_.capitalize()}] + to-V"
            # Dense Object Complement / Causative (make/find/render/keep + Object + Adj)
            for t in doc:
                if t.lemma_ in ("make", "find", "render", "keep") and t.pos_ in ("VERB", "AUX"):
                    for c in t.children:
                        if c.dep_ in ("ccomp", "oprd") and c.pos_ == "ADJ":
                            return f"[Subject] + {t.lemma_} + [Object] + [Adj]"
            # Dummy-It Subject Extraposition
            _AMBIENT_PREDS = {
                "daylight", "dark", "night", "morning", "afternoon", "evening", "noon", "midnight",
                "dawn", "dusk", "winter", "summer", "spring", "autumn", "fall", "cold", "hot", "warm",
                "cool", "sunny", "rainy", "snowy", "windy", "cloudy", "late", "early", "o'clock"
            }
            if any(t.text.lower() == "it" and t.dep_ in ("expl", "nsubj") for t in doc):
                for t in doc:
                    if t.text.lower() == "it" and t.head.lemma_ in ("be", "seem") and not any(c.dep_ == "advmod" and c.lower_ in ("when", "where") for c in t.head.children):
                        eval_children = [
                            c for c in t.head.children
                            if c.dep_ in ("acomp", "attr") and c.lemma_.lower() not in _AMBIENT_PREDS
                        ]
                        comp_children = [
                            c for c in t.head.children
                            if c.dep_ in ("ccomp", "csubj", "xcomp")
                            and c.tag_ not in ("VBG", "VBN")
                            and not any(gc.dep_ == "advmod" and gc.lower_ in ("when", "where") for gc in c.children)
                        ]
                        if eval_children and comp_children and comp_children[0].i > eval_children[0].i:
                            return "It + [be] + [Evaluative Adj] + to-V/that + [Clause]"
            # Participial adjunct (strictly requiring non-subject participle, comma boundary when post-positioned, verbal dependents, and actual main subject)
            for token in doc:
                if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                    if token.i > token.head.i and not any(t.text == "," for t in doc[token.head.i : token.i]):
                        continue
                    if token.i + 1 < len(doc) and doc[token.i + 1].pos_ in ("NOUN", "PROPN") and doc[token.i + 1].dep_ in ("dobj", "pobj", "nsubj", "attr"):
                        continue
                    if not any(c.dep_ in ("dobj", "prep", "advmod", "acomp", "xcomp", "ccomp", "prt", "agent", "npadvmod", "oprd") for c in token.children):
                        continue
                    if not any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children) and token.lemma_.lower() not in ("include", "accord", "regard"):
                        # Ensure main verb actually has a subject, otherwise it might be a gerund subject misclassified as advcl
                        main_has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in token.head.children)
                        if main_has_subj:
                            is_fronted = token.i < token.head.i
                            has_having = any(c.dep_ in ("aux", "auxpass") and c.lower_ == "having" for c in token.children)
                            has_being = any(c.dep_ in ("aux", "auxpass") and c.lower_ == "being" for c in token.children)
                            if has_having:
                                v_type = "Having + V-ed"
                            elif has_being:
                                v_type = "Being + V-ed"
                            elif token.tag_ == "VBG":
                                v_type = "V-ing"
                            else:
                                v_type = "V-ed"
                            if is_fronted:
                                return f"[{v_type} Phrase], [Subject] + [VP]"
                            else:
                                return f"[Subject] + [VP], [{v_type} Phrase]"
            # Relative clauses (which, that, who, whom, whose)
            for token in doc:
                if token.dep_ == "relcl":
                    rel_pron = next((c.text.lower() for c in token.children if c.text.lower() in ("which", "that", "who", "whom", "whose")), None)
                    if rel_pron:
                        return f"[NP] + {rel_pron} + [VP]"
            # Elaborative clause
            if re.search(r",\s*which\s+[a-z]+", text_lower):
                return "[Subject] + [VP], which + [VP]"

            # Passive voice
            for token in doc:
                if token.dep_ in ("nsubjpass", "auxpass") and token.head.lemma_.lower() not in {
            "go", "come", "fall", "rise", "arrive", "depart", "vanish", "disappear", "die", "remain", "stay",
            "finish", "done", "tired", "worried", "interested", "bored", "excited", "married", "engaged",
            "lost", "prepared", "convinced", "pleased", "satisfied", "embarrassed", "confused", "scared",
            "shocked", "surprised", "dressed", "located", "situated", "based", "involved", "related",
            "connected", "accustomed", "opposed", "dedicated", "devoted", "committed", "crowded", "filled",
            "covered", "equipped"
        }:
                    return "[Subject] + [be/modal be] + [V-ed]"

        # Default fallback standard formula
        return "[Subject] + [VP] + [Clause]"

    @classmethod
    def mine_grammar_skeletons(
        cls,
        sentence_pool: Dict[str, str],
        target_count: int = 5,
        syllabus_grammar: Optional[List[str]] = None,
        cefr_band: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Deterministically mines genuine, non-trivial grammar structural skeletons from the sentence pool.
        Selects sentences exhibiting rich dependency structures across the Four Macro Domains,
        prioritized by the 3-band CEFR priority sequence (A1_A2, B1_B2, C1_C2) and intra-pattern complexity.
        Returns a list of dicts: [{'sid': 'S-9', 'quote': '...', 'category': '...', 'pattern_formula': '...'}]
        """
        raw_skeletons: List[Dict[str, Any]] = []
        band = cefr_band or cls.resolve_grammar_cefr_band(sentence_pool)

        nlp = cls.get_spacy()
        from librarian.grammar_patterns import GrammarPatternEngine
        gp_engine = GrammarPatternEngine.get_engine(nlp)

        for sid, sent in sentence_pool.items():
            sent_clean = sent.strip()
            # Skip very short sentences or editorial bylines
            if len(sent_clean.split()) < 7 or cls._is_byline_or_meta_line(sent_clean):
                continue
            # Skip passage title (S-1) when passage has multiple sentences, section headers, and syllabus word-lists
            if sid == "S-1" and len(sentence_pool) > 1:
                continue
            # If sentence does not end with standard sentence punctuation, skip (headings/titles)
            if not sent_clean.endswith((".", "!", "?", '"', "'", "”", "’")):
                continue
            # If sentence is a colon-separated headline/title (e.g. "Title: Subtitle") without terminal punctuation
            if ":" in sent_clean and not any(sent_clean.endswith(p) for p in (".", "!", "?")):
                continue

            sent_norm = sent_clean.replace('\u2019', "'").replace('\u2018', "'")
            doc = nlp(sent_norm)
            matched_patterns = gp_engine.match_sentence(doc)

            if matched_patterns:
                for match_item in matched_patterns:
                    cat = match_item["category"]
                    formula = match_item.get("formula", "")
                    if cat and formula and not formula.startswith("[Subject] + [VP] + [Clause]") and "[" in formula:
                        raw_skeletons.append({
                            "sid": sid,
                            "quote": sent_clean,
                            "category": cat,
                            "pattern_formula": formula,
                        })
            else:
                cat = cls.classify_grammar_dependency(sent_clean)
                cand_cats = [cat] if cat else []
                for extra_c in ("Rhetoric & Emphasis", "Logic & Stance", "Cohesion & Framing", "Information Packaging"):
                    if extra_c not in cand_cats:
                        cand_cats.append(extra_c)
                for c_try in cand_cats:
                    formula = cls.generate_cobuild_formula(sent_clean, category=c_try) if c_try else ""
                    if c_try and formula and not formula.startswith("[Subject] + [VP] + [Clause]") and "[" in formula:
                        raw_skeletons.append({
                            "sid": sid,
                            "quote": sent_clean,
                            "category": c_try,
                            "pattern_formula": formula,
                        })

        if not raw_skeletons:
            return []

        # Score each skeleton using 3-band CEFR priority + Intra-Pattern Syntactic Complexity
        def _score_skeleton(item: Dict[str, Any]) -> Tuple[float, int]:
            formula_str = item["pattern_formula"]
            quote_str = item["quote"]
            cat_str = item["category"]
            prof = cls.match_cobuild_grammar_profile(formula_str, quote_str, category=cat_str, cefr_band=band)
            base_prio = float(prof.get("priority", 70))
            prof_key = prof.get("profile_key", "")

            words = quote_str.split()
            w_cnt = len(words)
            q_low = quote_str.lower()
            f_low = formula_str.lower()

            bonus = 0.0
            # Perfective / passive participle or coordinated non-finite clauses boost
            if "having + v-ed" in f_low or "being + v-ed" in f_low or re.search(r"\bhaving\s+[a-z]+ed\b", q_low):
                bonus += 10.0
            # Optimal pedagogical sentence length boost (14-38 words)
            if 14 <= w_cnt <= 38:
                bonus += 6.0
            elif w_cnt < 10:
                bonus -= 8.0
            elif w_cnt > 48:
                bonus -= 5.0

            # In B1_B2 and C1_C2, penalize bare restrictive relative clauses (that/who/which without pied-piped prepositions)
            # so non-finite participial adjuncts, clefts, inversions, and degree results win the macro-domain slots
            if band in ("B1_B2", "C1_C2") and prof_key == "relative_clause_restrictive":
                has_pied_piping = bool(re.search(r"\b(?:in|on|at|from|to|with|by|through|for|of|under|beyond|within|upon)\s+(?:which|whom)\b", q_low))
                if has_pied_piping:
                    bonus += 12.0
                else:
                    bonus -= 15.0

            sid_m = re.search(r"\d+", item.get("sid", "999"))
            sid_num = int(sid_m.group(0)) if sid_m else 999
            return (-round(base_prio + bonus, 2), sid_num)

        ranked_skeletons = sorted(raw_skeletons, key=_score_skeleton)

        # Diverse selection algorithm across macro domains, distinct formulas, and strictly unique quotes
        def _norm_structure(formula_str: str) -> str:
            res = re.sub(
                r"\[NP\]\s*(?:\+|,\s*)\s*(?:who|which|that|whom|whose)\s*\+\s*\[VP\]",
                "[NP] + rel_pron + [VP]",
                formula_str,
                flags=re.IGNORECASE
            )
            res = re.sub(
                r"^(?:if|unless|as long as|provided that)\s*\+\s*\[Clause\]",
                "[Condition] + [Clause]",
                res,
                flags=re.IGNORECASE
            )
            res = re.sub(
                r"^\[(?:Having \+ V-ed|Being \+ V-ed|V-ing|V-ed) Phrase\],\s*\[Subject\] \+ \[VP\]",
                "[Participial Phrase], [Subject] + [VP]",
                res,
                flags=re.IGNORECASE
            )
            return res

        selected: List[Dict[str, str]] = []
        seen_cats: Set[str] = set()
        seen_structures: Set[str] = set()
        seen_profiles: Set[str] = set()
        seen_quotes: Set[str] = set()

        def _get_prof_key(item: Dict[str, str]) -> str:
            prof = cls.match_cobuild_grammar_profile(
                item["pattern_formula"], item["quote"], category=item["category"], cefr_band=band
            )
            return str(prof.get("profile_key") or "")

        # Pass 1: maximize category diversity across the 4 macro domains (highest priority first)
        for item in ranked_skeletons:
            cat = item["category"]
            quote = item["quote"]
            struct = _norm_structure(item["pattern_formula"])
            pkey = _get_prof_key(item)
            if quote not in seen_quotes and cat not in seen_cats and struct not in seen_structures and pkey not in seen_profiles:
                selected.append(item)
                seen_cats.add(cat)
                seen_structures.add(struct)
                if pkey:
                    seen_profiles.add(pkey)
                seen_quotes.add(quote)
            if len(selected) >= target_count:
                break

        # Pass 2: fill remaining slots with distinct structural formulas, distinct profiles, and unique quotes
        if len(selected) < target_count:
            for item in ranked_skeletons:
                quote = item["quote"]
                struct = _norm_structure(item["pattern_formula"])
                pkey = _get_prof_key(item)
                if quote not in seen_quotes and struct not in seen_structures and pkey not in seen_profiles:
                    selected.append(item)
                    seen_structures.add(struct)
                    if pkey:
                        seen_profiles.add(pkey)
                    seen_quotes.add(quote)
                if len(selected) >= target_count:
                    break

        # Pass 3: if still under target_count, append remaining distinct quotes with unused profiles
        if len(selected) < target_count:
            for item in ranked_skeletons:
                quote = item["quote"]
                pkey = _get_prof_key(item)
                if quote not in seen_quotes and pkey not in seen_profiles:
                    selected.append(item)
                    seen_quotes.add(quote)
                    if pkey:
                        seen_profiles.add(pkey)
                if len(selected) >= target_count:
                    break
        # Pass 3b: allow a second pattern from an already-used sentence ONLY if its profile is completely unused
        if len(selected) < target_count:
            for item in ranked_skeletons:
                pkey = _get_prof_key(item)
                if pkey and pkey not in seen_profiles and item not in selected:
                    selected.append(item)
                    seen_profiles.add(pkey)
                if len(selected) >= target_count:
                    break

        # Restore natural passage appearance order (by sentence ID) for coherent reading flow
        def _sid_order(item: Dict[str, str]) -> int:
            m = re.search(r"\d+", item.get("sid", "999"))
            return int(m.group(0)) if m else 999

        selected.sort(key=_sid_order)
        return selected[:target_count]


    # =========================================================================
    _grammar_profiles_cache: Optional[Dict[str, Any]] = None
    _grammar_profiles_mtime: float = 0.0

    @classmethod
    def get_grammar_profiles_data(cls) -> Dict[str, Any]:
        """
        Loads and hot-reloads `librarian/data/grammar_profiles.json` using `st_mtime`.
        Decouples all 3-band priority weights, pedagogical functions, imitation examples,
        and common mistakes from Python source code.
        """
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "grammar_profiles.json")
        try:
            mtime = os.stat(json_path).st_mtime
            if cls._grammar_profiles_cache is not None and mtime == cls._grammar_profiles_mtime:
                return cls._grammar_profiles_cache
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cls._grammar_profiles_cache = data
            cls._grammar_profiles_mtime = mtime
            return data
        except Exception as e:
            logger.warning(f"Failed to load grammar_profiles.json ({e}); using cached or empty fallback.")
            return cls._grammar_profiles_cache or {"profiles": {}, "fallbacks": {}}

    @classmethod
    def resolve_grammar_cefr_band(cls, text_or_pool: Union[str, Dict[str, str], List[str]]) -> str:
        """
        Deterministically resolves a passage's grammar pedagogical band ('A1_A2', 'B1_B2', or 'C1_C2')
        using `calculate_text_cefr` and lexical sophistication metrics (`b2_plus_ratio`, `c1_plus_ratio`, `avg_cefr`).
        """
        if isinstance(text_or_pool, dict):
            full_text = " ".join(text_or_pool.values())
        elif isinstance(text_or_pool, list):
            full_text = " ".join(text_or_pool)
        else:
            full_text = str(text_or_pool or "")

        if not full_text.strip():
            return "B1_B2"

        cefr = cls.calculate_text_cefr(full_text)
        if cefr in ("A1", "A2"):
            return "A1_A2"
        if cefr in ("C1", "C2"):
            return "C1_C2"

        try:
            analyzer = cls.get_cefr_analyzer()
            words = re.findall(r"[a-zA-Z]+", full_text.lower())
            levels: List[float] = []
            for w in words:
                lvl = analyzer.get_average_word_level_float(w)
                if lvl is not None:
                    levels.append(float(lvl))
            if levels:
                n = len(levels)
                avg_cefr = sum(levels) / n
                b2_plus_ratio = sum(1 for x in levels if x >= 3.5) / n
                c1_plus_ratio = sum(1 for x in levels if x >= 4.5) / n
                if cefr == "B1" and avg_cefr < 1.75 and b2_plus_ratio < 0.088:
                    return "A1_A2"
                if cefr == "B2" and (avg_cefr >= 1.93 or (c1_plus_ratio >= 0.055 and avg_cefr >= 1.90)):
                    return "C1_C2"
        except Exception:
            pass

        return "B1_B2"

    @classmethod
    def match_cobuild_grammar_profile(
        cls,
        formula: str,
        quote: str,
        category: str = "",
        cefr_band: str = "B1_B2"
    ) -> Dict[str, Any]:
        """
        Deterministically matches a mined syntax formula and quote to the most specific
        COBUILD Grammar Profile from `librarian/data/grammar_profiles.json` and resolves
        the target CEFR band's (`A1_A2`, `B1_B2`, `C1_C2`) priority, pedagogical function,
        imitation example, and common mistakes (0 Token Cost).
        """
        band = cefr_band if cefr_band in ("A1_A2", "B1_B2", "C1_C2") else "B1_B2"
        data = cls.get_grammar_profiles_data()
        profiles = data.get("profiles", {})
        fallbacks = data.get("fallbacks", {})

        def _resolve_profile(profile_key: str, fallback_cat: str = "") -> Dict[str, Any]:
            raw_prof = profiles.get(profile_key)
            if not raw_prof:
                return _resolve_fallback(fallback_cat or category)
            tiers = raw_prof.get("tiers", {})
            tier_data = tiers.get(band) or tiers.get("B1_B2") or tiers.get("A1_A2") or {}
            prio_map = raw_prof.get("priority", {})
            prio_val = int(prio_map.get(band, prio_map.get("B1_B2", 70)))
            return {
                "profile_key": profile_key,
                "name": raw_prof.get("name", "Syntactic Pattern"),
                "category": raw_prof.get("category", category or "Information Packaging"),
                "pattern_formula": raw_prof.get("pattern_formula", formula),
                "priority": prio_val,
                "pedagogical_function": tier_data.get("pedagogical_function", ""),
                "imitation_example": tier_data.get("imitation_example", quote),
                "common_mistakes": tier_data.get("common_mistakes", ""),
                "variants": tier_data.get("variants", [])
            }

        def _resolve_fallback(cat_name: str) -> Dict[str, Any]:
            c_low = (cat_name or "").lower()
            if c_low == "rhetoric & emphasis":
                fb_key = "Rhetoric & Emphasis"
            elif c_low == "logic & stance":
                fb_key = "Logic & Stance"
            elif c_low == "cohesion & framing":
                fb_key = "Cohesion & Framing"
            else:
                fb_key = "Information Packaging"
            raw_fb = fallbacks.get(fb_key, {})
            tiers = raw_fb.get("tiers", {})
            tier_data = tiers.get(band) or tiers.get("B1_B2") or tiers.get("A1_A2") or {}
            prio_map = raw_fb.get("priority", {})
            prio_val = int(prio_map.get(band, 45))
            return {
                "profile_key": f"fallback_{fb_key.lower().replace(' ', '_')}",
                "name": raw_fb.get("name", f"{fb_key} Pattern"),
                "category": fb_key,
                "pattern_formula": formula,
                "priority": prio_val,
                "pedagogical_function": tier_data.get("pedagogical_function", f"Synthesizes discourse via {fb_key.lower()} syntax."),
                "imitation_example": tier_data.get("imitation_example", quote),
                "common_mistakes": tier_data.get("common_mistakes", "Grammar Warning (Syntax): Learners frequently misapply structural boundaries."),
                "variants": tier_data.get("variants", [])
            }

        f_lower = (formula or "").lower()
        q_lower = (quote or "").lower()
        c_lower = (category or "").lower()

        # 0. Explicit formula-first routing so formula and profile never diverge when a quote contains multiple clauses
        if any(f_lower.startswith(p) for p in ("had + [subject]", "were + [subject]", "should + [subject]")) or re.match(r"^(?:had|were|should)\s+(?:i|you|he|she|it|we|they|the\b|a\b|an\b|this\b|that\b|these\b|those\b|[a-z]+)\s+(?:not\s+)?(?:been|had|known|done|to\b|[a-z]+ed\b)", q_lower.strip()):
            if "inverted_conditional_subjunctive" in profiles:
                return _resolve_profile("inverted_conditional_subjunctive")
        if "negative/restrictive" in f_lower or "inversion" in f_lower or any(f_lower.startswith(p) for p in ("never +", "rarely +", "seldom +", "little +", "only +", "not only + [aux", "hardly +", "scarcely +", "no sooner +", "under no circumstances", "in no way", "at no time", "on no account")):
            return _resolve_profile("fronted_negative_inversion")
        if f_lower.startswith("with + [np]"):
            return _resolve_profile("with_absolute_adjunct")
        if f_lower.startswith("[np] + [v-ing") or f_lower.startswith("[np] + [v-ed") or "nominative absolute" in f_lower:
            if "nominative_absolute" in profiles:
                return _resolve_profile("nominative_absolute")
        if any(f_lower.startswith(p) for p in ("although +", "though +", "even though +", "even if +", "while +", "whereas +", "despite +", "in spite of +", "much as +", "albeit +", "notwithstanding +")):
            return _resolve_profile("subordinate_concessive")
        if any(f_lower.startswith(p) for p in ("because +", "since +", "because/since", "because/since/as")):
            if "explanatory_causal_clause" in profiles:
                return _resolve_profile("explanatory_causal_clause")
        if "inanimate" in f_lower or "enable/allow/permit" in f_lower or "[inanimate" in f_lower:
            if "enable" in f_lower or "allow" in f_lower or "causation" in f_lower:
                if "inanimate_causation_enablement" in profiles:
                    return _resolve_profile("inanimate_causation_enablement")
            if "inanimate_subject_packaging" in profiles:
                return _resolve_profile("inanimate_subject_packaging")
        if any(f_lower.startswith(p) for p in ("as long as +", "provided that +", "on condition that +", "so long as +")):
            return _resolve_profile("conditional_clause_as_long_as")
        if any(f_lower.startswith(p) for p in ("if +", "once +", "unless +", "if/unless +")):
            return _resolve_profile("conditional_clause_if")
        if ("so + [" in f_lower or "such + [" in f_lower) and "that" in f_lower:
            return _resolve_profile("resultative_so_such_that")
        if "the + [comparative]" in f_lower:
            return _resolve_profile("proportional_comparative")
        if "as + [adj/adv] + as" in f_lower or ("as + [" in f_lower and "+ as" in f_lower):
            return _resolve_profile("correlative_comparative_as_as")
        if "by + [v-ing" in f_lower:
            return _resolve_profile("prepositional_means_by_ving")
        if any(k in f_lower for k in ("parataxis", "verb-stack", "verb stack", "and [v3]", "and v3")):
            if "parataxis_verb_stacking" in profiles:
                return _resolve_profile("parataxis_verb_stacking")
        if "[v-ing / gerund phrase]" in f_lower or "[v-ing phrase] + [v]" in f_lower:
            return _resolve_profile("gerund_subject_nominalization")
        if "[subject] + [vp], [" in f_lower and ("v-ing" in f_lower or "having +" in f_lower):
            return _resolve_profile("participial_adjunct_post_ving")
        if "[subject] + [vp], [" in f_lower and ("v-ed" in f_lower or "being + v-ed" in f_lower):
            return _resolve_profile("participial_adjunct_post_ved")
        if any(p in f_lower for p in ("[v-ing phrase]", "[v-ed phrase]", "[having + v-ed phrase]", "[being + v-ed phrase]")):
            return _resolve_profile("participial_adjunct")
        if "passive" in f_lower or ("[be/modal be]" in f_lower and "[v-ed]" in f_lower):
            return _resolve_profile("passive_voice_agentless")
        if any(p in f_lower for p in ("in order to +", "so as to +", "to + [vp], [subject]")):
            return _resolve_profile("purpose_infinitive_adjunct")
        if f_lower.startswith("to + [vp] + [v"):
            if "infinitive_subject_nominalization" in profiles:
                return _resolve_profile("infinitive_subject_nominalization")
        if "which + [interpretive" in f_lower or f_lower.startswith(", which +"):
            if "propositional_encapsulation_which" in profiles:
                return _resolve_profile("propositional_encapsulation_which")
            return _resolve_profile("relative_clause_elaborative")
        if "[np], which" in f_lower or "[np], who" in f_lower or "[np] , which" in f_lower or "[np] , who" in f_lower:
            return _resolve_profile("relative_clause_elaborative")
        if "shell noun" in f_lower or "that/of + [proposition" in f_lower or "fact + that" in f_lower or "[abstract np] + that" in f_lower:
            return _resolve_profile("appositive_complement_that")
        if "focal element" in f_lower or "cleft" in f_lower:
            return _resolve_profile("cleft_focus")
        if "evaluative adj" in f_lower or "it + [be] + [evaluative" in f_lower or "it is + [adj" in f_lower or "it was + [adj" in f_lower or "dummy_it" in f_lower:
            return _resolve_profile("dummy_it_extraposition")
        if any(p in f_lower for p in ("[np] + which", "[np] + that", "[np] + who", "[np] + whom", "[np] + whose", "[np] + [prep] + which", "[np] + [prep] + whom")):
            return _resolve_profile("relative_clause_restrictive")

        # 1. Cleft Focus vs Dummy-It Extraposition (check extraposition predicates FIRST even if formula says Focal Element)
        if re.search(r"\bit(?:'s|\s+(?:was|is|has\s+been|seemed|seems|appeared|appears|turned\s+out|proved|would\s+be|could\s+be))\s+(?:no\s+|not\s+a\s+|little\s+|hardly\s+a\s+)?(?:clear|obvious|important|necessary|essential|vital|critical|imperative|crucial|possible|impossible|likely|unlikely|certain|true|evident|apparent|plain|notable|remarkable|worth\s+noting|fair|safe|reasonable|natural|normal|common|rare|strange|odd|ironic|paradoxical|surprising|unsurprising|astonishing|shocking|sad|tragic|unfortunate|regrettable|fortunate|lucky|good|better|best|wise|foolish|easy|hard|difficult|simple|complex|challenging|said|reported|believed|thought|assumed|estimated|expected|known|understood|agreed|accepted|argued|suggested|claimed|noted|observed|found|shown|proven|established|demonstrated|revealed|discovered|decided|arranged|recommended|proposed|demanded|required|requested|urged|advised|surprise|wonder|secret|doubt|coincidence|accident|pity|shame|mystery|miracle|comfort|relief|consolation|honor|honour|privilege|pleasure|joy|delight|tragedy|disgrace|scandal|fact|matter|question|issue|problem|challenge|duty|responsibility|task|job|custom|tradition|habit|rule|policy|practice|mistake|error|fault|crime|sin|blessing|curse|waste|use|good|point)\b", q_lower):
            return _resolve_profile("dummy_it_extraposition")
        if "focal element" in f_lower or "cleft" in f_lower:
            return _resolve_profile("cleft_focus")
        if re.search(r"\bit(?:'s|\s+(?:was|is))\s+.*?\b(?:that|who)\b", q_lower):
            return _resolve_profile("cleft_focus")

        # 2. Correlative Parallelism: not only... but also
        if "not only" in f_lower or ("not only" in q_lower and ("but also" in q_lower or "but " in q_lower)):
            if re.search(r"(?:^|[,;:]\s*)not\s+only\s+(?:did|do|does|had|have|has|was|were|is|are|can|could|will|would|should|must|may|might)\b", q_lower.strip()):
                return _resolve_profile("fronted_negative_inversion")
            return _resolve_profile("correlative_parallelism")

        # 2b. Antithesis Coordination: not... but... / instead of / rather than
        if "instead of +" in f_lower or "rather than +" in f_lower:
            if "prepositional_contrast_instead_of" in profiles:
                return _resolve_profile("prepositional_contrast_instead_of")
        if "not + [prepp/np], but" in f_lower or re.search(r",\s*not\b.*?\bbut\b", q_lower) or ("not + [" in f_lower and "but +" in f_lower):
            return _resolve_profile("antithesis_parallelism")

        # 3a. Causative bare infinitive: make/have/let + Object + [Bare Infinitive]
        if "bare infinitive" in f_lower:
            return _resolve_profile("causative_bare_infinitive")

        # 3b. Dummy-It Object Extraposition: make/find/think/consider + it + [Adj/NP] + to/that
        if "+ it + [" in f_lower and ("to + [vp]" in f_lower or "that + [clause]" in f_lower):
            if "dummy_it_object" in profiles:
                return _resolve_profile("dummy_it_object")

        # 3c. Object Complement / Complex Transitive (Noun Complement): make + Object + [Noun/NP]
        if ("make +" in f_lower or "find +" in f_lower or "appoint +" in f_lower or "consider +" in f_lower) and "[noun/np]" in f_lower:
            return _resolve_profile("object_complement_noun")

        # 3d. Object Complement / Complex Transitive (Adjective Complement): make/find/keep + Object + [Adj]
        if ("make +" in f_lower or "render +" in f_lower or "find +" in f_lower or "keep +" in f_lower or "leave +" in f_lower or "deem +" in f_lower) and ("[adj]" in f_lower or " + it + [" in f_lower or "[object]" in f_lower):
            return _resolve_profile("object_complement_adj")
        if re.search(r"\bmake\b.*?\b(?:possible|impossible|easy|clear|accessible|redundant)\b", q_lower):
            return _resolve_profile("object_complement_adj")

        # 4. Causative bare infinitive fallback: make/have/let + sb + do
        if "make + [" in f_lower or "let + [" in f_lower:
            return _resolve_profile("causative_bare_infinitive")

        # 5. Shell Noun / Appositive That-Clause: [Shell Noun] + that + [Clause]
        if "shell noun" in f_lower or "fact + that" in f_lower or "[abstract np] + that" in f_lower or re.search(r"\bthe\s+(?:fact|idea|belief|notion|view|possibility|chance|hope|fear|news|evidence|proof|assumption|hypothesis|premise|paradox|irony|conviction|realization|consensus|contention|proposition|myth|delusion|impression)\s+that\b", q_lower):
            return _resolve_profile("appositive_complement_that")

        # 5b. Participial Adjuncts (differentiating post-positioned V-ed vs V-ing, checked before relative clauses)
        if "[subject] + [vp], [" in f_lower and ("v-ing" in f_lower or "having +" in f_lower):
            return _resolve_profile("participial_adjunct_post_ving")
        if "[subject] + [vp], [" in f_lower and ("v-ed" in f_lower or "being + v-ed" in f_lower):
            return _resolve_profile("participial_adjunct_post_ved")
        if any(p in f_lower for p in ("[v-ing phrase]", "[v-ed phrase]", "[having + v-ed phrase]", "[being + v-ed phrase]")) or "particip" in c_lower:
            return _resolve_profile("participial_adjunct")

        # 6. Non-restrictive / Sentential Relative Clauses
        if "which + [interpretive" in f_lower or re.search(r",\s*which\s+(?:means|suggests|indicates|shows|implies|demonstrates|proves|reveals|reflects|underscores|highlights|explains|makes|leaves|renders|allows|enables|prevents|leads|results|contributes)\b", q_lower):
            if "propositional_encapsulation_which" in profiles:
                return _resolve_profile("propositional_encapsulation_which")
            return _resolve_profile("relative_clause_elaborative")
        if (", which" in q_lower and "fact + that" not in f_lower and "[shell noun]" not in f_lower) or "[np], which" in f_lower or "[np], who" in f_lower or "[np] , which" in f_lower or "[np] , who" in f_lower:
            return _resolve_profile("relative_clause_elaborative")

        # 7. Restrictive & Human Relative Clauses: [NP] + which/that/who + [VP] or Prep + which/whom
        if any(p in f_lower for p in ("[np] + which", "[np] + that", "[np] + who", "[np] + whom", "[np] + whose", "[np] + [prep] + which", "[np] + [prep] + whom")):
            return _resolve_profile("relative_clause_restrictive")

        # 8. Proportional Comparative & Equative Comparative (as ... as)
        if "the + [comparative]" in f_lower:
            return _resolve_profile("proportional_comparative")
        if "as + [adj/adv] + as" in f_lower or ("as + [" in f_lower and "+ as" in f_lower):
            return _resolve_profile("correlative_comparative_as_as")

        # 8b. Resultative Degree Constructions (so ... that / such ... that)
        if ("so + [" in f_lower or "such + [" in f_lower) and "that" in f_lower:
            return _resolve_profile("resultative_so_such_that")

        # 9. Dummy-It Extraposition: It is adj to/that
        if "it + [be] + [adj" in f_lower or "it is + [adj" in f_lower or "dummy" in f_lower:
            return _resolve_profile("dummy_it_extraposition")

        # 10. Purpose Infinitive Adjuncts (In order to / So as to / To + [VP]) & Infinitive Subject
        if any(p in f_lower for p in ("in order to +", "so as to +", "to + [vp], [subject]")):
            return _resolve_profile("purpose_infinitive_adjunct")
        if f_lower.startswith("to + [vp] + [v"):
            if "infinitive_subject_nominalization" in profiles:
                return _resolve_profile("infinitive_subject_nominalization")

        # 12. Gerund Subject Nominalization & Prepositional Means
        if "by + [v-ing" in f_lower or q_lower.startswith("by "):
            return _resolve_profile("prepositional_means_by_ving")
        if "[v-ing / gerund phrase]" in f_lower or "[v-ing phrase] + [v]" in f_lower:
            return _resolve_profile("gerund_subject_nominalization")

        # 13. Adversative & Concession
        if "but [subject] + [also]" in f_lower or ", but" in q_lower:
            return _resolve_profile("adversative_concession")
        if ", while + " in f_lower or ", whereas + " in f_lower or (any(w in q_lower for w in (", while ", "; while ", ", whereas ", "; whereas ")) and not q_lower.startswith(("while ", "whereas ", "but while "))):
            return _resolve_profile("subordinate_contrastive_while")
        if any(w in f_lower for w in ("although", "though", "while", "even if", "whereas")) or any(q_lower.startswith(w) for w in ("although ", "though ", "even though ", "even if ", "while ", "whereas ")):
            return _resolve_profile("subordinate_concessive")

        # 14. Discourse transition: However / Nevertheless / Nonetheless / Conversely / Instead
        if any(q_lower.startswith(w) or w in f_lower for w in ("however", "nevertheless", "nonetheless", "conversely", "in contrast", "instead,")):
            return _resolve_profile("discourse_transition_however")

        # 15. Causal & Explanatory Clauses (because / since / as / now that)
        if any(w in f_lower for w in ("because", "since + [clause]", "now that", "[causal clause]", "given that", "in that", "seeing that")):
            return _resolve_profile("explanatory_causal_clause")

        # 16. Comparative Manner & Analogy Clauses (just as / as if / as though)
        if any(w in f_lower for w in ("just as", "as if", "as though", "as + [clause]")):
            return _resolve_profile("manner_similarity_clause")

        # 17. Epistemic Stance & Hedging (suggest / indicate / show / argue that)
        if "stance" in f_lower or "epistemic" in f_lower or ("[verb] + that + [clause]" in f_lower and c_lower == "logic & stance"):
            return _resolve_profile("epistemic_stance_hedging")

        # 18. Passive voice
        if "passive" in f_lower or "passive" in c_lower or ("[be/modal be]" in f_lower and "[v-ed]" in f_lower):
            return _resolve_profile("passive_voice_agentless")

        # 19. Conditional clauses (if / as long as / unless / once)
        if "as long as" in f_lower or "as long as" in q_lower or "provided that" in f_lower or "on condition that" in f_lower or "so long as" in f_lower:
            return _resolve_profile("conditional_clause_as_long_as")
        if f_lower.startswith("if +") or f_lower.startswith("once +") or f_lower.startswith("unless +") or "if/unless" in f_lower or (q_lower.startswith("if ") or ", if " in q_lower or "so, if " in q_lower or "similarly, if " in q_lower):
            return _resolve_profile("conditional_clause_if")

        # Dynamic Category-Aware Fallback from grammar_profiles.json (3-Band Calibrated)
        return _resolve_fallback(category)

    @classmethod
    def _select_profile_variant(
        cls,
        profile: Dict[str, Any],
        quote: str,
        full_text: str,
        used_imitations: Set[str],
        used_mistakes: Set[str],
        formula: str = ""
    ) -> Tuple[str, str]:
        """
        Deterministically selects the best contextually-aligned (imitation_example, common_mistakes)
        variant from a profile's thematic bank based on lexical overlap with the quote and passage text,
        and structural alignment with the pattern formula's literal anchors.
        Guarantees band-calibrated examples and prevents duplicate templates.
        """
        default_imit = profile.get("imitation_example", quote)
        default_mist = profile.get("common_mistakes", "Grammar Warning (Syntax): Learners frequently misapply structural boundaries.")
        variants = profile.get("variants")

        # Extract literal anchor groups from formula (e.g. 'Although' or 'that/to')
        anchor_groups: List[List[str]] = []
        if formula:
            cobuild_pos = {
                "np", "vp", "v", "be", "aux", "v-ed", "v3", "v-ing", "to-v", "to-inf", "inf",
                "adj", "adv", "det", "prep", "conj", "clause", "s", "n", "pron", "modal"
            }
            raw_lits = re.findall(r"[a-zA-Z0-9_\'/]+", re.sub(r"\[.*?\]|\(.*?\)", " ", formula))
            for lit in raw_lits:
                alts = [a.strip().lower() for a in lit.split("/") if a.strip()]
                valid_alts = [a for a in alts if len(a) >= 2 and a not in cobuild_pos]
                if valid_alts:
                    anchor_groups.append(valid_alts)

        f_low = (formula or "").lower()
        requires_ving = ("[v-ing]" in f_low or "[v-ing phrase]" in f_low or "[v-ing / gerund phrase]" in f_low) and ("/" not in f_low.split("[v-ing")[1].split("]")[0] or "gerund" in f_low) if "[v-ing" in f_low else False

        def _instantiates_anchors(imit_text: str) -> bool:
            if not imit_text:
                return True
            imit_low = imit_text.lower()
            if requires_ving and not re.search(r"\b(?:[a-z]{3,}ing|being|doing|going)\b", imit_low):
                return False
            if not anchor_groups:
                return True
            imit_toks = set(re.findall(r"[a-z0-9']+", imit_low))
            for alts in anchor_groups:
                if any(a in imit_toks or cls.text_contains_form(imit_low, a) for a in alts):
                    continue
                # Closed-paradigm equivalence matching evaluator.py
                if any(re.match(r"^(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?|underscor(?:es|ed))$", a) for a in alts):
                    if re.search(r"\b(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?|underscor(?:es|ed))\b", imit_low):
                        continue
                if any(a in ("although", "though", "even", "while", "whereas") for a in alts):
                    if re.search(r"\b(?:although|though|even though|even if|while|whereas)\b", imit_low):
                        continue
                if any(a in ("fact", "idea", "belief", "assumption", "notion", "claim", "possibility", "view", "evidence", "news", "impression", "argument", "thought", "hope", "fear", "doubt", "contention", "recognition", "realization", "premise", "hypothesis", "consensus", "conclusion") for a in alts):
                    if re.search(r"\b(?:fact|idea|belief|assumption|notion|claim|possibility|view|evidence|news|impression|argument|thought|hope|fear|doubt|contention|recognition|realization|premise|hypothesis|consensus|conclusion)\b", imit_low):
                        continue
                if any(a in ("however", "nevertheless", "nonetheless", "conversely") for a in alts):
                    if re.search(r"\b(?:however|nevertheless|nonetheless|conversely)\b", imit_low):
                        continue
                return False
            return True

        if not variants:
            return default_imit, default_mist

        q_words = set(re.findall(r"[a-z]{3,}", (quote or "").lower()))
        t_words = set(re.findall(r"[a-z]{3,}", (full_text or "").lower()))

        best_var = None
        best_score = -1000.0
        for idx_v, var in enumerate(variants):
            kws = set(var.get("keywords", ()))
            # Quote keyword match carries 5x weight; passage theme match carries 1x weight;
            # deterministic unit+quote CRC32 tie-breaker prevents cross-unit variant repetition when keyword overlap ties at 0.
            tie_break = (zlib.crc32(f"{(full_text or '')[:240]}|{(quote or '')[:120]}|{idx_v}".encode("utf-8")) % 997) / 10000.0
            score = float(len(kws & q_words) * 5 + len(kws & t_words)) + tie_break
            v_imit = var.get("imitation_example", default_imit)
            v_mist = var.get("common_mistakes", default_mist)
            if _instantiates_anchors(v_imit):
                score += 50
            else:
                score -= 200
            if v_imit in used_imitations or v_mist in used_mistakes:
                score -= 300
            if score > best_score:
                best_score = score
                best_var = var

        default_already_used = (default_imit in used_imitations) or (default_mist in used_mistakes)
        default_instantiates = _instantiates_anchors(default_imit)
        if best_var:
            v_imit_best = best_var.get("imitation_example", default_imit)
            v_mist_best = best_var.get("common_mistakes", default_mist)
            var_instantiates = _instantiates_anchors(v_imit_best)
            var_unused = (v_imit_best not in used_imitations) and (v_mist_best not in used_mistakes)
            if (
                best_score >= 0
                or (default_already_used and var_unused and var_instantiates)
                or (not default_instantiates and var_instantiates)
            ):
                return v_imit_best, v_mist_best
        return default_imit, default_mist

    @classmethod
    def extract_deterministic_grammar(
        cls,
        text: str,
        sentence_pool: Optional[Dict[str, str]] = None,
        target_count: int = 5,
        syllabus_grammar: Optional[List[str]] = None,
        cefr_band: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Deterministically extracts grammar patterns from the text at 0 token cost,
        calibrating pattern priority, pedagogical functions, imitation examples, and
        empirical common mistakes to the passage's 3-Band CEFR level ('A1_A2', 'B1_B2', 'C1_C2').
        """
        if not sentence_pool:
            _, sentence_pool = cls.tokenize_and_index_sentences(text)

        resolved_band = cefr_band or cls.resolve_grammar_cefr_band(text if text else sentence_pool)

        skeletons = cls.mine_grammar_skeletons(
            sentence_pool,
            target_count=target_count,
            syllabus_grammar=syllabus_grammar,
            cefr_band=resolved_band
        )
        if not skeletons:
            return []

        results: List[Dict[str, Any]] = []
        used_imitations: Set[str] = set()
        used_mistakes: Set[str] = set()
        for s in skeletons:
            sid = s.get("sid", "S-1")
            quote = s.get("quote", "")
            cat = s.get("category", "Information Packaging")
            formula = s.get("pattern_formula", "[Subject] + [VP]")

            profile = cls.match_cobuild_grammar_profile(
                formula,
                quote,
                category=cat,
                cefr_band=resolved_band
            )

            prof_name = profile.get("name", "Syntactic Pattern")
            prof_cat = profile.get("category", cat)
            if profile.get("profile_key") == "fronted_negative_inversion" and "aux" not in formula.lower():
                formula = "Not only + [Aux] + [Subject] + [VP], but + [Subject] + [also] + [VP]"
            elif profile.get("profile_key") == "dummy_it_extraposition":
                formula = profile.get("pattern_formula", formula)
            ped_func = profile.get("pedagogical_function", f"Synthesizes discourse via {prof_cat.lower()} syntax.")

            imitation, clean_mistake = cls._select_profile_variant(
                profile,
                quote=quote,
                full_text=text,
                used_imitations=used_imitations,
                used_mistakes=used_mistakes,
                formula=formula
            )

            # PedLex Authentic Grammar Alert & Don't Say diagnostics ONLY for generic fallback profiles
            try:
                STOP_WORDS_FOR_ALERTS = {"that", "not", "also", "it", "which", "this", "what", "there", "long", "like", "well", "such"}
                GENERIC_PROFILES = {
                    "Syntactic Pattern",
                    "Rhetorical Emphasis & Contrast",
                    "Logical Transition & Epistemic Stance",
                    "Discursive Cohesion & Structural Framing",
                    "Information Packaging & Syntactic Integration"
                }

                is_specific_prof = prof_name not in GENERIC_PROFILES and bool(clean_mistake)

                if not is_specific_prof:
                    formula_clean = re.sub(r'\[.*?\]|\(.*?\)', ' ', formula)
                    formula_lits = [tok.lower() for tok in re.findall(r"\b[a-zA-Z]{3,}\b", formula_clean) if tok.lower() not in STOP_WORDS_FOR_ALERTS]

                    quote_lits: List[str] = []
                    if not formula_lits:
                        quote_lits = [tok.lower() for tok in re.findall(r"\b[a-zA-Z]{3,}\b", quote)
                                      if tok.lower() in ("make", "want", "enjoy", "too", "enough", "such", "need", "own", "used", "help", "prefer", "suggest", "while", "if", "because", "unless", "since", "both", "neither")]

                    ordered_candidates = []
                    seen_cands = set()
                    for tok in formula_lits + quote_lits:
                        if tok not in seen_cands:
                            seen_cands.add(tok)
                            ordered_candidates.append(tok)

                    found_alert_pitfall = False
                    for aw in ordered_candidates:
                        alert = cls.get_ldoce_grammar_alert(aw, quote=quote)
                        if alert and ("Don't say" in alert.get("content", "") or "✗" in alert.get("content", "")):
                            raw_content = alert.get("content", "")
                            parts = [p.strip() for p in re.split(r'[•\n]+', raw_content) if p.strip()]
                            pitfalls = [p for p in parts if ("Don't say" in p or "✗" in p)]
                            for pit in pitfalls:
                                chosen = re.sub(r'\s+', ' ', pit).strip()
                                chosen = re.sub(r'^GRAMMAR:?\s*', '', chosen, flags=re.IGNORECASE).strip()
                                chosen = re.sub(r'\s*Grammar guide\s*[-‒–—].*$', '', chosen, flags=re.IGNORECASE).strip()
                                cand_text = f"Grammar Warning ({alert.get('word', '').upper()}): {chosen}"
                                if cand_text not in used_mistakes:
                                    clean_mistake = cand_text
                                    found_alert_pitfall = True
                                    break
                        if found_alert_pitfall:
                            break
            except Exception:
                pass
            used_imitations.add(imitation)
            used_mistakes.add(clean_mistake)

            results.append({
                "quote": quote,
                "pattern_formula": formula,
                "category": prof_cat,
                "syntax_topic": prof_name,
                "pedagogical_function": ped_func,
                "imitation_example": imitation,
                "common_mistakes": clean_mistake,
                "cefr_band": resolved_band,
                "design_audit": f"AUDIT: [{sid}] -> [{formula}] -> [BAND:{resolved_band}] -> [VERBATIM_CONFIRMED]"
            })

        return results
