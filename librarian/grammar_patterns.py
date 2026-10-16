"""
Declarative Grammar Pattern Definitions and DependencyMatcher Engine.
Provides a declarative syntax pattern DSL for zero-token grammar extraction.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import re
import spacy
from spacy.matcher import DependencyMatcher


# ------------------------------------------------------------------------------
# 1. Declarative Grammar Patterns DSL
# ------------------------------------------------------------------------------
# Each pattern defines:
#   - id: Unique pattern identifier
#   - category: One of the 4 Macro Domains (Rhetoric & Emphasis, Logic & Stance, Cohesion & Framing, Information Packaging)
#   - priority: Integer score (higher priority matched first when selecting representative pattern)
#   - name: Descriptive grammar name
#   - formula_template: COBUILD-style slot formula or formula generator type
#   - tree_patterns: List of dependency pattern specifications for spaCy DependencyMatcher
#   - slot_mapping: Mapping of matched node names to pedagogical slots
# ------------------------------------------------------------------------------

DECLARATIVE_GRAMMAR_PATTERNS: List[Dict[str, Any]] = [
    # =========================================================================
    # A. Rhetoric & Emphasis
    # =========================================================================
    {
        "id": "cleft_it_be_relcl",
        "category": "Rhetoric & Emphasis",
        "priority": 100,
        "name": "It-Cleft Focus Construction",
        "formula": "It + be + [Focal Element] + that/who + [Clause]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "copula", "RIGHT_ATTRS": {"LEMMA": "be"}},
                {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "dummy_it", "RIGHT_ATTRS": {"LOWER": "it", "DEP": {"IN": ["nsubj", "expl"]}}},
                {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "focal_elem", "RIGHT_ATTRS": {"DEP": {"IN": ["attr", "dobj", "pobj", "prep", "npadvmod"]}, "LOWER": {"NOT_IN": ["time"]}}},
                {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "cleft_clause", "RIGHT_ATTRS": {"DEP": {"IN": ["relcl", "ccomp", "advcl"]}}},
            ],
            [
                {"RIGHT_ID": "copula", "RIGHT_ATTRS": {"LEMMA": "be"}},
                {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "dummy_it", "RIGHT_ATTRS": {"LOWER": "it", "DEP": {"IN": ["nsubj", "expl"]}}},
                {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "focal_elem", "RIGHT_ATTRS": {"DEP": {"IN": ["attr", "dobj", "pobj", "prep"]}, "LOWER": {"NOT_IN": ["time"]}}},
                {"LEFT_ID": "focal_elem", "REL_OP": ">", "RIGHT_ID": "cleft_clause", "RIGHT_ATTRS": {"DEP": "relcl"}},
            ]
        ]
    },
    {
        "id": "fronted_negative_inversion",
        "category": "Rhetoric & Emphasis",
        "priority": 95,
        "name": "Negative Adverbial Fronted Inversion",
        "formula": "[Negative Adverbial] + [Auxiliary] + [Subject] + [Main Verb]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "root_verb", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "root_verb", "REL_OP": ">", "RIGHT_ID": "aux_verb", "RIGHT_ATTRS": {"DEP": {"IN": ["aux", "auxpass"]}}},
                {"LEFT_ID": "root_verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
                {"LEFT_ID": "root_verb", "REL_OP": ">", "RIGHT_ID": "neg_adv", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no"]},
                    "DEP": {"IN": ["advmod", "neg"]}
                }},
            ]
        ]
    },
    {
        "id": "inverted_conditional_subjunctive",
        "category": "Logic & Stance",
        "priority": 94,
        "name": "Inverted Conditional Clause (Had/Were/Should)",
        "formula": "Had/Were/Should + [Subject] + [VP], [Main Clause]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "cond_verb", "RIGHT_ATTRS": {"DEP": {"IN": ["advcl", "ccomp"]}}},
                {"LEFT_ID": "cond_verb", "REL_OP": ">", "RIGHT_ID": "inv_aux", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["have", "shall"]},
                    "DEP": "aux"
                }},
                {"LEFT_ID": "cond_verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["nsubj", "nsubjpass"]}
                }},
            ],
            [
                {"RIGHT_ID": "inv_were", "RIGHT_ATTRS": {
                    "LEMMA": "be",
                    "TAG": "VBD",
                    "DEP": {"IN": ["advcl", "ccomp"]}
                }},
                {"LEFT_ID": "inv_were", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["nsubj", "nsubjpass"]}
                }},
                {"LEFT_ID": "inv_were", "REL_OP": ">", "RIGHT_ID": "to_comp", "RIGHT_ATTRS": {
                    "DEP": "xcomp"
                }},
            ]
        ]
    },
    {
        "id": "correlative_not_only_but_also",
        "category": "Rhetoric & Emphasis",
        "priority": 93,
        "name": "Correlative Coordination (Not only... but also)",
        "formula": "not only + [A] + but (also) + [B]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "anchor_root", "RIGHT_ATTRS": {}},
                {"LEFT_ID": "anchor_root", "REL_OP": ">>", "RIGHT_ID": "neg_part", "RIGHT_ATTRS": {"LOWER": "not", "DEP": {"IN": ["neg", "advmod", "preconj"]}}},
                {"LEFT_ID": "anchor_root", "REL_OP": ">>", "RIGHT_ID": "coord_conj", "RIGHT_ATTRS": {"LOWER": "but", "POS": "CCONJ"}},
            ]
        ]
    },
    {
        "id": "antithesis_not_but",
        "category": "Rhetoric & Emphasis",
        "priority": 92,
        "name": "Antithesis Parallel Coordination (not... but...)",
        "formula": "[Subject] + [Verb] , not [A] , but [B]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "anchor_root", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "anchor_root", "REL_OP": ">>", "RIGHT_ID": "neg_not", "RIGHT_ATTRS": {"LOWER": "not", "DEP": "neg"}},
                {"LEFT_ID": "anchor_root", "REL_OP": ">>", "RIGHT_ID": "coord_but", "RIGHT_ATTRS": {"LOWER": "but", "POS": "CCONJ"}},
            ]
        ]
    },
    {
        "id": "resultative_so_such_that",
        "category": "Rhetoric & Emphasis",
        "priority": 91,
        "name": "Resultative Degree Constructions (So + Adj / Such + NP + that ...)",
        "formula": "so + [Adj/Adv] + that + [Result Clause]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "head_pred", "RIGHT_ATTRS": {"POS": {"IN": ["ADJ", "ADV", "VERB", "AUX"]}}},
                {"LEFT_ID": "head_pred", "REL_OP": ">>", "RIGHT_ID": "so_mod", "RIGHT_ATTRS": {"LOWER": "so", "DEP": "advmod"}},
                {"LEFT_ID": "head_pred", "REL_OP": ">>", "RIGHT_ID": "res_clause", "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "advcl"]}}},
                {"LEFT_ID": "res_clause", "REL_OP": ">", "RIGHT_ID": "that_mark", "RIGHT_ATTRS": {"LOWER": "that", "DEP": "mark"}},
            ],
            [
                {"RIGHT_ID": "head_np", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
                {"LEFT_ID": "head_np", "REL_OP": ">>", "RIGHT_ID": "such_mod", "RIGHT_ATTRS": {"LOWER": "such", "DEP": {"IN": ["det", "predet", "amod"]}}},
                {"LEFT_ID": "head_np", "REL_OP": ">>", "RIGHT_ID": "res_clause", "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "advcl", "acl", "relcl"]}}},
                {"LEFT_ID": "res_clause", "REL_OP": ">", "RIGHT_ID": "that_mark", "RIGHT_ATTRS": {"LOWER": "that", "DEP": {"IN": ["mark", "dobj", "nsubj"]}}},
            ]
        ]
    },

    # =========================================================================
    # B. Logic & Stance
    # =========================================================================
    {
        "id": "concession_clausal",
        "category": "Logic & Stance",
        "priority": 85,
        "name": "Concessive Subordinate Clause",
        "formula": "although/even though/while + [Clause], [Main Clause]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "main_verb", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "main_verb", "REL_OP": ">", "RIGHT_ID": "sub_verb", "RIGHT_ATTRS": {"DEP": "advcl"}},
                {"LEFT_ID": "sub_verb", "REL_OP": ">", "RIGHT_ID": "concessive_marker", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["although", "though", "while", "whereas"]},
                    "DEP": {"IN": ["mark", "advmod"]}
                }},
            ]
        ]
    },
    {
        "id": "conditional_clausal",
        "category": "Logic & Stance",
        "priority": 85,
        "name": "Conditional Subordinate Clause",
        "formula": "if/unless + [Condition Clause], [Result Clause]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "main_verb", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "main_verb", "REL_OP": ">", "RIGHT_ID": "sub_verb", "RIGHT_ATTRS": {"DEP": "advcl"}},
                {"LEFT_ID": "sub_verb", "REL_OP": ">", "RIGHT_ID": "cond_marker", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["if", "unless", "provided", "providing"]},
                    "DEP": "mark"
                }},
            ],
            [
                {"RIGHT_ID": "main_verb", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "main_verb", "REL_OP": ">", "RIGHT_ID": "cond_adv", "RIGHT_ATTRS": {
                    "LOWER": "long",
                    "DEP": "advmod"
                }},
                {"LEFT_ID": "cond_adv", "REL_OP": ">", "RIGHT_ID": "sub_verb", "RIGHT_ATTRS": {"DEP": "advcl"}},
                {"LEFT_ID": "sub_verb", "REL_OP": ">", "RIGHT_ID": "sub_as", "RIGHT_ATTRS": {
                    "LOWER": "as",
                    "DEP": "mark"
                }},
            ]
        ]
    },

    # =========================================================================
    # C. Cohesion & Framing
    # =========================================================================
    {
        "id": "propositional_encapsulation_which",
        "category": "Cohesion & Framing",
        "priority": 90,
        "name": "Propositional Encapsulation (, which + verb + that)",
        "formula": "[Proposition] , which + [Interpretive Verb] + that [Clause]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "interp_verb", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["relcl", "advcl"]},
                    "LEMMA": {"IN": ["mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"]}
                }},
                {"LEFT_ID": "interp_verb", "REL_OP": ">", "RIGHT_ID": "which_subj", "RIGHT_ATTRS": {
                    "LOWER": "which",
                    "DEP": {"IN": ["nsubj", "nsubjpass"]}
                }},
                {"LEFT_ID": "interp_verb", "REL_OP": ">", "RIGHT_ID": "that_comp", "RIGHT_ATTRS": {
                    "DEP": "ccomp"
                }},
            ]
        ]
    },
    {
        "id": "shell_noun_complement",
        "category": "Cohesion & Framing",
        "priority": 80,
        "name": "Shell Noun Complement Clause",
        "formula": "[Shell Noun] + that [Proposition Clause]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "shell_noun", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "evidence", "assumption"]},
                    "POS": "NOUN"
                }},
                {"LEFT_ID": "shell_noun", "REL_OP": ">", "RIGHT_ID": "comp_clause", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["acl", "appos", "ccomp"]}
                }},
            ]
        ]
    },

    # =========================================================================
    # D. Information Packaging
    # =========================================================================
    {
        "id": "evaluative_dummy_it_subject",
        "category": "Information Packaging",
        "priority": 85,
        "name": "Evaluative Dummy-It Extraposition",
        "formula": "It is + [Adj/Noun] + that/to-inf [Clause]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "copula", "RIGHT_ATTRS": {"LEMMA": "be"}},
                {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "dummy_it", "RIGHT_ATTRS": {"LOWER": "it", "DEP": {"IN": ["nsubj", "expl"]}}},
                {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "eval_adj", "RIGHT_ATTRS": {"DEP": {"IN": ["acomp", "attr"]}}},
                {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "extraposed_clause", "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "csubj", "xcomp", "advcl"]}}},
            ]
        ]
    },
    {
        "id": "complex_transitive_adj_complement",
        "category": "Information Packaging",
        "priority": 80,
        "name": "Complex Transitive Object Complement",
        "formula": "[Subject] + make/find/keep + [Object] + [Adj/Complement]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "causative_verb", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["make", "find", "render", "keep", "leave", "consider"]},
                    "POS": {"IN": ["VERB", "AUX"]}
                }},
                {"LEFT_ID": "causative_verb", "REL_OP": ">", "RIGHT_ID": "dobj", "RIGHT_ATTRS": {"DEP": "dobj"}},
                {"LEFT_ID": "causative_verb", "REL_OP": ">", "RIGHT_ID": "adj_comp", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["oprd", "ccomp", "acomp"]},
                    "POS": {"IN": ["ADJ", "NOUN", "PROPN"]}
                }},
            ],
            [
                {"RIGHT_ID": "causative_verb", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["make", "find", "render", "keep", "leave", "consider"]},
                    "POS": {"IN": ["VERB", "AUX"]}
                }},
                {"LEFT_ID": "causative_verb", "REL_OP": ">", "RIGHT_ID": "adj_comp", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["ccomp", "oprd", "acomp"]},
                    "POS": {"IN": ["ADJ", "NOUN", "PROPN"]}
                }},
                {"LEFT_ID": "adj_comp", "REL_OP": ">", "RIGHT_ID": "obj_subj", "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass", "dobj"]}}},
            ]
        ]
    },
    {
        "id": "causative_bare_infinitive",
        "category": "Information Packaging",
        "priority": 82,
        "name": "Causative Bare-Infinitive Construction",
        "formula": "[Subject] + make/let/have + [Object] + [Bare Infinitive]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "causative_verb", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["make", "let", "have"]},
                    "POS": {"IN": ["VERB", "AUX"]}
                }},
                {"LEFT_ID": "causative_verb", "REL_OP": ">", "RIGHT_ID": "inf_comp", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["ccomp", "xcomp"]},
                    "TAG": "VB"
                }},
                {"LEFT_ID": "inf_comp", "REL_OP": ">", "RIGHT_ID": "obj_subj", "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "dobj"]}}},
            ],
            [
                {"RIGHT_ID": "causative_verb", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["make", "let", "have"]},
                    "POS": {"IN": ["VERB", "AUX"]}
                }},
                {"LEFT_ID": "causative_verb", "REL_OP": ">", "RIGHT_ID": "dobj", "RIGHT_ATTRS": {"DEP": "dobj"}},
                {"LEFT_ID": "causative_verb", "REL_OP": ">", "RIGHT_ID": "inf_comp", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["ccomp", "xcomp"]},
                    "TAG": "VB"
                }},
            ]
        ]
    },
    {
        "id": "nonfinite_participial_adjunct",
        "category": "Information Packaging",
        "priority": 75,
        "name": "Non-finite Participial Adjunct",
        "formula": "[V-ing / V-ed Phrase] , [Main Subject] + [Main VP]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "main_verb", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "main_verb", "REL_OP": ">", "RIGHT_ID": "part_verb", "RIGHT_ATTRS": {
                    "DEP": "advcl",
                    "TAG": {"IN": ["VBG", "VBN"]}
                }},
            ]
        ]
    },
    {
        "id": "non_restrictive_relative_clause",
        "category": "Information Packaging",
        "priority": 70,
        "name": "Relative Clause",
        "formula": "[NP] + which/that/who + [VP]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "antecedent", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}},
                {"LEFT_ID": "antecedent", "REL_OP": ">", "RIGHT_ID": "rel_verb", "RIGHT_ATTRS": {"DEP": "relcl"}},
                {"LEFT_ID": "rel_verb", "REL_OP": ">", "RIGHT_ID": "rel_pron", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["which", "who", "whom", "whose", "that"]},
                    "DEP": {"IN": ["nsubj", "nsubjpass", "dobj", "pobj"]}
                }},
            ]
        ]
    },
    {
        "id": "correlative_comparative_the_the",
        "category": "Information Packaging",
        "priority": 85,
        "name": "Proportional Comparative (The more..., the more...)",
        "formula": "The + [Comparative] ..., the + [Comparative] ...",
        "tree_patterns": [
            [
                {"RIGHT_ID": "main_comp", "RIGHT_ATTRS": {"DEGREE": "CMPR", "POS": {"IN": ["ADJ", "ADV"]}}},
                {"LEFT_ID": "main_comp", "REL_OP": ">", "RIGHT_ID": "det_the_1", "RIGHT_ATTRS": {"LOWER": "the", "DEP": "det"}},
            ]
        ]
    },
    {
        "id": "with_compound_construction",
        "category": "Information Packaging",
        "priority": 88,
        "name": "With Absolute / Compound Construction",
        "formula": "with + [NP] + [Participle/Adj/PrepP]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "with_prep", "RIGHT_ATTRS": {"LOWER": "with", "DEP": "prep"}},
                {"LEFT_ID": "with_prep", "REL_OP": ">", "RIGHT_ID": "part_pred", "RIGHT_ATTRS": {
                    "DEP": "pcomp",
                    "TAG": {"IN": ["VBG", "VBN", "JJ"]}
                }},
                {"LEFT_ID": "part_pred", "REL_OP": ">", "RIGHT_ID": "obj_subj", "RIGHT_ATTRS": {
                    "DEP": "nsubj"
                }},
            ]
        ]
    },
    {
        "id": "absolute_construction",
        "category": "Information Packaging",
        "priority": 87,
        "name": "Nominative Absolute Construction",
        "formula": "[NP] + [V-ing / V-ed Phrase] , [Main Subject] + [Main VP]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "main_verb", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "main_verb", "REL_OP": ">", "RIGHT_ID": "abs_verb", "RIGHT_ATTRS": {
                    "DEP": "advcl",
                    "TAG": {"IN": ["VBG", "VBN"]}
                }},
                {"LEFT_ID": "abs_verb", "REL_OP": ">", "RIGHT_ID": "abs_subj", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["nsubj", "nsubjpass"]}
                }},
            ]
        ]
    }
]


# ------------------------------------------------------------------------------
# 2. Grammar Pattern Matcher Engine
# ------------------------------------------------------------------------------
class GrammarPatternEngine:
    """
    Compiles and executes declarative DependencyMatcher patterns against spaCy Docs.
    Provides fast (<1ms), zero-token, robust syntactic pattern matching and slot extraction.
    """
    _instance: Optional["GrammarPatternEngine"] = None
    _matcher: Optional[DependencyMatcher] = None
    _pattern_map: Dict[str, Dict[str, Any]] = {}

    def __init__(self, vocab: spacy.vocab.Vocab):
        self.vocab = vocab
        self.matcher = DependencyMatcher(vocab)
        self.pattern_map: Dict[str, Dict[str, Any]] = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Compiles declarative patterns into the DependencyMatcher."""
        for pat in DECLARATIVE_GRAMMAR_PATTERNS:
            pat_id = pat["id"]
            self.pattern_map[pat_id] = pat
            tree_patterns = pat.get("tree_patterns", [])
            if tree_patterns:
                try:
                    self.matcher.add(pat_id, tree_patterns)
                except Exception as e:
                    # Keep compilation fault-tolerant
                    pass

    @classmethod
    def get_engine(cls, nlp: spacy.language.Language) -> "GrammarPatternEngine":
        """Singleton accessor for engine instance bound to vocabulary."""
        if cls._instance is None or cls._instance.vocab != nlp.vocab:
            cls._instance = cls(nlp.vocab)
        return cls._instance

    def match_sentence(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """
        Matches doc against compiled dependency trees.
        Returns list of matched patterns sorted by priority descending, with dynamically resolved formulas.
        """
        matches = self.matcher(doc)
        if not matches:
            return []

        matched_results: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        for match_id, token_indices in matches:
            string_id = self.vocab.strings[match_id]
            if string_id in seen_ids:
                continue
            pattern_meta = self.pattern_map.get(string_id, {})
            base_formula = pattern_meta.get("formula", "[Subject] + [VP]")

            # Semantic/topological validation for inversion:
            # In genuine fronted inversion, the negative adverbial must appear BEFORE the subject and aux
            if string_id == "fronted_negative_inversion":
                tokens = [doc[i] for i in token_indices]
                neg_adv = next((t for t in tokens if t.dep_ in ("advmod", "neg") and t.lower_ in ("hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no")), None)
                subj = next((t for t in tokens if t.dep_ in ("nsubj", "nsubjpass")), None)
                aux = next((t for t in tokens if t.dep_ in ("aux", "auxpass")), None)
                if neg_adv and subj and aux:
                    # Inverted order: NegAdv must be before Aux, and Aux before Subj (e.g. Never had he...)
                    if not (neg_adv.i < aux.i < subj.i):
                        continue
                else:
                    continue

            # Topological validation for inverted conditional subjunctive:
            # The auxiliary verb (Had / Were / Should) must strictly precede the subject (aux.i < subj.i)
            # This prevents false positives on standard past perfect (e.g. "I had switched")
            if string_id == "inverted_conditional_subjunctive":
                tokens = [doc[i] for i in token_indices]
                inv_aux = next((t for t in tokens if t.lower_ in ("had", "were", "should")), None)
                subj = next((t for t in tokens if t.dep_ in ("nsubj", "nsubjpass")), None)
                if inv_aux and subj:
                    if not (inv_aux.i < subj.i):
                        continue
                    # Must also NOT have an ordinary subordinate conjunction (mark) introducing it
                    head_verb = inv_aux.head if inv_aux.pos_ == "AUX" and inv_aux.dep_ == "aux" else inv_aux
                    if any(c.dep_ == "mark" for c in head_verb.children):
                        continue
                else:
                    continue

            # Semantic/topological validation for antithesis coordination (not... but...):
            # 'not' must strictly precede 'but' (not.i < but.i) to prevent '..., but not ...' from misfiring,
            # and must NOT be 'not only/merely/just/simply' (which belongs to correlative_not_only_but_also).
            if string_id == "antithesis_not_but":
                tokens = [doc[i] for i in token_indices]
                not_tok = next((t for t in tokens if t.lower_ == "not" and t.dep_ == "neg"), None)
                but_tok = next((t for t in tokens if t.lower_ == "but" and t.pos_ == "CCONJ"), None)
                if not_tok and but_tok:
                    if not (not_tok.i < but_tok.i):
                        continue
                    next_after_not = doc[not_tok.i + 1].lower_ if not_tok.i + 1 < len(doc) else ""
                    if next_after_not in ("only", "merely", "just", "simply"):
                        continue
                else:
                    continue

            if string_id == "resultative_so_such_that":
                tokens = [doc[i] for i in token_indices]
                deg_tok = next((t for t in tokens if t.lower_ in ("so", "such")), None)
                that_tok = next((t for t in tokens if t.lower_ == "that"), None)
                if not deg_tok or not that_tok or deg_tok.i >= that_tok.i:
                    continue
                if deg_tok.lower_ == "so" and that_tok.i == deg_tok.i + 1:
                    continue

            # Semantic/topological validation for it-cleft sentences:
            # The subordinate cleft clause must be introduced by a relative/subordinate marker (that, who, whom, etc.),
            # the copula must NOT have an adjectival complement (acomp, e.g., "It is also true that..."),
            # and the focal element must be a substantive phrase (not a bare discourse adverb or extraposition noun).
            if string_id == "cleft_it_be_relcl":
                tokens = [doc[i] for i in token_indices]
                copula = next((t for t in tokens if t.lemma_ == "be"), None)
                cleft_clause = next((t for t in tokens if t != copula and t.dep_ in ("relcl", "ccomp", "advcl")), None)
                focal_elem = next((t for t in tokens if t != copula and t != cleft_clause and t.lower_ != "it"), None)
                if copula and any(c.dep_ == "acomp" for c in copula.children):
                    continue
                _EXTRAPOSITION_NOUNS = {
                    "surprise", "wonder", "secret", "doubt", "coincidence", "accident",
                    "pity", "shame", "mystery", "fact", "truth", "news", "use", "good",
                    "point", "time", "question", "matter"
                }
                if (
                    not focal_elem
                    or focal_elem.lower_ in ("also", "not", "never", "even", "only", "still", "just", "merely")
                    or focal_elem.lemma_.lower() in _EXTRAPOSITION_NOUNS
                ):
                    continue
                if cleft_clause:
                    has_cleft_marker = any(
                        c.lower_ in ("that", "who", "whom", "which", "where", "when")
                        for c in cleft_clause.children
                    )
                    if not has_cleft_marker:
                        continue
                else:
                    continue

            # Semantic validation for nonfinite participial adjunct:
            # Must NOT be introduced by interrogative/subordinating adverbs/marks like 'why', 'how', 'when', 'if', 'because'
            # Must NOT have finite auxiliary verbs (e.g. 'was presented' -> finite passive, not nonfinite participle;
            # note that non-finite 'having' or 'being' as in 'Having seen...' or 'Being delayed...' are valid non-finite auxiliaries!)
            if string_id == "nonfinite_participial_adjunct":
                tokens = [doc[i] for i in token_indices]
                part_v = next((t for t in tokens if t.tag_ in ("VBG", "VBN")), None)
                if part_v:
                    has_subordinator = any(child.dep_ in ("mark", "advmod") and child.lemma_.lower() in ("why", "how", "when", "where", "if", "because", "although", "while") for child in part_v.children)
                    has_finite_aux = any(
                        child.dep_ in ("aux", "auxpass") and child.lower_ not in ("having", "being")
                        for child in part_v.children
                    )
                    # If part_v has a mark or adverbial subordinator, or finite auxiliary, it is not a bare nonfinite participial adjunct
                    if has_subordinator or has_finite_aux:
                        continue
                    # Also check if it's introduced by a 'why/how' attached to its head
                    if part_v.head and any(c.lemma_.lower() in ("why", "how", "that") and c.i < part_v.i for c in part_v.head.children):
                        continue

            # Topological validation for shell_noun_complement:
            # Verify that the complement clause is genuinely introduced by appositive 'that'/'whether' or 'of',
            # rather than a restrictive relative clause ('which'/'that' acting as nsubj/dobj)
            if string_id == "shell_noun_complement":
                tokens = [doc[i] for i in token_indices]
                shell_tok = next((t for t in tokens if t.pos_ == "NOUN"), None)
                comp_tok = next((t for t in tokens if t != shell_tok), None)
                if not shell_tok or not comp_tok:
                    continue
                has_appos_marker = any(
                    c.dep_ == "mark" and c.lower_ in ("that", "whether")
                    for c in comp_tok.children
                ) or any(
                    c.dep_ == "prep" and c.lower_ == "of"
                    for c in shell_tok.children
                )
                if not has_appos_marker:
                    continue

            # Topological validation for absolute construction:
            # The independent participial construction must NOT be introduced by subordinating conjunctions (mark)
            # and should not have auxiliary verbs (like "was exhausted") which indicate a finite subordinate clause.
            if string_id == "absolute_construction":
                tokens = [doc[i] for i in token_indices]
                abs_v = next((t for t in tokens if t.tag_ in ("VBG", "VBN")), None)
                if abs_v:
                    has_subordinator = any(child.dep_ == "mark" for child in abs_v.children)
                    has_aux = any(child.dep_ in ("aux", "auxpass") and child.lower_ not in ("having", "being") for child in abs_v.children)
                    if has_subordinator or has_aux:
                        continue

            if string_id == "correlative_not_only_but_also":
                tokens = [doc[i] for i in token_indices]
                neg_tok = next((t for t in tokens if t.lemma_.lower() == "not"), None)
                cc_tok = next((t for t in tokens if t.lemma_.lower() == "but"), None)
                if not neg_tok or not cc_tok or neg_tok.i >= cc_tok.i:
                    continue
                next_after_neg = doc[neg_tok.i + 1].lower_ if neg_tok.i + 1 < len(doc) else ""
                if next_after_neg not in ("only", "merely", "just", "simply"):
                    continue

            if string_id == "evaluative_dummy_it_subject":
                tokens = [doc[i] for i in token_indices]
                cop_tok = next((t for t in tokens if t.lemma_ == "be"), None)
                eval_tok = next((t for t in tokens if t != cop_tok and t.dep_ in ("acomp", "attr", "oprd") and t.pos_ in ("ADJ", "NOUN")), None)
                ext_tok = next((t for t in tokens if t != cop_tok and t != eval_tok and t.dep_ in ("ccomp", "csubj", "xcomp", "advcl")), None)
                if not eval_tok or not ext_tok or ext_tok.i <= eval_tok.i:
                    continue
                if ext_tok.tag_ in ("VBG", "VBN"):
                    continue
                has_to_or_comp = any(
                    (c.lower_ == "to" and c.dep_ in ("aux", "mark"))
                    or c.lower_ in ("that", "whether", "if", "how", "why", "what", "when", "where", "who")
                    for c in ext_tok.children
                ) or any(
                    t.lower_ in ("that", "to", "whether", "if")
                    for t in doc[eval_tok.i + 1 : ext_tok.i + 1]
                )
                if not has_to_or_comp:
                    continue

            if string_id == "complex_transitive_adj_complement":
                tokens = [doc[i] for i in token_indices]
                if len(tokens) >= 3:
                    v_tok, dobj_tok, comp_tok = tokens[0], tokens[1], tokens[2]
                    if comp_tok.dep_ == "conj" or (comp_tok.i > 0 and doc[comp_tok.i - 1].lower_ in ("and", "or", "but")):
                        continue
                    if any(t.dep_ == "prep" for t in doc[dobj_tok.i : comp_tok.i]):
                        continue

            # Resolve dynamic slots based on actual tokens in the sentence
            resolved_formula = self._resolve_formula(string_id, base_formula, doc, token_indices)
            seen_ids.add(string_id)

            matched_results.append({
                "pattern_id": string_id,
                "category": pattern_meta.get("category", "Information Packaging"),
                "priority": pattern_meta.get("priority", 50),
                "name": pattern_meta.get("name", "Syntactic Pattern"),
                "formula": resolved_formula,
                "token_indices": token_indices,
                "matched_tokens": [doc[i].text for i in token_indices]
            })

        # Sort by priority descending
        matched_results.sort(key=lambda x: x["priority"], reverse=True)
        return matched_results

    def _resolve_formula(self, pattern_id: str, base_formula: str, doc: spacy.tokens.Doc, token_indices: List[int]) -> str:
        """Dynamically adapts the COBUILD formula to the sentence's actual lexical triggers."""
        try:
            tokens = [doc[i] for i in token_indices]
            if pattern_id == "fronted_negative_inversion":
                neg_adv = next((t for t in tokens if t.dep_ in ("advmod", "neg")), tokens[0])
                return f"{neg_adv.text.capitalize()} + [aux/be] + [Subject] + [VP]"
            elif pattern_id == "resultative_so_such_that":
                has_such = any(t.lower_ == "such" for t in tokens)
                if has_such:
                    return "such + [NP] + that + [Result Clause]"
                return "so + [Adj/Adv] + that + [Result Clause]"
            elif pattern_id == "concession_clausal":
                marker = next((t for t in tokens if t.dep_ in ("mark", "advmod")), tokens[0])
                next_tok = doc[marker.i + 1].lower_ if marker.i + 1 < len(doc) else ""
                prev_tok = doc[marker.i - 1].lower_ if marker.i > 0 else ""
                if marker.lower_ == "even" and next_tok in ("if", "though"):
                    m_str = f"Even {next_tok}"
                elif prev_tok == "even" and marker.lower_ in ("if", "though"):
                    m_str = f"Even {marker.lower_}"
                else:
                    m_str = marker.text.capitalize()
                return f"{m_str} + [Clause], [Subject] + [VP]"
            elif pattern_id == "conditional_clausal":
                doc_text_low = doc.text.lower()
                if "as long as" in doc_text_low:
                    return "As long as + [Clause], [Subject] + [VP]"
                marker = next((t for t in tokens if t.dep_ == "mark"), tokens[0])
                prev_tok = doc[marker.i - 1].lower_ if marker.i > 0 else ""
                if prev_tok == "even" and marker.lower_ == "if":
                    return "Even if + [Clause], [Subject] + [VP]"
                return f"{marker.text.capitalize()} + [Clause], [Subject] + [VP]"
            elif pattern_id == "propositional_encapsulation_which":
                interp = next((t for t in tokens if t.pos_ in ("VERB", "AUX")), None)
                v_lemma = interp.lemma_.lower() if interp else "suggest"
                if v_lemma.endswith("y") and len(v_lemma) > 1 and v_lemma[-2] not in "aeiou":
                    v_3sg = f"{v_lemma[:-1]}ies"
                elif v_lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
                    v_3sg = f"{v_lemma}es"
                else:
                    v_3sg = f"{v_lemma}s"
                return f", which + {v_3sg} + that + [Proposition Clause]"
            elif pattern_id == "shell_noun_complement":
                shell = next((t for t in tokens if t.pos_ == "NOUN"), None)
                n_lemma = shell.lemma_.lower() if shell else "fact"
                return f"The + {n_lemma} + that/of + [Proposition Clause]"
            elif pattern_id == "complex_transitive_adj_complement":
                caus = next((t for t in tokens if t.pos_ in ("VERB", "AUX")), None)
                v_lemma = caus.lemma_.lower() if caus else "make"
                comp = next((t for t in tokens if t != caus and t.dep_ in ("oprd", "ccomp", "acomp")), None)
                comp_slot = "[Noun/NP]" if comp and comp.pos_ in ("NOUN", "PROPN") else "[Adj]"
                return f"[Subject] + {v_lemma} + [Object] + {comp_slot}"
            elif pattern_id == "causative_bare_infinitive":
                caus = next((t for t in tokens if t.lemma_.lower() in ("make", "let", "have")), None)
                v_lemma = caus.lemma_.lower() if caus else "make"
                return f"[Subject] + {v_lemma} + [Object] + [Bare Infinitive]"
            elif pattern_id == "non_restrictive_relative_clause":
                rel_pron = next((t for t in tokens if t.lemma_.lower() in ("which", "who", "whom", "whose", "that")), None)
                pron_str = rel_pron.text.lower() if rel_pron else "which/who"
                antecedent = next((t for t in tokens if t.dep_ != "relcl" and t.pos_ in ("NOUN", "PROPN", "PRON")), None)
                has_comma = False
                if rel_pron and rel_pron.i > 0:
                    prev_tok = doc[rel_pron.i - 1]
                    if prev_tok.text == ",":
                        has_comma = True
                if has_comma:
                    return f"[NP], {pron_str} + [VP]"
                else:
                    return f"[NP] + {pron_str} + [VP]"
            elif pattern_id == "nonfinite_participial_adjunct":
                part = next((t for t in tokens if t.tag_ in ("VBG", "VBN")), None)
                if part:
                    has_having = any(c.dep_ in ("aux", "auxpass") and c.lower_ == "having" for c in part.children)
                    has_being = any(c.dep_ in ("aux", "auxpass") and c.lower_ == "being" for c in part.children)
                    if has_having:
                        v_type = "Having + V-ed"
                    elif has_being:
                        v_type = "Being + V-ed"
                    elif part.tag_ == "VBG":
                        v_type = "V-ing"
                    else:
                        v_type = "V-ed"
                    main_v = next((t for t in tokens if t != part and t.pos_ in ("VERB", "AUX")), None)
                    if main_v and part.i < main_v.i:
                        return f"[{v_type} Phrase], [Subject] + [VP]"
                    else:
                        return f"[Subject] + [VP], [{v_type} Phrase]"
            elif pattern_id == "cleft_it_be_relcl":
                return "It + [be] + [Focal Element] + that/who + [Clause]"
            elif pattern_id == "antithesis_not_but":
                return "[Subject] + [VP], not + [PrepP/NP], but + [PrepP/NP]"
            elif pattern_id == "correlative_not_only_but_also":
                if re.search(r"(?:^|[,;—–-]\s*)not\s+only\s+(?:can|could|will|would|shall|should|may|might|must|do|does|did|have|has|had|am|is|are|was|were)\b", doc.text.lower()):
                    return "Not only + [Aux] + [Subject] + [VP], but + [Subject] + [also] + [VP]"
                return "[Subject] + not only + [VP], but also + [VP]"
            elif pattern_id == "inverted_conditional_subjunctive":
                inv_tok = next((t for t in tokens if t.lower_ in ("had", "were", "should")), None)
                first_w = inv_tok.text.capitalize() if inv_tok else "Had"
                return f"{first_w} + [Subject] + [VP], [Main Clause]"
            elif pattern_id == "with_compound_construction":
                return "with + [NP] + [Participle/Adj/PrepP]"
            elif pattern_id == "absolute_construction":
                return "[NP] + [V-ing / V-ed Phrase], [Main Subject] + [Main VP]"
        except Exception:
            pass
        return base_formula

