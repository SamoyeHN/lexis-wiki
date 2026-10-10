"""
Declarative Grammar Pattern Definitions and DependencyMatcher Engine.
Implements Scheme 1: Declarative Syntax Pattern DSL for robust, zero-token grammar extraction.
Decouples syntactic tree topologies, slot extraction, and COBUILD formula bindings from monolithic procedural code.
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
                {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "focal_elem", "RIGHT_ATTRS": {"DEP": {"IN": ["attr", "dobj", "pobj", "prep", "npadvmod", "advmod"]}}},
                {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "cleft_clause", "RIGHT_ATTRS": {"DEP": {"IN": ["relcl", "ccomp", "advcl"]}}},
            ],
            [
                {"RIGHT_ID": "copula", "RIGHT_ATTRS": {"LEMMA": "be"}},
                {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "dummy_it", "RIGHT_ATTRS": {"LOWER": "it", "DEP": {"IN": ["nsubj", "expl"]}}},
                {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "focal_elem", "RIGHT_ATTRS": {"DEP": {"IN": ["attr", "dobj", "pobj", "prep"]}}},
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
        "id": "correlative_not_only_but_also",
        "category": "Rhetoric & Emphasis",
        "priority": 90,
        "name": "Correlative Coordination (Not only... but also)",
        "formula": "not only + [A] + but also + [B]",
        "tree_patterns": [
            [
                {"RIGHT_ID": "coord_conj", "RIGHT_ATTRS": {"LOWER": "but", "DEP": "cc"}},
                {"LEFT_ID": "coord_conj", "REL_OP": "<", "RIGHT_ID": "anchor_head", "RIGHT_ATTRS": {}},
                {"LEFT_ID": "anchor_head", "REL_OP": ">", "RIGHT_ID": "neg_part", "RIGHT_ATTRS": {"LOWER": "not", "DEP": "neg"}},
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
                    "LEMMA": {"IN": ["although", "though", "while", "whereas", "even"]},
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
                {"LEFT_ID": "main_verb", "REL_OP": ">", "RIGHT_ID": "sub_verb", "RIGHT_ATTRS": {"DEP": "advcl"}},
                {"LEFT_ID": "sub_verb", "REL_OP": ">", "RIGHT_ID": "cond_as", "RIGHT_ATTRS": {
                    "LOWER": "as",
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
                {"LEFT_ID": "causative_verb", "REL_OP": ">", "RIGHT_ID": "adj_comp", "RIGHT_ATTRS": {"DEP": {"IN": ["oprd", "ccomp", "acomp"]}}},
            ],
            [
                {"RIGHT_ID": "causative_verb", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["make", "find", "render", "keep", "leave", "consider"]},
                    "POS": {"IN": ["VERB", "AUX"]}
                }},
                {"LEFT_ID": "causative_verb", "REL_OP": ">", "RIGHT_ID": "adj_comp", "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "oprd", "acomp"]}}},
                {"LEFT_ID": "adj_comp", "REL_OP": ">", "RIGHT_ID": "obj_subj", "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass", "dobj"]}}},
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

            # Semantic validation for nonfinite participial adjunct:
            # Must NOT be introduced by interrogative/subordinating adverbs/marks like 'why', 'how', 'when', 'if', 'because'
            # Must NOT be inside quotes as a gerund phrase, and must function as genuine non-finite modifier.
            if string_id == "nonfinite_participial_adjunct":
                tokens = [doc[i] for i in token_indices]
                part_v = next((t for t in tokens if t.tag_ in ("VBG", "VBN")), None)
                if part_v:
                    has_subordinator = any(child.dep_ in ("mark", "advmod") and child.lemma_.lower() in ("why", "how", "when", "where", "if", "because", "although", "while") for child in part_v.children)
                    # If part_v has a mark or adverbial subordinator, it is an adverbial/wh clause, not a bare participial adjunct
                    if has_subordinator:
                        continue
                    # Also check if it's introduced by a 'why/how' attached to its head
                    if part_v.head and any(c.lemma_.lower() in ("why", "how", "that") and c.i < part_v.i for c in part_v.head.children):
                        continue

            # Topological validation for absolute construction:
            # The independent participial construction must NOT be introduced by subordinating conjunctions (mark)
            # and should not have auxiliary verbs (like "was exhausted") which indicate a finite subordinate clause.
            if string_id == "absolute_construction":
                tokens = [doc[i] for i in token_indices]
                abs_v = next((t for t in tokens if t.tag_ in ("VBG", "VBN")), None)
                if abs_v:
                    has_subordinator = any(child.dep_ == "mark" for child in abs_v.children)
                    has_aux = any(child.dep_ in ("aux", "auxpass") for child in abs_v.children)
                    if has_subordinator or has_aux:
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
            elif pattern_id == "concession_clausal":
                marker = next((t for t in tokens if t.dep_ in ("mark", "advmod")), tokens[0])
                return f"{marker.text.capitalize()} + [Clause], [Subject] + [VP]"
            elif pattern_id == "conditional_clausal":
                doc_text_low = doc.text.lower()
                if "as long as" in doc_text_low:
                    return "As long as + [Clause], [Subject] + [VP]"
                marker = next((t for t in tokens if t.dep_ == "mark"), tokens[0])
                return f"{marker.text.capitalize()} + [Clause], [Subject] + [VP]"
            elif pattern_id == "propositional_encapsulation_which":
                interp = next((t for t in tokens if t.pos_ in ("VERB", "AUX")), None)
                v_lemma = interp.lemma_.lower() if interp else "suggest"
                return f", which + {v_lemma}s + that + [Proposition Clause]"
            elif pattern_id == "shell_noun_complement":
                shell = next((t for t in tokens if t.pos_ == "NOUN"), None)
                n_lemma = shell.lemma_.lower() if shell else "fact"
                return f"The + {n_lemma} + that/of + [Proposition Clause]"
            elif pattern_id == "complex_transitive_adj_complement":
                caus = next((t for t in tokens if t.pos_ in ("VERB", "AUX")), None)
                v_lemma = caus.lemma_.lower() if caus else "make"
                return f"[Subject] + {v_lemma} + [Object] + [Adj]"
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
                    v_type = "V-ing" if part.tag_ == "VBG" else "V-ed"
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

