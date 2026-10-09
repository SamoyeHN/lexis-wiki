"""
Declarative Multi-Word Expression Pattern Definitions and ExpressionPatternEngine.
Implements Scheme 3: Declarative Multi-Word Expression Pattern DSL for robust, zero-token phrase and collocation extraction.
Decouples syntactic tree topologies, slot extraction, transitivity valency, and canonical formula generation from monolithic procedural code.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import re
import spacy
from spacy.matcher import DependencyMatcher


COMMON_PARTICLES: Set[str] = {
    "up", "down", "out", "in", "off", "on", "away", "back", "over",
    "around", "round", "along", "through", "about", "ahead", "apart", "aside",
    "forward", "backward"
}

DEPENDENT_PREPS: Set[str] = {
    "to", "for", "with", "from", "on", "in", "at", "into", "as",
    "against", "towards", "toward", "upon", "about", "of"
}

IDIOMATIC_POSS_NOUNS: Set[str] = {
    "best", "mind", "temper", "breath", "time", "part", "way", "heart",
    "step", "place", "voice", "promise", "life", "potential", "future",
    "passion", "dream", "goal", "duty", "responsibility", "effort", "stride",
    "horizon", "footing", "view", "interest", "role", "purpose"
}

LIGHT_VERB_IDIOM_HEADS: Set[str] = {"take", "bring", "put", "call", "set", "keep", "bear"}
LIGHT_VERB_IDIOM_POBJS: Set[str] = {"account", "consideration", "effect", "play", "question", "mind"}

FRAME_PREPOSITION_NOUNS: Set[str] = {
    "response", "terms", "expense", "addition", "behalf", "favour", "favor",
    "spite", "light", "view", "case", "place", "respect", "regard", "relation",
    "aid", "search", "charge", "accordance", "conformity", "compliance", "contrast",
    "comparison", "virtue", "means", "line", "front", "top", "middle", "honour", "honor"
}

COLLOCATION_VERB_PREPS: Dict[str, str] = {
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
    "worry": "about",
    "care": "about",
    "complain": "about",
    "wonder": "about",
    "belong": "to",
    "listen": "to",
    "agree": "with",
    "disagree": "with",
    "deal": "with",
    "wait": "for",
    "search": "for",
    "apologize": "for",
    "look": "for",
}

ADJ_DEPENDENT_PREPS: Dict[str, str] = {
    "aware": "of",
    "conscious": "of",
    "capable": "of",
    "tired": "of",
    "fond": "of",
    "proud": "of",
    "typical": "of",
    "characteristic": "of",
    "worthy": "of",
    "guilty": "of",
    "interested": "in",
    "involved": "in",
    "engaged": "in",
    "successful": "in",
    "rich": "in",
    "responsible": "for",
    "famous": "for",
    "notorious": "for",
    "eligible": "for",
    "suitable": "for",
    "prepared": "for",
    "similar": "to",
    "related": "to",
    "accustomed": "to",
    "dedicated": "to",
    "committed": "to",
    "exposed": "to",
    "prone": "to",
    "vulnerable": "to",
    "opposed": "to",
    "superior": "to",
    "inferior": "to",
    "familiar": "with",
    "compatible": "with",
    "consistent": "with",
    "associated": "with",
    "patient": "with",
    "keen": "on",
    "dependent": "on",
    "reliant": "on",
    "based": "on",
    "different": "from",
    "safe": "from",
}

NOUN_DEPENDENT_PREPS: Dict[str, str] = {
    "access": "to",
    "approach": "to",
    "attitude": "to",
    "solution": "to",
    "response": "to",
    "reaction": "to",
    "key": "to",
    "threat": "to",
    "contribution": "to",
    "damage": "to",
    "resistance": "to",
    "introduction": "to",
    "invitation": "to",
    "demand": "for",
    "need": "for",
    "reason": "for",
    "request": "for",
    "preference": "for",
    "respect": "for",
    "talent": "for",
    "appetite": "for",
    "effect": "on",
    "impact": "on",
    "influence": "on",
    "emphasis": "on",
    "pressure": "on",
    "focus": "on",
    "ban": "on",
    "restriction": "on",
    "relationship": "with",
    "connection": "with",
    "contact": "with",
    "difficulty": "with",
    "decrease": "in",
    "increase": "in",
    "rise": "in",
    "drop": "in",
    "fall": "in",
    "interest": "in",
    "confidence": "in",
    "belief": "in",
    "experience": "in",
}


# ------------------------------------------------------------------------------
# 1. Declarative Expression Patterns DSL
# ------------------------------------------------------------------------------
DECLARATIVE_EXPRESSION_PATTERNS: List[Dict[str, Any]] = [
    # =========================================================================
    # A. Phrasal Verbs (Verb + Particle / Verb + Particle + Preposition)
    # =========================================================================
    {
        "id": "verb_particle_prep_three_part",
        "type": "phrasal verb",
        "priority": 95,
        "name": "Three-Part Phrasal Prepositional Verb",
        "tree_patterns": [
            [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "particle", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["prt", "advmod"]},
                    "LOWER": {"IN": list(COMMON_PARTICLES)}
                }},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {
                    "DEP": "prep",
                    "LOWER": {"IN": list(DEPENDENT_PREPS)}
                }},
            ]
        ]
    },
    {
        "id": "verb_touch_in_with",
        "type": "idiom",
        "priority": 94,
        "name": "Keep/Stay/Get in Touch Construction",
        "tree_patterns": [
            [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["keep", "get", "stay", "lose", "be"]},
                    "POS": {"IN": ["VERB", "AUX"]}
                }},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep_in", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["prep", "prt"]},
                    "LOWER": "in"
                }},
                {"LEFT_ID": "prep_in", "REL_OP": ">", "RIGHT_ID": "noun_touch", "RIGHT_ATTRS": {
                    "LOWER": "touch"
                }},
            ],
            [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["keep", "get", "stay", "lose", "be"]},
                    "POS": {"IN": ["VERB", "AUX"]}
                }},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep_in", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["prep", "prt"]},
                    "LOWER": "in"
                }},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "noun_touch", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["dobj", "pobj"]},
                    "LOWER": "touch"
                }},
            ]
        ]
    },
    {
        "id": "verb_particle_standard",
        "type": "phrasal verb",
        "priority": 90,
        "name": "Standard Phrasal Verb (Verb + Particle)",
        "tree_patterns": [
            [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "particle", "RIGHT_ATTRS": {
                    "DEP": "prt"
                }},
            ],
            [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "particle", "RIGHT_ATTRS": {
                    "DEP": "advmod",
                    "LOWER": {"IN": list(COMMON_PARTICLES)}
                }},
            ]
        ]
    },

    # =========================================================================
    # B. Idiomatic & Possessive Frames
    # =========================================================================
    {
        "id": "light_verb_prepositional_idiom",
        "type": "idiom",
        "priority": 92,
        "name": "Prepositional Verbal Idiom (take into account)",
        "tree_patterns": [
            [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": list(LIGHT_VERB_IDIOM_HEADS)},
                    "POS": {"IN": ["VERB", "AUX"]}
                }},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {
                    "DEP": "prep",
                    "LOWER": {"IN": ["into", "to", "in", "under"]}
                }},
                {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "pobj", "RIGHT_ATTRS": {
                    "DEP": "pobj",
                    "LEMMA": {"IN": list(LIGHT_VERB_IDIOM_POBJS)}
                }},
            ]
        ]
    },
    {
        "id": "idiomatic_possessive_frame",
        "type": "collocation",
        "priority": 88,
        "name": "Idiomatic Possessive Construction",
        "tree_patterns": [
            [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "noun", "RIGHT_ATTRS": {
                    "DEP": "dobj",
                    "LEMMA": {"IN": list(IDIOMATIC_POSS_NOUNS)}
                }},
                {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "poss", "RIGHT_ATTRS": {
                    "DEP": "poss"
                }},
            ]
        ]
    },

    # =========================================================================
    # C. Prepositional Collocations & Bound Prepositions
    # =========================================================================
    {
        "id": "verb_transitive_preposition_collocation",
        "type": "collocation",
        "priority": 85,
        "name": "Verb + Object + Preposition Collocation",
        "tree_patterns": [
            [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {
                    "LEMMA": {"IN": list(COLLOCATION_VERB_PREPS.keys())},
                    "POS": {"IN": ["VERB", "AUX"]}
                }},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "dobj", "RIGHT_ATTRS": {
                    "DEP": "dobj"
                }},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {
                    "DEP": "prep"
                }},
            ]
        ]
    },
    {
        "id": "verb_intransitive_dependent_preposition",
        "type": "phrasal verb",
        "priority": 82,
        "name": "Prepositional Verb (Verb + Preposition)",
        "tree_patterns": [
            [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {
                    "DEP": "prep",
                    "LOWER": {"IN": list(DEPENDENT_PREPS)}
                }},
            ]
        ]
    },

    # =========================================================================
    # D. Prepositional Frames & Non-Verbal Collocations
    # =========================================================================
    {
        "id": "prepositional_noun_frame",
        "type": "set phrase",
        "priority": 86,
        "name": "Three-Part Prepositional Noun Frame (in response to / at the expense of)",
        "tree_patterns": [
            [
                {"RIGHT_ID": "prep1", "RIGHT_ATTRS": {"POS": "ADP"}},
                {"LEFT_ID": "prep1", "REL_OP": ">", "RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN", "DEP": "pobj"}},
                {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "prep2", "RIGHT_ATTRS": {"POS": "ADP", "DEP": "prep"}},
                {"LEFT_ID": "prep2", "REL_OP": ">", "RIGHT_ID": "pobj", "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "pcomp"]}}}
            ]
        ]
    },
    {
        "id": "adj_dependent_preposition",
        "type": "collocation",
        "priority": 84,
        "name": "Adjective + Preposition Collocation (aware of / interested in)",
        "tree_patterns": [
            [
                {"RIGHT_ID": "adj", "RIGHT_ATTRS": {"POS": "ADJ"}},
                {"LEFT_ID": "adj", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {
                    "DEP": "prep",
                    "LOWER": {"IN": list(DEPENDENT_PREPS)}
                }},
                {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "pobj", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["pobj", "pcomp"]}}}
            ]
        ]
    },
    {
        "id": "noun_dependent_preposition",
        "type": "collocation",
        "priority": 80,
        "name": "Noun + Preposition Collocation (access to / demand for)",
        "tree_patterns": [
            [
                {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN"}},
                {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {
                    "POS": "ADP",
                    "DEP": "prep",
                    "LOWER": {"IN": list(DEPENDENT_PREPS)}
                }},
                {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "pobj", "RIGHT_ATTRS": {
                    "DEP": {"IN": ["pobj", "pcomp"]}}}
            ]
        ]
    }
]


# ------------------------------------------------------------------------------
# 2. Expression Pattern Matcher Engine
# ------------------------------------------------------------------------------
class ExpressionPatternEngine:
    """
    Compiles and executes declarative DependencyMatcher patterns for multi-word expressions.
    Provides fast (<1ms), zero-token, robust multi-word expression and collocation extraction.
    """
    _instance: Optional["ExpressionPatternEngine"] = None
    _matcher: Optional[DependencyMatcher] = None
    _pattern_map: Dict[str, Dict[str, Any]] = {}

    def __init__(self, vocab: spacy.vocab.Vocab):
        self.vocab = vocab
        self.matcher = DependencyMatcher(vocab)
        self.pattern_map: Dict[str, Dict[str, Any]] = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Compiles declarative patterns into the DependencyMatcher."""
        for pat in DECLARATIVE_EXPRESSION_PATTERNS:
            pat_id = pat["id"]
            self.pattern_map[pat_id] = pat
            tree_patterns = pat.get("tree_patterns", [])
            if tree_patterns:
                try:
                    self.matcher.add(pat_id, tree_patterns)
                except Exception:
                    pass

    @classmethod
    def get_engine(cls, nlp_or_vocab: Any) -> "ExpressionPatternEngine":
        """Singleton factory pattern to cache compiled matcher across requests."""
        vocab = getattr(nlp_or_vocab, "vocab", nlp_or_vocab)
        if cls._instance is None or cls._instance.vocab != vocab:
            cls._instance = cls(vocab)
        return cls._instance

    def match_sentence(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """
        Matches sentence Doc against compiled expression patterns.
        Resolves phrase, canonical formula, type, and contextual transitivity.
        """
        matches = self.matcher(doc)
        if not matches:
            return []

        matched_results: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[str, int]] = set()

        for match_id, token_indices in matches:
            string_id = self.vocab.strings[match_id]
            pattern_meta = self.pattern_map.get(string_id, {})
            tokens = [doc[i] for i in token_indices]
            root_token = tokens[0] if tokens else None
            if not root_token:
                continue

            # Deduplicate by (pattern_id, root_token.i)
            dedup_key = (string_id, root_token.i)
            if dedup_key in seen_keys:
                continue

            extracted = self._resolve_expression(string_id, pattern_meta, doc, tokens)
            if extracted:
                seen_keys.add(dedup_key)
                matched_results.append(extracted)

        # Sort by priority descending
        matched_results.sort(key=lambda x: x["priority"], reverse=True)

        # Token-overlap containment filter: If a lower-priority match has its core token indices
        # completely contained within a higher-priority match (specifically for multi-word prepositional
        # frames like 'in response to' superseding 'response to', or 'at the expense of' superseding 'expense of'),
        # suppress the subsumed sub-fragment.
        filtered_results: List[Dict[str, Any]] = []
        for res in matched_results:
            res_indices = set(res.get("token_indices", []))
            if not res_indices:
                filtered_results.append(res)
                continue
            is_subsumed = False
            for higher in filtered_results:
                higher_indices = set(higher.get("token_indices", []))
                # Suppress lower-priority sub-fragments subsumed by a higher-priority unit
                # e.g., 'keep in' ([14, 15]) subsumed by 'keep in touch with' ([14, 15, 16])
                # e.g., 'response to' subsumed by 'in response to'
                if res_indices.issubset(higher_indices):
                    is_subsumed = True
                    break
            if not is_subsumed:
                filtered_results.append(res)

        return filtered_results

    def _resolve_expression(
        self,
        pattern_id: str,
        meta: Dict[str, Any],
        doc: spacy.tokens.Doc,
        tokens: List[spacy.tokens.Token]
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts phrase, canonical formula, and metadata based on the pattern and sentence context.
        """
        root_verb = tokens[0]
        v_lemma = root_verb.lemma_.lower()
        expr_type = meta.get("type", "collocation")
        priority = meta.get("priority", 50)

        # 1. Three-part phrasal prepositional verb (e.g. look forward to, run out of)
        if pattern_id == "verb_particle_prep_three_part":
            particle = next((t for t in tokens if t.dep_ in ("prt", "advmod") and t.i > root_verb.i), None)
            prep = next((t for t in tokens if t.dep_ == "prep" and t.i > root_verb.i), None)
            if particle and prep and (0 < prep.i - particle.i <= 2):
                p_text = particle.text.lower()
                prep_text = prep.text.lower()
                cand_phrase = f"{v_lemma} {p_text} {prep_text}"
                # Must be attested in LDOCE as a true phrasal verb unit or explicit pattern
                # to prevent incidental adjunct prepositions (e.g. 'show up on time') from hijacking
                try:
                    from librarian.linguistics import LinguisticEngine
                    entry = LinguisticEngine.get_ldoce_entry(v_lemma)
                    pvs = (entry.get("phrasal_verbs") or []) if entry else []
                    is_true_three_part = any(pv.get("phrase") == cand_phrase for pv in pvs)
                    if not is_true_three_part:
                        base_pv = f"{v_lemma} {p_text}"
                        for pv in pvs:
                            if pv.get("phrase") == base_pv:
                                for s in pv.get("senses") or []:
                                    for pat in s.get("patterns") or []:
                                        if cand_phrase in pat.lower():
                                            is_true_three_part = True
                                            break
                                    if is_true_three_part:
                                        break
                    if not is_true_three_part:
                        return None
                except Exception:
                    pass

                phrase = cand_phrase
                formula = f"{v_lemma} {p_text} {prep_text} [sth/sb]"
                return {
                    "pattern_id": pattern_id,
                    "phrase": phrase,
                    "pattern_formula": formula,
                    "type": "phrasal verb",
                    "priority": priority,
                    "token_indices": [root_verb.i, particle.i, prep.i]
                }
            return None

        # 2. Keep/Stay/Get in touch (with)
        elif pattern_id == "verb_touch_in_with":
            has_with = any(c.text.lower() == "with" for c in root_verb.children)
            if not has_with:
                for c in root_verb.children:
                    if any(gc.text.lower() == "with" for gc in c.children):
                        has_with = True
                        break
            phrase = f"{v_lemma} in touch with" if has_with else f"{v_lemma} in touch"
            formula = f"{v_lemma} in touch with [sb]" if has_with else f"{v_lemma} in touch"
            return {
                "pattern_id": pattern_id,
                "phrase": phrase,
                "pattern_formula": formula,
                "type": "idiom",
                "priority": priority,
                "token_indices": [t.i for t in tokens]
            }

        # 3. Standard phrasal verb (Verb + Particle)
        elif pattern_id == "verb_particle_standard":
            particle = next((t for t in tokens if t.dep_ in ("prt", "advmod") and t.i > root_verb.i), None)
            if not particle:
                return None
            p_text = particle.text.lower()
            if p_text not in COMMON_PARTICLES:
                return None

            # Contextual Transitivity Audit
            has_dobj = any(c.dep_ in ("dobj", "obj") for c in root_verb.children)
            is_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in root_verb.children)

            phrase = f"{v_lemma} {p_text}"
            if has_dobj:
                formula = f"{v_lemma} [sb/sth] {p_text}"
            elif is_passive:
                formula = f"{v_lemma} {p_text} [sth/sb]"
            else:
                formula = f"{v_lemma} {p_text}"

            return {
                "pattern_id": pattern_id,
                "phrase": phrase,
                "pattern_formula": formula,
                "type": "phrasal verb",
                "priority": priority,
                "token_indices": [root_verb.i, particle.i]
            }

        # 4. Light verb prepositional idiom (e.g. take into account)
        elif pattern_id == "light_verb_prepositional_idiom":
            prep = next((t for t in tokens if t.dep_ == "prep"), None)
            pobj = next((t for t in tokens if t.dep_ == "pobj"), None)
            if prep and pobj and (pobj.i - prep.i == 1):
                p_text = prep.text.lower()
                noun_text = pobj.lemma_.lower()
                phrase = f"{v_lemma} {p_text} {noun_text}"
                formula = f"{v_lemma} {p_text} {noun_text} [sth]"
                return {
                    "pattern_id": pattern_id,
                    "phrase": phrase,
                    "pattern_formula": formula,
                    "type": "idiom",
                    "priority": priority,
                    "token_indices": [root_verb.i, prep.i, pobj.i]
                }
            return None

        # 5. Idiomatic possessive construction (e.g. make up [one's] mind, lose [one's] temper)
        elif pattern_id == "idiomatic_possessive_frame":
            noun = next((t for t in tokens if t.dep_ == "dobj"), None)
            if not noun:
                return None
            noun_lemma = noun.lemma_.lower()
            if noun_lemma not in IDIOMATIC_POSS_NOUNS and noun.text.lower() not in IDIOMATIC_POSS_NOUNS:
                return None
            target_noun = noun_lemma if noun_lemma in IDIOMATIC_POSS_NOUNS else noun.text.lower()

            prt = next((c for c in root_verb.children if c.dep_ == "prt"), None)
            if prt:
                phrase = f"{v_lemma} {prt.text.lower()} [one's] {target_noun}"
                formula = f"{v_lemma} {prt.text.lower()} [one's] {target_noun}"
            else:
                phrase = f"{v_lemma} [one's] {target_noun}"
                formula = f"{v_lemma} [one's] {target_noun}"

            return {
                "pattern_id": pattern_id,
                "phrase": phrase,
                "pattern_formula": formula,
                "type": "collocation",
                "priority": priority,
                "token_indices": [t.i for t in tokens]
            }

        # 6. Transitive verb + preposition collocation (e.g. provide [sb] with [sth])
        elif pattern_id == "verb_transitive_preposition_collocation":
            prep = next((t for t in tokens if t.dep_ == "prep" and t.i > root_verb.i), None)
            if not prep:
                return None
            p_text = prep.text.lower()
            expected_prep = COLLOCATION_VERB_PREPS.get(v_lemma)
            if expected_prep != p_text:
                return None

            phrase = f"{v_lemma} ... {p_text}"
            formula = f"{v_lemma} [sb] {p_text} [sth]"
            return {
                "pattern_id": pattern_id,
                "phrase": phrase,
                "pattern_formula": formula,
                "type": "collocation",
                "priority": priority,
                "token_indices": [t.i for t in tokens]
            }

        # 7. Intransitive prepositional verb (e.g. rely on, focus on, consist of)
        elif pattern_id == "verb_intransitive_dependent_preposition":
            prep = next((t for t in tokens if t.dep_ == "prep" and t.i > root_verb.i), None)
            if not prep or (prep.i - root_verb.i > 2):
                return None
            # Must not have direct object
            if any(c.dep_ in ("dobj", "obj") for c in root_verb.children):
                return None
            p_text = prep.text.lower()
            if p_text not in DEPENDENT_PREPS and p_text != COLLOCATION_VERB_PREPS.get(v_lemma):
                return None

            pobj_lemma = None
            pobj_proper = False
            pobj_nodes = [p for p in prep.children if p.dep_ == "pobj"]
            if pobj_nodes:
                pobj_tok = pobj_nodes[0]
                pobj_lemma = pobj_tok.lemma_.lower()
                pobj_proper = pobj_tok.pos_ == "PROPN" or bool(
                    pobj_tok.text[:1].isupper() and not pobj_tok.is_sent_start
                )

            phrase = f"{v_lemma} {p_text}"
            formula = f"{v_lemma} {p_text} [sth/sb]"
            return {
                "pattern_id": pattern_id,
                "phrase": phrase,
                "pattern_formula": formula,
                "pobj": pobj_lemma,
                "pobj_proper": pobj_proper,
                "type": "phrasal verb",
                "priority": priority,
                "token_indices": [root_verb.i, prep.i]
            }

        # 8. Three-part prepositional noun frame (e.g. in response to, at the expense of, in addition to)
        elif pattern_id == "prepositional_noun_frame":
            prep1 = tokens[0]
            noun = tokens[1]
            prep2 = tokens[2]
            p1_text = prep1.text.lower()
            p2_text = prep2.text.lower()
            n_lemma = noun.lemma_.lower()
            n_text = noun.text.lower()
            if n_lemma not in FRAME_PREPOSITION_NOUNS and n_text not in FRAME_PREPOSITION_NOUNS:
                return None

            noun_surface = n_text if n_text in FRAME_PREPOSITION_NOUNS else n_lemma
            has_the = any(c.text.lower() == "the" for c in noun.children)
            det_prefix = "the " if has_the else ""
            phrase = f"{p1_text} {det_prefix}{noun_surface} {p2_text}".strip()
            formula = f"{phrase} [sth/sb]"

            return {
                "pattern_id": pattern_id,
                "phrase": phrase,
                "pattern_formula": formula,
                "type": "set phrase",
                "priority": priority,
                "token_indices": [t.i for t in tokens]
            }

        # 9. Adjective + dependent preposition collocation (e.g. aware of, interested in, capable of)
        elif pattern_id == "adj_dependent_preposition":
            adj_tok = tokens[0]
            prep = tokens[1]
            a_lemma = adj_tok.lemma_.lower()
            p_text = prep.text.lower()

            # Calibrate against ADJ_DEPENDENT_PREPS
            expected_prep = ADJ_DEPENDENT_PREPS.get(a_lemma)
            if not expected_prep or expected_prep != p_text:
                return None

            phrase = f"{a_lemma} {p_text}"
            formula = f"{a_lemma} {p_text} [sth/sb]"

            return {
                "pattern_id": pattern_id,
                "phrase": phrase,
                "pattern_formula": formula,
                "type": "collocation",
                "priority": priority,
                "token_indices": [adj_tok.i, prep.i]
            }

        # 10. Noun + dependent preposition collocation (e.g. access to, demand for, effect on)
        elif pattern_id == "noun_dependent_preposition":
            noun_tok = tokens[0]
            prep = tokens[1]
            n_lemma = noun_tok.lemma_.lower()
            p_text = prep.text.lower()

            # Calibrate against NOUN_DEPENDENT_PREPS
            expected_prep = NOUN_DEPENDENT_PREPS.get(n_lemma)
            if not expected_prep or expected_prep != p_text:
                return None

            # Check if noun has an adjectival modifier forming an authentic compound collocation (e.g. 'adverse effect on')
            amod_toks = [c for c in noun_tok.children if c.dep_ == "amod"]
            amod_prefix = ""
            if amod_toks:
                amod_candidate = amod_toks[0].lemma_.lower()
                from librarian.linguistics import LinguisticEngine
                if LinguisticEngine.is_attested_phrase(f"{amod_candidate} {n_lemma}"):
                    amod_prefix = f"{amod_candidate} "

            phrase = f"{amod_prefix}{n_lemma} {p_text}".strip()
            formula = f"{amod_prefix}{n_lemma} {p_text} [sth/sb]".strip()

            return {
                "pattern_id": pattern_id,
                "phrase": phrase,
                "pattern_formula": formula,
                "type": "collocation",
                "priority": priority,
                "token_indices": [noun_tok.i, prep.i]
            }

        return None
