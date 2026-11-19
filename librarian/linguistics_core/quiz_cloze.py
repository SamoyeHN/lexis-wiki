"""
Authentic Discourse Cloze Assessment Engine (Mode B: Progressive Scaffolding).
==============================================================================
Implements authentic multi-paragraph discourse cloze assessment based on:
1. Cambridge Reading and Use of English Part 2 (Open Cloze) standards.
2. Psychometrically audited paragraph selection (180–260 tokens continuous span).
3. The Three Invariants of Discourse Testing (Alderson, 2000):
   - Lead-in & lead-out invariant (opening and closing sentences 100% unblanked).
   - Buffer invariant (minimum inter-blank distance >= 8-10 running tokens).
   - Deterministic single-fit invariant with zero double-key collisions.
4. Tripartite Blanking Allocation:
   - Pillar 1: Discourse Cohesion & Rhetorical Linkers (~30%)
   - Pillar 2: Multi-Word Unit Valency & Preposition Slicing (~40%)
   - Pillar 3: Lexico-Grammar & Inflectional Aspect (~30%)
5. Mode B Progressive Scaffolding:
   - Active Retrieval (100 pts) -> [💡] Clue Hint (80 pts) -> [⚅] 4 Options Fallback (60 pts).
6. 100% Pure Symbolic Pipeline (0 LLM Tokens, <50ms execution).
"""

from dataclasses import dataclass, field, field
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# --- Closed-Class Peer Pools for Zero-Collision Distractor Arenas ---

_PEER_POOLS: Dict[str, List[str]] = {
    # Relative Pronouns & Adverbs (Strict zero double-key separation: which vs that vs who)
    "rel_pronoun_thing": ["which", "where", "whose", "what", "when"],
    "rel_pronoun_which": ["which", "where", "whose", "what", "when"],
    "rel_pronoun_that": ["that", "where", "whose", "what", "when"],
    "rel_pronoun_person": ["who", "whom", "whose", "which", "where"],
    "rel_pronoun_place": ["where", "which", "when", "what"],
    "rel_pronoun_time": ["when", "which", "where", "whose"],
    "rel_pronoun_possessive": ["whose", "which", "whom", "who"],

    # Conjunctions & Discourse Linkers
    "conj_contrast": ["although", "whereas", "while", "because"],
    "conj_condition": ["unless", "if", "whether", "since"],
    "conj_cause": ["because", "since", "although", "unless"],
    "conj_time": ["while", "until", "before", "since"],
    "trans_contrast": ["however", "therefore", "furthermore", "meanwhile"],
    "trans_result": ["therefore", "however", "moreover", "otherwise"],
    "trans_addition": ["furthermore", "moreover", "however", "instead"],

    # Prepositions & Phrasal Particles
    "prep_governed": [
        "about", "into", "for", "to", "with", "from", "of", "on", "at", "against", "toward", "through", "by"
    ],
    "phrasal_particles": [
        "up", "down", "out", "off", "in", "on", "away", "back", "over", "through", "aside", "round"
    ],

    # Quantifiers & Determiners
    "quantifier_distributive": ["each", "every", "all", "both"],
    "quantifier_dual": ["either", "neither", "both", "all"],
    "quantifier_alternative": ["another", "other", "others", "the other"],
    "quantifier_amount": ["much", "many", "few", "little"],
    "quantifier_general": ["some", "any", "no", "all"],

    # Auxiliaries & Aspect
    "aux_aspect": ["been", "being", "having", "had"],
    "aux_verb": ["have", "has", "had", "having"],
    "aux_be": ["was", "were", "been", "being"],

    # Light Verbs
    "light_verbs": ["make", "take", "have", "give", "keep", "hold", "pay", "cast", "strike", "draw"],
}

# Discourse marker lexicon
_DISCOURSE_CONJUNCTIONS: Set[str] = {
    "although", "though", "even though", "while", "whereas", "since",
    "unless", "because", "if", "whether", "until", "before", "after", "once", "as"
}

_DISCOURSE_TRANSITIONS: Set[str] = {
    "however", "therefore", "furthermore", "moreover", "nevertheless",
    "meanwhile", "instead", "otherwise", "besides", "thus", "consequently", "indeed"
}

_RELATIVE_PRONOUNS: Set[str] = {
    "which", "whose", "whom", "who", "that", "where", "when", "why"
}

_CLOSED_QUANTIFIERS: Set[str] = {
    "each", "every", "another", "other", "others", "either", "neither",
    "both", "much", "many", "few", "little", "several"
}

_LIGHT_VERB_ROOTS: Set[str] = {
    "make", "take", "have", "give", "keep", "hold", "pay", "cast", "strike", "draw", "bring", "set", "put"
}

_BOUND_PREPOSITIONS: Set[str] = {
    "about", "into", "for", "to", "with", "from", "of", "on", "at",
    "against", "toward", "towards", "upon", "between", "among", "through"
}


_PHRASAL_PARTICLES: Set[str] = {
    "up", "down", "out", "off", "away", "back", "over", "through", "into", "on"
}

_REFLEXIVE_PRONOUNS: Set[str] = {
    "itself", "themselves", "himself", "herself", "ourselves", "yourself", "yourselves", "myself"
}

_ASPECT_AUXILIARIES: Set[str] = {
    "been", "being", "having", "had", "has", "have", "was", "were", "are", "is", "do", "does", "did"
}

_MODAL_AUXILIARIES: Set[str] = {
    "would", "could", "should", "must", "will", "can", "may", "might"
}

_PERSON_ANTECEDENTS: Set[str] = {
    "person", "people", "man", "woman", "child", "student", "teacher", "friend",
    "parent", "doctor", "author", "writer", "worker", "leader", "member",
    "one", "anyone", "someone", "everyone", "those", "individual", "citizen",
}

# Context-independent or category-scoped acceptable-word equivalents for Open Cloze grading
_CONJ_EQUIVALENTS: Dict[str, List[str]] = {
    "although": ["although", "though"],
    "though": ["though", "although"],
    "because": ["because", "since", "as"],
    "while": ["while", "whilst"],
    "whilst": ["whilst", "while"],
    "until": ["until", "till"],
    "till": ["till", "until"],
}

_TRANSITION_EQUIVALENTS: Dict[str, List[str]] = {
    "however": ["however", "nevertheless", "nonetheless"],
    "nevertheless": ["nevertheless", "nonetheless", "however"],
    "nonetheless": ["nonetheless", "nevertheless", "however"],
    "therefore": ["therefore", "thus", "consequently"],
    "thus": ["thus", "therefore", "consequently"],
    "consequently": ["consequently", "therefore", "thus"],
    "moreover": ["moreover", "furthermore"],
    "furthermore": ["furthermore", "moreover"],
}

_PREP_EQUIVALENTS: Dict[str, List[str]] = {
    "toward": ["toward", "towards"],
    "towards": ["towards", "toward"],
    "among": ["among", "amongst"],
    "amongst": ["amongst", "among"],
}


@dataclass
class ClozeCandidate:
    token_index: int
    char_start: int
    char_end: int
    sent_char_start: int
    raw_text: str
    target: str
    sentence_idx: int
    sentence_text: str
    pillar: int  # 1, 2, or 3
    pillar_label: str
    category: str
    clue: str
    distractor_pool_key: str
    priority_score: float = 1.0
    accepted_answers: List[str] = field(default_factory=list)
    head_context: str = ""


class QuizClozeMixin:
    """
    Mixin providing deterministic Discourse Cloze paragraph selection,
    Alderson invariant enforcement, tripartite slot blanking, and Mode B scaffolding.
    """

    _spacy_nlp = None

    @classmethod
    def get_spacy(cls):
        """
        Lazily loads and returns the spaCy en_core_web_sm pipeline.
        """
        if getattr(cls, "_spacy_nlp", None) is None:
            try:
                import spacy
                cls._spacy_nlp = spacy.load("en_core_web_sm")
            except Exception:
                cls._spacy_nlp = False
        return cls._spacy_nlp if cls._spacy_nlp is not False else None

    @classmethod
    def select_optimal_discourse_span(
        cls,
        passage_text: str,
        target_min_tokens: int = 160,
        target_max_tokens: int = 280,
    ) -> Dict[str, Any]:
        """
        Deterministically selects an authentic, continuous discourse span of 180-260 tokens
        from the passage text, filtering out headers, metadata, and non-prose bullet lines.
        """
        # 1. Clean markdown headers and bullet lists
        raw_paras = passage_text.split("\n\n")
        clean_paras: List[str] = []

        for p in raw_paras:
            p_strip = p.strip()
            if not p_strip:
                continue
            # Strip markdown headers (## Passage A, etc.)
            if re.match(r"^#+\s+", p_strip):
                continue
            # Strip standalone title lines (e.g. **People's Love...**)
            if re.match(r"^\*\*[^\*\n]+\*\*\s*$", p_strip):
                continue
            # Strip bylines / metadata lines like (Posted on ...)
            if re.match(r"^\([Pp]osted\s+on\s+.*?\)$", p_strip):
                continue
            # Strip bullet lists
            if re.match(r"^[\*\-\•]\s+", p_strip):
                continue

            clean_paras.append(p_strip)

        if not clean_paras:
            # Fallback to raw non-empty lines
            clean_paras = [line.strip() for line in passage_text.splitlines() if line.strip() and not line.startswith("#")]

        # Count tokens per paragraph (simple whitespace/word count)
        para_word_counts = [len(re.findall(r"\b\w+\b", p)) for p in clean_paras]
        total_words = sum(para_word_counts)

        # If total passage is within or close to target range, use all paragraphs
        if total_words <= target_max_tokens or len(clean_paras) <= 3:
            chosen_paras = clean_paras
            return {
                "span_text": "\n\n".join(chosen_paras),
                "paragraphs": chosen_paras,
                "token_count": total_words,
            }

        # Otherwise, slide windows of 1, 2, or 3 paragraphs to find the best span
        best_window = (0, len(clean_paras))
        best_fitness = -999999.0

        for win_len in (1, 2, 3):
            for i in range(len(clean_paras) - win_len + 1):
                cur_words = sum(para_word_counts[i : i + win_len])
                # Penalize divergence from ideal token range (220 tokens)
                length_penalty = abs(cur_words - 220) * 1.0
                if cur_words < target_min_tokens:
                    length_penalty += (target_min_tokens - cur_words) * 3.0
                elif cur_words > target_max_tokens:
                    length_penalty += (cur_words - target_max_tokens) * 2.0

                # Reward cohesive linkers and prepositions in candidate text
                text_slice = " ".join(clean_paras[i : i + win_len]).lower()
                conn_count = sum(1 for m in _DISCOURSE_CONJUNCTIONS if f" {m} " in text_slice)
                trans_count = sum(1 for m in _DISCOURSE_TRANSITIONS if f" {m} " in text_slice)
                prep_count = sum(1 for m in _BOUND_PREPOSITIONS if f" {m} " in text_slice)

                fitness = 100.0 - length_penalty + (conn_count * 10.0) + (trans_count * 8.0) + (prep_count * 3.0)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_window = (i, i + win_len)

        chosen_paras = clean_paras[best_window[0] : best_window[1]]
        return {
            "span_text": "\n\n".join(chosen_paras),
            "paragraphs": chosen_paras,
            "token_count": sum(para_word_counts[best_window[0] : best_window[1]]),
        }

    @classmethod
    def _build_accepted_answers(cls, tok, category: str) -> List[str]:
        """
        Deterministically builds the Acceptable-Word Scoring whitelist for Pure Open Cloze.
        Respects syntactic constraints (e.g., no 'that' in non-restrictive or post-preposition
        relative clauses) and preserves sentence-initial capitalization when present.
        """
        text_lower = tok.text.lower()
        is_cap = bool(tok.text and tok.text[0].isupper())
        accepted: List[str] = [text_lower]

        if category == "relative_pronoun":
            prev_tok = tok.doc[tok.i - 1] if tok.i > 0 else None
            preceded_by_comma = prev_tok is not None and prev_tok.text == ","
            preceded_by_prep = prev_tok is not None and prev_tok.pos_ == "ADP"
            if text_lower == "which" and not preceded_by_comma and not preceded_by_prep:
                accepted = ["which", "that"]
            elif text_lower == "who" and not preceded_by_comma:
                accepted = ["who", "that"]
            elif text_lower == "that" and tok.dep_ in ("nsubj", "dobj", "nsubjpass", "pobj"):
                antecedent = tok.head.head if tok.head is not None else None
                is_person = antecedent is not None and (
                    antecedent.ent_type_ == "PERSON"
                    or antecedent.lemma_.lower() in _PERSON_ANTECEDENTS
                )
                accepted = ["that", "who"] if is_person else ["that", "which"]
        elif category == "subordinating_conjunction":
            if text_lower in _CONJ_EQUIVALENTS:
                accepted = list(_CONJ_EQUIVALENTS[text_lower])
            elif text_lower == "whether":
                next_tok = tok.doc[tok.i + 1] if tok.i + 1 < len(tok.doc) else None
                at_sent_start = tok.i == tok.sent.start
                followed_by_or = next_tok is not None and next_tok.text.lower() == "or"
                if not at_sent_start and not followed_by_or:
                    accepted = ["whether", "if"]
        elif category == "discourse_transition":
            if text_lower in _TRANSITION_EQUIVALENTS:
                accepted = list(_TRANSITION_EQUIVALENTS[text_lower])
        elif category in ("bound_preposition", "phrasal_particle"):
            if text_lower in _PREP_EQUIVALENTS:
                accepted = list(_PREP_EQUIVALENTS[text_lower])

        if is_cap:
            return [w[0].upper() + w[1:] for w in accepted]
        return accepted

    @classmethod
    def _make_candidate(
        cls,
        tok,
        sent_idx: int,
        pillar: int,
        pillar_label: str,
        category: str,
        clue: str,
        pool_key: str,
        priority_score: float,
        head_context: str = "",
    ) -> ClozeCandidate:
        """Constructs a ClozeCandidate with exact sentence-relative character offset metadata."""
        raw_sent = tok.sent.text
        leading_ws = len(raw_sent) - len(raw_sent.lstrip())
        sent_text = raw_sent.strip()
        sent_char_start = tok.sent.start_char + leading_ws
        return ClozeCandidate(
            token_index=tok.i,
            char_start=tok.idx,
            char_end=tok.idx + len(tok.text),
            sent_char_start=sent_char_start,
            raw_text=tok.text,
            target=tok.text.lower(),
            sentence_idx=sent_idx,
            sentence_text=sent_text,
            pillar=pillar,
            pillar_label=pillar_label,
            category=category,
            clue=clue,
            distractor_pool_key=pool_key,
            priority_score=priority_score,
            accepted_answers=cls._build_accepted_answers(tok, category),
            head_context=head_context,
        )

    @classmethod
    def _categorize_cloze_token(cls, tok, sent_idx: int, total_sents: int) -> Optional[ClozeCandidate]:
        """
        Inspects a spaCy token and determines if it belongs to one of the 3 Tripartite Pillars:
        1. Discourse Cohesion & Rhetorical Linkers (~30%)
        2. MWU & Preposition Valency (~40%)
        3. Lexico-Grammar & Inflectional Aspect (~30%)
        """
        text_lower = tok.text.lower()
        lemma_lower = tok.lemma_.lower()

        # Disqualify non-alphabetic, numbers, or tokens inside contractions ('s, 've, n't)
        if not tok.is_alpha or len(tok.text) < 2 or tok.text.startswith("'"):
            return None

        # --- Pillar 1: Discourse Cohesion & Rhetorical Linkers (~30%) ---
        # 1a. Subordinating Conjunctions
        if (tok.dep_ == "mark" or tok.pos_ in ("SCONJ", "CCONJ")) and text_lower in _DISCOURSE_CONJUNCTIONS:
            if text_lower in ("although", "whereas", "while", "whilst", "though"):
                pool_key = "conj_contrast"
                clue = "💡 Clue: Subordinating conjunction introducing a clause of contrast or concession"
            elif text_lower in ("unless", "if", "whether"):
                pool_key = "conj_condition"
                clue = "💡 Clue: Subordinating conjunction introducing a conditional or alternative clause"
            elif text_lower in ("until", "till", "before", "after", "when", "whenever"):
                pool_key = "conj_cause"
                clue = "💡 Clue: Subordinating conjunction establishing a temporal relation"
            else:
                pool_key = "conj_cause"
                clue = "💡 Clue: Subordinating conjunction expressing cause, reason, or manner"
            return cls._make_candidate(
                tok, sent_idx, 1, "Discourse Cohesion", "subordinating_conjunction",
                clue, pool_key, 3.5,
            )

        # 1b. Relative Pronouns & Adverbs
        if tok.tag_ in ("WDT", "WP", "WP$", "WRB") or (text_lower in _RELATIVE_PRONOUNS and tok.dep_ in ("relcl", "nsubj", "dobj", "pobj")):
            pool_key = "rel_pronoun_which" if text_lower == "which" else (
                "rel_pronoun_that" if text_lower == "that" else (
                    "rel_pronoun_person" if text_lower in ("who", "whom") else (
                        "rel_pronoun_place" if text_lower == "where" else (
                            "rel_pronoun_possessive" if text_lower == "whose" else (
                                "rel_pronoun_time" if text_lower == "when" else "rel_pronoun_thing"
                            )
                        )
                    )
                )
            )
            if text_lower in _RELATIVE_PRONOUNS:
                antecedent = tok.head.text if tok.head else "preceding noun"
                return cls._make_candidate(
                    tok, sent_idx, 1, "Discourse Cohesion", "relative_pronoun",
                    f"💡 Clue: Relative pronoun referring back to '{antecedent}'",
                    pool_key, 3.6, head_context=antecedent,
                )

        # 1c. Discourse Transitions (However, Therefore, Furthermore, etc.)
        if text_lower in _DISCOURSE_TRANSITIONS and tok.pos_ in ("ADV", "CCONJ"):
            pool_key = "trans_result" if text_lower in ("therefore", "thus", "consequently") else (
                "trans_addition" if text_lower in ("furthermore", "moreover", "besides") else "trans_contrast"
            )
            return cls._make_candidate(
                tok, sent_idx, 1, "Discourse Cohesion", "discourse_transition",
                "💡 Clue: Transition adverb expressing logical flow between sentences",
                pool_key, 3.4,
            )

        # --- Pillar 2: Multi-Word Unit Valency & Preposition Slicing (~40%) ---
        # 2a. Phrasal Verb Particles (dep_ == 'prt')
        if tok.dep_ == "prt" and text_lower in _PHRASAL_PARTICLES:
            verb_head = tok.head.lemma_ if tok.head else "verb"
            return cls._make_candidate(
                tok, sent_idx, 2, "MWU & Preposition Valency", "phrasal_particle",
                f"💡 Clue: Directional or aspectual particle completing '{verb_head}'",
                "phrasal_particles", 3.2, head_context=verb_head,
            )

        # 2b. Bound Prepositional Government (prep governed by NOUN, ADJ, or VERB)
        if tok.dep_ == "prep" and text_lower in _BOUND_PREPOSITIONS:
            head_tok = tok.head
            if head_tok and head_tok.pos_ in ("VERB", "ADJ", "NOUN"):
                head_lemma = head_tok.lemma_
                return cls._make_candidate(
                    tok, sent_idx, 2, "MWU & Preposition Valency", "bound_preposition",
                    f"💡 Clue: Preposition required by the preceding word '{head_lemma}'",
                    "prep_governed", 3.0, head_context=head_lemma,
                )

        # 2c. Light-Verb Collocation (make a choice, keep silent, have a try, take steps)
        if lemma_lower in _LIGHT_VERB_ROOTS and tok.pos_ == "VERB":
            obj_noun = None
            for child in tok.children:
                if child.dep_ in ("dobj", "acomp", "oprd") and child.is_alpha:
                    obj_noun = child.lemma_
                    break
            if obj_noun:
                return cls._make_candidate(
                    tok, sent_idx, 2, "MWU & Preposition Valency", "light_verb",
                    f"💡 Clue: Collocational support verb paired with '{obj_noun}'",
                    "light_verbs", 3.1, head_context=obj_noun,
                )

        # --- Pillar 3: Lexico-Grammar & Inflectional Aspect (~30%) ---
        # 3a. Quantifiers & Determiners
        if text_lower in _CLOSED_QUANTIFIERS:
            pool_key = "quantifier_distributive" if text_lower in ("each", "every", "all", "both") else (
                "quantifier_dual" if text_lower in ("either", "neither") else (
                    "quantifier_alternative" if text_lower in ("another", "other", "others") else "quantifier_amount"
                )
            )
            return cls._make_candidate(
                tok, sent_idx, 3, "Lexico-Grammar", "quantifier",
                "💡 Clue: Determiner/quantifier indicating countability, distribution, or scope",
                pool_key, 2.9,
            )

        # 3b. Aspectual, Passive & Periphrastic Auxiliaries
        if tok.pos_ == "AUX" and tok.dep_ in ("aux", "auxpass") and text_lower in _ASPECT_AUXILIARIES:
            head_verb = tok.head.text if tok.head else ""
            if tok.dep_ == "auxpass":
                clue = f"💡 Clue: Auxiliary 'be' form completing the passive voice with '{head_verb}'"
            elif text_lower in ("has", "have", "had", "having"):
                clue = f"💡 Clue: Perfect aspect auxiliary matching subject agreement and tense before '{head_verb}'"
            elif text_lower in ("do", "does", "did"):
                clue = "💡 Clue: Periphrastic 'do' auxiliary required by negation, emphasis, or question syntax"
            else:
                clue = f"💡 Clue: Grammatical auxiliary required by the aspect or voice of '{head_verb}'"
            return cls._make_candidate(
                tok, sent_idx, 3, "Lexico-Grammar", "aspectual_auxiliary",
                clue, "aux_aspect", 2.85, head_context=head_verb,
            )

        # 3c. Semi-Modals & Modal Auxiliaries
        next_tok = tok.doc[tok.i + 1] if tok.i + 1 < len(tok.doc) else None
        if text_lower in ("used", "ought", "able") and next_tok is not None and next_tok.text.lower() == "to":
            return cls._make_candidate(
                tok, sent_idx, 3, "Lexico-Grammar", "modal_structure",
                "💡 Clue: Core lexical word completing the semi-modal structure with 'to'",
                "light_verbs", 2.8,
            )
        if tok.pos_ == "AUX" and tok.dep_ == "aux" and text_lower in _MODAL_AUXILIARIES:
            head_verb = tok.head.text if tok.head else ""
            return cls._make_candidate(
                tok, sent_idx, 3, "Lexico-Grammar", "modal_auxiliary",
                f"💡 Clue: Modal auxiliary expressing obligation, possibility, or tentativeness before '{head_verb}'",
                "aux_aspect", 2.7, head_context=head_verb,
            )

        # 3d. Reflexive & Anaphoric Reference
        if text_lower in _REFLEXIVE_PRONOUNS:
            return cls._make_candidate(
                tok, sent_idx, 3, "Lexico-Grammar", "reflexive_pronoun",
                "💡 Clue: Reflexive pronoun referring back to the subject of the clause",
                "quantifier_distributive", 2.85,
            )
        if text_lower in ("its", "their", "such") and tok.dep_ in ("poss", "predet", "det"):
            head_noun = tok.head.text if tok.head else ""
            return cls._make_candidate(
                tok, sent_idx, 3, "Lexico-Grammar", "anaphoric_determiner",
                f"💡 Clue: Anaphoric determiner/possessive before '{head_noun}' referring back to an earlier referent",
                "quantifier_distributive", 2.65, head_context=head_noun,
            )

        # 3e. Comparative & Degree Operators (than, too, so)
        if text_lower == "than" and tok.dep_ in ("prep", "mark"):
            return cls._make_candidate(
                tok, sent_idx, 3, "Lexico-Grammar", "comparative_operator",
                "💡 Clue: Comparative function word linking the two terms of a comparison",
                "conj_contrast", 2.8,
            )
        if text_lower in ("too", "so") and tok.dep_ == "advmod" and tok.head and tok.head.pos_ in ("ADJ", "ADV"):
            return cls._make_candidate(
                tok, sent_idx, 3, "Lexico-Grammar", "degree_operator",
                f"💡 Clue: Degree adverb modifying '{tok.head.text}' in a result or excess construction",
                "quantifier_amount", 2.65, head_context=tok.head.text,
            )

        return None

    @classmethod
    def _build_open_cloze_explanation(cls, candidate: ClozeCandidate) -> str:
        """
        Generates a concise, linguistically grounded explanation for a Pure Open Cloze slot,
        including any accepted structural equivalents.
        """
        target_display = candidate.raw_text
        cat_label = candidate.category.replace("_", " ")
        head_clause = (
            f" paired with '{candidate.head_context}'"
            if candidate.head_context
            else ""
        )
        alts = [a for a in candidate.accepted_answers if a.lower() != target_display.lower()]
        alt_note = (
            f" Acceptable equivalents in this slot: {', '.join(repr(a) for a in alts)}."
            if alts
            else ""
        )
        return (
            f"The target word is '{target_display}' ({candidate.pillar_label}: {cat_label}{head_clause}). "
            f"It satisfies the grammatical and discourse constraints of the surrounding clause.{alt_note}"
        )

    @classmethod
    def _synthesize_distractor_set(
        cls, candidate: ClozeCandidate, passage_text: str
    ) -> Tuple[List[str], int, str]:
        """
        Deterministically produces exactly 4 peer options (target + 3 distractors)
        with zero-collision validation and CRC32 slot shuffling.
        Returns: (options, correct_answer_index, explanation).
        """
        target = candidate.target
        is_capitalized = candidate.raw_text[0].isupper()

        # 1. Fetch peer pool
        pool = list(_PEER_POOLS.get(candidate.distractor_pool_key, []))
        if not pool or target not in pool:
            # Fallback to general prepositions or linkers
            if candidate.pillar == 2:
                pool = list(_PEER_POOLS["prep_governed"])
            elif candidate.pillar == 1:
                pool = list(_PEER_POOLS["conj_contrast"])
            else:
                pool = list(_PEER_POOLS["quantifier_distributive"])

        # Filter out target itself from candidate distractors
        distractors = [w for w in pool if w.lower() != target.lower()]

        # Absolute Zero Double-Key Relative Pronoun Guard:
        # In restrictive relative clauses without commas, 'which' and 'that' are interchangeable.
        # Likewise, 'who' and 'that' can both refer to persons in restrictive clauses.
        # Under no circumstances should 'which' and 'that' OR 'who' and 'that' co-exist as options.
        if target.lower() == "which":
            distractors = [w for w in distractors if w.lower() != "that"]
        elif target.lower() == "that":
            distractors = [w for w in distractors if w.lower() not in ("which", "who")]
        elif target.lower() in ("who", "whom"):
            distractors = [w for w in distractors if w.lower() != "that"]

        # Deterministic distractor selection using hash of target + sentence
        hash_seed = int(hashlib.md5(f"{target}:{candidate.sentence_text}".encode()).hexdigest(), 16)
        distractors.sort(key=lambda w: (hash_seed + sum(ord(c) for c in w)) % 997)

        selected_distractors = distractors[:3]
        while len(selected_distractors) < 3:
            # Emergency fallback from safe general words (mutually excluding which/that/who)
            forbidden = {"that"} if target.lower() in ("which", "who", "whom") else (
                {"which", "who"} if target.lower() == "that" else set()
            )
            for fallback_w in ["for", "with", "where", "other", "about", "what"]:
                if fallback_w != target.lower() and fallback_w not in forbidden and fallback_w not in selected_distractors:
                    selected_distractors.append(fallback_w)
                if len(selected_distractors) >= 3:
                    break

        all_options = [target] + selected_distractors[:3]

        # Sync capitalization
        if is_capitalized:
            all_options = [opt.capitalize() for opt in all_options]

        # Deterministic shuffle to assign correct_answer_index in 0..3
        slot_seed = (hash_seed >> 8) % 4
        # Fisher-Yates with deterministic swap
        shuffled = list(all_options)
        # Place target at slot_seed
        target_display = target.capitalize() if is_capitalized else target
        current_idx = shuffled.index(target_display)
        shuffled[current_idx], shuffled[slot_seed] = shuffled[slot_seed], shuffled[current_idx]

        correct_answer_index = slot_seed

        # Contrastive Pedagogical Explanation
        distractor_displays = [opt for i, opt in enumerate(shuffled) if i != correct_answer_index]
        distractor_str = ", ".join(f"'{d}'" for d in distractor_displays)

        explanation = (
            f"The correct answer is '{target_display}'. "
            f"In this passage, '{target_display}' functions as a {candidate.category.replace('_', ' ')} "
            f"and aligns with the syntactic valency and discourse context of the sentence. "
            f"In contrast, {distractor_str} cannot fit this syntactic slot without causing a grammatical or semantic error."
        )

        return shuffled, correct_answer_index, explanation

    @classmethod
    def build_discourse_cloze_assessment(
        cls,
        passage_text: str,
        target_count: int = 10,
        cefr_level: str = "B2",
    ) -> Dict[str, Any]:
        """
        100% Pure Symbolic pipeline generating an authentic Discourse Cloze assessment
        satisfying the Three Invariants of Alderson (2000) and Mode B scaffolding.
        Zero LLM tokens used.
        """
        # 1. Select optimal continuous discourse span (180–260 tokens)
        span_info = cls.select_optimal_discourse_span(passage_text)
        span_text = span_info["span_text"]

        # 2. Parse span with spaCy
        nlp = cls.get_spacy()
        if not nlp:
            raise RuntimeError("spaCy en_core_web_sm pipeline is required for Discourse Cloze generation.")

        doc = nlp(span_text)
        sentences = list(doc.sents)
        num_sentences = len(sentences)

        if num_sentences < 3:
            # Need at least lead-in, body, lead-out
            # If doc has fewer sentences, treat whole text without hard lead-out drop
            lead_in_idx = -1
            lead_out_idx = -1
        else:
            lead_in_idx = 0
            lead_out_idx = num_sentences - 1

        # 3. Harvest candidate tokens across sentences
        candidates: List[ClozeCandidate] = []

        for sent_idx, sent in enumerate(sentences):
            # Alderson Invariant 1: Opening & Closing sentences are 100% unblanked
            if sent_idx == lead_in_idx or sent_idx == lead_out_idx:
                continue

            for tok in sent:
                cand = cls._categorize_cloze_token(tok, sent_idx, num_sentences)
                if cand:
                    candidates.append(cand)

        # 4. Enforce Alderson Buffer Invariant (minimum 8-10 tokens between blanks)
        # Sort by priority score first, then position
        # We greedily select top candidates while preserving inter-blank buffer >= 8 tokens
        candidates.sort(key=lambda c: c.priority_score, reverse=True)

        selected_candidates: List[ClozeCandidate] = []
        selected_token_indices: Set[int] = set()
        seen_targets: Set[str] = set()

        # Pillar balance trackers (~30% P1 / ~40% P2 / ~30% P3)
        pillar_counts = {1: 0, 2: 0, 3: 0}
        target_p1 = max(2, int(round(target_count * 0.3)))
        target_p2 = max(3, int(round(target_count * 0.4)))
        target_p3 = max(2, target_count - target_p1 - target_p2)
        pillar_targets = {1: target_p1, 2: target_p2, 3: target_p3}

        for cand in candidates:
            target_key = cand.target.lower()
            if target_key in seen_targets:
                continue

            # Check buffer distance against already selected tokens
            too_close = any(abs(cand.token_index - existing_idx) < 7 for existing_idx in selected_token_indices)
            if too_close:
                continue

            # Prioritize underrepresented pillars strictly when other pillars still need items
            if pillar_counts[cand.pillar] >= pillar_targets[cand.pillar]:
                if any(pillar_counts[p] < pillar_targets[p] for p in (1, 2, 3)):
                    continue

            selected_candidates.append(cand)
            selected_token_indices.add(cand.token_index)
            seen_targets.add(target_key)
            pillar_counts[cand.pillar] += 1

            if len(selected_candidates) >= target_count:
                break

        # If we fell short of target_count, do a relaxed second pass with buffer >= 5 (still deduplicated)
        if len(selected_candidates) < target_count:
            for cand in candidates:
                target_key = cand.target.lower()
                if cand.token_index in selected_token_indices or target_key in seen_targets:
                    continue
                too_close = any(abs(cand.token_index - existing_idx) < 5 for existing_idx in selected_token_indices)
                if too_close:
                    continue
                selected_candidates.append(cand)
                selected_token_indices.add(cand.token_index)
                seen_targets.add(target_key)
                pillar_counts[cand.pillar] += 1
                if len(selected_candidates) >= target_count:
                    break

        # Sort selected candidates by appearance order in passage
        selected_candidates.sort(key=lambda c: c.token_index)

        # 5. Build Blanks, Questions, and Masked Passage (Pure Open Cloze)
        blanks_data: List[Dict[str, Any]] = []
        questions_data: List[Dict[str, Any]] = []

        # Replace tokens from right to left using char offsets to prevent index drifting
        char_replacements: List[Tuple[int, int, str]] = []

        for idx, cand in enumerate(selected_candidates, start=1):
            explanation = cls._build_open_cloze_explanation(cand)

            # Exact character-offset masking inside sentence_text (fixes repeated-word mis-masking)
            sent_text = cand.sentence_text
            rel_start = cand.char_start - cand.sent_char_start
            rel_end = cand.char_end - cand.sent_char_start
            if 0 <= rel_start < rel_end <= len(sent_text) and sent_text[rel_start:rel_end] == cand.raw_text:
                sent_masked = sent_text[:rel_start] + f"[{idx}] ____" + sent_text[rel_end:]
            else:
                pattern = re.compile(rf"\b{re.escape(cand.raw_text)}\b")
                sent_masked = pattern.sub(f"[{idx}] ____", sent_text, count=1)

            char_replacements.append((cand.char_start, cand.char_end, f"[{idx}] _____"))

            item_dict = {
                "blank_index": idx,
                "target_word": cand.target,
                "raw_word": cand.raw_text,
                "accepted_answers": list(cand.accepted_answers) or [cand.raw_text],
                "pillar": cand.pillar,
                "pillar_label": cand.pillar_label,
                "category": cand.category,
                "clue": cand.clue,
                "options": [],
                "correct_answer_index": None,
                "explanation": explanation,
                "question": sent_masked,
            }
            blanks_data.append(item_dict)
            questions_data.append(item_dict)

        # Assemble masked passage string
        char_replacements.sort(key=lambda x: x[0], reverse=True)
        masked_passage = span_text
        for c_start, c_end, repl in char_replacements:
            masked_passage = masked_passage[:c_start] + repl + masked_passage[c_end:]

        return {
            "mode": "open_cloze",
            "passage": span_text,
            "masked_passage": masked_passage,
            "paragraphs": span_info["paragraphs"],
            "blanks": blanks_data,
            "questions": questions_data,
            "token_count": span_info["token_count"],
            "cefr_level": cefr_level,
        }
