"""
Corpus anchor evidence scoring, token selection, dependency richness, and sense alignment.
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


class AnchorEvidenceMixin:
    _COMMON_GRAMMAR_WORDS = {
        "in", "on", "at", "to", "for", "with", "of", "by", "from", "about",
        "into", "onto", "upon", "over", "under", "behind", "between", "among",
        "through", "during", "before", "after", "as", "against", "without",
        "the", "a", "an", "this", "that", "these", "those", "and", "or", "but"
    }

    @classmethod
    def _in_forbidden_word_set(cls, token: str, forbidden: Optional[Set[str]]) -> bool:
        """
        True if `token` equals or inflects/derives from any word in `forbidden`.
        Used to keep collocational anchors away from words that are themselves
        targets of the current batch (a batch-target anchor is unsatisfiable:
        the evaluator demands it in the stem while its leakage gate bans it).
        """
        if not token or not forbidden:
            return False
        t = token.lower().strip()
        if not t:
            return False
        for fw in forbidden:
            if not fw:
                continue
            if t == fw:
                return True
            if t in cls._COMMON_GRAMMAR_WORDS and fw not in cls._COMMON_GRAMMAR_WORDS:
                continue
            if len(t) >= 4 and len(fw) >= 4 and (t.startswith(fw[:4]) or fw.startswith(t[:4])):
                return True
            if cls.are_same_word_family(t, fw):
                return True
        return False

    @classmethod
    def _pick_anchor_token(
        cls,
        candidates: List[str],
        target_word: Optional[str] = None,
        forbidden: Optional[Set[str]] = None,
        prefer_pos: Optional[str] = None
    ) -> Optional[str]:
        """
        Selects the first meaningful single-word anchor from a list of collocation
        phrases. Rejects truncated function-word tokens (e.g. 'be', 'have', 'of')
        produced by naively splitting multi-word collocations, so collision
        clearance operates on a real lexical headword instead of a stopword.
        Also strictly rejects the target_word itself and its simple inflections,
        and any word in the optional `forbidden` set (batch-target words).
        `prefer_pos` ('verb' / 'noun' / 'adj') re-orders the pick with spaCy: a
        cascade that needs a governing verb must get the verb out of
        'a crime carries a penalty' ('carry'), not the first content word ('crime').
        """
        tw_clean = target_word.lower().strip() if target_word else ""

        def acceptable(token: str) -> bool:
            if not token or len(token) < 3 or token in cls._ANCHOR_STOPWORDS:
                return False
            if token.startswith("somebod") or token.startswith("someth"):
                return False
            if tw_clean and (token == tw_clean or cls.are_same_word_family(token, tw_clean)):
                return False
            if forbidden and cls._in_forbidden_word_set(token, forbidden):
                return False
            return True

        nlp = cls.get_spacy() if prefer_pos else None
        fallback: Optional[str] = None

        for cand in candidates:
            if not cand:
                continue
            # Split by whitespace, slashes, and common punctuation to find the authentic colloc partner
            parts = re.split(r"[/,\s]+", cand)
            clean = []
            for p in parts:
                token = re.sub(r"[^a-zA-Z]", "", p).lower()
                if not acceptable(token):
                    continue
                clean.append(token)
                if fallback is None:
                    fallback = token
            if not clean or nlp is None:
                continue
            try:
                doc = nlp(cand)
            except Exception:
                continue
            for tok in doc:
                if not cls._pos_matches_preference(tok, prefer_pos) or not tok.is_alpha:
                    continue
                lem = (tok.lemma_ or tok.text).lower()
                if acceptable(lem):
                    return lem
        return fallback

    @staticmethod
    def _pos_matches_preference(tok, prefer_pos: Optional[str]) -> bool:
        """spaCy POS gate used by _pick_anchor_token's prefer_pos re-ordering."""
        if prefer_pos == "verb":
            return tok.pos_ in ("VERB", "AUX")
        if prefer_pos == "noun":
            return tok.pos_ in ("NOUN", "PROPN")
        if prefer_pos == "adj":
            return tok.pos_ == "ADJ"
        return False

    @staticmethod
    def _token_in_phrases(token: str, phrases: List[str]) -> bool:
        """True when the token appears as a whole word inside any stored collocation
        phrase ('great' inside 'great luxury'). Raw-list membership fails for phrases,
        which is what made adjective collocations be typed as governing verbs."""
        if not token:
            return False
        pattern = re.compile(rf"\b{re.escape(token)}\b")
        return any(pattern.search((p or "").lower()) for p in phrases or [])

    @classmethod
    def _select_sense_aligned_anchor(
        cls,
        candidates: List[str],
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        target_word: Optional[str] = None,
        forbidden: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Selects the most semantically aligned single-word anchor from collocation candidates,
        grounded in the target item's authentic quote and curriculum definition (Sense-Specific Primacy).
        Prevents sense-drift where polysemous words select collocations from irrelevant senses.
        Strictly excludes the target_word itself from being selected as its own anchor, and
        excludes every word in the optional `forbidden` set (other batch targets).
        """
        if not candidates:
            return None

        quote_clean = (quote or "").lower()
        def_clean = (definition or "").lower()
        tw_clean = target_word.lower().strip() if target_word else ""

        # 1. Level 1 (Authentic Quote Match): If any collocation candidate occurs directly
        # in the source sentence quote, pick it immediately (zero hallucination, verbatim fidelity)
        for cand in candidates:
            token = cls._pick_anchor_token([cand], target_word=tw_clean, forbidden=forbidden)
            if token and re.search(r"\b" + re.escape(token) + r"\b", quote_clean):
                return token

        # 2. Level 2 (Definition Semantic Overlap): Score candidates by word overlap with definition and quote
        context_tokens = set(re.findall(r"[a-zA-Z]{3,}", (quote_clean + " " + def_clean)))
        context_tokens = context_tokens - cls._ANCHOR_STOPWORDS
        if tw_clean:
            context_tokens.discard(tw_clean)

        scored_candidates: List[Tuple[int, str]] = []
        for cand in candidates:
            token = cls._pick_anchor_token([cand], target_word=tw_clean, forbidden=forbidden)
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
    def _quote_evidence_strength(cls, quote: Optional[str]) -> Tuple[str, int]:
        """
        Grades the curriculum quote as physical evidence for sense-locking and anchor
        extraction. A two-word headline fragment ('Punctuality Pays!') is not a clause:
        it cannot license a valency frame nor disambiguate a polysemous headword, so it
        must not be allowed to define an anchor or lock a sense. Weak quotes push anchor
        selection onto the dictionary path and are reported in the blueprint as
        quote_provenance='weak' so downstream consumers can see the evidence tier.
        Returns (provenance, token_count) with provenance in {'strong', 'weak'}.
        """
        text = (quote or "").strip()
        tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
        if len(tokens) < 5:
            return "weak", len(tokens)
        nlp = cls.get_spacy()
        if nlp is None:
            return "weak", len(tokens)
        try:
            doc = nlp(text)
            # A usable quote must contain a real predicate: either a lexical verb,
            # or a complete copular clause with a subject and complement (e.g. 'The hospital staff were very helpful.')
            has_lexical_verb = any(
                tok.pos_ == "VERB" and tok.dep_ not in ("aux", "auxpass", "cop")
                for tok in doc
            )
            has_copular_clause = any(
                tok.pos_ in ("AUX", "VERB", "ADJ")
                and (tok.dep_ in ("ROOT", "cop") or tok.head == tok)
                and any(c.dep_ in ("acomp", "attr", "dobj", "prep", "advmod") for c in tok.children)
                and any(c.dep_ in ("nsubj", "nsubjpass") for c in tok.children)
                for tok in doc
            )
            if not has_lexical_verb and not has_copular_clause:
                return "weak", len(tokens)
        except Exception:
            return "weak", len(tokens)
        return "strong", len(tokens)

    # ---------------------------------------------------------------------
    # B1 - anchor evidence quality. A sentence may only stand as an anchor's
    # physical proof when it is a clause that actually shows the target word at
    # work: a heading ('**Punctuality Pays!**') and a bare fragment ('Why?')
    # demonstrate nothing, and among the sentences that do, the one where the
    # target carries the richest dependency structure is the one worth locking
    # a sense against.
    # ---------------------------------------------------------------------
    @classmethod
    def anchor_evidence_ok(cls, sentence: Optional[str], target_word: Optional[str] = None) -> bool:
        """Is this sentence physical evidence for the target word, or just a label?"""
        text = (sentence or "").strip()
        if not text:
            return False
        tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
        if len(tokens) < cls.ANCHOR_MIN_WORDS:
            return False
        # A bold heading line is typography, not usage.
        if text.startswith("**") and text.endswith("**"):
            return False
        # Metadata signature lines (e.g. blog post timestamps/bylines in parentheses) are metadata, not usage evidence.
        if text.startswith("(") and text.endswith(")"):
            inner = text[1:-1].strip()
            if re.search(r"\b(?:posted\s+on|written\s+by|published\s+on|by\s+[A-Z])\b", inner, re.IGNORECASE):
                return False
        if target_word and not cls.text_contains_form(text, target_word):
            return False
        nlp = cls.get_spacy()
        if nlp is None:
            return False
        try:
            doc = nlp(text)
        except Exception:
            return False
        # The sentence must contain a real predicate: 'Why?' has no clause to read evidence from.
        roots = [tok for tok in doc if tok.dep_ == "ROOT"]
        has_root_verb = bool(roots and any(tok.pos_ in ("VERB", "AUX") for tok in roots))
        if not has_root_verb:
            # Check for embedded finite clausal verbs (e.g. 'One of the things that tend to annoy... has to do with...')
            has_clause_verb = any(
                tok.pos_ in ("VERB", "AUX") and tok.dep_ in ("relcl", "ccomp", "xcomp", "advcl", "conj")
                for tok in doc
            )
            # Check for imperative sentence starter (e.g. 'Schedule meetings and appointments in advance.')
            # where spaCy misparses an initial verb as a compound noun modifier.
            has_imperative_starter = False
            first_tok = doc[0] if len(doc) > 0 else None
            if first_tok and first_tok.text[0].isupper():
                wn = cls.get_wordnet()
                first_clean = first_tok.text.lower()
                if wn and any(s.pos == "v" for s in wn.synsets(first_clean)):
                    has_imperative_starter = True

            if not (has_clause_verb or has_imperative_starter):
                return False
        return True

    @classmethod
    def anchor_dependency_richness(cls, sentence: Optional[str], target_word: Optional[str]) -> int:
        """How much syntax the target word actually participates in inside this sentence."""
        text = (sentence or "").strip()
        tw = (target_word or "").strip().lower()
        if not text or not tw:
            return 0
        nlp = cls.get_spacy()
        if nlp is None:
            return 0
        try:
            doc = nlp(text)
        except Exception:
            return 0
        forms = set(cls.inflected_forms(tw)) | {tw}
        best = 0
        for tok in doc:
            if tok.text.lower() not in forms and tok.lemma_.lower() not in forms:
                continue
            deps = {ch.dep_ for ch in tok.children}
            score = 0
            if deps & {"nsubj", "nsubjpass"}:
                score += 3
            if deps & {"obj", "dobj", "iobj", "attr", "acomp", "oprd", "pred"}:
                score += 3
            if deps & {"prep", "obl", "pobj", "agent", "dative", "complement"}:
                score += 2
            if deps & {"advcl", "ccomp", "xcomp", "acl", "relcl", "partmod", "infmod"}:
                score += 2
            if deps & {"advmod", "mod", "neg", "aux", "auxpass", "compound", "amod", "npadvmod"}:
                score += 1
            if tok.dep_ == "ROOT":
                score += 2
            if tok.pos_ == "ADJ" and tok.head is not None and tok.head.pos_ == "NOUN":
                score += 2
            best = max(best, score)
        return best

    @classmethod
    def pick_anchor_sentence(
        cls,
        target_word: str,
        sentences: Iterable[str]
    ) -> Tuple[str, int]:
        """
        The best qualifying sentence for an anchor, scored by how much syntax the target word
        governs in it. Returns ("", -1) when no candidate is evidence at all - the caller must
        then label its quote as unqualified instead of pretending it proved something.
        """
        best_sentence, best_score = "", -1
        for sent in sentences or []:
            if not cls.anchor_evidence_ok(sent, target_word):
                continue
            score = cls.anchor_dependency_richness(sent, target_word)
            if score > best_score:
                best_sentence, best_score = sent, score
        return best_sentence, best_score

    @classmethod
    def _sense_frame_patterns(cls, sense: Dict[str, Any], target_word: str) -> List[str]:
        """
        Returns only those PedLex patterns of a sense that actually contain the headword
        ('correlate with', 'serious about', 'it annoys somebody'). Patterns such as
        'got on his nerves' or 'keeps' are idiom/example fragments, not valency frames,
        and must never count as evidence that a preposition belongs to the target.
        """
        tw = (target_word or "").strip().lower()
        if not sense or not tw:
            return []
        nlp = cls.get_spacy()
        frames: List[str] = []
        for pat in sense.get("patterns", []) or []:
            pl = (pat or "").lower()
            if not pl:
                continue
            if re.search(rf"\b{re.escape(tw)}\b", pl):
                frames.append(pat)
                continue
            if nlp:
                try:
                    if any(t.lemma_.lower() == tw for t in nlp(pl)):
                        frames.append(pat)
                except Exception:
                    pass
        return frames

    @classmethod
    def _prep_anchor_sense_conflict(
        cls,
        target_word: str,
        prep: str,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        canonical_pos: Optional[str] = None
    ) -> bool:
        """
        Locked-Sense Frame Concordance Gate .

        A bound-preposition anchor is only legitimate when the PedLex sense this item
        actually locks (from its curriculum definition + authentic quote) licenses it.
        'annoy' locks to 'to make someone feel slightly angry', whose authentic frame is
        'it annoys somebody when/how/that' and whose authentic example is 'What annoyed
        him most was that he had received no apology.' — no preposition at all. The
        dictionary's registered 'annoy somebody on' belongs to a different construction,
        so welding 'on' into the frame produced an unsatisfiable item.

        The gate only vetoes prepositions a dictionary actively registers (a genuine
        valency claim). A preposition harvested from the authentic quote by the syntactic
        walker ('degrade into', 'autonomy from', 'log into') carries no dictionary claim
        to contradict, so the Pillar-1 syntactic path stays untouched.
        """
        tw = (target_word or "").strip().lower()
        p_low = (prep or "").strip().lower()
        if not tw or not p_low:
            return False
        entry = cls.get_ldoce_entry(tw)
        if not entry:
            return False
        senses = entry.get("senses", []) or []
        if not senses:
            return False
        locked_idx = cls._lock_sense(
            entry, definition=definition, quote=quote, target_pos=canonical_pos
        )
        if locked_idx is None or locked_idx >= len(senses):
            return False
        locked_sense = senses[locked_idx]

        frames = cls._sense_frame_patterns(locked_sense, tw)
        if not frames:
            return False  # locked sense publishes no valency frame → nothing to contradict
        if any(re.search(rf"\b{re.escape(p_low)}\b", f.lower()) for f in frames):
            return False  # locked sense explicitly licenses this preposition

        # The locked sense's own authentic examples may exhibit the frame directly
        for ex in locked_sense.get("examples", []) or []:
            if re.search(rf"\b{re.escape(tw)}\w*\s+(?:\w+\s+)?{re.escape(p_low)}\b", (ex or "").lower()):
                return False

        registered = p_low in cls.get_ldoce_preps(tw)
        if not registered:
            ped_pos = {"verb": "verb", "noun": "noun", "adj": "adj"}.get((canonical_pos or "").lower())
            registered = p_low in cls.get_pedlex_collocations(tw, pos=ped_pos).get("prep", [])
        if not registered:
            return False  # pure syntactic complement harvested from the quote — keep it

        return True  # registered valency claim that the locked sense does not license

