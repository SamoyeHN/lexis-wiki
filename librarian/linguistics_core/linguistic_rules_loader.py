# -*- coding: utf-8 -*-
"""Linguistic rules loader with cached hot-reloading and graceful fallbacks.

Centralizes declarative valency maps, collective noun ontologies, inversion triggers,
and academic register filters from `librarian/data/linguistic_rules.json`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("librarian")

_RULES_CACHE: Optional[Dict[str, Any]] = None
_RULES_MTIME: float = 0.0


def _get_rules_json_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "linguistic_rules.json"


def load_linguistic_rules() -> Dict[str, Any]:
    """Loads and caches linguistic rules from `librarian/data/linguistic_rules.json`."""
    global _RULES_CACHE, _RULES_MTIME
    json_path = _get_rules_json_path()
    try:
        if json_path.exists():
            mtime = json_path.stat().st_mtime
            if _RULES_CACHE is not None and mtime == _RULES_MTIME:
                return _RULES_CACHE
            data = json.loads(json_path.read_text(encoding="utf-8"))
            _RULES_CACHE = data
            _RULES_MTIME = mtime
            return data
    except Exception as e:
        logger.warning(f"Failed to load linguistic_rules.json ({e}); using memory fallback.")
    return _RULES_CACHE or {}


def get_adj_prep_valency_map() -> Dict[str, List[str]]:
    rules = load_linguistic_rules()
    return rules.get("adj_prep_valency_map", {
        "to": ["vulnerable", "immune", "conducive", "susceptible", "allergic", "attentive", "adjacent", "subordinate", "accessible"],
        "of": ["aware", "capable", "characteristic", "deprived", "indicative", "reminiscent", "composed", "devoid", "conscious"],
        "for": ["eligible", "notorious", "responsible", "suitable", "renowned", "adequate", "prepared", "qualified"],
        "with": ["consistent", "compatible", "acquainted", "associated", "compliant", "congruent", "comparable"],
        "from": ["distinct", "exempt", "absent", "derived", "detached", "remote"],
        "in": ["proficient", "inherent", "rich", "experienced", "versed", "lacking"],
        "on": ["reliant", "dependent", "contingent", "intent", "keen"],
    })


def get_academic_adj_families() -> List[List[str]]:
    rules = load_linguistic_rules()
    return rules.get("academic_adj_families", [
        ["comprehensive", "extensive", "thorough", "limited", "exhaustive", "restricted", "fragmented"],
        ["available", "accessible", "obtainable", "unavailable", "inaccessible", "restricted", "exclusive"],
        ["stable", "volatile", "constant", "variable", "consistent", "fluctuating", "fragile"],
        ["crucial", "fundamental", "marginal", "negligible", "vital", "trivial", "secondary"],
        ["vulnerable", "susceptible", "immune", "resistant", "resilient", "defenseless"],
        ["feasible", "viable", "impractical", "attainable", "unrealistic", "plausible"],
        ["explicit", "ambiguous", "apparent", "obscure", "evident", "vague", "distinct"],
        ["substantial", "considerable", "minimal", "negligible", "significant", "modest"],
        ["coherent", "consistent", "incompatible", "contradictory", "harmonious", "conflicting"],
    ])


def get_register_banned_words() -> Set[str]:
    rules = load_linguistic_rules()
    raw = rules.get("register_banned_words")
    if raw:
        return set(raw)
    return {
        "dwarf", "beast", "fauna", "flora", "benthos", "heterotroph", "amphidiploid", "diploid",
        "ogre", "giant", "goblin", "witch", "wizard", "fairy", "elf", "gnome", "troll",
        "swell", "cracking", "bully", "smashing", "peachy", "dandy", "nifty", "corking", "groovy", "slap-up", "bang-up",
        "sex", "sexy", "sexual", "erotic", "porn", "pornography", "nude", "naked", "suicide", "murder",
    }


def get_anchor_stopwords() -> Set[str]:
    rules = load_linguistic_rules()
    raw = rules.get("anchor_stopwords")
    if raw:
        return set(raw)
    return {
        "a", "an", "the", "of", "to", "in", "on", "for", "with", "and", "or", "at",
        "by", "from", "into", "onto", "out", "off", "up", "down", "as", "be", "is",
        "are", "was", "were", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "shall", "should", "can", "could", "may", "might",
        "must", "that", "this", "these", "those", "it", "its", "he", "she", "we",
        "they", "you", "i", "me", "my", "your", "his", "her", "their", "our",
        "each", "every", "some", "any", "all", "both", "few", "sth", "sb",
        "more", "most", "less", "least", "many", "much", "very", "quite",
        "rather", "such", "own", "other", "another", "same", "different",
        "particular", "certain", "various", "several", "enough", "only",
        "just", "even", "also", "well", "still", "yet", "already",
        "thing", "things", "stuff", "person", "people", "way", "ways", "item", "items",
        "something", "anything", "everything", "nothing", "someone", "somebody",
        "anyone", "anybody", "everyone", "everybody", "nobody", "one", "ones",
        "lot", "lots", "bit", "bits", "kind", "kinds", "sort", "sorts",
        "piece", "pieces", "pair", "pairs", "number", "amount", "rest",
    }


def get_adjunct_prep_pairs() -> Set[Tuple[str, str]]:
    rules = load_linguistic_rules()
    raw = rules.get("adjunct_prep_pairs")
    if raw:
        return {tuple(p) for p in raw if len(p) == 2}
    return {
        ("for", "example"), ("for", "instance"), ("in", "fact"), ("in", "general"),
        ("at", "first"), ("on", "average"), ("in", "particular"), ("in", "summary"),
        ("in", "total"), ("in", "brief"), ("in", "short"), ("in", "part"),
    }


def get_typical_complement_preps() -> Set[str]:
    rules = load_linguistic_rules()
    raw = rules.get("typical_complement_preps")
    if raw:
        return set(raw)
    return {"into", "to", "from"}


def get_collective_noun_ontology() -> Dict[str, List[Tuple[str, ...]]]:
    rules = load_linguistic_rules()
    raw = rules.get("collective_noun_ontology")
    if raw and isinstance(raw, dict):
        return {
            category: [tuple(group) for group in groups]
            for category, groups in raw.items()
        }
    return {
        "human_role": [
            ("staff", "personnel", "workforce"),
            ("crew",),
            ("cast", "troupe"),
            ("faculty",),
            ("jury", "panel", "tribunal"),
            ("audience", "crowd", "congregation", "throng", "mob"),
            ("committee", "council", "board", "commission", "assembly", "delegation"),
            ("band", "choir", "chorus", "orchestra"),
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


def get_predicative_only_adjs() -> Set[str]:
    rules = load_linguistic_rules()
    raw = rules.get("predicative_only_adjs")
    if raw:
        return set(raw)
    return {
        "asleep", "afraid", "alive", "alone", "ashamed", "aware", "awake", "content", "unable", "prone", "glad",
    }


def get_negative_inversion_adverbs() -> Set[str]:
    rules = load_linguistic_rules()
    raw = rules.get("negative_inversion_adverbs")
    if raw:
        return set(raw)
    return {
        "hardly", "scarcely", "barely",
        "seldom", "rarely",
        "little", "no sooner",
    }


def get_negative_inversion_phrases() -> Set[str]:
    rules = load_linguistic_rules()
    raw = rules.get("negative_inversion_phrases")
    if raw:
        return set(raw)
    return {
        "no sooner", "not only", "not until",
        "under no circumstances", "on no account", "in no way",
        "at no time", "by no means", "only then", "only later",
        "only when", "only after", "only by", "only if",
    }


def get_all_negative_inversion_triggers() -> Set[str]:
    return get_negative_inversion_adverbs() | get_negative_inversion_phrases()


def get_delexical_verbs() -> Set[str]:
    rules = load_linguistic_rules()
    raw = rules.get("delexical_verbs")
    if raw:
        return set(raw)
    return {
        "like", "want", "need", "get", "have", "has", "make", "use", "give", "take",
        "put", "see", "know", "feel", "seem", "look", "hear", "go", "come", "say", "try",
        "help", "let", "call", "find", "keep", "set", "turn", "show", "ask", "tell",
        "mean", "include", "bring", "send", "add", "appear", "become",
    }


def get_delexical_quantifier_nouns() -> Set[str]:
    rules = load_linguistic_rules()
    raw = rules.get("delexical_quantifier_nouns")
    if raw:
        return set(raw)
    return {
        "lot", "lots", "bit", "bits", "sort", "sorts", "kind", "kinds", "type", "types",
        "couple", "number", "majority", "minority", "plenty", "rest", "bunch", "load",
        "loads", "few", "many", "mass", "stuff", "thing", "things", "amount", "quantity",
    }


def get_noun_taxonomy_exclusion() -> Dict[str, Any]:
    rules = load_linguistic_rules()
    raw = rules.get("noun_taxonomy_exclusion")
    if raw and isinstance(raw, dict):
        return raw
    return {
        "lexfile_to_category": {
            "noun.person": "role_agent",
            "noun.group": "collective_agent",
            "noun.act": "event_action",
            "noun.event": "event_action",
            "noun.process": "event_action",
            "noun.location": "facility_place",
            "noun.artifact": "concrete_object",
            "noun.object": "concrete_object",
            "noun.substance": "concrete_object",
            "noun.food": "concrete_object",
            "noun.plant": "concrete_object",
            "noun.animal": "concrete_object",
            "noun.body": "concrete_object",
            "noun.quantity": "metonymic_container",
            "noun.attribute": "abstract_attribute",
            "noun.state": "abstract_attribute",
            "noun.cognition": "abstract_attribute",
            "noun.feeling": "abstract_attribute",
            "noun.relation": "abstract_attribute",
            "noun.phenomenon": "abstract_attribute",
        },
        "forbidden_categories": {
            "role_agent": ["event_action", "facility_place", "concrete_object"],
            "collective_agent": ["facility_place", "metonymic_container", "concrete_object"],
            "event_action": ["role_agent", "facility_place", "concrete_object"],
            "facility_place": ["role_agent", "collective_agent"],
            "concrete_object": ["abstract_attribute", "event_action"],
            "metonymic_container": ["collective_agent", "abstract_attribute"],
            "abstract_attribute": ["concrete_object", "role_agent"],
        },
        "metonymy_clash_pairs": {
            "staff": ["suite", "room", "bread", "stave"],
            "crew": ["deck", "cabin", "vessel"],
            "faculty": ["facility", "hall", "building"],
            "applicant": ["application", "appliance"],
        },
    }


def get_gradability_rules() -> Dict[str, List[str]]:
    rules = load_linguistic_rules()
    raw = rules.get("gradability_rules")
    if raw and isinstance(raw, dict):
        return raw
    return {
        "gradable_scalar_adjectives": [
            "tired", "cold", "hot", "happy", "sad", "difficult", "hard", "challenging",
            "easy", "expensive", "cheap", "heavy", "light", "important", "popular",
            "hungry", "angry", "big", "small", "good", "bad", "interesting", "surprised", "frightened",
        ],
        "extreme_limit_adjectives": [
            "exhausted", "freezing", "boiling", "ecstatic", "miserable", "impossible",
            "dead", "vital", "essential", "unique", "starving", "furious", "enormous",
            "tiny", "wonderful", "terrible", "fascinating", "astonished", "terrified",
            "perfect", "empty", "full",
        ],
        "scalar_degree_adverbs": [
            "very", "extremely", "fairly", "rather", "slightly", "somewhat", "moderately",
        ],
        "totality_degree_adverbs": [
            "completely", "utterly", "absolutely", "totally", "entirely",
        ],
    }
