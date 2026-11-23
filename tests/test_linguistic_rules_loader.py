# -*- coding: utf-8 -*-
"""Unit tests for linguistic rules loader and decoupled declarative taxonomies."""

from pathlib import Path

from librarian.linguistics import LinguisticEngine
from librarian.linguistics_core.linguistic_rules_loader import (
    _get_rules_json_path,
    get_academic_adj_families,
    get_adj_prep_valency_map,
    get_adjunct_prep_pairs,
    get_all_negative_inversion_triggers,
    get_anchor_stopwords,
    get_collective_noun_ontology,
    get_delexical_quantifier_nouns,
    get_delexical_verbs,
    get_negative_inversion_adverbs,
    get_negative_inversion_phrases,
    get_predicative_only_adjs,
    get_register_banned_words,
    get_typical_complement_preps,
    load_linguistic_rules,
)


def test_linguistic_rules_json_exists():
    path = _get_rules_json_path()
    assert isinstance(path, Path)
    assert path.exists()
    assert path.is_file()


def test_load_linguistic_rules_structure():
    rules = load_linguistic_rules()
    assert isinstance(rules, dict)
    assert "_meta" in rules
    assert "adj_prep_valency_map" in rules
    assert "academic_adj_families" in rules
    assert "register_banned_words" in rules
    assert "anchor_stopwords" in rules
    assert "collective_noun_ontology" in rules
    assert "negative_inversion_adverbs" in rules
    assert "delexical_verbs" in rules


def test_typed_getters():
    valency_map = get_adj_prep_valency_map()
    assert isinstance(valency_map, dict)
    assert "to" in valency_map
    assert "vulnerable" in valency_map["to"]

    families = get_academic_adj_families()
    assert isinstance(families, list)
    assert len(families) >= 9

    banned = get_register_banned_words()
    assert isinstance(banned, set)
    assert "dwarf" in banned
    assert "sex" in banned

    stopwords = get_anchor_stopwords()
    assert isinstance(stopwords, set)
    assert "the" in stopwords
    assert "thing" in stopwords

    adjunct_pairs = get_adjunct_prep_pairs()
    assert isinstance(adjunct_pairs, set)
    assert ("for", "example") in adjunct_pairs

    typical_preps = get_typical_complement_preps()
    assert isinstance(typical_preps, set)
    assert "into" in typical_preps

    ontology = get_collective_noun_ontology()
    assert isinstance(ontology, dict)
    assert "human_role" in ontology
    assert ("staff", "personnel", "workforce") in ontology["human_role"]

    pred_adjs = get_predicative_only_adjs()
    assert isinstance(pred_adjs, set)
    assert "asleep" in pred_adjs

    inv_advs = get_negative_inversion_adverbs()
    assert isinstance(inv_advs, set)
    assert "hardly" in inv_advs

    inv_phrases = get_negative_inversion_phrases()
    assert isinstance(inv_phrases, set)
    assert "no sooner" in inv_phrases

    all_triggers = get_all_negative_inversion_triggers()
    assert inv_advs.issubset(all_triggers)
    assert inv_phrases.issubset(all_triggers)

    delex_verbs = get_delexical_verbs()
    assert isinstance(delex_verbs, set)
    assert "make" in delex_verbs

    delex_nouns = get_delexical_quantifier_nouns()
    assert isinstance(delex_nouns, set)
    assert "lot" in delex_nouns


def test_linguistic_engine_class_attributes_integrity():
    assert "to" in LinguisticEngine._ADJ_PREP_VALENCY_MAP
    assert "vulnerable" in LinguisticEngine._ADJ_PREP_VALENCY_MAP["to"]
    assert "human_role" in LinguisticEngine._COLLECTIVE_NOUN_ONTOLOGY
    assert "hardly" in LinguisticEngine._NEGATIVE_INVERSION_ADVERBS
    assert "make" in LinguisticEngine._DELEXICAL_VERBS
    assert "lot" in LinguisticEngine._DELEXICAL_QUANTIFIER_NOUNS


def test_syntactic_slot_clash_gate_regression():
    has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
        target="vulnerable",
        candidate="eligible",
        anchor="to",
        anchor_type="prep",
        pos="adj"
    )
    # vulnerable takes 'to', eligible takes 'for' -> slot clash!
    assert has_clash is True
    assert reason != ""
