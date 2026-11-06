# -*- coding: utf-8 -*-
"""
Tests for Verb Patient Ontology Classification and Cloze Slot Frame Integration.
Verifies First-Principles categorization of:
1. Concrete Physical Action Verbs (Physical objects/materials/animals)
2. Abstract Event / Situation / Conflict / Problem / Process Verbs
3. Human Recipient / Experiencer Verbs
"""

import re
import pytest
from librarian.linguistics import LinguisticEngine


class TestVerbPatientOntology:
    """Verifies noun patient ontology classification and verb patient extraction."""

    def test_classify_noun_patient_ontology(self):
        # Concrete physical nouns
        assert LinguisticEngine._classify_noun_patient_ontology("lid") == "physical"
        assert LinguisticEngine._classify_noun_patient_ontology("coat") == "physical"
        assert LinguisticEngine._classify_noun_patient_ontology("belt") == "physical"
        assert LinguisticEngine._classify_noun_patient_ontology("potato") == "physical"
        assert LinguisticEngine._classify_noun_patient_ontology("dirt") == "physical"
        assert LinguisticEngine._classify_noun_patient_ontology("car") == "physical"
        assert LinguisticEngine._classify_noun_patient_ontology("elephant") == "physical"

        # Abstract event / situation / conflict / problem / process nouns
        assert LinguisticEngine._classify_noun_patient_ontology("crisis") == "abstract_event"
        assert LinguisticEngine._classify_noun_patient_ontology("conflict") == "abstract_event"
        assert LinguisticEngine._classify_noun_patient_ontology("tension") == "abstract_event"
        assert LinguisticEngine._classify_noun_patient_ontology("dispute") == "abstract_event"
        assert LinguisticEngine._classify_noun_patient_ontology("war") == "abstract_event"
        assert LinguisticEngine._classify_noun_patient_ontology("problem") == "abstract_event"
        assert LinguisticEngine._classify_noun_patient_ontology("issue") == "abstract_event"
        assert LinguisticEngine._classify_noun_patient_ontology("process") == "abstract_event"
        assert LinguisticEngine._classify_noun_patient_ontology("negotiation") == "abstract_event"
        assert LinguisticEngine._classify_noun_patient_ontology("pain") == "abstract_event"
        assert LinguisticEngine._classify_noun_patient_ontology("risk") == "abstract_event"
        assert LinguisticEngine._classify_noun_patient_ontology("threat") == "abstract_event"

        # Person / group nouns
        assert LinguisticEngine._classify_noun_patient_ontology("student") == "person"
        assert LinguisticEngine._classify_noun_patient_ontology("parent") == "person"
        assert LinguisticEngine._classify_noun_patient_ontology("passenger") == "person"
        assert LinguisticEngine._classify_noun_patient_ontology("staff") == "person"
        assert LinguisticEngine._classify_noun_patient_ontology("victim") == "person"

    def test_concrete_physical_verbs(self):
        for v in ["fasten", "peel", "clean"]:
            entry = LinguisticEngine.get_ldoce_entry(v)
            s = (entry or {}).get("senses", [{}])[0]
            ex = (s.get("examples") or [""])[0]
            res = LinguisticEngine._extract_verb_patient_ontology(
                word=v,
                canonical_pos="verb",
                target_entry=entry,
                locked_sense=s,
                definition=s.get("definition"),
                authentic_example=ex,
            )
            assert res is not None, f"Expected ontology for physical verb {v}"
            assert res["ontology_type"] == "concrete_physical"
            assert "concrete physical" in res["slot_label"]
            assert len(res["exemplars"]) >= 1

    def test_abstract_event_crisis_verbs(self):
        # escalate
        entry_esc = LinguisticEngine.get_ldoce_entry("escalate")
        s_esc = (entry_esc or {}).get("senses", [{}])[0]
        res_esc = LinguisticEngine._extract_verb_patient_ontology(
            word="escalate",
            canonical_pos="verb",
            target_entry=entry_esc,
            locked_sense=s_esc,
            definition=s_esc.get("definition"),
            authentic_example=s_esc.get("examples", [""])[0],
        )
        assert res_esc is not None
        assert res_esc["ontology_type"] == "situation_crisis"
        assert "[situation/crisis" in res_esc["slot_label"]

        # defuse
        entry_def = LinguisticEngine.get_ldoce_entry("defuse")
        s_def = (entry_def or {}).get("senses", [{}])[0]
        res_def = LinguisticEngine._extract_verb_patient_ontology(
            word="defuse",
            canonical_pos="verb",
            target_entry=entry_def,
            locked_sense=s_def,
            definition=s_def.get("definition"),
            authentic_example=s_def.get("examples", [""])[0],
        )
        assert res_def is not None
        assert res_def["ontology_type"] == "situation_crisis"
        assert "[situation/crisis" in res_def["slot_label"]

    def test_abstract_process_problem_verbs(self):
        # facilitate
        entry_fac = LinguisticEngine.get_ldoce_entry("facilitate")
        s_fac = (entry_fac or {}).get("senses", [{}])[0]
        res_fac = LinguisticEngine._extract_verb_patient_ontology(
            word="facilitate",
            canonical_pos="verb",
            target_entry=entry_fac,
            locked_sense=s_fac,
            definition=s_fac.get("definition"),
            authentic_example=s_fac.get("examples", [""])[0],
        )
        assert res_fac is not None
        assert res_fac["ontology_type"] == "process_activity"
        assert "[process/activity" in res_fac["slot_label"]

        # tackle
        entry_tac = LinguisticEngine.get_ldoce_entry("tackle")
        s_tac = (entry_tac or {}).get("senses", [{}])[0]
        res_tac = LinguisticEngine._extract_verb_patient_ontology(
            word="tackle",
            canonical_pos="verb",
            target_entry=entry_tac,
            locked_sense=s_tac,
            definition=s_tac.get("definition"),
            authentic_example=s_tac.get("examples", [""])[0],
        )
        assert res_tac is not None
        assert res_tac["ontology_type"] == "problem_issue"
        assert "[problem/issue" in res_tac["slot_label"]

    def test_person_recipient_verbs(self):
        # reassure
        entry_rea = LinguisticEngine.get_ldoce_entry("reassure")
        s_rea = (entry_rea or {}).get("senses", [{}])[0]
        res_rea = LinguisticEngine._extract_verb_patient_ontology(
            word="reassure",
            canonical_pos="verb",
            target_entry=entry_rea,
            locked_sense=s_rea,
            definition=s_rea.get("definition"),
            authentic_example="Teachers reassured anxious parents.",
        )
        assert res_rea is not None
        assert res_rea["ontology_type"] == "person_group"
        assert "[person/group" in res_rea["slot_label"]

    def test_backward_compatibility_alias(self):
        """_extract_physical_patient_ontology is an alias and returns the exact same result."""
        res_new = LinguisticEngine._extract_verb_patient_ontology(
            word="fasten",
            canonical_pos="verb",
            target_entry=None,
            locked_sense={"definition": "to join together two sides of a coat"},
            definition="to join together two sides of a coat",
            authentic_example="Fasten your seat belt.",
        )
        res_legacy = LinguisticEngine._extract_physical_patient_ontology(
            word="fasten",
            canonical_pos="verb",
            target_entry=None,
            locked_sense={"definition": "to join together two sides of a coat"},
            definition="to join together two sides of a coat",
            authentic_example="Fasten your seat belt.",
        )
        assert res_new == res_legacy
        assert "concrete physical" in res_legacy["slot_label"]

    def test_derive_cloze_slot_frame_integration(self):
        # 1. Passive projection for abstract crisis verb (escalate)
        ont_slot = "[situation/crisis, e.g. the war / the conflict]"
        frame_vb = LinguisticEngine.derive_cloze_slot_frame_from_example(
            example_text="The crisis was escalated by foreign military intervention.",
            final_target="escalate",
            base_word="escalate",
            canonical_pos="verb",
            verb_form_tag="VB",
            patient_ontology_slot=ont_slot,
        )
        assert frame_vb == f"... ____ {ont_slot}"

        # 2. Passive VBN frame preserves patient as subject
        frame_vbn = LinguisticEngine.derive_cloze_slot_frame_from_example(
            example_text="The crisis was escalated by foreign military intervention.",
            final_target="escalated",
            base_word="escalate",
            canonical_pos="verb",
            verb_form_tag="VBN",
            patient_ontology_slot=ont_slot,
        )
        assert frame_vbn == f"... {ont_slot} was/were ____"

        # 3. Pronominal subject in ergative/intransitive ('It escalated rapidly')
        frame_pron = LinguisticEngine.derive_cloze_slot_frame_from_example(
            example_text="It escalated rapidly during the summer.",
            final_target="escalated",
            base_word="escalate",
            canonical_pos="verb",
            verb_form_tag="VBD",
            patient_ontology_slot=ont_slot,
        )
        assert ont_slot in frame_pron
        assert not re.search(r"\bit\b", frame_pron, re.I)
