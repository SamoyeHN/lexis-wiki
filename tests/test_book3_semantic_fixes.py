"""
Neuro-symbolic regression tests for Book 3 Units 1-6 semantic, metadata,
collocation/example, and grammar extraction quality fixes.
"""

import pytest
from librarian.linguistics import LinguisticEngine as LE
from librarian.processor import WikiProcessor


def test_british_cross_reference_sizeable_and_derived_projection():
    """Module 1: British cross-reference spelling redirect ('sizeable' -> 'sizable')
    and strict derived exclusion ('weightlessness', 'furiously' do not scavenge base-word senses)."""
    # 1. 'sizeable' redirects to 'sizable' and gets genuine adjective definition & examples
    e_size = LE.get_ldoce_entry("sizeable")
    assert e_size is not None and e_size.get("senses")
    s0 = e_size["senses"][0]
    assert "fairly large" in s0.get("definition", "").lower()
    assert s0.get("examples")

    # 2. Run-on derived entries ('weightlessness', 'furiously') have kind='derived' and do not project base-word senses
    e_weight = LE.get_ldoce_entry("weightlessness")
    assert e_weight is not None and e_weight.get("kind") == "derived"
    assert e_weight.get("senses") == []

    e_fur = LE.get_ldoce_entry("furiously")
    assert e_fur is not None and e_fur.get("kind") == "derived"
    assert e_fur.get("senses") == []


def test_tablet_sense_anchored_corpus_rescue_and_domain_wsd():
    """Module 1: 'tablet' locks to stone/wood/metal inscription sense and enforces strict sense-isolated example gate."""
    e_tab = LE.get_ldoce_entry("tablet")
    quote_tab = "The ancient laws were carved on stone tablets and displayed in the city center."
    idx_tab = LE._lock_sense(e_tab, quote=quote_tab, target_pos="noun")
    locked_tab = e_tab["senses"][idx_tab]
    assert "stone" in locked_tab.get("definition", "").lower()
    # Sense 1 ('tablet' = stone/clay slab) has 0 examples in LDOCE; polysemous entries never scavenge corpus examples across senses
    ex_tab = LE.ensure_distinct_example("tablet", "noun", "", quote_tab, sense=locked_tab)
    assert ex_tab == ""


def test_phrase_grounding_make_room_wind_up_take_back_train_for():
    """Module 1: Multi-word expression sense grounding and definition composition."""
    # 1. 'make room for [sth/sb]' composes a full verb phrase definition (not bare noun 'room')
    def_mr, src_mr, _ = LE.expression_definition_evidence(
        "make room for [sth/sb]",
        "We need to make room for the new equipment in the laboratory.",
    )
    assert def_mr.lower().startswith("to ")
    assert "space" in def_mr.lower()

    # 2. 'wind up' locks to end up / finish in a state or place, not wind up a watch
    cands_wu = LE.build_top_k_expression_candidates(
        "wind up",
        "If he keeps spending money like that, he will wind up in debt.",
        expr_type="phrasal verb",
    )
    assert cands_wu
    assert "watch" not in cands_wu[0]["definition"].lower()
    assert "machine" not in cands_wu[0]["definition"].lower()

    # 3. 'take [sb] back to [sth]' includes the memory/past-time sense in the Top-K symbolic candidate pool
    cands_tb = LE.build_top_k_expression_candidates(
        "take [sb] back to [sth]",
        "Hearing that old song always takes me back to my childhood summers.",
        expr_type="phrasal verb",
    )
    assert cands_tb
    assert any(
        any(w in c["definition"].lower() for w in ("remember", "past", "time", "memory"))
        for c in cands_tb
    )

    # 4. 'train for [sth/sb]' locks to sports preparation sense
    cands_tf = LE.build_top_k_expression_candidates(
        "train for [sth/sb]",
        "She has been training for the marathon for six months.",
        expr_type="collocation",
    )
    assert cands_tf
    assert "prepare" in cands_tf[0]["definition"].lower() or "sport" in cands_tf[0]["definition"].lower()


def test_vocab_card_enrichment_hygiene_and_grammar_box():
    """Module 2 & 3: Grammar Box stripping on attributive quotes, verb noun-unit pattern filtering,
    bare V-ing collocation filtering, and cross-sense collocation filtering."""
    # 1. Attributive noun modifier quote ('contentious issue') strips '[not before noun]' from Grammar Box
    e_cont = LE.get_ldoce_entry("contentious")
    s_cont = e_cont["senses"][0]
    enr_cont = LE.get_vocab_card_enrichment(
        "contentious",
        pos="adjective",
        definition=s_cont.get("definition", ""),
        quote="Water pricing remains a deeply contentious issue in the region.",
    )
    assert "[not before noun]" not in enr_cont.get("grammar_tag", "")
    assert "[not before noun]" not in enr_cont.get("grammar_usage", "")

    # 2. Verb card 'exile' filters out noun unit 'go into exile / be in exile' from Pattern
    e_exile = LE.get_ldoce_entry("exile")
    verb_senses = [s for s in e_exile.get("senses", []) if s.get("pos") == "verb"]
    if verb_senses:
        enr_exile = LE.get_vocab_card_enrichment(
            "exile",
            pos="verb",
            definition=verb_senses[0].get("definition", ""),
        )
        assert "in exile" not in enr_exile.get("pattern_note", "").lower()
        assert "into exile" not in enr_exile.get("pattern_note", "").lower()

    # 3. Thesaurus notes do not have trailing colons before —
    e_spec = LE.get_ldoce_entry("spectator")
    if e_spec and e_spec.get("senses"):
        enr_spec = LE.get_vocab_card_enrichment(
            "spectator",
            pos="noun",
            definition=e_spec["senses"][0].get("definition", ""),
        )
        assert ": —" not in enr_spec.get("thesaurus_note", "")


def test_grammar_gerund_subject_vs_fronted_participle_and_so_that():
    """Module 4: Subjectless copula + fronted VBG is classified as gerund_subject_nominalization,
    and purpose 'so that' is not misclassified as resultative 'so ... that'."""
    from librarian.grammar_patterns import GrammarPatternEngine

    nlp = LE.get_spacy()
    g_engine = GrammarPatternEngine.get_engine(nlp)
    # 1. Subjectless copula + fronted VBG ('Being on wheels also means having a tiny kitchen.')
    sent_ger = "That is why being on wheels also means having a tiny kitchen and very little storage space."
    doc_ger = nlp(sent_ger)
    matches_ger = g_engine.match_sentence(doc_ger)
    ids_ger = [m["id"] for m in matches_ger]
    assert "nonfinite_participial_adjunct" not in ids_ger
    prof_ger = LE.match_cobuild_grammar_profile(
        LE.generate_cobuild_formula(sent_ger, category="Information Packaging"),
        sent_ger,
        category="Information Packaging",
    )
    assert prof_ger["profile_key"] == "gerund_subject_nominalization"

    # 2. Purpose 'so that' ('... so that he could ...') is not matched as resultative 'so ... that'
    doc_purp = nlp("He spoke very slowly and clearly so that everyone in the hall could understand him.")
    matches_purp = g_engine.match_sentence(doc_purp)
    ids_purp = [m["id"] for m in matches_purp]
    assert "resultative_so_such_that" not in ids_purp


def test_book3_unit6_section_a_distractor_and_skeleton_fixes():
    """
    Regression tests for Book 3 Unit 6 Section A:
    1. Plural NNS countability gate ('subsidiaries' must never produce 'pharmas' or 'subsidences').
    2. Polysemous WordNet synset domain disambiguation ('loan' must never produce 'contraction' or 'synonym').
    3. Hypernym coordinate domain tightening ('slump' must never produce 'celibacy' or 'juncture').
    4. Slash-pattern modified_noun normalization ('regulatory body/authority/agency') and
       possessive placeholder preservation ('a roof over (one's) head' -> "a roof over one's head").
    """
    from librarian.linguistics import LinguisticEngine

    # 1. Plural noun countability & morphological gate
    assert not LinguisticEngine.can_pluralize_noun_distractor("pharma")
    assert not LinguisticEngine.can_pluralize_noun_distractor("subsidence")
    assert not LinguisticEngine.can_pluralize_noun_distractor("subspecies")
    assert LinguisticEngine.can_pluralize_noun_distractor("distributor")

    # 2. 'loan' (financial sense) must not match metalinguistic 'loanword' synset ('contraction', 'synonym')
    loan_dists = LinguisticEngine.generate_vocab_distractors(
        "loan",
        pos="noun",
        target_count=3,
        definition="an amount of money that you borrow from a bank etc.",
        quote="She took out a loan of 10,000 pounds.",
        context_anchor="of",
        anchor_type="prep",
    )
    assert "contraction" not in loan_dists
    assert "synonym" not in loan_dists
    assert any(w in loan_dists for w in ("overdraft", "credit", "load"))

    # 3. 'slump' (economic recession) must not pull mega-bucket grand-hypernym 'condition' -> 'celibacy'
    slump_dists = LinguisticEngine.generate_vocab_distractors(
        "slump",
        pos="noun",
        target_count=3,
        definition="a period when there is a big reduction in sales or business activity",
        quote="The country was in an economic slump.",
        context_anchor="economic",
        anchor_type="adj",
    )
    assert "celibacy" not in slump_dists
    assert "juncture" not in slump_dists
    assert "boom" in slump_dists

    # 4. 'a roof over (one's) head' normalization & phrase distractors
    roof_dists = LinguisticEngine.generate_phrase_distractors("a roof over one's head", count=3)
    assert len(roof_dists) == 3
    assert "a roof over your head" not in roof_dists
    assert "a roof over one's head" not in roof_dists
    assert all(d.startswith(("a ", "an ", "the ")) for d in roof_dists)
    # No two distractors should share the exact same 4-word prefix ('a mine of information ...')
    prefixes = [" ".join(d.split()[:-1]) for d in roof_dists]
    assert len(set(prefixes)) == len(prefixes)

    # 5. End-to-end skeleton check for 'subsidiaries', 'regulatory', and "a roof over (one's) head"
    items = [
        {
            "word": "subsidiary",
            "part_of_speech": "noun",
            "definition": "a company that is owned or controlled by another larger company",
            "quote": "Many foreign subsidiaries operate in the region.",
        },
        {
            "word": "regulatory",
            "part_of_speech": "adjective",
            "definition": "having the purpose of controlling an activity or process, especially by means of rules",
            "quote": "The plan must be approved by the regulatory authority.",
        },
        {
            "word": "a roof over (one's) head",
            "part_of_speech": "idiom",
            "definition": "a place to live",
            "quote": "At least we have a roof over our heads.",
        },
    ]
    skeletons = LinguisticEngine.build_precomputed_target_skeletons(items, target_count=3)
    skeleton_md = str(skeletons)
    assert "pharmas" not in skeleton_md
    assert "subsidences" not in skeleton_md
    reg_skel = next(s for s in skeletons if s["target_word"] == "regulatory")
    assert "body/authority/agency" not in reg_skel["micro_task"]
    assert "____ authority" in reg_skel["micro_task"]
    assert "a roof over his head" in skeleton_md


def test_possessive_idiom_whole_preservation_and_pronoun_tense_sync():
    """
    Whole-Idiom Preservation + Concrete Possessive Pronoun & Tense Synchronization:
    - Idioms with internal `one's` slots keep the entire idiom in `options` (no single-word splitting that destroys idiomatic meaning).
    - All 4 options synchronize `one's` to a concrete 3rd-person possessive (`her`/`his`/`their`/`its`) and synchronize leading verb inflection (`VBD`/`VBZ`/`VB`).
    - Strategy A2 never returns truncated phrasal-verb prefixes (`keep one's hands`, `let one's guard`), near-synonym double-keys (`do one's best` vs `do one's utmost`, `turn one's mind to` vs `turn one's attention to`), or undignified idioms (`pick one's nose`).
    """
    from librarian.linguistics import LinguisticEngine
    items = [
        {
            "word": "come to one's aid",
            "part_of_speech": "idiom",
            "definition": "to help someone who is in trouble or difficulty",
            "quote": "Friends came to her aid.",
        },
        {
            "word": "a roof over one's head",
            "part_of_speech": "idiom",
            "definition": "a place to live",
            "quote": "Struggling for years to make ends meet and keep a roof over their heads.",
        },
        {
            "word": "come into one's mind",
            "part_of_speech": "idiom",
            "definition": "if an idea or thought comes into your mind, you suddenly think of it",
            "quote": "Then a story she had heard as a child came into her mind.",
        },
        {
            "word": "turn one's attention to",
            "part_of_speech": "collocation",
            "definition": "to begin to think about, deal with, or look at something else",
            "quote": "He then turned his attention to the financial crisis.",
        },
        {
            "word": "do one's utmost",
            "part_of_speech": "collocation",
            "definition": "to try as hard as you can to do something",
            "quote": "She did her utmost to protect the children.",
        },
    ]
    skeletons = LinguisticEngine.build_precomputed_target_skeletons(items, target_count=5)
    by_base = {s["base_headword"]: s for s in skeletons}

    # 1. come to one's aid -> synchronized to VBD + 'her'
    aid_skel = by_base["come to one's aid"]
    assert aid_skel["target_word"] == "came to her aid"
    assert all("one's" not in opt and " her " in opt for opt in aid_skel["prescribed_options"])
    assert "to ____" not in (aid_skel["cloze_slot_frame"] or "")

    # 2. a roof over one's head -> singular head noun guard normalizes 'their' -> 'his'
    roof_skel = by_base["a roof over one's head"]
    assert roof_skel["target_word"] == "a roof over his head"
    assert all("one's" not in opt and " his " in opt for opt in roof_skel["prescribed_options"])

    # 3. come into one's mind -> kept whole, synchronized to VBD + 'her'
    mind_skel = by_base["come into one's mind"]
    assert mind_skel["target_word"] == "came into her mind"
    assert all("one's" not in opt and " her " in opt for opt in mind_skel["prescribed_options"])

    # 4. turn one's attention to -> relational prep 'to' preserved across all 4 options, no 'turn his mind to' or 'turn his nose up'
    att_skel = by_base["turn one's attention to"]
    assert att_skel["target_word"] == "turned his attention to"
    assert all("one's" not in opt and " his " in opt for opt in att_skel["prescribed_options"])
    assert all("mind to" not in opt and "nose up" not in opt for opt in att_skel["prescribed_options"])

    # 5. do one's utmost -> 3-word idiom distractors are complete idioms (no 'keep her hands', 'let her guard', 'do her best')
    utmost_skel = by_base["do one's utmost"]
    assert utmost_skel["target_word"] == "did her utmost"
    assert all("one's" not in opt and " her " in opt for opt in utmost_skel["prescribed_options"])
    for bad_frag in ("hands", "guard", "faith", "confidence", "best", "bit", "part"):
        assert all(not opt.endswith(" " + bad_frag) for opt in utmost_skel["prescribed_options"])


def test_catch_up_with_patient_ontology_and_frame():
    """Verify that phrasal/prepositional verbs like 'catch up with' do not leak base-verb direct
    object collocations (e.g. 'the act', 'the attention' from 'catch') into patient ontology,
    and resolve the human patient from prepositional pronoun objects ('her', 'him') and definition."""
    res = LE._extract_verb_patient_ontology(
        word="catch up with",
        canonical_pos="phrase",
        target_entry=None,
        locked_sense=None,
        definition="if something bad from the past catches up with you, you cannot avoid dealing with it any longer",
        authentic_example="At the end of the movie his murky past catches up with him.",
        quote="Eventually, the bureaucracy caught up with her, and she was given 10 days to leave her apartment for good.",
    )
    assert res is not None
    assert res["ontology_type"] == "person_group"
    assert "person" in res["slot_label"].lower() or "someone" in res["slot_label"].lower()
    assert "act" not in res["slot_label"].lower()
    assert "attention" not in res["slot_label"].lower()

    frame = LE.derive_cloze_slot_frame_from_example(
        example_text="At the end of the movie his murky past catches up with him.",
        final_target="catch up with",
        base_word="catch up with",
        canonical_pos="phrase",
        patient_ontology_slot=res.get("slot_label"),
    )
    assert frame is not None
    assert "act" not in frame
    assert "attention" not in frame
    assert "person" in frame or "someone" in frame


def test_five_paradigm_verb_preposition_patient_ontology():
    """
    Verify the 5 structural paradigms of verb/phrase + preposition patient ontology:
    1. Transitive phrasal verb ('pack up', 'clean up'): physical action on 'things/something'
       never hijacked by locative/purpose adjunct prepositions ('into cases', 'for a trip').
    2. Prepositional verb ('run into (sb)', 'consult with (sb)'): resolves human patient
       without being hijacked by manner adjuncts ('by chance').
    3. Phrasal-prepositional verb ('run out of', 'come down with'): traverses 2-step
       dependency chain ('run -> out -> of -> fuel').
    4. Verb + Noun + Prep idiom ('take care of', 'make sense of', 'pay attention to', 'pose a threat to'):
       blocks internal noun self-cannibalization ('the care', 'the sense', 'the attention', 'the threat')
       and resolves the external prepositional object.
    5. Closed / intransitive idioms ('make a difference', "come to (one's) aid", 'wind up', 'turn up'):
       return None instead of fabricating a post-phrase object slot.
    """
    # Paradigm I: 'pack up' -> concrete_physical (never 'the cases / the trip')
    res_pack = LE._extract_verb_patient_ontology(
        word="pack up",
        canonical_pos="phrase",
        target_entry=None,
        locked_sense={"definition": "to put things into cases, bags etc ready for a trip somewhere"},
        definition="to put things into cases, bags etc ready for a trip somewhere",
        authentic_example="Most of the holidaymakers had packed up and gone.",
        quote="Facing tenant eviction after several months of unpaid rent, Sue Johnson packed up whatever she could fit into her two-door automobile and drove out of town.",
    )
    assert res_pack is not None
    assert res_pack["ontology_type"] == "concrete_physical"
    assert "cases" not in res_pack["slot_label"]
    assert "trip" not in res_pack["slot_label"]
    assert "concrete physical object" in res_pack["slot_label"]

    frame_pack = LE.derive_cloze_slot_frame_from_example(
        example_text="Facing tenant eviction after several months of unpaid rent, Sue Johnson packed up whatever she could fit into her two-door automobile and drove out of town.",
        final_target="pack up",
        base_word="pack up",
        canonical_pos="phrase",
        patient_ontology_slot=res_pack["slot_label"],
    )
    assert frame_pack == f"... ____ {res_pack['slot_label']}"

    # Paradigm II: 'run into (sb)' -> person_group (never 'the chance' from 'by chance')
    res_run_into = LE._extract_verb_patient_ontology(
        word="run into (sb)",
        canonical_pos="phrase",
        target_entry=None,
        locked_sense={"definition": "[informal] to meet someone by chance"},
        definition="[informal] to meet someone by chance",
        authentic_example="I ran into an old friend at the station.",
    )
    assert res_run_into is not None
    assert res_run_into["ontology_type"] == "person_group"
    assert "chance" not in res_run_into["slot_label"]
    assert "station" not in res_run_into["slot_label"]

    # Paradigm III: 'run out of' -> traverses 'run -> out -> of -> fuel'
    res_run_out = LE._extract_verb_patient_ontology(
        word="run out of",
        canonical_pos="phrase",
        target_entry=None,
        locked_sense={"definition": "to use all of something so that there is none left"},
        definition="to use all of something so that there is none left",
        authentic_example="They had run out of fuel in the desert.",
    )
    assert res_run_out is not None
    assert res_run_out["ontology_type"] == "concrete_physical"
    assert "the fuel" in res_run_out["exemplars"]
    assert "the desert" not in res_run_out["exemplars"]

    # Paradigm IV: 'take care of', 'make sense of', 'pay attention to', 'pose a threat to'
    for phrase, defn, ex, banned_internal in [
        ("take care of", "to look after someone or something", "Who is taking care of the dog while you are away?", "the care"),
        ("make sense of", "to understand something that is complicated", "Can you make sense of these instructions?", "the sense"),
        ("pay attention to", "to listen, look at, or think about something carefully", "Please pay attention to the safety briefing.", "the attention"),
        ("pose a threat to (sth/sb)", "to exist in a way that may cause a problem, danger, or harm", "Pollution poses a threat to the marine ecology.", "the threat"),
    ]:
        res_idm = LE._extract_verb_patient_ontology(
            word=phrase,
            canonical_pos="phrase",
            target_entry=None,
            locked_sense={"definition": defn},
            definition=defn,
            authentic_example=ex,
        )
        assert res_idm is not None, f"Expected patient ontology for {phrase}"
        assert banned_internal not in res_idm["exemplars"], f"Internal noun {banned_internal} leaked for {phrase}"

    # Paradigm V / Closed intransitive idioms -> return None
    for closed_phrase, defn, ex in [
        ("make a difference", "to have an important effect or influence on something", "Exercise can make a big difference to your health."),
        ("come to (one's) aid", "help someone", "Passers-by came to her aid after the accident."),
        ("wind up", "[informal] to be in an unpleasant situation or place after doing something", "He wound up in a cheap motel."),
        ("turn up", "to arrive at a place, especially in a way that is unexpected", "She turned up late for the meal."),
    ]:
        res_closed = LE._extract_verb_patient_ontology(
            word=closed_phrase,
            canonical_pos="phrase",
            target_entry=None,
            locked_sense={"definition": defn},
            definition=defn,
            authentic_example=ex,
        )
        assert res_closed is None, f"Expected None for closed/intransitive phrase {closed_phrase}, got {res_closed}"


def test_tumble_verb_sense_isolation_and_frame():
    """
    Verify that polysemous verbs whose sense-locked LDOCE examples contain only inflected
    forms (e.g. 'tumble' Sense 0 verb: 'She lost her balance and tumbled backwards.')
    are not falsely flagged as sparse, never scavenge cross-POS noun examples
    ('take a tumble next year'), and never slice temporal determiners ('... ____ next').
    """
    items = [
        {
            "word": "tumble",
            "part_of_speech": "verb",
            "definition": "to fall down quickly and suddenly, especially with a rolling movement",
            "quote": "Sue and her two kids tumbled into a motel micro-apartment, which had no stove or refrigerator.",
        }
    ]
    skels = LE.build_precomputed_target_skeletons(items, target_count=1)
    assert len(skels) == 1
    skel = skels[0]
    assert "take a tumble" not in skel["authentic_example"].lower()
    assert skel["cloze_slot_frame"] != "... ____ next"
    assert skel["cloze_slot_frame"] in ("... ____ backwards", "... ____ down the cliff")


def test_orthographic_equivalence_closure_ame_bre():
    """
    Verify 100% Orthographic Equivalence Closure across American <-> British English pairs:
    - `orthographic_variants(us)` and `orthographic_variants(uk)` mutually include each other.
    - `get_pedlex_and_oxford_synonyms_antonyms`, `get_strict_pedlex_and_oxford_synonyms`,
      `semantic_fields`, `get_ldoce_preps`, `get_word_cxn_specs`, and `get_word_error_traps`
      return identical sets/counts for AmE and BrE spellings.
    - Pattern bundle extraction and phrase lookups bridge regional spelling differences
      (e.g. 'skeptical' matches 'sceptical about/of', 'in favor of' matches 'in favour of').
    """
    pairs = [
        ("enroll", "enrol"), ("fulfill", "fulfil"), ("instill", "instil"), ("distill", "distil"), ("appall", "appal"),
        ("skeptical", "sceptical"), ("skeptic", "sceptic"), ("artifact", "artefact"),
        ("color", "colour"), ("behavior", "behaviour"), ("favor", "favour"), ("honor", "honour"), ("labor", "labour"),
        ("center", "centre"), ("theater", "theatre"), ("fiber", "fibre"), ("meter", "metre"), ("liter", "litre"),
        ("organize", "organise"), ("realize", "realise"), ("recognize", "recognise"), ("analyze", "analyse"), ("paralyze", "paralyse"),
        ("defense", "defence"), ("offense", "offence"), ("pretense", "pretence"),
        ("catalog", "catalogue"), ("dialog", "dialogue"),
        ("traveler", "traveller"), ("counselor", "counsellor"), ("marvelous", "marvellous"),
        ("gray", "grey"), ("inquire", "enquire"), ("inquiry", "enquiry"), ("program", "programme"),
        ("sizable", "sizeable"), ("judgment", "judgement"), ("acknowledgment", "acknowledgement"),
    ]
    for us, uk in pairs:
        assert uk in LE.orthographic_variants(us), f"{uk} missing from orthographic_variants({us})"
        assert us in LE.orthographic_variants(uk), f"{us} missing from orthographic_variants({uk})"
        syns_us, ants_us = LE.get_pedlex_and_oxford_synonyms_antonyms(us)
        syns_uk, ants_uk = LE.get_pedlex_and_oxford_synonyms_antonyms(uk)
        assert syns_us == syns_uk, f"Synonym mismatch for {us}/{uk}: {syns_us ^ syns_uk}"
        assert set(ants_us) == set(ants_uk), f"Antonym mismatch for {us}/{uk}: {ants_us} vs {ants_uk}"
        assert LE.get_strict_pedlex_and_oxford_synonyms(us) == LE.get_strict_pedlex_and_oxford_synonyms(uk)
        assert LE.semantic_fields(us) == LE.semantic_fields(uk), f"semantic_fields mismatch for {us}/{uk}"
        assert set(LE.get_ldoce_preps(us)) == set(LE.get_ldoce_preps(uk)), f"Preps mismatch for {us}/{uk}"
        assert len(LE.get_word_cxn_specs(us)) == len(LE.get_word_cxn_specs(uk)), f"CXN mismatch for {us}/{uk}"
        assert len(LE.get_word_error_traps(us)) == len(LE.get_word_error_traps(uk)), f"Traps mismatch for {us}/{uk}"

    # Verify 'skeptical' extracts 'about'/'of' from British pattern 'sceptical about/of'
    s_skep = LE.get_ldoce_entry("skeptical")["senses"][0]
    bundles = LE.extract_sense_pattern_bundles(s_skep, target_word="skeptical")
    assert bundles and bundles[0]["prep"] == "about" and set(bundles[0]["preps"]) == {"about", "of"}
    assert LE.lookup_ldoce_phrase("curry favor with") is not None
    assert LE.lookup_ldoce_phrase("without fear or favor") is not None
    assert LE.is_attested_phrase("in favor of") is True


def test_enroll_qualia_scenario_preposition_valency_distractors():
    """
    Verify Pustejovsky Qualia Scenario Lifecycle Verb Bridge + Intransitive Preposition Valency Clash:
    - For `enroll` (`... ____ in [classes/school]`), `build_precomputed_target_skeletons` and
      `generate_vocab_distractors` harvest same-scenario education lifecycle verbs with contrasting
      valency (`apply` [+for/to], `attend` [transitive V+N], `withdraw` [+from]) plus 1 polar antonym (`leave`),
      and never fall back to unrelated corpus verbs (`worship`, `orbit`).
    - `verb_requires_object` is False when the selected cloze slot is an intransitive prepositional frame (`... ____ in`).
    """
    items = [
        {
            "word": "enroll",
            "part_of_speech": "verb",
            "definition": "to officially arrange to join a school, university, or course, or to arrange for someone else to do this",
            "quote": "Sue enrolled her kids in a nearby public school.",
        }
    ]
    skels = LE.build_precomputed_target_skeletons(items, target_count=1)
    assert len(skels) == 1
    skel = skels[0]
    assert skel["cloze_slot_frame"] == "... ____ in"
    assert skel["verb_requires_object"] is False
    opts = set(skel["prescribed_options"])
    assert "enroll" in opts
    assert "worship" not in opts and "orbit" not in opts
    assert "corpus_fallback" not in skel["distractor_mechanisms"].values()
    assert opts == {"enroll", "leave", "apply", "attend"}


def test_relational_and_deverbal_adjective_tier3b_distractors():
    """
    Verify Tier 3b Relational Adjective Pertainym / Derivation Graph Traversal (Noun + Verb roots)
    and Second Defense `-ory`/`-ive` morphological peers:
    - `regulatory` (`... the ____ authority`), `compensatory`, and `descriptive` must never fall back to
      unrelated `corpus_fallback` words (`instinctive`, `blind`, `reassuring`, `asleep`).
    """
    skels = LE.build_precomputed_target_skeletons(
        [
            {
                "word": "regulatory",
                "part_of_speech": "adjective",
                "definition": "[formal] a regulatory authority has the official power to control an activity and to make sure that it is done in a satisfactory way",
                "quote": "The plan must be approved by the regulatory authority.",
            }
        ],
        target_count=1,
    )
    reg_skel = skels[0]
    assert reg_skel["cloze_slot_frame"] == "... the ____ authority"
    assert "corpus_fallback" not in reg_skel["distractor_mechanisms"].values()
    assert "instinctive" not in reg_skel["prescribed_options"]
    assert "blind" not in reg_skel["prescribed_options"]
    assert set(reg_skel["prescribed_options"]) == {"regulatory", "limiting", "corrective", "decisive"}

    for w in ("compensatory", "descriptive"):
        dists, meta = LE.generate_vocab_distractors(w, pos="adj", target_count=3, return_metadata=True)
        assert len(dists) == 3
        assert "corpus_fallback" not in meta.values(), f"{w} fell back to corpus_fallback: {meta}"


def test_originated_intransitive_preposition_and_valency_distractors():
    """
    Verify Intransitive Preposition Valency Contrast + Antonym Fallback + Paronym Blending:
    - For `originated` (`... ____ in`), `build_precomputed_target_skeletons` produces:
      1 polar antonym (`terminated`),
      1 preposition valency clash / thesaurus peer (`derived` [+from vs +in]),
      1 Latinate prefix/suffix paronym (`orientated`),
      with 0 corpus fallback and 0 double-key leak (`occurred in` intercepted by `double_key_prep`).
    """
    items = [
        {
            "word": "originate",
            "part_of_speech": "verb",
            "definition": "to begin to happen or exist, especially in a particular place or situation",
            "quote": "The economic slump that so many people suffered through originated in the United States and quickly spread to Europe.",
        }
    ]
    skels = LE.build_precomputed_target_skeletons(items, target_count=1)
    assert len(skels) == 1
    skel = skels[0]
    assert skel["target_word"] == "originated"
    assert skel["cloze_slot_frame"] == "... ____ in"
    assert skel["verb_requires_object"] is False
    opts = set(skel["prescribed_options"])
    assert opts == {"originated", "terminated", "derived", "orientated"}
    assert "corpus_fallback" not in skel["distractor_mechanisms"].values()
    assert "occurred" not in opts
    assert skel["distractor_mechanisms"]["terminated"] == "antonym"
    assert skel["distractor_mechanisms"]["derived"] == "ldoce_thesaurus"
    assert skel["distractor_mechanisms"]["orientated"] == "orthographic_paronym"


def test_esp_financial_distractor_mechanisms_and_frame_slicing():
    """
    Verify ESP Financial/Commercial Lexicon Computation:
    1. Information Gain Frame Slicing:
       - Delexical verb 'get' in 'get a mortgage of £60,000' promotes bound prep 'of' to anchor,
         yielding the two-sided constraint frame '... get a ____ of'.
    2. Continuous CEFR Cognitive Reachability:
       - Obscure C2 terms like 'lien' (CEFR float 5.42, no core level) are mathematically penalized
         and excluded in favor of B1/B2 pedagogical core words ('credit', 'scholarship', 'fee').
    3. Strict Synonym vs Domain Co-Hyponym separation:
       - LDOCE Thesaurus domain coordinates are not falsely suppressed by the synonym-pile gate.
    """
    items = [
        {
            "word": "mortgage",
            "part_of_speech": "noun",
            "definition": "the amount of money you borrow in the form of a mortgage",
            "quote": "The economic slump that so many people suffered through originated in the United States, with a regulatory failure of mortgages rated less risky than they turned out to be.",
        }
    ]
    skels = LE.build_precomputed_target_skeletons(items, target_count=1)
    assert len(skels) == 1
    skel = skels[0]
    assert skel["target_word"] == "mortgage"
    assert skel["cloze_slot_frame"] == "... get a ____ of"
    opts = skel["prescribed_options"]
    assert "mortgage" in opts
    assert len(opts) == 4
    # Obscure C2 legal terms must be absent
    assert "lien" not in opts
    assert "pledge" not in opts
    # Prescribed options must be pedagogical core terms
    for dist in [o for o in opts if o != "mortgage"]:
        c_prof = LE._lexicon_profile(dist)
        assert c_prof[2] in ("A1", "A2", "B1", "B2"), f"Distractor {dist} exceeded B2 ceiling: {c_prof[2]}"


def test_lla_get_off_definition_repair():
    """
    Verify Longman Language Activator (LLA) gloss repair:
    'get off' under concept 'to get off a bus, plane etc' was truncated to 'to'.
    Now restored to full authentic gloss:
    'to leave a bus, train, etc at the end of a journey' across target entries.
    """
    EXPECTED_GLOSS = "to leave a bus, train, etc at the end of a journey"
    for w in ["disembark", "dismount", "get", "leave"]:
        entry = LE.get_ldoce_entry(w)
        assert entry is not None, f"Entry {w} not found"
        found = False
        for la in entry.get("language_activator", []):
            for item in la.get("words", []):
                if item.get("word") == "get off" and "bus" in la.get("concept", ""):
                    assert item.get("definition") == EXPECTED_GLOSS, (
                        f"Expected {EXPECTED_GLOSS!r} for {w}, got {item.get('definition')!r}"
                    )
                    found = True
                    break
            if found:
                break
        assert found, f"'get off' bus concept not found in LLA for {w}"






