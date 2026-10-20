"""
Neuro-symbolic regression tests for Book 3 Units 1-6 semantic, metadata,
collocation/example, and grammar extraction quality fixes.
"""

import pytest
from librarian.linguistics import LinguisticEngine as LE
from librarian.processor import WikiProcessor


def test_british_cross_reference_sizeable_and_derived_projection():
    """Module 1: British cross-reference spelling redirect ('sizeable' -> 'sizable')
    and derived base-sense projection ('weightlessness', 'furiously')."""
    # 1. 'sizeable' redirects to 'sizable' and gets genuine adjective definition & examples
    e_size = LE.get_ldoce_entry("sizeable")
    assert e_size is not None and e_size.get("senses")
    s0 = e_size["senses"][0]
    assert "fairly large" in s0.get("definition", "").lower()
    assert s0.get("examples")

    # 2. 'weightlessness' projects the space/gravity sense of 'weightless'
    e_weight = LE.get_ldoce_entry("weightlessness")
    assert e_weight is not None and len(e_weight.get("senses", [])) >= 2
    quote_w = "For the first few days of his trip, he experienced weightlessness in the spacecraft."
    idx_w = LE._lock_sense(e_weight, quote=quote_w, target_pos="noun")
    locked_w = e_weight["senses"][idx_w]
    assert "gravity" in locked_w.get("definition", "").lower() or "space" in locked_w.get("definition", "").lower()
    assert "[not before noun]" not in (locked_w.get("grammar") or "")

    # 3. 'furiously' projects both anger and high-speed/effort senses of 'furious'
    e_fur = LE.get_ldoce_entry("furiously")
    assert e_fur is not None and len(e_fur.get("senses", [])) >= 2
    quote_f = "He worked furiously all night to finish the report before the deadline."
    idx_f = LE._lock_sense(e_fur, quote=quote_f, target_pos="adverb")
    locked_f = e_fur["senses"][idx_f]
    assert "effort" in locked_f.get("definition", "").lower() or "speed" in locked_f.get("definition", "").lower()


def test_tablet_sense_anchored_corpus_rescue_and_domain_wsd():
    """Module 1: 'tablet' locks to stone/wood/metal inscription sense and rescues a stone/clay corpus example."""
    e_tab = LE.get_ldoce_entry("tablet")
    quote_tab = "The ancient laws were carved on stone tablets and displayed in the city center."
    idx_tab = LE._lock_sense(e_tab, quote=quote_tab, target_pos="noun")
    locked_tab = e_tab["senses"][idx_tab]
    assert "stone" in locked_tab.get("definition", "").lower()
    ex_tab = LE.ensure_distinct_example("tablet", "noun", "", quote_tab, sense=locked_tab)
    assert ex_tab
    assert any(w in ex_tab.lower() for w in ("stone", "clay", "marble", "memorial", "carved", "inscribed", "ancient"))


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

    # 3. 'take [sb] back to [sth]' locks to memory/past-time sense, not return goods to a shop
    cands_tb = LE.build_top_k_expression_candidates(
        "take [sb] back to [sth]",
        "Hearing that old song always takes me back to my childhood summers.",
        expr_type="phrasal verb",
    )
    assert cands_tb
    assert any(w in cands_tb[0]["definition"].lower() for w in ("remember", "past", "time", "memory"))

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
    assert "a roof over one's head" in skeleton_md

