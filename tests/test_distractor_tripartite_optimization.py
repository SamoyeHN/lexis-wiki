# -*- coding: utf-8 -*-
"""Regression tests for the tripartite distractor optimization (Pillar 1/2/3)."""

from __future__ import annotations

from librarian.linguistics import LinguisticEngine as LE


def test_inflected_verb_paronym_symmetric_projection():
    """Inflected verb targets unwrap to lemma for paronym lookup and project back symmetrically."""
    inflected_res = LE.get_orthographic_paronym_distractors("condensed", pos="v", target_count=3)
    assert inflected_res, "Expected non-empty inflected paronyms for 'condensed'"
    assert all(w != "condensed" and w.endswith("ed") for w in inflected_res), (
        f"Expected '-ed' projected paronyms for 'condensed', got {inflected_res}"
    )

    base_res = LE.get_orthographic_paronym_distractors("condense", pos="v", target_count=3)
    assert base_res, "Expected non-empty base paronyms for 'condense'"
    assert not any(w.endswith("ed") for w in base_res), (
        f"Expected uninflected base lemmas for 'condense', got {base_res}"
    )


def test_adjective_valency_and_paronym_soft_quota():
    """Preposition-governing adjectives cap valency distractors at <= 2 and reserve Slot 3."""
    dists, meta = LE.generate_vocab_distractors(
        "allergic",
        pos="adj",
        context_anchor="to",
        anchor_type="prep",
        target_count=3,
        return_metadata=True,
    )
    assert len(dists) == 3, f"Expected 3 distractors, got {dists}"
    prep_val_count = sum(1 for d in dists if meta.get(d) == "preposition_valency")
    assert prep_val_count <= 2, f"Expected <= 2 preposition_valency distractors, got {meta}"


def test_noun_ontology_negative_filter():
    """Noun ontology exclusion acts as a negative filter and NEVER a positive slot clash."""
    assert LE.is_noun_ontology_compatible("applicant", "application") is False
    assert LE.is_noun_ontology_compatible("staff", "suite") is False
    assert LE.is_noun_ontology_compatible("staff", "room") is False
    assert LE.is_noun_ontology_compatible("staff", "crew") is True
    assert LE.evaluate_syntactic_slot_clash("staff", "suite", pos="noun") == (False, "")
    assert LE.is_safe_orthographic_paronym("applicant", "appliance", pos="n") is False
    assert "appliance" not in LE.generate_vocab_distractors("applicant", pos="noun", target_count=3)


def test_degree_adverb_gradability_clash():
    """Scalar vs totality degree adverbs mutually clash; dual-compatible adverbs never clash."""
    has_clash_adv, reason_adv = LE.evaluate_syntactic_slot_clash(
        "completely",
        "very",
        anchor="exhausted",
        pos="adv",
        quote="The soldiers were completely exhausted.",
    )
    assert has_clash_adv is True
    assert reason_adv == "scalar_adverb_in_extreme_adj_slot"

    has_clash_adj, reason_adj = LE.evaluate_syntactic_slot_clash(
        "impossible",
        "difficult",
        anchor="utterly",
        pos="adj",
        quote="The proposal was utterly impossible.",
    )
    assert has_clash_adj is True
    assert reason_adj == "scalar_adj_in_totality_adverb_slot"

    dual_clash, _ = LE.evaluate_syntactic_slot_clash(
        "completely",
        "quite",
        anchor="exhausted",
        pos="adv",
        quote="The soldiers were completely exhausted.",
    )
    assert dual_clash is False


def test_degree_adverb_arena_generation():
    """Generating distractors for a totality adverb modifying an extreme adj yields scalar foils."""
    dists, meta = LE.generate_vocab_distractors(
        "completely",
        pos="adv",
        context_anchor="exhausted",
        quote="The soldiers were completely exhausted.",
        target_count=3,
        return_metadata=True,
    )
    assert len(dists) == 3
    assert any(d in ("very", "extremely", "fairly", "rather", "slightly", "somewhat", "moderately") for d in dists), (
        f"Expected scalar degree adverb distractors, got {dists} ({meta})"
    )
    assert any(meta.get(d) == "gradability_contrast" for d in dists)


def test_cefr_unlisted_and_zero_example_gates_across_b2_c1_c2():
    """Unlisted/zero-example obscure words and cefrpy pseudo-A1/A2 artifacts are gated across B2-C2."""
    assert LE.is_cefr_compliant_distractor("tauten", "constrict", passage_level="B2") is False
    assert LE.is_cefr_compliant_distractor("tauten", "constrict", passage_level="C1") is False
    assert LE.is_cefr_compliant_distractor("advertorial", "obituary", passage_level="B2") is False
    assert LE.is_cefr_compliant_distractor("clamorous", "constrict", passage_level="B2") is False
    assert LE.is_cefr_compliant_distractor("clamorous", "constrict", passage_level="C1") is False
    assert LE.get_word_cefr("pretext") in ("C1", "C2")
    assert LE.is_cefr_compliant_distractor("pretext", "reason", passage_level="B1") is False


def test_book4_unit5_double_key_and_cefr_fixes():
    """Book_4_Unit_5_Section_A eliminates fascinated/interested, tautens, advertorial, and impolite."""
    from pathlib import Path

    vocab_path = Path("wiki/Book_4_Unit_5_Section_A/extractions/Book_4_Unit_5_Section_A_vocabulary.md")
    if not vocab_path.exists():
        return
    skels = LE.build_precomputed_target_skeletons(vocab_path.read_text(encoding="utf-8"), target_count=10)
    by_tw = {s.get("target_word"): set(s.get("prescribed_options") or []) for s in skels}

    if "intrigued" in by_tw:
        assert "fascinated" not in by_tw["intrigued"]
        assert "interested" not in by_tw["intrigued"]
        assert not (by_tw["intrigued"] & {"vulnerable", "immune", "susceptible"})
        assert sum(1 for w in by_tw["intrigued"] if w.endswith("ed")) >= 2
    if "constricts" in by_tw:
        assert "tautens" not in by_tw["constricts"]
    if "obituary" in by_tw:
        assert "advertorial" not in by_tw["obituary"]
    if "earthy" in by_tw:
        assert "impolite" not in by_tw["earthy"]


def test_inanimate_causal_subject_preserved_in_verb_frame():
    """Inanimate causal subjects (e.g. 'caffeine') are preserved even with 3-word objects and trailing preps."""
    frame = LE.derive_cloze_slot_frame_from_example(
        example_text="Caffeine constricts the blood vessels in your body.",
        final_target="constricts",
        base_word="constrict",
        canonical_pos="verb",
        anchor="vessels",
        anchor_type="object",
    )
    assert frame is not None
    frame_low = str(frame).lower()
    assert "caffeine" in frame_low, f"Expected causal subject 'caffeine' in frame, got {frame!r}"
    assert "blood vessels" in frame_low, f"Expected object 'blood vessels' in frame, got {frame!r}"


def test_antonym_polarity_lock_injected_in_prompt_context(monkeypatch):
    """Prompt context injects a Polarity Lock note when options contain an antonym distractor."""
    from types import SimpleNamespace
    from librarian.processor_core.quiz_prompt_context import QuizPromptContext, _build_vocabulary_context

    stub_processor = SimpleNamespace(
        _sanitize_vocab_for_quiz=lambda raw: (raw, [], []),
        _render_llm_surface_frame=lambda raw, sk: (raw, []),
    )
    ctx = QuizPromptContext()
    monkeypatch.setattr(
        LE,
        "build_precomputed_target_skeletons",
        classmethod(lambda cls, *a, **kw: [{
            "target_word": "constricts",
            "base_headword": "constrict",
            "part_of_speech": "verb",
            "definition": "to make something narrower or tighter",
            "cloze_slot_frame": "... caffeine ____ the blood vessels in",
            "prescribed_options": ["expands", "constricts", "assists", "passes"],
            "correct_answer_index": 1,
            "anchor_type": "object",
            "distractor_mechanisms": {"expands": "antonym", "assists": "antonym", "passes": "coordinate"},
            "micro_task": "Fit into frame '... caffeine ____ the blood vessels in'.",
        }]),
    )
    _build_vocabulary_context(stub_processor, {"content": "## [[constrict]]", "cefr_level": "B2"}, 1, ctx)
    vocab_content = ctx.kwargs["vocabulary_content"]
    assert "Polarity Lock: options include antonym 'expands'" in vocab_content


def test_cross_attested_everyday_collocation_and_sense_lock():
    """Cross-attested everyday collocations boost matching sense examples without breaking Sense Lock."""
    constrict_ent = LE.get_ldoce_entry("constrict")
    constrict_s0 = constrict_ent["senses"][0]
    best_ex, _, _ = LE.select_optimal_five_source_example(
        final_target="constrict",
        base_word="constrict",
        canonical_pos="verb",
        target_entry=constrict_ent,
        locked_sense=constrict_s0,
        t_s_idx=0,
        definition=constrict_s0.get("definition"),
    )
    assert best_ex and "throat" in best_ex.lower(), f"Expected 'throat' example for constrict, got {best_ex!r}"

    ext_ex = LE.ensure_distinct_example(
        "constrict",
        "verb",
        "",
        "No activist had managed to stop a single experiment or improve the lives of animals living in tiny, constricted enclosures.",
        sense=constrict_s0,
    )
    assert ext_ex and "throat" in ext_ex.lower(), f"Expected 'throat' from ensure_distinct_example, got {ext_ex!r}"

    blunt_ent = LE.get_ldoce_entry("blunt")
    blunt_s1 = blunt_ent["senses"][1]
    blunt_ex, _, _ = LE.select_optimal_five_source_example(
        final_target="blunt",
        base_word="blunt",
        canonical_pos="adj",
        target_entry=blunt_ent,
        locked_sense=blunt_s1,
        t_s_idx=1,
        definition=blunt_s1.get("definition"),
    )
    assert blunt_ex, "Expected non-empty example for blunt sense 1"
    for s0_noun in ("instrument", "knife", "knives", "pencil", "edge"):
        assert s0_noun not in blunt_ex.lower(), f"Sense 0 noun '{s0_noun}' leaked into blunt Sense 1: {blunt_ex!r}"


def test_verb_never_sliced_as_prep_noun_slot_on_spacy_nns_misparse():
    """Finite transitive verbs mis-tagged by spaCy as NNS/pobj (e.g. 'Fear of crime constricts ... lives')
    must fall through to verb-object slicing rather than emitting '... fear of ____'."""
    frame = LE.derive_cloze_slot_frame_from_example(
        example_text="Fear of crime constricts many people's lives.",
        final_target="constrict",
        base_word="constrict",
        canonical_pos="verb",
    )
    assert frame is not None
    frame_low = str(frame).lower()
    assert "of ____" not in frame_low, f"Broken noun-prep frame emitted for verb: {frame!r}"
    assert "lives" in frame_low, f"Expected verb object 'lives' in frame, got {frame!r}"


def test_wsd_local_window_prevents_distant_clause_drift_on_constrict():
    """Distant clause tokens ('lives' in 'improve the lives') must not override local
    syntactic context ('tiny, constricted enclosures') during WSD sense locking."""
    entry = LE.get_ldoce_entry("constrict")
    assert entry is not None
    quote = (
        "No activist had managed to stop a single experiment or improve the lives "
        "of animals living in tiny, constricted enclosures."
    )
    best_idx = LE._lock_sense(entry, quote=quote, target_pos="verb")
    assert best_idx == 0, f"Expected Sense 0 (narrower/tighter), got Sense {best_idx}"


def test_corpus_fallback_enforces_cefr_floor_and_ceiling_on_c1_passage():
    """Corpus fallback must enforce both CEFR floor (no A1/A2 giveaways like 'nice'/'dark' on C1)
    and CEFR ceiling even when cefrpy rates the target word itself as A1/B1."""
    for target in ("renewable", "extinct"):
        cands = LE.get_corpus_fallback_candidates(
            target, pos="a", target_count=5, passage_level="C1"
        )
        assert len(cands) == 5, f"Expected 5 C1 fallback candidates for {target}, got {cands}"
        assert "nice" not in cands and "dark" not in cands, f"Basic A1 word leaked into C1 fallback: {cands}"
        for c in cands:
            lvl = LE.get_word_cefr(c)
            if lvl:
                assert LE.CEFR_ORDER[lvl] >= 3, f"Sub-B1 fallback '{c}' ({lvl}) leaked for {target} on C1"
            assert LE.is_cefr_compliant_distractor(c, target, passage_level="C1"), (
                f"Fallback '{c}' violated C1 ceiling for {target}"
            )

    a2_cands = LE.get_corpus_fallback_candidates("simple", pos="a", target_count=3, passage_level="A2")
    assert len(a2_cands) == 3, f"Expected 3 A2 fallback candidates without starvation, got {a2_cands}"


