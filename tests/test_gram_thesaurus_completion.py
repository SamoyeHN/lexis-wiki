import pytest
from librarian.linguistics import LinguisticEngine as LE


def test_pedlex_gram_and_thesaurus_enrichment():
    """Step 1: Verify entryhead <gram>, THESAURUS exponent <gram>, and structured grammar_boxes in pedlex.db."""
    # Entryhead / THESAURUS <gram> inheritance & cross-backfill
    added_prof = LE.get_word_gram_profile("added", "adjective")
    assert added_prof["only_before_noun"] is True

    additional_prof = LE.get_word_gram_profile("additional", "adjective")
    assert additional_prof["only_before_noun"] is True

    extra_thes = {x["word"]: x.get("grammar") for x in LE.get_ldoce_thesaurus("extra") if x.get("grammar")}
    assert extra_thes.get("spare") == "[only before noun]"
    assert extra_thes.get("additional") == "[only before noun]"
    assert extra_thes.get("added") == "[only before noun]"

    extra_prof = LE.get_word_gram_profile("extra", "adjective")
    assert extra_prof["only_before_noun"] is False
    assert extra_prof["not_before_noun"] is False

    done_prof = LE.get_word_gram_profile("done", "adjective")
    assert done_prof["not_before_noun"] is True

    advice_prof = LE.get_word_gram_profile("advice", "noun")
    assert advice_prof["pure_uncountable"] is True
    assert "countable_use" in advice_prof["forbidden_frames"]
    assert "advise" in advice_prof["comparison_peers"]

    suggest_prof = LE.get_word_gram_profile("suggest", "verb")
    assert "V_sb_to_do" in suggest_prof["forbidden_frames"]
    assert "V_to_do" in suggest_prof["forbidden_frames"]
    assert "V_doing" in suggest_prof["allowed_frames"]

    admire_prof = LE.get_word_gram_profile("admire", "verb")
    assert "progressive" in admire_prof["forbidden_frames"]


def test_sense_locking_syntactic_slot_alignment():
    """Step 2: Verify _lock_sense uses positional (<gram>) and grammar_box constraints to disambiguate senses."""
    e_comp = LE.get_ldoce_entry("complete")
    # Attributive slot ('a complete list') must NOT lock onto sense 2 ('[not before noun] finished')
    idx_attr = LE._lock_sense(e_comp, "adjective", "We need a complete list of items.")
    assert "[not before noun]" not in (e_comp["senses"][idx_attr].get("grammar") or "")

    # Predicative slot ('is now complete') MUST lock onto sense 2 ('[not before noun] finished')
    idx_pred = LE._lock_sense(e_comp, "adjective", "The building work is now complete.")
    assert "[not before noun]" in (e_comp["senses"][idx_pred].get("grammar") or "")


def test_evaluate_syntactic_slot_clash_and_double_key_exemption():
    """Step 3 & Step 4: Verify syntactic slot clash detection and double_key_collision exemption."""
    # 1. [only before noun] adjective ('additional', 'added', 'spare') in predicative slot ('Postage is extra.')
    for cand in ("additional", "added", "spare"):
        clash, reason = LE.evaluate_syntactic_slot_clash(
            "extra", cand, pos="adjective", quote="Postage is extra."
        )
        assert clash is True
        assert reason == "adj_only_before_noun_in_predicative_slot"
        is_dk, dk_kind = LE.double_key_collision(
            "extra", cand, pos="adjective", quote="Postage is extra."
        )
        assert is_dk is False
        assert dk_kind == "frame_only"

    # Conversely, in an attributive slot ('They need extra income.'), 'additional' has NO slot clash and IS a double-key
    clash_attr, _ = LE.evaluate_syntactic_slot_clash(
        "extra", "additional", pos="adjective", quote="They need extra income."
    )
    assert clash_attr is False
    is_dk_attr, dk_kind_attr = LE.double_key_collision(
        "extra", "additional", pos="adjective", quote="They need extra income."
    )
    assert is_dk_attr is True
    assert dk_kind_attr == "double_key_slot"

    # 2. [not before noun] adjective ('done') in attributive slot ('We need a complete list.')
    clash_done, reason_done = LE.evaluate_syntactic_slot_clash(
        "complete", "done", pos="adjective", quote="We need a complete list."
    )
    assert clash_done is True
    assert reason_done == "adj_not_before_noun_in_attributive_slot"

    # 3. Pure [uncountable] noun ('advice') in countable determiner slot ('He gave me a useful suggestion.')
    clash_adv, reason_adv = LE.evaluate_syntactic_slot_clash(
        "suggestion", "advice", pos="noun", quote="He gave me a useful suggestion."
    )
    assert clash_adv is True
    assert reason_adv == "uncountable_noun_in_countable_slot"
    is_dk_adv, _ = LE.double_key_collision(
        "suggestion", "advice", pos="noun", quote="He gave me a useful suggestion."
    )
    assert is_dk_adv is False

    # 4. Forbidden verb complement frame ('suggest' in 'V_sb_to_do' slot of 'advise')
    clash_sug, reason_sug = LE.evaluate_syntactic_slot_clash(
        "advise", "suggest", anchor_type="sb_to_do", pos="verb", quote="She advised him to leave early."
    )
    assert clash_sug is True
    assert reason_sug == "forbidden_frame_sb_to_do"
    is_dk_sug, _ = LE.double_key_collision(
        "advise", "suggest", anchor_type="sb_to_do", pos="verb", quote="She advised him to leave early."
    )
    assert is_dk_sug is False


def test_vocab_card_enrichment_and_semantic_discriminator():
    """Step 5: Verify zero-token vocabulary card enrichment and grammatical discriminator notes in quiz blueprints."""
    from librarian.processor import processor as proc

    # 1. Card enrichment for 'suggest' (Grammar Box + Collocations + Thesaurus)
    sug_enr = LE.get_vocab_card_enrichment(
        "suggest",
        pos="verb",
        definition="to tell someone your ideas about what they should do, where they should go etc",
        quote="She suggested going out for a walk."
    )
    assert "Don't say" in sug_enr["grammar_usage"] or "Pattern:" in sug_enr["grammar_usage"]

    # 2. Card enrichment for 'complete' in predicative slot ('[not before noun]')
    comp_enr = LE.get_vocab_card_enrichment(
        "complete",
        pos="adjective",
        definition="[not before noun] finished",
        quote="The building work is now complete."
    )
    assert "[not before noun]" in comp_enr["grammar_usage"]

    # 2b. Card enrichment for 'jargon': excludes 'jargon' itself and ranks closest peers (terminology, slang, term)
    # Also verify 'name' inside 'word' thesaurus has no false '[singular]' cross-sense pollution.
    word_thes_name = next((t for t in LE.get_ldoce_thesaurus("word") if t.get("word") == "name"), {})
    assert not word_thes_name.get("grammar")

    jargon_enr = LE.get_vocab_card_enrichment(
        "jargon",
        pos="noun",
        definition="[uncountable] words and expressions used in a particular profession or by a particular group of people, which are difficult for other people to understand – often used to show disapproval",
        quote="Your vocabulary is wide, you use professional jargon effortlessly."
    )
    assert "**jargon**" not in jargon_enr["thesaurus_note"]
    assert "**terminology** `[formal]`" in jargon_enr["thesaurus_note"]
    assert "**slang**" in jargon_enr["thesaurus_note"]
    assert "**term**" in jargon_enr["thesaurus_note"]

    # 2c. Card enrichment for 'prominence': 100% authentic dictionary collocations, zero '(a/the)' splices
    prom_enr = LE.get_vocab_card_enrichment(
        "prominence",
        pos="noun",
        definition="the fact of being important and well known",
        quote="Cliff came to prominence again in 1997."
    )
    assert "(a/the)" not in prom_enr["collocations"]
    assert "come to/rise to/achieve prominence (as something)" in prom_enr["collocations"]
    assert "prominence of" in prom_enr["collocations"]

    # 3. _semantic_discriminator_note surfaces syntactic_slot_clash and Grammar Box warnings
    note_extra = LE._semantic_discriminator_note(
        "extra",
        "more of something, in addition to the usual or standard amount or number",
        ["spare", "additional", "added"],
        dist_meta={
            "spare": "syntactic_slot_clash:adj_only_before_noun_in_predicative_slot",
            "additional": "syntactic_slot_clash:adj_only_before_noun_in_predicative_slot",
            "added": "syntactic_slot_clash:adj_only_before_noun_in_predicative_slot",
        },
        pos="adjective",
        quote="Postage is extra.",
    )
    assert "[only before noun]" in note_extra

    note_sug = LE._semantic_discriminator_note(
        "suggest",
        "to tell someone your ideas about what they should do, where they should go etc",
        ["advise", "recommend", "propose"],
        dist_meta={"advise": "grambox_peer:grammar_box_contrast"},
        pos="verb",
        quote="She suggested going out for a walk.",
    )
    assert "Grammar Box:" in note_sug

    # 4. Markdown rendering and structured quiz sanitization
    md_out = proc._format_as_markdown(
        {
            "title": "Test Unit Vocabulary",
            "vocabulary": [
                {
                    "word": "suggest",
                    "part_of_speech": "verb",
                    "definition": "to tell someone your ideas about what they should do, where they should go etc",
                    "quote": "She suggested going out for a walk.",
                    "example": "I suggest trying the local seafood.",
                }
            ],
        },
        "vocabulary",
        source_filename="Test_Unit.md",
    )
    assert "## [[suggest]] verb" in md_out
    assert "**Pattern**:" in md_out
    sanitized, _, _ = proc._sanitize_vocab_for_quiz([
        {
            "word": "suggest",
            "part_of_speech": "verb",
            "definition": "to tell someone your ideas about what they should do, where they should go etc",
            "quote": "She suggested going out for a walk.",
            "example": "I suggest trying the local seafood.",
        }
    ])
    assert "**Pattern**:" not in sanitized
    assert "**Collocations**:" not in sanitized
    assert "**Thesaurus Note**:" not in sanitized
    assert "## [[suggest]]" in sanitized


def test_main_entry_and_sense_grammar_edge_cases():
    """Verify compound adjective grammar isolation, partial-sense grammar disambiguation, and _verb_takes_object grammar awareness."""
    # 1. Compound adjective synthesized from verb ('energy-saving') must NOT inherit verb '[transitive]'
    es_entry = LE.get_ldoce_entry("energy-saving")
    assert es_entry is not None
    assert all("[transitive]" not in (s.get("grammar") or "") for s in es_entry.get("senses", []))

    # 2. Polysemous adjectives where only 1 sense is restricted ('complete', 'absent', 'blue') must NOT be globally restricted
    for w in ("complete", "absent", "blue"):
        prof = LE.get_word_gram_profile(w, "adjective")
        assert prof["only_before_noun"] is False, f"{w} falsely marked only_before_noun"
        assert prof["not_before_noun"] is False, f"{w} falsely marked not_before_noun"

    # Conversely, entryhead-restricted words ('afraid', 'lone', 'mere') MUST be globally restricted
    assert LE.get_word_gram_profile("afraid", "adjective")["not_before_noun"] is True
    assert LE.get_word_gram_profile("lone", "adjective")["only_before_noun"] is True
    assert LE.get_word_gram_profile("mere", "adjective")["only_before_noun"] is True

    # 3. _verb_takes_object uses PedLex verb sense grammar ([intransitive] vs [transitive])
    # Pure [intransitive] verbs (even with prepositional 'somebody' patterns like 'happen to somebody') must return False
    for intr_v in ("happen", "occur", "thrive", "arrive"):
        assert LE._verb_takes_object(intr_v) is False, f"{intr_v} should be intransitive"
    assert LE._verb_takes_object("achieve") is True

    # 4. Derived entries with runon_gram preserve their own grammar rather than dropping it or copying base verb grammar
    synth_derived = LE._resolve_ldoce_kind(
        "test-derived",
        "test-derived",
        {"pos": ["noun"], "runon_gram": "[uncountable]", "senses": []},
        "derived",
        "achieve",
        depth=0,
    )
    assert synth_derived["senses"][0].get("grammar") == "[uncountable]"


def test_derived_ly_ing_definitions_and_examples():
    """Verify derived -ly, -ing, and other entries in pedlex.db properly resolve definitions and examples."""
    # 1. Previously empty derived definitions (-ly adverbs and -ing nouns) now synthesize clean definitions from base_word
    for w in ("blushingly", "complaisantly", "discernibly", "ergonomically", "insouciantly", "psychokinetically",
              "belly dancing", "clanking", "kayaking", "strip mining", "terracing", "troubleshooting", "vlogging"):
        e = LE.get_ldoce_entry(w)
        assert e is not None and e.get("senses"), f"Missing entry/senses for {w}"
        defn = LE.format_ldoce_definition(e["senses"][0]) or e["senses"][0].get("definition", "")
        assert len(defn.strip()) > 10, f"Empty/short definition for {w}: {defn!r}"

    # 2. Previously polluted derived definitions have trailing '<base_word> <prep>/<prep>' pattern leaks stripped
    for w, bw in (("agonizing", "agonize"), ("benchmarking", "benchmark"), ("fumbling", "fumble"), ("milking", "milk"), ("piggybacking", "piggyback")):
        e = LE.get_ldoce_entry(w)
        defn = e["senses"][0].get("definition", "")
        assert not defn.lower().endswith(bw), f"Pattern leak still present in {w}: {defn!r}"
        assert "/" not in defn.split()[-1], f"Slash pattern leak in {w}: {defn!r}"

    # 3. Derived entries harvest authentic examples containing the derived word from base_data collocations/senses/corpus
    for w in ("striving", "enlightening", "increasing", "sharing"):
        e = LE.get_ldoce_entry(w)
        exs = e["senses"][0].get("examples") or []
        assert len(exs) >= 1, f"Expected harvested examples for {w}, got {exs}"

    # 4. ensure_distinct_example resolves authentic examples for derived -ly and -ing words via corpus and WordNet
    for w, pos in (("attentively", "adverb"), ("intensively", "adverb"), ("persevering", "adjective"), ("radiantly", "adverb"), ("ruthlessly", "adverb")):
        e = LE.get_ldoce_entry(w)
        s0 = e["senses"][0]
        ex = LE.ensure_distinct_example(w, pos, s0, f"Stub quote with {w}.")
        assert ex and w in ex.lower(), f"Failed to resolve distinct example for {w} ({pos}): {ex!r}"

    # 5. determine_contextual_pos accurately distinguishes participial adjective ('persevering') and verbal noun ('His striving')
    assert LE.determine_contextual_pos("persevering", "with persevering industry she revived the failing business") == "adjective"
    assert LE.determine_contextual_pos("striving", "His constant striving for perfection inspired the team.") == "noun"

