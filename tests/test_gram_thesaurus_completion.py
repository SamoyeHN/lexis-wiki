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

    # 4. Markdown rendering and quiz sanitization
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
    assert "**Grammar & Usage**:" in md_out
    sanitized = proc._sanitize_vocab_for_quiz(md_out)
    assert "**Grammar & Usage**:" not in sanitized
    assert "**Collocations**:" not in sanitized
    assert "**Thesaurus Note**:" not in sanitized

