"""Unit tests for the Orthographic / Paronym Distractor Tier and Cognate Near-Synonym Guard."""

from librarian.linguistics import LinguisticEngine as LE


def test_cognate_near_synonym_guard_blocks_ambiguous_pairs():
    """Verify that root-sharing near-synonyms (-ic/-ical, continual/continuous, specimen/species, activism/activity)
    are strictly rejected by is_safe_orthographic_paronym and never appear in paronym distractors."""
    blocked_pairs = [
        ("historic", "historical", "a"),
        ("historical", "historic", "a"),
        ("classic", "classical", "a"),
        ("economic", "economical", "a"),
        ("electric", "electrical", "a"),
        ("comic", "comical", "a"),
        ("continual", "continuous", "a"),
        ("specimen", "species", "n"),
        ("activism", "activity", "n"),
        ("activism", "action", "n"),
        ("imaginative", "unimaginative", "a"),
        ("considerate", "inconsiderate", "a"),
        ("considerate", "considered", "a"),
    ]
    for target, cand, pos in blocked_pairs:
        assert not LE.is_safe_orthographic_paronym(target, cand, pos=pos), (
            f"Expected Cognate Near-Synonym Guard to block ({target}, {cand}, {pos})"
        )
        paronyms = LE.get_orthographic_paronym_distractors(target, pos=pos, target_count=6)
        assert cand not in paronyms, (
            f"Expected '{cand}' NOT to appear in orthographic paronyms for '{target}': {paronyms}"
        )


def test_orthographic_paronym_harvester_finds_true_paronyms():
    """Verify that classic confusable paronyms and orthogonal-domain root peers are harvested."""
    expected_paronyms = [
        ("adapt", "v", "adopt"),
        ("complement", "v", "compliment"),
        ("simulate", "v", "stimulate"),
        ("industrial", "a", "industrious"),
        ("imaginative", "a", "imaginable"),
        ("considerate", "a", "considerable"),
        ("respectful", "a", "respective"),
    ]
    for target, pos, expected in expected_paronyms:
        assert LE.is_safe_orthographic_paronym(target, expected, pos=pos), (
            f"Expected ({target}, {expected}, {pos}) to pass is_safe_orthographic_paronym"
        )
        cands = LE.get_orthographic_paronym_distractors(target, pos=pos, target_count=5)
        assert expected in cands, (
            f"Expected '{expected}' in paronym candidates for '{target}', got {cands}"
        )


def test_orthogonal_domain_root_peers_for_specimen():
    """Verify that orthogonal-domain Latin root peers (specimen -> spectrum, spectacle, speculation)
    pass the guard while same-domain 'species' is blocked."""
    cands = LE.get_orthographic_paronym_distractors("specimen", pos="n", target_count=6)
    assert "species" not in cands
    orthogonal_hits = {"spectrum", "spectacle", "speculation"} & set(cands)
    assert len(orthogonal_hits) >= 2, f"Expected at least 2 of spectrum/spectacle/speculation in {cands}"


def test_generate_vocab_distractors_blends_or_fills_orthographic_paronym():
    """Verify that generate_vocab_distractors integrates orthographic_paronym metadata and discriminator notes."""
    dists, meta = LE.generate_vocab_distractors("specimen", pos="n", return_metadata=True)
    assert len(dists) == 3
    assert "species" not in dists
    assert any(v == "orthographic_paronym" for v in meta.values()), (
        f"Expected at least one orthographic_paronym in specimen distractors: {dists}, {meta}"
    )
    note = LE._semantic_discriminator_note(
        "specimen",
        "a single example of something, often an animal or plant",
        dists,
        meta,
        pos="n",
    )
    assert "orthographic look-alike / paronym" in note or "Contrast:" in note


def test_potluck_excludes_taboo_paronym_and_ranks_primary_domain_coordinates():
    """Verify that taboo/proper-noun paronyms (e.g. polack) are blocked and primary-sense domain
    alignment ranks genuine meal coordinates (feast, supper, picnic, banquet, buffet, brunch) ahead
    of secondary-sense homographs like 'mess'."""
    assert not LE.is_safe_orthographic_paronym("potluck", "polack", pos="n")
    dists, meta = LE.generate_vocab_distractors("potluck", pos="n", return_metadata=True)
    assert len(dists) == 3
    assert "polack" not in dists
    assert "mess" not in dists
    meal_coords = {"feast", "supper", "snack", "picnic", "banquet", "buffet", "brunch", "dinner", "lunch", "breakfast"}
    assert set(dists).issubset(meal_coords), f"Expected genuine meal coordinates, got {dists} ({meta})"


def test_snappywords_indirect_antonym_bridging():
    """Verify Upgrade 1: cross-POS pertainym/derivation antonym projection (reluctantly -> willingly,
    scarcity -> abundance) and synset-peer / satellite-head antonym bridging."""
    adv_ants = LE.get_antonyms("reluctantly", pos="r")
    assert "willingly" in adv_ants, f"Expected 'willingly' via cross-POS pertainym projection, got {adv_ants}"

    noun_ants = LE.get_antonyms("scarcity", pos="n")
    assert "abundance" in noun_ants, f"Expected 'abundance' via derivation projection, got {noun_ants}"

    verb_ants = LE.get_antonyms("win", pos="v", definition="be the winner in a contest or competition; be victorious")
    assert "lose" in verb_ants or "fail" in verb_ants, f"Expected 'lose' or 'fail' in {verb_ants}"

    # Verify generate_vocab_distractors harvests adverb antonyms as Slot 1
    dists, meta = LE.generate_vocab_distractors("reluctantly", pos="r", return_metadata=True)
    assert len(dists) == 3
    assert any(v == "antonym" for v in meta.values()), f"Expected antonym slot for 'reluctantly', got {dists} ({meta})"


def test_snappywords_verb_entails_causes_and_noun_comeronyms():
    """Verify Upgrades 2 & 3: verb script relations (entails/causes, e.g. snore -> sleep)
    and noun co-meronyms (eyelid -> retina/cornea/lens under holonym 'eye')."""
    v_dists, v_meta = LE.generate_vocab_distractors("snore", pos="v", return_metadata=True)
    assert "sleep" in v_dists, f"Expected entailed verb 'sleep' in snore distractors, got {v_dists} ({v_meta})"

    n_dists, n_meta = LE.generate_vocab_distractors("eyelid", pos="n", return_metadata=True)
    eye_parts = {"retina", "cornea", "lens", "iris", "pupil", "eyeball"}
    assert len(set(n_dists) & eye_parts) >= 2, (
        f"Expected at least 2 co-meronym eye parts for 'eyelid', got {n_dists} ({n_meta})"
    )


def test_find_rhymes_phonetic_skeleton_and_rhyme_tail_tricks():
    """Verify Find Rhymes Tricks 1 & 2: consonant skeleton ('rel_cns') and rhyme tail ('rel_rhy')
    index and boost phonetic-orthographic confusables."""
    # Trick 1: Consonant skeleton equivalence across vowel variations and c/s/k/ph/qu/x rules
    assert LE._phonetic_consonant_skeleton("adapt") == LE._phonetic_consonant_skeleton("adopt") == LE._phonetic_consonant_skeleton("adept") == "dpt"
    assert LE._phonetic_consonant_skeleton("prescribe") == LE._phonetic_consonant_skeleton("proscribe") == "prskrb"
    assert LE._phonetic_consonant_skeleton("precede") == LE._phonetic_consonant_skeleton("proceed") == "prsd"
    assert LE._phonetic_consonant_skeleton("complement") == LE._phonetic_consonant_skeleton("compliment") == "kmplmnt"
    assert LE._phonetic_consonant_skeleton("elicit") == LE._phonetic_consonant_skeleton("illicit") == "lst"
    assert LE._phonetic_consonant_skeleton("deceased") == LE._phonetic_consonant_skeleton("diseased") == "dsd"

    # Trick 2: Rhyme tail normalization (-cede/-ceed -> -seed, syllable alignment)
    assert LE._phonetic_rhyme_tail("precede") == LE._phonetic_rhyme_tail("recede") == LE._phonetic_rhyme_tail("secede")
    assert LE._estimate_syllables("precede") == LE._estimate_syllables("proceed") == 2

    # Verify precede harvests rhyme/skeleton peers (recede, secede, preclude, proceed)
    precede_cands = LE.get_orthographic_paronym_distractors("precede", pos="v", target_count=5)
    assert len({"recede", "secede", "preclude", "proceed"} & set(precede_cands)) >= 2, (
        f"Expected rhyme/skeleton paronyms for 'precede', got {precede_cands}"
    )
    # Verify elicit harvests rhyme peer solicit
    elicit_cands = LE.get_orthographic_paronym_distractors("elicit", pos="v", target_count=5)
    assert "solicit" in elicit_cands, f"Expected rhyme paronym 'solicit' for 'elicit', got {elicit_cands}"

    # Verify adapt excludes nominal-primary homographs (pitch, gear, wire, shoehorn) and blends adopt
    adapt_dists, adapt_meta = LE.generate_vocab_distractors("adapt", pos="v", return_metadata=True)
    for bad_nom in ("pitch", "gear", "wire", "shoehorn", "square", "tame"):
        assert bad_nom not in adapt_dists, f"Expected nominal/adjectival homograph '{bad_nom}' excluded from adapt: {adapt_dists}"
    assert "adopt" in adapt_dists, f"Expected paronym 'adopt' in adapt distractors, got {adapt_dists} ({adapt_meta})"



