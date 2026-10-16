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
