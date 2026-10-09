from librarian.linguistics import LinguisticEngine
from librarian.evaluator import _score_pedagogy, W_PEDAGOGY


def test_quantifier_phrase_distractors():
    # Calling generate_phrase_distractors directly for a quantifier frame
    dists = LinguisticEngine.generate_phrase_distractors("a piece of", count=3)
    assert len(dists) == 3
    for d in dists:
        # All distractors must be quantifier frames of the form 'a/an [Noun] of'
        assert d.endswith(" of")
        assert d.startswith("a ") or d.startswith("an ")
        assert "thinking" not in d
        assert "bring" not in d


def test_quantifier_head_blanking_skeleton():
    markdown_vocab = """## [[a piece of]]
- **Part of Speech**: phrase
- **Definition**: an individual article or amount of something
- **Quoted Sentence**: She found a piece of evidence.
"""
    skeletons = LinguisticEngine.build_precomputed_target_skeletons(markdown_vocab)
    assert len(skeletons) == 1
    skel = skeletons[0]

    assert skel["base_headword"] == "a piece of"
    assert skel["target_word"] == "piece"
    assert skel["part_of_speech"] == "phrase"
    assert skel["inflection"] == "singular quantifier noun"

    # Options must be head nouns, not full phrases
    assert "piece" in skel["prescribed_options"]
    assert len(skel["prescribed_options"]) == 4
    for opt in skel["prescribed_options"]:
        assert " " not in opt  # Each option is a single quantifier noun like 'piece', 'bit', 'slice'

    # Micro task must explicitly instruct quantifier frame blanking
    assert "____ of" in skel["micro_task"]
    assert "piece" in skel["micro_task"]
    assert "Fit into frame" in skel["micro_task"]


def test_quantifier_evaluator_gate_acceptance():
    # Test that evaluator accepts both target_word='a piece of' and target_word='piece'
    # when the stem is 'She discovered a ____ of evidence.' and selected option is 'piece'
    stem = "She discovered a ____ of evidence."
    options = ["piece", "bit", "slice", "sheet"]
    
    quiz_items_1 = [{
        "target_word": "a piece of",
        "question": stem,
        "options": options,
        "correct_answer_index": 0,
        "explanation": "A piece of is correct here."
    }]
    score1, flags1 = _score_pedagogy(quiz_items_1, "quiz")
    assert score1 == W_PEDAGOGY
    assert not any("not matching options" in f for f in flags1)

    quiz_items_2 = [{
        "target_word": "piece",
        "question": stem,
        "options": options,
        "correct_answer_index": 0,
        "explanation": "Piece is correct here."
    }]
    score2, flags2 = _score_pedagogy(quiz_items_2, "quiz")
    assert score2 == W_PEDAGOGY
    assert not any("not matching options" in f for f in flags2)
