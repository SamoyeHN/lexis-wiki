"""
Unit tests for Declarative Multi-Word Expression Pattern Engine.
Validates phrasal verbs, prepositional verbs, three-part combinations, idiomatic possessive frames, and light verbs.
"""

import pytest
import spacy
from librarian.expression_patterns import ExpressionPatternEngine


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="module")
def engine(nlp):
    return ExpressionPatternEngine.get_engine(nlp)


def test_standard_intransitive_phrasal_verb(nlp, engine):
    doc = nlp("He woke up early in the morning and showed up on time.")
    matches = engine.match_sentence(doc)
    assert len(matches) >= 2
    phrases = [m["phrase"] for m in matches]
    assert "wake up" in phrases
    assert "show up" in phrases
    wake_m = next(m for m in matches if m["phrase"] == "wake up")
    assert wake_m["pattern_formula"] == "wake up"
    assert wake_m["type"] == "phrasal verb"


def test_transitive_phrasal_verb_with_object(nlp, engine):
    doc = nlp("The engineers carried out the complex experiment yesterday.")
    matches = engine.match_sentence(doc)
    assert any(m["phrase"] == "carry out" for m in matches)
    carry_m = next(m for m in matches if m["phrase"] == "carry out")
    assert carry_m["pattern_formula"] == "carry [sb/sth] out" or "carry out" in carry_m["phrase"]
    assert carry_m["type"] == "phrasal verb"


def test_three_part_phrasal_verb(nlp, engine):
    doc = nlp("We look forward to meeting our new colleagues.")
    matches = engine.match_sentence(doc)
    assert any(m["phrase"] == "look forward to" for m in matches)
    look_m = next(m for m in matches if m["phrase"] == "look forward to")
    assert look_m["pattern_formula"] == "look forward to [sth/sb]"
    assert look_m["type"] == "phrasal verb"


def test_touch_construction(nlp, engine):
    doc = nlp("They always keep in touch with former classmates.")
    matches = engine.match_sentence(doc)
    assert any("keep in touch with" in m["phrase"] for m in matches)
    touch_m = next(m for m in matches if "keep in touch with" in m["phrase"])
    assert touch_m["type"] == "idiom"
    assert touch_m["pattern_formula"] == "keep in touch with [sb]"


def test_idiomatic_possessive_frame(nlp, engine):
    doc = nlp("She finally made up her mind after long discussions.")
    matches = engine.match_sentence(doc)
    assert any(m["phrase"] == "make up [one's] mind" for m in matches)
    mind_m = next(m for m in matches if m["phrase"] == "make up [one's] mind")
    assert mind_m["type"] == "collocation"
    assert mind_m["pattern_formula"] == "make up [one's] mind"


def test_light_verb_prepositional_idiom(nlp, engine):
    doc = nlp("We must take into account all experimental variables.")
    matches = engine.match_sentence(doc)
    assert any(m["phrase"] == "take into account" for m in matches)
    take_m = next(m for m in matches if m["phrase"] == "take into account")
    assert take_m["type"] == "idiom"
    assert take_m["pattern_formula"] == "take into account [sth]"


def test_verb_dependent_preposition(nlp, engine):
    doc = nlp("Students rely on accurate data to substantiate claims.")
    matches = engine.match_sentence(doc)
    assert any(m["phrase"] == "rely on" for m in matches)
    rely_m = next(m for m in matches if m["phrase"] == "rely on")
    assert rely_m["type"] == "phrasal verb"
    assert rely_m["pattern_formula"] == "rely on [sth/sb]"


def test_transitive_verb_preposition_collocation(nlp, engine):
    doc = nlp("The organization will provide students with modern equipment.")
    matches = engine.match_sentence(doc)
    assert any(m["phrase"] == "provide ... with" for m in matches)
    prov_m = next(m for m in matches if m["phrase"] == "provide ... with")
    assert prov_m["type"] == "collocation"
    assert prov_m["pattern_formula"] == "provide [sb] with [sth]"


def test_adjective_dependent_preposition(nlp, engine):
    doc = nlp("Students must be aware of potential hazards and interested in environmental science.")
    matches = engine.match_sentence(doc)
    phrases = [m["phrase"] for m in matches]
    assert "aware of" in phrases
    assert "interested in" in phrases

    aware_m = next(m for m in matches if m["phrase"] == "aware of")
    assert aware_m["type"] == "collocation"
    assert aware_m["pattern_formula"] == "aware of [sth/sb]"

    int_m = next(m for m in matches if m["phrase"] == "interested in")
    assert int_m["type"] == "collocation"
    assert int_m["pattern_formula"] == "interested in [sth/sb]"


def test_three_part_prepositional_noun_frame(nlp, engine):
    doc = nlp("The government took action in response to public concerns at the expense of economic speed.")
    matches = engine.match_sentence(doc)
    phrases = [m["phrase"] for m in matches]
    assert "in response to" in phrases
    assert "at the expense of" in phrases

    resp_m = next(m for m in matches if m["phrase"] == "in response to")
    assert resp_m["type"] == "set phrase"
    assert resp_m["pattern_formula"] == "in response to [sth/sb]"

    exp_m = next(m for m in matches if m["phrase"] == "at the expense of")
    assert exp_m["type"] == "set phrase"
    assert exp_m["pattern_formula"] == "at the expense of [sth/sb]"


def test_noun_dependent_preposition(nlp, engine):
    doc = nlp("Students have access to online academic journals and a talent for scientific research.")
    matches = engine.match_sentence(doc)
    phrases = [m["phrase"] for m in matches]
    assert "access to" in phrases
    assert "talent for" in phrases

    acc_m = next(m for m in matches if m["phrase"] == "access to")
    assert acc_m["type"] == "collocation"
    assert acc_m["pattern_formula"] == "access to [sth/sb]"


def test_phrase_distractors_and_authentic_cloze():
    from librarian.linguistics import LinguisticEngine

    # 1. Distractor generation for declarative patterns
    adj_d = LinguisticEngine.generate_phrase_distractors("aware of", count=3)
    assert len(adj_d) == 3
    assert any("of" in d for d in adj_d)

    frame_d = LinguisticEngine.generate_phrase_distractors("at the expense of", count=3)
    assert len(frame_d) == 3
    assert any("of" in d for d in frame_d)

    # 2. Authentic cloze matching for multi-word phrases
    vocab_text = """
## [[keep in touch with (sb)]]
- **Part of Speech**: phrasal verb
- **Definition**: to maintain communication with someone
- **Quoted Sentence**: "We decided to keep in touch with our old colleagues."

## [[aware of]]
- **Part of Speech**: collocation
- **Definition**: knowing about something
- **Quoted Sentence**: "She was fully aware of the upcoming changes."
"""
    items = LinguisticEngine.build_authentic_cloze_items(vocab_text, target_count=5)
    assert len(items) == 2
    
    t_words = [it["target_word"] for it in items]
    assert "keep in touch with" in t_words
    assert "aware of" in t_words

    for it in items:
        assert it["question"].count("____") == 1
        assert len(it["precomputed_distractors"]) >= 2
        assert it["is_multi_word"] is True


def test_single_and_multi_word_quota_balance():
    from librarian.linguistics import LinguisticEngine

    vocab_text = """
## [[wordone]]
- **Part of Speech**: noun
- **Definition**: word one definition
- **Quoted Sentence**: "This is wordone in sentence."

## [[wordtwo]]
- **Part of Speech**: verb
- **Definition**: word two definition
- **Quoted Sentence**: "They wordtwo every day."

## [[wordthree]]
- **Part of Speech**: adjective
- **Definition**: word three definition
- **Quoted Sentence**: "She is very wordthree."

## [[wordfour]]
- **Part of Speech**: noun
- **Definition**: word four definition
- **Quoted Sentence**: "Here is wordfour."

## [[wordfive]]
- **Part of Speech**: noun
- **Definition**: word five definition
- **Quoted Sentence**: "Look at wordfive."

## [[wordsix]]
- **Part of Speech**: noun
- **Definition**: word six definition
- **Quoted Sentence**: "And wordsix too."

## [[phrase one]]
- **Part of Speech**: collocation
- **Definition**: phrase one definition
- **Quoted Sentence**: "We use phrase one here."

## [[phrase two]]
- **Part of Speech**: collocation
- **Definition**: phrase two definition
- **Quoted Sentence**: "And phrase two there."

## [[phrase three]]
- **Part of Speech**: collocation
- **Definition**: phrase three definition
- **Quoted Sentence**: "Also phrase three now."

## [[phrase four]]
- **Part of Speech**: collocation
- **Definition**: phrase four definition
- **Quoted Sentence**: "Finally phrase four done."
"""
    items = LinguisticEngine.build_authentic_cloze_items(vocab_text, target_count=10)
    assert len(items) == 10
    multis = [it for it in items if it.get("is_multi_word")]
    singles = [it for it in items if not it.get("is_multi_word")]
    # Target 60% single words + 40% multi-words
    assert len(multis) == 4
    assert len(singles) == 6

    # Verify interleaved distribution: no single group monopolizes the beginning
    assert not items[0].get("is_multi_word")
    assert not items[1].get("is_multi_word")
    assert items[2].get("is_multi_word")


