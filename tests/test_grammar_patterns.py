import pytest
import spacy
from librarian.grammar_patterns import GrammarPatternEngine, DECLARATIVE_GRAMMAR_PATTERNS


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="module")
def engine(nlp):
    return GrammarPatternEngine.get_engine(nlp)


def test_declarative_grammar_patterns_integrity():
    """Verify that all patterns have required fields and valid schema."""
    assert len(DECLARATIVE_GRAMMAR_PATTERNS) >= 10
    valid_categories = {
        "Rhetoric & Emphasis",
        "Logic & Stance",
        "Cohesion & Framing",
        "Information Packaging"
    }
    for pat in DECLARATIVE_GRAMMAR_PATTERNS:
        assert "id" in pat
        assert pat["category"] in valid_categories
        assert "priority" in pat and isinstance(pat["priority"], int)
        assert "formula" in pat
        assert len(pat["tree_patterns"]) >= 1


def test_match_cleft_sentence(nlp, engine):
    """Test accurate detection of It-cleft sentences."""
    sent = "It was in the library that John found the ancient manuscript."
    doc = nlp(sent)
    matches = engine.match_sentence(doc)
    assert any(m["pattern_id"] == "cleft_it_be_relcl" for m in matches)
    top_match = next(m for m in matches if m["pattern_id"] == "cleft_it_be_relcl")
    assert top_match["category"] == "Rhetoric & Emphasis"


def test_match_fronted_inversion(nlp, engine):
    """Test detection of negative adverb fronted inversion."""
    sent = "Never had he witnessed such an extraordinary spectacle."
    doc = nlp(sent)
    matches = engine.match_sentence(doc)
    assert any(m["pattern_id"] == "fronted_negative_inversion" for m in matches)
    match = next(m for m in matches if m["pattern_id"] == "fronted_negative_inversion")
    assert match["category"] == "Rhetoric & Emphasis"


def test_match_concessive_clause(nlp, engine):
    """Test detection of concessive subordinate clauses."""
    sent = "Although the experiment faced severe technical obstacles, the research team persevered."
    doc = nlp(sent)
    matches = engine.match_sentence(doc)
    assert any(m["pattern_id"] == "concession_clausal" for m in matches)
    match = next(m for m in matches if m["pattern_id"] == "concession_clausal")
    assert match["category"] == "Logic & Stance"


def test_match_conditional_clause(nlp, engine):
    """Test detection of conditional subordinate clauses."""
    sent = "If the temperature drops below zero, the water in the pipes will freeze."
    doc = nlp(sent)
    matches = engine.match_sentence(doc)
    assert any(m["pattern_id"] == "conditional_clausal" for m in matches)
    match = next(m for m in matches if m["pattern_id"] == "conditional_clausal")
    assert match["category"] == "Logic & Stance"


def test_match_propositional_encapsulation(nlp, engine):
    """Test detection of ', which + interpretive verb + that' framing."""
    sent = "The unemployment rate dropped significantly, which suggests that the economic reform is taking effect."
    doc = nlp(sent)
    matches = engine.match_sentence(doc)
    assert any(m["pattern_id"] == "propositional_encapsulation_which" for m in matches)
    match = next(m for m in matches if m["pattern_id"] == "propositional_encapsulation_which")
    assert match["category"] == "Cohesion & Framing"


def test_match_complex_transitive(nlp, engine):
    """Test detection of complex transitive structure (make/find/keep + obj + adj)."""
    sent = "The experienced teacher easily made the restless classroom quiet."
    doc = nlp(sent)
    matches = engine.match_sentence(doc)
    assert any(m["pattern_id"] == "complex_transitive_adj_complement" for m in matches)
    match = next(m for m in matches if m["pattern_id"] == "complex_transitive_adj_complement")
    assert match["category"] == "Information Packaging"


def test_match_nonfinite_participial_adjunct(nlp, engine):
    """Test detection of non-finite participial adjuncts."""
    sent = "Arriving at the summit before dawn, the hikers watched the sunrise in complete silence."
    doc = nlp(sent)
    matches = engine.match_sentence(doc)
    assert any(m["pattern_id"] == "nonfinite_participial_adjunct" for m in matches)
    match = next(m for m in matches if m["pattern_id"] == "nonfinite_participial_adjunct")
    assert match["category"] == "Information Packaging"


def test_match_inverted_conditional(nlp, engine):
    """Test detection of inverted conditionals (Had/Were/Should)."""
    sent = "Had I known about the changes earlier, I would have planned differently."
    doc = nlp(sent)
    matches = engine.match_sentence(doc)
    assert any(m["pattern_id"] == "inverted_conditional_subjunctive" for m in matches)
    match = next(m for m in matches if m["pattern_id"] == "inverted_conditional_subjunctive")
    assert match["category"] == "Logic & Stance"
    assert "Had" in match["formula"]


def test_match_with_compound_construction(nlp, engine):
    """Test detection of with + NP + participle construction."""
    sent = "With so many projects pending, the manager worked late into the night."
    doc = nlp(sent)
    matches = engine.match_sentence(doc)
    assert any(m["pattern_id"] == "with_compound_construction" for m in matches)
    match = next(m for m in matches if m["pattern_id"] == "with_compound_construction")
    assert match["category"] == "Information Packaging"


def test_match_absolute_construction(nlp, engine):
    """Test detection of nominative absolute construction."""
    sent = "Weather permitting, we will set out at dawn tomorrow."
    doc = nlp(sent)
    matches = engine.match_sentence(doc)
    assert any(m["pattern_id"] == "absolute_construction" for m in matches)
    match = next(m for m in matches if m["pattern_id"] == "absolute_construction")
    assert match["category"] == "Information Packaging"


def test_fronted_inversion_does_not_match_normal_negation(nlp, engine):
    """Ensure fronted_negative_inversion does not trigger on normal subject-verb-not sentences."""
    sent = "Similarly, if a friend sends you a photo of himself, you won't find it too cool."
    doc = nlp(sent)
    matches = engine.match_sentence(doc)
    assert not any(m["pattern_id"] == "fronted_negative_inversion" for m in matches)

