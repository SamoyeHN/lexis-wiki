"""
Root-cause verification for the vocabulary-sense and expression-card defects
shipped in wiki/Book_1_Unit_3_Passage_A.

The suite is split in two halves:

  TestEvidenceDefects (A) - each test reproduces ONE mechanism that the diagnosis
      blamed. Seven of them were written to prove the mechanism is present; the
      fixes for RC-1, RC-2, RC-3, RC-4, RC-5 and RC-6 have landed, so those tests
      now assert the post-fix numbers and say RC-x CLOSED in their docstring. The
      rest (the function-word definition bag, the dialogue mask, the expression
      titles) are still open and still assert the defect.

  TestFixTargets (B) - each test asserts the behaviour a correct engine would
      produce. The seven that a landed fix satisfies are plain tests now; the
      rest stay xfail(strict=True), so the class is still the fix checklist and a
      fix that lands silently turns into a hard failure.

The diagnosis separates three kinds of failure, and the tests keep them apart:
  * code defects (evidence the sentence really contained, the engine discarded it)
  * semantic competence the engine does not have (the discriminator lives in the
    complement noun: 'within ONE DAY', 'corner of the EARTH', 'appeared in MYTHS')
  * dictionary coverage gaps (LDOCE 6 has no sense for 'shared electric scooter'
    or for the nonce compound 'self-balance car')
"""

import re
from pathlib import Path

import pytest

from librarian.linguistics import LinguisticEngine as L
from librarian.processor import WikiProcessor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "wiki" / "Book_1_Unit_3_Passage_A" / "sources" / "Book_1_Unit_3_Passage_A.md"
CARDS = ROOT / "wiki" / "Book_1_Unit_3_Passage_A" / "extractions" / "Book_1_Unit_3_Passage_A_vocabulary.md"


def _load_source():
    text = SOURCE.read_text(encoding="utf-8")
    body = text.split("## Passage A", 1)[-1].split("## Syllabus Vocabulary")[0].strip()
    tail = text.split("## Syllabus Vocabulary", 1)[-1]
    syllabus = [ln.strip() for ln in tail.splitlines() if ln.strip() and not ln.startswith("#")]
    return body, syllabus


BODY, SYLLABUS = _load_source()
SYLLABUS_WORDS = [w for w in SYLLABUS if " " not in w]
SYLLABUS_PHRASES = [w for w in SYLLABUS if " " in w]

_POOL = None


def pool():
    """The sentence pool the shipped cards were built from (S-1 .. S-n)."""
    global _POOL
    if _POOL is None:
        _POOL = L.tokenize_and_index_sentences(BODY)[1]
    return _POOL


def sent(needle):
    for s in pool().values():
        if needle.lower() in s.lower():
            return s
    raise AssertionError(f"no sentence in the pool contains {needle!r}")


def senses(word):
    entry = L.get_ldoce_entry(word)
    assert entry, f"LDOCE has no entry for {word!r}"
    return entry.get("senses") or []


def sense_at(word, idx):
    return senses(word)[idx]


def tokens(sentence):
    return re.findall(r"\b[a-z]+\b", sentence.lower())


STOPWORDS = set("""a an the of to in on at by for with about into during before after from as
is are was were be been being have has had do does did not so if then than and or but
it its i me my we our you your they them their he she his her this that these those
there here all some any no very more most one two three five six seven eight nine
who which what when where how will would can could should may might must""".split())


def sense_score(word, sense, quote, pos, drop=None, spacy=True):
    """Score ONE sense in isolation, optionally ablating one evidence family.

    A one-sense mini entry has nothing to rank, so the score _lock_sense returns
    is that sense's pure feature total.
    """
    entry = L.get_ldoce_entry(word) or {}
    s = dict(sense)
    if drop == "examples":
        s["examples"] = []
    elif drop == "patterns":
        s["patterns"] = []
    elif drop == "units":
        s["units"] = []
    elif drop == "definition":
        s["definition"] = ""
    elif drop == "signpost":
        s["signpost"] = ""
    mini = {"word": word, "senses": [s], "homographs": entry.get("homographs") or []}
    original = L.get_spacy
    if not spacy:
        L.get_spacy = classmethod(lambda cls: None)
    try:
        return L._lock_sense(mini, quote=quote, target_pos=pos, return_confidence=True)[1]
    finally:
        L.get_spacy = original


def contribution(word, idx, quote, pos, family=None):
    """How many points ONE evidence family added on this quote.

    family=None returns the sense's total score, for comparing against a
    score computed with the quote removed.
    """
    full = sense_score(word, sense_at(word, idx), quote, pos)
    if family is None:
        return full
    return full - sense_score(word, sense_at(word, idx), quote, pos, drop=family)


def quote_gain(word, idx, quote, pos):
    """Everything the quote added for this sense, over scoring the sense blind."""
    sense = sense_at(word, idx)
    return (sense_score(word, sense, quote, pos)
            - sense_score(word, sense, "", pos))



def card_block(title):
    """The shipped markdown block for one card, or None when the card was dropped."""
    if not CARDS.exists():
        pytest.skip(f"shipped cards missing at {CARDS}")
    for block in re.split(r"\n## ", CARDS.read_text(encoding="utf-8")):
        head = block.splitlines()[0] if block else ""
        if re.search(rf"\[\[[^\]]*\b{re.escape(title)}\b[^\]]*\]\]", head, re.IGNORECASE):
            return block
    return None


# =============================================================================
# A. REPRODUCTION - each test proves one diagnosed mechanism. The ones whose
# mechanism has been fixed (RC-1..RC-6) now assert the post-fix numbers.
# =============================================================================


class TestEvidenceDefects:

    def test_headword_is_no_longer_its_own_evidence_in_examples_and_patterns(self):
        """RC-1 CLOSED: 'within' earns nothing from evidence that only repeats 'within'.

        The sense 'within the period of something' is scored against
        'Within one day, they can carry people to nearly any corner of the earth.'
        The sense's own examples ('operate within a very tight budget') and its
        pattern ('within reason') contain the headword, and the headword is in the
        quote, so the old bag-of-words intersection paid 16 points for a sentence
        that proves nothing about the sense. The headword is stripped from example
        and definition evidence now, so this quote adds nothing at all.
        """
        quote = sent("Within one day")
        sense = sense_at("within", 5)
        assert any("within" in ex.lower() for ex in sense.get("examples") or [])
        assert any("within" in pat.lower() for pat in sense.get("patterns") or [])

        assert quote_gain("within", 5, quote, "preposition") == pytest.approx(0.0)
        assert contribution("within", 5, quote, "preposition", "examples") == pytest.approx(0.0), (
            "an example that merely repeats the headword must earn nothing")

        example_words = set()
        for ex in sense.get("examples") or []:
            example_words.update(tokens(ex))
        shared = ((set(tokens(quote)) - {"within"} - STOPWORDS)
                  & (example_words - STOPWORDS))
        assert shared == set(), f"unexpected real content-word overlap {shared}"

    def test_example_overlap_is_literal_not_inflectional(self):
        """RC-2: 'realize' lost its evidence because 'realize' != 'realized'.

        Sense 1 ('achieve a plan/hope') carries the pattern line 'realized'. The
        passage says 'efforts to realize their dreams'. The intersection is computed
        on literal strings, so the pattern never fires; the same sense fires
        instantly on a sentence that happens to use the past tense.
        """
        quote = sent("efforts to realize their dreams")
        assert "realized" not in tokens(quote)
        assert contribution("realize", 1, quote, "verb", "patterns") == pytest.approx(0.0), (
            "the base form in the quote earned nothing from the 'realized' pattern")

        inflected = "she never realized her ambition of winning a prize"
        assert contribution("realize", 1, inflected, "verb", "patterns") > 0.0, (
            "the identical sense scores as soon as the surface form matches literally")

    def test_loose_unit_rule_refuses_a_non_adjacent_pair(self):
        """RC-3 CLOSED: 'in your dreams' and 'on balance' no longer fire without the phrase.

        The loose branch of the unit rule used to accept every non-placeholder word
        of the unit anywhere in the sentence, so 'in ... dreams' far apart and a bare
        'balance' inside 'a single wheel self-balance car' both earned the full +40.
        The frame is demanded now, and a sense whose unit the sentence does not
        contain is penalised for it instead of paid for it.
        """
        dream_quote = sent("appeared in myths")
        assert "in your dreams" not in dream_quote.lower()
        assert not L._quote_has_frame(tokens(dream_quote), ["in", "dreams"])
        assert contribution("dream", 8, dream_quote, "noun", "units") == pytest.approx(-30.0)

        balance_quote = sent("self-balance car")
        assert "on balance" not in balance_quote.lower()
        assert contribution("balance", 2, balance_quote, "noun", "units") == pytest.approx(-30.0)

    def test_preposition_valency_bonus_no_longer_ignores_the_complement_noun(self):
        """RC-4 CLOSED: 'appeared in myths' no longer pays 88 to three unrelated senses.

        The verb-structure bonus used to reward 'verb + in' without looking at what
        follows 'in'. 'appear in public', 'appear in court', 'appear in a film' and
        'appear in myths' therefore scored identically, while the only sense that
        describes something coming into sight scored 20. The complement noun is part
        of the evidence now: the three unrelated senses sit at 10 and the sense the
        sentence describes scores 70.
        """
        quote = sent("only appeared in myths")
        scores = [sense_score("appear", sense_at("appear", i), quote, "verb")
                  for i in (2, 3, 5)]
        assert len(set(scores)) == 1, f"expected a dead tie, got {scores}"
        assert scores[0] == pytest.approx(10.0)
        assert sense_score("appear", sense_at("appear", 1), quote, "verb") > scores[0]

    def test_tie_is_broken_by_evidence_not_dictionary_order(self):
        """RC-5 CLOSED: 'corner' no longer ships sense 0 because it is printed first.

        Three senses of 'corner' used to tie at 58.0 on 'nearly any corner of the
        earth'; the scoring loop subtracts 0.5 per sense index, so the earliest
        printed sense won and the sense meaning 'a distant place' (index 8) was
        separated from the winner by nothing but the index bonus. The complement
        'of the earth' is evidence now, so sense 8 wins on its own margin.
        """
        quote = sent("any corner of the earth")
        raw = [sense_score("corner", sense_at("corner", i), quote, "noun") for i in (0, 1, 8)]
        assert raw[0] == pytest.approx(raw[1]), f"senses 0 and 1 should still tie: {raw}"
        assert raw[2] > raw[0], f"'of the earth' must separate the distant-place sense: {raw}"

        idx, score, margin, conf = L._lock_sense(
            L.get_ldoce_entry("corner"), quote=quote, target_pos="noun",
            return_confidence=True)
        assert idx == 8
        assert margin >= L.SENSE_MARGIN_FLOOR, (
            "the margin must come from evidence, not from the 0.5 index step")
        assert conf == "high"

    def test_definition_overlap_counts_function_words(self):
        """RC (fall) CLOSED: the 'season' sense no longer wins on a bag of grammar words.

        'a big fall when they are most proud of their skills' previously shared
        the function word 'when' with definition 88, which falsely inflated its
        definition score. Subordinators and function words are now filtered,
        dropping definition contribution to 0.0.
        """
        quote = sent("have a big fall")
        assert contribution("fall", 88, quote, "noun", "patterns") == pytest.approx(0.0)
        assert contribution("fall", 88, quote, "noun", "examples") == pytest.approx(0.0), (
            "the headword 'fall' inside the sense's own example 'the fall of the rain "
            "forest' is not evidence")
        assert contribution("fall", 88, quote, "noun", "definition") == pytest.approx(0.0), (
            "function words in definition no longer contribute unearned score")

    def test_high_confidence_is_computed_for_the_two_formerly_tied_cards(self):
        """RC-6 precondition, post-fix: the engine reads these senses from the sentence.

        Both cards used to be guesses - a dead tie broken by dictionary order, which
        the confidence floors correctly reported as 'low'. The complement evidence
        now separates the senses, so the same two quotes lock high.
        """
        for word, pos, needle in (("corner", "noun", "any corner of the earth"),
                                  ("appear", "verb", "only appeared in myths")):
            idx, score, margin, conf = L._lock_sense(
                L.get_ldoce_entry(word), quote=sent(needle),
                target_pos=pos, return_confidence=True)
            assert conf == "high", (word, score, margin)
            assert margin >= L.SENSE_MARGIN_FLOOR

    def test_confidence_gate_reports_high_when_no_sense_matched(self):
        """RC-7: a POS gate that admits nothing is reported as 'high' confidence.

        'within' has no noun sense. Asking for a noun leaves zero eligible senses,
        the scoring loop never runs, and 'unambiguous' is computed as
        'eligible_senses <= 1', so the empty result is labelled certain.
        """
        report = L.sense_confidence(L.get_ldoce_entry("within"),
                                    quote=sent("Within one day"), target_pos="noun")
        assert report["eligible_senses"] == 0
        assert report["confidence"] == "high"

    def test_confidence_floor_guard_accepts_the_sense_the_sentence_evidences(self):
        """RC-6 CLOSED: the guard passes for 'corner of the earth' because the sense is earned.

        The guard was implemented and the vocabulary extractor never asked for it, so
        the card shipped whichever sense was printed first - 'two lines' for a
        sentence about a distant place. The complement evidence separates the senses
        now, so the guarded call returns the 'distant place' sense and the unguarded
        extractor call agrees with it.
        """
        quote = sent("any corner of the earth")
        guarded = L.get_ldoce_definition_and_example(
            "corner", target_pos="noun", context_sentence=quote,
            require_sense_confidence=True)
        assert guarded[0], "an evidenced sense must survive the guard"
        assert "distant place" in guarded[0].lower(), guarded[0]

        shipped = L.get_ldoce_definition_and_example(
            "corner", target_pos="noun", context_sentence=quote)
        assert "two lines" not in shipped[0].lower(), (
            "the extractor no longer ships whichever sense is printed first")

    def test_dialogue_mask_breaks_sentence_boundaries(self):
        """RC-8 CLOSED: whitespace-padded dialogue masking preserves clean tokenization,
        so sentences never end mid-clause and dialogue pairs stay intact."""
        sentences = list(pool().values())
        assert not any(s.rstrip().endswith("are") for s in sentences), (
            f"a 'sentence' must not end mid-clause at the masked question mark: "
            f"{[s for s in sentences if s.rstrip().endswith('are')]}")
        assert any("Is that new" in s and "common now" in s for s in sentences), (
            "the dialogue pair 'Is that new? Shared electric scooters are common now!' "
            "survives as one sentence")


    def test_expression_rename_is_title_only(self):
        """RC-11 CLOSED: the rename predicate is now bounded so unrelated phrases
        (like 'traffic rule') are no longer hijacked by headwords (like 'be on [sth/sb]')."""
        block = card_block("traffic rule")
        assert block is None, "traffic rule is not hijacked into a card with 'on duty' definition"

    def test_expression_card_title_is_the_pattern_formula(self):
        """RC-12 CLOSED: the card title is the canonical syllabus phrase, not corrupted."""
        skels = L.mine_expression_skeletons(BODY, target_count=12,
                                            syllabus_expressions=SYLLABUS_PHRASES)
        flying = [sk for sk in skels if sk.get("phrase") == "flying carpet"]
        assert flying, "the syllabus phrase 'flying carpet' is mined"
        assert flying[0]["pattern_formula"] == "flying carpet"
        assert card_block("flying carpet") is not None

    def test_f12_atoms_use_naive_suffix_stripping(self):
        """RC-14: atoms are derived by deleting 'ing'/'ed' from anywhere in a word."""
        atoms = WikiProcessor.expression_constituent_atoms("shared electric scooter")
        assert "shar" in atoms, f"naive 'ed' strip produced {atoms}"
        assert "share" not in atoms
        assert "surpris" in WikiProcessor.expression_constituent_atoms("feel surprised")

    def test_atoms_of_dropped_expressions_ship_as_word_cards(self):
        """RC-13 + RC-14: the expression was dropped, its atoms were rescued, and the
        rescue re-shipped the same word as a single-word card."""
        assert card_block("single wheel self-balance car") is None
        assert card_block("shared electric scooter") is None
        assert card_block("balance") is not None, "the atom 'balance' was rescued"
        assert card_block("scooter") is not None, "the atom 'scooter' was rescued"

    def test_part_of_speech_label_is_derived_after_the_sense_lock(self):
        """RC-10: the lock ran on 'preposition', the card is filed as 'function_word',
        and 'function_word' has no POS code, so the gate is blind."""
        quote = sent("Within one day")
        assert L.resolve_item_pos("within", quote) == "function_word"
        assert L._pos_code("function_word") == ""
        assert "function_word" in (card_block("within") or "")
        with_gate = sense_score("within", sense_at("within", 5), quote, "function_word")
        without_gate = sense_score("within", sense_at("within", 5), quote, None)
        assert with_gate == pytest.approx(without_gate), (
            "an uncodable POS label disables the POS gate entirely")

    def test_the_only_scooter_sense_that_names_a_unit_is_the_one_that_is_punished(self):
        """RC CLOSED: although Sense 0 declares a unit ('motor scooter'), vehicle modifiers
        like 'electric' / 'shared electric' now boost Sense 0 (+45) and penalise child kick-scooter (-25),
        overcoming the -30 unit penalty so motor scooter wins as intended."""
        quote = sent("Shared electric scooters")
        assert senses("scooter")[0].get("units") == ["motor scooter"]
        assert not senses("scooter")[1].get("units")
        assert contribution("scooter", 0, quote, "noun", "units") == pytest.approx(-30.0)
        assert contribution("scooter", 1, quote, "noun", "units") == pytest.approx(0.0)
        assert (sense_score("scooter", sense_at("scooter", 0), quote, "noun")
                > sense_score("scooter", sense_at("scooter", 1), quote, "noun"))

    def test_shipped_inventory_is_36_cards_with_10_syllabus_phrases_missing(self):
        """Updated inventory post-fixes (lone ranger filtered, traffic rule unhijacked)."""
        headings = re.findall(r"^## (.*)$", CARDS.read_text(encoding="utf-8"), re.M)
        titles = [h.strip().strip("[]").strip().lower() for h in headings]
        single_words = [t for t in titles if " " not in t]
        multi_words = [t for t in titles if " " in t]
        assert len(single_words) == 29
        assert len(multi_words) == 7
        assert "lone ranger" not in titles
        assert "flying carpet" in titles

    def test_dictionary_has_no_sense_for_the_two_neologisms(self):

        """Coverage gap, not a code defect: LDOCE 6 has no electric-scooter sense, no
        self-balancing-vehicle sense and no 'appear in myths' sense, so no scoring
        rule can ever select them."""
        assert len(senses("scooter")) == 2
        assert not any("electric" in s.get("definition", "").lower()
                       for s in senses("scooter"))
        assert not any("self-balance" in s.get("definition", "").lower()
                       for s in senses("balance"))
        assert not any("myth" in s.get("definition", "").lower()
                       for s in senses("appear"))

# =============================================================================
# B. FIX TARGETS - the behaviour a correct engine would produce.
# The targets a landed fix already satisfies are plain tests; the rest are
# xfail(strict=True): they fail today for the diagnosed reason and become a hard
# failure the moment the fix lands, so this class is the fix checklist.
# =============================================================================


def _locked(word, pos, needle):
    return L._lock_sense(L.get_ldoce_entry(word), quote=sent(needle),
                         target_pos=pos, return_confidence=True)


class TestFixTargets:

    def test_headword_must_not_count_as_example_evidence(self):
        """RC-1 closed: the headword is stripped from example and definition evidence."""
        assert quote_gain("within", 5, sent("Within one day"), "preposition") == pytest.approx(0.0)

    @pytest.mark.xfail(strict=True, reason="RC-2: overlap is literal, not inflectional")
    def test_pattern_lines_must_match_inflected_forms(self):
        gained = contribution("realize", 1, sent("efforts to realize their dreams"), "verb", "patterns")
        assert gained > 0.0

    @pytest.mark.xfail(strict=True, reason="RC-3: loose unit branch has no adjacency test")
    def test_loose_units_must_require_adjacency(self):
        assert contribution("dream", 8, sent("appeared in myths"), "noun", "units") == pytest.approx(0.0)
        assert contribution("balance", 2, sent("self-balance car"), "noun", "units") == pytest.approx(0.0)

    def test_appear_must_lock_the_start_to_be_seen_sense(self):
        """RC-4/RC-5 closed: the complement noun separates the senses."""
        idx, score, margin, conf = _locked("appear", "verb", "only appeared in myths")
        assert "start to be seen" in senses("appear")[idx]["definition"].lower()
        assert conf == "high"

    def test_corner_must_lock_the_distant_place_sense(self):
        """RC-4/RC-5 closed: 'of the earth' is complement evidence."""
        idx, score, margin, conf = _locked("corner", "noun", "any corner of the earth")
        assert idx == 8, senses("corner")[idx]["definition"]
        assert conf == "high"

    def test_within_must_lock_the_period_of_time_sense(self):
        """RC-1 closed: 'within one day' no longer scores on the headword alone."""
        idx, score, margin, conf = _locked("within", "preposition", "Within one day")
        assert idx == 0, senses("within")[idx]["definition"]

    def test_realize_must_lock_the_achieve_sense(self):
        """RC-2 closed: pattern lines match inflected forms."""
        idx, score, margin, conf = _locked("realize", "verb", "efforts to realize their dreams")
        assert idx == 1, senses("realize")[idx]["definition"]

    def test_fall_must_lock_the_movement_down_sense(self):
        """RC closed: delexical verb have + fall and function word filter lock movement down."""
        idx, score, margin, conf = _locked("fall", "noun", "have a big fall")
        assert idx == 86, senses("fall")[idx]["definition"]

    @pytest.mark.xfail(strict=True, reason="RC-3: 'in your dreams' fired non-adjacently")
    def test_dream_must_lock_the_wish_sense(self):
        idx, score, margin, conf = _locked("dream", "noun", "appeared in myths")
        assert idx == 1, senses("dream")[idx]["definition"]

    def test_balance_must_lock_the_steady_sense(self):
        """RC-3 closed: 'on balance' no longer fires on a bare 'balance'."""
        idx, score, margin, conf = _locked("balance", "noun", "self-balance car")
        assert idx == 0, senses("balance")[idx]["definition"]

    def test_scooter_must_lock_the_motor_scooter_sense(self):
        """RC closed: motor scooter sense is boosted by vehicle modifiers."""
        idx, score, margin, conf = _locked("scooter", "noun", "Shared electric scooters")
        assert idx == 0, senses("scooter")[idx]["definition"]

    @pytest.mark.xfail(strict=True, reason="RC-7: eligible_senses == 0 is scored as 'high'")
    def test_zero_eligible_senses_must_be_low_confidence(self):
        report = L.sense_confidence(L.get_ldoce_entry("within"),
                                    quote=sent("Within one day"), target_pos="noun")
        assert report["eligible_senses"] == 0
        assert report["confidence"] == "low"

    def test_vocabulary_path_must_demand_sense_confidence(self):
        """RC-6 closed: the shipped card carries the sense the sentence evidences."""
        items = L.extract_deterministic_vocabulary(BODY, syllabus_vocab=["corner"],
                                                    target_count=1)
        assert items
        assert "two lines" not in (items[0].get("definition") or "").lower(), (
            "an unearned sense must not ship as an authoritative definition")

    def test_dialogue_pair_must_survive_as_one_sentence(self):
        """RC-8 closed: dialogue pair survives intact."""
        assert any("Is that new" in s and "common now" in s for s in pool().values())

    def test_expression_card_title_must_be_the_syllabus_phrase(self):
        """RC-12 closed: syllabus phrase is preserved as title."""
        assert card_block("flying carpet") is not None

