from pathlib import Path
from librarian.linguistics import LinguisticEngine


def test_step0_sanitize_and_wiki_sources():
    raw_sample = (
        "---\n"
        'title: "Unit 1:   Test   Title"\n'
        "---\n\n"
        "# The Power of Habit\n\n"
        "(Posted on January 22, 2015 by Karen)\n\n"
        'The    posture  you take sends an"upright"message to your   audience\n\n'
        "Many students find it difficult to balance their academic studies and part-time jobs\n\n"
        "## Syllabus Vocabulary\n"
        "posture   n.\n"
        "upright   adj.\n"
    )
    sanitized = LinguisticEngine.sanitize_source_markdown(raw_sample)
    # 1. Frontmatter & Syllabus untouched
    assert 'title: "Unit 1:   Test   Title"' in sanitized
    assert "posture   n." in sanitized
    # 2. Title-case heading without period untouched
    assert "# The Power of Habit\n" in sanitized
    # 3. Byline not given a trailing period
    assert "(Posted on January 22, 2015 by Karen)\n" in sanitized
    # 4. Double spaces collapsed, stuck quotes separated, missing terminal period added
    assert 'The posture you take sends an "upright" message to your audience.' in sanitized
    assert "Many students find it difficult to balance their academic studies and part-time jobs." in sanitized

    # Run sanitize_source_markdown on all wiki/*/sources/*.md and verify 100% Verbatim Quote Gate
    wiki_root = Path("wiki")
    if wiki_root.exists():
        verbatim_misses = []
        for src_file in sorted(wiki_root.glob("*/sources/*.md")):
            orig = src_file.read_text(encoding="utf-8")
            cleaned = LinguisticEngine.sanitize_source_markdown(orig, source_path=src_file)
            _, pool = LinguisticEngine.tokenize_and_index_sentences(cleaned)
            norm_src = " ".join(cleaned.split())
            for sid, sent in pool.items():
                norm_sent = " ".join(sent.split())
                if norm_sent not in norm_src:
                    verbatim_misses.append((src_file.as_posix(), sid, sent))
        assert not verbatim_misses, f"Verbatim misses found: {verbatim_misses[:5]}"


def test_step1_quote_and_sentence_boundaries():
    # Case 1A: Quote ending with '.' followed by an independent capitalized sentence -> MUST split into 2 sentences
    p1 = 'The coach shouted, "Never give up." Everyone cheered loudly.'
    _, pool1 = LinguisticEngine.tokenize_and_index_sentences(p1)
    sents1 = list(pool1.values())
    assert len(sents1) == 2, f"Expected 2 sentences, got {sents1}"
    assert sents1[0] == 'The coach shouted, "Never give up."'
    assert sents1[1] == "Everyone cheered loudly."

    # Case 1B: Multi-sentence dialogue inside curly quotes -> MUST stay unified as 1 sentence
    p2 = "Their attitude was “Ah! You’re here! We can start now!” Next morning they left."
    _, pool2 = LinguisticEngine.tokenize_and_index_sentences(p2)
    sents2 = list(pool2.values())
    assert len(sents2) == 2, f"Expected 2 sentences, got {sents2}"
    assert sents2[0] == 'Their attitude was "Ah! You\'re here! We can start now!"'
    assert sents2[1] == "Next morning they left."

    # Case 1C: Multi-sentence quote followed by lowercase attribution -> MUST stay unified as 1 sentence
    p3 = '"We must act now. Tomorrow is too late," added the director.'
    _, pool3 = LinguisticEngine.tokenize_and_index_sentences(p3)
    sents3 = list(pool3.values())
    assert len(sents3) == 1, f"Expected 1 sentence, got {sents3}"
    assert sents3[0] == '"We must act now. Tomorrow is too late," added the director.'

    # Case 1D: Quote followed by Proper-Noun speech attribution -> MUST stay unified as 1 sentence
    p4 = '“We have no choice!” Mary said quietly. Then she walked out.'
    _, pool4 = LinguisticEngine.tokenize_and_index_sentences(p4)
    sents4 = list(pool4.values())
    assert len(sents4) == 2, f"Expected 2 sentences, got {sents4}"
    assert sents4[0] == '"We have no choice!" Mary said quietly.'
    assert sents4[1] == "Then she walked out."


def test_step2_all_caps_pos():
    cases = [
        ("warning", "WARNING: HIGH VOLTAGE CAN CAUSE SEVERE INJURY.", "noun"),
        ("high", "WARNING: HIGH VOLTAGE CAN CAUSE SEVERE INJURY.", "adjective"),
        ("voltage", "WARNING: HIGH VOLTAGE CAN CAUSE SEVERE INJURY.", "noun"),
        ("can", "WARNING: HIGH VOLTAGE CAN CAUSE SEVERE INJURY.", "verb"),
        ("cause", "WARNING: HIGH VOLTAGE CAN CAUSE SEVERE INJURY.", "verb"),
        ("severe", "WARNING: HIGH VOLTAGE CAN CAUSE SEVERE INJURY.", "adjective"),
        ("injury", "WARNING: HIGH VOLTAGE CAN CAUSE SEVERE INJURY.", "noun"),
        ("do", "DO NOT THROW AWAY.", "verb"),
        ("throw", "DO NOT THROW AWAY.", "verb"),
        ("away", "DO NOT THROW AWAY.", "adverb"),
        ("keep", "KEEP OUT OF REACH OF CHILDREN.", "verb"),
        ("reach", "KEEP OUT OF REACH OF CHILDREN.", "noun"),
    ]
    for w, sent, expected in cases:
        got = LinguisticEngine.determine_contextual_pos(w, sent)
        assert got == expected, f"ALL-CAPS POS fail for {w!r} in {sent!r}: got {got!r}, expected {expected!r}"


def test_step3_parenthetical_main_verb():
    cases = [
        ("devastated", "The crisis—which lasted three years—devastated trade.", "verb"),
        ("lasted", "The crisis—which lasted three years—devastated trade.", "verb"),
        ("trade", "The crisis—which lasted three years—devastated trade.", "noun"),
        ("devastated", "The crisis, which lasted three years, devastated trade.", "verb"),
        ("damaged", "The storm—which struck at midnight—damaged crops.", "verb"),
        ("transformed", "The policy (introduced last month) transformed education.", "verb"),
        ("collapsed", "The bridge—built in 1920—collapsed yesterday.", "verb"),
        ("devastated", "The tall, devastated soldiers marched home.", "adjective"),
        ("marched", "The tall, devastated soldiers marched home.", "verb"),
    ]
    for w, sent, expected in cases:
        got = LinguisticEngine.determine_contextual_pos(w, sent)
        assert got == expected, f"Parenthetical POS fail for {w!r} in {sent!r}: got {got!r}, expected {expected!r}"


def test_step4_byline_filtering():
    passage = (
        "# Global Trade Shifts\n\n"
        "(Posted on January 22, 2015 by Karen)\n\n"
        "International trade has expanded rapidly over the past decade.\n\n"
        "(Updated on March 3, 2020 by Editorial Staff)\n\n"
        "Although tariffs rose slightly, global supply chains remained resilient."
    )
    _, pool = LinguisticEngine.tokenize_and_index_sentences(passage)
    for sid, s in pool.items():
        assert "Posted on" not in s and "Updated on" not in s, f"Byline leaked into sentence_pool: {sid}: {s}"
    assert len(pool) == 2, f"Expected 2 sentences in pool, got {pool}"
    g_skels = LinguisticEngine.mine_grammar_skeletons(pool, target_count=5)
    for g in g_skels:
        assert "Posted on" not in g["quote"]


def test_same_sentence_multi_occurrence_homographs():
    # 1. Exact surface match priority over lemma-only match ("costs" vs "cost" in same sentence)
    sent1 = "The high cost of materials costs the company millions."
    assert LinguisticEngine.determine_contextual_pos("cost", sent1) == "noun"
    assert LinguisticEngine.determine_contextual_pos("costs", sent1) == "verb"

    # 2. Identical surface form occurring twice with different POS roles ("cost" noun + "cost" verb)
    sent2 = "The cost of the damage will cost millions."
    all_poses = LinguisticEngine.determine_all_contextual_poses("cost", sent2)
    assert [p for _, _, p in all_poses] == ["noun", "verb"]
    assert LinguisticEngine.determine_contextual_pos("cost", sent2, occurrence=0) == "noun"
    assert LinguisticEngine.determine_contextual_pos("cost", sent2, occurrence=1) == "verb"
    assert LinguisticEngine.determine_contextual_pos("cost (v.)", sent2) == "verb"
    assert LinguisticEngine.determine_contextual_pos("cost (n.)", sent2) == "noun"
    assert LinguisticEngine.determine_contextual_pos("cost", sent2, preferred_pos="verb") == "verb"

    # 3. Another identical surface homograph pair ("work" verb + "work" noun)
    sent3 = "They work hard at their work."
    all_work = LinguisticEngine.determine_all_contextual_poses("work", sent3)
    assert [p for _, _, p in all_work] == ["verb", "noun"]
    assert LinguisticEngine.determine_contextual_pos("work", sent3, occurrence=0) == "verb"
    assert LinguisticEngine.determine_contextual_pos("work", sent3, occurrence=1) == "noun"
    assert LinguisticEngine.determine_contextual_pos("work [noun]", sent3) == "noun"


def test_paragraph_labels_with_single_letter_sentence_start_and_expr_dedup():
    raw_labeled = (
        "Taking soundings\n"
        "A Until recently it was thought that dolphins were the only mammals to use echolocation.\n"
        "B The ability of great whales to use sound to communicate has been known for decades.\n"
        "C Echolocation is a different form of sensory perception.\n"
        "D A major concern of environmentalists is that high-power military sonar might dis\n"
        "orientate or harm whales.\n"
        "E The phenomenon of stranding is not well understood. When a single whale is found dead on a beach "
        "it might have died at sea and been washed up on shore. In other cases the carcasses would have "
        "washed up along different parts of the shoreline.\n"
    )
    cleaned = LinguisticEngine.sanitize_source_markdown(raw_labeled)
    paras = [p for p in cleaned.splitlines() if p.strip()]
    assert len(paras) == 6
    assert paras[4].startswith("A major concern of environmentalists")
    assert "disorientate" in paras[4]
    assert paras[5].startswith("The phenomenon of stranding")

    exprs = LinguisticEngine.extract_deterministic_expressions(
        cleaned, target_count=10, enable_llm_disambiguation=False
    )
    expr_words = [e["word"] for e in exprs]
    # Ensure wash up and wash up on do not collide in the same extraction batch
    for i, w1 in enumerate(expr_words):
        for w2 in expr_words[i + 1:]:
            assert not LinguisticEngine.are_same_word_family(w1, w2), f"Collision: {w1} vs {w2}"


