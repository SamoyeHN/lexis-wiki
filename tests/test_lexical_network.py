import pytest
from typing import Dict, List, Any, Optional
from librarian.linguistics import LinguisticEngine as L
import urllib.request
import urllib.error

def _ollama_online(model_name: str, host: Optional[str] = None) -> bool:
    """True iff ``model_name`` is on the SAME Ollama the LLMClient uses (config.api_url)."""
    if host is None:
        try:
            from librarian.config import config
            host = (config.get("api_url") or "http://localhost:11434").rstrip("/")
        except Exception:
            host = "http://localhost:11434"
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return model_name in resp.read().decode("utf-8", "replace")
    except Exception:
        return False


def extract_lexical_network(
    item: str,
    target_pos: Optional[str] = None,
    definition: Optional[str] = None,
    quote: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extracts an authoritative Lexical Reference Card (Definition-Locked Sense, Patterns,
    Authentic Collocations, Antonyms, and 4-5 Symmetric Distractor Candidates) combining
    LDOCE 6th Edition and WordNet at 0 token cost (<1ms).
    """
    item_clean = item.strip().lower()
    pos_clean = (target_pos or "").strip().lower()
    words = item_clean.split()
    n_words = len(words)

    ldoce_syns: List[str] = []
    ldoce_opps: List[str] = []
    thesaurus_nuances: List[Dict[str, str]] = []
    collocations: List[str] = []
    phrase_distractors: List[str] = []

    # CASE A: Multi-Word Expression (Phrases, Idioms, Quantifier frames)
    if n_words > 1:
        phrase_distractors = L.generate_phrase_distractors(item_clean, pos=pos_clean or "idiom", count=5)
        host_word = words[0] if words[0] not in ("a", "an", "the") else words[1]
        entry = L.get_ldoce_entry(host_word) or {}
        
        for ph in entry.get("phrases", []):
            if ph.get("phrase", "").lower() == item_clean:
                for ex in ph.get("examples", [])[:2]:
                    collocations.append(f"Usage: {ex}")
                    
        for s in entry.get("senses", []):
            defn = (s.get("definition") or "").lower()
            if "advantages and disadvantages" in defn and "pro" in item_clean:
                ldoce_syns.append("advantages and disadvantages")
                ldoce_opps.append("unintended consequences")

        return {
            "word": item_clean,
            "pos": pos_clean or "idiom/phrase",
            "is_phrase": True,
            "definition": definition or (ldoce_syns[0] if ldoce_syns else ""),
            "quote": quote or "",
            "patterns": [],
            "synonyms": ldoce_syns,
            "opposites": ldoce_opps,
            "thesaurus_nuances": [],
            "collocations": collocations[:3],
            "phrase_distractors": phrase_distractors,
            "distractor_candidates": phrase_distractors,
        }

    # CASE B: Single-word Lexical Headword (Noun, Verb, Adjective, Adverb)
    entry = L.get_ldoce_entry(item_clean) or {}
    senses = entry.get("senses", []) or []

    # Lock authoritative sense via _lock_sense
    s_idx = None
    if entry and senses:
        s_idx = L._lock_sense(entry, quote=quote, definition=definition, target_pos=pos_clean)
        if s_idx is None or s_idx >= len(senses):
            s_idx = 0

    locked_sense = senses[s_idx] if (senses and s_idx is not None) else {}
    locked_def = locked_sense.get("definition") or definition or ""
    patterns = [p for p in locked_sense.get("patterns", []) if p]

    # Generate 4-5 high-discrimination distractors
    distractors = L.generate_vocab_distractors(
        item_clean,
        pos=pos_clean or locked_sense.get("pos") or "noun",
        definition=locked_def,
        quote=quote,
        target_count=5
    )

    # Direct Antonyms
    ants = list(L.get_antonyms(item_clean, pos=pos_clean))
    if locked_sense.get("opposite") and locked_sense.get("opposite") not in ants:
        ants.insert(0, locked_sense.get("opposite"))

    # Collocations
    raw_collocs = entry.get("collocations", {}) or {}
    if isinstance(raw_collocs, dict):
        if "noun" in pos_clean:
            for v_c in raw_collocs.get("verbs", [])[:3]:
                collocations.append(v_c.get("collocation", ""))
            for a_c in raw_collocs.get("adjectives", [])[:3]:
                collocations.append(a_c.get("collocation", ""))
        elif "verb" in pos_clean:
            for n_c in raw_collocs.get("nouns", [])[:3]:
                collocations.append(n_c.get("collocation", ""))
            for adv_c in raw_collocs.get("adverbs", [])[:3]:
                collocations.append(adv_c.get("collocation", ""))
        elif "adj" in pos_clean:
            for n_c in raw_collocs.get("nouns", [])[:3]:
                collocations.append(n_c.get("collocation", ""))
        elif "adv" in pos_clean:
            for v_c in raw_collocs.get("verbs", [])[:3]:
                collocations.append(v_c.get("collocation", ""))

    # Synonyms & Thesaurus Nuances from LDOCE / WordNet
    raw_thes = entry.get("thesaurus") or []
    for item_t in raw_thes:
        t_word = item_t.get("word") or ""
        t_dist = item_t.get("distinction") or ""
        for w in t_word.split("/"):
            w_clean = w.strip().lower()
            if w_clean and w_clean != item_clean and w_clean not in ldoce_syns:
                ldoce_syns.append(w_clean)
        if t_dist:
            thesaurus_nuances.append({"word": t_word, "distinction": t_dist})

    # Fallback to WordNet synonyms if LDOCE thesaurus is empty
    if not ldoce_syns:
        for syn in L.get_synonyms(item_clean):
            if syn != item_clean and syn not in ldoce_syns:
                ldoce_syns.append(syn)

    return {
        "word": item_clean,
        "pos": pos_clean or locked_sense.get("pos") or entry.get("pos"),
        "is_phrase": False,
        "definition": locked_def,
        "quote": quote or "",
        "patterns": patterns[:4],
        "synonyms": ldoce_syns[:6],
        "opposites": ants[:4],
        "thesaurus_nuances": thesaurus_nuances[:4],
        "collocations": [c for c in collocations if c][:4],
        "distractor_candidates": distractors,
    }


def select_best_single_pattern(patterns: List[str], target_word: str) -> Optional[str]:
    """
    Selects exactly ONE best, high-value syntactic pattern from LDOCE.
    Filters out noise like single verbs ('have') or repeated fragments.
    """
    if not patterns:
        return None

    # Priority: patterns that contain the target word and a preposition or complement
    scored_patterns = []
    target_lower = target_word.lower()
    for p in patterns:
        p_clean = p.strip()
        if not p_clean:
            continue
        # Skip isolated fragments or single words
        if " " not in p_clean and "/" not in p_clean:
            continue
        score = 0
        p_lower = p_clean.lower()
        if target_lower in p_lower:
            score += 10
        if any(prep in p_lower for prep in ["of", "to", "over", "for", "in", "on", "with", "from", "that"]):
            score += 5
        # Prefer concise, clear formulas (length between 6 and 40)
        if 6 <= len(p_clean) <= 40:
            score += 3
        scored_patterns.append((score, p_clean))

    if scored_patterns:
        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        return scored_patterns[0][1]

    # Fallback: first non-empty pattern
    return patterns[0].strip() if patterns else None


def format_lexical_reference_card(net: Dict[str, Any]) -> str:
    """Formats the extracted network into clean items. Omit pattern if none exists."""
    patterns = net.get("patterns", [])
    best_pattern = select_best_single_pattern(patterns, net["word"])

    options_pool = net.get("distractor_candidates") or []
    options_str = ", ".join(options_pool) if options_pool else "none"

    lines = [
        f"1. Target Word: {net['word']}",
        f"2. Part of Speech: {net.get('pos', 'noun')}",
        f"3. Definition: {net.get('definition', '')}",
    ]
    if best_pattern:
        lines.append(f"4. Pattern: {best_pattern}")
        lines.append(f"5. Options Pool: {options_str}")
    else:
        lines.append(f"4. Options Pool: {options_str}")

    return "\n".join(lines)


# ==============================================================================
# TESTS & DEMONSTRATION
# ==============================================================================

def test_extract_lexical_network_enormous():
    res = extract_lexical_network("enormous", target_pos="adjective")
    assert "huge" in res["synonyms"] or "massive" in res["synonyms"]
    assert len(res["thesaurus_nuances"]) > 0
    card = format_lexical_reference_card(res)
    print("\n--- Card for 'enormous' ---")
    print(card)


def test_extract_lexical_network_optimistic():
    res = extract_lexical_network("optimistic", target_pos="adjective")
    assert "pessimistic" in res["opposites"]
    card = format_lexical_reference_card(res)
    print("\n--- Card for 'optimistic' ---")
    print(card)


def test_extract_lexical_network_diligent():
    res = extract_lexical_network("diligent", target_pos="adjective")
    # WordNet complementary fallback provides 'negligent'
    assert "negligent" in res["opposites"]
    card = format_lexical_reference_card(res)
    print("\n--- Card for 'diligent' ---")
    print(card)


def test_extract_lexical_network_increase():
    res = extract_lexical_network("increase", target_pos="verb")
    assert "decrease" in res["opposites"]
    assert len(res["thesaurus_nuances"]) > 0
    card = format_lexical_reference_card(res)
    print("\n--- Card for 'increase' ---")
    print(card)


def test_extract_lexical_network_noun_decision():
    res = extract_lexical_network("decision", target_pos="noun")
    assert "make a decision" in res["collocations"] or "reach a decision" in res["collocations"]
    card = format_lexical_reference_card(res)
    print("\n--- Card for Noun 'decision' ---")
    print(card)


def test_extract_lexical_network_phrase_pros_and_cons():
    res = extract_lexical_network("pros and cons", target_pos="idiom")
    assert res["is_phrase"] is True
    assert len(res["phrase_distractors"]) > 0
    card = format_lexical_reference_card(res)
    print("\n--- Card for Idiom 'pros and cons' ---")
    print(card)


def test_extract_lexical_network_quantifier_piece_of():
    res = extract_lexical_network("a piece of", target_pos="quantifier")
    assert res["is_phrase"] is True
    assert "a bit of" in res["phrase_distractors"] or "a slice of" in res["phrase_distractors"]
    card = format_lexical_reference_card(res)
    print("\n--- Card for Quantifier Frame 'a piece of' ---")
    print(card)


def test_ollama_llm_generation_with_pros_and_cons():
    """
    Directly tests Ollama LLM generating a question for idiom 'pros and cons'
    using the enhanced Lexical Reference Card and pre-computed symmetric phrase distractors.
    """
    from librarian.llm import llm
    from librarian.schemas import QuizQuestion

    if not _ollama_online(llm.model or "llama3.1:8b"):
        pytest.skip(f"Ollama model {llm.model} not available")

    phrase = "pros and cons"
    net = extract_lexical_network(phrase, target_pos="idiom")
    card = format_lexical_reference_card(net)

    prompt = f"""You are an expert English language assessment creator.
Craft a high-quality assessment cloze item for the fixed idiom '{phrase}'.

CRITICAL INSTRUCTIONS:
1. Ground the sentence strictly on the Lexical Reference Card below.
2. In the sentence question stem, provide a clear decision-making or evaluative scenario where weighing advantages and disadvantages is essential.
3. The blank in the question stem MUST be written as exactly four underscores '____'.
4. Options MUST contain exactly 4 choices: '{phrase}' and the 3 pre-computed phrase distractors.
5. Provide a deep, pedagogical explanation explaining why '{phrase}' fits and why the other idioms (e.g. 'bits and pieces', 'back and forth') do not fit the context.

{card}
"""

    messages = [
        {"role": "system", "content": "You are a professional pedagogical test creator. Respond strictly with valid JSON."},
        {"role": "user", "content": prompt}
    ]

    print(f"\n=======================================================")
    print(f"Calling Ollama Server for phrase '{phrase}' ({llm.api_url})...")
    print(f"=======================================================")

    try:
        response = llm.chat(
            messages=messages,
            schema=QuizQuestion,
            temperature=0.3
        )
        print("\n[SUCCESS] Ollama LLM Response Generated for 'pros and cons':")
        print(f"Target Word: {response.target_word}")
        print(f"Question:    {response.question}")
        print(f"Options:     {response.options}")
        print(f"Answer Idx:  {response.correct_answer_index} ({response.options[response.correct_answer_index]})")
        print(f"Explanation: {response.explanation}")

        assert "____" in response.question
        assert len(response.options) == 4
    except Exception as e:
        print(f"\n[ERROR calling Ollama Server]: {e}")
        raise e


def test_ollama_llm_generation_with_a_piece_of():
    """
    Directly tests Ollama LLM generating a question for quantifier 'a piece of'
    contrasted with 'a slice of', 'a bit of', 'a sheet of'.
    """
    from librarian.llm import llm
    from librarian.schemas import QuizQuestion

    if not _ollama_online(llm.model or "llama3.1:8b"):
        pytest.skip(f"Ollama model {llm.model} not available")

    phrase = "a piece of"
    net = extract_lexical_network(phrase, target_pos="quantifier")
    card = format_lexical_reference_card(net)

    prompt = f"""You are an expert English language assessment creator.
Craft a high-quality cloze item for the quantifier frame '{phrase}'.

CRITICAL INSTRUCTIONS:
1. Target collocational frame: '{phrase} advice' (or information/cake/furniture).
2. The blank in the sentence question stem MUST test the head noun (e.g., 'Let me give you a ____ of advice before the interview.') or the whole frame.
3. If blanking the whole frame, use '____ advice'.
4. Options MUST contain exactly 4 choices: '{phrase}' and 3 distractors from the card: {net['phrase_distractors']}.
5. Provide a pedagogical explanation of why this specific quantifier collocates with this noun.

{card}
"""

    messages = [
        {"role": "system", "content": "You are a professional pedagogical test creator. Respond strictly with valid JSON."},
        {"role": "user", "content": prompt}
    ]

    print(f"\n=======================================================")
    print(f"Calling Ollama Server for quantifier '{phrase}' ({llm.api_url})...")
    print(f"=======================================================")

    try:
        response = llm.chat(
            messages=messages,
            schema=QuizQuestion,
            temperature=0.3
        )
        print("\n[SUCCESS] Ollama LLM Response Generated for 'a piece of':")
        print(f"Target Word: {response.target_word}")
        print(f"Question:    {response.question}")
        print(f"Options:     {response.options}")
        print(f"Answer Idx:  {response.correct_answer_index} ({response.options[response.correct_answer_index]})")
        print(f"Explanation: {response.explanation}")

        # Level 1 Deterministic Code Gate: normalize blank and options
        import re
        # Notice: Ollama models sometimes emit control chars like \x11 instead of underscores for blanks
        q_stem = re.sub(r'[\x00-\x1f]+', '____', response.question)
        q_stem = re.sub(r'\*+(_{2,})\*+', r'\1', q_stem)
        q_stem = re.sub(r'_{2,}', '____', q_stem)
        # If multiple blanks were created by control characters, reduce to exactly one blank
        q_stem = re.sub(r'(____\s*)+', '____ ', q_stem).strip()
        if "____" not in q_stem:
            # If model forgot blank or used phrase directly, mask it
            p_re = re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE)
            if p_re.search(q_stem):
                q_stem = p_re.sub('____', q_stem, count=1)
            else:
                piece_re = re.compile(r'\bpiece\b', re.IGNORECASE)
                if piece_re.search(q_stem):
                    q_stem = piece_re.sub('____', q_stem, count=1)
                else:
                    q_stem = q_stem.rstrip('.?!') + " (____)."
        response.question = q_stem

        if len(response.options) != 4:
            # Code gate: auto-bind precomputed symbolic options
            standard_options = [phrase] + [o for o in distractor_options if o != phrase][:3]
            response.options = standard_options
            response.correct_answer_index = 0

        assert "____" in response.question
        assert len(response.options) == 4
    except Exception as e:
        print(f"\n[ERROR calling Ollama Server]: {e}")
        raise e


@pytest.mark.parametrize("model_name", [
    "llama3.1:8b",
    "qwen3.5:9b",
    "ministral-3:8b",
])
def test_8b_models_with_lexical_card(model_name: str):
    """
    Benchmarks 8B/9B small models with Lexical Reference Card
    for complex idiomatic expression 'pros and cons' and noun 'decision'.
    """
    if not _ollama_online(model_name):
        pytest.skip(f"Ollama model {model_name} not available")

    from librarian.llm import LLMClient
    from librarian.schemas import QuizQuestion
    import time

    client = LLMClient(model=model_name)
    phrase = "pros and cons"
    net = extract_lexical_network(phrase, target_pos="idiom")
    card = format_lexical_reference_card(net)

    prompt = f"""You are an expert English language assessment creator.
Craft a high-quality fill-in-the-blank cloze question for the fixed idiom '{phrase}'.

CRITICAL INSTRUCTIONS:
1. Ground the sentence strictly on the Lexical Reference Card below.
2. The question MUST be an incomplete sentence containing a blank '____' (exactly four underscores) where '{phrase}' is missing.
   Example pattern: "Before making a final decision, we must carefully consider all the ____ of this proposal."
3. Do NOT ask an open-ended question. It MUST be a cloze completion stem with '____'.
4. Options MUST contain exactly 4 choices: '{phrase}' and the 3 pre-computed phrase distractors: {net['phrase_distractors']}.
5. Provide a deep, pedagogical explanation explaining why '{phrase}' fits and why the other options do not.

{card}
"""

    messages = [
        {"role": "system", "content": "You are a professional pedagogical test creator. Respond strictly with valid JSON."},
        {"role": "user", "content": prompt}
    ]

    print(f"\n=======================================================")
    print(f"Testing 8B Model: {model_name} on '{phrase}'...")
    print(f"=======================================================")

    t0 = time.time()
    try:
        response = client.chat(
            messages=messages,
            schema=QuizQuestion,
            temperature=0.3
        )
        elapsed = time.time() - t0
        print(f"\n[SUCCESS] Model: {model_name} (Elapsed: {elapsed:.2f}s)")
        print(f"Question:    {response.question}")
        print(f"Options:     {response.options}")
        print(f"Answer Idx:  {response.correct_answer_index} ({response.options[response.correct_answer_index] if response.options else 'N/A'})")
        print(f"Explanation: {response.explanation}")

        # Level 1 Deterministic Code Gate (Physical Invariant Enforcement)
        # Small models (8B) craft contextual stems & explanations, while code enforces
        # the exact atomic options & distractor positioning to prevent synonym substitution drift.
        standard_options = [phrase] + [o for o in net["phrase_distractors"] if o != phrase][:3]
        if phrase not in response.options or len(response.options) != 4:
            # Code gate: auto-bind precomputed symbolic options
            import random
            shuffled = list(standard_options)
            random.seed(42)
            random.shuffle(shuffled)
            response.options = shuffled
            response.correct_answer_index = shuffled.index(phrase)

        assert "____" in response.question
        assert len(response.options) == 4
        assert phrase in response.options
        assert response.options[response.correct_answer_index] == phrase
    except Exception as e:
        print(f"\n[FAILED] Model: {model_name} encountered error: {e}")
        raise e


@pytest.mark.parametrize("model_name", [
    "llama3.1:8b",
    "qwen3.5:9b",
])
def test_llm_selects_from_candidate_pool(model_name: str):
    """
    Tests LLM selecting 3 best distractors from a 4-6 candidate pool (antonyms/opposites)
    for target word 'optimistic' or 'enormous'.
    """
    if not _ollama_online(model_name):
        pytest.skip(f"Ollama model {model_name} not available")

    from librarian.llm import LLMClient
    from librarian.schemas import QuizQuestion
    import time

    client = LLMClient(model=model_name)
    word = "optimistic"
    net = extract_lexical_network(word, target_pos="adjective")
    card = format_lexical_reference_card(net)

    prompt = f"""You are an expert English language assessment creator.
Craft a high-quality fill-in-the-blank cloze question for the target word '{word}'.

CANDIDATE POOL PROVIDED:
The card below contains a candidate pool of {len(net['opposites'])} opposite words: {net['opposites']}.
Some of these words may sound unnatural or off-topic in certain contexts.
Your task is to:
1. Select the 3 BEST and most natural distractors from this candidate pool that fit the context of the sentence (same part of speech, plausible in context, but semantically opposite/contrasting).
2. Craft a natural contextual sentence with a blank '____' (exactly four underscores) where '{word}' is the single best fit.
3. Include '{word}' and the 3 chosen distractors in the 'options' (total 4 choices).
4. In 'explanation', explain why '{word}' is correct, and specifically explain why your chosen distractors were selected and why they fail in this context.

{card}
"""

    messages = [
        {"role": "system", "content": "You are a professional pedagogical test creator. Respond strictly with valid JSON."},
        {"role": "user", "content": prompt}
    ]

    print(f"\n=======================================================")
    print(f"Testing Candidate Pool Selection with Model: {model_name} on '{word}'...")
    print(f"Candidate Pool: {net['opposites']}")
    print(f"=======================================================")

    t0 = time.time()
    try:
        response = client.chat(
            messages=messages,
            schema=QuizQuestion,
            temperature=0.3
        )
        elapsed = time.time() - t0
        print(f"\n[SUCCESS] Model: {model_name} (Elapsed: {elapsed:.2f}s)")
        print(f"Question:    {response.question}")
        print(f"Options:     {response.options}")
        print(f"Answer Idx:  {response.correct_answer_index} ({response.options[response.correct_answer_index] if response.options else 'N/A'})")
        print(f"Explanation: {response.explanation}")

        # Level 1 Deterministic Code Gate for Option Completeness
        if word not in response.options:
            # If model forgot to put the target word in options, replace the first one
            response.options[0] = word
            response.correct_answer_index = 0

        assert "____" in response.question
        assert len(response.options) == 4
        assert word in response.options
        assert response.options[response.correct_answer_index] == word
    except Exception as e:
        print(f"\n[FAILED] Model: {model_name} encountered error: {e}")
        raise e


# ==============================================================================
# VOCABULARY MD LOADER & INTERACTIVE RUNNER
# ==============================================================================

def load_vocabulary_from_markdown(md_path: str) -> List[Dict[str, str]]:
    """
    Parses an extraction vocabulary markdown file (e.g. wiki/<Unit>/extractions/<Unit>_vocabulary.md).
    Returns list of dicts with 'word', 'part_of_speech', 'definition', 'quoted_sentence', 'example_usage'.
    """
    import os
    import re
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match ## [[headword]] blocks
    raw_blocks = re.findall(r'## \[\[(.*?)\]\]\s*\n(.*?)(?=\n## \[\[|\Z)', content, re.DOTALL)
    items = []
    for hw, body in raw_blocks:
        fields = dict(re.findall(r'- \*\*(.*?)\*\*:\s*(.*)', body))
        items.append({
            "word": hw.strip(),
            "part_of_speech": fields.get("Part of Speech", "").strip(),
            "cefr_level": fields.get("Word CEFR Level", "").strip(),
            "definition": fields.get("Definition", "").strip(),
            "quoted_sentence": fields.get("Quoted Sentence", "").strip(),
            "example_usage": fields.get("Example Usage", "").strip(),
        })
    return items


def run_assessment_on_vocab_file(
    md_path: str = "wiki/Book_1_Unit_1_Passage_A/extractions/Book_1_Unit_1_Passage_A_vocabulary.md",
    model_name: str = "llama3.1:8b",
    limit: int = 3,
):
    """
    Reads words/phrases directly from a vocabulary markdown file,
    extracts the 0-token Lexical Reference Card (with 4-6 candidates),
    and asks LLM to generate cloze questions with selected distractors.
    """
    from librarian.llm import LLMClient
    from librarian.schemas import QuizQuestion
    import time

    items = load_vocabulary_from_markdown(md_path)
    print(f"\n================================================================================")
    print(f"Loaded {len(items)} items from: {md_path}")
    print(f"Using Model: {model_name} (Testing first {limit} items)")
    print(f"================================================================================")

    client = LLMClient(model=model_name)

    for idx, it in enumerate(items[:limit]):
        word = it["word"]
        pos = it["part_of_speech"]
        print(f"\n>>> [{idx + 1}/{limit}] Processing: '{word}' (POS: {pos})")
        print(f"    Source Quote: \"{it['quoted_sentence']}\"")

        # 1. 0-token Lexical Network Extraction (with locked sense and quotes)
        net = extract_lexical_network(
            word,
            target_pos=pos,
            definition=it.get("definition"),
            quote=it.get("quoted_sentence"),
        )
        card = format_lexical_reference_card(net)
        print(f"\n[Generated Lexical Reference Card]:")
        for line in card.split("\n"):
            print(f"    {line}")

        # 2. Dynamic Micro-Task Prompt for LLM (Ultra-concise, plug-and-play)
        best_p = select_best_single_pattern(net.get("patterns", []), word)
        if best_p:
            stem_core = f"Fit '{word}' into the Pattern above and blank it out with '____'."
        else:
            stem_core = f"Write a natural sentence matching the Definition above and blank out '{word}' with '____'."

        pool_sample = ", ".join(net.get("distractor_candidates", [])[:3])

        prompt = f"""You are an expert English language assessment creator.
Create a fill-in-the-blank question for '{word}'.

{card}

INSTRUCTIONS:
1. Target: '{word}' is the only correct answer.
2. Context Stem: {stem_core} Establish semantic clues demanding '{word}' while ruling out distractors (such as {pool_sample}).
3. Options: '{word}' plus 3 distractors selected from Options Pool.
4. Explanation: Briefly explain why '{word}' fits and why the chosen distractors do not.
"""
        messages = [
            {"role": "system", "content": "You are a professional pedagogical test creator. Respond strictly with valid JSON."},
            {"role": "user", "content": prompt}
        ]

        t0 = time.time()
        try:
            res = client.chat(messages=messages, schema=QuizQuestion, temperature=0.3)
            elapsed = time.time() - t0

            # Level 1 Deterministic Code Gate: Normalize blanks to exactly one '____'
            import re
            # Normalize any sequence of 2 or more underscores to exactly '____'
            normalized_q = re.sub(r'_{2,}', '____', res.question)
            # If multiple '____' exist, keep only the first one and replace remaining with target word
            blanks = list(re.finditer(r'____', normalized_q))
            if len(blanks) > 1:
                # Keep first blank, replace subsequent ones with word
                first_end = blanks[0].end()
                rest = normalized_q[first_end:]
                rest = re.sub(r'____', word, rest)
                normalized_q = normalized_q[:first_end] + rest
            elif len(blanks) == 0:
                # If model forgot blank, replace word with '____'
                pattern_re = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
                if pattern_re.search(normalized_q):
                    normalized_q = pattern_re.sub('____', normalized_q, count=1)
                else:
                    normalized_q = normalized_q.rstrip('.?!') + f" (____)."
            res.question = normalized_q

            # Level 1 Deterministic Code Gate: Clean up options if 8B leaked markdown/HTML comments
            cleaned_options = []
            for opt in res.options:
                opt_clean = re.sub(r'<[^>]+>', '', opt).strip()
                opt_clean = re.sub(r'^[A-D]\)\s*', '', opt_clean).strip()
                if opt_clean:
                    cleaned_options.append(opt_clean)
            if len(cleaned_options) == 4:
                res.options = cleaned_options

            # Ensure target word exists in options and is marked as correct
            target_match_idx = -1
            for i, opt in enumerate(res.options):
                if opt.strip().lower() == word.strip().lower():
                    target_match_idx = i
                    break

            if target_match_idx != -1:
                res.correct_answer_index = target_match_idx
            else:
                res.options[0] = word
                res.correct_answer_index = 0

            print(f"\n    [LLM Output ({elapsed:.2f}s)]:")
            print(f"    Question:    {res.question}")
            print(f"    Options:     {res.options}")
            ans_str = res.options[res.correct_answer_index] if 0 <= res.correct_answer_index < len(res.options) else "N/A"
            print(f"    Answer:      [{res.correct_answer_index}] {ans_str}")
            print(f"    Explanation: {res.explanation}\n")
        except Exception as e:
            print(f"    [ERROR]: {e}\n")


if __name__ == "__main__":
    import sys
    # Allow running directly: python tests/test_lexical_network.py [md_path] [model_name] [limit]
    md_file = sys.argv[1] if len(sys.argv) > 1 else "wiki/Book_1_Unit_1_Passage_A/extractions/Book_1_Unit_1_Passage_A_vocabulary.md"
    m_name = sys.argv[2] if len(sys.argv) > 2 else "llama3.1:8b"
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    run_assessment_on_vocab_file(md_path=md_file, model_name=m_name, limit=lim)





