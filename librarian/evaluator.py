import re
import json
import unicodedata
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Any, Optional, Set, Tuple, get_args

from .config import config
from .schemas import PARTS_OF_SPEECH, EXPRESSION_TYPES
from .linguistics import LinguisticEngine

# --- Dimension weights (max points per dimension). Single source of truth. ---
W_SCHEMA = 25.0
W_VERBATIM = 30.0
W_PEDAGOGY = 25.0
W_UNIQUENESS = 20.0

# Valid Part of Speech enum set derived directly from schemas.py
VALID_POS_SET = set(get_args(PARTS_OF_SPEECH))
VALID_EXPR_POS_SET = set(get_args(EXPRESSION_TYPES))

# Top-level arrays that may hold gradeable items, in priority order.
ITEM_KEYS = ["vocabulary", "expressions", "grammar_patterns", "questions", "concepts", "branches"]

# Module-level stop-slots set for verbatim checking (avoids re-allocation on every item)
STOP_SLOTS = frozenset({
    "something", "somebody", "ones", "someone", "sb", "sth", "entity",
    "field", "area", "role", "object", "domain", "type", "situation", "goal"
})

# Grammar pattern slot whitelist for bracketed formulas (case-insensitive)
ALLOWED_GRAMMAR_SLOTS = frozenset({
    "s", "np", "vp", "v", "be", "aux", "v-ed", "v3", "v-ing", "to-v", "adj", "adv",
    "det", "prep", "conj", "clause", "noun phrase", "verb phrase", "predicate", "subject",
    "object", "complement", "adverbial", "focus element", "focal element", "subordinate clause",
    "main clause", "dependent clause", "modal", "modal/vp", "copula", "past participle",
    "present participle", "infinitive", "gerund", "adjective phrase", "adverb phrase",
    "prepositional phrase", "quotation", "wh-word", "wh-clause", "head noun",
    "verb-ing", "verb-ed", "prepp/np", "adj/np", "comparative", "proposition clause", "also"
})

# Standard COBUILD bare POS tokens for unbracketed pattern formulas
COBUILD_POS_TOKENS = frozenset({
    "np", "vp", "v", "be", "aux", "v-ed", "v3", "v-ing", "to-v", "adj", "adv",
    "det", "prep", "conj", "clause", "s", "n", "pron", "modal"
})
# Canonical identities for COBUILD slot labels. Two labels that name the same
# constituent are interchangeable inside a formula ('[NP]' == '[Noun Phrase]'),
# which is what makes a formula comparison casing- and wording-insensitive.
GRAMMAR_SLOT_IDENTITIES = {
    "np": "np", "n": "np", "noun": "np", "noun phrase": "np", "n phrase": "np",
    "vp": "vp", "verb phrase": "vp", "predicate": "vp",
    "s": "s", "subject": "s",
    "v": "v", "verb": "v",
    "v-ing": "v-ing", "verb-ing": "v-ing", "v ing": "v-ing", "participle": "v-ing",
    "v-ing phrase": "v-ing", "participial phrase": "v-ing",
    "v-ed": "v-ed", "verb-ed": "v-ed", "v ed": "v-ed", "v3": "v-ed",
    "past participle": "v-ed", "v-ed phrase": "v-ed",
    "adj": "adj", "adjective": "adj", "adj phrase": "adj",
    "adv": "adv", "adverb": "adv", "adv phrase": "adv",
    "prep": "prep", "prepp": "prep", "preposition": "prep",
    "det": "det", "determiner": "det",
    "conj": "conj", "conjunction": "conj",
    "modal": "modal", "modal verb": "modal", "modal vp": "modal",
    "clause": "clause", "main clause": "clause", "subordinate clause": "clause",
    "to-v": "to-v", "to verb": "to-v", "infinitive": "to-v",
    "wh": "wh", "wh word": "wh", "relative pronoun": "wh",
    "o": "o", "object": "o",
}


def _formula_slot_identity(slot: str) -> str:
    """Canonical identity of one slot label ('Noun Phrase' -> 'np')."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(slot).lower()).strip()
    return GRAMMAR_SLOT_IDENTITIES.get(cleaned, cleaned)


def canonical_formula_key(formula: str) -> str:
    """Casing-, spacing- and separator-insensitive identity of a COBUILD formula.

    'It + [be] + [NP] + that + [S]' and 'it be [Noun Phrase] that [Subject]' share
    one key, so a formula that differs only in presentation is not a deviation.
    """
    if not formula:
        return ""
    keyed = re.sub(
        r"\[([^\]]*)\]",
        lambda m: f"[{_formula_slot_identity(m.group(1))}]",
        str(formula).lower(),
    )
    keyed = keyed.replace("+", " ")
    keyed = re.sub(r"\s*,\s*", ",", keyed)
    keyed = re.sub(r"[^a-z0-9()\[\]/']+", " ", keyed)
    keyed = re.sub(r"\s+", " ", keyed)
    return keyed.strip().strip(".")


def degenerate_grammar_slots(formula: str) -> List[str]:
    """Slots that offer two names for the SAME constituent: '[NP/NP]', 'v/v'.

    The lexicon never mines such a slot: it teaches nothing, and it is the
    signature of a formula that was rewritten instead of copied.
    """
    offenders: List[str] = []
    text = str(formula or "")
    for slot in re.findall(r"\[([^\]]+)\]", text):
        parts = [p for p in slot.split("/") if p.strip()]
        if len(parts) >= 2 and len({_formula_slot_identity(p) for p in parts}) == 1:
            offenders.append(slot.strip())
    for token in re.sub(r"\[[^\]]*\]", " ", text).split():
        parts = [p for p in token.split("/") if p.strip()]
        if len(parts) >= 2 and len({_formula_slot_identity(p) for p in parts}) == 1:
            offenders.append(token.strip())
    return offenders



# Fatal QA flags that invalidate pedagogical delivery and trigger quarantine / retry
FATAL_QA_FLAGS = (
    "does not appear in quoted sentence",
    "Non-verbatim quote detected",
    "duplicate",
    "copy-pasted definition",
    "Selection clustering",
    "Redundant headwords",
    "Overly basic general-English",
    "Grammar formula anchor",
    "Hallucinated quote",
    "pure fabrication",
    "Incomplete target coverage",
    "ungrounded in quote",
    "not a recognized English word",
    "Multiple blanks",
    "Target word leaks",
    "missing fill-in-the-blank slot",
    "indefinite article leakage",
    "cross-target leakage",
    "Stem verbatim from dictionary example",
    "Stem verbatim from curriculum quote",
    "Inflection discordance",
    # F10 rule 0: these are no longer fatal Level-1 flags. 'blank slot POS mismatch',
    # 'Anchor missing in question stem' and 'Explanation anchor not grounded' all require
    # reading a sentence, so they belong to the Level-2 expert audit; 'Distractor slot
    # illegality' is a dictionary comparison that stays reported but no longer caps the score.
    # Backlog D1: skeleton-to-output alignment defects
    "Prescribed options altered",
    "Prescribed answer index altered",
    "degenerate formula slot",
    "deviates from the pre-extracted skeleton",
    "duplicates the source quote",

)

# Module-level irregular verbs mapping (avoids re-allocation on every item)
COMMON_IRREGULARS = {
    'lead': ('led',), 'led': ('lead',),
    'take': ('took', 'taken'), 'took': ('take',), 'taken': ('take',),
    'bring': ('brought',), 'brought': ('bring',),
    'lay': ('laid',), 'laid': ('lay',),
    'make': ('made',), 'made': ('make',),
    'give': ('gave', 'given'), 'gave': ('give',), 'given': ('give',),
    'come': ('came',), 'came': ('come',),
    'go': ('went', 'gone'), 'went': ('go',), 'gone': ('go',),
    'keep': ('kept',), 'kept': ('keep',),
    'hold': ('held',), 'held': ('hold',),
    'find': ('found',), 'found': ('find',),
    'break': ('broke', 'broken'), 'broke': ('break',), 'broken': ('break',),
    'choose': ('chose', 'chosen'), 'chose': ('choose',), 'chosen': ('choose',),
    'run': ('ran',), 'ran': ('run',),
    'see': ('saw', 'seen'), 'saw': ('see',), 'seen': ('see',),
    'speak': ('spoke', 'spoken'), 'spoke': ('speak',), 'spoken': ('speak',),
    'write': ('wrote', 'written'), 'wrote': ('write',), 'written': ('write',),
    'build': ('built',), 'built': ('build',),
    'lose': ('lost',), 'lost': ('lose',),
    'pay': ('paid',), 'paid': ('pay',),
    'say': ('said',), 'said': ('say',),
    'send': ('sent',), 'sent': ('send',),
    'spend': ('spent',), 'spent': ('spend',),
    'stand': ('stood',), 'stood': ('stand',),
    'tell': ('told',), 'told': ('tell',),
    'think': ('thought',), 'thought': ('think',),
    'understand': ('understood',), 'understood': ('understand',),
    'win': ('won',), 'won': ('win',),
    'catch': ('caught',), 'caught': ('catch',),
    'draw': ('drew', 'drawn'), 'drew': ('draw',), 'drawn': ('draw',),
    'grow': ('grew', 'grown'), 'grew': ('grow',), 'grown': ('grow',),
    'hear': ('heard',), 'heard': ('hear',),
    'hide': ('hid', 'hidden'), 'hid': ('hide',), 'hidden': ('hide',),
    'know': ('knew', 'known'), 'knew': ('know',), 'known': ('know',),
    'leave': ('left',), 'left': ('leave',),
    'meet': ('met',), 'met': ('meet',),
    'read': ('read',),
    'rise': ('rose', 'risen'), 'rose': ('rise',), 'risen': ('rise',),
    'wear': ('wore', 'worn'), 'wore': ('wear',), 'worn': ('wear',),
    'drive': ('drove', 'driven'), 'drove': ('drive',), 'driven': ('drive',),
    'fall': ('fell', 'fallen'), 'fell': ('fall',), 'fallen': ('fall',),
    'feel': ('felt',), 'felt': ('feel',),
}


def _form_in_text(tok: str, text: str) -> bool:
    """
    Backlog A1: the single word-form matcher behind every quote-evidence check.

    Delegates to LinguisticEngine so an inflected occurrence counts ('attaches' for the
    headword 'attach'), then keeps the historical stem-prefix heuristic - which also
    accepts derivationally related forms ('degradation' for 'degrade') that no inflection
    table covers - and the closed irregular-verb table.
    """
    if not tok or not text:
        return False
    if LinguisticEngine.text_contains_form(text, tok):
        return True
    if tok in text or (len(tok) >= 4 and tok[:4] in text):
        return True
    if tok in COMMON_IRREGULARS and any(ir in text for ir in COMMON_IRREGULARS[tok]):
        return True
    return False

# Common English contractions for contraction-aware anchor matching
CONTRACTIONS_MAP = {
    "it's": "it is", "it’s": "it is", "its": "it is",  # in context of contracted copula
    "that's": "that is", "that’s": "that is",
    "there's": "there is", "there’s": "there is",
    "what's": "what is", "what’s": "what is",
    "here's": "here is", "here’s": "here is",
    "he's": "he is", "he’s": "he is",
    "she's": "she is", "she’s": "she is",
    "who's": "who is", "who’s": "who is",
    "i'm": "i am", "i’m": "i am",
    "you're": "you are", "you’re": "you are",
    "we're": "we are", "we’re": "we are",
    "they're": "they are", "they’re": "they are",
    "can't": "cannot", "can’t": "cannot",
    "won't": "will not", "won’t": "will not",
    "don't": "do not", "don’t": "do not",
    "doesn't": "does not", "doesn’t": "does not",
    "didn't": "did not", "didn’t": "did not",
    "isn't": "is not", "isn’t": "is not",
    "aren't": "are not", "aren’t": "are not",
    "wasn't": "was not", "wasn’t": "was not",
    "weren't": "were not", "weren’t": "were not",
    "haven't": "have not", "haven’t": "have not",
    "hasn't": "has not", "hasn’t": "has not",
    "hadn't": "had not", "hadn’t": "had not",
    "wouldn't": "would not", "wouldn’t": "would not",
    "shouldn't": "should not", "shouldn’t": "should not",
    "couldn't": "could not", "couldn’t": "could not",
    "mustn't": "must not", "mustn’t": "must not",
    "i've": "i have", "i’ve": "i have",
    "you've": "you have", "you’ve": "you have",
    "we've": "we have", "we’ve": "we have",
    "they've": "they have", "they’ve": "they have",
    "i'll": "i will", "i’ll": "i will",
    "you'll": "you will", "you’ll": "you will",
    "he'll": "he will", "he’ll": "he will",
    "she'll": "she will", "she’ll": "she will",
    "we'll": "we will", "we’ll": "we will",
    "they'll": "they will", "they’ll": "they will",
    "i'd": "i would", "i’d": "i would",
    "you'd": "you would", "you’d": "you would",
    "he'd": "he would", "he’d": "he would",
    "she'd": "she would", "she’d": "she would",
    "we'd": "we would", "we’d": "we would",
    "they'd": "they would", "they’d": "they would",
}

_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTIONS_MAP.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE
)

def _expand_contractions(text: str) -> str:
    """Expands English contractions to full words for robust anchor matching."""
    if not text:
        return ""
    def _repl(m: re.Match) -> str:
        tok = m.group(1).lower()
        return CONTRACTIONS_MAP.get(tok, tok)
    return _CONTRACTION_RE.sub(_repl, text)

def auto_remap_grammar_category(item: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Deterministic Level 1 Code Gate: Auto-Remap Grammar Category.
    Combines spaCy computational dependency parsing with physical marker regexes.
    If a quote has structural markers pointing to a specific macro domain,
    but the LLM mislabeled it, automatically remap the category in place
    and return a diagnostic notice.
    """
    if not isinstance(item, dict):
        return item, None
    quote = str(item.get("quote", "")).strip()
    category = str(item.get("category", "")).strip()
    if not quote or not category:
        return item, None

    # Only auto-remap if the category is claimed to be one of the 4 Macro Domains
    # If the item uses a legacy fine-grained label (e.g. 'Concessive clauses'), let it pass without remap
    FOUR_DOMAINS = {"Rhetoric & Emphasis", "Cohesion & Framing", "Information Packaging", "Logic & Stance"}
    if category not in FOUR_DOMAINS:
        return item, None

    INTERPRETIVE_VERBS_REGEX = (
        r"(?:(?:would|could|might|may|can|will)\s+)?"
        r"(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?)"
    )

    # Step 1: Query the spaCy Computational Dependency Classification
    dep_category = LinguisticEngine.classify_grammar_dependency(quote)
    if dep_category and dep_category != category:
        orig_cat = category
        item["category"] = dep_category
        # Detect descriptive marker for diagnostic transparency
        marker = "structural dependency pattern"
        if re.search(r"\bnot\s+.*?\s*,\s*but\b", quote, re.IGNORECASE):
            marker = "antithesis 'not... but...'"
        elif re.search(rf",\s*which\s+{INTERPRETIVE_VERBS_REGEX}\s+that\b", quote, re.IGNORECASE):
            marker = "propositional encapsulation ', which + [interpretive verb] + that'"
        notice = f"ℹ️ Deterministic Auto-Remap (spaCy): Corrected category from '{orig_cat}' to '{dep_category}' (Marker: {marker})"
        return item, notice

    # Step 2: Fallback to fast-path regex checks for patterns spaCy tree might span across fragments
    # 1. Cohesion & Framing ironclad markers:
    has_which_propositional = bool(re.search(rf",\s*which\s+{INTERPRETIVE_VERBS_REGEX}\s+that\b", quote, re.IGNORECASE))
    has_shell_noun_frame = bool(re.search(r"\bthe\s+(?:fact|idea|notion|reason|belief|claim|argument|possibility|question|view|conclusion)\s+that\b", quote, re.IGNORECASE))
    if (has_which_propositional or has_shell_noun_frame) and category != "Cohesion & Framing":
        orig_cat = category
        item["category"] = "Cohesion & Framing"
        marker_name = "propositional encapsulator ', which + [interpretive verb] + that'" if has_which_propositional else "shell noun complement frame"
        notice = f"ℹ️ Deterministic Auto-Remap: Corrected category from '{orig_cat}' to 'Cohesion & Framing' (Marker: {marker_name})"
        return item, notice

    # 2. Rhetoric & Emphasis ironclad markers:
    has_antithesis = bool(re.search(r"\bnot\s+.*?\s*,\s*but\b", quote, re.IGNORECASE))
    has_correlative = bool(re.search(r"\b(?:not\s+only\b.*?\bbut\s+also|either\b.*?\bor\b|neither\b.*?\bnor\b)\b", quote, re.IGNORECASE))
    has_fronted_inversion = bool(re.search(r"^\s*(?:not\s+only|only\s+(?:if|when|after|by)|never|seldom|hardly|scarcely)\b", quote, re.IGNORECASE))
    has_cleft = bool(
        re.search(r"\bIt\s+(?:is|was|were|'s)\s+(?:only|not\s+only|because|when|in|on|at|by|with|[a-z]{3,}\s+who|[a-z]{3,}\s+that)\b", quote, re.IGNORECASE)
        and not re.search(r"\bIt\s+(?:is|was|were|'s)\s+(?:said|thought|believed|reported|expected|suggested|indicated|argued|claimed|known|important|essential|necessary|likely|clear|obvious|vital|crucial|apparent|natural|possible)\s+that\b", quote, re.IGNORECASE)
    )
    
    if (has_antithesis or has_correlative or has_fronted_inversion or has_cleft) and category != "Rhetoric & Emphasis":
        orig_cat = category
        item["category"] = "Rhetoric & Emphasis"
        marker_name = "antithesis 'not... but...'" if has_antithesis else ("correlative coordination" if has_correlative else ("fronted inversion" if has_fronted_inversion else "structural cleft"))
        notice = f"ℹ️ Deterministic Auto-Remap: Corrected category from '{orig_cat}' to 'Rhetoric & Emphasis' (Marker: {marker_name})"
        return item, notice

    # 3. Information Packaging ironclad markers:
    has_evaluative_it = bool(re.search(r"\bIt\s+(?:is|was|were|'s|has\s+been)\s+(?:[a-z]{4,}\s+)?(?:important|essential|necessary|likely|clear|obvious|vital|crucial|apparent|natural|possible|hard|easy|difficult|wise|useful)\s+(?:that|to\s+[a-z]+)\b", quote, re.IGNORECASE))
    has_dummy_it_obj = bool(re.search(r"\b(?:find|found|make|made|think|thought|consider|considered|deem|deemed)\s+it\s+(?:difficult|hard|easy|possible|impossible|necessary|vital|crucial|wise|useful)\s+to\b", quote, re.IGNORECASE))
    if (has_evaluative_it or has_dummy_it_obj) and category in ("Rhetoric & Emphasis", "Cohesion & Framing"):
        orig_cat = category
        item["category"] = "Information Packaging"
        notice = f"ℹ️ Deterministic Auto-Remap: Corrected category from '{orig_cat}' to 'Information Packaging' (Marker: Evaluative Dummy-It extraposition)"
        return item, notice

    # Ordinary elaborative relative clause (, which + VP without that) mislabeled as Cohesion & Framing
    has_ordinary_which = bool(re.search(r",\s*which\s+[a-z]+", quote, re.IGNORECASE) and not re.search(rf",\s*which\s+{INTERPRETIVE_VERBS_REGEX}\b", quote, re.IGNORECASE))
    if has_ordinary_which and category == "Cohesion & Framing":
        orig_cat = category
        item["category"] = "Information Packaging"
        notice = f"ℹ️ Deterministic Auto-Remap: Corrected category from '{orig_cat}' to 'Information Packaging' (Marker: Elaborative non-restrictive relative clause)"
        return item, notice

    return item, None


def _safe_str(val: Any, default: str = "") -> str:
    """Safely converts a value to string, mapping None to default rather than 'None'."""
    if val is None:
        return default
    return str(val).strip()


def _normalize_text(value: Any) -> str:
    """Normalize unicode, strip markdown formatting, brackets, ellipses, quotes, and whitespace."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).lower()
    # Remove markdown bold/italics/code/strikethrough markers
    text = re.sub(r"[\*\_`~]+", " ", text)
    # Remove ellipses and dots
    text = re.sub(r"(\.{2,}|…)", " ", text)
    # Remove quotes and apostrophes directly to keep words intact
    text = re.sub(r"[“”‘’\"'«»]+", "", text)
    # Strip injected sentence markers like [S-12] or S-12 to prevent verbatim mismatch
    text = re.sub(r"\[?\bS-\d+\b\]?[:\-]?", " ", text, flags=re.IGNORECASE)
    # Remove structural brackets/parentheses with whitespace separation
    text = re.sub(r"[\[\]\(\)\{\}\<\>]+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_core(text: str) -> str:
    """Strip all punctuation and non-alphanumeric chars (retaining spaces)."""
    norm = _normalize_text(text)
    return re.sub(r"[^\w\s]", "", norm).strip()


def _batch_target_conflict(word: str, other_headword: str) -> bool:
    """
    True if `word` IS, or is an inflected/derived form of, `other_headword`.
    Mirrors exactly the stem-stripping used by the Cross-Target Stem Leakage Gate so
    that the anchor-presence gate can never demand a word the leakage gate bans.
    """
    a = _clean_core(word)
    b = _clean_core(other_headword)
    if not a or not b:
        return False
    if a == b:
        return True
    _strip = r"(?:ed|ing|s|es|ly|tion|ment)$"
    stem_a = re.sub(_strip, "", a)
    stem_b = re.sub(_strip, "", b)
    return len(stem_a) >= 4 and len(stem_b) >= 4 and stem_a == stem_b


def _ngram_coverage(quote_clean: str, source_clean: str, n: int = 3) -> float:
    """Calculates n-gram overlap between cleaned quote and source text."""
    q_words = quote_clean.split()
    s_words = source_clean.split()
    if not q_words or not s_words:
        return 0.0
    if len(q_words) < n:
        if quote_clean in source_clean:
            return 1.0
        s_set = set(s_words)
        return sum(1 for w in q_words if w in s_set) / len(q_words)
    q_ngrams = set(tuple(q_words[i:i + n]) for i in range(len(q_words) - n + 1))
    s_ngrams = set(tuple(s_words[i:i + n]) for i in range(len(s_words) - n + 1))
    if not q_ngrams:
        return 0.0
    return len(q_ngrams.intersection(s_ngrams)) / len(q_ngrams)


_VERB_SUFFIXES = ("ies", "ied", "ing", "es", "ed", "s", "d")


def _inflected_variants_of(word: str) -> set:
    """Surface forms a target can plausibly take, without a spaCy round-trip per token."""
    w = re.sub(r"[^a-z]", "", (word or "").lower())
    forms = {w}
    if not w:
        return forms
    for suffix in ("s", "es", "d", "ed", "ing"):
        forms.add(w + suffix)
    if w.endswith("y") and len(w) > 2:
        forms.update({w[:-1] + "ies", w[:-1] + "ied"})
    if w.endswith("e") and len(w) > 3:
        forms.update({w[:-1] + "ing", w[:-1] + "s"})
    for suffix in _VERB_SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            base = w[: -len(suffix)]
            forms.update({base, base + "s", base + "es", base + "d", base + "ed", base + "ing"})
    return {f for f in forms if len(f) >= 3}


def _source_token_variants(source: str, target: str) -> List[List[str]]:
    """A corpus source, plus the same source with the target word blanked out.

    The most common writer shortcut is to take the blueprint's own example and drop the
    target into a blank. That hole breaks the source's 5-grams, so a short example (every
    5-gram of it contains the target) slips past a plain n-gram test. Comparing the stem
    against the masked source as well closes that hole.
    """
    tokens = re.sub(r"[^\w\s]", " ", (source or "").lower()).split()
    if not tokens:
        return []
    variants = [tokens]
    forms = _inflected_variants_of(target)
    kept = [t for t in tokens if t not in forms]
    if len(kept) != len(tokens):
        variants.append(kept)
    return variants


def _is_hallucinated_quote(quote: str) -> bool:
    """Detect if model explicitly notes quote is inferred or missing from text,
    or if it copied a pattern template (containing syntactic slot brackets like [S], [NP], [to-V]) into quote.
    """
    q_lower = quote.lower()
    indicators = [
        "not present in text",
        "not in text",
        "not found in text",
        "not in source",
        "inferred from",
        "implied by",
        "constructed from",
        "not explicitly mentioned",
    ]
    if any(ind in q_lower for ind in indicators):
        return True
    # If quote contains grammatical slot placeholders like [S], [NP], [to-V], [VP], it's a copied template, not an authentic passage quote
    if re.search(r"\[(S|NP|VP|V|to-V|V3|V-ing|adj|adv|be|aux|modal)\]", quote, re.IGNORECASE):
        return True
    return False


def _extract_source_content(user_prompt: str) -> str:
    """Extracts isolated source text from user prompt (under ### SOURCE TEXT ###, CONTENT:, # Source Material, etc.)
    Returns empty string if no authentic content block is found to prevent instruction text from
    being falsely matched as verbatim source.
    Also strips any appended QA review / retry critique blocks so error reports are never treated as source material.
    """
    if not user_prompt:
        return ""
    # 1. Primary: Standard ### PASSAGE..., ### SOURCE TEXT ### or CONTENT: marker
    match = re.search(r"(?:###\s*(?:PASSAGE[^\n#]*|SOURCE\s*TEXT)\s*###|CONTENT:)\s*\n(.*)", user_prompt, re.DOTALL | re.IGNORECASE)
    content = ""
    if match:
        content = match.group(1).strip()
    else:
        # 2. Secondary: Other standard headings if primary markers were omitted
        match_sec = re.search(r"(?:#+\s*(?:Source Material|Input Content|Text Context|Transcript))\s*\n(.*)", user_prompt, re.DOTALL | re.IGNORECASE)
        if match_sec:
            content = match_sec.group(1).strip()
        else:
            # 3. Tertiary: Raw markdown source with optional YAML frontmatter and section heading
            match_md = re.search(r"(?:^|\n)(?:---\s*\n.*?\n---\s*\n)?(?:##\s+[^\n]+\n+)(.*)", user_prompt, re.DOTALL)
            if match_md:
                content = match_md.group(1).strip()
    
    if content:
        # Strictly strip any trailing task blocks, target lists, skeletons, draft blocks, or retry critique blocks
        # This prevents target vocabulary/expression/grammar instruction blocks from contaminating authentic source text.
        content = re.split(
            r"\n\s*###+\s*(?:TARGET\s+VOCABULARY|TARGET\s+EXPRESSIONS|TARGET\s+GRAMMAR|DETERMINISTIC\s+TARGET|🚨|\[QUALITY AUDIT REVIEW|VOCABULARY DRAFT|GRAMMAR PATTERNS DRAFT|QUIZ DRAFT|DRAFT)",
            content,
            flags=re.IGNORECASE
        )[0].strip()

    return content


# Backlog C1: a passage sets ONE difficulty ceiling, and every support sentence the
# student is asked to read - the 'example_usage' of a vocabulary or expression item,
# the 'imitation_example' of a grammar pattern - has to respect it. Before this, each
# quiz audit in processor.py wrote its own ladder and the grammar / vocabulary
# extraction paths had none at all.
MIN_SOURCE_WORDS_FOR_CEILING = 40


def _cefr_source_ceiling(user_prompt: str) -> Tuple[Optional[str], Set[str]]:
    """(CEFR level of the authentic passage, the words that passage already uses).

    Returns (None, set()) when the prompt carries no real passage: instruction text,
    target lists and retry critiques must never be measured as source difficulty.
    """
    source = _extract_source_content(user_prompt)
    words = re.findall(r"[a-zA-Z]+", source.lower())
    if len(words) < MIN_SOURCE_WORDS_FOR_CEILING:
        return None, set()
    return LinguisticEngine.calculate_text_cefr(source), {w for w in words if len(w) >= 4}


def _extract_wordlist(user_prompt: str) -> List[str]:
    """Extract the supplied vocabulary headwords from the content block.

    Prefers Obsidian heading entries of the form `## [[word]]`; falls back to any
    `## heading` so that differently-formatted word lists still work.
    """
    source = _extract_source_content(user_prompt)
    if not source:
        return []
    # Match ## [[...]] where the inner content might contain nested slot brackets like [[pay [somebody] attention]]
    words = re.findall(r"^##\s*\[\[(.*?)(?:\]\]\s*$)", source, re.MULTILINE)
    if not words:
        words = re.findall(r"^##\s*\[\[([^\]]+)\]\]", source, re.MULTILINE)
    if not words:
        # Fallback to headings, but strictly filter out structural/enumerated headings
        # like 'Item 1', 'Question 2', 'Vocabulary List', 'Text A', etc.
        raw_headings = re.findall(r"^##\s+(.+?)\s*$", source, re.MULTILINE)
        filtered = []
        for h in raw_headings:
            h_clean = h.strip()
            # Skip structural markers and item numbers
            if re.match(r"^(?:Item|Question|Task|Section|Part|Unit|Text|Passage)\s+\d+", h_clean, re.IGNORECASE):
                continue
            if re.match(r"^(?:Vocabulary\s*List|Comprehension|Assessment|Questions|Overview)", h_clean, re.IGNORECASE):
                continue
            filtered.append(h_clean)
        words = filtered
    return [w.strip() for w in words if w and w.strip()]


def _extract_target_skeletons(user_prompt: str, task_type: str) -> List[Dict[str, str]]:
    """Extract deterministic target skeletons supplied in user_prompt for coverage checking."""
    if not user_prompt:
        return []
    skeletons = []
    norm_prompt = re.sub(r'\r\n|\r', '\n', user_prompt)
    if task_type == "grammar":
        # Matches: 1. [S-8] (Logic & Stance) Formula: `...` or 1. [S-8] Formula: `...`
        m = re.findall(r'(\d+)\.\s*\[([^\]]+)\](?:\s*\(([^)]+)\))?\s*Formula:\s*`([^`]+)`', norm_prompt)
        for num, sid, cat, formula in m:
            skeletons.append({"sid": sid.strip(), "category": (cat or "Academic Syntax").strip(), "formula": formula.strip()})
    elif task_type == "expressions":
        # Matches: 1. [S-10] (phrasal verb) Formula: `...` or 1. [S-10] Formula: `...`
        m = re.findall(r'(\d+)\.\s*\[([^\]]+)\](?:\s*\(([^)]+)\))?\s*Formula:\s*`([^`]+)`', norm_prompt)
        for num, sid, pos, formula in m:
            skeletons.append({"sid": sid.strip(), "pos": (pos or "expression").strip(), "formula": formula.strip()})
        if not skeletons:
            # Syllabus list fallback: ### TARGET EXPRESSIONS LIST ### \n - expression
            syl_m = re.search(r'###\s*TARGET\s+EXPRESSIONS\s+LIST\s*###(.*?)(?:###|\Z)', norm_prompt, re.DOTALL | re.IGNORECASE)
            if syl_m:
                bullets = re.findall(r'^[ \t]*-[ \t]*([^\r\n]+)', syl_m.group(1), re.MULTILINE)
                for b in bullets:
                    clean_b = re.sub(r'\(.*?\)', '', b).strip()
                    if clean_b and not clean_b.startswith("The text has"):
                        skeletons.append({"formula": clean_b, "word": clean_b})
    elif task_type == "vocabulary":
        # Matches: 1. [S-14] **word** (noun) or 1. [S-14] **word** [AWL]
        m = re.findall(r'(\d+)\.\s*\[([^\]]+)\]\s*\*\*([^*]+)\*\*(?:\s*\(([^)]+)\))?', norm_prompt)
        for num, sid, word, pos in m:
            skeletons.append({"sid": sid.strip(), "word": word.strip(), "pos": (pos or "noun").strip()})
        if not skeletons:
            # Syllabus list fallback: ### TARGET VOCABULARY LIST ### \n - word (pos)
            syl_m = re.search(r'###\s*TARGET\s+VOCABULARY\s+LIST\s*###(.*?)(?:###|\Z)', norm_prompt, re.DOTALL | re.IGNORECASE)
            if syl_m:
                bullets = re.findall(r'^[ \t]*-[ \t]*([^\r\n]+)', syl_m.group(1), re.MULTILINE)
                for b in bullets:
                    # Remove (pos) like (noun) and sentence anchor like [S-7]
                    clean_b = re.sub(r'\[S-\d+\]|\(.*?\)', '', b).strip()
                    if clean_b and not clean_b.startswith("The text has"):
                        skeletons.append({"word": clean_b})
    return skeletons


def _wordlist_matches(clean_target: str, wordlist: set) -> bool:
    """True if the (already cleaned) target word matches any headword in the list.

    Handles multi-word units and inflections via whole-token / shared-4-char-stem
    matching, while avoiding naive substring false positives (e.g. 'win' vs 'window').
    """
    if clean_target in wordlist:
        return True
    target_tokens = [t for t in clean_target.split() if t]
    if not target_tokens:
        return False
    for wl in wordlist:
        wl_tokens = [t for t in wl.split() if t]
        if all(
            any(
                t == w
                or (len(t) >= 4 and len(w) >= 4 and t[:4] == w[:4])
                for w in wl_tokens
            )
            for t in target_tokens
        ):
            return True
    return False


def _detect_task_type(task: str, parsed: Any) -> str:
    """Detect the logical task type from JSON structure first, task name as fallback."""
    t = (task or "").lower()
    # Turn 1 prose drafting is an intermediate natural language stage, not a final JSON output
    if "turn1_prose" in t or "prose_draft" in t:
        return "intermediate_prose"
    if "expert_audit" in t or "quality_audit" in t:
        return "expert_audit"
    if isinstance(parsed, dict):
        if "pass_audit" in parsed and "blind_solve_accuracy" in parsed:
            return "expert_audit"
        if isinstance(parsed.get("questions"), list):
            return "quiz"
        if isinstance(parsed.get("grammar_patterns"), list):
            return "grammar"
        if isinstance(parsed.get("expressions"), list):
            return "expressions"
        if isinstance(parsed.get("vocabulary"), list):
            return "vocabulary"
        if isinstance(parsed.get("concepts"), list):
            return "summary"
        if isinstance(parsed.get("branches"), list):
            return "mindmap"
    if "extract_grammar" in t:
        return "grammar"
    if "extract_expressions" in t:
        return "expressions"
    if "quiz" in t:
        return "quiz"
    if "extract_summary" in t or "summary" in t:
        return "summary"
    if "extract_mindmap" in t or "mindmap" in t:
        return "mindmap"
    if "vocabulary" in t:
        return "vocabulary"
    return "unknown"


def _extract_items(parsed: Any, task_type: str) -> List[Dict[str, Any]]:
    """Return the list of dict-items for the given task type (structure-aware)."""
    if not isinstance(parsed, dict):
        return []
    key_by_type = {
        "quiz": "questions",
        "expert_audit": "questions",
        "grammar": "grammar_patterns",
        "expressions": "expressions",
        "vocabulary": "vocabulary",
        "summary": "concepts",
        "mindmap": "branches",
    }
    preferred = key_by_type.get(task_type)
    candidates = ([preferred] if preferred else []) + [k for k in ITEM_KEYS if k != preferred]
    # First pass: try to find a candidate that contains valid dict items
    for key in candidates:
        value = parsed.get(key)
        if isinstance(value, list) and value:
            valid_items = [it for it in value if isinstance(it, dict)]
            if valid_items:
                return valid_items
    return []


def _extract_json(text: Any) -> Any:
    """Best-effort JSON extraction: direct, fenced ```json, or first balanced brace."""
    if not text or not isinstance(text, str):
        return None
    # Strip <think>...</think> reasoning blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                continue
    return None


def _score_schema(parsed: Any, raw_response: str = "", task_type: str = "") -> Tuple[Optional[float], List[str]]:
    """Dimension 1 (0–25). Always applicable."""
    if isinstance(parsed, dict):
        if len(parsed) == 0:
            return 0.0, ["❌ Valid JSON but empty object"]
        
        flags = []
        deduction = 0.0
        if raw_response and isinstance(raw_response, str):
            trimmed = raw_response.strip()
            # If closing brace was missing or raw string didn't end properly
            if trimmed.count("{") > trimmed.count("}") or trimmed.count("[") > trimmed.count("]"):
                deduction += 5.0
                flags.append("⚠️ Structural repair: unclosed braces/brackets in raw output")
        
        # Array element schema & type enforcement
        ARRAY_SPECS = {
            "mindmap": ("branches", 3, 5, "branch_name"),
            "vocabulary": ("vocabulary", 1, 50, "word"),
            "expressions": ("expressions", 1, 50, "word"),
            "grammar": ("grammar_patterns", 1, 20, "pattern_formula"),
            "summary": ("concepts", 1, 10, "concept_name"),
            "quiz": ("questions", 1, 50, "question"),
        }
        if task_type in ARRAY_SPECS:
            arr_key, min_items, max_items, required_field = ARRAY_SPECS[task_type]
            arr_val = parsed.get(arr_key)
            if not isinstance(arr_val, list):
                deduction += 15.0
                flags.append(f"❌ [INVALID_SCHEMA] Expected list for '{arr_key}', got {type(arr_val).__name__}")
            else:
                non_dict_count = sum(1 for el in arr_val if not isinstance(el, dict))
                if non_dict_count > 0:
                    deduction += 15.0
                    flags.append(f"❌ [INVALID_ITEM_TYPE] {non_dict_count}/{len(arr_val)} elements in '{arr_key}' are not objects (e.g. unparsed JSON strings)")
                valid_dict_count = len(arr_val) - non_dict_count
                if valid_dict_count < min_items:
                    deduction += 10.0
                    flags.append(f"❌ [INSUFFICIENT_ITEMS] '{arr_key}' contains only {valid_dict_count} valid objects (minimum required: {min_items})")

        score = max(0.0, W_SCHEMA - deduction)
        return score, flags
    return 0.0, ["❌ Invalid or missing JSON output"]


def _score_verbatim(items: List[Dict[str, Any]], task_type: str, user_prompt: str, context_prompt: str = "") -> Tuple[Optional[float], List[str]]:
    """Dimension 2 (0–30). Applies to extraction and quiz tasks; returns (None, []) otherwise.

    For extraction tasks this verifies quoted sentences are faithful to the source.
    For quiz tasks this verifies every target_word comes from the supplied word list
    (anti-hallucination: the assessment may not test fabricated vocabulary).
    """
    if task_type not in ("vocabulary", "expressions", "grammar", "quiz"):
        return None, []  # N/A -> normalized out of composite score
    if not items:
        return 0.0, ["❌ No items to evaluate for source faithfulness"]

    # --- Quiz: every target_word must be present in the supplied word list ---
    if task_type == "quiz":
        flags: List[str] = []
        effective_prompt = user_prompt if user_prompt.strip() else (context_prompt or "")
        wordlist = {_clean_core(w) for w in _extract_wordlist(effective_prompt)}
        wordlist = {w for w in wordlist if w}
        checks = matches = 0
        draft_prompt = re.split(r"\n\s*###+\s*🚨|\n\s*###+\s*\[QUALITY AUDIT REVIEW", effective_prompt, flags=re.IGNORECASE)[0]
        clean_draft = _clean_core(draft_prompt)

        for item in items:
            target = str(item.get("target_word") or "").strip()
            if not target:
                continue
            clean_target = _clean_core(target)
            if not clean_target:
                continue
            checks += 1
            if wordlist:
                if _wordlist_matches(clean_target, wordlist):
                    matches += 1
                elif clean_draft and clean_target in clean_draft:
                    # Target word matched in initial prompt draft text
                    matches += 1
                else:
                    flags.append(f"❌ Target word '{target}' not found in supplied word list (possible hallucination)")
            else:
                # If no wordlist extracted from prompt (e.g. self-correction defect ticket),
                # assume compliant with blueprint to avoid false-positive draft content warnings
                matches += 1
        if checks == 0:
            # No target_word items (e.g. reading/translation quizzes) -> nothing to verify -> N/A
            return None, flags
        return max(0.0, round((matches / checks) * W_VERBATIM, 1)), flags

    flags: List[str] = []
    source = _extract_source_content(user_prompt)
    if not source and context_prompt:
        source = _extract_source_content(context_prompt)
    core_src = _clean_core(source)
    checks = matches = 0
    
    for item in items:
        quote = item.get("quoted_sentence") or item.get("quote") or item.get("context_sentence")
        word = str(item.get("word") or item.get("pattern_formula") or "").strip()
        checks += 1

        if not (quote and isinstance(quote, str) and quote.strip()):
            flags.append(f"❌ Missing quote/quoted_sentence for item: '{word[:40]}'")
            continue
            
        # Check 1: Explicit hallucination acknowledgment
        if _is_hallucinated_quote(quote):
            flags.append(f"❌ Hallucinated quote (explicitly inferred/absent): '{quote[:50]}...'")
            continue

        # Check 1.5: Reject prompt instructions / task metadata leakage
        if re.search(r"###\s*(?:TARGET|DETERMINISTIC|PASSAGE|SYLLABUS|OUTPUT|CORE\s+PEDAGOGICAL|JSON\s+SCHEMA)|Extract these exact|syllabus items|Do not skip or omit", quote, re.IGNORECASE):
            flags.append(f"❌ Prompt instruction leakage in quote: '{quote[:50]}...'")
            matches = max(0, matches - 1)
            continue

        # Check 2: Target word must be present in the quoted sentence (for vocabulary & expressions)
        if word and task_type in ("vocabulary", "expressions"):
            clean_word_no_slots = re.sub(r"\[.*?\]|\(.*?\)", " ", word)
            clean_word = _clean_core(clean_word_no_slots)
            clean_quote = _clean_core(quote)
            word_tokens = [w for w in clean_word.split() if w and w not in STOP_SLOTS]
            
            if word_tokens:
                # Match token, stem/inflection (e.g. degrade -> degradation, took -> take, pose -> poses), or irregular forms
                def _tok_in_quote(tok: str) -> bool:
                    return _form_in_text(tok, clean_quote)

                matched_count = sum(1 for w in word_tokens if _tok_in_quote(w))
                min_needed = max(1, len(word_tokens) // 2 + (1 if len(word_tokens) % 2 == 1 else 0))
                word_in_quote = (matched_count >= min_needed)
            else:
                word_in_quote = (clean_word in clean_quote)
                
            if not word_in_quote:
                flags.append(f"❌ Target word '{word}' does not appear in quoted sentence: '{quote[:40]}...'")
                # Severe deduction: directly penalize verbatim score rather than neutral skip
                matches = max(0, matches - 1)
                continue

            # Check 2.5: The headword itself must be a real standalone unit —
            # a recognized English word, or a standalone token in the quoted
            # sentence. Catches lemmatizer corruption (e.g. 'embe') that Check 2's
            # substring matching lets through ('embe' is a prefix of 'embedded'),
            # without false-flagging legitimate words WordNet lacks ('ice-cream',
            # "don't") or text-specific terms that appear verbatim in the quote.
            if " " not in clean_word and clean_word.isalpha():
                if LinguisticEngine.is_known_english_word(clean_word) is not True:
                    if clean_word not in clean_quote.split():
                        flags.append(
                            f"❌ Headword '{word}' is not a recognized English word and does not appear "
                            f"as a standalone token in the quoted sentence (suspect lemmatization artifact)"
                        )
                        matches = max(0, matches - 1)
                        continue


        # Check 3: Cleaned quote in source or high n-gram coverage
        core_quote = _clean_core(quote)
        is_verbatim = False
        if core_quote:
            if core_quote in core_src or _ngram_coverage(core_quote, core_src, n=3) >= 0.80:
                is_verbatim = True
            elif "..." in quote or "…" in quote:
                # If model used ellipsis to omit middle parts of a long sentence, check each segment
                segments = [s.strip() for s in re.split(r'\.{3,}|…', quote) if s.strip()]
                meaningful_segs = [s for s in segments if len(_clean_core(s).split()) >= 2]
                if meaningful_segs and all(_clean_core(seg) in core_src or _ngram_coverage(_clean_core(seg), core_src, n=3) >= 0.80 for seg in meaningful_segs):
                    is_verbatim = True

        if is_verbatim:
            matches += 1
        else:
            flags.append(f"❌ Non-verbatim quote detected: '{quote[:40]}...'")
            
    if checks == 0:
        return 0.0, ["❌ No items to evaluate for source faithfulness"]
    return max(0.0, round((matches / checks) * W_VERBATIM, 1)), flags



def _score_pedagogy(items: List[Dict[str, Any]], task_type: str, user_prompt: str = "") -> Tuple[Optional[float], List[str]]:
    """Dimension 3 (0–25). Evaluates pedagogical quality across extraction and assessment types."""
    if task_type not in ("vocabulary", "expressions", "grammar", "quiz", "summary", "mindmap", "expert_audit"):
        return None, []
    if not items:
        return 0.0, [f"⚠️ Output list for '{task_type}' is empty."]
        
    flags: List[str] = []
    checks = passes = 0

    # Extract study list headwords and blueprint specifications from prompt if evaluating a quiz
    prompt_headwords = set()
    prompt_blueprints: Dict[str, Dict[str, Any]] = {}
    if task_type == "quiz" and user_prompt:
        for m in re.finditer(r"## \[\[(.*?)\]\]", user_prompt):
            prompt_headwords.add(m.group(1).strip().lower())
        if not prompt_headwords:
            for m in re.finditer(r"-\s*Target:\s*([^\n\r]+)", user_prompt):
                prompt_headwords.add(m.group(1).strip().lower())
        # Support Target Word from Authentic Corpus Blueprint
        for m in re.finditer(r"-\s*Target Word:\s*([^\n\r]+)", user_prompt, re.IGNORECASE):
            w = m.group(1).strip().lower()
            if w:
                prompt_headwords.add(w)

        # Parse full item blueprints: ### Item N ###
        blueprint_blocks = re.split(r"###\s*Item\s*\d+\s*###", user_prompt, flags=re.IGNORECASE)
        for block in blueprint_blocks:
            m_target = re.search(r"-\s*Target Word:\s*([^\n\r]+)", block, re.IGNORECASE)
            if not m_target:
                continue
            b_word = m_target.group(1).strip().lower()
            m_pos = re.search(r"-\s*Part of Speech:\s*([^\n\r]+)", block, re.IGNORECASE)
            m_infl = re.search(r"-\s*Inflectional Form:\s*([^\n\r]+)", block, re.IGNORECASE)
            m_anc = re.search(r"-\s*Collocational Anchor:\s*([^\n\r]+)", block, re.IGNORECASE)
            m_ex = re.search(
                r"(?:Authentic Corpus Blueprint:\s*'(.+?)'\s*$"
                r"|authentic_example['\"]\s*:\s*['\"]([^'\n\r]+)['\"])",
                block,
                re.IGNORECASE | re.MULTILINE,
            )
            
            anchor_val = ""
            anchor_type = ""
            if m_anc:
                raw_anc = m_anc.group(1).strip()
                m_type = re.match(r"^(.*?)\s*\(([^)]+)\)$", raw_anc)
                if m_type:
                    anchor_val = m_type.group(1).strip().lower()
                    anchor_type = m_type.group(2).strip().lower()
                else:
                    anchor_val = raw_anc.lower()

            ex_val = ""
            if m_ex:
                ex_val = (m_ex.group(1) or m_ex.group(2) or "").strip()

            # Quote provenance as rendered by the blueprint: a licensed quote is corpus evidence
            # strong enough to be modelled, a display-only quote may never be reused at all.
            m_lq = re.search(
                r"Curriculum Quote \(licensed\):\s*'(.+?)'\s*$", block, re.IGNORECASE | re.MULTILINE
            )
            m_cq = re.search(
                r"Curriculum Quote \(display-only.*?\):\s*'(.+?)'\s*$", block, re.IGNORECASE | re.MULTILINE
            )
            licensed_quote_val = m_lq.group(1).strip() if m_lq else ""
            candidate_quote_val = m_cq.group(1).strip() if m_cq else ""

            # Backlog D1: the blueprint also prescribes the four options and the answer
            # index; the writer is only allowed to author stem, explanation and audit.
            m_opts = re.search(r"-\s*Prescribed Options:\s*\[([^\]]*)\]", block, re.IGNORECASE)
            prescribed_opts_val = [o.strip() for o in m_opts.group(1).split(",") if o.strip()] if m_opts else []
            m_pidx = re.search(r"-\s*Correct Answer Index:\s*(\d+)", block, re.IGNORECASE)
            prescribed_index_val = int(m_pidx.group(1)) if m_pidx else None


            prompt_blueprints[b_word] = {
                "target_word": b_word,
                "part_of_speech": m_pos.group(1).strip().lower() if m_pos else "",
                "inflection": m_infl.group(1).strip().lower() if m_infl else "",
                "anchor": anchor_val,
                "anchor_type": anchor_type,
                "example": ex_val,
                "licensed_quote": licensed_quote_val,
                "candidate_quote": candidate_quote_val,
                "quote_provenance": "weak" if candidate_quote_val else ("strong" if licensed_quote_val else ""),
                "prescribed_options": prescribed_opts_val,
                "prescribed_index": prescribed_index_val,

            }

    # Backlog C1: one ceiling for every support sentence the student reads.
    ceiling_level, ceiling_words = (None, set())
    if task_type in ("vocabulary", "expressions", "grammar") and user_prompt:
        ceiling_level, ceiling_words = _cefr_source_ceiling(user_prompt)
    elif task_type == "quiz" and user_prompt:
        m_cefr = re.search(r'(?:overall_)?cefr(?:_level)?:\s*["\']?([A-C][1-2])["\']?|CEFR\s+([A-C][1-2])', user_prompt, re.IGNORECASE)
        if m_cefr:
            ceiling_level = (m_cefr.group(1) or m_cefr.group(2)).upper()

    def _over_example_ceiling(sentence: str, own_words: Iterable[str] = ()) -> List[Tuple[str, str]]:
        """Words of an example sentence that the source passage does not license.

        `own_words` are exempt: an example is required to contain the headword it
        teaches, however hard that headword is.
        """
        if not ceiling_level or not sentence:
            return []
        allow = set(ceiling_words)
        for word_text in own_words:
            lowered = str(word_text).lower()
            allow.update(re.findall(r"[a-z]{4,}", lowered))
            # 're-schedule' is one headword; its example may show 'rescheduled',
            # which the lemma lookup then resolves back to the exempt form.
            joined = re.sub(r"[^a-z]+", "", lowered)
            if joined:
                allow.add(joined)
        return LinguisticEngine.over_ceiling_tokens(sentence, ceiling_level, mode="text", allow=allow)

    def _ceiling_reason(over: List[Tuple[str, str]], field: str) -> str:
        offenders = ", ".join(f"'{w}' ({lvl})" for w, lvl in over[:3])
        extra = f" (+{len(over) - 3} more)" if len(over) > 3 else ""
        band = LinguisticEngine.cefr_ceiling(ceiling_level, mode="text")
        return (
            f"{field} exceeds the CEFR {band} ceiling of the CEFR {ceiling_level} "
            f"source passage: {offenders}{extra}"
        )

    # Backlog D1: the formulas the lexicon pre-extracted, indexed by a presentation-
    # insensitive key. An item consumes a skeleton when it copies it; a same-category
    # skeleton left unconsumed means the model rewrote the formula it was handed.
    grammar_skeletons: List[Dict[str, str]] = []
    skeleton_keys: Dict[str, List[Dict[str, str]]] = {}
    if task_type == "grammar" and user_prompt:
        grammar_skeletons = _extract_target_skeletons(user_prompt, "grammar")
        from .processor import WikiProcessor
        for _sk in grammar_skeletons:
            _raw_formula = str(_sk.get("formula", ""))
            # Index the skeleton both as mined and as the pipeline would canonicalize it:
            # the pedagogy branch unwraps literal brackets ('[Main Clause]' -> '[S]')
            # before comparing, so both presentations are the same formula.
            _skel_keys = {
                canonical_formula_key(_raw_formula),
                canonical_formula_key(WikiProcessor.unwrap_literal_brackets(_raw_formula)),
            }
            for _key in _skel_keys:
                if _key:
                    skeleton_keys.setdefault(_key, []).append(_sk)
    consumed_skeletons: set = set()

    def _align_grammar_formula(pattern: str, category: str) -> Tuple[bool, str]:
        """(copied a skeleton formula?, the skeleton formula this item should have copied).

        A formula that equals a mined skeleton is aligned even if another item already
        delivered that skeleton — duplicated coverage is the coverage gate's business,
        not an alignment defect. Only a formula that equals NO mined skeleton is a rewrite.
        """
        if not grammar_skeletons:
            return True, ""
        key = canonical_formula_key(pattern)
        matched = skeleton_keys.get(key, [])
        for sk in matched:
            if id(sk) not in consumed_skeletons:
                consumed_skeletons.add(id(sk))
                return True, str(sk.get("formula", ""))
        if matched:
            return True, ""
        wanted = ""
        for sk in grammar_skeletons:
            if id(sk) in consumed_skeletons:
                continue
            if str(sk.get("category", "")).strip().lower() == str(category).strip().lower():
                wanted = str(sk.get("formula", ""))
                break
        return False, wanted

    for item in items:
        if task_type == "vocabulary":
            word = _safe_str(item.get("word"))
            pos = _safe_str(item.get("part_of_speech")).lower()
            quote = _safe_str(item.get("quoted_sentence"))
            is_expression = LinguisticEngine.is_multiword_expression(word)
            # A label is missing, or a multi-word unit carries the part of speech of one
            # of its tokens ('tap into' -> 'verb') instead of its expression type.
            needs_pos = bool(word) and (
                not pos or (is_expression and pos not in LinguisticEngine.EXPRESSION_TYPE_LABELS)
            )
            if needs_pos:
                # A multi-word unit is typed as an expression ('collocation', 'phrasal
                # verb', 'set phrase', 'idiom'); only a single headword is typed by the
                # parse of its quote.
                if is_expression:
                    pos = LinguisticEngine.classify_expression_type(word, quote) or "collocation"
                else:
                    pos = LinguisticEngine.determine_contextual_pos(word, quote)
                item["part_of_speech"] = pos
            definition = _safe_str(item.get("definition"))
            example = _safe_str(item.get("example_usage"))
            checks += 1
            
            # Allow words/phrases with slots, hyphens, brackets, parentheses, apostrophes
            valid_word = bool(re.search(r"^[A-Za-z\s\-'\[\]\(\)]+$", word) and len(word) >= 2)
            # A multi-word row may carry an expression type, or (as a fallback the healing
            # above could not decide) the part of speech of a fixed frame it is taught as.
            allowed_pos = (VALID_POS_SET | VALID_EXPR_POS_SET) if is_expression else VALID_POS_SET
            valid_pos = pos in allowed_pos
            has_example = bool(example)
            is_distinct_example = bool(has_example and _clean_core(example) != _clean_core(quote))
            over_ceiling = _over_example_ceiling(example, [word]) if is_distinct_example else []
            # F2: 'A core academic term functioning as a noun.' is the engine's own filler,
            # not a definition. It fails the check the same way a missing one does.
            invented_definition = bool(definition) and LinguisticEngine.is_boilerplate_definition(definition)

            if valid_word and valid_pos and definition and not invented_definition and is_distinct_example and not over_ceiling:
                passes += 1
            else:
                reasons = []
                if not valid_word: reasons.append(f"invalid headword '{word}'")
                if not valid_pos: reasons.append(f"invalid PoS '{pos}'")
                if not definition: reasons.append("missing definition")
                elif invented_definition: reasons.append("invented fallback definition (no lexicon host grounds it)")
                if not has_example:
                    reasons.append("missing example_usage")
                elif not is_distinct_example:
                    reasons.append("example_usage is an unoriginal duplicate of quoted_sentence")
                elif over_ceiling:
                    reasons.append(_ceiling_reason(over_ceiling, "example_usage"))
                flags.append(f"⚠️ Vocabulary item '{word}' failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "expressions":
            word = _safe_str(item.get("word"))
            definition = _safe_str(item.get("definition"))
            example = _safe_str(item.get("example_usage"))
            quote = _safe_str(item.get("quoted_sentence"))
            checks += 1

            is_multiword = bool(re.search(r"\[.+?\]|one's", word, re.IGNORECASE) or len(word.split()) > 1)
            is_trivial = word.lower() in ("talk", "listen", "turn", "watch", "sit down", "talk to", "listen to", "look at")
            has_example = bool(example)
            is_distinct_example = bool(has_example and _clean_core(example) != _clean_core(quote))
            over_ceiling = _over_example_ceiling(example, [word]) if is_distinct_example else []
            # F2: the cascade's invented filler is not a definition. An expression the lexicon
            # does not define is a defect to be reported, not a row to be shipped.
            invented_definition = bool(definition) and LinguisticEngine.is_boilerplate_definition(definition)

            if is_multiword and not is_trivial and definition and not invented_definition and is_distinct_example and not over_ceiling:
                passes += 1
            else:
                reasons = []
                if not is_multiword or is_trivial:
                    reasons.append("lacks multi-word/slot form or is too basic")
                if not definition:
                    reasons.append("missing definition")
                elif invented_definition:
                    reasons.append("invented fallback definition (no lexicon host grounds it)")
                if not has_example:
                    reasons.append("missing example_usage")
                elif not is_distinct_example:
                    reasons.append("example_usage is an unoriginal duplicate of quoted_sentence")
                elif over_ceiling:
                    reasons.append(_ceiling_reason(over_ceiling, "example_usage"))
                flags.append(f"⚠️ Expression '{word}' failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "grammar":
            pattern = _safe_str(item.get("pattern_formula"))
            audit = _safe_str(item.get("design_audit"))
            quote = _safe_str(item.get("quote"))
            category = _safe_str(item.get("category"))
            if not category and quote:
                category = LinguisticEngine.classify_grammar_dependency(quote) or "Information Packaging"
                item["category"] = category
            checks += 1
            
            reasons = []
            imitation = _safe_str(item.get("imitation_example"))
            if not audit:
                reasons.append("missing design_audit")
            if not pattern:
                reasons.append("missing pattern_formula")
            if not imitation:
                reasons.append("missing imitation_example")
            else:
                if quote and _clean_core(imitation) == _clean_core(quote):
                    # Backlog D1: an imitation example that IS the passage sentence teaches
                    # no transfer — the student copies instead of re-using the pattern.
                    reasons.append("imitation_example duplicates the source quote")
                    flags.append(
                        f"❌ Grammar pattern '{pattern[:40]}': imitation_example duplicates the source quote "
                        "(an imitation must be a new sentence, not the quoted one)"
                    )

                over_ceiling = _over_example_ceiling(imitation)
                if over_ceiling:
                    # Backlog C1: a pattern modelled from an A2 passage must not be
                    # demonstrated with C2 vocabulary the student has not met.
                    reasons.append(_ceiling_reason(over_ceiling, "imitation_example"))
            if not _safe_str(item.get("common_mistakes")):
                reasons.append("missing common_mistakes")
            
            # Auto-unwrap accidental brackets around literal functional words (e.g. [it], [that], [if])
            from .processor import WikiProcessor
            pattern = WikiProcessor.unwrap_literal_brackets(pattern)

            # Minimal Anti-Triviality Gate: penalize simple patterns whose ONLY anchor is a standalone conversational filler/coordinator (e.g. 'But [S]', 'And [S]')
            slot_count = len(re.findall(r"\[.*?\]", pattern))
            lits = [
                w for w in re.sub(r"\[.*?\]|\(.*?\)|[+,/]", " ", pattern).split()
                if len(_clean_core(w)) >= 2
            ]
            TRIVIAL_ANCHOR_WORDS = frozenset({"but", "and", "so", "or", "of", "course", "well", "now", "here"})
            TRIVIAL_PHRASES = frozenset({"but", "and", "so", "or", "of course", "here is", "heres"})
            clean_lits = [re.sub(r"[^\w\s]", "", l).lower().strip() for l in lits if re.sub(r"[^\w\s]", "", l).strip()]
            combined_lit_phrase = " ".join(clean_lits)
            # Reject un-abstracted trivial formulas anchored only by sentence-connectors (e.g. 'However, [S]', 'Therefore, [Clause]')
            if slot_count <= 1:
                if (combined_lit_phrase in TRIVIAL_PHRASES) or (clean_lits and all(l in TRIVIAL_ANCHOR_WORDS for l in clean_lits)):
                    reasons.append(f"trivial formula anchored only by conversational filler or conjunction: '{pattern}'")
                elif clean_lits and any(l in {"however", "therefore", "moreover", "furthermore", "meanwhile", "nevertheless", "in fact", "actually"} for l in clean_lits):
                    reasons.append(f"un-abstracted superficial pattern anchored only on discourse adverbial connector: '{pattern}'")

            # Structural Consistency: Literal keywords in formula MUST exist in quote AND imitation_example
            # (e.g. if formula requires 'while', 'not... but...', 'which', or '[V-ing Phrase]',
            # both the quoted text and the pedagogical imitation sentence must physically instantiate them)
            clean_quote = _clean_core(quote)
            clean_quote_raw = quote.lower()
            clean_imitation = _clean_core(imitation)
            clean_imitation_raw = imitation.lower()
            if pattern:
                # Check for explicit literal connector / subordinator words in formula (supporting slash alternatives e.g. is/was)
                raw_lits = re.findall(r"[a-zA-Z0-9_\'/]+", re.sub(r"\[.*?\]|\(.*?\)", " ", pattern))
                expanded_quote = _clean_core(_expand_contractions(quote))
                expanded_imitation = _clean_core(_expand_contractions(imitation))
                quote_tokens = set(clean_quote.split()) | set(expanded_quote.split())
                imitation_tokens = set(clean_imitation.split()) | set(expanded_imitation.split())
                missing_anchors_quote = []
                missing_anchors_imitation = []
                for lit in raw_lits:
                    alts = [a.strip() for a in lit.split("/") if a.strip()]
                    valid_alts = [a for a in alts if len(_clean_core(a)) >= 2 and _clean_core(a) not in COBUILD_POS_TOKENS]
                    if not valid_alts:
                        continue
                    if not any(_clean_core(a) in quote_tokens for a in valid_alts):
                        missing_anchors_quote.append(lit)
                    if imitation and len(imitation.split()) >= 3:
                        is_instantiated = any(_clean_core(a) in imitation_tokens for a in valid_alts)
                        # Backlog F4 tolerance: propositional encapsulation interpretive verbs (show/suggest/mean/indicate)
                        # allow cross-verb instantiation within the closed interpretive verb paradigm
                        if not is_instantiated and any(re.match(r"^(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?)$", a, re.IGNORECASE) for a in valid_alts):
                            if re.search(r"\b(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?)\b", imitation, re.IGNORECASE):
                                is_instantiated = True
                        if not is_instantiated:
                            missing_anchors_imitation.append(lit)

                if missing_anchors_quote:
                    reasons.append(f"formula anchor '{', '.join(missing_anchors_quote)}' ungrounded in quote")
                if missing_anchors_imitation:
                    # Backlog F4: An imitation sentence must instantiate its own formula
                    reasons.append(f"imitation_example does not instantiate formula anchor '{', '.join(missing_anchors_imitation)}'")
                    flags.append(
                        f"❌ Grammar pattern '{pattern[:40]}': imitation_example does not instantiate formula anchor "
                        f"'{', '.join(missing_anchors_imitation)}' (an imitation must instantiate the formula it models)"
                    )

                # Check for participle requirement
                if "[v-ing" in pattern.lower() or "[participle" in pattern.lower():
                    ING_EXCLUDE = frozenset({"morning", "evening", "thing", "something", "nothing", "everything", "anything", "during", "ceiling"})
                    ing_tokens_quote = [w for w in re.findall(r"\b[a-zA-Z]{3,}ing\b", clean_quote_raw) if w not in ING_EXCLUDE]
                    if not ing_tokens_quote:
                        reasons.append("formula requires '[V-ing]' but quote contains no participle verb ungrounded in quote")
                    if imitation:
                        ing_tokens_imitation = [w for w in re.findall(r"\b[a-zA-Z]{3,}ing\b", clean_imitation_raw) if w not in ING_EXCLUDE]
                        if not ing_tokens_imitation:
                            reasons.append("formula requires '[V-ing]' but imitation_example contains no participle verb")

            # Backlog D1: Degenerate Slot Gate — '[NP/NP]' names the same constituent
            # twice, teaches nothing, and is never mined by the lexicon.
            degenerate_slots = degenerate_grammar_slots(pattern)
            if degenerate_slots:
                reasons.append(f"degenerate formula slot {degenerate_slots} names the same constituent twice")
                flags.append(
                    f"❌ Grammar pattern '{pattern[:40]}': degenerate formula slot {degenerate_slots} "
                    "(a slot may not be offered to itself)"
                )

            # Backlog D1: Skeleton Alignment Gate — the formula the lexicon mined is the
            # formula the item must ship. Casing, '+' separators and slot synonyms are
            # presentation; a rewritten structure is a deviation.
            aligned, wanted_formula = _align_grammar_formula(pattern, category)
            if not aligned and wanted_formula:
                reasons.append(f"pattern_formula deviates from the pre-extracted skeleton '{wanted_formula}'")
                flags.append(
                    f"❌ Grammar pattern '{pattern[:40]}' deviates from the pre-extracted skeleton "
                    f"formula '{wanted_formula}'"
                )
            elif not aligned:
                flags.append(
                    f"⚠️ Grammar pattern '{pattern[:40]}' matches none of the pre-extracted skeleton formulas"
                )


            # Auto-Remap explicit category mismatches if not already healed
            _, remap_notice = auto_remap_grammar_category(item)
            if remap_notice:
                flags.append(remap_notice)
                category = str(item.get("category", "")).strip()

            # Four Macro Functional Domains deterministic checks
            # 1. Rhetoric & Emphasis: Check for inversion markers, genuine cleft relative clauses, or parallelism coordination
            if category == "Rhetoric & Emphasis":
                has_inversion_trigger = bool(re.search(r"\b(?:only|never|hardly|scarcely|seldom|rarely|little|not only|neither|nor|no sooner)\b", quote, re.IGNORECASE))
                has_cleft = bool(re.search(r"\bIt\s+(?:is|was|were|'s|has\s+been|had\s+been)\b", quote, re.IGNORECASE) and re.search(r"\b(?:that|who|whom|which)\b", quote, re.IGNORECASE))
                has_wh_cleft = bool(re.search(r"\bWhat\s+[a-z0-9_']+\s+(?:is|was|were)\b", quote, re.IGNORECASE))
                has_parallel_coordination = bool(
                    re.search(r"\b(?:not only\b.*?\bbut\b|either\b.*?\bor\b|neither\b.*?\bnor\b|both\b.*?\band\b|not\s+.*?\s*,\s*but\b)\b", quote, re.IGNORECASE)
                    or ";" in quote or "," in quote
                )
                if not (has_inversion_trigger or has_cleft or has_wh_cleft or has_parallel_coordination):
                    reasons.append("category 'Rhetoric & Emphasis' assigned to quote lacking rhetorical markers (inversion, cleft focus, or structural balance)")

            # 2. Logic & Stance: Check for conditional, concessive, epistemic hedging, or stance modal assertions
            elif category == "Logic & Stance":
                has_cond = bool(re.search(r"\b(?:if|unless|provided\s+that|supposing|as\s+long\s+as|had\s+\w+\s+\w+|should\s+\w+\s+\w+|were\s+\w+\s+\w+)\b", quote, re.IGNORECASE))
                has_concessive = bool(re.search(r"\b(?:although|though|even though|even if|while|whereas|despite|in spite of)\b", quote, re.IGNORECASE))
                has_stance_or_hedging = bool(re.search(r"\b(?:can|cannot|can't|could|may|might|must|should|would|will|suggest|indicate|appear|seem|likely|probably|presumably|arguably|tends?\s+to)\b", quote, re.IGNORECASE))
                if not (has_cond or has_concessive or has_stance_or_hedging):
                    reasons.append("category 'Logic & Stance' assigned to quote lacking logical condition, concessive linker, or epistemic/stance modal assertion")

            # 3. Information Packaging: Check for non-finite verb forms, nominalization, dummy-it extraposition, or elaborative non-restrictive clause
            elif category == "Information Packaging":
                ING_NON_VERBS = frozenset({"morning", "evening", "thing", "something", "nothing", "everything", "anything", "ring", "spring", "king", "wing", "sing", "during", "ceiling"})
                ing_tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}ing\b", quote) if w.lower() not in ING_NON_VERBS]
                has_participle = bool(ing_tokens or re.search(r"\b(?:having\s+\w+(?:ed|en|t)|surrounded|driven|given|taken|seen|known|based|built|born|located|situated|reminded|confronted|faced)\b", quote, re.IGNORECASE))
                TO_NOUNS = frozenset({"school", "work", "bed", "church", "college", "prison", "jail", "court", "sea", "town", "home", "market", "class", "them", "him", "her", "us", "me", "you", "it", "this", "that", "these", "those"})
                to_matches = [m.group(1).lower() for m in re.finditer(r"\bto\s+([a-z]{2,})\b", quote, re.IGNORECASE)]
                has_infinitive_clause = any(word not in TO_NOUNS for word in to_matches)
                has_dummy_it = bool(re.search(r"\bit(?:'s|\s+is|\s+was|\s+has\s+been)\s+(?:[a-z]{4,}\s+)?(?:that|to\s+[a-z]+)\b", quote, re.IGNORECASE))
                has_nominalization = bool(re.search(r"\b[a-z]{3,}(?:tion|sion|ment|ance|ence|ity|ness)\b", quote, re.IGNORECASE))
                has_elaborative_relative = bool(re.search(r",\s*which\s+[a-z]+", quote, re.IGNORECASE))
                if not (has_participle or has_infinitive_clause or has_dummy_it or has_nominalization or has_elaborative_relative):
                    reasons.append("category 'Information Packaging' assigned to quote lacking non-finite clauses, evaluative it-extraposition, dense nominalization, or elaborative clause")

            # 4. Cohesion & Framing: Check for shell nouns, complement that-clauses, or encapsulation
            elif category == "Cohesion & Framing":
                has_shell_frame = bool(re.search(r"\b(?:the\s+(?:fact|idea|notion|reason|belief|claim|argument|possibility|question|view|conclusion)\s+that|this\s+(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|led\s+to|leads\s+to)|the\s+extent\s+to\s+which)\b", quote, re.IGNORECASE))
                has_anaphoric_encapsulation = bool(re.search(r"\b(?:this|these|such)\s+[a-z]{4,}\b", quote, re.IGNORECASE) or bool(re.search(r"\bthat\s+is\s+why\b", quote, re.IGNORECASE)))
                # Closed set of interpretive verbs across all inflectional forms (base, 3sg, past, participle) + optional modal auxiliaries
                INTERPRETIVE_VERBS_REGEX = (
                    r"(?:(?:would|could|might|may|can|will)\s+)?"
                    r"(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?)"
                )
                has_which_propositional = bool(re.search(rf",\s*which\s+{INTERPRETIVE_VERBS_REGEX}\b", quote, re.IGNORECASE))
                has_rel_frame = bool(re.search(r"\b(?:in\s+which|by\s+which|through\s+which|whereby)\b", quote, re.IGNORECASE))
                if not (has_shell_frame or has_anaphoric_encapsulation or has_which_propositional or has_rel_frame):
                    reasons.append("category 'Cohesion & Framing' assigned to quote lacking abstract shell frame, prepositional relative, or discourse encapsulation")

            # Boundary tolerance (any-macro-domain rule): if the strict per-category gate above
            # rejected the model's chosen category, but the quote structurally fits ANOTHER of
            # the 4 macro functional domains, it is a defensible boundary case (boundary
            # sentences sit on two domains and have no unique label). Tolerate it instead of
            # rejecting — only reject quotes that match NO domain (kept as the original fatal
            # reason, so genuinely marker-less quotes are still quarantined).
            if any(r.startswith("category '") for r in reasons):
                _ING_EXCLUDE = frozenset({"morning", "evening", "thing", "something", "nothing", "everything", "anything", "ring", "spring", "king", "wing", "sing", "during", "ceiling"})
                _TO_EXCLUDE = frozenset({"school", "work", "bed", "church", "college", "prison", "jail", "court", "sea", "town", "home", "market", "class", "them", "him", "her", "us", "me", "you", "it", "this", "that", "these", "those"})
                _fits_rhetoric = (
                    re.search(r"\b(?:only|never|hardly|scarcely|seldom|rarely|little|not only|neither|nor|no sooner)\b", quote, re.IGNORECASE)
                    or (re.search(r"\bIt\s+(?:is|was|were|'s|has\s+been|had\s+been)\b", quote, re.IGNORECASE) and re.search(r"\b(?:that|who|whom|which)\b", quote, re.IGNORECASE))
                    or re.search(r"\bWhat\s+[a-z0-9_']+\s+(?:is|was|were)\b", quote, re.IGNORECASE)
                    or re.search(r"\b(?:not only\b.*?\bbut\b|either\b.*?\bor\b|neither\b.*?\bnor\b|both\b.*?\band\b|not\s+.*?\s*,\s*but\b)\b", quote, re.IGNORECASE)
                    or ";" in quote or "," in quote
                )
                _fits_logic = (
                    re.search(r"\b(?:if|unless|provided\s+that|supposing|as\s+long\s+as|had\s+\w+\s+\w+|should\s+\w+\s+\w+|were\s+\w+\s+\w+)\b", quote, re.IGNORECASE)
                    or re.search(r"\b(?:although|though|even though|even if|while|whereas|despite|in spite of)\b", quote, re.IGNORECASE)
                    or re.search(r"\b(?:can|cannot|can't|could|may|might|must|should|would|will|suggest|indicate|appear|seem|likely|probably|presumably|arguably|tends?\s+to)\b", quote, re.IGNORECASE)
                )
                _ing_tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}ing\b", quote) if w.lower() not in _ING_EXCLUDE]
                _to_words = [m.group(1).lower() for m in re.finditer(r"\bto\s+([a-z]{2,})\b", quote, re.IGNORECASE)]
                _fits_info = (
                    _ing_tokens
                    or re.search(r"\b(?:having\s+\w+(?:ed|en|t)|surrounded|driven|given|taken|seen|known|based|built|born|located|situated|reminded|confronted|faced)\b", quote, re.IGNORECASE)
                    or any(w not in _TO_EXCLUDE for w in _to_words)
                    or re.search(r"\bit(?:'s|\s+is|\s+was|\s+has\s+been)\s+(?:[a-z]{4,}\s+)?(?:that|to\s+[a-z]+)\b", quote, re.IGNORECASE)
                    or re.search(r"\b[a-z]{3,}(?:tion|sion|ment|ance|ence|ity|ness)\b", quote, re.IGNORECASE)
                    or re.search(r",\s*which\s+[a-z]+", quote, re.IGNORECASE)
                )
                _fits_cohesion = (
                    re.search(r"\b(?:the\s+(?:fact|idea|notion|reason|belief|claim|argument|possibility|question|view|conclusion)\s+that|this\s+(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|led\s+to|leads\s+to)|the\s+extent\s+to\s+which)\b", quote, re.IGNORECASE)
                    or re.search(r"\b(?:this|these|such)\s+[a-z]{4,}\b", quote, re.IGNORECASE)
                    or re.search(r"\bthat\s+is\s+why\b", quote, re.IGNORECASE)
                    or re.search(r",\s*which\s+(?:(?:would|could|might|may|can|will)\s+)?(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?)\b", quote, re.IGNORECASE)
                    or re.search(r"\b(?:in\s+which|by\s+which|through\s+which|whereby)\b", quote, re.IGNORECASE)
                )
                if _fits_rhetoric or _fits_logic or _fits_info or _fits_cohesion:
                    reasons[:] = [r for r in reasons if not r.startswith("category '")]
                    flags.append("ℹ️ Boundary pattern: chosen category rejected by strict gate, but quote fits another macro-domain — tolerated (any-macro-domain rule)")

            # Quote cleanliness warning: trailing ellipsis
            if quote.rstrip().endswith(("...", "…")):
                flags.append(f"⚠️ Quote ends with trailing ellipsis: '{quote[:40]}...'")

            if not reasons:
                passes += 1
            else:
                flags.append(f"⚠️ Grammar pattern '{pattern[:40]}' failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "quiz":
            options = item.get("options", [])
            idx = item.get("correct_answer_index")
            explanation = _safe_str(item.get("explanation") or item.get("diagnostic_critique"))
            question = _safe_str(item.get("question") or item.get("translated_sentence") or item.get("source_sentence"))
            target = _safe_str(item.get("target_word") or item.get("target_keyword") or item.get("word")).strip().lower()
            is_translation = bool(item.get("translated_sentence") or item.get("source_sentence") or item.get("english_skeleton") or item.get("target_keyword") or item.get("idiomatic_translation"))
            skeleton = _safe_str(item.get("english_skeleton"))
            is_comparative_translation = bool(item.get("idiomatic_translation") and item.get("flawed_translation"))
            is_comprehension = not target and not is_translation
            expected_opt_count = 2 if is_comparative_translation else 4
            checks += 1
            
            # 1. Option count and distinctness (case/space-insensitive uniqueness check)
            is_list_valid = isinstance(options, list) and len(options) == expected_opt_count
            has_no_duplicates = is_list_valid and len(set(str(o).strip().lower() for o in options)) == expected_opt_count
            
            # 2. Correct answer index validity
            valid_idx = isinstance(idx, int) and 0 <= idx < expected_opt_count
            
            # 3. Target word must be present in options and match the correct answer slot options[idx]
            target_in_options = True
            if target and is_list_valid and valid_idx:
                clean_target = _clean_core(target)
                if is_comparative_translation:
                    # In comparative translation, target_keyword must be in idiomatic_translation
                    idiomatic_ans = _safe_str(item.get("idiomatic_translation")).lower()
                    target_in_options = clean_target in _clean_core(idiomatic_ans) or (
                        len(clean_target) >= 4 and clean_target[:4] in _clean_core(idiomatic_ans)
                    )
                elif is_translation:
                    # In legacy translation quizzes, target_keyword can be in the options slot,
                    # or in the fixed english_skeleton / correct_english_answer!
                    full_ans = _safe_str(item.get("correct_english_answer")).lower()
                    skel_ans = _safe_str(item.get("english_skeleton")).lower()
                    opt_ans = _safe_str(options[idx]).lower()
                    target_in_options = (
                        clean_target in _clean_core(full_ans) or
                        clean_target in _clean_core(skel_ans) or
                        clean_target in _clean_core(opt_ans) or
                        (len(clean_target) >= 4 and any(
                            clean_target[:4] in _clean_core(text)
                            for text in [full_ans, skel_ans, opt_ans]
                        ))
                    )
                else:
                    selected_opt = _clean_core(str(options[idx]))
                    # Match exact or valid inflections (e.g. assert -> asserted, marathon -> marathons)
                    # or quantifier/phrase head blanking (e.g. clean_target="a piece of", selected_opt="piece")
                    target_in_options = (clean_target == selected_opt) or (
                        selected_opt in clean_target.split()
                    ) or (
                        len(clean_target) >= 4 and len(selected_opt) >= 4 and (
                            selected_opt.startswith(clean_target[:4]) or clean_target.startswith(selected_opt[:4]) or (clean_target in selected_opt)
                        )
                    )

            # 4. Blank verification for fill-in-the-blank questions (Strict Single Blank Gate)
            has_blank_when_expected = True
            is_single_blank = True
            no_stem_leak = True
            no_article_leak = True
            is_adequate_complexity = True
            # Gates below are only evaluated for lexical fill-in-the-blank items; they must
            # still be bound for translation / comprehension / comparative-translation items
            # or the shared item_passed computation below raises UnboundLocalError.
            no_cross_target_leak = True
            leaked_other_targets: List[str] = []
            no_verbatim_example = True
            verbatim_label = "dictionary example"
            inflection_agreed = True
            illegal_distractors: List[str] = []
            # Backlog D1: the Prescribed Options Contract is satisfied by default — it only
            # binds when the blueprint actually prescribes options (cloze and comprehension
            # blueprints do not).
            options_match_blueprint = True
            index_match_blueprint = True


            if is_comparative_translation or is_comprehension:
                # Comparative translation and Reading/Listening/Video comprehension questions do not require blanks
                has_blank_when_expected = True
            elif is_translation:
                if skeleton:
                    has_blank_when_expected = bool(re.search(r"\[\s*_{2,}\s*\]|_{2,}", skeleton))
            elif target and question:
                # Lexical multiple-choice fill-in-the-blank items MUST contain EXACTLY ONE blank '____'
                blanks = re.findall(r"_{2,}", question)
                if len(blanks) == 0:
                    has_blank_when_expected = False
                elif len(blanks) > 1:
                    is_single_blank = False
                    flags.append(
                        f"❌ Quiz item '{target}': Multiple blanks ({len(blanks)}) detected in question stem (only exactly ONE blank permitted)"
                    )

                # Target Stem Leakage Gate: target word/stem cannot leak into question outside the blank
                clean_target = _clean_core(target)
                target_stem = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', clean_target)
                stem_no_blank = re.sub(r'_{2,}', ' ', question)
                stem_words = re.findall(r'\b[a-zA-Z]+\b', stem_no_blank)
                leaked_words = [
                    w for w in stem_words
                    if w.lower() == clean_target or (len(target_stem) >= 4 and re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', w.lower()) == target_stem)
                ]
                if leaked_words:
                    no_stem_leak = False
                    flags.append(
                        f"❌ Quiz item '{target}': Target word leaks verbatim into question stem outside blank: {leaked_words}"
                    )

                # Article Leakage Gate (Pre-blank 'a' or 'an' leaking answer or invalidating distractors)
                no_article_leak = True
                if options and re.search(r"\b(?:a|an)\s+_{2,}\b", question, flags=re.IGNORECASE):
                    initials = {str(opt).strip()[:1].lower() for opt in options if str(opt).strip()}
                    has_vowel_init = any(init in 'aeiou' for init in initials)
                    has_cons_init = any(init.isalpha() and init not in 'aeiou' for init in initials)
                    if has_vowel_init and has_cons_init:
                        no_article_leak = False
                        flags.append(
                            f"❌ Quiz item '{target}': Question stem has indefinite article ('a/an') immediately preceding blank with mixed vowel/consonant options (fatal test-wiseness leakage); use 'the ____', plural, or possessive instead."
                        )

                # Cross-Target Stem Leakage Gate (Zero-Tolerance Physical Gate):
                # The question stem MUST NOT leak other targets from the current batch outside the blank
                no_cross_target_leak = True
                leaked_other_targets = []
                if prompt_headwords:
                    clean_target = _clean_core(target)
                    stem_no_blank = re.sub(r'_{2,}', ' ', question)
                    stem_words_lower = {w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', stem_no_blank)}
                    for other_hw in prompt_headwords:
                        if not other_hw or other_hw == clean_target or other_hw == target.lower():
                            continue
                        if other_hw in stem_words_lower:
                            leaked_other_targets.append(other_hw)
                        else:
                            other_stem = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', other_hw)
                            if len(other_stem) >= 4 and any(
                                (len(re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', sw)) >= 4 and re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', sw) == other_stem)
                                for sw in stem_words_lower
                            ):
                                leaked_other_targets.append(other_hw)
                if leaked_other_targets:
                    no_cross_target_leak = False
                    flags.append(
                        f"❌ Quiz item '{target}': Question stem leaks other batch target word(s) outside blank: {leaked_other_targets} (cross-target leakage)!"
                    )

                # Blank Syntactic Slot & POS Agreement — moved to Level 2 (F10 rule 0).
                # Filling the target back into the stem and letting spaCy judge the slot is a
                # sentence-level judgement: it cannot be re-checked next round by the same
                # comparison, so it is not a Level-1 gate. 'grant', 'conduct' and 'accustomed'
                # each collected two or three "a noun cannot fill a verb slot" errors from it,
                # all of them false, because the POS they were judged against was guessed.
                # What Level 1 keeps is the declaration itself and its provenance:
                # blueprint > item field > lexicon guess, and a guess may only warn.
                bp_info = prompt_blueprints.get(target.lower(), {}) if prompt_blueprints else {}
                expected_pos = _safe_str(bp_info.get("part_of_speech")).strip().lower()
                pos_source = "blueprint"
                if not expected_pos:
                    expected_pos = _safe_str(item.get("part_of_speech")).strip().lower()
                    pos_source = "item field"
                if not expected_pos:
                    expected_pos = LinguisticEngine.determine_contextual_pos(target, target)
                    pos_source = "lexicon guess"
                pos_is_guess = (pos_source == "lexicon guess")

                # Anchor Presence in Question Stem Gate — deleted from Level 1 (F10 rule 1.1).
                # 'sector' declares the anchor 'manufacture'; the model wrote 'the manufacturing
                # ____', a literal search found no 'manufacture', and that one false error -
                # reported once per item and once again in the summary - consumed two of five
                # repair slots and triggered a 44-second regeneration round. The anchor remains
                # a generation-time requirement (the micro-task states it, and the blueprint now
                # checks its own anchor against its own model sentence). Level 1 no longer
                # searches for a word inside a sentence. What it does keep is the blueprint
                # compared against itself, which needs no reading: an anchor that is itself
                # another batch target is a self-contradiction the blueprint can be told about.
                req_anchor = bp_info.get("anchor", "")
                if req_anchor and req_anchor not in ("general context", "semantic context clues") and prompt_headwords:
                    for _other_hw in prompt_headwords:
                        if not _other_hw or _batch_target_conflict(target, _other_hw):
                            continue
                        if _batch_target_conflict(req_anchor, _other_hw):
                            flags.append(
                                f"⚠️ Quiz item '{target}': blueprint self-conflict — declared anchor "
                                f"'{req_anchor}' is itself another batch target ('{_other_hw}'), so no "
                                f"stem can satisfy both the anchor requirement and the leakage ban"
                            )
                            break

                # Backlog D1: Prescribed Options Contract Gate — the blueprint owns the
                # options and the answer index (CRC32-positioned distractors); the writer
                # may only author the stem, explanation and audit.
                prescribed_opts = bp_info.get("prescribed_options") or []
                prescribed_idx = bp_info.get("prescribed_index")
                if prescribed_opts and is_list_valid:
                    wanted_opts = [str(o).strip().lower() for o in prescribed_opts]
                    shipped_opts = [str(o).strip().lower() for o in options]
                    if shipped_opts != wanted_opts:
                        options_match_blueprint = False
                        substituted = sorted(set(wanted_opts) - set(shipped_opts))
                        introduced = sorted(set(shipped_opts) - set(wanted_opts))
                        flags.append(
                            f"❌ Quiz item '{target}': Prescribed options altered — blueprint prescribed "
                            f"[{', '.join(str(o) for o in prescribed_opts)}], item shipped "
                            f"[{', '.join(str(o) for o in options)}]"
                            + (f" (substituted: {substituted}; introduced: {introduced})" if substituted or introduced else "")
                        )
                    if isinstance(prescribed_idx, int) and isinstance(idx, int) and idx != prescribed_idx:
                        index_match_blueprint = False
                        flags.append(
                            f"❌ Quiz item '{target}': Prescribed answer index altered — blueprint index "
                            f"{prescribed_idx}, item shipped {idx}"
                        )


                # Explanation Grounding Gate — moved to Level 2 (F10 rule 1.3).
                # "Does the collocation this explanation claims actually land in the stem?" is a
                # reading task: the next round cannot re-run the same comparison on the same
                # strings, so by rule 0 it is not a Level-1 gate. Level 1 keeps the declaration —
                # the blueprint states the anchor, the micro-task repeats it — and the Level-2
                # expert audit (scripts/l2_expert_audit.py) is the one that reads prose.

                # Verbatim Copy Gate (P1-3): a stem may emulate a corpus model's syntax and
                # register, but it may never reuse its wording. Both corpus models are checked —
                # the LDOCE example and the curriculum quote that was strong enough to license
                # evidence. Each source is compared twice: as written, and with the target word
                # blanked out, because the classic shortcut is to mask the blueprint's own
                # example and that hole otherwise breaks every 5-gram of a short example.
                # A display-only quote is too short for an n-gram test, so reusing any part of
                # it counts as copying outright.
                no_verbatim_example = True
                verbatim_label = "dictionary example"
                verbatim_hit = ""
                stem_clean = re.sub(r"_{2,}", " ", question.lower())
                stem_tokens = re.sub(r"[^\w\s]", " ", stem_clean).split()
                stem_5grams = {tuple(stem_tokens[i:i + 5]) for i in range(len(stem_tokens) - 4)}
                for _label, _source in (
                    ("dictionary example", bp_info.get("example", "")),
                    ("curriculum quote", bp_info.get("licensed_quote", "")),
                ):
                    for src_clean in _source_token_variants(_source, target):
                        if len(src_clean) < 5 or not stem_5grams:
                            continue
                        src_5grams = {tuple(src_clean[i:i + 5]) for i in range(len(src_clean) - 4)}
                        overlap_5grams = src_5grams.intersection(stem_5grams)
                        if overlap_5grams:
                            no_verbatim_example = False
                            verbatim_label = _label
                            verbatim_hit = " ".join(next(iter(overlap_5grams)))
                            break
                    if not no_verbatim_example:
                        break
                weak_quote = (
                    bp_info.get("candidate_quote", "")
                    if bp_info.get("quote_provenance") == "weak" else ""
                )
                weak_tokens = re.sub(r"[^\w\s]", " ", weak_quote.lower()).split()
                if no_verbatim_example and weak_tokens and len(weak_tokens) < 5:
                    if " ".join(weak_tokens) in " ".join(stem_tokens):
                        no_verbatim_example = False
                        verbatim_label = "curriculum quote"
                        verbatim_hit = " ".join(weak_tokens)
                if not no_verbatim_example:
                    flags.append(
                        f"❌ Quiz item '{target}': Stem verbatim from {verbatim_label} "
                        f"(detected n-gram overlap: '{verbatim_hit}...')!"
                    )

                # Inflection Concordance Gate (P1-2)
                # The blueprint declares a verb form, the stem physically demands one, and every
                # option must be cast in it. A declared 'past tense (VBD)' target sitting in a
                # 'will ____' slot — or offered as the bare base form — makes the item unanswerable.
                inflection_agreed = True
                declared_tag = LinguisticEngine.inflection_tag_from_label(bp_info.get("inflection", ""))
                if declared_tag and expected_pos in ("verb", "v") and target and "____" in question:
                    base_verb = LinguisticEngine.lemma_of(target)
                    expected_form = LinguisticEngine.verb_form_for_tag(base_verb, declared_tag)
                    target_tag = LinguisticEngine.verb_form_tag_of(target)
                    # Concordant when the option equals the deterministically inflected lemma, or
                    # when the option itself parses as the declared form — the fallback covers
                    # irregular and dictionary-silent forms the inflector cannot reproduce.
                    form_agreed = (
                        not expected_form
                        or target.strip().lower() == expected_form
                        or target_tag == declared_tag
                    )
                    if not form_agreed:
                        inflection_agreed = False
                        flags.append(
                            f"❌ Quiz item '{target}': Inflection discordance: blueprint declares "
                            f"'{bp_info.get('inflection')}' (expected form '{expected_form}') "
                            f"but the answer option is '{target}'!"
                        )
                    # 'does the stem slot contradict the declared form?' — deleted from Level 1
                    # (F10 rule 1.4). The auxiliary table below read the finished sentence and
                    # guessed which form the slot demanded; a detector built that way covers
                    # 'is currently ____' and misses 'has been ____', 'to have ____' and
                    # 'the report that the firm ____ last year'. What Level 1 keeps is the pure
                    # form comparison above: the tag the blueprint declared against the form the
                    # answer option actually carries, plus the option-set form check below.
                    # Parallel inflection across options (soft warning): a mixed-form option set
                    # lets the item be solved by morphology instead of meaning.
                    if declared_tag in ("VBD", "VBG", "VBZ", "VBN", "VB", "VBP"):
                        form_family = {"VBD", "VBN"} if declared_tag in ("VBD", "VBN") else {declared_tag}
                        for o_idx, opt in enumerate(options):
                            if o_idx == idx:
                                continue
                            opt_l = str(opt).strip().lower()
                            opt_tag = LinguisticEngine.verb_form_tag_of(opt_l)
                            if opt_tag and LinguisticEngine.lemma_of(opt_l) != opt_l and opt_tag not in form_family:
                                flags.append(
                                    f"⚠️ Quiz item '{target}': distractor '{opt_l}' carries verb form "
                                    f"{opt_tag} while the blueprint declares '{bp_info.get('inflection')}'"
                                )

                # Distractor Slot Legality Gate (P1-1)
                # Every option must be able to occupy the target's own syntactic slot;
                # 'log in to a ____' offering the adverb 'somehow' is solved by morphology
                # instead of meaning. Only positive dictionary evidence vetoes an option.
                if is_list_valid and valid_idx and expected_pos:
                    requires_obj = (
                        expected_pos in ("verb", "v")
                        and LinguisticEngine.verb_takes_object(LinguisticEngine.lemma_of(target)) is True
                    )
                    for o_idx, opt in enumerate(options):
                        if o_idx == idx:
                            continue
                        opt_l = str(opt).strip().lower()
                        if not opt_l or opt_l == str(target).strip().lower():
                            continue
                        if not LinguisticEngine.distractor_occupies_slot(
                            opt_l, expected_pos, requires_object=requires_obj
                        ):
                            illegal_distractors.append(opt_l)
                if illegal_distractors:
                    # Downgraded from fatal to a report (F10 改哪几处 2). The check itself is a
                    # pure dictionary comparison, so it is a legitimate Level-1 finding and it is
                    # re-checkable next round — but the options are blueprint-owned (Backlog D1),
                    # so the writer is forbidden from changing them. A ❌ here therefore triggers a
                    # regeneration that is not allowed to fix the thing it was called for. It is
                    # reported for human review instead of capping the score.
                    _pos_note = (
                        " [POS inferred from the lexicon — needs human review]"
                        if pos_is_guess else ""
                    )
                    flags.append(
                        f"⚠️ Quiz item '{target}': Distractor slot illegality: "
                        f"{illegal_distractors} cannot occupy the {expected_pos} slot the "
                        f"target '____' occupies{_pos_note} (blueprint-owned options)"
                    )

                # Stem Length Check (Physical Sanity Gate - Soft Warning)
                # Ensure stems provide sufficient context (> 3 words) without forcing artificial clausal complexity.
                if len(stem_words) < 4:
                    is_adequate_complexity = False
                    flags.append(f"⚠️ Quiz item '{target}': Inadequate context stem (< 4 words)")

            # 5. In-List Distractor Recycling check (Zero-Tolerance Level 1 Gate)
            recycled_in_distractors = []
            if prompt_headwords and is_list_valid and valid_idx:
                clean_target = _clean_core(target) if target else ""
                clean_hw_set = {_clean_core(hw) for hw in prompt_headwords if hw}

                for o_idx, opt in enumerate(options):
                    if o_idx == idx:
                        continue
                    opt_clean = str(opt).strip().lower()
                    opt_core = _clean_core(opt_clean)

                    # If this option is derived from the item's own target keyword (e.g. preserves, preserving, to preserve),
                    # it is a legitimate morphological/syntactic distractor trap, NOT recycling an external word!
                    if clean_target and (
                        opt_core == clean_target or 
                        clean_target in opt_clean or 
                        (len(clean_target) >= 4 and opt_clean.startswith(clean_target[:4]))
                    ):
                        continue

                    # Now check if it recycles another DIFFERENT headword from the study list
                    if opt_clean in prompt_headwords or (opt_core and opt_core in clean_hw_set and opt_core != clean_target):
                        recycled_in_distractors.append(opt_clean)
                    elif not is_translation:
                        # For vocabulary multiple-choice, check suffix-stripped stems
                        opt_stem = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', opt_clean)
                        for hw in prompt_headwords:
                            hw_clean = str(hw).strip().lower()
                            if clean_target and (hw_clean == clean_target or _clean_core(hw_clean) == clean_target):
                                continue
                            hw_stem = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', hw_clean)
                            if (len(hw_stem) >= 4 and len(opt_stem) >= 4 and hw_stem == opt_stem) or \
                               (len(hw_clean) >= 5 and opt_clean.startswith(hw_clean[:5])) or \
                               (len(opt_clean) >= 5 and hw_clean.startswith(opt_clean[:5])):
                                recycled_in_distractors.append(opt_clean)
                                break

            no_in_list_recycling = len(recycled_in_distractors) == 0

            # Level-1 pass list (F10 rule 0). Every term below is a field-to-field or
            # string-to-string comparison that the next round can re-run identically. The four
            # sentence-level terms that used to sit in this list — blank POS agreement, anchor
            # presence in the stem, explanation grounding, distractor slot legality — are gone:
            # the first three belong to the Level-2 expert audit, the fourth is blueprint-owned
            # and is reported as a warning instead of capping the score.
            item_passed = (
                is_list_valid and
                has_no_duplicates and
                valid_idx and
                target_in_options and
                explanation and
                question and
                has_blank_when_expected and
                is_single_blank and
                no_stem_leak and
                no_cross_target_leak and
                no_article_leak and
                no_verbatim_example and
                inflection_agreed and
                no_in_list_recycling and
                options_match_blueprint and
                index_match_blueprint
            )

            if item_passed:
                passes += 1
            else:
                reasons = []
                if not is_list_valid: reasons.append(f"options count != {expected_opt_count}")
                elif not has_no_duplicates: reasons.append("duplicate options detected")
                if not valid_idx: reasons.append("invalid answer index")
                if not target_in_options: reasons.append(f"target '{target}' not matching options[{idx}]")
                if not explanation: reasons.append("missing explanation")
                if not question: reasons.append("missing question text")
                if not has_blank_when_expected: reasons.append("missing fill-in-the-blank slot (____)")
                if not is_single_blank: reasons.append("multiple blanks detected in stem")
                if not no_stem_leak: reasons.append("target word leaks into question stem")
                if not no_cross_target_leak: reasons.append(f"cross-target leakage {leaked_other_targets}")
                if not no_article_leak: reasons.append("indefinite article leakage before blank")
                if not no_verbatim_example: reasons.append(f"stem verbatim from {verbatim_label}")
                if not inflection_agreed: reasons.append(f"inflection discordance with blueprint '{bp_info.get('inflection')}'")
                if not options_match_blueprint:
                    reasons.append(f"prescribed options altered from blueprint {bp_info.get('prescribed_options')}")
                if not index_match_blueprint:
                    reasons.append(f"prescribed answer index altered from blueprint index {bp_info.get('prescribed_index')}")

                if not no_in_list_recycling:
                    reasons.append(f"recycles study list headwords in distractors {recycled_in_distractors}")
                    flags.append(f"❌ Quiz item '{target or question[:25]}' in-list distractor recycling: {recycled_in_distractors}")
                elif not has_blank_when_expected:
                    flags.append(f"❌ Quiz item '{target}': missing fill-in-the-blank slot (____)")
                elif not is_single_blank:
                    flags.append(f"❌ Quiz item '{target}': Multiple blanks detected in stem")
                elif not no_stem_leak:
                    flags.append(f"❌ Quiz item '{target}': Target word leaks into question stem: {leaked_words}")
                elif not no_cross_target_leak:
                    flags.append(f"❌ Quiz item '{target}': cross-target leakage: {leaked_other_targets}")
                elif not no_article_leak:
                    flags.append(f"❌ Quiz item '{target}': indefinite article leakage before blank")
                elif not no_verbatim_example:
                    flags.append(f"❌ Quiz item '{target}': Stem verbatim from {verbatim_label}")
                elif not inflection_agreed:
                    flags.append(f"❌ Quiz item '{target}': Inflection discordance with blueprint declaration")
                else:
                    flags.append(f"⚠️ Quiz question failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "summary":
            name = _safe_str(item.get("concept_name"))
            sig = _safe_str(item.get("educational_significance"))
            details = item.get("key_details", [])
            sub_concepts = item.get("sub_concepts", [])
            checks += 1
            # If sub_concepts are present, validate their internal structure
            sub_ok = True
            if isinstance(sub_concepts, list) and sub_concepts:
                for sub in sub_concepts:
                    if not isinstance(sub, dict):
                        sub_ok = False
                        break
                    s_name = _safe_str(sub.get("sub_concept_name"))
                    s_sig = _safe_str(sub.get("significance_or_takeaway"))
                    s_pts = sub.get("key_points", [])
                    if not (s_name and s_sig and isinstance(s_pts, list) and len(s_pts) > 0):
                        sub_ok = False
                        break
            if name and sig and isinstance(details, list) and len(details) > 0 and sub_ok:
                passes += 1
            else:
                reason = "missing educational significance or key details" if not (name and sig and isinstance(details, list) and len(details) > 0) else "malformed sub_concepts"
                flags.append(f"⚠️ Concept '{name}' {reason}")
        elif task_type == "mindmap":
            name = _safe_str(item.get("branch_name"))
            checks += 1
            if name:
                passes += 1
            else:
                flags.append("⚠️ MindMap branch missing branch name")
        elif task_type == "expert_audit":
            checks += 1
            raw_distractors = item.get("distractors", [])
            # Filter out empty or non-dict objects (e.g. trailing {} produced during JSON truncation)
            distractors = [
                d for d in raw_distractors 
                if isinstance(d, dict) and any(d.get(k) for k in ("option_text", "trap_type", "elimination_rationale"))
            ] if isinstance(raw_distractors, list) else []
            bs_idx = item.get("blind_solved_index")
            feedback = _safe_str(item.get("diagnostic_feedback"))
            score = item.get("pedagogical_score", 100)
            single_valid = item.get("single_fit_valid", True)
            is_valid_dist = len(distractors) in (1, 2, 3, 4)
            is_valid_idx = isinstance(bs_idx, int) and -1 <= bs_idx <= 3
            # Feedback is required only if the item has defects (score < 90 or invalid single fit);
            # for flawless items (score >= 90), empty feedback is valid and expected.
            feedback_ok = bool(feedback) if (score < 90 or not single_valid) else True
            triage = item.get("triage_action", "PASS")
            cured_q = item.get("cured_question")
            is_valid_triage = triage in ("PASS", "REPAIR", "REWRITE")
            cured_ok = bool(cured_q) if triage in ("REPAIR", "REWRITE") else True
            if is_valid_dist and is_valid_idx and feedback_ok and is_valid_triage and cured_ok:
                passes += 1
            else:
                reasons = []
                if not is_valid_dist: reasons.append(f"distractor count != 1 to 4 (got {len(distractors) if isinstance(distractors, list) else 0})")
                if not is_valid_idx: reasons.append("invalid blind solve index")
                if not feedback_ok: reasons.append("missing diagnostic feedback for defective item")
                if not is_valid_triage: reasons.append(f"invalid triage_action '{triage}'")
                if not cured_ok: reasons.append(f"missing cured_question for triage '{triage}'")
                flags.append(f"⚠️ Audit item failed validation: {', '.join(reasons)}")
                
    if checks == 0:
        return 0.0, [f"⚠️ Output list for '{task_type}' is empty."]
    
    # Check for copy-pasted/homogenized definitions in vocabulary
    if task_type in ("vocabulary", "expressions"):
        definitions = [_clean_core(item.get("definition", "")) for item in items if item.get("definition")]
        if len(definitions) > 1:
            dup_defs = len(definitions) - len(set(definitions))
            if dup_defs > 0:
                flags.append(f"❌ Found {dup_defs} duplicate or copy-pasted definition(s) across different terms")
                passes = max(0, passes - dup_defs)

    raw_score = round((passes / checks) * W_PEDAGOGY, 1)
    # Apply soft deduction for auto-remapped grammar categories (2.0 pts per remapped item)
    if task_type == "grammar":
        remap_count = sum(1 for f in flags if "Deterministic Auto-Remap" in f)
        if remap_count > 0:
            raw_score = max(0.0, raw_score - (remap_count * 2.0))

    return max(0.0, raw_score), flags


def _score_uniqueness(items: List[Dict[str, Any]], task_type: str, user_prompt: str = "") -> Tuple[Optional[float], List[str]]:
    """Dimension 4 (0–20). Deduplicate by the task's primary identifying field."""
    if not items:
        return 0.0, ["❌ No items to evaluate for uniqueness"]
    key_by_type = {
        "vocabulary": ("word",),
        "expressions": ("word",),
        "grammar": ("quote", "pattern_formula"),
        "quiz": ("target_word", "question", "translated_sentence", "correct_english_answer"),
        "summary": ("concept_name",),
        "mindmap": ("branch_name",),
        "expert_audit": ("item_index",),
    }
    keys = key_by_type.get(task_type, ("word", "quote", "question"))
    headwords = []
    for item in items:
        if task_type == "grammar":
            # Composite identity for grammar: quote is primary, formula is secondary
            q_val = str(item.get("quote") or "").strip().lower()
            f_val = str(item.get("pattern_formula") or "").strip().lower()
            if q_val:
                headwords.append(q_val)
            elif f_val:
                headwords.append(f_val)
            continue
        for key in keys:
            value = item.get(key)
            if value:
                headwords.append(str(value).strip().lower())
                break
    if not headwords:
        return 0.0, ["⚠️ Missing identifying keys for uniqueness check"]

    unique_flags = []
    unique = len(set(headwords))
    earned = W_UNIQUENESS
    if unique < len(headwords):
        dup = len(headwords) - unique
        # Stricter deduction: duplicate items heavily reduce score
        ratio = unique / len(headwords)
        earned = 0.0 if ratio < 0.8 else round(ratio * W_UNIQUENESS, 1)
        unique_flags.append(f"❌ Found {dup} duplicate item(s)")

    # For vocabulary: audit morphological word family collisions (e.g. 'recognize' vs 'recognition')
    if task_type == "vocabulary" and headwords:
        family_clusters: List[List[str]] = []
        for hw in headwords:
            clean_hw = re.sub(r'\[.*?\]|\(.*?\)', '', str(hw)).strip().lower()
            if not clean_hw:
                continue
            placed = False
            for fc in family_clusters:
                if any(LinguisticEngine.are_same_word_family(clean_hw, fchw) for fchw in fc):
                    fc.append(clean_hw)
                    placed = True
                    break
            if not placed:
                family_clusters.append([clean_hw])
        
        mult_clusters = [fc for fc in family_clusters if len(fc) > 1]
        if mult_clusters:
            coll_details = ", ".join(f"[{'/'.join(c)}]" for c in mult_clusters[:3])
            unique_flags.append(f"❌ Morphological word-family collisions detected: {coll_details}")
            # Penalize uniqueness dimension
            earned = max(0.0, earned - len(mult_clusters) * 3.0)

    # For grammar: audit duplicate common_mistakes across patterns
    if task_type == "grammar":
        all_mistakes = [str(it.get("common_mistakes") or "").strip() for it in items if str(it.get("common_mistakes") or "").strip()]
        if len(all_mistakes) > len(set(all_mistakes)):
            from collections import Counter
            m_counts = Counter(all_mistakes)
            dup_mistakes = [m for m, c in m_counts.items() if c > 1]
            if dup_mistakes:
                short_dup = [f"'{m[:40]}...' ({m_counts[m]}x)" for m in dup_mistakes[:2]]
                unique_flags.append(f"❌ Found duplicate common_mistakes across grammar patterns: {', '.join(short_dup)}")
                earned = max(0.0, earned - len(dup_mistakes) * 3.0)

    # For quizzes: penalize identical distractors recycled repeatedly across different questions
    if task_type == "quiz":
        all_distractors = []
        for item in items:
            opts = item.get("options", [])
            idx = item.get("correct_answer_index")
            if isinstance(opts, list) and isinstance(idx, int):
                for o_i, o in enumerate(opts):
                    if o_i != idx:
                        all_distractors.append(str(o).strip().lower())
        if all_distractors:
            from collections import Counter
            counts = Counter(all_distractors)
            repeated = {w: c for w, c in counts.items() if c >= 3 and len(w) >= 3}
            if repeated:
                rep_details = ", ".join(f"'{w}' ({c}x)" for w, c in list(repeated.items())[:3])
                unique_flags.append(f"❌ Distractors recycled repeatedly across quiz: {rep_details}")
    return earned, unique_flags


def prune_hallucinated_items(parsed_data: Any, user_prompt: str, task_type: str = "") -> Tuple[Any, List[str]]:
    """Deterministic Level 1 Hallucination Pruning Gate.
    
    If the model extracts vocabulary/expressions where the headword neither appears
    in its quoted sentence nor in the source text, it is a pure hallucination.
    Rather than blindly looping retries that ask the model to fix nonexistent words,
    we surgically prune the hallucinated items deterministically.
    
    Returns:
        (pruned_parsed_data, list_of_pruned_flags)
    """
    if not isinstance(parsed_data, dict):
        return parsed_data, []
    
    if not task_type or task_type == "unknown" or task_type not in ("vocabulary", "expressions", "grammar"):
        detected = _detect_task_type(task_type or "", parsed_data)
        if detected in ("vocabulary", "expressions", "grammar"):
            task_type = detected
        else:
            return parsed_data, []

    items = _extract_items(parsed_data, task_type)
    if not items:
        return parsed_data, []

    source = _extract_source_content(user_prompt)
    core_src = _clean_core(source)
    if not core_src:
        return parsed_data, []

    # Initialize spaCy sentence pool for deterministic boundary snapping
    _, sentence_pool = LinguisticEngine.tokenize_and_index_sentences(source)

    key_by_type = {
        "vocabulary": "vocabulary",
        "expressions": "expressions",
        "grammar": "grammar_patterns",
    }
    array_key = key_by_type.get(task_type)
    if not array_key or array_key not in parsed_data or not isinstance(parsed_data[array_key], list):
        return parsed_data, []

    surviving_items = []
    pruned_flags = []
    seen_dedup_keys = set()

    for item in parsed_data[array_key]:
        if not isinstance(item, dict):
            surviving_items.append(item)
            continue

        # In-Place Self-Healing: Quote boundary magnetic snapping using sentence pool
        # Snaps [S-id] or partial quote with '...' to the pristine, authentic source sentence
        quote_field = "quoted_sentence" if "quoted_sentence" in item else ("quote" if "quote" in item else None)
        if quote_field and item.get(quote_field):
            raw_q = str(item[quote_field]).strip()
            snapped = LinguisticEngine.snap_to_sentence_pool(raw_q, sentence_pool)
            if snapped and snapped != raw_q:
                item[quote_field] = snapped
                pruned_flags.append(f"ℹ️ In-Place Self-Healing: Snapped quote '{raw_q[:30]}...' -> full authentic sentence")
            else:
                # Strip internal [S-id] prefix if quote was verbatim with S-id attached
                cleaned_q = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", raw_q, flags=re.IGNORECASE).strip()
                item[quote_field] = cleaned_q

        # In-Place Quote Repair: the quoted sentence MUST contain the headword.
        # If the model cited the wrong sentence (headword physically present in the
        # source text but absent from the quote), deterministically swap in the
        # shortest authentic source sentence containing it — instead of burning LLM
        # retries that cannot reliably self-correct this failure mode.
        if task_type in ("vocabulary", "expressions"):
            headword = str(item.get("word") or item.get("pattern_formula") or item.get("target_word") or "").strip()
            if headword and quote_field and sentence_pool:
                rw_clean_word = _clean_core(re.sub(r"\[.*?\]|\(.*?\)", " ", headword))
                rw_tokens = [w for w in rw_clean_word.split() if w and w not in STOP_SLOTS]
                if rw_tokens:
                    def _rw_tok_in(text_core: str, tok: str) -> bool:
                        return _form_in_text(tok, text_core)

                    rw_current_core = _clean_core(str(item.get(quote_field) or ""))
                    if rw_current_core:
                        rw_matched = sum(1 for t in rw_tokens if _rw_tok_in(rw_current_core, t))
                        rw_min_needed = max(1, len(rw_tokens) // 2 + (1 if len(rw_tokens) % 2 == 1 else 0))
                        rw_needs_repair = rw_matched < rw_min_needed
                    else:
                        rw_needs_repair = True

                    # Only repair when the headword physically exists in the source;
                    # otherwise it is a pure hallucination and the prune gate handles it.
                    if rw_needs_repair and all(_rw_tok_in(core_src, t) for t in rw_tokens):
                        rw_best, rw_best_len = None, None
                        for _sid, _sent in sentence_pool.items():
                            _sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", _sent, flags=re.IGNORECASE).strip()
                            _sc = _clean_core(_sent_clean)
                            if not _sc or not all(_rw_tok_in(_sc, t) for t in rw_tokens):
                                continue
                            if rw_best_len is None or len(_sent_clean) < rw_best_len:
                                rw_best, rw_best_len = _sent_clean, len(_sent_clean)
                        if rw_best:
                            item[quote_field] = rw_best
                            pruned_flags.append(
                                f"🩹 In-Place Quote Repair: headword '{headword}' was absent from the quoted sentence; "
                                f"swapped in authentic source sentence: '{rw_best[:60]}...'"
                            )

        # Deterministic deduplication check
        if task_type == "grammar":
            raw_quote = str(item.get("quote", "")).strip()
            # 1. Physical sentence-level invariant dedup (Sentence pool ID)
            sent_id = LinguisticEngine.get_sentence_id(raw_quote, sentence_pool)
            if sent_id:
                sent_dedup_key = f"sent_id:{sent_id}"
                if sent_dedup_key in seen_dedup_keys:
                    pruned_flags.append(f"✂️ Pruned duplicate item: duplicate sentence {sent_id} ({raw_quote[:35]}...)")
                    continue
                seen_dedup_keys.add(sent_dedup_key)

            # 2. Dependency Syntax Signature Dedup (Macro domain + Dep type + Core anchor)
            if not item.get("category"):
                classified_cat = LinguisticEngine.classify_grammar_dependency(raw_quote)
                item["category"] = classified_cat or "Information Packaging"
            cat_val = str(item.get("category", "")).strip()
            fp = LinguisticEngine.extract_grammar_fingerprint(raw_quote, category=cat_val)
            if fp[1] != "generic" and fp[2]:
                fp_key = f"syntax_fp:{fp[0]}:{fp[1]}:{fp[2]}"
                if fp_key in seen_dedup_keys:
                    pruned_flags.append(f"✂️ Pruned duplicate item: homogeneous grammar pattern ({fp[0]} -> {fp[1]}:{fp[2]}): {raw_quote[:35]}...")
                    continue
                seen_dedup_keys.add(fp_key)

            formula_val = str(item.get("pattern_formula", "")).strip().lower()
            quote_val = raw_quote.lower()
            dedup_key = (formula_val, quote_val)
        else:
            word_val = str(item.get("word", "")).strip().lower()
            clean_word_val = re.sub(r'\[.*?\]|\(.*?\)', '', word_val).strip()
            dedup_key = (clean_word_val,)
            
            # Word-Family Deduplication Gate: reject items belonging to an already seen word family
            if task_type == "vocabulary" and clean_word_val:
                is_family_dup = False
                for seen_k in seen_dedup_keys:
                    if isinstance(seen_k, tuple) and seen_k and isinstance(seen_k[0], str):
                        prev_w = seen_k[0]
                        if prev_w and LinguisticEngine.are_same_word_family(clean_word_val, prev_w):
                            pruned_flags.append(f"✂️ Pruned duplicate word-family item '{clean_word_val}' (subsumed by '{prev_w}')")
                            is_family_dup = True
                            break
                if is_family_dup:
                    continue
            
        if any(dedup_key) and dedup_key in seen_dedup_keys:
            pruned_flags.append(f"✂️ Pruned duplicate item: {dedup_key[0]}")
            continue
        if any(dedup_key):
            seen_dedup_keys.add(dedup_key)

        word = str(item.get("word") or item.get("pattern_formula") or "").strip()
        quote = str(item.get("quoted_sentence") or item.get("quote") or "").strip()
        
        if not word and not quote:
            continue

        # Specialized pruning for grammar
        if task_type == "grammar":
            core_quote = _clean_core(quote)
            imitation = str(item.get("imitation_example") or "").strip()
            mistakes = str(item.get("common_mistakes") or "").strip()

            if not (core_quote and imitation and mistakes):
                missing_fields = []
                if not core_quote: missing_fields.append("quote")
                if not imitation: missing_fields.append("imitation_example")
                if not mistakes: missing_fields.append("common_mistakes")
                pruned_flags.append(f"✂️ Pruned incomplete grammar pattern '{word[:30]}' (missing required: {', '.join(missing_fields)})")
                continue

            # Auto-hydrate category if missing from simplified schema
            if not item.get("category"):
                classified_cat = LinguisticEngine.classify_grammar_dependency(quote)
                item["category"] = classified_cat or "Information Packaging"

            # Quote must exist in source text (verbatim or high n-gram coverage)
            quote_in_src = (core_quote in core_src or _ngram_coverage(core_quote, core_src, n=3) >= 0.80)
            if not quote_in_src:
                # Check segment coverage if ellipsis is present
                if "..." in quote or "…" in quote:
                    segments = [s.strip() for s in re.split(r'\.{3,}|…', quote) if s.strip()]
                    meaningful_segs = [s for s in segments if len(_clean_core(s).split()) >= 2]
                    if meaningful_segs and all(_clean_core(seg) in core_src or _ngram_coverage(_clean_core(seg), core_src, n=3) >= 0.80 for seg in meaningful_segs):
                        quote_in_src = True

            if not quote_in_src:
                pruned_flags.append(f"✂️ Pruned hallucinated grammar pattern '{word[:30]}' (quote not found in source text)")
                continue

            # Anchor verification: literal functional anchors must exist in quote
            raw_lits = re.findall(r"[a-zA-Z0-9_\'/]+", re.sub(r"\[.*?\]|\(.*?\)", " ", word))
            expanded_quote = _clean_core(_expand_contractions(quote))
            quote_tokens = set(core_quote.split()) | set(expanded_quote.split())
            missing_anchors = []
            for lit in raw_lits:
                alts = [a.strip() for a in lit.split("/") if a.strip()]
                valid_alts = [a for a in alts if len(_clean_core(a)) >= 2 and _clean_core(a) not in COBUILD_POS_TOKENS]
                if not valid_alts:
                    continue
                if not any(_clean_core(a) in quote_tokens for a in valid_alts):
                    missing_anchors.append(lit)
            
            if missing_anchors:
                # Level 1 In-Place Self-Healing: Attempt to synthesize canonical formula from quote
                canonical_formula = LinguisticEngine.generate_cobuild_formula(quote, category=item.get("category"))
                if canonical_formula and canonical_formula != word and canonical_formula != "[Subject] + [VP] + [Clause]":
                    item["pattern_formula"] = canonical_formula
                    pruned_flags.append(f"ℹ️ In-Place Self-Healing: Repaired mismatched formula '{word[:30]}' -> '{canonical_formula}' (anchors {missing_anchors} not in quote)")
                    word = canonical_formula
                else:
                    pruned_flags.append(f"✂️ Pruned mismatched grammar pattern '{word[:30]}' (anchor '{missing_anchors[0]}' missing from quote)")
                    continue

            # Level 1 Code Gate: Auto-remap explicit category mismatches
            item, remap_notice = auto_remap_grammar_category(item)
            if remap_notice:
                pruned_flags.append(remap_notice)

            # Level 1 Code Gate: Auto-repair/canonicalize COBUILD formula if trivial or missing
            current_formula = str(item.get("pattern_formula", "")).strip()
            if not current_formula or "[" not in current_formula or current_formula.lower() in ("the + [noun] + [vp]", "[subject] + [vp]"):
                canonical_formula = LinguisticEngine.generate_cobuild_formula(quote, category=item.get("category"))
                if canonical_formula and canonical_formula != current_formula:
                    item["pattern_formula"] = canonical_formula
                    pruned_flags.append(f"ℹ️ In-Place Self-Healing: Standardized COBUILD formula -> '{canonical_formula}'")

            surviving_items.append(item)
            continue

        clean_word_no_slots = re.sub(r"\[.*?\]|\(.*?\)", " ", word)
        clean_word = _clean_core(clean_word_no_slots)
        clean_quote = _clean_core(quote)
        word_tokens = [w for w in clean_word.split() if w and w not in STOP_SLOTS]

        # Check A: Does the word appear in its claimed quote?
        def _tok_matches(tok: str, target_text: str) -> bool:
            return _form_in_text(tok, target_text)

        if word_tokens:
            min_needed = max(1, len(word_tokens) // 2 + (1 if len(word_tokens) % 2 == 1 else 0))
            word_in_quote = (sum(1 for w in word_tokens if _tok_matches(w, clean_quote)) >= min_needed)
            word_in_source = (sum(1 for w in word_tokens if _tok_matches(w, core_src)) >= min_needed)
        else:
            word_in_quote = (clean_word in clean_quote)
            word_in_source = (clean_word in core_src)

        # If it's absent from quote AND absent from entire source text -> pure fabrication
        if not word_in_quote and not word_in_source:
            pruned_flags.append(f"✂️ Pruned hallucinated item '{word}' (not present in quoted sentence or source text)")
            continue

        # If it's absent from its claimed quote but the quote itself is not in source
        if not word_in_quote:
            core_quote = _clean_core(quote)
            if core_quote and (core_quote not in core_src and _ngram_coverage(core_quote, core_src, n=3) < 0.85):
                pruned_flags.append(f"✂️ Pruned hallucinated item '{word}' (unmatched word in unverified quote)")
                continue

        # Level 1 In-Place Self-Healing: Canonical Lemmatization for single-word vocabulary
        if task_type == "vocabulary" and "word" in item:
            orig_word = str(item["word"]).strip()
            # If word is inflected (ends with -ed, -ing, -s) and does not contain brackets
            if orig_word and "[" not in orig_word and "(" not in orig_word:
                lemmatized = LinguisticEngine.lemmatize_headword(orig_word, quote)
                if lemmatized and lemmatized.lower() != orig_word.lower():
                    item["word"] = lemmatized
                    pruned_flags.append(f"ℹ️ In-Place Self-Healing: Lemmatized headword '{orig_word}' -> '{lemmatized}'")

        surviving_items.append(item)

    if pruned_flags:
        parsed_data = dict(parsed_data)
        parsed_data[array_key] = surviving_items
    return parsed_data, pruned_flags


class LogEvaluator:
    """
    Evaluates LLM execution logs across 4 pedagogical quality dimensions:
    1. Schema & Structural Adherence (25%)
    2. Source Faithfulness & Verbatim Verification (30%)
    3. Pedagogical & Slot-Filling Quality (25%)
    4. Uniqueness & Deduplication (20%)

    Scoring convention:
    - Dimensions returning None (N/A) are excluded from composite calculation.
    - Composite Score is strictly normalized to 0–100 based on applicable dimensions.
    """

    # Cache & thread-lock for repeated audits (dashboard polls /api/hero-board).
    _AUDIT_LOCK = threading.Lock()
    _AUDIT_CACHE: Dict[str, Any] = {"key": None, "value": None}

    @classmethod
    def parse_log_file(cls, log_path: Path) -> Optional[Dict[str, Any]]:
        """Parses a single .log file and extracts metadata, status, prompt content, and JSON response."""
        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        task_match = re.search(r"=== TASK:\s*(.+?)\s*===", content)
        model_match = re.search(r"=== MODEL:\s*(.+?)\s*===", content)
        time_match = re.search(r"=== TIMESTAMP:\s*(.+?)\s*===", content)
        status_match = re.search(r"=== STATUS:\s*(.+?)\s*===", content)
        fail_cat_match = re.search(r"=== FAILURE_CATEGORY:\s*(.+?)\s*===", content)

        if not task_match or not model_match:
            return None

        task = task_match.group(1).strip()
        model = model_match.group(1).strip()
        timestamp = time_match.group(1).strip() if time_match else ""
        status = status_match.group(1).strip() if status_match else "SUCCESS"
        failure_category = fail_cat_match.group(1).strip() if fail_cat_match else None

        pre_cure_match = re.search(r"=== PRE_CURE_SCORE:\s*([\d\.]+)%\s*===", content)
        pre_cure_score = float(pre_cure_match.group(1)) if pre_cure_match else None

        post_cure_match = re.search(r"=== POST_CURE_SCORE:\s*([\d\.]+)%\s*===", content)
        post_cure_score = float(post_cure_match.group(1)) if post_cure_match else None

        # Extract SYSTEM PROMPT content block (fallback context)
        system_prompt = ""
        system_prompt_match = re.search(r"--- SYSTEM PROMPT ---\n(.*?)(?=\n--- USER PROMPT ---|\n--- RAW RESPONSE ---|\n===|\Z)", content, re.DOTALL)
        if system_prompt_match:
            system_prompt = system_prompt_match.group(1).strip()

        # Extract USER PROMPT content block
        user_prompt = ""
        user_prompt_match = re.search(r"--- USER PROMPT ---\n(.*?)(?=\n--- RAW RESPONSE ---|\n===|\Z)", content, re.DOTALL)
        if user_prompt_match:
            user_prompt = user_prompt_match.group(1).strip()

        # Extract RAW RESPONSE
        raw_response = ""
        response_match = re.search(r"--- RAW RESPONSE ---\n(.*?)(?=\n===|\Z)", content, re.DOTALL)
        if response_match:
            raw_response = response_match.group(1).strip()

        # Parse JSON (direct, fenced ```json, or first balanced brace)
        parsed_json = _extract_json(raw_response)

        return {
            "log_name": log_path.name,
            "task": task,
            "model": model,
            "status": status,
            "failure_category": failure_category,
            "timestamp": timestamp,
            "pre_cure_score": pre_cure_score,
            "post_cure_score": post_cure_score,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_json": parsed_json,
        }

    @classmethod
    def evaluate_log(cls, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the 4-dimension audit on a parsed log entry."""
        log_name = log_data.get("log_name", "")
        task = log_data.get("task", "")
        model = log_data.get("model", "")
        status = log_data.get("status", "SUCCESS")
        failure_category = log_data.get("failure_category")
        pre_cure_score = log_data.get("pre_cure_score")
        post_cure_score = log_data.get("post_cure_score")
        user_prompt = log_data.get("user_prompt", "")
        system_prompt = log_data.get("system_prompt", "")
        context_prompt = log_data.get("context_prompt", "") or system_prompt
        parsed = log_data.get("parsed_json")

        raw_response = log_data.get("raw_response", "")
        task_type = _detect_task_type(task, parsed)
        items = _extract_items(parsed, task_type)

        flags: List[str] = []
        schema_score, f1 = _score_schema(parsed, raw_response, task_type=task_type)
        verbatim_score, f2 = _score_verbatim(items, task_type, user_prompt, context_prompt=context_prompt)
        effective_prompt = user_prompt if user_prompt.strip() else context_prompt
        pedagogy_score, f3 = _score_pedagogy(items, task_type, user_prompt=effective_prompt)
        uniqueness_score, f4 = _score_uniqueness(items, task_type, user_prompt=effective_prompt)
        flags.extend(f1 + f2 + f3 + f4)

        # Dimension scores & weights map
        dim_results = [
            ("schema_adherence", schema_score, W_SCHEMA),
            ("verbatim_faithfulness", verbatim_score, W_VERBATIM),
            ("pedagogical_quality", pedagogy_score, W_PEDAGOGY),
            ("uniqueness", uniqueness_score, W_UNIQUENESS),
        ]

        applicable_weight = sum(w for _, score, w in dim_results if score is not None)
        earned_score = sum(score for _, score, _ in dim_results if score is not None)

        if applicable_weight > 0:
            composite_score = round((earned_score / applicable_weight) * 100.0, 1)
        else:
            composite_score = 0.0

        # Deterministic Target Coverage Gate: check if model delivered all requested target items
        if task_type in ("grammar", "expressions", "vocabulary"):
            skeletons = _extract_target_skeletons(effective_prompt, task_type)
            expected_count = len(skeletons)
            if expected_count > 0:
                delivered_count = len(items)
                if delivered_count < expected_count:
                    coverage_ratio = delivered_count / expected_count
                    # Detect whether targets come from auto-mining (AWL suggestions) vs teacher-provided syllabus
                    # "DETERMINISTIC TARGET VOCABULARY" = auto-mined AWL words (soft recommendations)
                    # "DETERMINISTIC TARGET PATTERNS"   = in-text grammar patterns (strict, they ARE in the text)
                    is_auto_mined = "DETERMINISTIC TARGET VOCABULARY" in (effective_prompt or "")

                    # Name the missing targets so the retry critique can target them specifically
                    name_key = "word" if task_type == "vocabulary" else "formula"
                    missing_names: List[str] = []
                    if name_key:
                        delivered_names = {
                            _clean_core(str(it.get(name_key) or it.get("word") or it.get("pattern_formula") or "")).lower()
                            for it in items
                        }
                        for sk in skeletons:
                            sk_name = str(sk.get(name_key) or "").strip()
                            if sk_name and _clean_core(sk_name).lower() not in delivered_names:
                                missing_names.append(sk_name)
                    missing_txt = f" — missing: {', '.join(missing_names[:12])}" if missing_names else ""

                    target_source_desc = "Auto-mined" if is_auto_mined else "Syllabus"
                    # Unified Target Coverage Gate (85% tolerance):
                    # Accounts for local LLM (e.g. gemma4) tendency to occasionally deliver e.g. 19/20 or 4/5 items.
                    # >= 85% coverage: Informational warning (non-fatal), slight score adjustment (at most 10%).
                    # < 85% coverage: FATAL defect (❌ [INCOMPLETE_COVERAGE]), triggers retry/correction and proportional score scaling.
                    if coverage_ratio >= 0.85:
                        flags.append(
                            f"⚠️ [PARTIAL_COVERAGE] {target_source_desc} target coverage: delivered {delivered_count}/{expected_count} targets{missing_txt}"
                        )
                        composite_score = max(0.0, round(composite_score * max(coverage_ratio, 0.9), 1))
                    else:
                        flags.append(
                            f"❌ [INCOMPLETE_COVERAGE] Incomplete target coverage: delivered only {delivered_count}/{expected_count} targets{missing_txt}"
                        )
                        composite_score = round(composite_score * coverage_ratio, 1)

        # Special handling for expert_audit task: align composite_score with deterministic code gate
        if task_type == "expert_audit" and isinstance(parsed, dict):
            try:
                from .expert_auditor import ExpertAuditor
                infer_type = "vocabulary"
                task_lower = (task or "").lower()
                prompt_lower = (user_prompt or "").lower()
                if "translation" in task_lower or "translation" in prompt_lower:
                    infer_type = "translation"
                elif "reading" in task_lower or "reading" in prompt_lower:
                    infer_type = "reading"
                
                divs = parsed.get("blind_solve_confident_divergences", 0)
                ExpertAuditor.reconcile_report_scores(parsed, quiz_type=infer_type, confident_divergences=divs)
                raw_audited_score = round(float(parsed.get("overall_quality_score", composite_score)), 1)
                
                if pre_cure_score is None:
                    pre_cure_score = raw_audited_score
                if post_cure_score is not None:
                    composite_score = post_cure_score
                else:
                    composite_score = raw_audited_score
                
                flaws = parsed.get("flawed_item_indices", [])
                if flaws:
                    flags.insert(0, f"❌ [FATAL FLAW] {len(flaws)} item(s) failed single-fit validation: #{', #'.join(str(i) for i in flaws)}")
                if not parsed.get("pass_audit", False):
                    base = parsed.get("base_quality_score", raw_audited_score)
                    if base > raw_audited_score:
                        flags.insert(1, f"⚠️ [L2_PENALTY] Capped from Base {base}% to {raw_audited_score}% due to fatal flaw(s)")
                if divs > 0:
                    flags.insert(0, f"❌ [CRITICAL FLAW] {divs} confident double-key divergence(s) detected in blind solve")
                if post_cure_score is not None and post_cure_score > raw_audited_score:
                    flags.insert(0, f"✅ [SURGICAL CURE] In-place healed from {pre_cure_score}% to Final {post_cure_score}%")
            except Exception as err:
                composite_score = round(float(parsed.get("overall_quality_score", composite_score)), 1)

        # Fatal QA Flag Gate: If fatal pedagogical flags are detected, composite score cannot claim passing (>=80%)
        # Automatically cap composite score below 60.0 to reflect failed/review status
        has_fatal_flags = any(
            any(fatal in f for fatal in FATAL_QA_FLAGS)
            for f in flags
        )
        if has_fatal_flags:
            composite_score = min(composite_score, 59.0)
            if status == "SUCCESS":
                status = "REVIEW_NEEDED"
                if not failure_category:
                    failure_category = "QA_FATAL_FLAG"

        # Sort flags: critical errors (❌) first, then warnings (⚠️)
        flags.sort(key=lambda x: (0 if x.startswith("❌") else (1 if x.startswith("⚠️") else 2)))

        return {
            "log_name": log_name,
            "task": task,
            "task_type": task_type,
            "model": model,
            "status": status,
            "failure_category": failure_category,
            "n_items": len(items),
            "composite_score": composite_score,
            "pre_cure_score": pre_cure_score,
            "post_cure_score": post_cure_score,
            "scores": {
                "schema_adherence": schema_score,
                "verbatim_faithfulness": verbatim_score,
                "pedagogical_quality": pedagogy_score,
                "uniqueness": uniqueness_score,
            },
            "flags": flags,
        }

    @classmethod
    def audit_all_logs(cls, logs_dir: Path = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Scans and audits all .log files in the logs/ directory.
        Returns the overall Hero Board leaderboard and individual log audits.
        Results are cached by MD5 hash of entries' metadata so repeated calls stay fast and thread-safe.
        """
        if logs_dir is None:
            logs_dir = config.project_root / "logs"
        logs_dir = Path(logs_dir)

        if not logs_dir.exists():
            return {"hero_board": [], "audits": []}

        entries = []
        for path in logs_dir.glob("*.log"):
            try:
                st = path.stat()
            except OSError:
                continue
            entries.append((path, st.st_mtime, st.st_size))

        if not entries:
            return {"hero_board": [], "audits": []}

        # Build compact hash-based cache key using sha256 (FIPS compliant)
        sig_data = f"{logs_dir.as_posix()}:{len(entries)}:" + ":".join(f"{p.name}:{mt}:{sz}" for (p, mt, sz) in entries)
        cache_key = hashlib.sha256(sig_data.encode("utf-8")).hexdigest()

        with cls._AUDIT_LOCK:
            if use_cache and cls._AUDIT_CACHE["key"] == cache_key and cls._AUDIT_CACHE["value"] is not None:
                return cls._AUDIT_CACHE["value"]

        # Newest first
        entries.sort(key=lambda e: e[1], reverse=True)

        audits: List[Dict[str, Any]] = []
        model_stats: Dict[str, Dict[str, Any]] = {}

        # Worker for ThreadPoolExecutor
        def _process_one(log_file: Path) -> Optional[Dict[str, Any]]:
            parsed = cls.parse_log_file(log_file)
            if not parsed:
                return None
            evaluation = cls.evaluate_log(parsed)
            if evaluation.get("task_type") == "intermediate_prose":
                return None
            return evaluation

        # Parallelize log parsing and evaluation (preserves recency order)
        with ThreadPoolExecutor() as executor:
            raw_evaluations = list(executor.map(_process_one, [e[0] for e in entries]))

        for evaluation in raw_evaluations:
            if not evaluation:
                continue

            audits.append(evaluation)

            model = evaluation["model"]
            if model not in model_stats:
                model_stats[model] = {
                    "model": model,
                    "runs": 0,
                    "success_runs": 0,
                    "failed_runs": 0,
                    "total_score": 0.0,
                    "schema_sum": 0.0,
                    "schema_runs": 0,
                    "verbatim_sum": 0.0,
                    "verbatim_runs": 0,
                    "pedagogy_sum": 0.0,
                    "pedagogy_runs": 0,
                    "uniqueness_sum": 0.0,
                    "uniqueness_runs": 0,
                }

            s = model_stats[model]
            s["runs"] += 1

            # Exclude FAILED or RETRYING status logs from the quality leaderboard composite score
            if evaluation.get("status") in ("FAILED", "RETRYING"):
                s["failed_runs"] += 1
                continue

            s["success_runs"] += 1
            s["total_score"] += evaluation["composite_score"]

            sc = evaluation["scores"]
            if sc["schema_adherence"] is not None:
                s["schema_sum"] += sc["schema_adherence"]
                s["schema_runs"] += 1
            if sc["verbatim_faithfulness"] is not None:
                s["verbatim_sum"] += sc["verbatim_faithfulness"]
                s["verbatim_runs"] += 1
            if sc["pedagogical_quality"] is not None:
                s["pedagogy_sum"] += sc["pedagogical_quality"]
                s["pedagogy_runs"] += 1
            if sc["uniqueness"] is not None:
                s["uniqueness_sum"] += sc["uniqueness"]
                s["uniqueness_runs"] += 1

        # Build Hero Board Leaderboard
        hero_board = []
        for model, s in model_stats.items():
            succ = s["success_runs"]
            if succ == 0:
                comp = 0.0
                sch_avg = vbt_avg = ped_avg = unq_avg = 0.0
            else:
                comp = round(s["total_score"] / succ, 1)
                sch_avg = round((s["schema_sum"] / (s["schema_runs"] * W_SCHEMA)) * 100.0, 1) if s["schema_runs"] else 100.0
                vbt_avg = round((s["verbatim_sum"] / (s["verbatim_runs"] * W_VERBATIM)) * 100.0, 1) if s["verbatim_runs"] else 100.0
                ped_avg = round((s["pedagogy_sum"] / (s["pedagogy_runs"] * W_PEDAGOGY)) * 100.0, 1) if s["pedagogy_runs"] else 100.0
                unq_avg = round((s["uniqueness_sum"] / (s["uniqueness_runs"] * W_UNIQUENESS)) * 100.0, 1) if s["uniqueness_runs"] else 100.0

            hero_board.append({
                "model": model,
                "runs": s["runs"],
                "success_runs": succ,
                "failed_runs": s["failed_runs"],
                "composite_score": comp,
                "schema_adherence_avg": sch_avg,
                "verbatim_faithfulness_avg": vbt_avg,
                "pedagogical_quality_avg": ped_avg,
                "uniqueness_avg": unq_avg,
            })

        # Sort Leaderboard descending by composite score
        hero_board.sort(key=lambda x: x["composite_score"], reverse=True)

        result = {"hero_board": hero_board, "audits": audits}
        with cls._AUDIT_LOCK:
            cls._AUDIT_CACHE["key"] = cache_key
            cls._AUDIT_CACHE["value"] = result
        return result
