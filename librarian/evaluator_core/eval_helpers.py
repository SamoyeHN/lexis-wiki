"""Shared evaluation helpers, normalization utilities, and schema constants for Lexis Wiki.

Extracted from evaluator.py to provide a dependency leaf for evaluator_core.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, get_args

from ..config import config
from ..linguistics import LinguisticEngine
from ..schemas import EXPRESSION_TYPES, PARTS_OF_SPEECH

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
    "verb-ing", "verb-ed", "v-ing phrase", "v-ed phrase", "prepp/np", "adj/np", "np/adj",
    "comparative", "proposition clause", "also"
})

# Standard COBUILD bare POS tokens for unbracketed pattern formulas
COBUILD_POS_TOKENS = frozenset({
    "np", "vp", "v", "be", "aux", "v-ed", "v3", "v-ing", "to-v", "to-inf", "inf", "adj", "adv",
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
    "duplicate item",
    "duplicate options",
    "duplicate common_mistakes",
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
    "Stem verbatim from dictionary example",
    "Stem verbatim from curriculum quote",
    "Prescribed options altered",
    "Prescribed answer index altered",
    "degenerate formula slot",
    "deviates from the pre-extracted skeleton",
    "duplicates the source quote",
    "Audit truncated early",
    "Comparative translation defect",
    "Minimal-Pair contract violation",
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
    """Checks whether `tok` (including inflected or irregular forms) appears in `text`."""
    if not tok or not text:
        return False
    if LinguisticEngine.text_contains_form(text, tok):
        return True
    # Derived -ly adverbs (e.g. 'effectively') must appear as the adverb itself, not as base adjective ('effective')
    if len(tok) >= 5 and tok.endswith("ly"):
        return bool(re.search(rf"\b{re.escape(tok)}\b", text))
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

    # Deterministic grammar items carry an authoritative syntax_topic and macro-domain
    # from grammar_profiles.json; do not override with whole-sentence surface regexes.
    if item.get("syntax_topic"):
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
        re.search(
            r"\bIt\s+(?:is|was|were|'s)\s+(?:only|not\s+only|because|when|in|on|at|by|with|(?:(?:a|an|the|this|that|these|those|my|your|his|her|its|our|their|[a-z]{2,})\s+){1,8}(?:who|whom|that))\b",
            quote,
            re.IGNORECASE,
        )
        and not re.search(
            r"\bIt\s+(?:is|was|were|'s)\s+(?:(?:not|also|still|only|even|very|quite|highly|widely|generally|commonly)\s+)*(?:said|thought|believed|reported|expected|suggested|indicated|argued|claimed|known|assumed|estimated|found|shown|noted|agreed|important|essential|necessary|likely|unlikely|clear|obvious|vital|crucial|apparent|natural|possible|impossible|true|certain|evident|surprising|remarkable|common|rare|normal|strange|odd|sad|pity|shame|wonder|secret|mystery|doubt|fact|time)\b",
            quote,
            re.IGNORECASE,
        )
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
            # 3. Tertiary: Raw markdown source with optional YAML frontmatter and section heading (# or ##)
            match_md = re.search(r"(?:^|\n)(?:---\s*\n.*?\n---\s*\n)?(?:#{1,3}\s+[^\n]+\n+)(.*)", user_prompt, re.DOTALL)
            if match_md:
                content = match_md.group(1).strip()
            else:
                # 4. Plain-text / headingless raw source fallback (when no LLM instruction boilerplate is present)
                stripped_raw = re.sub(r"^\s*---\s*\n.*?\n---\s*\n?", "", user_prompt, count=1, flags=re.DOTALL).strip()
                if (
                    stripped_raw
                    and len(stripped_raw.split()) >= 20
                    and not re.search(
                        r"(?:You are an expert|###\s*(?:OUTPUT|INSTRUCTIONS|TASK|SCHEMA)|JSON\s+SCHEMA|Return\s+ONLY\s+valid\s+JSON)",
                        stripped_raw,
                        re.IGNORECASE,
                    )
                ):
                    content = stripped_raw
    
    if content:
        # Strictly strip any trailing task blocks, target lists, skeletons, draft blocks, or retry critique blocks
        # This prevents target vocabulary/expression/grammar instruction blocks from contaminating authentic source text.
        content = re.split(
            r"\n\s*###+\s*(?:TARGET\s+VOCABULARY|TARGET\s+EXPRESSIONS|TARGET\s+GRAMMAR|DETERMINISTIC\s+TARGET|🚨|\[QUALITY AUDIT REVIEW|VOCABULARY DRAFT|GRAMMAR PATTERNS DRAFT|QUIZ DRAFT|DRAFT)",
            content,
            flags=re.IGNORECASE
        )[0].strip()

    return content


MIN_SOURCE_WORDS_FOR_CEILING = 40


def _cefr_source_ceiling(user_prompt: str) -> Tuple[Optional[str], Set[str]]:
    """Returns (CEFR level of the authentic passage, words used in that passage)."""
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
    if "extract_wsd" in t or "wsd_arbitration" in t or "chat_text_mode" in t:
        return "wsd_arbitration"
    if "expert_audit" in t or "quality_audit" in t or "_audit_" in t or t.endswith("_audit") or t.startswith("audit_"):
        return "expert_audit"
    if isinstance(parsed, dict):
        if ("pass_audit" in parsed and "blind_solve_accuracy" in parsed) or (
            "summary_verdict" in parsed and isinstance(parsed.get("questions"), list)
        ):
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

