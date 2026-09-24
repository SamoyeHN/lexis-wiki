import re
import json
import unicodedata
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, get_args

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
    "object", "complement", "adverbial", "focus element", "subordinate clause",
    "main clause", "dependent clause", "modal", "copula", "past participle",
    "present participle", "infinitive", "gerund", "adjective phrase", "adverb phrase",
    "prepositional phrase", "quotation", "wh-word", "wh-clause", "head noun",
    "verb-ing", "verb-ed"
})

# Standard COBUILD bare POS tokens for unbracketed pattern formulas
COBUILD_POS_TOKENS = frozenset({
    "np", "vp", "v", "be", "aux", "v-ed", "v3", "v-ing", "to-v", "adj", "adv",
    "det", "prep", "conj", "clause", "s", "n", "pron", "modal"
})

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
    "Multiple blanks",
    "Target word leaks",
    "missing fill-in-the-blank slot",
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
    # Remove structural brackets/parentheses with whitespace separation
    text = re.sub(r"[\[\]\(\)\{\}\<\>]+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_core(text: str) -> str:
    """Strip all punctuation and non-alphanumeric chars (retaining spaces)."""
    norm = _normalize_text(text)
    return re.sub(r"[^\w\s]", "", norm).strip()


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
        # Strictly strip any trailing draft blocks or retry critique blocks
        content = re.split(
            r"\n\s*###+\s*(?:🚨|\[QUALITY AUDIT REVIEW|VOCABULARY DRAFT|GRAMMAR PATTERNS DRAFT|QUIZ DRAFT|DRAFT)",
            content,
            flags=re.IGNORECASE
        )[0].strip()

    return content


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
        # Matches: 1. [S-8] (Logic & Stance) Formula: `...`
        m = re.findall(r'(\d+)\.\s*\[([^\]]+)\]\s*\(([^)]+)\)\s*Formula:\s*`([^`]+)`', norm_prompt)
        for num, sid, cat, formula in m:
            skeletons.append({"sid": sid.strip(), "category": cat.strip(), "formula": formula.strip()})
    elif task_type == "expressions":
        # Matches: 1. [S-10] (phrasal verb) Formula: `...`
        m = re.findall(r'(\d+)\.\s*\[([^\]]+)\]\s*\(([^)]+)\)\s*Formula:\s*`([^`]+)`', norm_prompt)
        for num, sid, pos, formula in m:
            skeletons.append({"sid": sid.strip(), "pos": pos.strip(), "formula": formula.strip()})
    elif task_type == "vocabulary":
        # Matches: 1. [S-14] **word** (noun)
        m = re.findall(r'(\d+)\.\s*\[([^\]]+)\]\s*\*\*([^*]+)\*\*\s*\(([^)]+)\)', norm_prompt)
        for num, sid, word, pos in m:
            skeletons.append({"sid": sid.strip(), "word": word.strip(), "pos": pos.strip()})
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


def _score_schema(parsed: Any, raw_response: str = "") -> Tuple[Optional[float], List[str]]:
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
            if wordlist and _wordlist_matches(clean_target, wordlist):
                matches += 1
            elif not wordlist:
                # Packaging / conversion phase: check if target word exists in the drafted prompt text
                if clean_target in clean_draft:
                    matches += 1
                else:
                    flags.append(f"⚠️ Target '{target}' not found in draft content")
            else:
                # Robust secondary check: wordlist was extracted, but target was not matched against it.
                # If target is solidly grounded in the drafted prompt text (e.g. Turn 1 prose draft),
                # treat as valid to prevent false-positive hallucination flags during conversion.
                if clean_target in clean_draft:
                    matches += 1
                else:
                    flags.append(f"❌ Target word '{target}' not found in supplied word list (possible hallucination)")
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
        quote = item.get("quoted_sentence") or item.get("quote")
        word = str(item.get("word") or item.get("pattern_formula") or "").strip()
        checks += 1

        if not (quote and isinstance(quote, str) and quote.strip()):
            flags.append(f"❌ Missing quote/quoted_sentence for item: '{word[:40]}'")
            continue
            
        # Check 1: Explicit hallucination acknowledgment
        if _is_hallucinated_quote(quote):
            flags.append(f"❌ Hallucinated quote (explicitly inferred/absent): '{quote[:50]}...'")
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
                    if tok in clean_quote or (len(tok) >= 4 and tok[:4] in clean_quote):
                        return True
                    if tok in COMMON_IRREGULARS and any(ir in clean_quote for ir in COMMON_IRREGULARS[tok]):
                        return True
                    return False

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

    # Extract study list headwords from prompt if evaluating a quiz
    prompt_headwords = set()
    if task_type == "quiz" and user_prompt:
        for m in re.finditer(r"## \[\[(.*?)\]\]", user_prompt):
            prompt_headwords.add(m.group(1).strip().lower())
        if not prompt_headwords:
            for m in re.finditer(r"-\s*Target:\s*([^\n\r]+)", user_prompt):
                prompt_headwords.add(m.group(1).strip().lower())

    for item in items:
        if task_type == "vocabulary":
            word = _safe_str(item.get("word"))
            pos = _safe_str(item.get("part_of_speech")).lower()
            definition = _safe_str(item.get("definition"))
            example = _safe_str(item.get("example_usage"))
            quote = _safe_str(item.get("quoted_sentence"))
            checks += 1
            
            # Allow words/phrases with slots, hyphens, brackets, parentheses, apostrophes
            valid_word = bool(re.search(r"^[A-Za-z\s\-'\[\]\(\)]+$", word) and len(word) >= 2)
            valid_pos = pos in VALID_POS_SET
            has_example = bool(example)
            is_distinct_example = bool(has_example and _clean_core(example) != _clean_core(quote))
            
            if valid_word and valid_pos and definition and is_distinct_example:
                passes += 1
            else:
                reasons = []
                if not valid_word: reasons.append(f"invalid headword '{word}'")
                if not valid_pos: reasons.append(f"invalid PoS '{pos}'")
                if not definition: reasons.append("missing definition")
                if not has_example:
                    reasons.append("missing example_usage")
                elif not is_distinct_example:
                    reasons.append("example_usage is an unoriginal duplicate of quoted_sentence")
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

            if is_multiword and not is_trivial and definition and is_distinct_example:
                passes += 1
            else:
                reasons = []
                if not is_multiword or is_trivial:
                    reasons.append("lacks multi-word/slot form or is too basic")
                if not definition:
                    reasons.append("missing definition")
                if not has_example:
                    reasons.append("missing example_usage")
                elif not is_distinct_example:
                    reasons.append("example_usage is an unoriginal duplicate of quoted_sentence")
                flags.append(f"⚠️ Expression '{word}' failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "grammar":
            pattern = _safe_str(item.get("pattern_formula"))
            audit = _safe_str(item.get("design_audit"))
            quote = _safe_str(item.get("quote"))
            category = _safe_str(item.get("category"))
            checks += 1
            
            reasons = []
            if not audit:
                reasons.append("missing design_audit")
            if not pattern:
                reasons.append("missing pattern_formula")
            if not _safe_str(item.get("imitation_example")):
                reasons.append("missing imitation_example")
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

            # Structural Consistency: Literal keywords in formula MUST exist in quote
            # (e.g. if formula requires 'while', 'not... but...', 'which', or '[V-ing Phrase]', quote must possess corresponding physical markers)
            clean_quote = _clean_core(quote)
            clean_quote_raw = quote.lower()
            if pattern:
                # Check for explicit literal connector / subordinator words in formula
                formula_anchors = [
                    re.sub(r"[^\w\s]", "", lit).lower().strip() 
                    for lit in re.sub(r"\[.*?\]|\(.*?\)|[+,/]", " ", pattern).split() 
                    if len(re.sub(r"[^\w\s]", "", lit).strip()) >= 3 and lit.lower() not in COBUILD_POS_TOKENS
                ]
                missing_anchors = [anc for anc in formula_anchors if anc not in clean_quote]
                if missing_anchors:
                    reasons.append(f"formula anchor '{', '.join(missing_anchors)}' ungrounded in quote")
                # Check for participle requirement
                if "[v-ing" in pattern.lower() or "[participle" in pattern.lower():
                    ING_EXCLUDE = frozenset({"morning", "evening", "thing", "something", "nothing", "everything", "anything", "during", "ceiling"})
                    ing_tokens = [w for w in re.findall(r"\b[a-zA-Z]{3,}ing\b", clean_quote_raw) if w not in ING_EXCLUDE]
                    if not ing_tokens:
                        reasons.append("formula requires '[V-ing]' but quote contains no participle verb ungrounded in quote")

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
                    target_in_options = (clean_target == selected_opt) or (
                        len(clean_target) >= 4 and len(selected_opt) >= 4 and (
                            selected_opt.startswith(clean_target[:4]) or clean_target.startswith(selected_opt[:4]) or (clean_target in selected_opt)
                        )
                    )

            # 4. Blank verification for fill-in-the-blank questions (Strict Single Blank Gate)
            has_blank_when_expected = True
            is_single_blank = True
            no_stem_leak = True
            is_adequate_complexity = True

            if is_comparative_translation:
                # Comparative translation does not require blanks
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

                # Syntax Complexity Check (CEFR Level & Clause Check - Soft Warning)
                # Stems should ideally have >= 9 words, or contain a subordinate/coordinate clause marker or comma
                if len(stem_words) < 9:
                    clause_markers = ('although', 'though', 'while', 'whereas', 'because', 'since', 'if', 'unless', 'which', 'that', 'who', 'whom', 'whose', 'where', 'when', 'after', 'before', 'until', 'so that')
                    has_clause = any(re.search(rf'\b{re.escape(cm)}\b', question, flags=re.IGNORECASE) for cm in clause_markers)
                    has_comma = (',' in question or ';' in question)
                    if not (has_clause or has_comma):
                        is_adequate_complexity = False
                        flags.append(f"⚠️ Quiz item '{target}': Trivial short stem (< 9 words without subordinate/coordinate clause)")

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
                no_in_list_recycling
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
                if not no_in_list_recycling:
                    reasons.append(f"recycles study list headwords in distractors {recycled_in_distractors}")
                    flags.append(f"❌ Quiz item '{target or question[:25]}' in-list distractor recycling: {recycled_in_distractors}")
                else:
                    flags.append(f"⚠️ Quiz question failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "summary":
            name = _safe_str(item.get("concept_name"))
            sig = _safe_str(item.get("educational_significance"))
            details = item.get("key_details", [])
            checks += 1
            if name and sig and isinstance(details, list) and len(details) > 0:
                passes += 1
            else:
                flags.append(f"⚠️ Concept '{name}' missing educational significance or key details")
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
            is_valid_idx = isinstance(bs_idx, int) and 0 <= bs_idx <= 3
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
                cleaned_q = cleaned_q.strip("\"'“”‘’").strip()
                item[quote_field] = cleaned_q

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
            dedup_key = (word_val,)
            
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
            category = str(item.get("category") or "").strip()
            imitation = str(item.get("imitation_example") or "").strip()
            mistakes = str(item.get("common_mistakes") or "").strip()

            if not (core_quote and category and imitation and mistakes):
                missing_fields = []
                if not core_quote: missing_fields.append("quote")
                if not category: missing_fields.append("category")
                if not imitation: missing_fields.append("imitation_example")
                if not mistakes: missing_fields.append("common_mistakes")
                pruned_flags.append(f"✂️ Pruned incomplete grammar pattern '{word[:30]}' (missing required: {', '.join(missing_fields)})")
                continue

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
            if tok in target_text or (len(tok) >= 4 and tok[:4] in target_text):
                return True
            if tok in COMMON_IRREGULARS and any(ir in target_text for ir in COMMON_IRREGULARS[tok]):
                return True
            return False

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
        schema_score, f1 = _score_schema(parsed, raw_response)
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
                    flags.append(
                        f"❌ [INCOMPLETE_COVERAGE] Incomplete target coverage: delivered only {delivered_count}/{expected_count} targets"
                    )
                    # Proportionately scale composite score so that 1/5 outputs score ~20%, NOT 100%!
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
