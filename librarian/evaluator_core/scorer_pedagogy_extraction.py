"""Item-level pedagogical validation for extraction and discourse tasks:
- vocabulary
- expressions
- grammar
- summary
- mindmap
- expert_audit
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..linguistics import LinguisticEngine
from ..processor import WikiProcessor
from .eval_helpers import (
    COBUILD_POS_TOKENS,
    VALID_EXPR_POS_SET,
    VALID_POS_SET,
    _clean_core,
    _expand_contractions,
    _form_in_text,
    _safe_str,
    auto_remap_grammar_category,
    degenerate_grammar_slots,
)


def validate_vocabulary_item(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single vocabulary extraction item."""
    flags: List[str] = []
    word = _safe_str(item.get("word"))
    pos = _safe_str(item.get("part_of_speech")).lower()
    quote = _safe_str(item.get("quoted_sentence"))
    is_expression = LinguisticEngine.is_multiword_expression(word)
    needs_pos = bool(word) and (
        not pos or (is_expression and pos not in LinguisticEngine.EXPRESSION_TYPE_LABELS)
    )
    if needs_pos:
        if is_expression:
            pos = LinguisticEngine.classify_expression_type(word, quote) or "collocation"
        else:
            pos = LinguisticEngine.determine_contextual_pos(word, quote)
        item["part_of_speech"] = pos
    definition = _safe_str(item.get("definition"))
    example = _safe_str(item.get("example_usage"))

    valid_word = bool(re.search(r"^[A-Za-z\s\-'\[\]\(\)/]+$", word) and len(word) >= 2)
    allowed_pos = (VALID_POS_SET | VALID_EXPR_POS_SET) if is_expression else VALID_POS_SET
    valid_pos = pos in allowed_pos
    has_example = bool(example)
    is_distinct_example = bool(has_example and _clean_core(example) != _clean_core(quote))
    invented_definition = bool(definition) and LinguisticEngine.is_boilerplate_definition(definition)

    if valid_word and valid_pos and definition and not invented_definition and is_distinct_example:
        return True, flags

    reasons = []
    if not valid_word:
        reasons.append(f"invalid headword '{word}'")
    if not valid_pos:
        reasons.append(f"invalid PoS '{pos}'")
    if not definition:
        reasons.append("missing definition")
    elif invented_definition:
        reasons.append("invented fallback definition (no lexicon host grounds it)")
    if not has_example:
        reasons.append("missing example_usage")
    elif not is_distinct_example:
        reasons.append("example_usage is an unoriginal duplicate of quoted_sentence")
    flags.append(f"⚠️ Vocabulary item '{word}' failed pedagogy check: {', '.join(reasons)}")
    return False, flags


def validate_expressions_item(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single multi-word expression extraction item."""
    flags: List[str] = []
    word = _safe_str(item.get("word"))
    definition = _safe_str(item.get("definition"))
    example = _safe_str(item.get("example_usage"))
    quote = _safe_str(item.get("quoted_sentence"))

    is_multiword = bool(re.search(r"\[.+?\]|one's", word, re.IGNORECASE) or len(word.split()) > 1)
    is_trivial = word.lower() in ("talk", "listen", "turn", "watch", "sit down", "talk to", "listen to", "look at")
    has_example = bool(example)
    is_distinct_example = bool(has_example and _clean_core(example) != _clean_core(quote))
    invented_definition = bool(definition) and LinguisticEngine.is_boilerplate_definition(definition)

    if is_multiword and not is_trivial and definition and not invented_definition and is_distinct_example:
        return True, flags

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
    flags.append(f"⚠️ Expression '{word}' failed pedagogy check: {', '.join(reasons)}")
    return False, flags


def validate_grammar_item(
    item: Dict[str, Any],
    align_formula_fn: Callable[[str, str], Tuple[bool, str]],
    src_cefr_level: Optional[str] = None,
    src_passage_words: Optional[Set[str]] = None,
) -> Tuple[bool, List[str]]:
    """Validate a single grammar pattern extraction item."""
    flags: List[str] = []
    if src_passage_words is None:
        src_passage_words = set()

    pattern = _safe_str(item.get("pattern_formula"))
    audit = _safe_str(item.get("design_audit"))
    quote = _safe_str(item.get("quote"))
    category = _safe_str(item.get("category"))
    if not category and quote:
        category = LinguisticEngine.classify_grammar_dependency(quote) or "Information Packaging"
        item["category"] = category

    reasons: List[str] = []
    imitation = _safe_str(item.get("imitation_example"))
    if not audit:
        reasons.append("missing design_audit")
    if not pattern:
        reasons.append("missing pattern_formula")
    if not imitation:
        reasons.append("missing imitation_example")
    else:
        if quote and _clean_core(imitation) == _clean_core(quote):
            reasons.append("imitation_example duplicates the source quote")
            flags.append(
                f"❌ Grammar pattern '{pattern[:40]}': imitation_example duplicates the source quote "
                "(an imitation must be a new sentence, not the quoted one)"
            )
    if not _safe_str(item.get("common_mistakes")):
        reasons.append("missing common_mistakes")

    pattern = WikiProcessor.unwrap_literal_brackets(pattern)

    # Minimal Anti-Triviality Gate
    slot_count = len(re.findall(r"\[.*?\]", pattern))
    lits = [
        w for w in re.sub(r"\[.*?\]|\(.*?\)|[+,/]", " ", pattern).split()
        if len(_clean_core(w)) >= 2
    ]
    TRIVIAL_ANCHOR_WORDS = frozenset({"but", "and", "so", "or", "of", "course", "well", "now", "here"})
    TRIVIAL_PHRASES = frozenset({"but", "and", "so", "or", "of course", "here is", "heres"})
    clean_lits = [re.sub(r"[^\w\s]", "", l).lower().strip() for l in lits if re.sub(r"[^\w\s]", "", l).strip()]
    combined_lit_phrase = " ".join(clean_lits)
    if slot_count <= 1:
        if (combined_lit_phrase in TRIVIAL_PHRASES) or (clean_lits and all(l in TRIVIAL_ANCHOR_WORDS for l in clean_lits)):
            reasons.append(f"trivial formula anchored only by conversational filler or conjunction: '{pattern}'")
        elif clean_lits and any(l in {"however", "therefore", "moreover", "furthermore", "meanwhile", "nevertheless", "in fact", "actually"} for l in clean_lits):
            reasons.append(f"un-abstracted superficial pattern anchored only on discourse adverbial connector: '{pattern}'")

    # Structural Consistency: Literal keywords in formula MUST exist in quote AND imitation_example
    clean_quote = _clean_core(quote)
    clean_quote_raw = quote.lower()
    clean_imitation = _clean_core(imitation)
    clean_imitation_raw = imitation.lower()
    if pattern:
        raw_lits = re.findall(r"[a-zA-Z0-9_\'/-]+", re.sub(r"\[.*?\]|\(.*?\)", " ", pattern))
        expanded_quote = _clean_core(_expand_contractions(quote))
        expanded_imitation = _clean_core(_expand_contractions(imitation))
        quote_tokens = set(clean_quote.split()) | set(expanded_quote.split())
        imitation_tokens = set(clean_imitation.split()) | set(expanded_imitation.split())
        missing_anchors_quote = []
        missing_anchors_imitation = []
        for lit in raw_lits:
            alts = [a.strip() for a in lit.split("/") if a.strip()]
            expanded_alts = []
            for a in alts:
                if a.lower() in ("to-v", "to-inf"):
                    expanded_alts.append("to")
                elif len(_clean_core(a)) >= 2 and _clean_core(a) not in COBUILD_POS_TOKENS and a.lower() not in COBUILD_POS_TOKENS:
                    expanded_alts.append(a)
            valid_alts = expanded_alts
            if not valid_alts:
                continue
            quote_has_anchor = any(_clean_core(a) in quote_tokens or _form_in_text(_clean_core(a), clean_quote) for a in valid_alts)
            if not quote_has_anchor and any(a.lower() in ("is", "was", "were", "are", "be", "been") for a in valid_alts):
                if re.search(r"\b(?:is|was|were|are|be|been|'s|'re)\b", clean_quote, re.IGNORECASE):
                    quote_has_anchor = True
            if not quote_has_anchor:
                missing_anchors_quote.append(lit)
            if imitation and len(imitation.split()) >= 3:
                is_instantiated = any(_clean_core(a) in imitation_tokens or _form_in_text(_clean_core(a), clean_imitation) for a in valid_alts)
                if not is_instantiated and any(a.lower() in ("is", "was", "were", "are", "be", "been") for a in valid_alts):
                    if re.search(r"\b(?:is|was|were|are|be|been|'s|'re)\b", clean_imitation, re.IGNORECASE):
                        is_instantiated = True
                if not is_instantiated and any(re.match(r"^(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?|underscor(?:es|ed))$", a, re.IGNORECASE) for a in valid_alts):
                    if re.search(r"\b(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?|underscor(?:es|ed))\b", imitation, re.IGNORECASE):
                        is_instantiated = True
                if not is_instantiated and any(a.lower() == "that" for a in valid_alts) and re.search(r",\s*which\b", pattern, re.IGNORECASE):
                    if re.search(r",\s*which\s+(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?|underscor(?:es|ed))\b", imitation, re.IGNORECASE):
                        is_instantiated = True
                if not is_instantiated and (
                    any(a.lower() in ("although", "though", "even", "while", "whereas") for a in valid_alts)
                    or (any(a.lower() == "if" for a in valid_alts) and re.search(r"\beven\s+if\b", pattern, re.IGNORECASE))
                ):
                    if re.search(r"\b(?:although|though|even though|even if|while|whereas)\b", imitation, re.IGNORECASE):
                        is_instantiated = True
                if not is_instantiated and any(a.lower() in ("fact", "idea", "belief", "assumption", "notion", "claim", "possibility", "view", "evidence", "news", "impression", "argument", "thought", "hope", "fear", "doubt", "contention", "recognition", "realization", "premise", "hypothesis", "consensus", "conclusion", "paradox", "irony", "conviction") for a in valid_alts):
                    if re.search(r"\b(?:fact|idea|belief|assumption|notion|claim|possibility|view|evidence|news|impression|argument|thought|hope|fear|doubt|contention|recognition|realization|premise|hypothesis|consensus|conclusion|paradox|irony|conviction)\b", imitation, re.IGNORECASE):
                        is_instantiated = True
                if not is_instantiated and any(a.lower() in ("make", "makes", "made", "find", "finds", "found", "keep", "keeps", "kept", "render", "renders", "rendered", "deem", "deems", "deemed", "consider", "considers", "considered", "leave", "leaves", "left", "let", "lets", "have", "has", "had") for a in valid_alts):
                    if re.search(r"\b(?:make|makes|made|making|find|finds|found|finding|keep|keeps|kept|keeping|render|renders|rendered|rendering|deem|deems|deemed|consider|considers|considered|leave|leaves|left|leaving|let|lets|letting|have|has|had|having)\b", imitation, re.IGNORECASE):
                        is_instantiated = True
                if not is_instantiated and any(a.lower() in ("however", "nevertheless", "nonetheless", "conversely") for a in valid_alts):
                    if re.search(r"\b(?:however|nevertheless|nonetheless|conversely)\b", imitation, re.IGNORECASE):
                        is_instantiated = True
                if not is_instantiated and any(a.lower() in ("which", "that", "who", "whom") for a in valid_alts) and re.match(r"^\[np\]\s*\+\s*(?:which|that|who)", pattern.strip(), re.IGNORECASE):
                    if re.search(r"\b(?:which|that|who|whom)\b", imitation, re.IGNORECASE):
                        is_instantiated = True
                if not is_instantiated:
                    missing_anchors_imitation.append(lit)

        if missing_anchors_quote:
            reasons.append(f"formula anchor '{', '.join(missing_anchors_quote)}' ungrounded in quote")
        if missing_anchors_imitation:
            reasons.append(f"imitation_example does not instantiate formula anchor '{', '.join(missing_anchors_imitation)}'")
            flags.append(
                f"❌ Grammar pattern '{pattern[:40]}': imitation_example does not instantiate formula anchor "
                f"'{', '.join(missing_anchors_imitation)}' (an imitation must instantiate the formula it models)"
            )

        # Participle requirement
        ING_EXCLUDE = frozenset({"morning", "evening", "thing", "something", "nothing", "everything", "anything", "during", "ceiling", "ring", "spring", "king", "wing", "sing", "bring", "cling", "fling", "sting", "string", "swing"})
        pat_low = pattern.lower()
        requires_strict_ving = bool(re.search(r"\[v-ing(?:\s+phrase)?\]", pat_low))
        requires_any_participle = ("[v-ing" in pat_low or "[participle" in pat_low) and not requires_strict_ving
        if requires_strict_ving:
            ing_tokens_quote = [w for w in re.findall(r"\b(?:being|doing|going|[a-zA-Z]{3,}ing)\b", clean_quote_raw) if w.lower() not in ING_EXCLUDE]
            if not ing_tokens_quote:
                reasons.append("formula requires '[V-ing]' but quote contains no participle verb ungrounded in quote")
            if imitation:
                ing_tokens_imitation = [w for w in re.findall(r"\b(?:being|doing|going|[a-zA-Z]{3,}ing)\b", clean_imitation_raw) if w.lower() not in ING_EXCLUDE]
                if not ing_tokens_imitation:
                    reasons.append("formula requires '[V-ing]' but imitation_example contains no participle verb")
        elif requires_any_participle:
            pass

    # Ceiling Gate
    if src_cefr_level and imitation:
        over = LinguisticEngine.over_ceiling_tokens(imitation, src_cefr_level, mode="text", allow=src_passage_words)
        if over:
            offenders = ", ".join(f"'{w}' ({lvl})" for w, lvl in over[:3])
            band = LinguisticEngine.cefr_ceiling(src_cefr_level, mode="text")
            reason_msg = f"imitation_example exceeds the CEFR {band} ceiling of the CEFR {src_cefr_level} source passage: {offenders}"
            reasons.append(reason_msg)
            flags.append(f"❌ Grammar pattern '{pattern[:40]}': {reason_msg}")

    # Degenerate Slot Gate
    degenerate_slots = degenerate_grammar_slots(pattern)
    if degenerate_slots:
        reasons.append(f"degenerate formula slot {degenerate_slots} names the same constituent twice")
        flags.append(
            f"❌ Grammar pattern '{pattern[:40]}': degenerate formula slot {degenerate_slots} "
            "(a slot may not be offered to itself)"
        )

    # Skeleton Alignment Gate
    aligned, wanted_formula = align_formula_fn(pattern, category)
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

    # Auto-Remap explicit category mismatches
    _, remap_notice = auto_remap_grammar_category(item)
    if remap_notice:
        flags.append(remap_notice)
        category = str(item.get("category", "")).strip()

    # Four Macro Functional Domains deterministic checks
    if not item.get("syntax_topic") and category == "Rhetoric & Emphasis":
        has_inversion_trigger = bool(re.search(r"\b(?:only|never|hardly|scarcely|seldom|rarely|little|not only|neither|nor|no sooner)\b", quote, re.IGNORECASE))
        has_cleft = bool(re.search(r"\bIt\s+(?:is|was|were|'s|has\s+been|had\s+been)\b", quote, re.IGNORECASE) and re.search(r"\b(?:that|who|whom|which)\b", quote, re.IGNORECASE))
        has_wh_cleft = bool(re.search(r"\bWhat\s+[a-z0-9_']+\s+(?:is|was|were)\b", quote, re.IGNORECASE))
        has_parallel_coordination = bool(
            re.search(r"\b(?:not only\b.*?\bbut\b|either\b.*?\bor\b|neither\b.*?\bnor\b|both\b.*?\band\b|not\s+.*?\s*,\s*but\b)\b", quote, re.IGNORECASE)
            or ";" in quote or "," in quote
        )
        if not (has_inversion_trigger or has_cleft or has_wh_cleft or has_parallel_coordination):
            reasons.append("category 'Rhetoric & Emphasis' assigned to quote lacking rhetorical markers (inversion, cleft focus, or structural balance)")

    elif not item.get("syntax_topic") and category == "Logic & Stance":
        has_cond = bool(re.search(r"\b(?:if|unless|provided\s+that|supposing|as\s+long\s+as|had\s+\w+\s+\w+|should\s+\w+\s+\w+|were\s+\w+\s+\w+)\b", quote, re.IGNORECASE))
        has_concessive = bool(re.search(r"\b(?:although|though|even though|even if|while|whereas|despite|in spite of)\b", quote, re.IGNORECASE))
        has_stance_or_hedging = bool(re.search(r"\b(?:can|cannot|can't|could|may|might|must|should|would|will|suggest|indicate|appear|seem|likely|probably|presumably|arguably|tends?\s+to)\b", quote, re.IGNORECASE))
        if not (has_cond or has_concessive or has_stance_or_hedging):
            reasons.append("category 'Logic & Stance' assigned to quote lacking logical condition, concessive linker, or epistemic/stance modal assertion")

    elif not item.get("syntax_topic") and category == "Information Packaging":
        ING_NON_VERBS = frozenset({"morning", "evening", "thing", "something", "nothing", "everything", "anything", "ring", "spring", "king", "wing", "sing", "during", "ceiling"})
        ing_tokens = [w.lower() for w in re.findall(r"\b(?:being|doing|going|[a-zA-Z]{3,}ing)\b", quote) if w.lower() not in ING_NON_VERBS]
        has_participle = bool(ing_tokens or re.search(r"\b(?:having\s+\w+(?:ed|en|t)|surrounded|driven|given|taken|seen|known|based|built|born|located|situated|reminded|confronted|faced|with\s+\w+)\b", quote, re.IGNORECASE))
        TO_NOUNS = frozenset({"school", "work", "bed", "church", "college", "prison", "jail", "court", "sea", "town", "home", "market", "class", "them", "him", "her", "us", "me", "you", "it", "this", "that", "these", "those"})
        to_matches = [m.group(1).lower() for m in re.finditer(r"\bto\s+([a-z]{2,})\b", quote, re.IGNORECASE)]
        has_infinitive_clause = any(word not in TO_NOUNS for word in to_matches)
        has_dummy_it = bool(re.search(r"\bit(?:'s|\s+is|\s+was|\s+has\s+been)\s+(?:[a-z]{4,}\s+)?(?:that|to\s+[a-z]+)\b", quote, re.IGNORECASE))
        has_nominalization = bool(re.search(r"\b[a-z]{3,}(?:tion|sion|ment|ance|ence|ity|ness)\b", quote, re.IGNORECASE))
        has_elaborative_relative = bool(re.search(r",\s*which\s+[a-z]+|\b(?:which|that|who|whom)\s+[a-z]+", quote, re.IGNORECASE))
        has_complex_trans_or_passive = bool(re.search(r"\b(?:make|makes|made|find|finds|found|keep|keeps|kept|render|renders|rendered|deem|deems|deemed|let|lets|have|has|had|is|are|was|were|be|been)\b", quote, re.IGNORECASE))
        if not (has_participle or has_infinitive_clause or has_dummy_it or has_nominalization or has_elaborative_relative or has_complex_trans_or_passive):
            reasons.append("category 'Information Packaging' assigned to quote lacking non-finite clauses, evaluative it-extraposition, dense nominalization, or elaborative clause")

    elif not item.get("syntax_topic") and category == "Cohesion & Framing":
        has_shell_frame = bool(re.search(r"\b(?:the\s+(?:fact|idea|notion|reason|belief|claim|argument|possibility|question|view|conclusion)\s+that|this\s+(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|led\s+to|leads\s+to)|the\s+extent\s+to\s+which)\b", quote, re.IGNORECASE))
        has_anaphoric_encapsulation = bool(re.search(r"\b(?:this|these|such)\s+[a-z]{4,}\b", quote, re.IGNORECASE) or bool(re.search(r"\bthat\s+is\s+why\b", quote, re.IGNORECASE)))
        INTERPRETIVE_VERBS_REGEX = (
            r"(?:(?:would|could|might|may|can|will)\s+)?"
            r"(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?)"
        )
        has_which_propositional = bool(re.search(rf",\s*which\s+{INTERPRETIVE_VERBS_REGEX}\b", quote, re.IGNORECASE))
        has_rel_frame = bool(re.search(r"\b(?:in\s+which|by\s+which|through\s+which|whereby)\b", quote, re.IGNORECASE))
        if not (has_shell_frame or has_anaphoric_encapsulation or has_which_propositional or has_rel_frame):
            reasons.append("category 'Cohesion & Framing' assigned to quote lacking abstract shell frame, prepositional relative, or discourse encapsulation")

    # Boundary tolerance
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
        _ing_tokens = [w.lower() for w in re.findall(r"\b(?:being|doing|going|[a-zA-Z]{3,}ing)\b", quote) if w.lower() not in _ING_EXCLUDE]
        _to_words = [m.group(1).lower() for m in re.finditer(r"\bto\s+([a-z]{2,})\b", quote, re.IGNORECASE)]
        _fits_info = (
            _ing_tokens
            or re.search(r"\b(?:having\s+\w+(?:ed|en|t)|surrounded|driven|given|taken|seen|known|based|built|born|located|situated|reminded|confronted|faced|with\s+\w+)\b", quote, re.IGNORECASE)
            or any(w not in _TO_EXCLUDE for w in _to_words)
            or re.search(r"\bit(?:'s|\s+is|\s+was|\s+has\s+been)\s+(?:[a-z]{4,}\s+)?(?:that|to\s+[a-z]+)\b", quote, re.IGNORECASE)
            or re.search(r"\b[a-z]{3,}(?:tion|sion|ment|ance|ence|ity|ness)\b", quote, re.IGNORECASE)
            or re.search(r",\s*which\s+[a-z]+|\b(?:which|that|who|whom)\s+[a-z]+", quote, re.IGNORECASE)
            or re.search(r"\b(?:make|makes|made|find|finds|found|keep|keeps|kept|render|renders|rendered|deem|deems|deemed|let|lets|have|has|had|is|are|was|were|be|been)\b", quote, re.IGNORECASE)
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

    # Trailing ellipsis check
    if quote.rstrip().endswith(("...", "…")):
        flags.append(f"⚠️ Quote ends with trailing ellipsis: '{quote[:40]}...'")

    if not reasons:
        return True, flags
    flags.append(f"⚠️ Grammar pattern '{pattern[:40]}' failed pedagogy check: {', '.join(reasons)}")
    return False, flags


def validate_summary_item(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single summary concept item."""
    flags: List[str] = []
    name = _safe_str(item.get("concept_name"))
    sig = _safe_str(item.get("educational_significance"))
    details = item.get("key_details", [])
    sub_concepts = item.get("sub_concepts", [])

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
        return True, flags
    reason = "missing educational significance or key details" if not (name and sig and isinstance(details, list) and len(details) > 0) else "malformed sub_concepts"
    flags.append(f"⚠️ Concept '{name}' {reason}")
    return False, flags


def validate_mindmap_item(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single mindmap branch item."""
    flags: List[str] = []
    name = _safe_str(item.get("branch_name"))
    if name:
        return True, flags
    flags.append("⚠️ MindMap branch missing branch name")
    return False, flags


def validate_expert_audit_item(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single expert audit item."""
    flags: List[str] = []
    raw_distractors = item.get("distractors", [])
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
    feedback_ok = bool(feedback) if (score < 90 or not single_valid) else True
    triage = item.get("triage_action", "PASS")
    cured_q = item.get("cured_question")
    is_valid_triage = triage in ("PASS", "REPAIR", "REWRITE")
    cured_ok = bool(cured_q) if triage in ("REPAIR", "REWRITE") else True

    if is_valid_dist and is_valid_idx and feedback_ok and is_valid_triage and cured_ok:
        return True, flags

    reasons = []
    if not is_valid_dist:
        reasons.append(f"distractor count != 1 to 4 (got {len(distractors) if isinstance(distractors, list) else 0})")
    if not is_valid_idx:
        reasons.append("invalid blind solve index")
    if not feedback_ok:
        reasons.append("missing diagnostic feedback for defective item")
    if not is_valid_triage:
        reasons.append(f"invalid triage_action '{triage}'")
    if not cured_ok:
        reasons.append(f"missing cured_question for triage '{triage}'")
    flags.append(f"⚠️ Audit item failed validation: {', '.join(reasons)}")
    return False, flags
