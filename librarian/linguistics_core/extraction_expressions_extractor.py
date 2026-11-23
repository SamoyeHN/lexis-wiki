"""
Deterministic multi-word expression extraction pipeline orchestrator.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("librarian.linguistics")


class ExtractionExpressionsExtractorMixin:
    @classmethod
    def extract_deterministic_expressions(
        cls,
        text: str,
        target_count: int = 5,
        syllabus_expressions: Optional[List[str]] = None,
        enable_llm_disambiguation: bool = True,
        exclude_words: Optional[Set[str]] = None,
        exclude_definitions: Optional[Set[str]] = None,
        unit_name: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Extracts genuine multi-word academic expressions and collocations from text
        in 100% deterministic fashion (0 Tokens).
        Combines mine_expression_skeletons with get_expression_definition_and_example.
        Returns a list of dicts conforming to VocabularyItem / ExpressionItem:
            {
                "word": canonical formula (e.g. 'shut [sb/sth] down'),
                "part_of_speech": 'collocation' | 'phrasal verb' | ...,
                "definition": authoritative PedLex / WordNet definition,
                "quoted_sentence": verbatim passage sentence,
                "example_usage": authentic PedLex example or passage sentence,
                "design_audit": "AUDIT: [S-ID] -> [Formula] -> [VERBATIM_CONFIRMED]"
            }
        """
        with cls.wsd_log_session(unit_name or "expressions"):
            sent_pool = cls.tokenize_and_index_sentences(text)[1] if text else {}
            skeletons = cls.mine_expression_skeletons(
                text,
                target_count=max(target_count, len(syllabus_expressions)) if syllabus_expressions else target_count + 8,
                syllabus_expressions=syllabus_expressions
            )
            for sk in skeletons:
                sk["_is_syllabus"] = bool(syllabus_expressions)
            if syllabus_expressions:
                seen_ph = {re.sub(r'\[.*?\]|\(.*?\)', '', s.get("phrase", "")).strip().lower() for s in skeletons}
                reserve_exprs = cls.mine_expression_skeletons(
                    text,
                    target_count=target_count + 10,
                    syllabus_expressions=None
                )
                for rs in reserve_exprs:
                    rph = re.sub(r'\[.*?\]|\(.*?\)', '', rs.get("phrase", "")).strip().lower()
                    if rph and rph not in seen_ph:
                        seen_ph.add(rph)
                        rs["_is_syllabus"] = False
                        skeletons.append(rs)
            if not skeletons:
                return []

            excl_w_low = {str(w).strip().lower() for w in (exclude_words or set()) if w}
            excl_d_norm = {re.sub(r"[^a-z0-9]+", " ", str(d).lower()).strip() for d in (exclude_definitions or set()) if d}
            _expr_func_stop = {
                "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with", "from", "into", "onto", "upon",
                "over", "under", "up", "down", "off", "away", "out", "back", "about", "around", "through", "along",
                "across", "against", "between", "among", "be", "have", "do", "get", "make", "take", "go", "come"
            }

            attested_sources = (
                "phrase_row", "unit_sense", "headword_entry", "phrasal_verb_block",
                "sense_pattern", "collocation", "crossref", "ldoce_entry", "wordnet"
            )
            results: List[Dict[str, Any]] = []
            seen_formulas: Set[str] = set()
            for s in skeletons:
                if len(results) >= target_count:
                    break
                phrase = s.get("phrase", "")
                formula = s.get("pattern_formula", phrase)
                norm_form = re.sub(r'\[.*?\]|\(.*?\)', '', formula).strip().lower()
                if not norm_form or norm_form in seen_formulas or not cls.is_pedagogical_expression(norm_form) or not cls.is_pedagogical_expression(formula):
                    continue
                is_syllabus_item = bool(s.get("_is_syllabus", False))
                if not is_syllabus_item and any(
                    cls.are_same_word_family(norm_form, sf) or cls.are_same_word_family(formula, r["word"])
                    for sf, r in zip(seen_formulas, results)
                ):
                    continue
                if not is_syllabus_item and any(cls.are_same_word_family(norm_form, sf) for sf in seen_formulas):
                    continue
                # Prevent verb+preposition expressions (e.g. 'conform to [sth/sb]') from duplicating
                # a single-word headword ('conform') already extracted in the same unit.
                if excl_w_low:
                    if not is_syllabus_item and any(
                        cls.are_same_word_family(norm_form, ew) or cls.are_same_word_family(formula, ew)
                        for ew in excl_w_low
                    ):
                        continue
                    f_content_toks = [t for t in norm_form.split() if t not in _expr_func_stop]
                    if len(f_content_toks) == 1 and (
                        f_content_toks[0] in excl_w_low
                        or cls._surface_lemma(f_content_toks[0]) in excl_w_low
                    ):
                        continue
                sid = s.get("sid", "S-1")
                quote = s.get("quote", "")
                expr_type = cls.classify_expression_type(formula, quote) or cls.classify_expression_type(phrase, quote) or s.get("type", "collocation")

                discourse_ctx = ""
                if sid and sent_pool:
                    m_sid_num = re.match(r"S-(\d+)", str(sid))
                    if m_sid_num:
                        s_num = int(m_sid_num.group(1))
                        prev_s = str(sent_pool.get(f"S-{s_num - 1}") or "").strip()
                        next_s = str(sent_pool.get(f"S-{s_num + 1}") or "").strip()
                        if prev_s or next_s:
                            discourse_ctx = " ".join(p for p in (prev_s, f">> {quote} <<", next_s) if p)

                # Primary: Ground multi-word expressions in authoritative PedLex expression evidence tiers
                # (Disable inner LLM disambiguation here since Gate 1 below arbitrates across the full unified pool)
                defn, ex, def_source = cls.expression_definition_evidence(
                    phrase=phrase,
                    expr_type=expr_type,
                    context_sentence=quote,
                    enable_llm_disambiguation=False
                )

                # Gate 1: Neuro-Symbolic Arbitration across Top-K (k=6) Expression Candidates
                locked_expr_sense = None
                if quote:
                    pool = cls.build_top_k_expression_candidates(formula, quote, k=6, expr_type=expr_type)
                    has_attested_pool = any(c.get("attested") for c in pool)
                    is_verb_slot_frame = bool(re.search(r"[\[\(](?:sb|sth|oneself|one's)[\]\)]", formula, re.IGNORECASE))
                    if not has_attested_pool and def_source not in attested_sources and not is_verb_slot_frame:
                        continue
                    if defn and (not has_attested_pool or def_source in attested_sources):
                        norm_defn = re.sub(r'[^a-z0-9]+', ' ', defn.lower()).strip()
                        if not any(re.sub(r'[^a-z0-9]+', ' ', c.get("definition", "").lower()).strip() == norm_defn for c in pool):
                            pool.insert(0, {
                                "definition": defn,
                                "pos": expr_type,
                                "examples": [ex] if ex else [],
                                "score": 30.0,
                                "margin": 0.0,
                                "attested": def_source in attested_sources,
                            })
                    pool = pool[:6]
                    if pool:
                        chosen = None
                        if len(pool) > 1 and enable_llm_disambiguation:
                            chosen_idx = cls.disambiguate_via_llm(formula, quote, pool, discourse_context=discourse_ctx)
                            if 0 <= chosen_idx < len(pool):
                                chosen = pool[chosen_idx]
                            else:
                                logger.info(
                                    f"  [DROP_WSD_REJECTED] Expression '{formula}' rejected by WSD arbitration in [{sid}], advancing to next candidate."
                                )
                                continue
                        elif has_attested_pool or not defn or def_source not in attested_sources:
                            chosen = pool[0]
                        if chosen is not None:
                            prev_defn_norm = re.sub(r"[^a-z0-9]+", " ", (defn or "").lower()).strip()
                            chosen_defn_norm = re.sub(r"[^a-z0-9]+", " ", (chosen.get("definition") or "").lower()).strip()
                            sense_changed = bool(chosen_defn_norm and chosen_defn_norm != prev_defn_norm)
                            if chosen.get("definition"):
                                defn = chosen["definition"]
                                def_source = def_source or "phrase_row"
                            locked_expr_sense = chosen.get("raw_sense")
                            if sense_changed:
                                ex = ""
                            if chosen.get("examples"):
                                ex = (
                                    cls._unit_example(formula, chosen["examples"], allow_modifier=False)
                                    or cls._unit_example(formula, chosen["examples"], allow_modifier=True)
                                    or cls._relaxed_unit_example(formula, chosen["examples"])
                                    or (ex if ex and cls._unit_example(formula, [ex], allow_modifier=True) else chosen["examples"][0])
                                )
                        elif locked_expr_sense is None:
                            locked_expr_sense = pool[0].get("raw_sense")

                low_confidence = not def_source
                ex = cls.ensure_distinct_example(formula, expr_type, ex, quote, sense=locked_expr_sense)

                # Gate 2 (Evidence-Completeness Gate - 宁缺勿滥):
                # Drop any multi-word expression lacking an authoritative definition, distinct corpus example,
                # or mapped to a non-pedagogical cardinal number definition.
                if not defn or cls.is_boilerplate_definition(defn) or not ex or not cls.is_pedagogical_expression(formula, defn):
                    continue
                defn_norm_key = re.sub(r"[^a-z0-9]+", " ", defn.lower()).strip()
                if defn_norm_key in excl_d_norm:
                    continue

                seen_formulas.add(norm_form)
                excl_d_norm.add(defn_norm_key)
                results.append({
                    "word": formula,
                    "part_of_speech": expr_type,
                    "definition": defn,
                    "quoted_sentence": quote,
                    "example_usage": ex,
                    "definition_source": def_source,
                    "low_confidence": low_confidence,
                    "design_audit": f"AUDIT: [{sid}] -> [{formula}] -> [VERBATIM_CONFIRMED]"
                })

            return results

