"""
Deterministic extraction pipeline orchestration (run_pipeline), grammar formula normalization, and wiki Q&A.
"""

import random
import os
import json
import re
import dataclasses
import base64
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor
from ..config import config
from ..llm import llm
from ..prompts import Prompts
from ..linguistics import LinguisticEngine

logger = logging.getLogger("librarian.processor")

from ..schemas import (
    VocabularyExtraction, GrammarExtraction, SummaryExtraction,
    VocabularyItem, GrammarItem,
    VocabularyQuiz, ReadingQuiz, TranslationQuiz, ListeningQuiz,
    RoutingResult, MindMapExtraction,
    validate_and_map
)

class ExtractionRunnerMixin:

    # Normalization mapping from legacy/verbose grammar slots to standard COBUILD tokens
    GRAMMAR_SLOT_NORMALIZATION: Dict[str, str] = {
        "clause": "[S]",
        "main clause": "[S]",
        "subordinate clause": "[S]",
        "dependent clause": "[S]",
        "subject": "[S]",
        "predicate": "[S]",
        "predicate verb": "[V]",
        "complement/object": "[NP/adj]",
        "noun phrase": "[NP]",
        "noun/np": "[NP]",
        "verb phrase": "[VP]",
        "focus element": "[Focal Element]",
        "focal element": "[Focal Element]",
        "object": "[NP]",
        "complement": "[NP/adj]",
        "copula": "[be]",
        "auxiliary": "[aux]",
        "modal": "[aux]",
        "past participle": "[V3]",
        "v-ed": "[V3]",
        "verb-ed": "[V3]",
        "present participle": "[V-ing]",
        "gerund": "[V-ing]",
        "verb-ing": "[V-ing]",
        "v-ing / gerund phrase": "[V-ing Phrase]",
        "base verb": "[V]",
        "finite verb": "[V]",
        "infinitive": "[to-V]",
        "adjective": "[adj]",
        "evaluative adjective": "[adj]",
        "comparative": "[adj]",
        "adverb": "[adv]",
        "preposition": "[prep]",
        "prepositional phrase": "[prep] + [NP]",
    }

    # Standard literal functional words/connectors commonly wrapped in brackets by LLMs
    LITERAL_FUNCTIONAL_WORDS = frozenset({
        "the", "a", "an", "that", "this", "these", "those", "such", "it", "there",
        "if", "then", "when", "while", "as", "since", "because", "although", "though", "even though",
        "to", "for", "of", "with", "by", "at", "in", "on", "from", "into", "onto",
        "have", "has", "had", "have to", "has to", "had to", "do", "does", "did",
        "not", "no", "never", "only", "also", "too", "either", "neither", "nor", "both",
        "so", "such that", "so that", "more", "most", "less", "least", "than"
    })

    # Canonical casing for standard COBUILD slots
    CANONICAL_SLOT_CASING = {
        "s": "[S]", "np": "[NP]", "vp": "[VP]", "v": "[V]", "be": "[be]", "aux": "[aux]",
        "v-ed": "[V3]", "v3": "[V3]", "v-ing": "[V-ing]", "to-v": "[to-V]",
        "adj": "[adj]", "adv": "[adv]", "prep": "[prep]", "det": "[det]", "conj": "[conj]",
        "clause": "[S]", "focal element": "[Focal Element]"
    }

    @classmethod
    def unwrap_literal_brackets(cls, formula: str) -> str:
        """
        Unwraps accidental square brackets around literal words/connectors in formulas.
        E.g. '[If] [S] [to-V], [then] [S] [have to] [V]' -> 'If [S] [to-V], then [S] have to [V]'
        """
        if not formula or "[" not in formula:
            return formula

        from ..evaluator import ALLOWED_GRAMMAR_SLOTS

        def _repl(match: re.Match) -> str:
            raw_slot = match.group(1).strip()
            slot_lower = raw_slot.lower()
            if slot_lower in cls.GRAMMAR_SLOT_NORMALIZATION:
                return cls.GRAMMAR_SLOT_NORMALIZATION[slot_lower]
            if "/" in slot_lower:
                sub_parts = [p.strip() for p in slot_lower.split("/") if p.strip()]
                from ..evaluator import degenerate_grammar_slots
                if degenerate_grammar_slots(f"[{raw_slot}]"):
                    return f"[{raw_slot}]"

                if all(p in ALLOWED_GRAMMAR_SLOTS or p in cls.GRAMMAR_SLOT_NORMALIZATION for p in sub_parts):
                    norm_parts = [
                        cls.GRAMMAR_SLOT_NORMALIZATION.get(p, cls.CANONICAL_SLOT_CASING.get(p, f"[{p}]")).strip("[]")
                        for p in sub_parts
                    ]
                    return f"[{'/'.join(norm_parts)}]"
            if slot_lower in cls.CANONICAL_SLOT_CASING:
                return cls.CANONICAL_SLOT_CASING[slot_lower]
            if slot_lower in ALLOWED_GRAMMAR_SLOTS:
                return f"[{raw_slot}]"
            if slot_lower in cls.LITERAL_FUNCTIONAL_WORDS:
                # It is a literal word or connective phrase (e.g. that, it, if, then, the, such, to, have to)
                # Unwrap bracket so it serves as natural functional anchor
                return raw_slot
            # Keep original bracket for unrecognized slot so evaluator catches it
            return f"[{raw_slot}]"

        unwrapped = re.sub(r'\[(.*?)\]', _repl, formula)
        return re.sub(r'\s+', ' ', unwrapped).strip()

    @classmethod
    def normalize_grammar_formula(cls, formula: str) -> str:
        """
        Normalizes legacy or verbose bracketed slots into concise, standard COBUILD tokens,
        and unwraps literal words from square brackets.
        """
        return cls.unwrap_literal_brackets(formula)

    def run_pipeline(self, source_filename: str, categories: List[str] = None) -> Tuple[str, List[str]]:
        """
        Modular pipeline for processing a raw unit file.
        Runs Vocabulary, Grammar, and Concept extractions in parallel for efficiency.
        """
        from ..config import normalize_name
        import shutil

        input_path = Path(source_filename)
        wiki_dir = self.config.wiki_content_path

        if input_path.exists() and input_path.is_file():
            # Explicit file path on disk (either absolute or relative)
            input_stem = input_path.stem
            normalized_input = normalize_name(input_stem)

            # Determine the canonical unit folder name
            unit_folder_name = None
            try:
                resolved_input = input_path.resolve()
                resolved_wiki = wiki_dir.resolve()
                if resolved_input.is_relative_to(resolved_wiki):
                    rel_parts = resolved_input.relative_to(resolved_wiki).parts
                    if len(rel_parts) >= 3 and rel_parts[1] == "sources":
                        unit_folder_name = rel_parts[0]
            except Exception:
                pass

            if not unit_folder_name:
                unit_folder_name = input_stem.replace(" ", "_")
                if wiki_dir.exists():
                    for child in wiki_dir.iterdir():
                        if child.is_dir() and normalize_name(child.name) == normalized_input:
                            unit_folder_name = child.name
                            break

            # Define paths
            unit_dir = wiki_dir / unit_folder_name
            sources_dir = unit_dir / "sources"
            extractions_dir = unit_dir / "extractions"
            handouts_dir = unit_dir / "handouts"

            sources_dir.mkdir(parents=True, exist_ok=True)
            extractions_dir.mkdir(parents=True, exist_ok=True)
            handouts_dir.mkdir(parents=True, exist_ok=True)

            # Compile directly from the original explicit file path without copying to sources_dir
            source_path = input_path
        else:
            # Treated as a unit name or filename under existing units
            input_stem = Path(source_filename).stem
            normalized_input = normalize_name(source_filename)

            unit_folder_name = None
            if wiki_dir.exists():
                for child in wiki_dir.iterdir():
                    if child.is_dir() and (normalize_name(child.name) == normalized_input or normalize_name(child.name) == normalize_name(input_stem)):
                        unit_folder_name = child.name
                        break

            if not unit_folder_name:
                return f"Error: Source file or unit folder '{source_filename}' not found.", []

            unit_dir = wiki_dir / unit_folder_name
            sources_dir = unit_dir / "sources"
            extractions_dir = unit_dir / "extractions"
            handouts_dir = unit_dir / "handouts"

            sources_dir.mkdir(parents=True, exist_ok=True)
            extractions_dir.mkdir(parents=True, exist_ok=True)
            handouts_dir.mkdir(parents=True, exist_ok=True)

            # Locate the source file in sources_dir
            text_sources = [f for f in sources_dir.iterdir() if f.is_file() and f.suffix in [".md", ".txt"]]
            if len(text_sources) == 1:
                source_path = text_sources[0]
            elif len(text_sources) > 1:
                matching = [f for f in text_sources if normalize_name(f.stem) == normalized_input]
                source_path = matching[0] if matching else text_sources[0]
            else:
                return f"Error: No source markdown or text file found under unit sources for {unit_folder_name}.", []

        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = LinguisticEngine.sanitize_source_markdown(content, source_path=source_path)

        # Ensure unit's sources/ directory holds the sanitized canonical source when imported from an external path
        existing_unit_sources = [
            f for f in sources_dir.iterdir() if f.is_file() and f.suffix.lower() in (".md", ".txt")
        ]
        if not existing_unit_sources:
            ext = source_path.suffix if source_path.suffix.lower() in (".md", ".txt") else ".md"
            unit_source_file = sources_dir / f"{unit_folder_name}{ext}"
            unit_source_file.write_text(content, encoding="utf-8")
            source_path = unit_source_file

        # Plan 1 (Single Source): Parse optional syllabus sections directly from source markdown
        clean_content, syllabus_vocab, syllabus_grammar, syllabus_expressions = self.parse_syllabus_sections(content)
        self._last_source_content = clean_content or content
        file_stem = unit_folder_name
        all_saved = []
        
        # Determine extraction counts from compile_defaults config (priority: wiki_config.json -> config.py DEFAULT_CONFIG -> fallback)
        compile_defaults = self.config.get("compile_defaults") or {}
        v_count = compile_defaults.get("vocabulary", 20)
        e_count = compile_defaults.get("expressions", 5)
        g_count = compile_defaults.get("grammar", 5)
        c_count = compile_defaults.get("concepts", 3)
        max_p = compile_defaults.get("max_parallel", 3)

        # 1. Prepare Extraction Tasks (Summary & MindMap use LLM; Vocabulary, Expressions & Grammar are 100% deterministic)
        s_prompt_template, s_schema = Prompts.get("extract_summary")
        m_prompt_template, m_schema = Prompts.get("extract_mindmap")

        # Pre-tokenize source text into an indexed sentence pool ([S-1], [S-2]) using spaCy
        raw_source_text = clean_content or content
        indexed_content, sentence_pool = LinguisticEngine.tokenize_and_index_sentences(raw_source_text)
        self._last_sentence_pool = sentence_pool

        target_vocab_count = v_count
        target_expr_count = e_count
        if syllabus_vocab:
            logger.info(f"📋 Detected {len(syllabus_vocab)} syllabus vocabulary item(s) in source markdown (default target count: {v_count}).")
        if syllabus_expressions:
            logger.info(f"📋 Detected {len(syllabus_expressions)} syllabus expression/phrase item(s) in source markdown (default target count: {e_count}).")
        if syllabus_grammar:
            logger.info(f"📋 Detected {len(syllabus_grammar)} syllabus grammar pattern(s) in source markdown.")

        s_kwargs = {"content": raw_source_text, "count": c_count}
        m_kwargs = {"content": raw_source_text}

        tasks = [
            ("summary", s_prompt_template.format(**s_kwargs), self._interpolate_schema(s_schema, s_kwargs)),
            ("mindmap", m_prompt_template.format(**m_kwargs), self._interpolate_schema(m_schema, m_kwargs))
        ]

        run_vocab = False
        run_expressions = False
        run_grammar = False
        if categories:
            normalized_cats = [c.lower() for c in categories]
            task_names_to_run = []
            for cat in normalized_cats:
                if cat == "vocabulary":
                    run_vocab = True
                    run_expressions = True
                elif cat == "expressions":
                    run_expressions = True
                elif cat == "grammar":
                    run_grammar = True
                else:
                    task_names_to_run.append(cat)
            tasks = [t for t in tasks if t[0] in task_names_to_run]
        else:
            run_vocab = True
            run_expressions = True
            run_grammar = True

        # Pre-Flight LLM Availability Gate:
        # Fail fast before running WSD or generative extraction tasks if the LLM endpoint/model is unreachable.
        needs_llm = bool(tasks) or run_vocab or run_expressions
        if needs_llm:
            preflight_task = (
                f"extract_wsd_{file_stem}"
                if (run_vocab or run_expressions)
                else f"extract_{tasks[0][0]}_{file_stem}"
            )
            try:
                llm.ensure_available(
                    task_name=preflight_task,
                    prompt_summary=(
                        f"Extraction pipeline pre-flight check for '{source_filename}' "
                        f"(categories={categories or ['vocabulary', 'expressions', 'grammar', 'summary', 'mindmap']})"
                    ),
                )
            except Exception as pre_err:
                logger.error(f"Pre-flight LLM check failed for {source_filename}: {pre_err}")
                return f"Error: {pre_err}", all_saved

        # Deterministic Expressions & Vocabulary Extraction
        # Run Expressions FIRST so we know the exact set of multi-word expressions that actually pass
        # Gate 1 (Sense Locking) and Gate 2 (Evidence Completeness) before holding back or padding single words.
        det_vocab_data = None
        det_expressions_data = None
        try:
            with LinguisticEngine.wsd_log_session(file_stem):
                e_items_list = []
                shipped_parent_by_atom: Dict[str, str] = {}
                shipped_parent_quotes: Dict[str, Set[str]] = {}
                shipped_atoms: Set[str] = set()

                if run_expressions:
                    logger.info(f"⚡ Deterministic Expressions Extraction executing locally via ACL & Pedagogical Lexicon/WordNet (0 tokens)...")
                    from ..schemas import ExpressionsExtraction, ExpressionItem
                    raw_e_items = LinguisticEngine.extract_deterministic_expressions(
                        raw_source_text,
                        target_count=target_expr_count,
                        syllabus_expressions=syllabus_expressions,
                    )
                    for it in raw_e_items:
                        if not str(it.get("definition", "")).strip() and syllabus_expressions:
                            w_raw = it.get("word", "")
                            clean_w = re.sub(r'\[.*?\]|\(.*?\)', '', w_raw).strip().lower()
                            if 'end of' in clean_w:
                                sub_def, sub_ex, sub_src = LinguisticEngine.expression_definition_evidence('the end of', 'set phrase')
                                if sub_def and sub_ex:
                                    it["definition"] = sub_def
                                    it["example_usage"] = sub_ex
                                    it["definition_source"] = sub_src
                                    logger.info(f"🎯 Grounded syllabus expression '{w_raw}' via sub-phrase 'the end of' ({sub_src})")

                        # Evidence-Completeness Gate: ship no expression lacking an authoritative definition or distinct corpus example.
                        if not str(it.get("definition", "")).strip() or not str(it.get("example_usage", "")).strip():
                            logger.warning(
                                f"🚫 Gate 2 (DROP_INCOMPLETE_LEXICON): incomplete lexicon evidence for '{it.get('word', '')}' "
                                f"(source tier: '{it.get('definition_source', '')}') - expression dropped"
                            )
                            continue
                        audit = it.get("design_audit", "")
                        if it.get("definition_source"):
                            audit = f"{audit} -> [DEF: {it['definition_source']}]"
                        e_item = ExpressionItem(
                            design_audit=audit,
                            word=it["word"],
                            definition=it["definition"],
                            quoted_sentence=it["quoted_sentence"],
                            example_usage=it["example_usage"],
                            part_of_speech=it.get("part_of_speech", "collocation")
                        )
                        e_items_list.append(e_item)

                    # Backfill genuine expressions from text when e_items_list has fewer than target_expr_count
                    if len(e_items_list) < target_expr_count:
                        existing_expr_tokens = {
                            re.sub(r'\[.*?\]|\(.*?\)', '', it.word).strip().lower()
                            for it in e_items_list
                        }
                        supplementary_items = LinguisticEngine.extract_deterministic_expressions(
                            raw_source_text,
                            target_count=target_expr_count + 5,
                            syllabus_expressions=None,
                        )
                        for sup in supplementary_items:
                            if len(e_items_list) >= target_expr_count:
                                break
                            sup_clean = re.sub(r'\[.*?\]|\(.*?\)', '', sup.get("word", "")).strip().lower()
                            if sup_clean not in existing_expr_tokens and str(sup.get("definition", "")).strip():
                                audit = sup.get("design_audit", "")
                                if sup.get("definition_source"):
                                    audit = f"{audit} -> [DEF: {sup['definition_source']}]"
                                e_items_list.append(ExpressionItem(
                                    design_audit=audit,
                                    word=sup["word"],
                                    definition=sup["definition"],
                                    quoted_sentence=sup["quoted_sentence"],
                                    example_usage=sup["example_usage"],
                                    part_of_speech=sup.get("part_of_speech", "collocation")
                                ))
                                existing_expr_tokens.add(sup_clean)
                                logger.info(f"✨ Backfilled genuine expression from text: '{sup['word']}' ({sup.get('definition_source')})")

                    _DELEXICAL_LIGHT_VERBS = {"give", "make", "take", "have", "do", "get"}
                    _nlp_delex = LinguisticEngine.get_spacy()
                    for e_row in e_items_list:
                        _e_atoms = self.expression_constituent_atoms(e_row.word)
                        for atom in _e_atoms:
                            shipped_atoms.add(atom)
                            shipped_parent_by_atom.setdefault(atom, e_row.word)
                            if e_row.quoted_sentence:
                                shipped_parent_quotes.setdefault(atom, set()).add(e_row.quoted_sentence.strip())
                        if e_row.quoted_sentence and _nlp_delex and _e_atoms:
                            _doc_e = _nlp_delex(e_row.quoted_sentence)
                            for _tok in _doc_e:
                                if _tok.lemma_.lower() in _e_atoms or _tok.text.lower() in _e_atoms:
                                    _head = _tok.head
                                    if (
                                        _head is not None
                                        and _head != _tok
                                        and _head.pos_ == "VERB"
                                        and _head.lemma_.lower() in _DELEXICAL_LIGHT_VERBS
                                        and _tok.dep_ in ("dobj", "obj", "attr", "oprd", "acomp", "nsubjpass", "nsubj:pass", "pobj")
                                    ):
                                        _lv = _head.lemma_.lower()
                                        shipped_atoms.add(_lv)
                                        shipped_parent_by_atom.setdefault(_lv, f"{_lv} {e_row.word}")
                                        shipped_parent_quotes.setdefault(_lv, set()).add(e_row.quoted_sentence.strip())

                    det_expressions_data = ExpressionsExtraction(
                        title=f"{file_stem.replace('_', ' ')} Expressions",
                        expressions=e_items_list
                    )

                if run_vocab:
                    logger.info(f"⚡ Deterministic Vocabulary Extraction executing locally via spaCy & Pedagogical Lexicon (0 tokens)...")
                    from ..schemas import VocabularyExtraction, VocabularyItem
                    cleaned_syllabus_vocab: List[str] = []
                    held_back_atoms: Set[str] = set()
                    if syllabus_vocab:
                        unshipped_declared_atoms: Set[str] = set()
                        for expr in (syllabus_expressions or []):
                            for atom in self.expression_constituent_atoms(expr):
                                if atom not in shipped_atoms:
                                    unshipped_declared_atoms.add(atom)

                        for w in syllabus_vocab:
                            w_clean = re.sub(r'\[.*?\]|\(.*?\)', '', w).strip().lower()
                            if not w_clean or w_clean in cleaned_syllabus_vocab:
                                continue
                            if w_clean in shipped_parent_by_atom:
                                parent_expr = shipped_parent_by_atom[w_clean]
                                parent_qs = shipped_parent_quotes.get(w_clean) or set()
                                has_independent_sent = bool(
                                    sentence_pool
                                    and any(
                                        s.strip() not in parent_qs
                                        and LinguisticEngine.text_contains_form(s, w_clean, standalone=True)
                                        for s in sentence_pool.values()
                                    )
                                )
                                if not has_independent_sent:
                                    held_back_atoms.add(w_clean)
                                    logger.info(f"🚫 Holding back '{w}' from vocabulary - shipped expression '{parent_expr}' already carries it.")
                                    continue
                            elif w_clean in unshipped_declared_atoms and sentence_pool:
                                if not any(
                                    LinguisticEngine.text_contains_form(s, w_clean, standalone=True)
                                    for s in sentence_pool.values()
                                ):
                                    held_back_atoms.add(w_clean)
                                    logger.info(
                                        f"🚫 [F12 Gate] Rejecting '{w}' from vocabulary: does not exist as "
                                        f"an independent word in source text (only as part of a compound/phrase)."
                                    )
                                    continue

                            e = LinguisticEngine.get_pedlex_entry(w_clean)
                            has_pedlex = bool(e and e.get("senses"))
                            if not has_pedlex:
                                wn_mod = LinguisticEngine.get_wordnet()
                                has_pedlex = bool(wn_mod and wn_mod.synsets(w_clean))
                            if not has_pedlex:
                                logger.info(f"🚫 [Lexicon Gate] Dropped syllabus candidate '{w_clean}' because it has no standalone Pedagogical Lexicon or WordNet entry.")
                                continue
                            cleaned_syllabus_vocab.append(w_clean)

                    effective_vocab_count = (
                        max(target_vocab_count, (v_count + e_count) - len(e_items_list))
                        if run_expressions
                        else target_vocab_count
                    )
                    vocab_exclude_words = (held_back_atoms | shipped_atoms) - set(cleaned_syllabus_vocab)
                    raw_v_items = LinguisticEngine.extract_deterministic_vocabulary(
                        raw_source_text,
                        syllabus_vocab=cleaned_syllabus_vocab if syllabus_vocab else None,
                        target_count=effective_vocab_count,
                        exclude_words=vocab_exclude_words if vocab_exclude_words else None,
                    )
                    v_items_list = []
                    for it in raw_v_items:
                        v_item = VocabularyItem(
                            design_audit=it.get("design_audit", ""),
                            word=it["word"],
                            definition=it["definition"],
                            quoted_sentence=it["quoted_sentence"],
                            example_usage=it["example_usage"],
                            part_of_speech=it.get("part_of_speech", "noun")
                        )
                        v_items_list.append(v_item)
                    det_vocab_data = VocabularyExtraction(
                        title=f"{file_stem.replace('_', ' ')} Vocabulary",
                        vocabulary=v_items_list
                    )
        except Exception as wsd_err:
            logger.error(f"WSD / vocabulary extraction aborted for {source_filename}: {wsd_err}", exc_info=True)
            return f"Error: Vocabulary/Expression WSD failed for {source_filename} (no fallback allowed): {wsd_err}", all_saved

        # Deterministic Grammar Extraction (0 Tokens, Pure Computational Syntax + COBUILD Common Learner Pitfalls)
        det_grammar_data = None
        if run_grammar:
            logger.info(f"⚡ Deterministic Grammar Extraction executing locally via spaCy & COBUILD Grammar Profiles (0 tokens)...")
            from ..schemas import GrammarExtraction, GrammarItem
            target_g_count = g_count
            raw_g_items = LinguisticEngine.extract_deterministic_grammar(
                raw_source_text,
                sentence_pool=sentence_pool,
                target_count=target_g_count,
                syllabus_grammar=syllabus_grammar
            )
            g_items_list = []
            for it in raw_g_items:
                g_item = GrammarItem(
                    quote=it["quote"],
                    pattern_formula=it["pattern_formula"],
                    pedagogical_function=it["pedagogical_function"],
                    design_audit=it["design_audit"],
                    imitation_example=it["imitation_example"],
                    common_mistakes=it["common_mistakes"]
                )
                setattr(g_item, "category", it.get("category", "Information Packaging"))
                if "syntax_topic" in it:
                    setattr(g_item, "syntax_topic", it["syntax_topic"])
                g_items_list.append(g_item)
            det_grammar_data = GrammarExtraction(
                title=f"{file_stem.replace('_', ' ')} Grammar",
                grammar_patterns=g_items_list
            )

        # 2. Run remaining extractions in parallel (Direct One-Shot JSON Structured Output)
        results = []
        if det_vocab_data:
            results.append(("vocabulary", det_vocab_data))
        if det_expressions_data:
            results.append(("expressions", det_expressions_data))
        if det_grammar_data:
            results.append(("grammar", det_grammar_data))

        try:
            with ThreadPoolExecutor(max_workers=max_p) as executor:
                futures = {
                    executor.submit(llm.chat, [{"role": "user", "content": prompt}], schema=schema, task_name=f"extract_{name}_{file_stem}"): name 
                    for name, prompt, schema in tasks
                }
                failed_tasks = []
                for future in futures:
                    name = futures[future]
                    try:
                        data = future.result()
                        if data:
                            results.append((name, data))
                    except Exception as task_err:
                        failed_tasks.append((name, str(task_err)))
                        logger.error(f"Extraction task '{name}' failed for {file_stem}: {task_err}", exc_info=True)

            # 3. Process, Merge, and Save Results
            vocab_data = None
            expressions_data = None
            other_results = []

            for category, data in results:
                if category == "vocabulary":
                    vocab_data = data
                elif category == "expressions":
                    expressions_data = data
                else:
                    other_results.append((category, data))

            # Merge expressions into vocabulary if both exist
            if vocab_data:
                merged_vocab = []
                if dataclasses.is_dataclass(vocab_data):
                    merged_vocab.extend(getattr(vocab_data, "vocabulary", []))
                elif isinstance(vocab_data, dict):
                    merged_vocab.extend(vocab_data.get("vocabulary", []))

                if expressions_data:
                    from ..schemas import VocabularyItem
                    expr_list = getattr(expressions_data, "expressions", []) if dataclasses.is_dataclass(expressions_data) else expressions_data.get("expressions", [])
                    for expr in expr_list:
                        if isinstance(expr, dict):
                            mapped_item = {
                                "design_audit": expr.get("design_audit", ""),
                                "word": expr.get("word", ""),
                                "part_of_speech": expr.get("part_of_speech", "collocation"),
                                "definition": expr.get("definition", ""),
                                "quoted_sentence": expr.get("quoted_sentence", ""),
                                "example_usage": expr.get("example_usage", ""),
                            }
                        else:
                            mapped_item = VocabularyItem(
                                design_audit=getattr(expr, "design_audit", ""),
                                word=getattr(expr, "word", ""),
                                definition=getattr(expr, "definition", ""),
                                quoted_sentence=getattr(expr, "quoted_sentence", ""),
                                example_usage=getattr(expr, "example_usage", ""),
                                part_of_speech=getattr(expr, "part_of_speech", "collocation")
                            )
                        merged_vocab.append(mapped_item)

                # Filter out redundant single words in vocabulary that were extracted solely as
                # decomposed fragments of a multi-word expression (e.g. 'pro' and 'con' when 'pros and cons' is extracted)
                if expressions_data:
                    multi_words = set()
                    for it in merged_vocab:
                        w = it.word if dataclasses.is_dataclass(it) else it.get("word", "")
                        clean_w = re.sub(r'\[.*?\]|\(.*?\)', '', str(w)).strip()
                        toks = [t.lower() for t in re.findall(r'[a-zA-Z]+', clean_w)]
                        if len(toks) > 1:
                            multi_words.add(" ".join(toks))

                    if multi_words:
                        filtered_vocab = []
                        for it in merged_vocab:
                            w = it.word if dataclasses.is_dataclass(it) else it.get("word", "")
                            clean_w = re.sub(r'\[.*?\]|\(.*?\)', '', str(w)).strip()
                            toks = [t.lower() for t in re.findall(r'[a-zA-Z]+', clean_w)]
                            # If it's a single word, check if it's an artificial fragment of an existing multi-word unit
                            # from the exact same quoted sentence
                            if len(toks) == 1:
                                single_tok = toks[0]
                                it_quote = it.quoted_sentence if dataclasses.is_dataclass(it) else it.get("quoted_sentence", "")
                                clean_it_quote = re.sub(r'[^\w\s]', ' ', str(it_quote).lower())
                                
                                # Check if any multi-word unit in the SAME sentence contains this single word
                                # e.g. single 'pro' / 'con' vs 'pros and cons'
                                is_fragment = False
                                for mw in multi_words:
                                    mw_toks = mw.split()
                                    if any(single_tok == mt or (len(single_tok) >= 3 and (mt.startswith(single_tok) or single_tok.startswith(mt))) for mt in mw_toks):
                                        # Verify matching sentence or syllabus provenance
                                        for other in merged_vocab:
                                            other_w = other.word if dataclasses.is_dataclass(other) else other.get("word", "")
                                            clean_other = re.sub(r'\[.*?\]|\(.*?\)', '', str(other_w)).strip()
                                            if clean_other.lower() == mw:
                                                other_quote = other.quoted_sentence if dataclasses.is_dataclass(other) else other.get("quoted_sentence", "")
                                                if clean_it_quote and clean_it_quote == re.sub(r'[^\w\s]', ' ', str(other_quote).lower()):
                                                    # Don't prune if the single word was explicitly declared in syllabus_vocab!
                                                    declared_in_syllabus = any(single_tok == sw.lower() or single_tok == re.sub(r'[^\w]', '', sw.lower()) for sw in (syllabus_vocab or []))
                                                    if not declared_in_syllabus:
                                                        is_fragment = True
                                                        break
                                        if is_fragment:
                                            break
                                if is_fragment:
                                    logger.info(f"🧹 Pruning redundant single-word fragment '{w}' subsumed by multi-word expression '{mw}'")
                                    continue
                            filtered_vocab.append(it)
                        merged_vocab = filtered_vocab

                # Level 1 Deterministic Code Gate: Post-merge Word-Family Deduplication & Anti-Clustering
                # Guarantees that even if LLM outputted multiple words from the same family (e.g. 'recognize' & 'recognition'),
                # or crammed multiple words into adjacent sentences, the finalized output remains 100% unique per family.
                # NOTE: For multi-word expressions, only apply exact duplicate checking so syllabus phrases are never swallowed!
                deduped_family_vocab = []
                seen_families: List[str] = []
                for it in merged_vocab:
                    w = it.word if dataclasses.is_dataclass(it) else it.get("word", "")
                    clean_w = re.sub(r'\[.*?\]|\(.*?\)', '', str(w)).strip().lower()
                    if not clean_w:
                        continue
                    # Check against existing items
                    is_family_duplicate = False
                    is_multiword = len(clean_w.split()) > 1
                    for existing_w in seen_families:
                        existing_is_multi = len(existing_w.split()) > 1
                        if is_multiword or existing_is_multi:
                            # Exact string match for multi-word expressions
                            if clean_w == existing_w:
                                is_family_duplicate = True
                                logger.info(f"✂️ Code Gate: Pruning exact duplicate item '{clean_w}'")
                                break
                        else:
                            # Single-word family deduplication across all single-word items
                            if LinguisticEngine.are_same_word_family(clean_w, existing_w):
                                is_family_duplicate = True
                                logger.info(f"✂️ Code Gate: Pruning duplicate word-family item '{clean_w}' (subsumed by '{existing_w}')")
                                break
                    if is_family_duplicate:
                        continue
                    seen_families.append(clean_w)
                    deduped_family_vocab.append(it)
                merged_vocab = deduped_family_vocab

                if dataclasses.is_dataclass(vocab_data):
                    vocab_data.vocabulary = merged_vocab
                elif isinstance(vocab_data, dict):
                    vocab_data["vocabulary"] = merged_vocab

                all_saved.extend(self._save_extraction_results(vocab_data, source_path.name, category_override="vocabulary"))
            elif expressions_data:
                from ..schemas import VocabularyExtraction, VocabularyItem
                vocab_list = []
                expr_list = getattr(expressions_data, "expressions", []) if dataclasses.is_dataclass(expressions_data) else expressions_data.get("expressions", [])
                for expr in expr_list:
                    resolved_word = getattr(expr, "word", "")
                    item = VocabularyItem(
                        design_audit=getattr(expr, "design_audit", ""),
                        word=resolved_word,
                        definition=getattr(expr, "definition", ""),
                        quoted_sentence=getattr(expr, "quoted_sentence", ""),
                        example_usage=getattr(expr, "example_usage", ""),
                        part_of_speech=getattr(expr, "part_of_speech", "collocation")
                    )
                    vocab_list.append(item)
                v_extracted = VocabularyExtraction(
                    title=f"{file_stem.replace('_', ' ')} Vocabulary",
                    vocabulary=vocab_list
                )
                all_saved.extend(self._save_extraction_results(v_extracted, source_path.name, category_override="vocabulary"))

            # Save other results (grammar, summary)
            for category, data in other_results:
                all_saved.extend(self._save_extraction_results(data, source_path.name, category_override=category))

            if failed_tasks:
                failed_names = ", ".join(name for name, _ in failed_tasks)
                first_err = failed_tasks[0][1]
                return (
                    f"Error: Extraction pipeline failed for {source_filename} "
                    f"(failed tasks: {failed_names}; reason: {first_err}).",
                    all_saved,
                )

            return f"Pipeline completed for {source_filename}. Generated {len(all_saved)} files.", all_saved

        except Exception as e:
            return f"Pipeline Error: {e}", all_saved

    def ask_wiki(self, query):
        """RAG-lite for querying wiki content."""
        inventory = self._get_wiki_inventory()
        inventory_str = "\n".join([f"- {i['name']} ({i['type']})" for i in inventory])
        
        routing_prompt = f"""You are a Librarian. Select up to 5 relevant files for: {query}\nINVENTORY:\n{inventory_str}"""

        try:
            llm.ensure_available(task_name="wiki_routing", prompt_summary=f"Lexis Ask query: {query}")
            route = llm.chat([{"role": "user", "content": routing_prompt}], schema=RoutingResult, task_name="wiki_routing")
            selected_names = route.selected_files if route else []
            
            # Helper to normalize names for comparison (ignores spaces, underscores, and extension)
            def normalize_name(n):
                if n.lower().endswith('.md'):
                    n = n[:-3]
                return re.sub(r'[^a-zA-Z0-9]', '', n).lower()

            context_parts = []
            for name in selected_names:
                norm_name = normalize_name(name)
                item = next((i for i in inventory if normalize_name(i['name']) == norm_name), None)
                if item:
                    with open(item['path'], "r", encoding="utf-8") as f:
                        context_parts.append(f"--- FILE: {item['name']} ---\n{f.read()}")

            # Robust Fallback: If LLM selected nothing, or all selected names failed to match any inventory file,
            # use a smart local keyword matcher
            if not context_parts:
                # Pre-process query to separate digits (e.g. "4unit" -> "4 unit", "book4" -> "book 4")
                clean_query = re.sub(r'([0-9]+)', r' \1 ', query).lower()
                tokens = []
                # Keep words length >= 2, or any length if they are unit digits
                for word in re.findall(r'[a-zA-Z0-9]+', clean_query):
                    if len(word) >= 2 or word.isdigit():
                        tokens.append(word)
                # Keep Chinese characters
                for char in re.findall(r'[\u4e00-\u9fff]', query):
                    tokens.append(char)

                scored_items = []
                for item in inventory:
                    name_lower = item['name'].lower()
                    score = 0
                    
                    # Category indicators boost matched types
                    if "vocabulary" in tokens and item['type'] == "wiki_vocabulary":
                        score += 5
                    if "grammar" in tokens and item['type'] == "wiki_grammar":
                        score += 5
                    if "summary" in tokens and item['type'] == "wiki_summaries":
                        score += 5
                    
                    for t in tokens:
                        if t in name_lower:
                            score += 10
                            if t.isdigit():
                                score += 10  # heavy priority match on unit/book numbers
                    
                    if score > 15:
                        scored_items.append((item['name'], score))
                
                if scored_items:
                    scored_items.sort(key=lambda x: x[1], reverse=True)
                    # Take top 3 matching items to avoid bloating prompt context
                    top_matches = [name for name, _ in scored_items[:3]]
                    for name in top_matches:
                        item = next((i for i in inventory if i['name'] == name), None)
                        if item:
                            with open(item['path'], "r", encoding="utf-8") as f:
                                context_parts.append(f"--- FILE: {item['name']} ---\n{f.read()}")

            # Smart fallback for source/original text queries not yet resolved
            if not context_parts:
                query_lower = query.lower()
                is_source_query = any(
                    kw in query_lower
                    for kw in ["original text", "full text", "source text", "complete text",
                                "原文", "全文", "完整内容", "课文", "文章"]
                )

                if is_source_query:
                    # Extract unit identifier from query (e.g. "Book 4 Unit 5" -> "book_4_unit_5")
                    source_tokens = []
                    for word in re.findall(r'[a-zA-Z0-9]+', clean_query):
                        if len(word) >= 2 or word.isdigit():
                            source_tokens.append(word.lower())

                    # Collect names already used as context to avoid duplicates
                    existing_names = {re.sub(r'[^a-z0-9]', '', n).lower() for n in inventory}

                    seen_units = set()
                    for item in inventory:
                        if item['type'] != 'raw_source':
                            continue
                        # Check path and name for unit identifier tokens
                        item_path = str(item['path']).replace('\\', '/')
                        item_name = (item['name'] + " " + item_path).lower()
                        matched_tokens = [t for t in source_tokens if t in item_name]
                        if not matched_tokens:
                            continue

                        # Derive unit folder name from path (wiki/<unit>/sources/...)
                        wiki_idx = item_path.find('/wiki/')
                        if wiki_idx >= 0:
                            after_wiki = item_path[wiki_idx + 6:]
                            parts = after_wiki.split('/')
                            if len(parts) >= 3 and parts[1] == 'sources':
                                unit_folder = parts[0]
                            else:
                                continue
                        else:
                            # Bare filename like "Book_4_Unit_5.md" -> use stem as unit name
                            unit_folder = Path(item['name']).stem

                        if unit_folder in seen_units:
                            continue
                        norm_key = re.sub(r'[^a-z0-9]', '', unit_folder).lower()
                        if norm_key in existing_names:
                            continue
                        seen_units.add(unit_folder)
                        existing_names.add(norm_key)

                        with open(item['path'], "r", encoding="utf-8") as f:
                            context_parts.append(
                                f"--- SOURCE FILE ({unit_folder}): {item['name']} ---\n{f.read()}"
                            )

            full_context = "\n\n".join(context_parts)
            answer_prompt = f"""Answer based on context:\n{full_context}\n\nQUERY: {query}"""
            return llm.chat([{"role": "user", "content": answer_prompt}], json_format=False)

        except Exception as e:
            return f"Wiki Query Error: {e}"
