"""
Syllabus wordlist parsing, Markdown formatting, mindmap rendering, and disk I/O.
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

class SyllabusAndMarkdownMixin:
    def __init__(self):
        self.config = config

    @staticmethod
    def parse_raw_headwords(raw_input: Any) -> List[str]:
        """
        Smart parser for messy textbook vocabulary lists.
        Handles dirty pastes with numbers, IPA, PoS abbreviations, Chinese translations,
        bullet points, colons, commas, semicolons, and newlines.
        """
        if not raw_input:
            return []

        if isinstance(raw_input, list):
            lines = [str(x) for x in raw_input]
        else:
            lines = str(raw_input).strip().split('\n')

        headwords = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Remove leading Markdown bullets, numbers, colons
            line_clean = re.sub(r'^(?:[\d\.\-\*\•\–\—\)\s]+)', '', line_str).strip()
            if not line_clean:
                continue

            # Strip ', etc.' or 'etc.' before comma-splitting so items like
            # 'enlist sb.'s help / services, etc.' or 'devote one's time / money, etc. to sth.' do not fracture.
            line_clean = re.sub(r',?\s*\betc\.?(?=\s|$|[,;])', '', line_clean, flags=re.IGNORECASE)

            # If separated by commas or semicolons:
            # Do not split on internal commas inside multi-clause proverbs/expressions
            # (e.g. "where there's a will, there's a way") where every comma-separated clause has >= 3 words.
            semi_chunks = re.split(r';+', line_clean)
            chunks = []
            for sc in semi_chunks:
                comma_parts = [p.strip() for p in sc.split(',') if p.strip()]
                if len(comma_parts) == 2 and all(len(p.split()) >= 3 for p in comma_parts):
                    chunks.append(sc.strip())
                else:
                    chunks.extend(comma_parts)
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue

                # Normalize curly apostrophes, spaced possessive apostrophes (' s -> 's), and ellipsis before splitting/cleaning
                chunk = chunk.replace('\u2019', "'").replace('\u2018', "'").replace('…', '...')
                chunk = re.sub(r"'\s+s\b", "'s", chunk)

                # Normalize slash-separated word alternatives (e.g. 'get / feel / be lost' -> 'get/feel/be lost')
                # so middle segments ('/ feel /') are never mistaken for IPA transcriptions
                chunk = re.sub(r"([A-Za-z'’\-])\s*/\s*(?=[A-Za-z'’\-])", r"\1/", chunk)

                # Remove IPA pronunciations like /'bɜːɡləri/ or [ˈfæsɪneɪtɪŋ]
                chunk = re.sub(r'(?<!\S)\/[^\/]+\/(?!\S)|\[[^\]]+\]', ' ', chunk)

                # Split out Chinese translations or English definitions following colons, dashes, tabs, or spaces
                chunk = re.split(r'[\u4e00-\u9fa5]|(?<!\w)[—–\-:]+(?!\w)|\t', chunk)[0].strip()

                # Clean common PoS abbreviations
                chunk = re.sub(r'\b(?:n|v|vt|vi|adj|adv|prep|conj|pron|art|num|phr|idiom)\.?\b', ' ', chunk, flags=re.IGNORECASE)

                # Normalize sb.'s / sb. / sth. periods before stripping trailing punctuation
                chunk = re.sub(r"\bsb\.'s\b", "sb's", chunk, flags=re.IGNORECASE)
                chunk = re.sub(r'\b(sth|sb)\.', r'\1', chunk, flags=re.IGNORECASE)

                # Normalize whitespace
                chunk = re.sub(r'\s+', ' ', chunk).strip()

                # Filter out numbers and punctuation (preserving internal '...' for slot ellipsis)
                if chunk and re.search(r'[A-Za-z]', chunk):
                    cleaned_hw = re.sub(r'[^\w\s\-\'\[\]\(\)\/\.]', '', chunk).strip('. ').strip()
                    if cleaned_hw and len(cleaned_hw) >= 2:
                        headwords.append(cleaned_hw)

        # Deduplicate while preserving order
        seen = set()
        final_list = []
        for w in headwords:
            w_lower = w.lower()
            if w_lower not in seen:
                seen.add(w_lower)
                final_list.append(w)

        return final_list

    @classmethod
    def strip_non_prose_wordlists(cls, text: str) -> Tuple[str, List[str]]:
        """
        Detects and strips non-prose word lists, vocabulary appendices, and glossaries
        embedded in source text (>= 5 consecutive lines with <= 3 words and no finite verbs).
        """
        if not text:
            return "", []

        lines = text.split("\n")
        n = len(lines)
        nlp = None  # lazy load spaCy only if block matches line criteria

        extracted_words: List[str] = []
        is_list_line = [False] * n

        i = 0
        while i < n:
            line = lines[i].strip()
            # Skip empty lines or markdown horizontal rules
            if not line or line.startswith("---") or line.startswith("==="):
                i += 1
                continue

            # Candidate line check: strip bullets/numbers and check word count
            c_line = re.sub(r'^(?:[\d\.\-\*\•\–\—\)\s]+)', '', line).strip()
            # Strip definitions/translations and common PoS tags before checking punctuation
            c_core = re.split(r'[\u4e00-\u9fa5]|(?<!\w)[—–\-:]+(?!\w)|\t', c_line)[0].strip()
            c_core = re.sub(r'\b(?:n|v|vt|vi|adj|adv|prep|conj|pron|art|num|phr|idiom)\.?\b', ' ', c_core, flags=re.IGNORECASE).strip()
            words_in_line = c_core.split()

            # A non-prose line has <= 3 core words and doesn't end in full sentence punctuation (. ! ?)
            is_cand = 1 <= len(words_in_line) <= 3 and not re.search(r'[!?]\s*$', c_line)
            if is_cand and c_line.endswith("."):
                # Allow trailing period only if it's an abbreviation like n. / v. / etc.
                is_cand = bool(re.search(r'\b(?:n|v|vt|vi|adj|adv|prep|conj|pron|art|num|phr|idiom|etc|sb|sth)\.\s*$', c_line, re.IGNORECASE))

            if is_cand:
                # Potential start of a non-prose list block
                block_start = i
                block_end = i
                while block_end < n:
                    curr_line = lines[block_end].strip()
                    if not curr_line:
                        break
                    curr_clean = re.sub(r'^(?:[\d\.\-\*\•\–\—\)\s]+)', '', curr_line).strip()
                    curr_core = re.split(r'[\u4e00-\u9fa5]|(?<!\w)[—–\-:]+(?!\w)|\t', curr_clean)[0].strip()
                    curr_core = re.sub(r'\b(?:n|v|vt|vi|adj|adv|prep|conj|pron|art|num|phr|idiom)\.?\b', ' ', curr_core, flags=re.IGNORECASE).strip()
                    curr_words = curr_core.split()
                    curr_cand = 1 <= len(curr_words) <= 3 and not re.search(r'[!?]\s*$', curr_clean)
                    if curr_cand and curr_clean.endswith("."):
                        curr_cand = bool(re.search(r'\b(?:n|v|vt|vi|adj|adv|prep|conj|pron|art|num|phr|idiom|etc|sb|sth)\.\s*$', curr_clean, re.IGNORECASE))

                    if curr_cand:
                        block_end += 1
                    else:
                        break

                block_len = block_end - block_start
                if block_len >= 5:
                    # Check for finite verbs using spaCy to ensure it's not a short poem / dialogue
                    block_text = "\n".join(lines[block_start:block_end])
                    if nlp is None:
                        nlp = LinguisticEngine.get_spacy()
                    doc = nlp(block_text)
                    # Finite verbs have pos_ == "VERB" and tag_ in ("VBD", "VBZ", "VBP", "MD")
                    finite_verbs = [
                        token for token in doc
                        if token.pos_ in ("VERB", "AUX") and token.tag_ in ("VBD", "VBZ", "VBP", "MD")
                    ]
                    if not finite_verbs:
                        # Genuine non-prose word list!
                        for idx in range(block_start, block_end):
                            is_list_line[idx] = True
                        # Mine words
                        extracted_words.extend(cls.parse_raw_headwords(block_text))
                        # Also check if previous line was a list heading (e.g. ## Words to Learn)
                        if block_start > 0 and lines[block_start - 1].strip().startswith("#"):
                            is_list_line[block_start - 1] = True
                    i = block_end
                    continue

            i += 1

        clean_lines = [lines[idx] for idx in range(n) if not is_list_line[idx]]
        # Clean up any excessive blank lines left over
        clean_text = re.sub(r'\n{3,}', '\n\n', "\n".join(clean_lines)).strip()
        return clean_text, extracted_words

    @classmethod
    def expression_constituent_atoms(cls, expression: str) -> Set[str]:
        """Returns the single-word constituent atoms covered by a multi-word expression."""
        atoms: Set[str] = set()
        for st in re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)*", (expression or "").lower()):
            if "-" in st:
                atoms.add(st)
                continue
            atoms.add(st)
            if st == "granted":
                atoms.add("grant")
            elif st.endswith("ing"):
                atoms.add(st[:-3])
            elif st.endswith("ed"):
                atoms.add(st[:-2])
        return atoms

    @classmethod
    def parse_syllabus_sections(cls, content: str) -> Tuple[str, List[str], List[str], List[str]]:
        """
        Parses source Markdown for syllabus sections (clean_body, syllabus_vocab,
        syllabus_grammar, syllabus_expressions).
        """
        if not content:
            return "", [], [], []

        # Find any ## Syllabus Vocabulary, ## Syllabus Grammar, or ## Syllabus Expressions sections
        # Stops at subsequent Markdown headers (##) or full-prose paragraphs (\n\n followed by sentence capital letter)
        section_boundary = r'(?=\n##+|\n\n(?=[A-Z][a-z]+(?:\s+[a-z]+){3,}\b)|\Z)'
        vocab_matches = re.search(
            r'##+\s*(?:Syllabus\s+Vocabulary|Vocabulary\s+List|Words?\s+to\s+Learn|Target\s+Words|Word\s*List|Vocabulary|词汇表|生词表)[^\n]*\n([\s\S]*?)' + section_boundary,
            content,
            re.IGNORECASE
        )
        grammar_matches = re.search(
            r'##+\s*(?:Syllabus\s+Grammar|Grammar\s+Topics|Grammar\s+List|Target\s+Grammar|Grammar|语法点|语法表)[^\n]*\n([\s\S]*?)' + section_boundary,
            content,
            re.IGNORECASE
        )
        expr_matches = re.search(
            r'##+\s*(?:Syllabus\s+Expressions|Syllabus\s+Phrases|Expressions\s+List|Phrases\s+List|Phrases\s+to\s+Learn|短语表|词组表)[^\n]*\n([\s\S]*?)' + section_boundary,
            content,
            re.IGNORECASE
        )

        raw_vocab = []
        if vocab_matches:
            raw_vocab_text = vocab_matches.group(1).strip()
            raw_vocab = cls.parse_raw_headwords(raw_vocab_text)

        syllabus_grammar = []
        if grammar_matches:
            raw_grammar_text = grammar_matches.group(1).strip()
            # Split grammar items by line / bullets / numbers
            for line in raw_grammar_text.split('\n'):
                line = re.sub(r'^(?:[\d\.\-\*\•\–\—\)\s]+)', '', line).strip()
                if line and len(line) >= 2:
                    syllabus_grammar.append(line)

        # Multi-word indicator test: spaces, slashes, placeholders (sb/sth/one's), parentheses
        def is_multiword(item: str) -> bool:
            clean_item = item.strip()
            return (
                " " in clean_item
                or "/" in clean_item
                or "sb." in clean_item.lower()
                or "sb " in clean_item.lower()
                or "sth." in clean_item.lower()
                or "sth " in clean_item.lower()
                or "one's" in clean_item.lower()
                or "(" in clean_item
            )

        # Partition raw_vocab into pure single words vs detected expressions
        pure_words = []
        detected_exprs = []
        for item in raw_vocab:
            if is_multiword(item):
                detected_exprs.append(item)
            else:
                if item.strip().lower() not in {
                    "on", "in", "at", "to", "of", "by", "for", "up", "as", "or", "so", "be", "do", "go", "no", "if", "it", "he", "we", "me", "us", "my", "an"
                }:
                    pure_words.append(item)

        # 1. Parse explicit expressions if section present
        syllabus_expressions = []
        if expr_matches:
            raw_expr_text = expr_matches.group(1).strip()
            syllabus_expressions = cls.parse_raw_headwords(raw_expr_text)
            # Merge any extra detected expressions from raw_vocab without duplicates
            seen_exprs = {e.lower() for e in syllabus_expressions}
            for e in detected_exprs:
                if e.lower() not in seen_exprs:
                    syllabus_expressions.append(e)
                    seen_exprs.add(e.lower())
        else:
            syllabus_expressions = detected_exprs

        # Promote 'take it for granted' into syllabus_expressions when 'grant' appears in that idiom
        body_lower = content.lower()
        if "grant" in [re.sub(r'\[.*?\]|\(.*?\)', '', w).strip().lower() for w in pure_words]:
            if "take it for granted" in body_lower or "take for granted" in body_lower or "taken for granted" in body_lower:
                promoted_expr = "take it for granted"
                if promoted_expr not in [e.lower() for e in syllabus_expressions]:
                    syllabus_expressions.insert(0, promoted_expr)
                    logger.info(f"✨ Promoted 'take it for granted' into syllabus expressions from body text evidence.")

        # Atom suppression is deferred to process_passage so atoms are only suppressed if their expression ships
        syllabus_vocab = pure_words

        # Strip syllabus sections from body text to avoid confusing general prompt
        clean_body = re.sub(
            r'##+\s*(?:Syllabus\s+Vocabulary|Vocabulary\s+List|Words?\s+to\s+Learn|Target\s+Words|Word\s*List|Vocabulary|词汇表|生词表)[^\n]*\n[\s\S]*?' + section_boundary,
            '',
            content,
            flags=re.IGNORECASE
        )
        clean_body = re.sub(
            r'##+\s*(?:Syllabus\s+Grammar|Grammar\s+Topics|Grammar\s+List|Target\s+Grammar|Grammar|语法点|语法表)[^\n]*\n[\s\S]*?' + section_boundary,
            '',
            clean_body,
            flags=re.IGNORECASE
        )
        clean_body = re.sub(
            r'##+\s*(?:Syllabus\s+Expressions|Syllabus\s+Phrases|Expressions\s+List|Phrases\s+List|Phrases\s+to\s+Learn|短语表|词组表)[^\n]*\n[\s\S]*?' + section_boundary,
            '',
            clean_body,
            flags=re.IGNORECASE
        ).strip()

        # Strip YAML frontmatter (metadata) so prompt body receives strictly pristine pedagogical prose
        clean_body = re.sub(r'^---\s*\n.*?\n---\s*\n', '', clean_body, flags=re.DOTALL).strip()

        # Strip blog/article metadata lines or parentheticals (e.g. '(Posted on 29 December, 2022 by Nazia Anderson)')
        clean_body = re.sub(r'\s*\(\s*(?:posted|updated|published|written)\s+(?:on|by)\b[^\n\)]+\)', '', clean_body, flags=re.IGNORECASE)
        clean_body = re.sub(r'(?m)^\s*(?:posted|updated|published|written)\s+(?:on|by)\b[^\n]+$', '', clean_body, flags=re.IGNORECASE).strip()

        # Strip non-prose word lists or glossaries at the bottom of texts
        clean_body, auto_vocab = cls.strip_non_prose_wordlists(clean_body)
        if not syllabus_vocab and auto_vocab:
            # If no syllabus section was explicitly defined, populate syllabus_vocab from stripped lists
            syllabus_vocab = auto_vocab

        return clean_body, syllabus_vocab, syllabus_grammar, syllabus_expressions

    def _save_extraction_results(self, data, source_filename, category_override=None):
        """Categorizes and saves extraction results."""
        path_obj = Path(source_filename)
        if path_obj.stem.lower() in ["subtitle", "transcript"] and path_obj.parent.name:
            filename_stem = path_obj.parent.name
        else:
            filename_stem = path_obj.stem
        saved_paths = []
        
        # Use explicit override if provided (most robust)
        category = category_override
        
        # Fallback to dynamic category detection for Virtual Schemas
        if not category:
            category = getattr(data, "_category", None)
        
        # Fallback to class-based detection for legacy/internal schemas
        if not category:
            if isinstance(data, VocabularyExtraction): category = "vocabulary"
            elif isinstance(data, GrammarExtraction): category = "grammar"
            elif isinstance(data, SummaryExtraction): category = "summary"
            elif isinstance(data, MindMapExtraction): category = "mindmap"

        unit_dir = self.config.wiki_content_path / filename_stem
        extractions_dir = unit_dir / "extractions"
        extractions_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # Transparent Delivery: Extractions always ship to extractions/
        # QA status & review flag are recorded in frontmatter for teacher visibility
        # -------------------------------------------------------------
        qa_audit = getattr(data, "_qa_audit", None) if dataclasses.is_dataclass(data) else (data.get("_qa_audit") if isinstance(data, dict) else None)
        if qa_audit is None and category in ("vocabulary", "grammar", "summary"):
            try:
                from ..evaluator import LogEvaluator
                import json as _json
                payload_dict = dataclasses.asdict(data) if dataclasses.is_dataclass(data) else (dict(data) if isinstance(data, dict) else {})
                payload_dict.pop("_qa_audit", None)
                src_ctx = getattr(self, "_last_source_content", "") or ""
                if not src_ctx:
                    src_md = unit_dir / "sources" / f"{filename_stem}.md"
                    if src_md.exists():
                        src_ctx = src_md.read_text(encoding="utf-8", errors="ignore")
                qa_audit = LogEvaluator.evaluate_log({
                    "task": f"extract_{category}_{filename_stem}",
                    "parsed_json": payload_dict,
                    "response": _json.dumps(payload_dict, ensure_ascii=False),
                    "user_prompt": src_ctx,
                    "context_prompt": src_ctx,
                })
                if dataclasses.is_dataclass(data):
                    object.__setattr__(data, "_qa_audit", qa_audit)
                elif isinstance(data, dict):
                    data["_qa_audit"] = qa_audit
            except Exception as e:
                logger.debug(f"Deterministic QA audit evaluation skipped for {category}: {e}")
        has_fatal_flags = False
        if qa_audit and isinstance(qa_audit, dict):
            composite = qa_audit.get("composite_score")
            flags = qa_audit.get("flags", [])
            from ..evaluator import FATAL_QA_FLAGS
            has_fatal_flags = any(
                any(fatal in f for fatal in FATAL_QA_FLAGS)
                for f in flags
            )
            if (composite is not None and composite < 60.0) or has_fatal_flags:
                logger.warning(
                    f"🚫 {category} extraction for {filename_stem} scored {composite}/100 (fatal defect / failed). "
                    f"Delivering to extractions/ with qa_status: 'failed' (body blocked from rendering)."
                )
            elif (composite is not None and composite < 80.0):
                logger.warning(
                    f"⚠️ {category} extraction for {filename_stem} scored {composite}/100 (review recommended). "
                    f"Delivering directly to extractions/ with qa_status: 'review_needed'."
                )

        if category == "vocabulary":
            # Deduplicate vocabulary list based on 'word' field (case-insensitive & stripped)
            seen = set()
            deduped_vocabulary = []
            if dataclasses.is_dataclass(data):
                vocab_list = getattr(data, "vocabulary", [])
                for item in vocab_list:
                    word_val = getattr(item, "word", "")
                    if isinstance(word_val, str):
                        w_clean = word_val.strip().lower().replace("[[", "").replace("]]", "")
                        if w_clean and w_clean not in seen:
                            seen.add(w_clean)
                            deduped_vocabulary.append(item)
                data.vocabulary = deduped_vocabulary
            elif isinstance(data, dict):
                vocab_list = data.get("vocabulary", [])
                for item in vocab_list:
                    word_val = item.get("word", "") if isinstance(item, dict) else getattr(item, "word", "")
                    if isinstance(word_val, str):
                        w_clean = word_val.strip().lower().replace("[[", "").replace("]]", "")
                        if w_clean and w_clean not in seen:
                            seen.add(w_clean)
                            deduped_vocabulary.append(item)
                data["vocabulary"] = deduped_vocabulary

            path = extractions_dir / f"{filename_stem}_vocabulary.md"
            content = self._format_as_markdown(data, "vocabulary", source_filename)
            with open(path, "w", encoding="utf-8") as f: f.write(content)
            saved_paths.append(str(path))
        
        elif category == "grammar":
            path = extractions_dir / f"{filename_stem}_grammar.md"
            content = self._format_as_markdown(data, "grammar", source_filename)
            with open(path, "w", encoding="utf-8") as f: f.write(content)
            saved_paths.append(str(path))

        elif category == "summary":
            path = extractions_dir / f"{filename_stem}_summary.md"
            content = self._format_as_markdown(data, "summary", source_filename)
            with open(path, "w", encoding="utf-8") as f: f.write(content)
            saved_paths.append(str(path))

        elif category == "mindmap":
            # 1. Save raw JSON file
            json_path = extractions_dir / f"{filename_stem}_mindmap.json"
            if dataclasses.is_dataclass(data):
                mindmap_dict = dataclasses.asdict(data)
            else:
                mindmap_dict = data
            branches = mindmap_dict.get("branches", mindmap_dict.get("nodes", [])) if isinstance(mindmap_dict, dict) else []
            mindmap_dict["item_count"] = len(branches)
            # Physical text CEFR calculation
            if hasattr(self, "_last_source_content") and self._last_source_content:
                mindmap_dict["overall_cefr_level"] = LinguisticEngine.calculate_text_cefr(self._last_source_content)
            elif "overall_cefr_level" not in mindmap_dict:
                mindmap_dict["overall_cefr_level"] = "B1"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(mindmap_dict, f, indent=2, ensure_ascii=False)
            saved_paths.append(str(json_path))

            # 2. Render and save standalone HTML to extractions directory
            html_content = self._render_mindmap(mindmap_dict)
            html_path = extractions_dir / f"{filename_stem}_mindmap.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            saved_paths.append(str(html_path))

        return saved_paths

    def _format_as_markdown(self, data, category, source_filename=None):
        """
        Converts an extraction object to a markdown string dynamically.
        Iterates over dataclass fields or dict keys to remain field-agnostic.
        """
        source_link = f"[[{source_filename}]]" if source_filename else "None"
        display_title = Path(source_filename).stem.replace("_", " ") if source_filename else "Unit"
        
        # 1. Handle Collection Extractions (Vocabulary, Grammar)
        if category in ["vocabulary", "grammar"]:
            # Identify top-level fields (like overall_cefr_level)
            if dataclasses.is_dataclass(data):
                fields = [(f.name, getattr(data, f.name)) for f in dataclasses.fields(data)]
            else:
                fields = list(data.items())

            # Identify the collection list & count
            items = []
            if dataclasses.is_dataclass(data):
                list_field = next((f for f in dataclasses.fields(data) if isinstance(getattr(data, f.name), list)), None)
                if list_field:
                    items = getattr(data, list_field.name)
            else:
                list_field_name = next((k for k, v in fields if isinstance(v, list)), None)
                if list_field_name: items = data.get(list_field_name, [])

            # Automatic Deduplication: filter out repeated entries with identical primary key (e.g. word / headword)
            seen_keys = set()
            unique_items = []
            for item in items:
                k_val = None
                if dataclasses.is_dataclass(item):
                    for attr in ["word", "name", "concept_name", "quote"]:
                        if hasattr(item, attr) and getattr(item, attr):
                            k_val = str(getattr(item, attr)).strip().lower()
                            break
                elif isinstance(item, dict):
                    for attr in ["word", "name", "concept_name", "quote"]:
                        if attr in item and item[attr]:
                            k_val = str(item[attr]).strip().lower()
                            break
                
                if k_val:
                    if k_val in seen_keys:
                        continue
                    seen_keys.add(k_val)
                unique_items.append(item)
            
            items = unique_items
            item_count = len(items)

            # Physical text CEFR calculation for collection extractions
            calc_overall_cefr = None
            if hasattr(self, "_last_source_content") and self._last_source_content:
                calc_overall_cefr = LinguisticEngine.calculate_text_cefr(self._last_source_content)
            elif category in ["vocabulary", "grammar"]:
                calc_overall_cefr = "B1"

            lines = [
                "---",
                f"title: \"{display_title}\"",
                f"source: \"{source_link}\"",
                f"category: [\"{category}\", \"extraction\"]",
            ]
            if calc_overall_cefr:
                lines.append(f"overall_cefr_level: \"{calc_overall_cefr}\"")
            lines.append(f"item_count: {item_count}")

            # Inject QA Audit Score if available
            qa_audit = getattr(data, "_qa_audit", None) if dataclasses.is_dataclass(data) else (data.get("_qa_audit") if isinstance(data, dict) else None)
            is_failed = False
            failed_flags = []
            if qa_audit and isinstance(qa_audit, dict):
                qa_composite = qa_audit.get("composite_score")
                if qa_composite is not None:
                    lines.append(f"qa_score: {round(float(qa_composite))}")
                    from ..evaluator import FATAL_QA_FLAGS
                    flags = qa_audit.get("flags", [])
                    has_fatal = any(any(fatal in f for fatal in FATAL_QA_FLAGS) for f in flags)
                    if (qa_composite < 60.0) or has_fatal:
                        qa_status = "failed"
                        is_failed = True
                        failed_flags = [f for f in flags if any(fatal in f for fatal in FATAL_QA_FLAGS) or f.startswith("❌")]
                    elif qa_composite >= 80.0:
                        qa_status = "passed"
                    else:
                        qa_status = "review_needed"
                    lines.append(f"qa_status: \"{qa_status}\"")

            for name, val in fields:
                if name not in ["title", "grammar", "grammar_patterns", "vocabulary", "concepts", "_category", "_qa_audit"]:
                    if val: lines.append(f"{name}: \"{val}\"")
            
            lines.extend(["---", "", f"# {category.title()}: {display_title}", ""])

            # If extraction failed QA gate (<60% or fatal defects), block contaminated body items from rendering
            if is_failed:
                flag_summary = f" (Reasons: {'; '.join(failed_flags[:2])})" if failed_flags else ""
                lines.extend([
                    f"> [!CAUTION]",
                    f"> **Extraction Quality Gate Failed (Score: {round(float(qa_composite or 0))}/100)**{flag_summary}",
                    f"> The generated {category} content contained severe defects and was blocked to prevent curricular contamination.",
                    f"> Please check the source text completeness, adjust model settings, or recompile."
                ])
                return "\n".join(lines) + "\n"

            for item in items:
                if dataclasses.is_dataclass(item):
                    item_fields = [(f.name, getattr(item, f.name)) for f in dataclasses.fields(item) if f.name != "design_audit"]
                else:
                    item_fields = [(k, v) for k, v in item.items() if k != "design_audit"]

                if not item_fields:
                    continue

                # For grammar items, prefer syntax_topic as header, fallback to category
                if category == "grammar":
                    topic_val = getattr(item, "syntax_topic", None) if dataclasses.is_dataclass(item) else (item.get("syntax_topic") if isinstance(item, dict) else None)
                    cat_val = getattr(item, "category", None) if dataclasses.is_dataclass(item) else (item.get("category") if isinstance(item, dict) else None)
                    if not cat_val:
                        cat_val = "Information Packaging"
                    header_val = topic_val if topic_val else cat_val
                    header_entry = ("syntax_topic" if topic_val else "category", header_val)
                    
                    if dataclasses.is_dataclass(item):
                        item_fields = [(f.name, getattr(item, f.name)) for f in dataclasses.fields(item) if f.name not in ("design_audit", "syntax_topic")]
                    else:
                        item_fields = [(k, v) for k, v in item.items() if k not in ("design_audit", "syntax_topic")]
                    
                    # Ensure Category is visible in item fields
                    body_entries = []
                    if topic_val and cat_val:
                        body_entries.append(("Category", cat_val))
                    body_entries.extend([entry for entry in item_fields if entry[0].lower() != "category"])
                    primary_key = header_entry[0]
                else:
                    # Locate canonical primary header field
                    primary_key = None
                    for hk in ["word", "category", "name", "concept_name", "title"]:
                        if any(k == hk for k, v in item_fields):
                            primary_key = hk
                            break

                    if primary_key:
                        header_entry = next((k, v) for k, v in item_fields if k == primary_key)
                        body_entries = [entry for entry in item_fields if entry[0] != primary_key]
                    else:
                        header_entry = item_fields[0]
                        body_entries = item_fields[1:]

                header_name, header_val = header_entry
                val_str = str(header_val or "").strip()
                if val_str.startswith("[[") and val_str.endswith("]]"):
                    inner = val_str[2:-2]
                else:
                    inner = val_str
                # Normalize inner brackets to parentheses so wikilinks [[...]] stay valid in Obsidian
                inner = inner.replace("[", "(").replace("]", ")")
                header_val = f"[[{inner}]]"

                # For vocabulary items, deterministically calculate Word CEFR Level, Part of Speech, and Enrichment
                header_suffix_parts: List[str] = []
                if category == "vocabulary":
                    has_cefr = any(k.lower() == "word_cefr_level" and v for k, v in body_entries)
                    derived_cefr = ""
                    if not has_cefr and primary_key == "word":
                        clean_target_word = re.sub(r'\[.*?\]|\(.*?\)', '', str(header_entry[1])).strip().lower()
                        # If multi-word expression, take first core content word (skipping determiners, prepositions, and articles)
                        STOP_WORDS_FOR_CEFR = {
                            'a', 'an', 'the', 'any', 'some', 'this', 'that', 'these', 'those', 'all',
                            'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'toward', 'towards',
                            'into', 'through', 'over', 'under', 'from', 'as'
                        }
                        tokens = [tok for tok in clean_target_word.split() if tok]
                        content_tokens = [tok for tok in tokens if tok not in STOP_WORDS_FOR_CEFR]
                        lookup_w = content_tokens[0] if content_tokens else (tokens[0] if tokens else clean_target_word)
                        derived_cefr = LinguisticEngine.get_word_cefr(lookup_w, default="B1")
                    else:
                        derived_cefr = str(next((v for k, v in body_entries if k.lower() == "word_cefr_level" and v), "") or "").strip()

                    authoritative_pos = ""
                    gram_tag_inline = ""
                    if primary_key == "word":
                        raw_quote = next((v for k, v in body_entries if k.lower() in ("quoted_sentence", "quote")), "")
                        raw_defn = next((v for k, v in body_entries if k.lower() == "definition"), "")
                        headword = str(header_entry[1])
                        current_pos_val = next((v for k, v in body_entries if k.lower() == "part_of_speech"), None)
                        authoritative_pos = LinguisticEngine.resolve_item_pos(headword, str(raw_quote), manual_pos=current_pos_val)

                        # Zero-Token Pedagogical Enrichment (<GRAM>, Collocations, Pattern Note, Thesaurus)
                        existing_keys = {k.lower() for k, _ in body_entries}
                        enrichment = LinguisticEngine.get_vocab_card_enrichment(
                            word=headword,
                            pos=authoritative_pos,
                            definition=str(raw_defn or ""),
                            quote=str(raw_quote or "")
                        )
                        gram_tag_inline = str(enrichment.get("grammar_tag") or "").strip()
                        if enrichment.get("collocations") and "collocations" not in existing_keys:
                            body_entries.append(("collocations", enrichment["collocations"]))
                        if enrichment.get("pattern_note") and "pattern" not in existing_keys and "grammar_usage" not in existing_keys:
                            body_entries.append(("pattern", enrichment["pattern_note"]))
                        if enrichment.get("thesaurus_note") and "thesaurus_note" not in existing_keys:
                            body_entries.append(("thesaurus_note", enrichment["thesaurus_note"]))

                    # Remove part_of_speech and word_cefr_level from bullet entries (moved to header line)
                    body_entries = [
                        (k, v) for k, v in body_entries
                        if k.lower() not in ("part_of_speech", "word_cefr_level")
                    ]

                    if authoritative_pos:
                        header_suffix_parts.append(authoritative_pos)
                    if derived_cefr:
                        header_suffix_parts.append(derived_cefr)
                    if gram_tag_inline:
                        fmt_gram = gram_tag_inline if gram_tag_inline.startswith("`") else f"`{gram_tag_inline}`"
                        header_suffix_parts.append(fmt_gram)

                header_line = f"## {header_val}"
                if header_suffix_parts:
                    header_line += " " + " ".join(header_suffix_parts)
                lines.append(header_line)

                # Iterate remaining fields as bullet points with canonical field ordering
                CANONICAL_FIELD_ORDERS = {
                    "grammar": [
                        "quote",
                        "pattern_formula",
                        "pedagogical_function",
                        "imitation_example",
                        "common_mistakes",
                        "cefr_level"
                    ],
                    "vocabulary": [
                        "definition",
                        "quoted_sentence",
                        "example_usage",
                        "collocations",
                        "pattern",
                        "grammar_usage",
                        "thesaurus_note"
                    ],
                    "expressions": [
                        "part_of_speech",
                        "word_cefr_level",
                        "definition",
                        "quoted_sentence",
                        "example_usage"
                    ]
                }

                preferred_order = CANONICAL_FIELD_ORDERS.get(category, [])
                def _field_sort_key(entry):
                    fname = entry[0].lower()
                    if fname in preferred_order:
                        return (0, preferred_order.index(fname))
                    return (1, fname)

                sorted_body_entries = sorted(body_entries, key=_field_sort_key)

                LABEL_OVERRIDES = {
                    "part_of_speech": "Part of Speech",
                    "word_cefr_level": "Word CEFR Level",
                    "cefr_level": "CEFR Level",
                    "collocations": "Collocations",
                    "pattern": "Pattern",
                    "grammar_usage": "Grammar & Usage",
                    "thesaurus_note": "Thesaurus Note",
                }

                # Pre-fetch sentence pool for deterministic quote snapping if source content is available
                sent_pool = None
                if hasattr(self, "_last_source_content") and self._last_source_content:
                    _, sent_pool = LinguisticEngine.tokenize_and_index_sentences(self._last_source_content)

                for fname, fval in sorted_body_entries:
                    label = LABEL_OVERRIDES.get(fname.lower(), fname.replace("_", " ").title())
                    if fname.lower() == "definition" and fval and \
                            LinguisticEngine.is_boilerplate_definition(str(fval)):
                        logger.warning(
                            f"🚫 Invented definition blocked for "
                            f"{header_val}: {str(fval).strip()[:70]}"
                        )
                        continue
                    if fval:
                        if category == "grammar" and fname == "pattern_formula":
                            fval = self.normalize_grammar_formula(str(fval))
                        elif fname.lower() in ("quote", "quoted_sentence"):
                            fval_str = str(fval).strip()
                            # Deterministically snap to authentic sentence in sentence pool if available
                            if sent_pool:
                                snapped = LinguisticEngine.snap_to_sentence_pool(fval_str, sent_pool)
                                if snapped:
                                    fval_str = snapped
                            # Strip internal sentence pool tokens like [S-26] or S-26: from front of quote if still present
                            fval_str = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", fval_str, flags=re.IGNORECASE).strip()
                            fval = fval_str
                        elif category == "grammar" and fname.lower() == "common_mistakes":
                            fval_str = str(fval).strip()
                            # Convert to Simplified Plan B layout if not already multiline
                            if "\n" not in fval_str:
                                # Strip prefix like 'PedLex Grammar Alert (TAG): ', 'COBUILD Warning (TAG): ', or 'Grammar Warning (TAG): '
                                m_tag = re.match(r'^(?:(?:PedLex|LDOCE)\s+Grammar\s+Alert|COBUILD\s+Warning|Grammar\s+Warning)\s*\(([^)]+)\)\s*:\s*(.*)', fval_str, re.DOTALL | re.IGNORECASE)
                                if m_tag:
                                    tag = m_tag.group(1).strip()
                                    body = m_tag.group(2).strip()
                                else:
                                    m_tag2 = re.match(r'^\(([^)]+)\)\s*(.*)', fval_str, re.DOTALL)
                                    if m_tag2:
                                        tag = m_tag2.group(1).strip()
                                        body = m_tag2.group(2).strip()
                                    else:
                                        tag = ""
                                        body = fval_str

                                tag_str = f"({tag}) " if tag else ""

                                # Pattern A: contains [INCORRECT: ... -> CORRECT: ...]
                                m_inc = re.search(r'^(.*?)\s*["“](.*?)\[INCORRECT:\s*(.*?)\s*->\s*CORRECT:\s*(.*?)\](.*?)["”](.*)$', body, re.DOTALL)
                                if m_inc:
                                    intro = m_inc.group(1).strip().rstrip(':').strip()
                                    pre = m_inc.group(2)
                                    wrong = m_inc.group(3).strip()
                                    right = m_inc.group(4).strip()
                                    post = m_inc.group(5)
                                    note = m_inc.group(6).strip()
                                    
                                    wrong_sent = f"{pre}*{wrong}*{post}".strip()
                                    right_sent = f"{pre}{right}{post}".strip()

                                    sub_lines = [f"{tag_str}{intro}:"]
                                    sub_lines.append(f"    ✗ {wrong_sent}")
                                    if right:
                                        sub_lines.append(f"    ✓ {right_sent}")
                                    if note:
                                        sub_lines.append(f"    {note}")
                                    fval = "\n".join(sub_lines)
                                # Pattern B: contains Don't say or ✗
                                elif "Don't say" in body or "✗" in body:
                                    parts = re.split(r"(?:✗\s*Don't say:\s*|✗Don't say:\s*|Don't say:\s*✗\s*|Don't say:\s*|✗\s*)", body)
                                    rule = parts[0].strip().rstrip(':').strip()
                                    non_empty = [p.strip() for p in parts[1:] if p.strip()]
                                    err = non_empty[-1] if non_empty else ""
                                    intro_line = f"{tag_str}{rule}".strip() if (tag_str or rule) else ""
                                    sub_lines = [intro_line] if intro_line else []
                                    if err:
                                        sub_lines.append(f"    ✗ {err}")
                                    fval = "\n".join(sub_lines)
                                else:
                                    fval = f"{tag_str}{body}"

                        lines.append(f"- **{label}**: {fval}")
                lines.append("")

            return "\n".join(lines)

        # 2. Handle Summary Extractions (with nested Concepts)
        elif category == "summary":
            if dataclasses.is_dataclass(data):
                title = getattr(data, "title", display_title)
                summary_text = getattr(data, "text_summary_or_plot", "")
                reading_time = getattr(data, "estimated_reading_time", "")
                questions = getattr(data, "essential_questions", [])
                lesson_hook = getattr(data, "lesson_hook", "")
                concepts = getattr(data, "concepts", [])
                overall_cefr = getattr(data, "overall_cefr_level", "B2")
            else:
                title = data.get("title", display_title)
                summary_text = data.get("text_summary_or_plot", "")
                reading_time = data.get("estimated_reading_time", "")
                questions = data.get("essential_questions", [])
                lesson_hook = data.get("lesson_hook", "")
                concepts = data.get("concepts", [])
                overall_cefr = data.get("overall_cefr_level", "B2")

            # Deterministic word count & reading time calculation (approx. 180-200 WPM)
            if hasattr(self, "_last_source_content") and self._last_source_content:
                words = len(re.findall(r'\b\w+\b', self._last_source_content))
                minutes = max(1, round(words / 180))
                reading_time = f"{words} words, approx. {minutes} min{'s' if minutes > 1 else ''} reading time"

            # Physical text CEFR calculation
            if hasattr(self, "_last_source_content") and self._last_source_content:
                overall_cefr = LinguisticEngine.calculate_text_cefr(self._last_source_content)
            elif not overall_cefr:
                overall_cefr = "B1"

            item_count = len(concepts)

            lines = [
                "---",
                f"title: \"{title}\"",
                f"source: \"{source_link}\"",
                "category: [\"summary\", \"extraction\"]",
                f"overall_cefr_level: \"{overall_cefr}\"",
                f"item_count: {item_count}",
                f"estimated_reading_time: \"{reading_time}\"",
            ]

            # Inject QA Audit Score if available
            qa_audit = getattr(data, "_qa_audit", None) if dataclasses.is_dataclass(data) else (data.get("_qa_audit") if isinstance(data, dict) else None)
            is_failed = False
            failed_flags = []
            if qa_audit and isinstance(qa_audit, dict):
                qa_composite = qa_audit.get("composite_score")
                if qa_composite is not None:
                    lines.append(f"qa_score: {round(float(qa_composite))}")
                    from ..evaluator import FATAL_QA_FLAGS
                    flags = qa_audit.get("flags", [])
                    has_fatal = any(any(fatal in f for fatal in FATAL_QA_FLAGS) for f in flags)
                    if (qa_composite < 60.0) or has_fatal:
                        qa_status = "failed"
                        is_failed = True
                        failed_flags = [f for f in flags if any(fatal in f for fatal in FATAL_QA_FLAGS) or f.startswith("❌")]
                    elif qa_composite >= 80.0:
                        qa_status = "passed"
                    else:
                        qa_status = "review_needed"
                    lines.append(f"qa_status: \"{qa_status}\"")

            lines.extend([
                "---",
                "",
                f"# Summary: {title}",
                "",
            ])

            # If extraction failed QA gate (<60% or fatal defects), block contaminated body from rendering
            if is_failed:
                flag_summary = f" (Reasons: {'; '.join(failed_flags[:2])})" if failed_flags else ""
                lines.extend([
                    f"> [!CAUTION]",
                    f"> **Extraction Quality Gate Failed (Score: {round(float(qa_composite or 0))}/100)**{flag_summary}",
                    f"> The generated summary content contained severe defects and was blocked to prevent curricular contamination.",
                    f"> Please check the source text completeness, adjust model settings, or recompile."
                ])
                return "\n".join(lines) + "\n"

            lines.extend([
                "## Narrative Overview",
                summary_text,
                "",
                "## Lesson Hook",
                lesson_hook,
                "",
                "## Essential Questions",
            ])
            for q in questions:
                lines.append(f"- {q}")
            lines.append("")
            
            lines.append("## Core Concepts")
            lines.append("")
            for concept in concepts:
                if dataclasses.is_dataclass(concept):
                    c_name = getattr(concept, "concept_name", "")
                    c_sig = getattr(concept, "educational_significance", "")
                    c_details = getattr(concept, "key_details", [])
                    c_conn = getattr(concept, "related_connections", [])
                else:
                    c_name = concept.get("concept_name", "")
                    c_sig = concept.get("educational_significance", "")
                    c_details = concept.get("key_details", [])
                    c_conn = concept.get("related_connections", [])

                # Add double-brackets around concept name for wikilinks
                c_header = str(c_name or "").strip()
                if c_header.startswith("[[") and c_header.endswith("]]"):
                    c_inner = c_header[2:-2]
                else:
                    c_inner = c_header
                c_inner = c_inner.replace("[", "(").replace("]", ")")
                c_header = f"[[{c_inner}]]"

                lines.append(f"### {c_header}")
                if c_sig:
                    lines.append(f"- **Educational Significance**: {c_sig}")
                if c_details:
                    lines.append("- **Key Details**:")
                    for detail in c_details:
                        lines.append(f"  - {detail}")
                
                # Render hierarchical sub-concepts if present
                sub_concepts = getattr(concept, "sub_concepts", []) if dataclasses.is_dataclass(concept) else concept.get("sub_concepts", [])
                if sub_concepts:
                    lines.append("- **Sub-concepts & Facets**:")
                    for sub in sub_concepts:
                        if dataclasses.is_dataclass(sub):
                            s_name = getattr(sub, "sub_concept_name", "")
                            s_sig = getattr(sub, "significance_or_takeaway", "")
                            s_points = getattr(sub, "key_points", [])
                        else:
                            s_name = sub.get("sub_concept_name", "")
                            s_sig = sub.get("significance_or_takeaway", "")
                            s_points = sub.get("key_points", [])
                        s_name_clean = str(s_name or "").strip().replace("[", "(").replace("]", ")")
                        if s_name_clean:
                            prefix = f"  - **[[{s_name_clean}]]**"
                            if s_sig:
                                prefix += f": {s_sig}"
                            lines.append(prefix)
                            for pt in s_points:
                                if pt:
                                    lines.append(f"    - *Key Point*: {pt}")

                if c_conn:
                    conn_links = []
                    for conn in c_conn:
                        clean_conn = str(conn or "").strip()
                        if "Connect to" in clean_conn or ":" in clean_conn:
                            m = re.search(r'(?:Connect to\s*)?\*?\*?([A-Za-z0-9\s/&#\-\[\]\(\)]+?)\*?\*?(?:\s*:|\s*$)', clean_conn, re.IGNORECASE)
                            if m and len(m.group(1).strip()) > 1:
                                clean_conn = m.group(1).strip()
                            elif ":" in clean_conn:
                                clean_conn = re.sub(r'^(?:Connect to\s*)?\*?\*?|\*?\*?$', '', clean_conn.split(":")[0]).strip()
                        if clean_conn:
                            if clean_conn.startswith("[[") and clean_conn.endswith("]]"):
                                clean_inner = clean_conn[2:-2]
                            else:
                                clean_inner = clean_conn
                            clean_inner = clean_inner.replace("[", "(").replace("]", ")")
                            conn_links.append(f"[[{clean_inner}]]")
                    if conn_links:
                        lines.append(f"- **Related Connections**: {', '.join(conn_links)}")
                lines.append("")
                lines.append("&nbsp;")
                lines.append("")

            return "\n".join(lines)

        return str(data)

    def _load_wiki_data(self, core_name, quiz_type):
        """Loads data from the wiki for quiz generation."""
        parts = Path(core_name).parts
        if "wiki" in parts:
            idx = parts.index("wiki")
            if idx + 1 < len(parts):
                core_name = parts[idx + 1]
        elif "raw" in parts:
            idx = parts.index("raw")
            if idx + 1 < len(parts):
                core_name = parts[idx + 1]
        else:
            core_name = parts[0] if parts else core_name
            
        if core_name.endswith(".md"): core_name = core_name[:-3]
        elif core_name.endswith(".txt"): core_name = core_name[:-4]

        # 1. Resolve unit directory case-insensitively using normalize_name
        from ..config import normalize_name
        norm_core = normalize_name(core_name)
        unit_dir = self.config.wiki_content_path / core_name
        if not unit_dir.exists() and self.config.wiki_content_path.exists():
            for child in self.config.wiki_content_path.iterdir():
                if child.is_dir() and normalize_name(child.name) == norm_core:
                    unit_dir = child
                    core_name = child.name
                    break

        # Resolve source path: wiki/<core_name>/sources/<core_name>.md or .txt
        source_path = unit_dir / "sources" / f"{core_name}.md"
        if not source_path.exists():
            source_path = unit_dir / "sources" / f"{core_name}.txt"
        if not source_path.exists():
            sources_dir = unit_dir / "sources"
            if sources_dir.exists() and sources_dir.is_dir():
                candidates = [f for f in sources_dir.iterdir() if f.is_file() and f.suffix in [".md", ".txt"]]
                if candidates:
                    source_path = candidates[0]

        cefr = "B2"
        if source_path and source_path.exists():
            try:
                raw_src_for_cefr = source_path.read_text(encoding="utf-8")
                clean_p_cefr, _, _, _ = self.parse_syllabus_sections(raw_src_for_cefr)
                det_c = LinguisticEngine.calculate_text_cefr(clean_p_cefr or raw_src_for_cefr)
                if det_c:
                    cefr = det_c
            except Exception:
                pass

        def _extract_fresh_vocab_items(src_path: Path) -> Tuple[List[VocabularyItem], str, str]:
            """Runs fresh deterministic vocabulary + expression extraction with F12 atom suppression/recovery, returning structured items."""
            with open(src_path, "r", encoding="utf-8") as f:
                src_text = f.read()
            clean_p, syl_v, syl_g, syl_e = self.parse_syllabus_sections(src_text)
            self._last_source_content = clean_p or src_text
            det_c = LinguisticEngine.calculate_text_cefr(clean_p) if clean_p else cefr
            resolved_cefr = det_c or cefr

            _, sent_pool = LinguisticEngine.tokenize_and_index_sentences(clean_p)
            f12_suppressed: List[Tuple[str, str]] = []
            filtered_syl_v = list(syl_v) if syl_v else []
            if syl_v and syl_e:
                f12_parent: Dict[str, str] = {}
                for expr in syl_e:
                    for atom in self.expression_constituent_atoms(expr):
                        f12_parent.setdefault(atom, expr)
                pending_v: List[str] = []
                for w in syl_v:
                    w_clean = re.sub(r'\[.*?\]|\(.*?\)', '', w).strip().lower()
                    parent_expr = f12_parent.get(w_clean)
                    if parent_expr:
                        f12_suppressed.append((w, parent_expr))
                        continue
                    pending_v.append(w)
                filtered_syl_v = pending_v

            compile_defaults = self.config.get("compile_defaults") or {}
            v_default_k = compile_defaults.get("vocabulary", 20)
            e_default_k = compile_defaults.get("expressions", 5)

            with LinguisticEngine.wsd_log_session(core_name):
                target_k = v_default_k
                det_v = LinguisticEngine.extract_deterministic_vocabulary(
                    clean_p,
                    syllabus_vocab=filtered_syl_v if filtered_syl_v else None,
                    target_count=target_k
                )
                v_items = [
                    VocabularyItem(
                        word=it['word'],
                        part_of_speech=it.get('part_of_speech', 'noun'),
                        definition=it.get('definition', ''),
                        quoted_sentence=it.get('quoted_sentence', ''),
                        example_usage=it.get('example_usage', ''),
                        design_audit=it.get('design_audit', '')
                    )
                    for it in (det_v or [])
                ]

                target_e_k = e_default_k
                det_e = LinguisticEngine.extract_deterministic_expressions(
                    clean_p,
                    target_count=target_e_k,
                    syllabus_expressions=syl_e
                )
                shipped_atoms: Set[str] = set()
                e_items = []
                for it in (det_e or []):
                    if not str(it.get("definition", "")).strip() or not str(it.get("example_usage", "")).strip():
                        continue
                    shipped_atoms |= self.expression_constituent_atoms(it.get("word", ""))
                    e_items.append(
                        VocabularyItem(
                            word=it['word'],
                            part_of_speech=it.get('part_of_speech', 'collocation'),
                            definition=it.get('definition', ''),
                            quoted_sentence=it.get('quoted_sentence', ''),
                            example_usage=it.get('example_usage', ''),
                            design_audit=it.get('design_audit', '')
                        )
                    )

                if f12_suppressed and len(v_items) < target_k:
                    rescued_raw = [
                        w for (w, _parent) in f12_suppressed
                        if re.sub(r'\[.*?\]|\(.*?\)', '', w).strip().lower() not in shipped_atoms
                    ]
                    rescued = [
                        w for w in rescued_raw
                        if not sent_pool or any(
                            LinguisticEngine.text_contains_form(s, re.sub(r'\[.*?\]|\(.*?\)', '', w).strip().lower(), standalone=True)
                            for s in sent_pool.values()
                        )
                    ]
                    if rescued:
                        existing_v_words = {v.word.strip().lower() for v in v_items}
                        rescued = [w for w in rescued if re.sub(r'\[.*?\]|\(.*?\)', '', w).strip().lower() not in existing_v_words]
                    if rescued:
                        remaining_slots = max(0, target_k - len(v_items))
                        for it in LinguisticEngine.extract_deterministic_vocabulary(
                            clean_p,
                            syllabus_vocab=rescued,
                            target_count=min(len(rescued), remaining_slots)
                        ):
                            if len(v_items) >= target_k:
                                break
                            if it['word'].strip().lower() not in existing_v_words:
                                existing_v_words.add(it['word'].strip().lower())
                                v_items.append(
                                    VocabularyItem(
                                        word=it['word'],
                                        part_of_speech=it.get('part_of_speech', 'noun'),
                                        definition=it.get('definition', ''),
                                        quoted_sentence=it.get('quoted_sentence', ''),
                                        example_usage=it.get('example_usage', ''),
                                        design_audit=it.get('design_audit', '')
                                    )
                                )

            all_v_items = v_items + e_items
            return all_v_items, resolved_cefr, clean_p

        if quiz_type in ("vocabulary", "vocabulary_quiz"):
            # Per Rule 3.1: Assessment truth must be grounded directly on authentic source text via structured items,
            # never downstream markdown extraction files.
            if not source_path or not source_path.exists():
                return None
            fresh_v_items, cefr, _ = _extract_fresh_vocab_items(source_path)
            if not fresh_v_items:
                return None
            return {"content": fresh_v_items, "cefr_level": cefr}

        elif quiz_type == "reading":
            if not source_path or not source_path.exists(): return None
            with open(source_path, "r", encoding="utf-8") as f: passage = f.read()
            # Clean syllabus sections (e.g. ## Syllabus Vocabulary, ## Syllabus Grammar) so only pure passage text is supplied
            clean_passage, _, _, _ = self.parse_syllabus_sections(passage)
            self._last_source_content = clean_passage or passage
            det_cefr = LinguisticEngine.calculate_text_cefr(clean_passage) if clean_passage else cefr
            if det_cefr:
                cefr = det_cefr
            return {"passage": clean_passage.strip() if clean_passage else passage, "cefr_level": cefr}

        elif quiz_type == "translation":
            # Per Rule 3.1: Assessment truth must be grounded directly on authentic source text via structured items.
            if not source_path or not source_path.exists():
                return None
            with open(source_path, "r", encoding="utf-8") as f:
                source_text = f.read()
            clean_passage, syl_v, syl_g, syl_e = self.parse_syllabus_sections(source_text)
            fresh_v_items, cefr, _ = _extract_fresh_vocab_items(source_path)
            if not fresh_v_items:
                return None

            indexed_text, pool = LinguisticEngine.tokenize_and_index_sentences(clean_passage)
            compile_defaults = self.config.get("compile_defaults") or {}
            target_k_g = compile_defaults.get("grammar", 5)
            det_g = LinguisticEngine.extract_deterministic_grammar(
                clean_passage,
                sentence_pool=pool,
                target_count=target_k_g,
                syllabus_grammar=syl_g
            )
            g_items = [
                GrammarItem(
                    quote=it['quote'],
                    pattern_formula=it['pattern_formula'],
                    pedagogical_function=it['pedagogical_function'],
                    design_audit=it.get('design_audit', ''),
                    imitation_example=it['imitation_example'],
                    common_mistakes=it['common_mistakes']
                )
                for it in (det_g or [])
            ]
            return {"vocab_list": fresh_v_items, "grammar_list": g_items, "cefr_level": cefr}

        elif quiz_type == "listening":
            if not source_path or not source_path.exists():
                return None
            fresh_v_items, cefr, _ = _extract_fresh_vocab_items(source_path)
            if not fresh_v_items:
                return None
            source_content = source_path.read_text(encoding="utf-8")
            clean_source, _, _, _ = self.parse_syllabus_sections(source_content)
            return {
                "vocab_list": fresh_v_items,
                "summary_content": "",
                "source_content": clean_source or source_content,
                "unit_core_name": core_name,
                "cefr_level": cefr
            }

        elif quiz_type == "video":
            video_url = ""
            video_type = ""
            from ..config import normalize_name
            normalized_core = normalize_name(core_name)
            unit_wiki_dir = self.config.wiki_content_path / core_name
            
            # Resolve actual unit_wiki_dir case-insensitively
            if self.config.wiki_content_path.exists():
                for child in self.config.wiki_content_path.iterdir():
                    if child.is_dir() and normalize_name(child.name) == normalized_core:
                        unit_wiki_dir = child
                        core_name = child.name  # Sync core_name
                        break

            source_path = None
            active_media_name = None
            if unit_wiki_dir.exists() and unit_wiki_dir.is_dir():
                # Dynamically resolve active media by scanning sources/media (most recently modified first)
                media_dir = unit_wiki_dir / "sources" / "media"
                if media_dir.exists() and media_dir.is_dir():
                    candidates = [f for f in media_dir.iterdir() if f.is_file() and f.suffix in [".md", ".txt"]]
                    if candidates:
                        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        source_path = candidates[0]
                        active_media_name = source_path.name

            # Fallback if no active_media or active_media not found
            if not source_path:
                if unit_wiki_dir.exists() and unit_wiki_dir.is_dir():
                    # Check sources/media/ first for any md/txt transcript files
                    media_dir = unit_wiki_dir / "sources" / "media"
                    if media_dir.exists() and media_dir.is_dir():
                        for f in media_dir.iterdir():
                            if f.is_file() and f.suffix in [".md", ".txt"]:
                                source_path = f
                                active_media_name = f.name
                                break
                    
                    if not source_path:
                        # Look for explicit video_transcript files first
                        for f in unit_wiki_dir.rglob("*.md"):
                            if f.is_file():
                                try:
                                    with open(f, "r", encoding="utf-8") as f_read:
                                        test_content = f_read.read()
                                    if 'category: "video_transcript"' in test_content or 'video_type:' in test_content or 'video_url:' in test_content or 'source_url:' in test_content:
                                        source_path = f
                                        break
                                except Exception:
                                    pass

                # If no explicit video transcript found, try standard source paths
                if not source_path or not source_path.exists():
                    source_path = self.config.wiki_content_path / core_name / "sources" / f"{core_name}.md"
                    if not source_path.exists():
                        source_path = self.config.wiki_content_path / core_name / "sources" / f"{core_name}.txt"
                    
                    # Fallback to any md file in the unit directory (excluding compiled node outputs)
                    if not source_path.exists() and unit_wiki_dir.exists() and unit_wiki_dir.is_dir():
                        for f in unit_wiki_dir.glob("*.md"):
                            if f.is_file() and not any(suffix in f.name for suffix in ["_vocabulary", "_grammar", "_summary"]):
                                source_path = f
                                break
                        if not source_path.exists():
                            for f in unit_wiki_dir.rglob("*.md"):
                                if f.is_file() and not any(suffix in f.name for suffix in ["_vocabulary", "_grammar", "_summary"]):
                                    source_path = f
                                    break

            if not source_path or not source_path.exists(): return None
            with open(source_path, "r", encoding="utf-8") as f: transcript = f.read()
            # Try multiple common keys for video URL (quoted or unquoted)
            for key in ["source_url", "video_url", "url", "source"]:
                m_url = re.search(fr'{key}:\s*["\']?([^\n"\']+)["\']?', transcript)
                if m_url and m_url.group(1).strip():
                    video_url = m_url.group(1).strip()
                    break
                    
            m_type = re.search(r'video_type:\s*["\']?([^\n"\']+)["\']?', transcript)
            if m_type:
                video_type = m_type.group(1).strip()
            elif video_url:
                # Auto-detect type based on URL
                url_lower = video_url.lower()
                if "youtube" in url_lower or "youtu.be" in url_lower:
                    video_type = "youtube"
                elif url_lower.endswith((".mp4", ".webm", ".ogg", ".mp3", ".wav")) or "/media/" in url_lower:
                    video_type = "local"
                else:
                    video_type = "youtube"  # default fallback if URL exists

            # If video_url is still empty and file is inside the media folder, resolve local companion
            if not video_url and ("media" in source_path.parts or (active_media_name and source_path.name == active_media_name)):
                parent_dir = source_path.parent
                for ext in [".mp4", ".mp3", ".webm", ".ogg", ".wav"]:
                    companion = parent_dir / f"{source_path.stem}{ext}"
                    if companion.exists():
                        video_url = f"/wiki/{core_name}/sources/media/{companion.name}"
                        video_type = "local"
                        break
            
            # For video quiz, if there is no valid video_url, skip video quiz gracefully rather than emitting 404 target
            if quiz_type == "video" and not video_url:
                return None

            # Rewrite local video URL to point to /wiki/<UnitName>/sources/media/
            if video_type == "local" and video_url:
                import os
                video_filename = os.path.basename(video_url)
                video_url = f"/wiki/{core_name}/sources/media/{video_filename}"
                
            return {"transcript": transcript, "video_url": video_url, "video_type": video_type, "cefr_level": cefr, "source_stem": source_path.stem}

        return None

    def _render_mindmap(self, mindmap_dict):
        """Renders the horizontal Mind Map HTML page with injected data."""
        template_path = Path(__file__).resolve().parent.parent / "templates" / "mindmap.html"
            
        if not template_path.exists():
            return f"<html><body><pre>{json.dumps(mindmap_dict, indent=2)}</pre></body></html>"

        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()
            
        json_data = json.dumps(mindmap_dict, ensure_ascii=False)
        
        if "const mindmapData =" in html:
            return re.sub(r'const mindmapData = .*?;', lambda _: f'const mindmapData = {json_data};', html, flags=re.DOTALL)
        return html.replace("</body>", f"<script>const mindmapData = {json_data};</script></body>")

    def _get_wiki_inventory(self):
        """Scans the wiki and raw directories recursively."""
        inventory = []
        wiki_dir = self.config.wiki_content_path
        if wiki_dir.exists():
            for f in wiki_dir.rglob("*.md"):
                if f.is_file():
                    if "_vocabulary" in f.name:
                        inventory.append({"name": f.stem, "type": "wiki_vocabulary", "path": f})
                    elif "_grammar" in f.name:
                        inventory.append({"name": f.stem, "type": "wiki_grammar", "path": f})
                    elif "_summary" in f.name:
                        inventory.append({"name": f.stem, "type": "wiki_summaries", "path": f})
                    else:
                        inventory.append({"name": f.stem, "type": "wiki_concepts", "path": f})
        
        # Add unit-specific sources to inventory as raw_source
        if wiki_dir.exists():
            for f in wiki_dir.rglob("sources/*.*"):
                if f.is_file() and f.suffix in [".md", ".txt"]:
                    try:
                        rel_name = str(f.relative_to(wiki_dir)).replace("\\", "/")
                        stem = f.stem
                        inventory.append({"name": rel_name, "type": "raw_source", "path": f})
                        inventory.append({"name": stem, "type": "raw_source", "path": f})
                    except Exception:
                        pass
        return inventory
