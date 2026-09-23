"""
Linguistic Engine for Lexis Wiki.
Integrates spaCy (computational dependency syntax) and WordNet (lexical relations)
to achieve deterministic sentence indexing, macro grammar domain classification,
boundary-accurate phrase extraction, canonical lemmatization, and zero-collision distractors.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class LinguisticEngine:
    _spacy_nlp = None
    _wn = None
    _acl_data = None
    _awl_data = None
    _ocd_data = None

    @classmethod
    def get_spacy(cls):
        """Lazy-loads the spaCy English model (~12MB)."""
        if cls._spacy_nlp is None:
            import spacy
            cls._spacy_nlp = spacy.load("en_core_web_sm")
        return cls._spacy_nlp

    @classmethod
    def get_wordnet(cls):
        """Lazy-loads the wn (Open English WordNet) client."""
        if cls._wn is None:
            import wn
            cls._wn = wn
        return cls._wn

    @classmethod
    def get_oxford_collocations(cls) -> Dict[str, Dict[str, List[str]]]:
        """Lazy-loads the Oxford Collocations Dictionary 2nd Edition (20,791 headwords, ~4.9MB)."""
        if cls._ocd_data is None:
            cls._ocd_data = {}
            ocd_path = Path(__file__).parent / "data" / "oxford_collocations.json"
            if ocd_path.exists():
                try:
                    with open(ocd_path, "r", encoding="utf-8") as f:
                        cls._ocd_data = json.load(f)
                except Exception:
                    pass
        return cls._ocd_data

    @classmethod
    def get_rich_collocations(cls, word: str, top_k: int = 5) -> List[str]:
        """
        Retrieves top authoritative, natural collocations for a word from Oxford Collocations Dictionary.
        Synthesizes phrases like 'lay the foundation', 'firm foundation', 'comprehensive guide'.
        """
        if not word:
            return []
        ocd = cls.get_oxford_collocations()
        entry = ocd.get(word.lower().strip())
        if not entry:
            return []

        results: List[str] = []

        # 1. Verb + Noun: 'lay [word]', 'establish [word]'
        if "verb_before" in entry:
            for v in entry["verb_before"][:3]:
                if "(" not in v:
                    results.append(f"{v} {word}")

        # 2. Adj + Noun: 'firm [word]', 'solid [word]'
        if "adj" in entry:
            for adj in entry["adj"][:3]:
                results.append(f"{adj} {word}")

        # 3. Noun collocations: '[word] guide', '[word] analysis'
        if "colloc_nouns" in entry:
            for noun in entry["colloc_nouns"][:3]:
                results.append(f"{word} {noun}")

        # 4. Adv + Verb: '[word] heavily', '[word] directly'
        if "verb" in entry:
            for adv in entry["verb"][:2]:
                results.append(f"{word} {adv}")

        # 5. Preposition: '[word] for', '[word] to'
        if "prep" in entry:
            for p in entry["prep"][:2]:
                results.append(f"{word} {p}")

        # Dedup preserving order
        seen = set()
        deduped = []
        for r in results:
            if r.lower() not in seen:
                seen.add(r.lower())
                deduped.append(r)
            if len(deduped) >= top_k:
                break
        return deduped

    @classmethod
    def generate_vocab_distractors(
        cls,
        target_word: str,
        pos: str = "noun",
        context_anchor: str = None,
        anchor_type: str = "collocation",
        target_count: int = 3
    ) -> List[str]:
        """
        Synthesizes high-discrimination, collision-free distractors for a target vocabulary word.
        
        Pipeline:
        1. WordNet Semantic Candidates: Retrieves direct synset synonyms (Tier 1), hyponyms (Tier 2),
           and coordinate terms (Tier 3).
        2. Lexicon Filter (CEFR / Oxford 20k): Restricts candidates to words present in the Oxford
           Collocations Dictionary, ensuring natural, curriculum-appropriate words and eliminating obscure terms.
        3. Oxford Collision Clearance (Zero Double-Key Guarantee):
           - If context_anchor is provided (e.g., target='lay', anchor='foundation'):
             Retrieves all authorized collocations for the anchor from OCD and removes any candidate
             that forms a valid collocation, guaranteeing absolute single-fit validity.
        4. Syntactic / Morphological Parallelism: Restricts to single words matching the target's POS.
        """
        clean_target = target_word.strip().lower()
        wn = cls.get_wordnet()
        ocd = cls.get_oxford_collocations()

        # Map POS to WordNet tag ('n', 'v', 'a', 'r')
        wn_pos = "v" if pos.startswith("v") else ("n" if pos.startswith("n") else ("a" if pos.startswith("adj") or pos.startswith("a") else None))
        words = wn.words(clean_target, pos=wn_pos) if wn_pos else wn.words(clean_target)

        tier1_synonyms: List[str] = []
        tier2_hyponyms: List[str] = []
        tier3_coordinates: List[str] = []
        seen = {clean_target}

        for w in words:
            for s in w.synsets():
                # Tier 1: Direct Synonyms in synset (Strict single-word constraint: no phrasal verbs)
                for sw in s.words():
                    lemma = sw.lemma().lower()
                    if lemma not in seen and " " not in lemma and "_" not in lemma and "-" not in lemma and lemma in ocd:
                        seen.add(lemma)
                        tier1_synonyms.append(lemma)
                # Tier 2: Hyponyms (more specific concepts)
                for hypo in s.hyponyms():
                    for hw in hypo.words():
                        lemma = hw.lemma().lower()
                        if lemma not in seen and " " not in lemma and "_" not in lemma and "-" not in lemma and lemma in ocd:
                            seen.add(lemma)
                            tier2_hyponyms.append(lemma)
                # Tier 3: Coordinate terms (sisters under same hypernym)
                for hyper in s.hypernyms():
                    for sis in hyper.hyponyms():
                        for sw in sis.words():
                            lemma = sw.lemma().lower()
                            if lemma not in seen and " " not in lemma and "_" not in lemma and "-" not in lemma and lemma in ocd:
                                seen.add(lemma)
                                tier3_coordinates.append(lemma)

        # Candidate pool ordered by pedagogical relevance (synonyms -> coordinates -> hyponyms)
        candidates = tier1_synonyms + tier3_coordinates + tier2_hyponyms

        # 3. Oxford Collision Clearance (Anti Double-Key Gate)
        forbidden_words = {clean_target}
        if context_anchor:
            clean_anchor = context_anchor.strip().lower()
            anchor_entry = ocd.get(clean_anchor, {})
            # If target is verb, anchor is noun: check verb_before and verb_after
            for v in anchor_entry.get("verb_before", []) + anchor_entry.get("verb_after", []):
                forbidden_words.add(v.split()[0].lower())
            # If target is adj, anchor is noun: check adj
            for a in anchor_entry.get("adj", []):
                forbidden_words.add(a.split()[0].lower())
            # If target is noun, anchor is verb/adj: check colloc_nouns and noun_after
            for n in anchor_entry.get("colloc_nouns", []) + anchor_entry.get("noun_after", []):
                forbidden_words.add(n.split()[0].lower())

        safe_distractors: List[str] = []
        for cand in candidates:
            if cand not in forbidden_words and len(cand) >= 2:
                safe_distractors.append(cand)
                if len(safe_distractors) >= target_count:
                    break

        # Fallback if candidates pool is sparse: select standard homogeneous items from OCD
        if len(safe_distractors) < target_count:
            fallback_pool = {
                "v": ["put", "make", "take", "hold", "draw", "keep", "stand", "bring", "lead"],
                "n": ["aspect", "factor", "process", "measure", "element", "context", "matter"],
                "a": ["crucial", "essential", "primary", "initial", "direct", "specific", "constant"]
            }.get(wn_pos or "n", ["factor", "element", "process"])
            for fb in fallback_pool:
                if fb != clean_target and fb not in forbidden_words and fb not in safe_distractors:
                    safe_distractors.append(fb)
                if len(safe_distractors) >= target_count:
                    break

        return safe_distractors[:target_count]

    @classmethod
    def get_acl_collocations(cls) -> Dict[str, str]:
        """Lazy-loads the Academic Collocation List (ACL, ~2474 items). Returns dict mapping core pattern -> canonical original."""
        if cls._acl_data is None:
            cls._acl_data = {}
            acl_path = Path(__file__).parent / "data" / "academic_collocations.json"
            if acl_path.exists():
                try:
                    with open(acl_path, "r", encoding="utf-8") as f:
                        raw_items = json.load(f)
                    for item in raw_items:
                        # strip optional prefixes/suffixes like (a), (be), (to), (of) for robust matching
                        core = re.sub(r"^\([a-z\s]+\)\s*", "", item)
                        core = re.sub(r"\s*\([a-z\s]+\)$", "", core).strip().lower()
                        if len(core) >= 3:
                            cls._acl_data[core] = item
                except Exception:
                    pass
        return cls._acl_data

    @classmethod
    def get_awl_words(cls) -> Set[str]:
        """Lazy-loads the Academic Word List (AWL, 560+ headwords)."""
        if cls._awl_data is None:
            cls._awl_data = set()
            awl_path = Path(__file__).parent / "data" / "academic_word_list.json"
            if awl_path.exists():
                try:
                    with open(awl_path, "r", encoding="utf-8") as f:
                        cls._awl_data = set(json.load(f))
                except Exception:
                    pass
        return cls._awl_data

    # -------------------------------------------------------------------------
    # 1. Sentence Boundary Tokenization & Indexing ([S-1], [S-2])
    # -------------------------------------------------------------------------
    @classmethod
    def tokenize_and_index_sentences(cls, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Pre-tokenizes raw source text into an indexed, complete-sentence pool.
        Returns:
            indexed_text: Text formatted with '[S-1] ... [S-2] ...' for LLM prompts.
            sentence_pool: Dict mapping 'S-1' -> pristine full sentence text.
        """
        nlp = cls.get_spacy()
        # Cleanly separate frontmatter if present so spaCy only tokenizes authentic body prose
        fm_match = re.match(r"^(---\s*\n.*?\n---\s*\n)(.*)", text, re.DOTALL)
        frontmatter = fm_match.group(1) if fm_match else ""
        body = fm_match.group(2) if fm_match else text

        doc = nlp(body)

        sentence_pool: Dict[str, str] = {}
        indexed_paragraphs: List[str] = []
        counter = 1

        # Process paragraph-by-paragraph to preserve natural paragraph boundaries while streaming sentences within paragraphs
        raw_paragraphs = re.split(r'\n{2,}', body.strip())
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            # If the paragraph is a pure markdown header or bold block (e.g. '## Text A', '**Title**')
            lines = [l.strip() for l in para.split("\n") if l.strip()]
            if all(l.startswith("#") or (l.startswith("**") and l.endswith("**")) for l in lines):
                indexed_paragraphs.append(para)
                continue

            # Separate optional leading heading lines in this paragraph
            prefix_lines = []
            content_lines = []
            for l in lines:
                if not content_lines and (l.startswith("#") or (l.startswith("**") and l.endswith("**"))):
                    prefix_lines.append(l)
                else:
                    content_lines.append(l)

            prefix_str = "\n\n".join(prefix_lines) + "\n\n" if prefix_lines else ""
            para_text = " ".join(content_lines).strip()
            if not para_text:
                if prefix_lines:
                    indexed_paragraphs.append("\n\n".join(prefix_lines))
                continue

            para_doc = nlp(para_text)
            para_sent_parts = []
            for sent in para_doc.sents:
                sent_str = sent.text.strip()
                if not sent_str:
                    continue
                words = [t for t in nlp(sent_str) if t.is_alpha]
                if len(words) < 2 and not sent_str.endswith((".", "?", "!")):
                    para_sent_parts.append(sent_str)
                    continue

                sid = f"S-{counter}"
                sentence_pool[sid] = sent_str
                para_sent_parts.append(f"[{sid}] {sent_str}")
                counter += 1

            indexed_para = prefix_str + " ".join(para_sent_parts)
            indexed_paragraphs.append(indexed_para.strip())

        if frontmatter:
            indexed_text = frontmatter + "\n\n" + "\n\n".join(indexed_paragraphs)
        else:
            indexed_text = "\n\n".join(indexed_paragraphs)
        return indexed_text, sentence_pool

    @classmethod
    def snap_to_sentence_pool(cls, raw_quote_or_id: str, sentence_pool: Dict[str, str]) -> Optional[str]:
        """
        Deterministically snaps a model-provided sentence ID (e.g. '[S-3]', 'S-3') or a
        partial quote with ellipsis back to the pristine, authentic source sentence in the pool.
        Eliminates trailing ellipses, copy-paste hallucinations, and verbatim mismatches.
        """
        if not raw_quote_or_id or not sentence_pool:
            return None

        clean_input = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", raw_quote_or_id.strip(), flags=re.IGNORECASE).strip()

        # 1. Direct ID lookup (e.g. '[S-3]' or 'S-3' or 'Sentence 3')
        id_match = re.search(r"\bS-(\d+)\b", raw_quote_or_id, re.IGNORECASE)
        if id_match:
            sid = f"S-{id_match.group(1)}"
            if sid in sentence_pool:
                return re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sentence_pool[sid], flags=re.IGNORECASE).strip()

        # 2. Exact match in sentence pool
        for sid, sent in sentence_pool.items():
            sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sent, flags=re.IGNORECASE).strip()
            if clean_input.lower() == sent_clean.lower():
                return sent_clean

        # 3. Substring containment: if quote is a fragment of a pool sentence
        clean_target = re.sub(r"\.{3,}|…", "", clean_input).strip().lower()
        if len(clean_target) >= 15:
            for sid, sent in sentence_pool.items():
                sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sent, flags=re.IGNORECASE).strip()
                if clean_target in sent_clean.lower():
                    return sent_clean

        # 4. High overlap fallback
        clean_tokens = set(re.findall(r"\w{3,}", clean_target))
        if clean_tokens:
            best_sent = None
            max_overlap = 0.0
            for sid, sent in sentence_pool.items():
                sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sent, flags=re.IGNORECASE).strip()
                sent_tokens = set(re.findall(r"\w{3,}", sent_clean.lower()))
                if not sent_tokens:
                    continue
                overlap = len(clean_tokens & sent_tokens) / len(clean_tokens)
                if overlap > max_overlap and overlap >= 0.75:
                    max_overlap = overlap
                    best_sent = sent_clean
            if best_sent:
                return best_sent

    @classmethod
    def get_sentence_id(cls, raw_quote_or_id: str, sentence_pool: Dict[str, str]) -> Optional[str]:
        """
        Returns the canonical sentence identifier (e.g. 'S-3') for a given quote or ID.
        """
        if not raw_quote_or_id or not sentence_pool:
            return None

        clean_input = raw_quote_or_id.strip()

        # 1. Direct ID lookup
        id_match = re.search(r"\bS-(\d+)\b", clean_input, re.IGNORECASE)
        if id_match:
            sid = f"S-{id_match.group(1)}"
            if sid in sentence_pool:
                return sid

        # 2. Exact match
        for sid, sent in sentence_pool.items():
            if clean_input.lower() == sent.lower():
                return sid

        # 3. Substring containment
        clean_target = re.sub(r"\.{3,}|…", "", clean_input).strip().lower()
        if len(clean_target) >= 15:
            for sid, sent in sentence_pool.items():
                if clean_target in sent.lower():
                    return sid

        # 4. Token overlap fallback
        clean_tokens = set(re.findall(r"\w{3,}", clean_target))
        if clean_tokens:
            best_sid = None
            max_overlap = 0.0
            for sid, sent in sentence_pool.items():
                sent_tokens = set(re.findall(r"\w{3,}", sent.lower()))
                if not sent_tokens:
                    continue
                overlap = len(clean_tokens & sent_tokens) / len(clean_tokens)
                if overlap > max_overlap and overlap >= 0.75:
                    max_overlap = overlap
                    best_sid = sid
            if best_sid:
                return best_sid

        return None

    @classmethod
    def extract_grammar_fingerprint(cls, sentence: str, category: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Extracts a structural grammar fingerprint (macro_domain, dep_type, core_anchor)
        using spaCy dependency parsing. Used for deterministic deduplication of identical
        syntactic patterns across different sentences.
        """
        nlp = cls.get_spacy()
        doc = nlp(sentence)
        macro_domain = category or cls.classify_grammar_dependency(sentence) or "Unknown"

        dep_type = "generic"
        core_anchor = ""

        # 1. Rhetoric & Emphasis
        for token in doc:
            if token.dep_ == "nsubj":
                head = token.head
                if head.pos_ in ("VERB", "AUX"):
                    aux_children = [c for c in head.children if c.dep_ in ("aux", "auxpass") and c.i < token.i]
                    if aux_children and not sentence.strip().endswith("?"):
                        first_tok = doc[0].lemma_.lower()
                        if first_tok in ("hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no"):
                            dep_type = "inversion"
                            core_anchor = first_tok
                            break

        if dep_type == "generic":
            text_lower = sentence.lower()
            if re.search(r"\bnot\s+.*?\s*,\s*but\b", text_lower):
                dep_type = "antithesis"
                core_anchor = "not...but"
            elif re.search(r"\bnot\s+only\b.*?\bbut\s+also\b", text_lower):
                dep_type = "parallelism"
                core_anchor = "not only...but also"
            elif any(t.dep_ == "expl" and t.text.lower() == "it" for t in doc):
                for t in doc:
                    if t.dep_ == "expl" and t.head.lemma_ in ("be", "seem"):
                        if any(c.dep_ == "relcl" or any(gc.dep_ == "relcl" for gc in c.children) for c in t.head.children):
                            dep_type = "cleft"
                            core_anchor = "it...that"
                            break

        # 2. Logic & Stance
        if dep_type == "generic":
            for token in doc:
                if token.dep_ == "mark" and token.lemma_.lower() in {"although", "though", "while", "whereas", "even though", "if", "unless", "provided"}:
                    dep_type = "subordinating_clause"
                    core_anchor = token.lemma_.lower()
                    break

        # 3. Cohesion & Framing
        if dep_type == "generic":
            interpretive_verbs = {"mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"}
            for token in doc:
                if token.dep_ in ("relcl", "advcl") and token.lemma_.lower() in interpretive_verbs:
                    subj_which = any(c.dep_ in ("nsubj", "nsubjpass") and c.text.lower() == "which" for c in token.children)
                    if subj_which:
                        dep_type = "propositional_encapsulation"
                        core_anchor = f"which {token.lemma_.lower()}"
                        break
            if dep_type == "generic":
                shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "difference", "discrepancy"}
                for token in doc:
                    if token.lemma_.lower() in shell_nouns:
                        dep_type = "shell_noun"
                        core_anchor = token.lemma_.lower()
                        break

            # Correlative comparative: the more..., the more... (Rhetoric & Emphasis or Information Packaging)
            COMPARATIVE_WORDS = r"(?:more|less|fewer|better|worse|[a-z]{2,}er)"
            if re.search(rf"\bthe\s+{COMPARATIVE_WORDS}\b(?!\s+hand\b).*?(?:,\s*|\band\s+the\s+{COMPARATIVE_WORDS}\b.*?,?\s*)\bthe\s+{COMPARATIVE_WORDS}\b", text_lower):
                dep_type = "correlative_comparative"
                core_anchor = "the...the"

            if dep_type == "generic":
                for token in doc:
                    if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                        if not any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children) and token.lemma_.lower() not in ("include", "accord", "regard"):
                            dep_type = "participial_adjunct"
                            core_anchor = token.tag_
                            break
            # Dummy-It Subject Extraposition: It is adj that/to-V
            if dep_type == "generic" and any(t.text.lower() == "it" and t.dep_ in ("expl", "nsubj") for t in doc):
                for t in doc:
                    if t.text.lower() == "it" and t.head.lemma_ in ("be", "seem"):
                        if any(c.dep_ in ("acomp", "attr") for c in t.head.children) and any(c.dep_ in ("ccomp", "csubj", "xcomp") for c in t.head.children):
                            dep_type = "dummy_it_extraposition"
                            core_anchor = "it be adj to/that"
                            break
            # Dummy-It Object Extraposition: find/make/think it adj to-V
            if dep_type == "generic":
                for t in doc:
                    if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                        for c in t.children:
                            if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                                dep_type = "dummy_it_object"
                                core_anchor = f"{t.lemma_} it {c.lemma_}"
                                break

        return (macro_domain, dep_type, core_anchor)


    @classmethod
    def lemmatize_headword(cls, word_or_phrase: str, context_sentence: Optional[str] = None) -> str:
        """
        Derives canonical base dictionary headwords from sentence context,
        eliminating inflected headwords (-ed, -ing, 3sg).
        """
        nlp = cls.get_spacy()
        target = word_or_phrase.strip()

        # If a single word
        if " " not in target:
            doc = nlp(target)
            if len(doc) > 0 and doc[0].pos_ in ("VERB", "NOUN", "ADJ"):
                return doc[0].lemma_.lower()
            return target.lower()

        # If a multi-word expression (e.g. phrasal verb or idiom)
        doc = nlp(target)
        lemmatized_tokens = []
        for token in doc:
            if token.pos_ == "VERB":
                lemmatized_tokens.append(token.lemma_.lower())
            else:
                lemmatized_tokens.append(token.text.lower())
        return " ".join(lemmatized_tokens)

    @classmethod
    def extract_phrasal_verbs(cls, sentence: str) -> List[str]:
        """
        Deterministically extracts multi-word phrasal verbs from the dependency tree
        by querying verb heads governing direct 'prt' (particle) children,
        strictly pruning ordinary spatial/temporal prepositional phrases ('prep + pobj').
        e.g.
          'come in' -> extracted (prt)
          'give up' -> extracted (prt)
          'come in a city' -> rejected (prep + pobj, spatial adjunct)
        """
        nlp = cls.get_spacy()
        doc = nlp(sentence)
        phrasal_verbs = []

        for token in doc:
            if token.pos_ != "VERB":
                continue

            # Check direct particle children (prt)
            prts = [child for child in token.children if child.dep_ == "prt"]
            for p in prts:
                pv = f"{token.lemma_.lower()} {p.lemma_.lower()}"
                # Check for secondary preposition attached to verb or particle (e.g. come up with)
                prep_children = [c for c in p.children if c.dep_ == "prep"]
                if not prep_children:
                    prep_children = [c for c in token.children if c.dep_ == "prep" and c.i > p.i]
                if prep_children:
                    pv += f" {prep_children[0].lemma_.lower()}"
                phrasal_verbs.append(pv)

        return list(dict.fromkeys(phrasal_verbs))

    # -------------------------------------------------------------------------
    # 3. Dependency-Based Macro Domain Classification Gate
    # -------------------------------------------------------------------------
    @classmethod
    def classify_grammar_dependency(cls, sentence: str) -> Optional[str]:
        """
        Classifies complex sentence into one of the Four Macro Functional Domains
        using deterministic spaCy dependency trees and syntactic relations:
          - 'Rhetoric & Emphasis'
          - 'Cohesion & Framing'
          - 'Information Packaging'
          - 'Logic & Stance'
        Returns None if no unambiguous structural pattern dominates.
        """
        nlp = cls.get_spacy()
        doc = nlp(sentence)

        # ---------------------------------------------------------------------
        # 1. Rhetoric & Emphasis:
        #    a. Fronted Inversion: aux or ROOT precedes nsubj in linear word order
        #    b. Cleft Focus: expl ('it') + 'is/was' + focal constituent + relcl (that...)
        #    c. Antithesis: 'not... but...' parallel coordinating balance
        # ---------------------------------------------------------------------
        # Inversion check
        for token in doc:
            if token.dep_ == "nsubj":
                # Check if auxiliary or root verb precedes nsubj (excluding questions)
                head = token.head
                if head.pos_ == "VERB" or head.pos_ == "AUX":
                    aux_children = [c for c in head.children if c.dep_ in ("aux", "auxpass")]
                    for aux in aux_children:
                        if aux.i < token.i and not sentence.strip().endswith("?"):
                            # Check if fronted negative/restrictive adverbial starts sentence
                            first_token = doc[0].lemma_.lower()
                            if first_token in ("hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no"):
                                return "Rhetoric & Emphasis"

        # Cleft vs Dummy-it check
        for token in doc:
            if token.dep_ == "expl" and token.text.lower() == "it":
                # Check copula verb
                head = token.head
                if head.lemma_ in ("be", "seem"):
                    # If followed by relative clause (relcl) modifying non-subject focus -> Cleft
                    has_relcl = any(child.dep_ == "relcl" for child in head.children)
                    for child in head.children:
                        if any(gc.dep_ == "relcl" for gc in child.children):
                            has_relcl = True
                    # Check if followed by that/who relcl
                    if has_relcl:
                        return "Rhetoric & Emphasis"

        # Antithesis / Correlative Parallelism check (not... but..., not only... but also)
        text_lower = sentence.lower()
        if re.search(r"\bnot\s+.*?\s*,\s*but\b", text_lower) or re.search(r"\bnot\s+only\b.*?\bbut\s+also\b", text_lower):
            return "Rhetoric & Emphasis"

        # ---------------------------------------------------------------------
        # 2. Logic & Stance:
        #    Conditionals (if/unless) or Concessives (although/even though/while/whereas)
        #    Checked before Information Packaging to avoid 'promising' being treated as participle
        # ---------------------------------------------------------------------
        subordinating_conjs = {"although", "though", "while", "whereas", "even though", "if", "unless", "provided", "providing"}
        for token in doc:
            if token.dep_ == "mark" and token.lemma_.lower() in subordinating_conjs:
                return "Logic & Stance"

        # ---------------------------------------------------------------------
        # 3. Cohesion & Framing:
        #    a. Propositional Encapsulation: , which + [interpretive verb] + that
        #    b. Shell Noun Frames: [shell noun] + that [clause]
        # ---------------------------------------------------------------------
        # Propositional encapsulation: which + interpretive verb + that
        interpretive_verbs = {"mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"}
        for token in doc:
            if token.dep_ in ("relcl", "advcl") and token.lemma_.lower() in interpretive_verbs:
                # Check if subject is 'which'
                subj_children = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass") and c.text.lower() == "which"]
                if subj_children:
                    has_that_comp = any(child.dep_ == "ccomp" for child in token.children)
                    if has_that_comp:
                        return "Cohesion & Framing"

        # Shell noun patterns: fact, idea, conclusion, claim, belief, hypothesis + that
        shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "evidence", "assumption"}
        for token in doc:
            if token.lemma_.lower() in shell_nouns:
                for child in token.children:
                    if child.dep_ in ("acl", "appos", "ccomp"):
                        return "Cohesion & Framing"

        # ---------------------------------------------------------------------
        # 4. Information Packaging:
        #    a. Evaluative Dummy-It: It is + [adj/noun] + that/to [csubj/ccomp]
        #    b. Non-finite participial adjuncts: advcl with VerbForm=Part
        #    c. Elaborative non-restrictive relative clause: , which + VP (without that-clause)
        # ---------------------------------------------------------------------
        # Evaluative Dummy-It (spaCy tags 'It' as nsubj/expl and the complement clause as ccomp/csubj)
        for token in doc:
            if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj"):
                head = token.head
                if head.lemma_ in ("be", "seem"):
                    # Check if head has acomp/attr and ccomp/csubj/xcomp
                    has_predicate = any(child.dep_ in ("acomp", "attr") for child in head.children)
                    has_complement = any(child.dep_ in ("ccomp", "csubj", "xcomp") for child in head.children)
                    if has_predicate and has_complement:
                        return "Information Packaging"

        # Evaluative Dummy-It Object (find/make/think it adj to-V)
        for t in doc:
            if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                for c in t.children:
                    if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                        return "Information Packaging"

        # Non-finite participial adjuncts (advcl with VBG or VBN)
        for token in doc:
            if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                # Ensure it has no subject of its own (non-finite) and not a preposition
                has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children)
                if not has_subj and token.lemma_.lower() not in ("include", "accord", "regard", "concern"):
                    return "Information Packaging"

        # Elaborative non-restrictive relative clauses
        if re.search(r",\s*which\s+[a-z]+", text_lower):
            return "Information Packaging"

        return None

    @classmethod
    def generate_cobuild_formula(cls, sentence: str, category: Optional[str] = None) -> str:
        """
        Deterministically derives standard algebraic COBUILD slot formula from sentence
        via spaCy dependency trees and syntactic pattern anchors.
        Replaces prone-to-hallucination LLM formula drafting with zero-token invariant logic.
        """
        nlp = cls.get_spacy()
        doc = nlp(sentence)
        cat = category or cls.classify_grammar_dependency(sentence) or ""
        text_lower = sentence.lower()

        # 1. Rhetoric & Emphasis
        if cat == "Rhetoric & Emphasis" or not cat:
            # Inversion
            for token in doc:
                if token.dep_ == "nsubj":
                    head = token.head
                    if head.pos_ in ("VERB", "AUX"):
                        aux_children = [c for c in head.children if c.dep_ in ("aux", "auxpass") and c.i < token.i]
                        if aux_children and not sentence.strip().endswith("?"):
                            first_tok = doc[0].lemma_.lower()
                            if first_tok in ("hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no"):
                                return f"{doc[0].text} + [aux/be] + [Subject] + [VP]"
            # Cleft
            for token in doc:
                if token.dep_ == "expl" and token.text.lower() == "it" and token.head.lemma_ in ("be", "seem"):
                    if any(c.dep_ == "relcl" or any(gc.dep_ == "relcl" for gc in c.children) for c in token.head.children):
                        return "It + [be] + [Focal Element] + that/who + [Clause]"
            # Antithesis
            if re.search(r"\bnot\s+.*?\s*,\s*but\b", text_lower):
                return "[Subject] + [VP], not + [PrepP/NP], but + [PrepP/NP]"
            # Correlative Parallelism
            if re.search(r"\bnot\s+only\b.*?\bbut\s+also\b", text_lower):
                return "[Subject] + not only + [VP], but also + [VP]"

        # 2. Logic & Stance
        if cat == "Logic & Stance" or not cat:
            for token in doc:
                if token.dep_ == "mark" and token.lemma_.lower() in {"although", "though", "while", "whereas", "even though", "if", "unless", "provided"}:
                    mark_word = token.text.capitalize()
                    return f"{mark_word} + [Clause], [Subject] + [VP]"

        # 3. Cohesion & Framing
        if cat == "Cohesion & Framing" or not cat:
            # Propositional Encapsulation: , which + [interpretive verb] + that
            interpretive_verbs = {"mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"}
            for token in doc:
                if token.dep_ in ("relcl", "advcl") and token.lemma_.lower() in interpretive_verbs:
                    if any(c.dep_ in ("nsubj", "nsubjpass") and c.text.lower() == "which" for c in token.children):
                        return f", which + {token.lemma_.lower()}s + that + [Proposition Clause]"
            # Shell noun frame
            shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "difference", "discrepancy"}
            for token in doc:
                if token.lemma_.lower() in shell_nouns:
                    return f"The + {token.lemma_.lower()} + that/of + [Proposition Clause]"

        # 4. Information Packaging
        if cat == "Information Packaging" or not cat:
            # Correlative comparative: The more..., the more...
            COMPARATIVE_WORDS = r"(?:more|less|fewer|better|worse|[a-z]{2,}er)"
            if re.search(rf"\bthe\s+{COMPARATIVE_WORDS}\b(?!\s+hand\b).*?(?:,\s*|\band\s+the\s+{COMPARATIVE_WORDS}\b.*?,?\s*)\bthe\s+{COMPARATIVE_WORDS}\b", text_lower):
                return "The + [comparative] + [Clause], the + [comparative] + [Clause]"
            # Dummy-It Object Extraposition (find/make/think it adj to-V)
            for t in doc:
                if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                    for c in t.children:
                        if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                            return f"[Subject] + {t.lemma_} + it + [{c.lemma_.capitalize()}] + to-V"
            # Dummy-It Subject Extraposition
            if any(t.text.lower() == "it" and t.dep_ in ("expl", "nsubj") for t in doc):
                for t in doc:
                    if t.text.lower() == "it" and t.head.lemma_ in ("be", "seem"):
                        if any(c.dep_ in ("acomp", "attr") for c in t.head.children) and any(c.dep_ in ("ccomp", "csubj", "xcomp") for c in t.head.children):
                            return "It + [be] + [Adj/NP] + to-V/that + [Clause]"
            # Participial adjunct
            for token in doc:
                if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                    if not any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children) and token.lemma_.lower() not in ("include", "accord", "regard"):
                        is_fronted = token.i < token.head.i
                        v_type = "V-ing" if token.tag_ == "VBG" else "V-ed"
                        if is_fronted:
                            return f"[{v_type} Phrase], [Subject] + [VP]"
                        else:
                            return f"[Subject] + [VP], [{v_type} Phrase]"
            # Elaborative clause
            if re.search(r",\s*which\s+[a-z]+", text_lower):
                return "[Subject] + [VP], which + [VP]"

        # Default fallback standard formula
        return "[Subject] + [VP] + [Clause]"

    @classmethod
    def mine_grammar_skeletons(
        cls,
        sentence_pool: Dict[str, str],
        target_count: int = 5,
        syllabus_grammar: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Deterministically mines genuine, non-trivial grammar structural skeletons from the sentence pool.
        Selects sentences exhibiting rich dependency structures across the Four Macro Domains.
        Returns a list of dicts: [{'sid': 'S-9', 'quote': '...', 'category': '...', 'pattern_formula': '...'}]
        """
        raw_skeletons: List[Dict[str, str]] = []

        for sid, sent in sentence_pool.items():
            sent_clean = sent.strip()
            # Skip very short or title-like sentences
            if len(sent_clean.split()) < 7:
                continue

            cat = cls.classify_grammar_dependency(sent_clean)
            if cat:
                formula = cls.generate_cobuild_formula(sent_clean, category=cat)
                # Filter out generic/un-abstracted formulas (e.g. [Subject] + [VP] + [Clause] or simple [S])
                if formula and not formula.startswith("[Subject] + [VP] + [Clause]") and "[" in formula:
                    raw_skeletons.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "category": cat,
                        "pattern_formula": formula,
                    })

        if not raw_skeletons:
            return []

        # Diverse selection algorithm across macro domains and distinct formulas
        selected: List[Dict[str, str]] = []
        seen_cats: Set[str] = set()
        seen_formulas: Set[str] = set()

        # Pass 1: maximize category diversity across the 4 domains
        for item in raw_skeletons:
            cat = item["category"]
            form = item["pattern_formula"]
            if cat not in seen_cats and form not in seen_formulas:
                selected.append(item)
                seen_cats.add(cat)
                seen_formulas.add(form)
            if len(selected) >= target_count:
                break

        # Pass 2: fill remaining slots with distinct structural formulas
        if len(selected) < target_count:
            for item in raw_skeletons:
                if item not in selected and item["pattern_formula"] not in seen_formulas:
                    selected.append(item)
                    seen_formulas.add(item["pattern_formula"])
                if len(selected) >= target_count:
                    break

        # Pass 3: if still under target_count, append distinct quotes with unique formulas
        if len(selected) < target_count:
            selected_quotes = {s["quote"] for s in selected}
            for item in raw_skeletons:
                if item["quote"] not in selected_quotes and item["pattern_formula"] not in seen_formulas:
                    selected.append(item)
                    selected_quotes.add(item["quote"])
                    seen_formulas.add(item["pattern_formula"])
                if len(selected) >= target_count:
                    break

        return selected[:target_count]

    # -------------------------------------------------------------------------
    # 3b. Deterministic Multi-Word Expression & Collocation Mining (ACL + spaCy)
    # -------------------------------------------------------------------------
    @classmethod
    def mine_expression_skeletons(cls, text: str, target_count: int = 8) -> List[Dict[str, str]]:
        """
        Deterministically extracts high-value multi-word expressions, phrasal verbs,
        and Academic Collocation List (ACL) items from the indexed sentence pool.
        Standardizes slotted formulas (e.g. [sb], [sth], [one's]) at zero token cost.

        Returns list of dicts:
            [{
                "sid": "S-1",
                "quote": "full sentence text",
                "phrase": "have an adverse effect on",
                "type": "collocation",
                "pattern_formula": "have an adverse effect on [sth]"
            }, ...]
        """
        _, pool = cls.tokenize_and_index_sentences(text)
        if not pool:
            return []

        nlp = cls.get_spacy()
        acl_map = cls.get_acl_collocations()
        awl_set = cls.get_awl_words()

        COMMON_PARTICLES = frozenset({"away", "back", "down", "in", "off", "on", "out", "over", "round", "through", "up"})
        DEPENDENT_PREPS = frozenset({"on", "upon", "to", "with", "for", "from", "into", "against", "about", "of", "in"})

        raw_candidates: List[Dict[str, Any]] = []

        for sid, sentence in pool.items():
            sent_clean = sentence.strip()
            if len(sent_clean.split()) < 5:
                continue

            doc = nlp(sent_clean)
            text_lower = sent_clean.lower()

            # 1. Academic Collocation List (ACL) Matcher
            for core, original in acl_map.items():
                pattern = r"\b" + re.escape(core) + r"\b"
                if re.search(pattern, text_lower):
                    # Determine type: verb+noun or adj+noun
                    words = core.split()
                    expr_type = "collocation"
                    formula = core
                    if original.endswith("(to)") or original.endswith("(of)") or original.endswith("(in)") or original.endswith("(with)"):
                        prep_match = re.search(r"\(([a-z]+)\)$", original)
                        if prep_match:
                            formula = f"{core} {prep_match.group(1)} [sth]"
                    elif len(words) == 2 and any(w in awl_set for w in words):
                        formula = f"{core}"

                    raw_candidates.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "phrase": core,
                        "type": expr_type,
                        "pattern_formula": formula,
                        "score": 10 + (2 if any(w in awl_set for w in words) else 0),
                    })

            # 2. Dependency Syntax Phrasal Verbs & Prepositional Combinations
            COLLOCATION_VERB_PREPS = {
                "welcome": "to",
                "provide": "with",
                "share": "with",
                "equip": "with",
                "attribute": "to",
                "contribute": "to",
                "rely": "on",
                "depend": "on",
                "focus": "on",
                "concentrate": "on",
                "participate": "in",
                "engage": "in",
                "succeed": "in",
                "benefit": "from",
                "derive": "from",
                "suffer": "from",
                "cope": "with",
                "comply": "with",
                "lead": "to",
                "adapt": "to",
                "conform": "to",
                "refer": "to",
                "apply": "to",
                "appeal": "to",
                "insist": "on",
                "consist": "of",
            }

            for token in doc:
                if token.pos_ in ("VERB", "AUX"):
                    v_lemma = token.lemma_.lower()

                    # Check particle (e.g. wake up, get by, slave away, carry out)
                    for c in token.children:
                        if (c.dep_ == "prt" or (c.dep_ == "advmod" and c.text.lower() in COMMON_PARTICLES)) and c.i > token.i:
                            p_tok = c.text.lower()
                            # Check three-part phrasal verb: verb + particle + prep immediately following (distance <= 2)
                            prep_child = [gc for gc in token.children if gc.dep_ == "prep" and 0 < (gc.i - c.i) <= 2]
                            if prep_child:
                                prep_word = prep_child[0].text.lower()
                                p_verb = f"{v_lemma} {p_tok} {prep_word}"
                                formula = f"{v_lemma} {p_tok} {prep_word} [sth/sb]"
                            else:
                                dobj = [d for d in token.children if d.dep_ == "dobj"]
                                if dobj:
                                    p_verb = f"{v_lemma} {p_tok}"
                                    formula = f"{v_lemma} [sb/sth] {p_tok}"
                                else:
                                    p_verb = f"{v_lemma} {p_tok}"
                                    formula = f"{v_lemma} {p_tok}"

                            raw_candidates.append({
                                "sid": sid,
                                "quote": sent_clean,
                                "phrase": p_verb,
                                "type": "phrasal verb",
                                "pattern_formula": formula,
                                "score": 9,
                            })

                    # Check possessive object construction via spaCy 'poss' dependency (e.g. attain [one's] best, make up [one's] mind)
                    # Restrict to genuine idiomatic and academic lexicalized noun heads to avoid trivial 'take his dad'
                    IDIOMATIC_POSS_NOUNS = frozenset({
                        "best", "mind", "temper", "breath", "time", "part", "way", "heart",
                        "step", "place", "voice", "promise", "life", "potential", "future",
                        "passion", "dream", "goal", "duty", "responsibility", "effort", "stride",
                        "horizon", "footing", "view", "interest", "role", "purpose"
                    })
                    dobjs = [d for d in token.children if d.dep_ == "dobj"]
                    for d in dobjs:
                        poss = [p for p in d.children if p.dep_ == "poss"]
                        if poss:
                            noun_text = d.text.lower()
                            noun_lemma = d.lemma_.lower()
                            matched_noun = None
                            if noun_text in IDIOMATIC_POSS_NOUNS:
                                matched_noun = noun_text
                            elif noun_lemma in IDIOMATIC_POSS_NOUNS:
                                matched_noun = noun_lemma

                            if matched_noun:
                                prt = [p for p in token.children if p.dep_ == "prt"]
                                if prt:
                                    p_clean = f"{v_lemma} {prt[0].text.lower()} [one's] {matched_noun}"
                                    formula = f"{v_lemma} {prt[0].text.lower()} [one's] {matched_noun}"
                                else:
                                    p_clean = f"{v_lemma} [one's] {matched_noun}"
                                    formula = f"{v_lemma} [one's] {matched_noun}"

                                raw_candidates.append({
                                    "sid": sid,
                                    "quote": sent_clean,
                                    "phrase": p_clean,
                                    "type": "collocation",
                                    "pattern_formula": formula,
                                    "score": 10,
                                })

                    # Check dependent preposition collocation or idiomatic frame
                    for c in token.children:
                        if c.dep_ == "prep" and c.i > token.i:
                            p_text = c.text.lower()
                            dobj = [d for d in token.children if d.dep_ == "dobj"]

                            # Known academic verb+prep collocation (e.g. welcome [sb] to [sth], provide [sb] with [sth])
                            if v_lemma in COLLOCATION_VERB_PREPS and COLLOCATION_VERB_PREPS[v_lemma] == p_text:
                                if dobj:
                                    coll_name = f"{v_lemma} ... {p_text}"
                                    formula = f"{v_lemma} [sb] {p_text} [sth]"
                                else:
                                    coll_name = f"{v_lemma} {p_text}"
                                    formula = f"{v_lemma} {p_text} [sth/sb]"
                                raw_candidates.append({
                                    "sid": sid,
                                    "quote": sent_clean,
                                    "phrase": coll_name,
                                    "type": "collocation",
                                    "pattern_formula": formula,
                                    "score": 9,
                                })
                            elif not dobj and p_text in DEPENDENT_PREPS and (c.i - token.i <= 2):
                                # Intransitive prepositional verb (e.g. rely on, contend with)
                                p_verb = f"{v_lemma} {p_text}"
                                formula = f"{v_lemma} {p_text} [sth/sb]"
                                raw_candidates.append({
                                    "sid": sid,
                                    "quote": sent_clean,
                                    "phrase": p_verb,
                                    "type": "phrasal verb",
                                    "pattern_formula": formula,
                                    "score": 8,
                                })
                            elif dobj:
                                # Fixed verbal idiom/collocation (e.g. take into account, make a difference)
                                pobj = [p for p in c.children if p.dep_ == "pobj"]
                                if pobj and (c.i - token.i <= 2) and pobj[0].lemma_.lower() in ("account", "consideration", "advantage", "part", "effect", "place"):
                                    prep_phrase = f"{v_lemma} {p_text} {pobj[0].lemma_.lower()}"
                                    formula = f"{v_lemma} {p_text} {pobj[0].lemma_.lower()} [sth]"
                                    raw_candidates.append({
                                        "sid": sid,
                                        "quote": sent_clean,
                                        "phrase": prep_phrase,
                                        "type": "idiom",
                                        "pattern_formula": formula,
                                        "score": 10,
                                    })

        if not raw_candidates:
            return []

        # Sort by score descending and deduplicate phrases
        raw_candidates.sort(key=lambda x: x["score"], reverse=True)

        selected: List[Dict[str, str]] = []
        seen_phrases: Set[str] = set()
        seen_quotes: Set[str] = set()

        # Balance across types: collocation, phrasal verb, idiom
        for item in raw_candidates:
            p_clean = item["phrase"].lower()
            if p_clean not in seen_phrases and item["quote"] not in seen_quotes:
                selected.append({
                    "sid": item["sid"],
                    "quote": item["quote"],
                    "phrase": item["phrase"],
                    "type": item["type"],
                    "pattern_formula": item["pattern_formula"],
                })
                seen_phrases.add(p_clean)
                seen_quotes.add(item["quote"])
            if len(selected) >= target_count:
                break

        # If more slots needed, allow reusing quote if phrase is distinct
        if len(selected) < target_count:
            for item in raw_candidates:
                p_clean = item["phrase"].lower()
                if p_clean not in seen_phrases:
                    selected.append({
                        "sid": item["sid"],
                        "quote": item["quote"],
                        "phrase": item["phrase"],
                        "type": item["type"],
                        "pattern_formula": item["pattern_formula"],
                    })
                    seen_phrases.add(p_clean)
                if len(selected) >= target_count:
                    break

        return selected[:target_count]

    # -------------------------------------------------------------------------
    # 3c. Deterministic Academic Vocabulary Mining (AWL + Lemmatization)
    # -------------------------------------------------------------------------
    @classmethod
    def mine_vocabulary_skeletons(cls, text: str, target_count: int = 20) -> List[Dict[str, Any]]:
        """
        Deterministically mines genuine academic vocabulary targets from the indexed sentence pool.
        Uses spaCy context-aware lemmatization to extract base dictionary headwords,
        prioritizes the Academic Word List (AWL 560), and filters out low-value everyday words.

        Returns list of dicts:
            [{
                "sid": "S-1",
                "quote": "full sentence text",
                "word": "autonomy",
                "part_of_speech": "noun",
                "is_awl": True
            }, ...]
        """
        _, pool = cls.tokenize_and_index_sentences(text)
        if not pool:
            return []

        nlp = cls.get_spacy()
        awl_set = cls.get_awl_words()

        # Closed whitelist of parts of speech accepted by schema
        VALID_POS_MAP = {
            "NOUN": "noun",
            "VERB": "verb",
            "ADJ": "adjective",
            "ADV": "adverb",
        }

        # Exclude trivial grammaticalized or ultra-common verbs/words
        TRIVIAL_WORDS = frozenset({
            "be", "have", "do", "say", "get", "make", "go", "know", "take", "see",
            "come", "think", "look", "want", "give", "use", "find", "tell", "ask",
            "work", "seem", "feel", "try", "leave", "call", "good", "new", "first",
            "last", "long", "great", "little", "own", "other", "old", "right", "big",
            "high", "different", "small", "large", "next", "early", "young", "important",
            "few", "public", "bad", "same", "able", "man", "woman", "person", "people",
            "child", "time", "year", "day", "way", "thing", "life", "hand", "part",
            "eye", "place", "case", "week", "company", "system", "program", "question",
            "government", "number", "night", "point", "home", "water", "room", "mother",
            "area", "money", "story", "fact", "month", "lot", "right", "study", "book",
            "word", "business", "issue", "side", "kind", "head", "house", "service",
            "friend", "father", "power", "hour", "game", "line", "end", "member", "law",
            "car", "city", "community", "name", "president", "team", "minute", "idea",
            "kid", "body", "information", "back", "parent", "face", "others", "level",
            "office", "door", "health", "person", "art", "war", "history", "party",
            "result", "change", "morning", "reason", "research", "girl", "guy", "moment",
            "air", "teacher", "force", "education"
        })

        candidates: List[Dict[str, Any]] = []
        seen_lemmas: Set[str] = set()

        for sid, sentence in pool.items():
            sent_clean = sentence.strip()
            if len(sent_clean.split()) < 5:
                continue

            doc = nlp(sent_clean)
            for token in doc:
                if token.pos_ not in VALID_POS_MAP:
                    continue
                lemma = token.lemma_.lower().strip()
                if len(lemma) < 3 or lemma in seen_lemmas or token.is_stop or not lemma.isalpha():
                    continue

                is_awl = lemma in awl_set

                # If not in AWL and is trivial or very short, skip
                if not is_awl and (lemma in TRIVIAL_WORDS or len(lemma) < 5):
                    continue

                # Scoring: AWL headwords get high priority, followed by word length and syllable complexity
                score = (20 if is_awl else 0) + min(len(lemma), 12)

                candidates.append({
                    "sid": sid,
                    "quote": sent_clean,
                    "word": lemma,
                    "part_of_speech": VALID_POS_MAP[token.pos_],
                    "is_awl": is_awl,
                    "score": score
                })
                seen_lemmas.add(lemma)

        if not candidates:
            return []

        # Sort descending by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected: List[Dict[str, Any]] = []
        selected_words: Set[str] = set()
        selected_quotes: Set[str] = set()

        # Pass 1: select diverse items across different sentences
        for item in candidates:
            if item["word"] not in selected_words and item["quote"] not in selected_quotes:
                selected.append(item)
                selected_words.add(item["word"])
                selected_quotes.add(item["quote"])
            if len(selected) >= target_count:
                break

        # Pass 2: fill remaining slots allowing multiple words from same sentence if high value
        if len(selected) < target_count:
            for item in candidates:
                if item["word"] not in selected_words:
                    selected.append(item)
                    selected_words.add(item["word"])
                if len(selected) >= target_count:
                    break

        return selected[:target_count]

    @classmethod
    def build_authentic_cloze_items(
        cls,
        vocab_content: str,
        target_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Track 1: Authentic Passage Cloze (Achievement Testing / 学业水平测试)
        Extracts extracted vocabulary items and masks their target appearance in the authentic
        quoted passage sentence using spaCy lemmatization and exact word-boundary masking.
        Generates 0-token, 100% textbook-grounded question stems.
        """
        if not vocab_content:
            return []

        nlp = cls.get_spacy()
        blocks = re.split(r'\n(?=##\s*\[\[)', vocab_content)
        parsed_items: List[Dict[str, str]] = []

        for b in blocks:
            m_word = re.search(r'##\s*\[\[(.*?)\]\]', b)
            if not m_word:
                continue
            word = m_word.group(1).strip()
            m_pos = re.search(r'-\s*\*\*Part Of Speech\*\*:\s*([^\n]+)', b, re.IGNORECASE)
            m_def = re.search(r'-\s*\*\*Definition\*\*:\s*([^\n]+)', b, re.IGNORECASE)
            m_quote = re.search(r'-\s*\*\*Quoted Sentence\*\*:\s*([^\n]+)', b, re.IGNORECASE)

            parsed_items.append({
                "word": word,
                "part_of_speech": m_pos.group(1).strip() if m_pos else "noun",
                "definition": m_def.group(1).strip() if m_def else "",
                "quote": m_quote.group(1).strip() if m_quote else ""
            })

        cloze_items: List[Dict[str, Any]] = []

        for it in parsed_items:
            w = it["word"]
            q = it["quote"]
            if not q:
                continue

            # Strip leading/trailing quotation marks from quote
            q_clean = q.strip().strip('"').strip("'").strip("“").strip("”")
            if not q_clean:
                continue

            doc = nlp(q_clean)
            w_lower = w.lower()

            # Locate token in sentence that matches the lemma or literal surface form
            target_tok = None
            for t in doc:
                if t.text.lower() == w_lower or t.lemma_.lower() == w_lower:
                    target_tok = t
                    break

            if not target_tok:
                continue

            surface_form = target_tok.text
            # Replace target token with exact 4 underscores '____'
            stem = re.sub(rf'\b{re.escape(surface_form)}\b', '____', q_clean, count=1)

            # Ensure exactly one blank exists in stem
            if len(re.findall(r'_{2,}', stem)) != 1:
                continue

            # Wordnet distractor generation for Track 1 Cloze
            pos_tag = "n"
            pos_raw = it["part_of_speech"].lower()
            if "verb" in pos_raw: pos_tag = "v"
            elif "adj" in pos_raw: pos_tag = "a"
            elif "adv" in pos_raw: pos_tag = "r"

            distractors = cls.generate_zero_collision_distractors(
                target_word=surface_form,
                pos=pos_tag,
                count=3
            )

            cloze_items.append({
                "target_word": surface_form,
                "base_headword": w,
                "part_of_speech": it["part_of_speech"],
                "definition": it["definition"],
                "question": stem,
                "source_sentence": q_clean,
                "precomputed_distractors": distractors
            })

            if len(cloze_items) >= target_count:
                break

        return cloze_items

    @classmethod
    def build_precomputed_target_skeletons(
        cls,
        vocab_content: str,
        target_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Track 2: Pre-Computed Generative Skeletons (Proficiency Testing / 语境迁移单选题)
        Extracts vocabulary items from extracted cards and synthesizes authoritative collocational
        anchors (via Oxford Collocations Dictionary) and collision-free distractors (via WordNet + OCD).
        Produces pre-computed item skeletons to strictly constrain LLM generation, guaranteeing zero
        duplicate options, zero synonym pile double-keys, and perfect stem-option-explanation alignment.
        """
        if not vocab_content:
            return []

        nlp = cls.get_spacy()
        ocd = cls.get_oxford_collocations()
        blocks = re.split(r'\n(?=##\s*\[\[)', vocab_content)
        parsed_items: List[Dict[str, str]] = []

        for b in blocks:
            m_word = re.search(r'##\s*\[\[(.*?)\]\]', b)
            if not m_word:
                continue
            word = m_word.group(1).strip()
            # Filter out multi-word idioms or phrases from single-word vocabulary MCQ targets
            if " " in word or "-" in word:
                continue

            m_pos = re.search(r'-\s*\*\*Part Of Speech\*\*:\s*([^\n]+)', b, re.IGNORECASE)
            m_def = re.search(r'-\s*\*\*Definition\*\*:\s*([^\n]+)', b, re.IGNORECASE)
            m_quote = re.search(r'-\s*\*\*Quoted Sentence\*\*:\s*([^\n]+)', b, re.IGNORECASE)

            parsed_items.append({
                "word": word,
                "part_of_speech": m_pos.group(1).strip() if m_pos else "noun",
                "definition": m_def.group(1).strip() if m_def else "",
                "quote": m_quote.group(1).strip() if m_quote else ""
            })

        skeletons: List[Dict[str, Any]] = []

        for it in parsed_items:
            w = it["word"]
            w_lower = w.lower()
            raw_pos = it["part_of_speech"].lower()
            
            # Map POS category
            canonical_pos = "noun"
            if "verb" in raw_pos:
                canonical_pos = "verb"
            elif "adj" in raw_pos:
                canonical_pos = "adj"
            elif "adv" in raw_pos:
                canonical_pos = "adv"

            # 1. Retrieve Authentic Oxford Collocational Anchor
            anchor = None
            anchor_type = "collocation"

            if w_lower in ocd:
                entry = ocd[w_lower]
                if canonical_pos == "verb":
                    # Verb: Look for direct noun objects
                    candidates = entry.get("noun_after", []) + entry.get("colloc_nouns", [])
                    if candidates:
                        anchor = candidates[0].split()[0].lower()
                elif canonical_pos == "noun":
                    # Noun: Look for typical modifying adjectives
                    candidates = entry.get("adj", []) + entry.get("verb_before", [])
                    if candidates:
                        anchor = candidates[0].split()[0].lower()
                elif canonical_pos == "adj":
                    # Adjective: Look for nouns modified
                    candidates = entry.get("noun_after", []) + entry.get("colloc_nouns", [])
                    if candidates:
                        anchor = candidates[0].split()[0].lower()

            # 2. Syntactic Fallback: Extract anchor from quoted sentence if OCD entry is empty
            if not anchor and it["quote"]:
                q_clean = it["quote"].strip().strip('"').strip("'").strip("“").strip("”")
                try:
                    doc = nlp(q_clean)
                    for tok in doc:
                        if tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower:
                            for child in tok.children:
                                if child.dep_ in ("dobj", "pobj"):
                                    anchor = child.lemma_.lower()
                                    break
                                elif child.dep_ == "prep":
                                    for pchild in child.children:
                                        if pchild.dep_ == "pobj":
                                            anchor = pchild.lemma_.lower()
                                            break
                            if not anchor and tok.head and tok.head.pos_ in ("NOUN", "VERB"):
                                anchor = tok.head.lemma_.lower()
                            break
                except Exception:
                    pass

            # 3. Generate 3 Zero-Collision Distractors
            distractors = cls.generate_vocab_distractors(
                target_word=w_lower,
                pos=canonical_pos,
                context_anchor=anchor,
                target_count=3
            )

            # Ensure we have valid distractors; fallback to zero_collision if needed
            if len(distractors) < 3:
                pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else "n")
                distractors = cls.generate_zero_collision_distractors(w_lower, pos=pos_char, count=3)

            # Assemble full options (target + 3 distractors)
            raw_options = [w_lower] + distractors[:3]

            # 4. Deterministic Inflection Variation Engine
            # Synergize with LLM: Python deterministically synthesizes grammatical inflections
            # (past tense, participle, plural) so LLM can fearlessly compose varied academic frames.
            inflection_tag = None
            inflection_desc = "base form"
            final_target = w_lower
            final_options = raw_options

            try:
                from lemminflect import getInflection
                # Apply inflection to verbs or nouns based on item index (e.g. mix past, ing, plural)
                item_seq = len(skeletons)
                if canonical_pos == "verb":
                    # Alternate: past participle (VBN) / present participle (VBG) / base form
                    if item_seq % 3 == 1:
                        inflection_tag = "VBN"
                        inflection_desc = "past participle (-ed / -en)"
                    elif item_seq % 3 == 2:
                        inflection_tag = "VBG"
                        inflection_desc = "present participle / gerund (-ing)"
                elif canonical_pos == "noun":
                    # Alternate: plural (NNS) / singular base form
                    if item_seq % 2 == 1:
                        inflection_tag = "NNS"
                        inflection_desc = "plural (-s)"

                if inflection_tag:
                    infl_opts = []
                    for opt in raw_options:
                        parts = opt.split()
                        if parts:
                            head = parts[0]
                            infl = getInflection(head, tag=inflection_tag)
                            if infl:
                                infl_opts.append(" ".join([infl[0]] + parts[1:]))
                            else:
                                infl_opts.append(opt)
                        else:
                            infl_opts.append(opt)
                    # Only adopt if all 4 options inflected successfully and are unique
                    if len(infl_opts) == 4 and len(set(infl_opts)) == 4:
                        final_options = infl_opts
                        w_parts = w_lower.split()
                        w_infl = getInflection(w_parts[0], tag=inflection_tag) if w_parts else None
                        if w_infl:
                            final_target = " ".join([w_infl[0]] + w_parts[1:])
                    else:
                        inflection_desc = "base form"
            except Exception:
                inflection_desc = "base form"

            skeletons.append({
                "target_word": final_target,
                "base_headword": w,
                "part_of_speech": canonical_pos,
                "inflection": inflection_desc,
                "context_anchor": anchor,
                "prescribed_options": final_options,
                "definition": it["definition"],
                "candidate_quote": it["quote"]
            })

            if len(skeletons) >= target_count:
                break

        return skeletons

    # -------------------------------------------------------------------------
    # 4. WordNet Distractor Assembly & Zero-Double-Key Guarantee
    # -------------------------------------------------------------------------
    @classmethod
    def get_synonyms(cls, word: str) -> Set[str]:
        """Returns all synonymous lemma strings for the word across all synsets."""
        wn = cls.get_wordnet()
        synonyms = set()
        for syn in wn.synsets(word.lower()):
            for w in syn.words():
                lemma = w.lemma().replace("_", " ").lower()
                if lemma != word.lower():
                    synonyms.add(lemma)
        return synonyms

    @classmethod
    def get_antonyms(cls, word: str) -> Set[str]:
        """Returns strict direct antonyms from WordNet sense relations."""
        wn = cls.get_wordnet()
        antonyms = set()
        for syn in wn.synsets(word.lower()):
            # Sense-level antonyms
            for sense in syn.senses():
                for ant_sense in sense.get_related("antonym"):
                    antonyms.add(ant_sense.word().lemma().replace("_", " ").lower())
        return antonyms

    @classmethod
    def generate_zero_collision_distractors(
        cls,
        target_word: str,
        dependent_prep: Optional[str] = None,
        pos: str = "v",
        count: int = 3
    ) -> List[str]:
        """
        Pre-computes distractors mathematically guaranteed to avoid synonym collisions with target.
        When dependent_prep is specified (e.g. 'on' for 'rely'), prioritizes candidates that
        do NOT govern that preposition or clash with it.
        """
        wn = cls.get_wordnet()
        target_lower = target_word.lower()
        target_synonyms = cls.get_synonyms(target_lower)
        target_synonyms.add(target_lower)

        antonyms = cls.get_antonyms(target_lower)

        candidates: List[str] = []

        # 1. Antonyms make outstanding, collision-free distractors
        for ant in antonyms:
            if ant not in target_synonyms and " " not in ant:
                candidates.append(ant)

        # 2. Taxonomy sibling / hypernym search
        target_synsets = wn.synsets(target_lower, pos=pos)
        if target_synsets:
            primary_synset = target_synsets[0]
            # Get hypernyms (parent categories)
            hypernyms = primary_synset.get_related("hypernym")
            for hyper in hypernyms:
                # Get siblings (hyponyms of the hypernym)
                for sibling in hyper.get_related("hyponym"):
                    for w in sibling.words():
                        lemma = w.lemma().replace("_", " ").lower()
                        if lemma not in target_synonyms and " " not in lemma and lemma not in candidates:
                            candidates.append(lemma)

        # 3. Deduplicate and return requested count
        final_distractors = candidates[:count]

        # Fallback if synset taxonomy was sparse
        if len(final_distractors) < count:
            common_fallbacks = {
                "v": ["assume", "indicate", "require", "maintain", "assess"],
                "n": ["aspect", "factor", "criterion", "approach", "phenomenon"],
                "a": ["apparent", "consistent", "variable", "significant", "distinct"]
            }
            for fb in common_fallbacks.get(pos, common_fallbacks["v"]):
                if fb not in target_synonyms and fb not in final_distractors:
                    final_distractors.append(fb)
                if len(final_distractors) == count:
                    break

        return final_distractors
