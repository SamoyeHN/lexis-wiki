"""
Glossary Ledger: Cumulative Lexical Ledger (CLL) for Multi-Book / Multi-Series Curriculum.
Maintains deterministic word timelines, detects first-exposure units,
prevents cross-unit vocabulary duplication, and stops quiz item / example collisions.
"""

from __future__ import annotations
import json
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import config


def normalize_series_slug(series_name: str) -> str:
    """Creates a deterministic filesystem slug for a textbook series."""
    slug = re.sub(r"[^\w\-]+", "_", series_name.strip()).strip("_")
    return slug or "General_Series"


def hash_example_text(text: str) -> str:
    """Computes a normalized SHA256 (first 12 chars) of an example sentence for collision detection."""
    norm = text.strip().lower()
    norm = re.sub(r"[\-_]+", " ", norm)
    norm = re.sub(r"[^\w\s]", "", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


def _normalize_series_text(s: str) -> str:
    """Normalizes textbook series titles for robust matching across editions and spacing."""
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[\s_\-]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip().lower()


def normalize_pos(raw_pos: str) -> str:
    """Normalizes part of speech strings (e.g. 'ad.' -> 'adv', 'a.' -> 'adj', 'noun' -> 'noun')."""
    p = raw_pos.strip().lower().rstrip(".")
    mapping = {
        "n": "noun",
        "noun": "noun",
        "v": "verb",
        "vt": "verb",
        "vi": "verb",
        "verb": "verb",
        "a": "adj",
        "adj": "adj",
        "adjective": "adj",
        "ad": "adv",
        "adv": "adv",
        "adverb": "adv",
        "prep": "prep",
        "preposition": "prep",
        "conj": "conj",
        "conjunction": "conj",
        "pron": "pron",
        "pronoun": "pron",
        "num": "num",
        "numeral": "num",
        "int": "interj",
        "interj": "interj",
        "interjection": "interj",
        "art": "art",
        "article": "art",
    }
    return mapping.get(p, p)


class GlossaryLedger:
    """
    Cumulative Lexical Ledger (CLL) for a specific (series, book) combination.
    Persisted deterministically at wiki/_glossary/<series_slug>/Book_<book>_glossary.json.
    """

    def __init__(self, series: str, book: int | str, project_root: Path | None = None):
        self.series = series.strip()
        self.book = str(book).strip()
        self.project_root = Path(project_root) if project_root else Path(config.project_root)
        self.slug = normalize_series_slug(self.series)
        self.ledger_dir = self.project_root / "wiki" / "_glossary" / self.slug
        self.ledger_path = self.ledger_dir / f"Book_{self.book}_glossary.json"

        # Data store
        self.data: dict[str, Any] = {
            "series": self.series,
            "book": self.book,
            "last_updated": None,
            "total_entries": 0,
            "entries": {},
            "unit_index": {},
            "used_example_hashes": {},
        }
        self.load()

    def load(self) -> None:
        """Loads ledger from disk if it exists, self-healing legacy schema and normalizing POS keys."""
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    disk_data = json.load(f)
                    if isinstance(disk_data, dict):
                        # Migrate & deduplicate entries by normalized lemma#pos
                        raw_entries = disk_data.get("entries", {})
                        migrated_entries: dict[str, dict[str, Any]] = {}
                        key_mapping: dict[str, str] = {}  # old_key -> canonical_key

                        for old_key, entry in raw_entries.items():
                            entry.pop("ipa", None)
                            entry.pop("chinese", None)
                            entry.pop("pos_raw", None)
                            lemma = str(entry.get("lemma", "")).strip().lower()
                            pos = normalize_pos(entry.get("pos", "noun"))
                            entry["pos"] = pos
                            entry["lemma"] = lemma
                            canonical_key = f"{lemma}#{pos}"
                            key_mapping[old_key] = canonical_key

                            if "acquisition_tier" not in entry:
                                entry["acquisition_tier"] = "target_new"

                            if canonical_key not in migrated_entries:
                                migrated_entries[canonical_key] = entry
                            else:
                                target = migrated_entries[canonical_key]
                                if not target.get("definition") and entry.get("definition"):
                                    target["definition"] = entry.get("definition")
                                if not target.get("cefr") and entry.get("cefr"):
                                    target["cefr"] = entry.get("cefr")
                                if not target.get("display_word") and entry.get("display_word"):
                                    target["display_word"] = entry.get("display_word")
                                existing_occ_units = {o.get("unit") for o in target.get("occurrences", [])}
                                for occ in entry.get("occurrences", []):
                                    if occ.get("unit") not in existing_occ_units:
                                        target.setdefault("occurrences", []).append(occ)
                                        existing_occ_units.add(occ.get("unit"))
                                for ex in entry.get("used_examples", []):
                                    if ex not in target.setdefault("used_examples", []):
                                        target["used_examples"].append(ex)

                        disk_data["entries"] = migrated_entries

                        # Remap unit_index keys to canonical keys
                        raw_u_idx = disk_data.get("unit_index", {})
                        new_u_idx: dict[str, list[str]] = {}
                        for u_name, k_list in raw_u_idx.items():
                            canonical_list = []
                            for k in k_list:
                                ck = key_mapping.get(k, k)
                                if ck in migrated_entries and ck not in canonical_list:
                                    canonical_list.append(ck)
                            new_u_idx[u_name] = canonical_list
                        disk_data["unit_index"] = new_u_idx

                        self.data.update(disk_data)
            except Exception:
                pass

        # If prior books exist, ensure all entries are classified
        if self.get_prior_books():
            needs_classification = any(
                "prior_exposure" not in e for e in self.data.get("entries", {}).values()
            )
            if needs_classification:
                self.classify_all_entries()

    def save(self) -> Path:
        """Persists ledger to disk with sorted keys for git stability."""
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.data["total_entries"] = len(self.data.get("entries", {}))
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        return self.ledger_path

    # =========================================================================
    # Pathway B: Official Syllabus Ingest (PDF Top-Down)
    # =========================================================================
    def import_pdf_glossary(self, pdf_path: Path | str) -> dict[str, Any]:
        """
        Parses official textbook Appendix Glossary PDF (e.g. 《新编实用英语》综合教程 1 附录词表).
        Extracts: word, ipa, pos, chinese, and unit tag (e.g. U1P1, U4S).
        Aligns unit codes to wiki unit slugs where available.
        """
        import pymupdf

        p = Path(pdf_path)
        if not p.exists():
            raise FileNotFoundError(f"Glossary PDF not found at: {p}")

        doc = pymupdf.open(str(p))
        entries_added = 0
        entries_updated = 0

        # Scan all pages
        for page in doc:
            page_text = page.get_text()
            lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
            i = 0
            while i < len(lines):
                ln = lines[i]
                # Pattern: Word /IPA/ POS Meaning
                m = re.match(r"^([A-Za-z\s\-\'\.]+?)\s+/(.*?)/\s+([a-z\.\s\/]+?)\s+(.+)$", ln)
                if m:
                    word = m.group(1).strip()
                    ipa = m.group(2).strip()
                    pos_raw = m.group(3).strip()
                    meaning = m.group(4).strip().rstrip("\x08").strip()

                    unit_tag = ""
                    if i + 1 < len(lines) and re.match(r"^U\d+[A-Z0-9]+$", lines[i + 1]):
                        unit_tag = lines[i + 1].strip()
                        i += 1

                    norm_lemma = word.lower()
                    pos_norm = normalize_pos(pos_raw)
                    entry_key = f"{norm_lemma}#{pos_norm}"

                    # Map unit tag to standard unit reference
                    resolved_unit = self._resolve_unit_code_to_unit_slug(unit_tag)

                    entries = self.data.setdefault("entries", {})
                    if entry_key not in entries:
                        entries[entry_key] = {
                            "lemma": norm_lemma,
                            "display_word": word,
                            "pos": pos_norm,
                            "unit_code": unit_tag,
                            "first_unit": resolved_unit or unit_tag,
                            "source_type": "official_pdf",
                            "occurrences": [],
                            "used_examples": [],
                        }
                        entries_added += 1
                    else:
                        existing = entries[entry_key]
                        existing["display_word"] = word
                        existing["unit_code"] = unit_tag
                        existing.pop("ipa", None)
                        existing.pop("chinese", None)
                        existing.pop("pos_raw", None)
                        if not existing.get("first_unit") and (resolved_unit or unit_tag):
                            existing["first_unit"] = resolved_unit or unit_tag
                        entries_updated += 1

                    # Add to unit_index
                    u_key = resolved_unit or unit_tag or "General"
                    u_idx = self.data.setdefault("unit_index", {})
                    u_list = u_idx.setdefault(u_key, [])
                    if entry_key not in u_list:
                        u_list.append(entry_key)

                i += 1

        doc.close()
        self.classify_all_entries()
        self.save()
        return {
            "status": "success",
            "pdf_path": str(p),
            "entries_added": entries_added,
            "entries_updated": entries_updated,
            "total_entries": len(self.data["entries"]),
        }

    def _resolve_unit_code_to_unit_slug(self, unit_tag: str) -> str | None:
        """
        Resolves unit codes like 'U1P1', 'U1P2', 'U2P1' to exact wiki unit slugs.
        E.g.: 'U1P1' + Book 1 -> 'Book_1_Unit_1_Passage_A'
        """
        if not unit_tag:
            return None

        m = re.match(r"^U(\d+)([A-Za-z0-9]+)$", unit_tag)
        if not m:
            return None

        u_num, section_code = int(m.group(1)), m.group(2).upper()
        wiki_dir = self.project_root / "wiki"
        if not wiki_dir.exists():
            return None

        # Determine passage/section letter
        section_char = "A" if section_code in ("P1", "A") else ("B" if section_code in ("P2", "B") else None)

        # Search existing units for matching book, unit, section
        for u_path in sorted(wiki_dir.iterdir()):
            if not u_path.is_dir() or u_path.name.startswith(("_", ".")):
                continue
            name = u_path.name
            # Check Book and Unit
            if f"Book_{self.book}" in name and f"Unit_{u_num}" in name:
                if section_char and (f"Passage_{section_char}" in name or f"Section_{section_char}" in name):
                    return name
                elif not section_char:
                    return name

        # Fallback synthesized slug
        if section_char:
            return f"Book_{self.book}_Unit_{u_num}_Passage_{section_char}"
        return f"Book_{self.book}_Unit_{u_num}"

    # =========================================================================
    # Pathway A: Bottom-Up Auto-Harvest from extractions
    # =========================================================================
    def sync_bottom_up(self) -> dict[str, Any]:
        """
        Scans all units in wiki/ matching this series and book in pedagogical unit sequence.
        Harvests headwords, senses, definitions, examples, and records first-exposure unit.
        """
        wiki_dir = self.project_root / "wiki"
        if not wiki_dir.exists():
            return {"status": "error", "message": "wiki directory not found"}

        # Find matching units
        matching_units: list[Path] = []
        for u_dir in wiki_dir.iterdir():
            if not u_dir.is_dir() or u_dir.name.startswith(("_", ".")):
                continue
            # Check source file frontmatter for series & book match
            src_file = u_dir / "sources" / f"{u_dir.name}.md"
            if not src_file.exists():
                for f in (u_dir / "sources").glob("*.md"):
                    src_file = f
                    break
            if not src_file.exists():
                continue

            try:
                head = src_file.read_text(encoding="utf-8")[:1000]
                m_ser = re.search(r'^series:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                m_bk = re.search(r'^book:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                u_ser = m_ser.group(1).strip() if m_ser else ""
                u_bk = m_bk.group(1).strip() if m_bk else ""

                if self._series_matches(u_ser) and str(u_bk) == str(self.book):
                    matching_units.append(u_dir)
            except Exception:
                pass

        # Sort units naturally: Unit 1 Passage A, Unit 1 Passage B, Unit 2...
        def _sort_key(p: Path) -> tuple[int, int, str]:
            m_u = re.search(r"Unit_(\d+)", p.name)
            u_num = int(m_u.group(1)) if m_u else 999
            is_p2 = 2 if ("Passage_B" in p.name or "Section_B" in p.name) else 1
            return (int(self.book) if self.book.isdigit() else 0, u_num, p.name)

        matching_units.sort(key=_sort_key)

        harvested_count = 0
        entries = self.data.setdefault("entries", {})
        unit_idx = self.data.setdefault("unit_index", {})
        used_hashes = self.data.setdefault("used_example_hashes", {})

        for u_dir in matching_units:
            unit_name = u_dir.name
            unit_vocab_file = u_dir / "extractions" / f"{unit_name}_vocabulary.md"
            if not unit_vocab_file.exists():
                continue

            try:
                vocab_text = unit_vocab_file.read_text(encoding="utf-8")
                unit_items = self._parse_vocabulary_markdown(vocab_text)

                for item in unit_items:
                    lemma = item["lemma"]
                    pos = normalize_pos(item.get("pos", "n"))
                    entry_key = f"{lemma}#{pos}"

                    # Example hash tracking
                    ex_text = item.get("example", "")
                    ex_hash = hash_example_text(ex_text) if ex_text else None
                    if ex_hash and ex_text:
                        used_hashes[ex_hash] = {
                            "unit": unit_name,
                            "lemma": lemma,
                            "example": ex_text,
                        }

                    if entry_key not in entries:
                        entries[entry_key] = {
                            "lemma": lemma,
                            "display_word": item.get("display_word", lemma),
                            "pos": pos,
                            "cefr": item.get("cefr"),
                            "definition": item.get("definition"),
                            "first_unit": unit_name,
                            "source_type": "bottom_up_extracted",
                            "occurrences": [
                                {
                                    "unit": unit_name,
                                    "role": "target_new",
                                    "example": ex_text,
                                    "example_hash": ex_hash,
                                }
                            ],
                            "used_examples": [ex_text] if ex_text else [],
                        }
                        harvested_count += 1
                    else:
                        ent = entries[entry_key]
                        if not ent.get("cefr") and item.get("cefr"):
                            ent["cefr"] = item.get("cefr")
                        if not ent.get("definition") and item.get("definition"):
                            ent["definition"] = item.get("definition")

                        # Record occurrence
                        occ_units = [o["unit"] for o in ent.get("occurrences", [])]
                        if unit_name not in occ_units:
                            ent.setdefault("occurrences", []).append({
                                "unit": unit_name,
                                "role": "recycled_review",
                                "example": ex_text,
                                "example_hash": ex_hash,
                            })
                        if ex_text and ex_text not in ent.setdefault("used_examples", []):
                            ent["used_examples"].append(ex_text)

                    # Update unit_index
                    u_list = unit_idx.setdefault(unit_name, [])
                    if entry_key not in u_list:
                        u_list.append(entry_key)

            except Exception:
                pass

        self.classify_all_entries()
        self.save()
        return {
            "status": "success",
            "series": self.series,
            "book": self.book,
            "matching_units_scanned": len(matching_units),
            "harvested_count": harvested_count,
            "total_entries": len(entries),
        }

    def _series_matches(self, unit_series: str) -> bool:
        """Determines if a unit's series matches this ledger's series."""
        if not unit_series:
            return False
        s1 = _normalize_series_text(self.series)
        s2 = _normalize_series_text(unit_series)
        return s1 in s2 or s2 in s1

    def _parse_vocabulary_markdown(self, text: str) -> list[dict[str, Any]]:
        """Extracts items from an extractions/<Unit>_vocabulary.md file."""
        items = []
        # Pattern: ## [[lemma]] pos CEFR `[constraints]`
        blocks = re.split(r"\n(?=##\s+\[\[)", text)
        for blk in blocks:
            m_header = re.search(r"^##\s+\[\[(.*?)\]\]\s+([a-z]+)\s+([A-C][1-2])?", blk, re.MULTILINE | re.IGNORECASE)
            if not m_header:
                continue

            lemma = m_header.group(1).strip().lower()
            pos = m_header.group(2).strip().lower()
            cefr = m_header.group(3).strip().upper() if m_header.group(3) else None

            # Definition
            m_def = re.search(r"-\s+\*\*Definition\*\*:\s*(.*?)$", blk, re.MULTILINE)
            definition = m_def.group(1).strip() if m_def else ""

            # Example Usage
            m_ex = re.search(r"-\s+\*\*Example Usage\*\*:\s*(.*?)$", blk, re.MULTILINE)
            example = m_ex.group(1).strip() if m_ex else ""

            items.append({
                "lemma": lemma,
                "display_word": m_header.group(1).strip(),
                "pos": pos,
                "cefr": cefr,
                "definition": definition,
                "example": example,
            })
        return items

    # =========================================================================
    # Collision Guards (for Quiz Generation & Unit Verification)
    # =========================================================================
    def is_headword_prior_target(self, headword: str, current_unit: str) -> tuple[bool, str | None]:
        """
        Checks whether headword has already been introduced as a target word in an earlier unit.
        Returns (is_prior, first_unit_name).
        """
        norm = headword.strip().lower()
        entries = self.data.get("entries", {})

        for key, entry in entries.items():
            if entry.get("lemma") == norm or key.split("#")[0] == norm:
                first_unit = entry.get("first_unit")
                if first_unit and first_unit != current_unit:
                    return True, first_unit
        return False, None

    def is_example_used(self, example_text: str) -> bool:
        """Checks if this specific example has already been used in any quiz or unit extraction."""
        if not example_text or not example_text.strip():
            return False
        h = hash_example_text(example_text)
        return h in self.data.get("used_example_hashes", {})

    def record_quiz_usage(self, headword: str, unit: str, example_text: str = "") -> None:
        """Records that an item and example were used in a generated quiz."""
        norm = headword.strip().lower()
        entries = self.data.setdefault("entries", {})
        for key, entry in entries.items():
            if entry.get("lemma") == norm or key.split("#")[0] == norm:
                quizzes = entry.setdefault("used_in_quizzes", [])
                if unit not in quizzes:
                    quizzes.append(unit)
                if example_text and example_text not in entry.setdefault("used_examples", []):
                    entry["used_examples"].append(example_text)

        if example_text:
            h = hash_example_text(example_text)
            self.data.setdefault("used_example_hashes", {})[h] = {
                "unit": unit,
                "lemma": norm,
                "example": example_text,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        self.save()

    # =========================================================================
    # Cross-Book SLA Spiral Progression & Quota Governor
    # =========================================================================
    def get_prior_books(self) -> list[str]:
        """
        Finds all prior books for this series in chronological order.
        For example, if self.book == '2', finds ['1'].
        If self.book == '4', finds ['1', '2', '3'].
        """
        if not self.ledger_dir.exists():
            return []

        current_num = int(self.book) if self.book.isdigit() else None
        if current_num is None or current_num <= 1:
            return []

        prior_books = []
        for b in range(1, current_num):
            p = self.ledger_dir / f"Book_{b}_glossary.json"
            if p.exists():
                prior_books.append(str(b))
        return prior_books

    def load_prior_ledgers(self) -> list[GlossaryLedger]:
        """Loads GlossaryLedger instances for all prior books in this series."""
        ledgers = []
        for b_str in self.get_prior_books():
            ledgers.append(GlossaryLedger(series=self.series, book=b_str, project_root=self.project_root))
        return ledgers

    def classify_cross_book_role(
        self,
        lemma: str,
        pos: str,
        definition: str = "",
        prior_ledgers: list[GlossaryLedger] | None = None,
    ) -> dict[str, Any]:
        """
        Determines SLA spiral progression acquisition tier across curriculum books:
        - 'target_new': Lemma has never appeared in any prior book (Core: ~80-85%).
        - 'recycled_sense_shift': Lemma appeared in an earlier book, but POS differs (e.g. schedule n in B1 vs v in B2)
                                 or definition/sense shifts significantly (Auxiliary: ~10-15%).
        - 'recycled_spaced_review': Lemma appeared in an earlier book with identical POS/sense (Auxiliary: ~0-5%).
        """
        norm_lemma = lemma.strip().lower()
        norm_pos = normalize_pos(pos)

        if prior_ledgers is None:
            prior_ledgers = self.load_prior_ledgers()

        if not prior_ledgers:
            return {
                "acquisition_tier": "target_new",
                "prior_exposure": None,
                "tier_label": "New Target Word",
            }

        found_matches: list[dict[str, Any]] = []

        for p_ledger in prior_ledgers:
            p_entries = p_ledger.data.get("entries", {})
            for p_key, p_entry in p_entries.items():
                p_lemma = p_entry.get("lemma", "").lower()
                if p_lemma == norm_lemma:
                    found_matches.append({
                        "book": p_ledger.book,
                        "unit": p_entry.get("first_unit") or p_entry.get("unit_code"),
                        "pos": normalize_pos(p_entry.get("pos", "")),
                        "definition": p_entry.get("definition", ""),
                    })

        if not found_matches:
            return {
                "acquisition_tier": "target_new",
                "prior_exposure": None,
                "tier_label": "New Target Word",
            }

        exact_pos_match = next((m for m in found_matches if m["pos"] == norm_pos), None)

        if exact_pos_match is None:
            earliest = found_matches[0]
            return {
                "acquisition_tier": "recycled_sense_shift",
                "prior_exposure": {
                    "book": earliest["book"],
                    "unit": earliest["unit"],
                    "pos": earliest["pos"],
                    "note": f"Recategorized from {earliest['pos']} (Book {earliest['book']}) to {norm_pos}",
                },
                "tier_label": f"Sense Shift (Book {earliest['book']} {earliest['pos']})",
            }

        return {
            "acquisition_tier": "recycled_spaced_review",
            "prior_exposure": {
                "book": exact_pos_match["book"],
                "unit": exact_pos_match["unit"],
                "pos": exact_pos_match["pos"],
                "note": f"Spaced review of Book {exact_pos_match['book']} target",
            },
            "tier_label": f"Spaced Review (Book {exact_pos_match['book']})",
        }

    def classify_all_entries(self) -> dict[str, int]:
        """
        Classifies all entries in this ledger according to cross-book ancestry.
        Returns the counts per acquisition tier.
        """
        prior_ledgers = self.load_prior_ledgers()
        tier_counts = {"target_new": 0, "recycled_sense_shift": 0, "recycled_spaced_review": 0}
        entries = self.data.get("entries", {})

        for key, entry in entries.items():
            lemma = entry.get("lemma", "")
            pos = entry.get("pos", "")
            definition = entry.get("definition", "")
            res = self.classify_cross_book_role(lemma, pos, definition, prior_ledgers=prior_ledgers)
            entry["acquisition_tier"] = res["acquisition_tier"]
            entry["prior_exposure"] = res["prior_exposure"]
            entry["tier_label"] = res["tier_label"]
            tier_counts[res["acquisition_tier"]] += 1

        return tier_counts

    def allocate_quiz_lexical_quota(
        self,
        candidate_entries: list[dict[str, Any]],
        total_count: int = 10,
        new_target_ratio: float = 0.8,
        sense_shift_ratio: float = 0.15,
        spaced_review_ratio: float = 0.05,
    ) -> dict[str, Any]:
        """
        Allocates items for a quiz (e.g. 10 items) adhering to the SLA Spiral Progression Model:
        - Primary (80-85%): New target words (target_new)
        - Secondary (10-15%): Sense shift / recategorized words (recycled_sense_shift)
        - Tertiary (0-5%): Spaced retrieval of prior book words (recycled_spaced_review)
        Gracefully backfills with target_new if auxiliary tiers have insufficient candidates.
        """
        if not candidate_entries:
            return {
                "selected_items": [],
                "quota_breakdown": {"target_new": 0, "recycled_sense_shift": 0, "recycled_spaced_review": 0},
                "total_allocated": 0,
            }

        buckets: dict[str, list[dict[str, Any]]] = {
            "target_new": [],
            "recycled_sense_shift": [],
            "recycled_spaced_review": [],
        }
        for item in candidate_entries:
            tier = item.get("acquisition_tier", "target_new")
            if tier not in buckets:
                tier = "target_new"
            buckets[tier].append(item)

        target_new_quota = max(1, round(total_count * new_target_ratio))
        sense_shift_quota = round(total_count * sense_shift_ratio)
        spaced_review_quota = total_count - target_new_quota - sense_shift_quota

        selected: list[dict[str, Any]] = []
        counts = {"target_new": 0, "recycled_sense_shift": 0, "recycled_spaced_review": 0}

        # 1. Select sense shifts up to quota
        shifts = buckets["recycled_sense_shift"][:sense_shift_quota]
        selected.extend(shifts)
        counts["recycled_sense_shift"] = len(shifts)

        # 2. Select spaced reviews up to quota
        reviews = buckets["recycled_spaced_review"][:spaced_review_quota]
        selected.extend(reviews)
        counts["recycled_spaced_review"] = len(reviews)

        # 3. Fill target_new up to remaining needed slots
        needed_new = total_count - len(selected)
        news = buckets["target_new"][:needed_new]
        selected.extend(news)
        counts["target_new"] = len(news)

        # 4. If still under total_count (e.g. target_new ran out), fallback to any remaining candidates
        if len(selected) < total_count:
            all_remaining = (
                buckets["target_new"][len(news):] +
                buckets["recycled_sense_shift"][len(shifts):] +
                buckets["recycled_spaced_review"][len(reviews):]
            )
            for item in all_remaining:
                if len(selected) >= total_count:
                    break
                selected.append(item)
                tier = item.get("acquisition_tier", "target_new")
                counts[tier] = counts.get(tier, 0) + 1

        return {
            "selected_items": selected,
            "quota_breakdown": counts,
            "total_allocated": len(selected),
        }

    # =========================================================================
    # Summary & Reporting
    # =========================================================================
    def get_summary(self) -> dict[str, Any]:
        """Returns statistical overview of the book's vocabulary ledger."""
        entries = self.data.get("entries", {})
        cefr_counts: dict[str, int] = {}
        unit_counts: dict[str, int] = {}
        tier_counts: dict[str, int] = {
            "target_new": 0,
            "recycled_sense_shift": 0,
            "recycled_spaced_review": 0,
        }

        for e in entries.values():
            lvl = e.get("cefr") or "Unassigned"
            cefr_counts[lvl] = cefr_counts.get(lvl, 0) + 1

            u = e.get("first_unit") or e.get("unit_code") or "Unknown"
            unit_counts[u] = unit_counts.get(u, 0) + 1

            tier = e.get("acquisition_tier") or "target_new"
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            "series": self.series,
            "book": self.book,
            "total_entries": len(entries),
            "last_updated": self.data.get("last_updated"),
            "cefr_breakdown": cefr_counts,
            "unit_distribution": dict(sorted(unit_counts.items())),
            "acquisition_tier_breakdown": tier_counts,
        }

    summary = get_summary


def get_or_create_ledger(series: str, book: int | str, project_root: Path | None = None) -> GlossaryLedger:
    """Factory function to get or create a GlossaryLedger instance."""
    return GlossaryLedger(series=series, book=book, project_root=project_root)
