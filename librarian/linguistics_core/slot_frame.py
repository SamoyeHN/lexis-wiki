"""Structured SlotFrame domain dataclass for Lexis Wiki.

Provides a structured contract for cloze slot frames while maintaining
100% backward compatibility with legacy string-based contracts and unit tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SlotFrame:
    """Structured representation of a cloze slot frame.

    Attributes:
        left_anchor: Preceding syntactic anchor (e.g. 'their', 'a major').
        aux_lock: Mandatory auxiliary or copula lock (e.g. 'is', 'have been').
        right_anchor: Trailing complement or clause boundary (e.g. 'into the market').
        prep: Bound preposition directly following the slot (e.g. 'into', 'for', 'to').
        ontology_hint: Semantic selection constraint (e.g. '[process/activity, e.g. production / distribution]').
        constraints: Lexico-syntactic constraints (e.g. ['to-infinitive', 'mandative-subjunctive', 'plural-only']).
        raw_symbolic: Pristine symbolic string representation if parsed from legacy source.
    """

    left_anchor: str = ""
    aux_lock: str = ""
    right_anchor: str = ""
    prep: str = ""
    ontology_hint: str = ""
    constraints: List[str] = field(default_factory=list)
    raw_symbolic: str = ""

    def to_symbolic_str(self) -> str:
        """Returns the canonical symbolic string representation (e.g. '... their ____ achievement')."""
        if self.raw_symbolic:
            return self.raw_symbolic

        parts: List[str] = []
        if self.left_anchor:
            parts.append(self.left_anchor.strip())
        if self.aux_lock:
            parts.append(f"[{self.aux_lock.strip()}]")
        parts.append("____")
        if self.prep:
            parts.append(self.prep.strip())
        if self.ontology_hint:
            parts.append(self.ontology_hint.strip())
        if self.right_anchor:
            parts.append(self.right_anchor.strip())

        joined = " ".join(p for p in parts if p)
        if not joined.startswith("...") and not joined.startswith("The "):
            joined = f"... {joined}"
        return joined

    def to_llm_surface(self, example_txt: str = "", skeleton: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
        """De-metasyntaxifies symbolic frame into natural-English prompt surface plus prompt rules.

        Replaces regex post-processing in _render_llm_surface_frame with direct structured rendering.
        """
        raw = self.to_symbolic_str()
        if not raw:
            return "... ____ ...", []

        surface = raw.strip()
        notes: List[str] = []

        # 1. Expand ontology hints `[category, e.g. sample1 / sample2]` -> concrete sample1 + separate rule
        def _repl_ontology(m: re.Match) -> str:
            cat = m.group(1).strip()
            examples_raw = m.group(2).strip()
            first_ex = re.split(r"\s*/\s*", examples_raw)[0].strip()
            if first_ex:
                if cat.lower() == "body part":
                    notes.append(
                        f"Body-part object: if the subject is a person, precede '{first_ex}' with a subject-matching "
                        f"possessive ('his/her/their/my'); if the subject is non-human or general, use a natural "
                        f"modifier or bare plural (e.g. 'blood {first_ex}')."
                    )
                    return first_ex
                if cat.lower() in ("person/group", "person", "someone"):
                    concrete_ex = [
                        ex.strip()
                        for ex in re.split(r"\s*/\s*", examples_raw)
                        if ex.strip() and ex.strip().lower() not in ("someone", "somebody", "person", "people")
                    ]
                    ex_hint = (
                        f" (e.g. '{concrete_ex[0]}', 'employees', 'the audience' — avoid generic 'someone')"
                        if concrete_ex
                        else " (e.g. 'the audience', 'employees', 'a colleague' — avoid generic 'someone')"
                    )
                    notes.append(f"Object slot requires a concrete person or group noun phrase{ex_hint}.")
                    return "..."
                prefix = "" if cat.lower().startswith("concrete") else "concrete "
                notes.append(f"Use a {prefix}{cat} noun (e.g. '{first_ex}' or similar).")
                return first_ex
            return "the issue"

        surface = re.sub(
            r"\[([^,\]]+),\s*e\.g\.\s*([^\]]+)\]",
            _repl_ontology,
            surface,
            flags=re.IGNORECASE,
        )

        # 2. Replace `to do [sth]` / `to do something` / `to [verb]` with a concrete infinitive or rule
        if re.search(r"\bto\s+(?:do\s+(?:\[sth\]|something)|\[verb\])", surface, flags=re.IGNORECASE):
            inf_v = ""
            if example_txt:
                tw_low = str((skeleton or {}).get("target_word") or "").strip().lower()
                bw_low = str((skeleton or {}).get("base_headword") or tw_low).strip().lower()
                prescribed = {str(opt).strip().lower() for opt in (skeleton or {}).get("prescribed_options") or []}
                disallowed = {tw_low, bw_low, "do", "be", "have", "get", "the", "a"} | prescribed
                for m_match in re.finditer(r"\bto\s+([a-z]{3,})\b", example_txt.lower()):
                    cand = m_match.group(1)
                    if (
                        cand not in disallowed
                        and not (bw_low and cand.startswith(bw_low))
                        and cand not in prescribed
                    ):
                        inf_v = cand
                        break
            if inf_v:
                surface = re.sub(r"\bto\s+(?:do\s+(?:\[sth\]|something)|\[verb\])", f"to {inf_v}", surface, flags=re.IGNORECASE)
            else:
                surface = re.sub(r"\bto\s+(?:do\s+(?:\[sth\]|something)|\[verb\])", "to", surface, flags=re.IGNORECASE)
                notes.append("Follow 'to' with a specific action verb (never write the literal phrase 'to do something').")

        # 2b. Replace `[doing sth]` / `doing something` with an authentic gerund or rule
        if re.search(r"\[doing\s+sth\]|\bdoing\s+something\b", surface, flags=re.IGNORECASE):
            gerund_v = ""
            corpus_sources = []
            if example_txt:
                corpus_sources.append(example_txt.lower())
            if skeleton:
                for q_field in ("candidate_quote", "licensed_quote", "quote"):
                    q_val = str(skeleton.get(q_field) or "").strip().lower()
                    if q_val:
                        corpus_sources.append(q_val)
            tw_low = str((skeleton or {}).get("target_word") or "").strip().lower()
            bw_low = str((skeleton or {}).get("base_headword") or tw_low).strip().lower()
            prescribed = {str(opt).strip().lower() for opt in (skeleton or {}).get("prescribed_options") or []}
            disallowed = {tw_low, bw_low, "doing", "being", "having", "going"} | prescribed
            for c_src in corpus_sources:
                for m_match in re.finditer(r"\b([a-z]{3,}ing)\b", c_src):
                    cand = m_match.group(1)
                    if (
                        cand not in disallowed
                        and not (bw_low and cand.startswith(bw_low))
                        and cand not in prescribed
                    ):
                        gerund_v = cand
                        break
                if gerund_v:
                    break
            if gerund_v:
                surface = re.sub(r"\[doing\s+sth\]|\bdoing\s+something\b", gerund_v, surface, flags=re.IGNORECASE)
            else:
                surface = re.sub(r"\[doing\s+sth\]|\bdoing\s+something\b", "[action in -ing form, e.g. making changes]", surface, flags=re.IGNORECASE)
                notes.append("Follow '____' directly with an action verb in -ing form (e.g. 'making changes' or 'taking risks' — never write literal 'doing something').")

        # 3. Replace mandative subjunctive `that [sb] (should) [V-base]` with an authentic model
        if re.search(r"that\s+\[sb\]\s*\(should\)\s*\[V-base\]", surface, flags=re.IGNORECASE):
            subj_model = ""
            corpus_sources = []
            if example_txt:
                corpus_sources.append(example_txt)
            if skeleton:
                for q_field in ("candidate_quote", "licensed_quote", "quote"):
                    q_val = str(skeleton.get(q_field) or "").strip()
                    if q_val:
                        corpus_sources.append(q_val)
            for c_src in corpus_sources:
                m_c = re.search(r"\bthat\s+((?:the\s+|a\s+|an\s+|[a-z]+\s+)[a-z]+\s+(?:be\b|should\s+[a-z]+|[a-z]+))\b", c_src, re.IGNORECASE)
                if m_c:
                    subj_model = f"that {m_c.group(1).strip()}"
                    break
            if subj_model:
                surface = re.sub(
                    r"that\s+\[sb\]\s*\(should\)\s*\[V-base\]",
                    subj_model,
                    surface,
                    flags=re.IGNORECASE,
                )
            else:
                surface = re.sub(
                    r"that\s+\[sb\]\s*\(should\)\s*\[V-base\]",
                    "that the committee review",
                    surface,
                    flags=re.IGNORECASE,
                )
            notes.append("Use a subjunctive base-form verb after 'that' (never write '(should)' or brackets).")

        # 4. Resolve remaining metasyntax brackets and number-neutral slash copulas
        def _repl_poss_placeholder(m: re.Match) -> str:
            head_n = m.group(1).strip()
            notes.append(
                f"Possessive slot: precede '{head_n}' with a context-matching possessive "
                f"(e.g. 'the manager's {head_n}', 'his {head_n}', or bare '{head_n}' — never write literal 'someone's')."
            )
            return head_n

        surface = re.sub(
            r"\b(?:someone's|somebody's|sb's|one's)\s+([a-zA-Z]+)\b",
            _repl_poss_placeholder,
            surface,
            flags=re.IGNORECASE,
        )
        if re.search(r"\[(?:sb|someone|person/group)\]", surface, flags=re.IGNORECASE):
            surface = re.sub(r"\[(?:sb|someone|person/group)\]", "...", surface, flags=re.IGNORECASE)
            if not any("concrete person or group" in n for n in notes):
                notes.append(
                    "Object slot requires a concrete person or group noun phrase "
                    "(e.g. 'the audience', 'employees', 'a colleague' — avoid generic 'someone')."
                )
        sth_model = ""
        if example_txt:
            m_np = re.search(r"\b(?:the|a|an|his|her|their|our)\s+([a-z]{3,})\b", example_txt.lower())
            if m_np and m_np.group(1) not in ("issue", "project", "way", "fact", "thing"):
                sth_model = f"the {m_np.group(1)}"
        surface = re.sub(r"\[sth\]", sth_model or "the situation", surface, flags=re.IGNORECASE)
        surface = re.sub(r"\[something\]", "something", surface, flags=re.IGNORECASE)
        surface = re.sub(r"\[(?:sb/sth|sth/sb)\]", sth_model or "the matter", surface, flags=re.IGNORECASE)
        surface = re.sub(r"\[V-base\]", "take action", surface, flags=re.IGNORECASE)
        surface = re.sub(r"\[noun\]", "system", surface, flags=re.IGNORECASE)
        surface = re.sub(
            r"\b([a-zA-Z]+)\s+is/are\b",
            lambda m: f"{m.group(1)} {'are' if (m.group(1).lower().endswith('s') and not m.group(1).lower().endswith(('ss', 'us', 'is'))) else 'is'}",
            surface,
            flags=re.IGNORECASE,
        )
        surface = re.sub(
            r"\b([a-zA-Z]+)\s+was/were\b",
            lambda m: f"{m.group(1)} {'were' if (m.group(1).lower().endswith('s') and not m.group(1).lower().endswith(('ss', 'us', 'is'))) else 'was'}",
            surface,
            flags=re.IGNORECASE,
        )
        surface = re.sub(r"\bis/are\b", "is", surface, flags=re.IGNORECASE)
        surface = re.sub(r"\bwas/were\b", "was", surface, flags=re.IGNORECASE)

        # 5. Lock passive auxiliary for VBN targets when example uses passive `be + VBN`
        if skeleton:
            v_tag = str(skeleton.get("verb_form_tag") or "").upper()
            tw_low = str(skeleton.get("target_word") or "").strip().lower()
            if (
                v_tag == "VBN"
                and tw_low
                and not re.search(r"\b(?:is|are|was|were|be|been|being|has|have|had|get|got)\s+_{2,}", surface, flags=re.IGNORECASE)
                and re.search(rf"\b(?:is|are|was|were|be|been|get|got)\s+(?:\w+\s+)?{re.escape(tw_low)}\b", example_txt, flags=re.IGNORECASE)
            ):
                surface = re.sub(r"____", "was ____", surface, count=1)
                notes.append("Keep a passive auxiliary ('was/were/is/are') directly before '____'.")

        # 5b. Subject-verb agreement guard when an auxiliary (`have`, `has`, `are`, `were`) is locked directly before `____`
        if re.search(r"\bhave\s+_{2,}", surface, flags=re.IGNORECASE):
            notes.append(
                "Subject-verb agreement: 'have ____' requires a plural subject (e.g. 'the pipes', 'years of neglect') "
                "or 'I/you/we/they' — never a singular noun."
            )
        elif re.search(r"\bhas\s+_{2,}", surface, flags=re.IGNORECASE):
            notes.append(
                "Subject-verb agreement: 'has ____' requires a 3rd-person singular subject (e.g. 'corrosion', 'the policy')."
            )
        elif re.search(r"\b(?:are|were)\s+_{2,}", surface, flags=re.IGNORECASE):
            notes.append(
                "Subject-verb agreement: 'are/were ____' requires a plural subject."
            )

        # 6. Universal Target Leakage Guard: Ensure target word or base word NEVER leaks into surface outside '____'
        if skeleton:
            tw_low = str(skeleton.get("target_word") or "").strip().lower()
            bw_low = str(skeleton.get("base_headword") or tw_low).strip().lower()
            for tgt in (tw_low, bw_low):
                if tgt and len(tgt) >= 3:
                    if re.search(rf"\bto\s+{re.escape(tgt)}\b", surface, flags=re.IGNORECASE):
                        surface = re.sub(rf"\bto\s+{re.escape(tgt)}\b", "to", surface, flags=re.IGNORECASE)
                        if "Follow 'to' with a specific action verb (never write the literal phrase 'to do something')." not in notes:
                            notes.append("Follow 'to' with a specific action verb (never write the literal phrase 'to do something').")
                    elif re.search(rf"\b{re.escape(tgt)}\b", surface, flags=re.IGNORECASE):
                        if "____" not in surface:
                            surface = re.sub(rf"\b{re.escape(tgt)}\b", "____", surface, count=1, flags=re.IGNORECASE)
                        surface = re.sub(rf"\b{re.escape(tgt)}\b", "", surface, flags=re.IGNORECASE).strip()

        surface = re.sub(r"(?:\.\.\.\s*){2,}", "... ", surface)
        surface = re.sub(r"\s+", " ", surface).strip()
        if not re.search(r"_{2,}", surface):
            surface = f"... {surface} ____ ..." if surface else "... ____ ..."
            surface = re.sub(r"\s+", " ", surface).strip()
        return surface, notes

    @classmethod
    def from_symbolic_str(cls, frame_str: str) -> SlotFrame:
        """Parses a legacy symbolic string into a structured SlotFrame instance."""
        if not frame_str:
            return cls(raw_symbolic="")

        raw = frame_str.strip()
        cleaned = re.sub(r"^\.\.\.\s*", "", raw)

        # Extract ontology hint if present
        ont_m = re.search(r"(\[[^,\]]+,\s*e\.g\.\s*[^\]]+\])", cleaned)
        ontology_hint = ont_m.group(1).strip() if ont_m else ""

        # Extract constraints
        constraints: List[str] = []
        if re.search(r"\bto\s+(?:do\s+(?:\[sth\]|something)|\[verb\])", cleaned, re.IGNORECASE):
            constraints.append("to-infinitive")
        if re.search(r"that\s+\[sb\]\s*\(should\)\s*\[V-base\]", cleaned, re.IGNORECASE):
            constraints.append("mandative-subjunctive")

        # Split around blank slot
        left_part = ""
        right_part = ""
        if "____" in cleaned:
            left_part, right_part = cleaned.split("____", 1)
            left_part = left_part.strip()
            right_part = right_part.strip()

        # Extract trailing bound preposition if immediately after blank
        prep = ""
        prep_m = re.match(r"^([a-z]{1,6})\b", right_part, re.IGNORECASE)
        common_preps = {
            "in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
            "into", "onto", "upon", "over", "after", "through", "against", "towards",
        }
        if prep_m and prep_m.group(1).lower() in common_preps:
            prep = prep_m.group(1).lower()
            right_part = right_part[prep_m.end():].strip()

        # Extract aux lock if bracketed in left part
        aux_lock = ""
        aux_m = re.search(r"\[(is|was|are|were|be|been|being|has|have|had)\]", left_part, re.IGNORECASE)
        if aux_m:
            aux_lock = aux_m.group(1).lower()
            left_part = re.sub(r"\[(is|was|are|were|be|been|being|has|have|had)\]", "", left_part, flags=re.IGNORECASE).strip()

        return cls(
            left_anchor=left_part,
            aux_lock=aux_lock,
            right_anchor=right_part,
            prep=prep,
            ontology_hint=ontology_hint,
            constraints=constraints,
            raw_symbolic=raw,
        )

    def __str__(self) -> str:
        return self.to_symbolic_str()

    def __repr__(self) -> str:
        return f"SlotFrame({self.to_symbolic_str()!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SlotFrame):
            return self.to_symbolic_str() == other.to_symbolic_str()
        if isinstance(other, str):
            return self.to_symbolic_str() == other
        return False

    def __hash__(self) -> int:
        return hash(self.to_symbolic_str())
