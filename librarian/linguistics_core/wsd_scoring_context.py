"""Data structures for neuro-symbolic WSD sense scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


@dataclass
class WsdContext:
    """Precomputed syntactic and contextual evidence for WSD sense locking."""

    entry: Dict[str, Any]
    target_pos: Optional[str]
    target_p: Optional[str]
    definition: Optional[str]
    quote: Optional[str]
    frame_evidence: Optional[str]
    stopwords: Set[str]
    q_tokens: Set[str]
    d_tokens: Set[str]
    q_low_all: str
    q_word_tokens: Set[str]
    q_seq: List[str]
    block_poses: Dict[Any, Set[str]]
    target_w_clean: str
    headword_tokens: Set[str]
    target_inflections: Set[str]
    has_pron_obj: bool
    target_tok_morph_number: Any
    target_tok_is_predicative: bool
    target_tok_is_attributive: bool
    target_v_has_dobj: bool
    target_v_is_bare_active_intrans: bool
    doc_q: Any
    meta_stopwords: Set[str]
    cue_scores: Dict[int, float]
    generic_signpost_stopwords: Set[str]
    second_order_domain_tokens: Set[str]
    discourse_tokens: Set[str]
    pv_sense_map: Dict[int, List[str]]
    senses: List[Dict[str, Any]]
    discourse_context: Optional[str] = None


@dataclass
class WsdSenseInfo:
    """Precomputed sense-level properties for WSD scoring stages."""

    s_pos: Optional[str]
    eff_pos: Optional[str]
    eff_p: Optional[str]
    s_defn: str
    s_def_tokens: Set[str]
    sp: str
    s_gram: str


__all__ = ["WsdContext", "WsdSenseInfo"]
