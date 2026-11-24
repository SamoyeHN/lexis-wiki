"""
Skeleton Context dataclass for generative vocabulary quiz skeletons.
Encapsulates state across preprocessing, anchor resolution, distractor coordination,
and micro-task assembly.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


CXN_ANCHOR_TYPES: Set[str] = {
    "it_extrapos",
    "gerund",
    "to_inf",
    "sb_to_do",
    "shell_to_do",
    "shell_of_doing",
    "shell_that",
    "correl_pair",
    "fronted_inv",
    "npi",
    "mandative_subj",
    "lv_triad",
    "v_obj_pp",
}


@dataclass
class SkeletonContext:
    """Carries vocabulary item state across the generative skeleton pipeline stages."""

    it: Dict[str, Any]
    w: str
    raw_w_item: str
    w_lower: str
    raw_pos: str
    canonical_pos: str

    # Dictionary / Sense Grounding
    passage_cefr: Optional[str] = None
    target_entry: Optional[Dict[str, Any]] = None
    entry_base: Optional[Dict[str, Any]] = None
    locked_sense: Optional[Dict[str, Any]] = None
    t_s_idx: Optional[int] = None
    target_p_code: Optional[str] = None
    quote_provenance: str = "strong"
    quote_wordcount: int = 0

    # Construction & Discontinuous Verb Metadata
    discontinuous_prep: Optional[str] = None
    orig_disc_prep: Optional[str] = None
    v_obj_pp_tail: Optional[str] = None
    causative_extraposition_pat: Optional[str] = None
    causative_extraposition_ex: Optional[str] = None
    correlative_second_arm: Optional[str] = None
    correlative_pat: Optional[str] = None
    correlative_frame: Optional[str] = None
    lv_triad_prep: Optional[str] = None
    lv_triad_pat: Optional[str] = None
    lv_triad_frame: Optional[str] = None
    cxn_override_pat: Optional[str] = None
    cxn_override_frame: Optional[str] = None
    is_idiom_slot: bool = False
    m_caus_adj: Optional[Any] = None

    # Atomic Sense Pattern Bundle & Anchors
    initial_atomic_bundle: Optional[Dict[str, Any]] = None
    atomic_bundle: Optional[Dict[str, Any]] = None
    sense_native_prep: Optional[str] = None
    anchor: Optional[str] = None
    anchor_type: Optional[str] = None
    anchor_source: str = "curriculum_collocation"
    prep_obj: Optional[str] = None
    quote_governed_prep: Optional[str] = None
    quote_shown_prep: bool = False
    quote_shown_partitive: bool = False
    quote_is_transitive: bool = False
    quote_adj_is_attributive: bool = False
    word_subcats: Set[str] = field(default_factory=set)
    target_inflections_set: Set[str] = field(default_factory=set)
    ldoce_pattern_example: Optional[str] = None
    ldoce_cloze_frame: Optional[str] = None
    pattern_anchor: Optional[str] = None
    best_pattern: Optional[str] = None

    # Target & Syntactic Flags
    final_target: str = ""
    verb_form_tag: str = "V-base"
    verb_requires_object: Optional[bool] = None
    attested_adj_preps: List[str] = field(default_factory=list)

    # Half-slice / Quantifier Frame Metadata
    is_quantifier_frame: bool = False
    quant_head_noun: Optional[str] = None
    half_slice_target: Optional[str] = None
    half_slice_distractors: Optional[List[str]] = None
    half_slice_frame: Optional[str] = None
    half_slice_meta: Optional[Dict[str, Any]] = None
    half_slice_explanation: Optional[str] = None

    # Distractor & Option State
    is_plural_noun: bool = False
    is_pedlex_plural: bool = False
    plural_target: str = ""
    distractors: List[str] = field(default_factory=list)
    dist_meta: Dict[str, Any] = field(default_factory=dict)
    distractors3: List[str] = field(default_factory=list)
    slot_target: str = ""
    slot: int = 0
    raw_options: List[str] = field(default_factory=list)
    inflection_desc: str = "base form"
    final_options: List[str] = field(default_factory=list)
    correct_answer_index: int = 0
    dist_list: List[str] = field(default_factory=list)
    dist_str: str = ""
    skeleton_definition: str = ""
    anchor_downgrade: Optional[str] = None
    has_multiple_candidates: bool = True
