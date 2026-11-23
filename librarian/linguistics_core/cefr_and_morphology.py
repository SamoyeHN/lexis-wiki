"""
Unified CEFR, Morphology, Contextual POS, and Multi-Word Syntax facade.
"""

import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from .cefr_calibration import CefrCalibrationMixin
from .morphology_engine import MorphologyEngineMixin
from .contextual_pos import ContextualPosMixin
from .multiword_syntax import MultiwordSyntaxMixin

__all__ = [
    "CefrCalibrationMixin",
    "MorphologyEngineMixin",
    "ContextualPosMixin",
    "MultiwordSyntaxMixin",
    "CefrAndMorphologyMixin",
]


class CefrAndMorphologyMixin(
    CefrCalibrationMixin,
    MorphologyEngineMixin,
    ContextualPosMixin,
    MultiwordSyntaxMixin,
):
    """
    Unified CEFR, Morphology, and Syntactic Analysis facade.
    Composes specialized CEFR calibration, morphological inflection,
    contextual POS resolution, and multi-word syntax domain mixins
    while preserving 100% backward compatibility.
    """
    _spacy_nlp = None
    _wn = None
    _cefr_analyzer = None
    _ldoce_conn = None
    _ldoce_cache: Dict[str, Any] = {}
    _ldoce_in_progress: Set[str] = set()
    _surface_lemma_cache: Dict[str, Optional[str]] = {}
    _attested_form_cache: Dict[str, bool] = {}
    _plural_form_cache: Dict[str, str] = {}
    _inflected_forms_cache: Dict[Tuple[str, str], Set[str]] = {}
    _ldoce_tables: Optional[Set[str]] = None
    _ldoce_compounds: Optional[Dict[str, str]] = None
    _ocd_lexfile_cache: Dict[str, List[str]] = {}
    _ocd_verb_lexfile_cache: Dict[str, List[str]] = {}
    _phrase_text_cache: Dict[str, List[Tuple[str, str]]] = {}
    _phrase_evidence_cache: Dict[Tuple[str, str], List[str]] = {}
    _wsd_llm_cache: Dict[str, Tuple[int, str]] = {}
    _wsd_session_local = threading.local()
    _orthographic_pos_index: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None
    _derivational_base_cache: Dict[str, Set[str]] = {}
    _phrase_by_key_cache: Optional[Dict[str, List[Tuple[str, Optional[int], str, str]]]] = None
    _phrase_by_raw_cache: Optional[Dict[str, List[Tuple[str, Optional[int], str, str]]]] = None
    _oxford_thesaurus_cache: Dict[str, List[Dict[str, Any]]] = {}
    _compound_nouns_cache: Optional[List[Tuple[str, bool]]] = None
