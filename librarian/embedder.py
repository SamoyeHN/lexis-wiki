"""
Embedder Module (Deprecated / Decoupled).

The project architecture has permanently migrated from heavy embedding models (all-MiniLM-L6-v2)
to Pure Symbolic Candidate Selection (POS Hard Gating + Syntactic Valency Gate + Pedagogical Lexicon Priority)
paired with Neuro-Symbolic Local LLM Reranking.

This stub is retained solely for backward compatibility with legacy scripts, returning None.
"""

from typing import Optional

def get_embedder() -> Optional[object]:
    """Deprecated: Embedding models have been permanently removed in favor of pure symbolic extraction."""
    return None
