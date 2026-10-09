"""Lightweight sentence embedding engine using ONNX Runtime and Tokenizers.

Uses all-MiniLM-L6-v2 (ONNX format, ~80MB) for ultra-fast, local CPU embedding and
cosine similarity calculation (<3ms per inference).
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
except ImportError:
    ort = None
    Tokenizer = None

_EMBEDDER_INSTANCE = None


class MiniLMEmbedder:
    """Singleton-style embedder wrapping all-MiniLM-L6-v2 ONNX model."""

    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        if ort is None or Tokenizer is None:
            raise RuntimeError("onnxruntime and tokenizers must be installed to use MiniLMEmbedder.")

        self.model_dir = Path(model_dir) if model_dir else self._find_or_download_model()
        
        tokenizer_file = self.model_dir / "tokenizer.json"
        onnx_file = self.model_dir / "onnx" / "model.onnx"
        if not onnx_file.exists():
            onnx_file = self.model_dir / "model.onnx"

        if not tokenizer_file.exists() or not onnx_file.exists():
            raise FileNotFoundError(f"Model files not found in {self.model_dir}")

        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=128)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_file),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

    @classmethod
    def _find_or_download_model(cls) -> Path:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
        if cache_root.exists():
            snapshots = sorted(cache_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for snap in snapshots:
                if (snap / "tokenizer.json").exists() and (
                    (snap / "onnx" / "model.onnx").exists() or (snap / "model.onnx").exists()
                ):
                    return snap

        # Fallback to downloading if not cached
        from huggingface_hub import hf_hub_download
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        tok_p = hf_hub_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", filename="tokenizer.json")
        model_p = hf_hub_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", filename="onnx/model.onnx")
        return Path(tok_p).parent

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of texts into normalized embedding vectors (N, 384)."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        clean_texts = [str(t).strip() if str(t).strip() else "[EMPTY]" for t in texts]
        encoded = self.tokenizer.encode_batch(clean_texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        token_embeddings = outputs[0]  # shape: [batch, seq_len, 384]

        # Mean pooling with attention mask
        mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), token_embeddings.shape)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        return embeddings.astype(np.float32)

    def similarity(self, query: str, candidates: List[str]) -> np.ndarray:
        """Returns 1D cosine similarities between single query and N candidate strings."""
        if not candidates:
            return np.array([], dtype=np.float32)
        all_vecs = self.encode([query] + list(candidates))
        q_vec = all_vecs[0]
        c_vecs = all_vecs[1:]
        return np.dot(c_vecs, q_vec)


def get_embedder() -> Optional[MiniLMEmbedder]:
    """Retrieves or initializes the global embedder instance. Returns None if dependencies missing."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        try:
            _EMBEDDER_INSTANCE = MiniLMEmbedder()
        except Exception:
            _EMBEDDER_INSTANCE = False
    if _EMBEDDER_INSTANCE is False:
        return None
    return _EMBEDDER_INSTANCE
