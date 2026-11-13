"""Text embedding models.

* ``SentenceEmbedder``: the configured model (default all-MiniLM-L6-v2, 384-d). Vectors are
  L2-normalized, so FAISS inner product == cosine similarity.
* ``HashingEmbedder``: deterministic bag-of-token hashing embedder used ONLY by fast unit
  tests as a fixture. It is never selected by the default configuration.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np


class SentenceEmbedder:
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self._model = SentenceTransformer(model_id, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        v = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
                               convert_to_numpy=True)
        return v.astype("float32")


class HashingEmbedder:
    """Test fixture: signed feature hashing of lower-cased word and identifier tokens."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model_id = f"hashing-{dim}(test-fixture)"

    def encode(self, texts: list[str], batch_size: int = 0) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9\-]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def build_embedder(cfg: dict, fixture: bool = False):
    if fixture:
        return HashingEmbedder()
    return SentenceEmbedder(cfg["models"]["embedding_id"])
