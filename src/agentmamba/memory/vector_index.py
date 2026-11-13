"""Persistent FAISS CPU index.

* ``IndexIDMap2(IndexFlatIP(dim))`` over L2-normalized vectors: inner product == cosine.
  Exact search; the corpus sizes here (<= tens of thousands) do not need approximation.
* External IDs are the SQLite ``vector_id`` values (stable integers, never reused).
* Persisted as ``<name>.faiss`` (serialized index) + ``<name>.meta.json`` (model id, dim,
  count, id checksum). ``validate`` compares the index with SQLite (source of truth).
* Metadata filtering: an ``IDSelectorBatch`` restricts search to the vector IDs that match
  SQLite filters; if the selector is unavailable we fall back to oversampled post-filtering.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim: int, model_id: str, path: str | Path | None = None):
        self.dim = dim
        self.model_id = model_id
        self.path = Path(path) if path else None
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    # ---------------------------------------------------------------- mutation
    def add(self, vectors: np.ndarray, ids: list[int]) -> None:
        if len(ids) == 0:
            return
        v = np.ascontiguousarray(vectors, dtype="float32")
        if v.ndim != 2 or v.shape[1] != self.dim:
            raise ValueError(f"expected vectors of dim {self.dim}, got {v.shape}")
        self.index.add_with_ids(v, np.asarray(ids, dtype="int64"))

    def remove(self, ids: list[int]) -> int:
        if not ids:
            return 0
        return int(self.index.remove_ids(np.asarray(ids, dtype="int64")))

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def ids(self) -> list[int]:
        if self.size == 0:
            return []
        return sorted(int(i) for i in faiss.vector_to_array(self.index.id_map))

    # ---------------------------------------------------------------- search
    def search(self, query: np.ndarray, k: int, allowed_ids: list[int] | None = None) -> list[tuple[int, float]]:
        if self.size == 0 or k <= 0:
            return []
        q = np.ascontiguousarray(query.reshape(1, -1), dtype="float32")
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            try:
                sel = faiss.IDSelectorBatch(np.asarray(allowed_ids, dtype="int64"))
                D, I = self.index.search(q, min(k, len(allowed_ids)), params=faiss.SearchParameters(sel=sel))
            except (TypeError, RuntimeError, AttributeError):
                allowed = set(allowed_ids)
                D, I = self.index.search(q, self.size)
                pairs = [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i in allowed]
                return pairs[:k]
        else:
            D, I = self.index.search(q, min(k, self.size))
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]

    # ---------------------------------------------------------------- persistence
    def _checksum(self) -> str:
        return hashlib.sha256(json.dumps(self.ids()).encode()).hexdigest()[:16]

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path or self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        buf = faiss.serialize_index(self.index)
        tmp = p.with_suffix(".faiss.tmp")
        tmp.write_bytes(buf.tobytes())
        tmp.replace(p.with_suffix(".faiss"))
        meta = {"dim": self.dim, "model_id": self.model_id, "count": self.size, "id_checksum": self._checksum(),
                "metric": "inner_product_on_l2_normalized (cosine)", "index_type": "IndexIDMap2(IndexFlatIP)"}
        p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return p.with_suffix(".faiss")

    @classmethod
    def load(cls, path: str | Path, expected_model_id: str | None = None, expected_dim: int | None = None) -> "VectorIndex":
        p = Path(path)
        fpath, mpath = p.with_suffix(".faiss"), p.with_suffix(".meta.json")
        if not fpath.exists() or not mpath.exists():
            raise FileNotFoundError(f"index artifacts missing at {fpath}; run POST /memory/rebuild-index "
                                    f"or scripts/build_memory.py --rebuild-index")
        meta = json.loads(mpath.read_text(encoding="utf-8"))
        if expected_model_id and meta["model_id"] != expected_model_id:
            raise ValueError(f"index built with {meta['model_id']!r} but {expected_model_id!r} configured; rebuild the index")
        if expected_dim and meta["dim"] != expected_dim:
            raise ValueError(f"index dim {meta['dim']} != configured {expected_dim}; rebuild the index")
        obj = cls(meta["dim"], meta["model_id"], p)
        obj.index = faiss.deserialize_index(np.frombuffer(fpath.read_bytes(), dtype="uint8"))
        if obj.size != meta["count"] or obj._checksum() != meta["id_checksum"]:
            raise ValueError("index file does not match its metadata (partial write?); rebuild the index")
        return obj

    def validate_against(self, sqlite_vector_ids: list[int]) -> dict:
        mine, theirs = set(self.ids()), set(sqlite_vector_ids)
        return {"ok": mine == theirs, "index_count": len(mine), "sqlite_count": len(theirs),
                "missing_in_index": sorted(theirs - mine)[:20], "stale_in_index": sorted(mine - theirs)[:20]}
