"""Gate features. Everything is computable at write time from the record itself and the
records already *considered* earlier in the stream; nothing uses questions, answers, future
outcomes or labels.

Feature groups (names are recorded in the artifact's feature schema):
  type:<event_type>, stage:<stage>      one-hot (vocabulary fitted on training events)
  severity_ord, severity_missing        recorded severity (none/info/minor/major/critical -> 0..4)
  len_chars_log, n_words_log, n_ids, n_obs, has_version, has_supersedes, n_related, digit_ratio
  idf_mean, idf_max                     token rarity; IDF fitted on TRAINING event text (no labels)
  novelty, redundancy_log               vs. the previous ``window`` considered records (cosine)
  emb_pca_<i>                           PCA of the sentence embedding (fit on training events)
  mamba_nll_mean, mamba_nll_max, mamba_pca_<i>   optional Mamba-derived features (ablation H)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import deque
from pathlib import Path

import numpy as np

from agentmamba.memory.schema import ID_FIELDS, Event

SEV_ORD = {None: 0, "info": 1, "minor": 2, "major": 3, "critical": 4}
FEATURE_SCHEMA_VERSION = "gate-features-v1"


class NoveltyTracker:
    """Online novelty: 1 - max cosine similarity to the last ``window`` records considered."""

    def __init__(self, window: int = 256, dup_threshold: float = 0.9):
        self.window = window
        self.dup_threshold = dup_threshold
        self.buf: deque = deque(maxlen=window)

    def update(self, vec: np.ndarray) -> tuple[float, float]:
        if self.buf:
            sims = np.asarray(self.buf) @ vec
            nov = 1.0 - float(sims.max())
            red = math.log1p(int((sims > self.dup_threshold).sum()))
        else:
            nov, red = 1.0, 0.0
        self.buf.append(vec)
        return nov, red


def stream_novelty(vectors: np.ndarray, window: int) -> np.ndarray:
    tr = NoveltyTracker(window)
    return np.array([tr.update(v) for v in vectors], dtype="float32").reshape(-1, 2)


class FeatureStore:
    """Disk cache of per-record embeddings / Mamba features, keyed by content hash."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, kind: str, model_id: str):
        tag = hashlib.sha1(model_id.encode()).hexdigest()[:8]
        return self.root / f"{kind}_{tag}.npy", self.root / f"{kind}_{tag}.keys.json"

    def get(self, kind: str, model_id: str, events: list[Event], compute, chunk: int = 1024) -> np.ndarray:
        """Missing rows are computed in chunks and saved after each chunk, so an interrupted
        run resumes where it stopped."""
        vpath, kpath = self._paths(kind, model_id)
        keys, mat = [], None
        if vpath.exists() and kpath.exists():
            keys = json.loads(kpath.read_text(encoding="utf-8"))
            mat = np.load(vpath)
        pos = {k: i for i, k in enumerate(keys)}
        missing = [e for e in dict((e.content_hash, e) for e in events).values() if e.content_hash not in pos]
        for start in range(0, len(missing), chunk):
            part = missing[start:start + chunk]
            new = compute([e.text for e in part]).astype("float32")
            mat = new if mat is None else np.vstack([mat, new])
            for e in part:
                pos[e.content_hash] = len(keys)
                keys.append(e.content_hash)
            np.save(vpath, mat)
            kpath.write_text(json.dumps(keys), encoding="utf-8")
        return mat[[pos[e.content_hash] for e in events]]


class GateFeaturizer:
    def __init__(self, use_mamba: bool, emb_pca_dims: int = 16, mamba_pca_dims: int = 16, window: int = 256):
        self.use_mamba = use_mamba
        self.emb_pca_dims = emb_pca_dims
        self.mamba_pca_dims = mamba_pca_dims
        self.window = window
        self.types: list[str] = []
        self.stages: list[str] = []
        self.idf: dict[str, float] = {}
        self.idf_default = 0.0
        self.emb_pca = None
        self.mamba_pca = None
        self.feature_names: list[str] = []

    @staticmethod
    def _tokens(text: str) -> list[str]:
        # identifiers are masked: an asset/heat number is not a reusable signal across plants
        text = re.sub(r"\b[A-Z0-9]+-[A-Z]+[-A-Z]*\d*\b", " <id> ", text)
        return re.findall(r"[a-z<>]+", text.lower())

    def fit(self, events: list[Event], emb: np.ndarray, mamba: np.ndarray | None) -> "GateFeaturizer":
        from sklearn.decomposition import PCA

        self.types = sorted({e.event_type for e in events})
        self.stages = sorted({e.stage for e in events})
        df: dict[str, int] = {}
        for e in events:
            for t in set(self._tokens(e.text)):
                df[t] = df.get(t, 0) + 1
        n = len(events)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.idf_default = math.log(1 + n) + 1
        self.emb_pca = PCA(self.emb_pca_dims, random_state=0).fit(emb)
        if self.use_mamba:
            if mamba is None:
                raise ValueError("use_mamba=True but no Mamba features supplied")
            self.mamba_pca = PCA(self.mamba_pca_dims, random_state=0).fit(mamba[:, 2:])
        self.feature_names = (
            [f"type:{t}" for t in self.types] + [f"stage:{s}" for s in self.stages]
            + ["severity_ord", "severity_missing", "len_chars_log", "n_words_log", "n_ids", "n_obs", "has_version",
               "has_supersedes", "n_related", "digit_ratio", "idf_mean", "idf_max", "novelty", "redundancy_log"]
            + [f"emb_pca_{i}" for i in range(self.emb_pca_dims)]
            + (["mamba_nll_mean", "mamba_nll_max"] + [f"mamba_pca_{i}" for i in range(self.mamba_pca_dims)]
               if self.use_mamba else []))
        return self

    def transform(self, events: list[Event], emb: np.ndarray, mamba: np.ndarray | None,
                  novelty: np.ndarray | None = None) -> np.ndarray:
        if novelty is None:
            novelty = stream_novelty(emb, self.window)
        rows = []
        for e in events:
            toks = self._tokens(e.text)
            idfs = [self.idf.get(t, self.idf_default) for t in toks] or [0.0]
            r = [1.0 if e.event_type == t else 0.0 for t in self.types]
            r += [1.0 if e.stage == s else 0.0 for s in self.stages]
            r += [SEV_ORD.get(e.severity, 0), 1.0 if e.severity is None else 0.0,
                  math.log1p(len(e.text)), math.log1p(len(toks)),
                  sum(1 for f in ID_FIELDS if getattr(e, f)), len(e.observations),
                  1.0 if e.version is not None else 0.0, 1.0 if e.supersedes else 0.0, len(e.related_event_ids),
                  sum(ch.isdigit() for ch in e.text) / max(1, len(e.text)),
                  float(np.mean(idfs)), float(np.max(idfs))]
            rows.append(r)
        X = np.hstack([np.asarray(rows, dtype="float32"), novelty, self.emb_pca.transform(emb)])
        if self.use_mamba:
            if mamba is None:
                raise ValueError("this gate requires Mamba features")
            X = np.hstack([X, mamba[:, :2], self.mamba_pca.transform(mamba[:, 2:])])
        return X.astype("float32")
