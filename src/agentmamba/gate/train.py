"""Train / validate / serialize the learned memory-writing gate.

* Fit: training-split events and training-split labels only.
* Model selection (regularization strength) and decision threshold: validation split only.
* The test split is never touched here.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from agentmamba.gate.features import FEATURE_SCHEMA_VERSION, GateFeaturizer
from agentmamba.reproducibility import utc_now

GATE_ARTIFACT_VERSION = "gate-artifact-v1"


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def classification_report(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    out = {"n": int(len(y)), "positives": int(y.sum()), "threshold": float(threshold),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "brier": float(brier_score_loss(y, p)), "ece": expected_calibration_error(y, p)}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
    return out


@dataclass
class GateArtifact:
    featurizer: GateFeaturizer
    model: object
    threshold: float
    meta: dict

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"artifact_version": GATE_ARTIFACT_VERSION, "feature_schema": FEATURE_SCHEMA_VERSION,
                         "featurizer": self.featurizer, "model": self.model, "threshold": self.threshold,
                         "meta": self.meta}, fh)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GateArtifact":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"trained gate artifact not found at {path}; run scripts/train_gate.py "
                                    f"(the primary pipeline never substitutes a heuristic)")
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if blob.get("artifact_version") != GATE_ARTIFACT_VERSION or blob.get("feature_schema") != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"gate artifact {path} has incompatible version {blob.get('artifact_version')}/"
                             f"{blob.get('feature_schema')}; retrain")
        return cls(blob["featurizer"], blob["model"], blob["threshold"], blob["meta"])


def _make_model(kind: str, c: float, seed: int):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0 / c, max_iter=600,
                                                         early_stopping=True, random_state=seed))


def train_gate(train_events, train_emb, train_mamba, y_train, val_events, val_emb, val_mamba, y_val, *,
               kind: str = "logreg", use_mamba: bool = True, emb_pca_dims: int = 16, mamba_pca_dims: int = 16,
               window: int = 256, seed: int = 13, c_grid=(0.01, 0.1, 1.0, 10.0), data_manifest: dict | None = None
               ) -> GateArtifact:
    fz = GateFeaturizer(use_mamba, emb_pca_dims, mamba_pca_dims, window).fit(train_events, train_emb, train_mamba)
    Xtr = fz.transform(train_events, train_emb, train_mamba if use_mamba else None)
    Xva = fz.transform(val_events, val_emb, val_mamba if use_mamba else None)
    search = []
    best = None
    for c in c_grid:
        m = _make_model(kind, c, seed).fit(Xtr, y_train)
        ap = float(average_precision_score(y_val, m.predict_proba(Xva)[:, 1]))
        search.append({"C": c, "val_average_precision": ap})
        if best is None or ap > best[0]:
            best = (ap, c, m)
    _, c_best, model = best
    p_val = model.predict_proba(Xva)[:, 1]
    # threshold maximizing validation F1 (used by threshold mode; budget mode ranks by score)
    grid = np.unique(np.round(p_val, 4))
    f1s = [(f1_score(y_val, (p_val >= t).astype(int), zero_division=0), t) for t in grid]
    thr = float(max(f1s)[1]) if f1s else 0.5
    meta = {
        "artifact_version": GATE_ARTIFACT_VERSION, "feature_schema": FEATURE_SCHEMA_VERSION, "model": kind,
        "use_mamba_features": use_mamba, "C": c_best, "c_search": search, "threshold": thr, "seed": seed,
        "n_features": len(fz.feature_names), "feature_names": fz.feature_names, "trained_at": utc_now(),
        "train": classification_report(y_train, model.predict_proba(Xtr)[:, 1], thr),
        "val": classification_report(y_val, p_val, thr), "data_manifest": data_manifest or {},
    }
    meta["gate_version"] = hashlib.sha256(json.dumps(
        {k: meta[k] for k in ("model", "use_mamba_features", "C", "seed", "feature_schema", "n_features")}
        | {"data": (data_manifest or {}).get("counts")}, sort_keys=True).encode()).hexdigest()[:12]
    return GateArtifact(fz, model, thr, meta)
