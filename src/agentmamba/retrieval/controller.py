"""Conditional-retrieval controller: RETURN_INITIAL | RETRIEVE | ABSTAIN_OR_CLARIFY.

Decision policy (documented):
  1. ABSTAIN_OR_CLARIFY if the question is classified ambiguous or names no identifier of the
     kind its intent requires (asking the engineer beats guessing an asset).
  2. In conditional mode, RETRIEVE if initial evidence is insufficient, partial, or conflicting.
  3. Otherwise p = P(initial context sufficient | assessment features) from a logistic
     regression fitted on TRAINING questions (label: annotation ``initial_sufficient``).
  4. RETURN_INITIAL if p >= tau, else RETRIEVE. tau is chosen on the VALIDATION split to
     maximize balanced accuracy of the retrieve/return decision.
Modes "always" and "never" bypass step 2-3 (baselines B / C).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentmamba.retrieval.answerability import FEATURES, Assessment

CONTROLLER_VERSION = "controller-v1"
RETURN, RETRIEVE, CLARIFY = "RETURN_INITIAL", "RETRIEVE", "ABSTAIN_OR_CLARIFY"


@dataclass
class Decision:
    action: str
    p_sufficient: float | None
    threshold: float | None
    reasons: list[str]

    def to_dict(self) -> dict:
        return {"action": self.action, "p_sufficient": None if self.p_sufficient is None else round(self.p_sufficient, 4),
                "threshold": self.threshold, "reasons": self.reasons}


class Controller:
    def __init__(self, model=None, threshold: float = 0.5, meta: dict | None = None):
        self.model = model
        self.threshold = threshold
        self.meta = meta or {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Controller":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        self.model.fit(X, y)
        coefs = self.model[-1].coef_[0]
        self.meta["coefficients"] = {k: round(float(c), 4) for k, c in zip(FEATURES, coefs)}
        return self

    def tune_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.model.predict_proba(X)[:, 1]
        best = (-1.0, 0.5)
        for t in np.linspace(0.05, 0.95, 91):
            pred = p >= t
            tpr = (pred & (y == 1)).sum() / max(1, (y == 1).sum())
            tnr = (~pred & (y == 0)).sum() / max(1, (y == 0).sum())
            best = max(best, ((tpr + tnr) / 2, float(round(t, 2))))
        self.threshold = best[1]
        self.meta["val_balanced_accuracy"] = round(float(best[0]), 4)
        self.meta["threshold"] = self.threshold
        return self.threshold

    def decide(self, a: Assessment, mode: str = "conditional") -> Decision:
        if a.ambiguous or a.missing_anchor:
            why = "question is ambiguous" if a.ambiguous else "question names no identifier of the required kind"
            return Decision(CLARIFY, None, None, [why])
        if mode == "always":
            return Decision(RETRIEVE, None, None, ["mode=always_retrieve"])
        if mode == "never":
            return Decision(RETURN, None, None, ["mode=no_retrieval"])
        if a.evidence_status == "supported":
            return Decision(RETURN, None, None, ["initial context contains verified evidence for every answer part"])
        if a.evidence_status in {"insufficient", "partial", "conflict"}:
            reasons = a.reasons + [f"initial evidence is {a.evidence_status}; retrieve before answering"]
            return Decision(RETRIEVE, None, None, reasons)
        if self.model is None:
            raise RuntimeError("controller not trained; run scripts/train_gate.py --controller")
        p = float(self.model.predict_proba(np.asarray([a.vector()]))[0, 1])
        if p >= self.threshold:
            return Decision(RETURN, p, self.threshold, [f"initial evidence judged sufficient (p={p:.2f} >= {self.threshold})"])
        return Decision(RETRIEVE, p, self.threshold, a.reasons + [f"p_sufficient={p:.2f} < {self.threshold}"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"version": CONTROLLER_VERSION, "model": self.model, "threshold": self.threshold,
                         "meta": self.meta, "features": FEATURES}, fh)

    @classmethod
    def load(cls, path: Path) -> "Controller":
        if not Path(path).exists():
            raise FileNotFoundError(f"controller artifact missing at {path}; run scripts/train_gate.py --controller")
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if blob.get("version") != CONTROLLER_VERSION or tuple(blob.get("features", ())) != FEATURES:
            raise ValueError("controller artifact incompatible; retrain")
        return cls(blob["model"], blob["threshold"], blob["meta"])
