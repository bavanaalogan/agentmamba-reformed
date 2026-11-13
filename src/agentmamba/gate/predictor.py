"""Write-time scoring with a trained gate (or an explicitly selected baseline policy).

``build_scorer`` is the single place that maps a configured policy to a scoring function.
For policy "learned" a trained artifact MUST exist; a missing artifact raises instead of
silently falling back to the heuristic.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from agentmamba.gate.budget import heuristic_score, random_score
from agentmamba.gate.features import stream_novelty
from agentmamba.gate.train import GateArtifact
from agentmamba.memory.schema import Event


class GatePredictor:
    def __init__(self, artifact: GateArtifact):
        self.artifact = artifact

    @property
    def needs_mamba(self) -> bool:
        return self.artifact.featurizer.use_mamba

    def score(self, events: list[Event], emb: np.ndarray, mamba: np.ndarray | None,
              novelty: np.ndarray | None = None) -> np.ndarray:
        fz = self.artifact.featurizer
        nov = novelty if novelty is not None else stream_novelty(emb, fz.window)
        X = fz.transform(events, emb, mamba if fz.use_mamba else None, nov)
        return self.artifact.predict_proba(X)


def build_scorer(policy: str, *, predictor: GatePredictor | None = None, seed: int = 13
                 ) -> Callable[..., np.ndarray]:
    """Every scorer has the signature (events, emb, mamba, novelty) -> scores."""
    if policy == "learned":
        if predictor is None:
            raise RuntimeError("memory.policy=learned requires a trained gate artifact")
        return predictor.score
    if policy == "heuristic":
        return lambda evs, emb, mamba, novelty=None: np.array([heuristic_score(e) for e in evs])
    if policy == "random":
        return lambda evs, emb, mamba, novelty=None: np.array([random_score(e, seed) for e in evs])
    if policy == "store_all":
        return lambda evs, emb, mamba, novelty=None: np.ones(len(evs))
    raise ValueError(f"unknown policy {policy!r}")
