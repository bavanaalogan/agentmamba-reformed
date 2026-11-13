"""Gate features, labels, training/reload, budget enforcement and baseline policies."""
from __future__ import annotations

import numpy as np
import pytest

from agentmamba.gate.budget import BudgetedSelector, heuristic_score, random_score
from agentmamba.gate.features import FeatureStore, GateFeaturizer, NoveltyTracker, stream_novelty
from agentmamba.gate.label_builder import build_labels
from agentmamba.gate.predictor import GatePredictor, build_scorer
from agentmamba.gate.train import GateArtifact, train_gate
from agentmamba.models.embedding_model import HashingEmbedder
from tests.conftest import make_event


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    evs, y = [], []
    for i in range(n):
        useful = i % 6 == 0
        evs.append(make_event(f"E{i:03d}", "ladle_inspection" if useful else "shift_log",
                              text=("hairline crack at slag line" if useful else "routine shift, no remarks") + f" #{i}",
                              severity="info" if useful else None))
        y.append(int(useful))
    emb = HashingEmbedder().encode([e.text for e in evs])
    mamba = rng.normal(size=(n, 2 + 24)).astype("float32")
    return evs, emb, mamba, np.array(y)


def test_novelty_tracker_detects_duplicates():
    tr = NoveltyTracker(window=4)
    v = np.array([1.0, 0.0])
    assert tr.update(v)[0] == 1.0          # first record is novel
    nov, red = tr.update(v)
    assert nov == pytest.approx(0.0) and red > 0


def test_features_shape_names_and_no_label_inputs():
    evs, emb, mamba, y = _toy()
    fz = GateFeaturizer(use_mamba=True, emb_pca_dims=4, mamba_pca_dims=3).fit(evs, emb, mamba)
    X = fz.transform(evs, emb, mamba)
    assert X.shape == (len(evs), len(fz.feature_names)) and np.isfinite(X).all()
    assert "mamba_nll_mean" in fz.feature_names and "novelty" in fz.feature_names
    with pytest.raises(ValueError):
        fz.transform(evs, emb, None)
    nomamba = GateFeaturizer(use_mamba=False, emb_pca_dims=4).fit(evs, emb, None)
    assert not any(n.startswith("mamba") for n in nomamba.feature_names)
    import inspect
    assert "label" not in inspect.signature(GateFeaturizer.transform).parameters


def test_label_builder(dataset, annotations):
    ev = dataset.events_for("train")
    y = build_labels(ev, annotations["train"])
    useful = {e for a in annotations["train"].values() for e in a.required_evidence_ids}
    assert y.sum() == len([e for e in ev if e.event_id in useful]) > 0


def test_gate_train_save_reload_identical(tmp_path):
    evs, emb, mamba, y = _toy()
    art = train_gate(evs[:80], emb[:80], mamba[:80], y[:80], evs[80:], emb[80:], mamba[80:], y[80:],
                     emb_pca_dims=4, mamba_pca_dims=3, c_grid=(0.1, 1.0))
    assert art.meta["val"]["roc_auc"] > 0.9 and 0 <= art.threshold <= 1
    p = art.save(tmp_path / "gate.pkl")
    again = GateArtifact.load(p)
    s1 = GatePredictor(art).score(evs, emb, mamba)
    s2 = GatePredictor(again).score(evs, emb, mamba)
    assert np.allclose(s1, s2) and again.meta["gate_version"] == art.meta["gate_version"]


def test_missing_gate_artifact_never_falls_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        GateArtifact.load(tmp_path / "nope.pkl")
    with pytest.raises(RuntimeError):
        build_scorer("learned", predictor=None)


def test_budget_never_exceeded_and_keeps_top_scores():
    sel = BudgetedSelector(100)
    rng = np.random.default_rng(1)
    scores = {f"E{i:02d}": float(s) for i, s in enumerate(rng.random(40))}
    for eid, s in scores.items():
        sel.offer(eid, 10, s)
        assert sel.used <= 100
    top = sorted(scores, key=lambda k: -scores[k])[:10]
    assert sel.retained() == set(top)


def test_budget_streaming_decisions_and_eviction():
    sel = BudgetedSelector(20)
    assert sel.offer("A", 10, 0.5) == ("WRITE", [])
    assert sel.offer("B", 10, 0.9) == ("WRITE", [])
    assert sel.offer("C", 10, 0.1) == ("SKIP", [])          # lowest -> skipped
    assert sel.offer("D", 10, 0.7) == ("WRITE", ["A"])      # evicts A
    assert sel.offer("BIG", 50, 1.0) == ("SKIP", [])        # larger than whole budget
    assert BudgetedSelector(None).offer("X", 10**9, 0.0) == ("WRITE", [])


def test_budget_ties_are_deterministic():
    runs = []
    for _ in range(2):
        sel = BudgetedSelector(30)
        for i in range(10):
            sel.offer(f"E{i}", 10, 0.5)
        runs.append(sel.retained())
    assert runs[0] == runs[1] == {"E0", "E1", "E2"}


def test_baseline_policies():
    crit = make_event("A", "ladle_incident", severity="critical")
    log = make_event("B", "shift_log")
    assert heuristic_score(crit) > heuristic_score(log)
    assert random_score(crit, 1) == random_score(crit, 1) != random_score(crit, 2)
    assert list(build_scorer("store_all")([crit, log], None, None)) == [1.0, 1.0]


def test_feature_store_caches_by_content_hash(tmp_path):
    calls = []
    fs = FeatureStore(tmp_path)

    def compute(texts):
        calls.append(len(texts))
        return np.ones((len(texts), 3))

    evs = [make_event(f"E{i}") for i in range(5)]
    fs.get("k", "m", evs, compute, chunk=2)
    fs.get("k", "m", evs, compute)
    assert calls == [2, 2, 1]       # chunked once, then fully cached
