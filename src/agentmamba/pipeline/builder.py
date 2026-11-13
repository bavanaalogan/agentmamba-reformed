"""Wiring helpers shared by scripts, API, dashboard and tests."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from agentmamba.config import resolve_path
from agentmamba.gate.features import FeatureStore
from agentmamba.gate.predictor import GatePredictor, build_scorer
from agentmamba.gate.train import GateArtifact
from agentmamba.memory.episodic_memory import EpisodicMemory
from agentmamba.memory.relationship_store import InMemoryRelationshipGraph
from agentmamba.memory.schema import Event, Question
from agentmamba.models.answerer import FirstCandidateAnswerer, MambaAnswerer, extract_candidates
from agentmamba.pipeline.agent import AgentMamba
from agentmamba.pipeline.ingestion import FeatureProvider, IngestionPipeline
from agentmamba.retrieval.answerability import assess
from agentmamba.retrieval.controller import Controller
from agentmamba.retrieval.query_processor import IntentClassifier, QueryProcessor
from agentmamba.verification.evidence_checker import EvidenceVerifier


def gate_path(cfg: dict, use_mamba: bool | None = None, kind: str | None = None) -> Path:
    um = cfg["gate"]["use_mamba_features"] if use_mamba is None else use_mamba
    kind = kind or cfg["gate"]["model"]
    return resolve_path(cfg, "gate.artifact_dir") / f"gate_{kind}_{'mamba' if um else 'nomamba'}.pkl"


def build_answerer(cfg: dict, backbone=None):
    if cfg["models"]["answerer"] == "first_candidate":
        return FirstCandidateAnswerer()
    if backbone is None:
        from agentmamba.models.mamba_backbone import MambaBackbone

        backbone = MambaBackbone(cfg["models"]["mamba_id"], cfg["models"]["mamba_threads"],
                                 cfg["models"]["mamba_max_tokens"],
                                 resolve_path(cfg, "paths.artifacts_dir") / "models" / "mamba_score_cache.sqlite3")
    return MambaAnswerer(backbone)


def train_intent(questions: list[Question], question_types: dict[str, str]) -> IntentClassifier:
    qs = [q for q in questions if q.question_id in question_types]
    return IntentClassifier().fit([q.text for q in qs], [question_types[q.question_id] for q in qs])


def initial_features(qp: QueryProcessor, answerer, question: Question, ev_index: dict[str, Event],
                     verifier: EvidenceVerifier | None = None):
    """Run the initial (context-only) stage and return (Assessment | None)."""
    verifier = verifier or EvidenceVerifier()
    parsed = qp.parse(question.text)
    if parsed.spec is None:
        return None
    ctx = [ev_index[i] for i in question.initial_context_ids]
    ranked = answerer.rank(question.text, ctx, extract_candidates(parsed.spec, ctx)).candidates
    ver = verifier.verify(parsed, ranked, {e.event_id: e for e in ctx}, InMemoryRelationshipGraph(ctx), store=None)
    a = assess(parsed, ranked, ver)
    return None if (a.ambiguous or a.missing_anchor) else a


def train_controller(qp, answerer, train_qs, train_anns, val_qs, val_anns, ev_index) -> Controller:
    def xy(qs, anns):
        X, y = [], []
        for q in qs:
            a = initial_features(qp, answerer, q, ev_index)
            if a is not None:
                X.append(a.vector())
                y.append(1 if anns[q.question_id].initial_sufficient else 0)
        return np.asarray(X), np.asarray(y)

    Xtr, ytr = xy(train_qs, train_anns)
    Xva, yva = xy(val_qs, val_anns)
    ctl = Controller().fit(Xtr, ytr)
    ctl.tune_threshold(Xva, yva)
    ctl.meta.update({"n_train": int(len(ytr)), "n_val": int(len(yva)), "train_pos": int(ytr.sum()),
                     "val_pos": int(yva.sum()), "answerer": getattr(answerer, "name", "?")})
    return ctl


def build_memory(cfg: dict, storage_dir: Path, events: list[Event], *, policy: str, budget_fraction: float,
                 embedder, feature_store: FeatureStore | None, backbone=None, gate: GateArtifact | None = None,
                 fresh: bool = True, run_tag: str = ""):
    """Create a memory directory and ingest ``events`` under a selection policy and budget
    (budget = fraction of the total candidate bytes; 1.0 == store-all)."""
    storage_dir = Path(storage_dir)
    if fresh and storage_dir.exists():
        shutil.rmtree(storage_dir)
    mem = EpisodicMemory(storage_dir, embedder)
    predictor = GatePredictor(gate) if policy == "learned" else None
    scorer = build_scorer(policy, predictor=predictor, seed=cfg["seed"])
    needs_mamba = bool(predictor and predictor.needs_mamba)
    fp = FeatureProvider(embedder, feature_store, backbone, cfg["models"]["mamba_feature_max_tokens"])
    pipe = IngestionPipeline(mem, fp, scorer, policy, needs_mamba, cfg["gate"]["novelty_window"], run_tag)
    total = sum(e.size_bytes() for e in events)
    budget = None if policy == "store_all" or budget_fraction >= 1.0 else int(total * budget_fraction)
    rep = pipe.ingest(events, budget)
    mem.store.set_meta("memory_config", {"policy": policy, "budget_fraction": budget_fraction, "budget_bytes": budget,
                                         "candidate_bytes": total, "gate_version": gate.meta["gate_version"] if gate else None,
                                         "run_tag": run_tag})
    return mem, rep


def build_agent(cfg: dict, memory, answerer=None, versions: dict | None = None) -> AgentMamba:
    paths = AgentMamba.artifact_paths(cfg)
    qp = QueryProcessor(IntentClassifier.load(paths["intent"]))
    ctl = Controller.load(paths["controller"])
    return AgentMamba(cfg, memory, answerer or build_answerer(cfg), qp, ctl,
                      versions={"controller_threshold": ctl.threshold, **(versions or {})})
