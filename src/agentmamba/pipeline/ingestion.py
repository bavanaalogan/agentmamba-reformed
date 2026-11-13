"""Ingestion and memory writing:
records -> normalization (loaders) -> stable IDs / relationships -> features -> gate score
-> budgeted WRITE/SKIP (+ evictions) -> SQLite + FAISS, with every decision logged."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from agentmamba.gate.budget import BudgetedSelector
from agentmamba.gate.features import FeatureStore, NoveltyTracker
from agentmamba.memory.lifecycle import update as lifecycle_update
from agentmamba.memory.schema import Event
from agentmamba.reproducibility import utc_now


class FeatureProvider:
    """Embeddings (always) and Mamba record features (only when the gate needs them),
    served from the content-hash cache and computed on a miss."""

    def __init__(self, embedder, feature_store: FeatureStore | None, backbone=None, mamba_max_tokens: int = 96):
        self.embedder = embedder
        self.fs = feature_store
        self.backbone = backbone
        self.mamba_max_tokens = mamba_max_tokens

    def embeddings(self, events: list[Event]) -> np.ndarray:
        if self.fs is None:
            return self.embedder.encode([e.text for e in events])
        return self.fs.get("emb", self.embedder.model_id, events, self.embedder.encode)

    def mamba(self, events: list[Event]) -> np.ndarray:
        if self.backbone is None:
            raise RuntimeError("Mamba features requested but no backbone configured")
        mt = self.mamba_max_tokens
        compute = lambda texts: self.backbone.record_features(texts, max_tokens=mt)  # noqa: E731
        if self.fs is None:
            return compute([e.text for e in events])
        return self.fs.get("mamba", self.backbone.model_id + f"@{mt}", events, compute)


@dataclass
class IngestReport:
    offered: int = 0
    written: int = 0
    skipped: int = 0
    evicted: int = 0
    duplicates: int = 0
    updated: int = 0
    budget_bytes: int | None = None
    used_bytes: int = 0
    policy: str = ""
    timings: dict = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)

    def to_dict(self, max_decisions: int = 200) -> dict:
        d = dict(self.__dict__)
        d["decisions"] = self.decisions[:max_decisions]
        return d


class IngestionPipeline:
    def __init__(self, memory, features: FeatureProvider, scorer, policy: str, needs_mamba: bool,
                 novelty_window: int = 256, run_tag: str = ""):
        self.memory = memory
        self.features = features
        self.scorer = scorer
        self.policy = policy
        self.needs_mamba = needs_mamba
        self.novelty = NoveltyTracker(novelty_window)
        self.run_tag = run_tag

    def _selector(self, budget_bytes: int | None) -> BudgetedSelector:
        sel = BudgetedSelector(budget_bytes)
        rows = self.memory.store.conn.execute(
            "SELECT event_id, size_bytes, COALESCE(gate_score, 0) FROM events WHERE status='active'").fetchall()
        for eid, size, score in rows:   # existing memory counts against the budget
            sel.offer(eid, size, score)
        return sel

    def ingest(self, events: list[Event], budget_bytes: int | None = None) -> IngestReport:
        rep = IngestReport(policy=self.policy, budget_bytes=budget_bytes)
        t0 = time.perf_counter()
        store = self.memory.store
        fresh: list[Event] = []
        seen = set()
        for ev in sorted(events, key=lambda e: (e.timestamp or "9999", e.event_id)):
            if ev.event_id in seen:
                rep.duplicates += 1
                continue
            seen.add(ev.event_id)
            row = store.row(ev.event_id)
            if row is None:
                fresh.append(ev)
            elif row["content_hash"] == ev.content_hash:
                rep.duplicates += 1
            else:
                lifecycle_update(self.memory, ev)
                rep.updated += 1
        rep.offered = len(fresh)
        if not fresh:
            rep.timings["total_s"] = time.perf_counter() - t0
            return rep
        for ev in fresh:
            ev.ingested_at = ev.ingested_at or utc_now()
        t = time.perf_counter()
        emb = self.features.embeddings(fresh)
        rep.timings["embedding_s"] = time.perf_counter() - t
        mamba = None
        if self.needs_mamba:
            t = time.perf_counter()
            mamba = self.features.mamba(fresh)
            rep.timings["mamba_features_s"] = time.perf_counter() - t
        nov = np.array([self.novelty.update(v) for v in emb], dtype="float32").reshape(-1, 2)
        t = time.perf_counter()
        scores = np.asarray(self.scorer(fresh, emb, mamba, nov), dtype="float64")
        rep.timings["gate_s"] = time.perf_counter() - t
        sel = self._selector(budget_bytes)
        pending: dict[str, int] = {}
        log_rows, stored_evictions = [], []
        for i, (ev, sc) in enumerate(zip(fresh, scores)):
            decision, evicted = sel.offer(ev.event_id, ev.size_bytes(), float(sc))
            if decision == "WRITE":
                pending[ev.event_id] = i
            for eid in evicted:
                if eid in pending:
                    del pending[eid]
                else:
                    stored_evictions.append(eid)
                log_rows.append((eid, "EVICT", None, self.policy, budget_bytes, sel.used,
                                 f"lowest score under budget when {ev.event_id} arrived", self.run_tag, utc_now()))
                rep.evicted += 1
            reason = ("within budget" if decision == "WRITE" else "score below retained minimum under budget")
            log_rows.append((ev.event_id, decision, float(sc), self.policy, budget_bytes, sel.used, reason,
                             self.run_tag, utc_now()))
            if len(rep.decisions) < 2000:
                rep.decisions.append({"event_id": ev.event_id, "event_type": ev.event_type, "decision": decision,
                                      "score": round(float(sc), 5), "size_bytes": ev.size_bytes()})
        # A record written and later evicted within this batch is logged WRITE then EVICT.
        t = time.perf_counter()
        if stored_evictions:
            self.memory.delete(stored_evictions, reason="evicted", save=False)
        idx = sorted(pending.values())
        self.memory.write([fresh[i] for i in idx], emb[idx], [float(scores[i]) for i in idx], save=True)
        store.log_decisions(log_rows)
        rep.timings["write_s"] = time.perf_counter() - t
        rep.written = len(idx)
        rep.skipped = rep.offered - rep.written
        rep.used_bytes = sel.used
        rep.timings["total_s"] = time.perf_counter() - t0
        store.audit("ingest", {k: v for k, v in rep.to_dict(0).items() if k != "decisions"})
        return rep
