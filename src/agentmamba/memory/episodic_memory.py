"""EpisodicMemory: SQLite (authoritative) + FAISS (similarity) kept in sync.

Writes go to SQLite first inside a transaction, then to FAISS; the index is saved after
each batch. On open, the index is validated against SQLite; a missing or inconsistent index
is rebuilt from the SQLite text (re-embedding), never the other way round.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from agentmamba.data.event_linker import link_event
from agentmamba.logging_utils import get_logger
from agentmamba.memory.schema import Event
from agentmamba.memory.sqlite_store import SQLiteStore
from agentmamba.memory.vector_index import VectorIndex

log = get_logger(__name__)


class EpisodicMemory:
    def __init__(self, storage_dir: str | Path, embedder, name: str = "memory", auto_recover: bool = True):
        self.dir = Path(storage_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.store = SQLiteStore(self.dir / f"{name}.sqlite3")
        self.index_path = self.dir / f"{name}"
        self.recovery_note: str | None = None
        try:
            self.index = VectorIndex.load(self.index_path, embedder.model_id, embedder.dim)
            check = self.index.validate_against([v for v, _, _ in self.store.active_vectors()])
            if not check["ok"]:
                raise ValueError(f"index inconsistent with SQLite: {check}")
        except (FileNotFoundError, ValueError) as exc:
            n_active = self.store.stats()["records"]
            self.index = VectorIndex(embedder.dim, embedder.model_id, self.index_path)
            if n_active and auto_recover:
                self.recovery_note = f"rebuilt index from SQLite: {exc}"
                log.warning(self.recovery_note)
                self.rebuild_index()
            elif n_active:
                self.recovery_note = f"index unusable ({exc}); call rebuild_index()"
            else:
                self.index.save()
        stored_model = self.store.get_meta("embedding_model")
        if stored_model is None:
            self.store.set_meta("embedding_model", embedder.model_id)

    # ---------------------------------------------------------------- write path
    def write(self, events: list[Event], vectors: np.ndarray | None = None, scores: list[float] | None = None,
              save: bool = True) -> list[int]:
        if not events:
            return []
        if vectors is None:
            vectors = self.embedder.encode([e.text for e in events])
        start = self.store.next_vector_id(len(events))
        vids = list(range(start, start + len(events)))
        with self.store.tx() as c:
            for ev, vid, sc in zip(events, vids, scores or [None] * len(events)):
                self.store.insert_event(ev, vid, sc, conn=c)
                self.store.insert_relationships(link_event(ev), conn=c)
        self.index.add(np.asarray(vectors, dtype="float32"), vids)
        if save:
            self.index.save()
        return vids

    def delete(self, event_ids: list[str], reason: str = "delete", save: bool = True) -> int:
        vids = []
        with self.store.tx() as c:
            for eid in event_ids:
                vid = self.store.hard_delete(eid, reason, conn=c)
                if vid is not None:
                    vids.append(vid)
        removed = self.index.remove(vids)
        if save:
            self.index.save()
        return removed

    def save(self) -> None:
        self.index.save()

    # ---------------------------------------------------------------- maintenance
    def rebuild_index(self, batch: int = 256) -> dict:
        rows = self.store.active_vectors()
        self.index = VectorIndex(self.embedder.dim, self.embedder.model_id, self.index_path)
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            self.index.add(self.embedder.encode([t for _, _, t in chunk]), [v for v, _, _ in chunk])
        self.index.save()
        self.store.set_meta("embedding_model", self.embedder.model_id)
        self.store.audit("rebuild_index", {"vectors": self.index.size})
        return self.validate()

    def validate(self) -> dict:
        return self.index.validate_against([v for v, _, _ in self.store.active_vectors()])

    def stats(self) -> dict:
        s = self.store.stats()
        faiss_file = self.index_path.with_suffix(".faiss")
        s.update({"vectors": self.index.size, "faiss_file_bytes": faiss_file.stat().st_size if faiss_file.exists() else 0,
                  "embedding_model": self.embedder.model_id, "index_consistency": self.validate(),
                  "recovery_note": self.recovery_note})
        return s

    def close(self) -> None:
        self.store.close()
