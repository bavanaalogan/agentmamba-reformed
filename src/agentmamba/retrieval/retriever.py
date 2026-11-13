"""Cross-stage evidence retrieval over persistent memory.

1. Embed the question; FAISS top-(k*oversample), restricted to vector IDs that pass SQLite
   metadata filters (asset / stage / heat / ... / time range / source type).
2. Resolve vector IDs to authoritative SQLite records.
3. Relationship expansion (optional): from identifiers named in the question (and in the
   permitted initial context), follow documented entity links (bounded BFS), then fetch
   records of answer-bearing types that mention reached entities. The temporal constraint of
   the parsed question is applied to these records as a metadata filter.
4. Pull in newer versions of superseded records; drop exact duplicates (content hash).
5. Rerank and keep top-k, plus the linking records that justify each kept expansion item.
6. Text channel: untyped records (documents, operational logs) whose TEXT names a question
   identifier (exact whole token), date-filtered in SQL when the question names a date.
Every item keeps its similarity, path and an explanation string.

Channels are configurable (``retrieval.channels``) for ablations: ``vector`` (FAISS),
``relationship`` (documented-link expansion), ``text`` (identifier text match). With the vector
channel off, items are ranked by lexical overlap with the question instead of cosine similarity.
For questions without a typed AnswerSpec, a stated date (and clock time) is applied as a
metadata filter to every channel: records dated otherwise are not returned.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from agentmamba.memory.relationship_store import InMemoryRelationshipGraph, SQLiteRelationshipGraph, UnionGraph
from agentmamba.memory.schema import Event
from agentmamba.retrieval.relationship_traversal import traverse
from agentmamba.retrieval.query_processor import SPECS
from agentmamba.retrieval.reranker import rerank
from agentmamba.verification.temporal_checker import check_constraint

CHANNELS = ("vector", "relationship", "text")
TYPED_TYPES = frozenset(t for sp in SPECS.values() for t in sp.event_types + sp.hypothesis_types)


@dataclass
class RetrievedItem:
    event: Event
    similarity: float
    sources: list[str]
    depth: int | None = None
    path: list[dict] = field(default_factory=list)
    score: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict:
        return {"record_id": self.event.event_id, "similarity": round(self.similarity, 4), "score": round(self.score, 4),
                "sources": self.sources, "depth": self.depth, "path": self.path, "explanation": self.explanation}


@dataclass
class RetrievalResult:
    items: list[RetrievedItem]
    relationships_traversed: list[dict]
    timings: dict
    n_vector_hits: int
    n_expansion_hits: int
    duplicates_dropped: list[str]
    n_filtered_by_time: int = 0      # candidates removed because their date/time contradicts the question
    n_text_hits: int = 0             # records found by the identifier-text channel


class Retriever:
    def __init__(self, memory, cfg: dict):
        self.memory = memory
        r = cfg["retrieval"]
        self.top_k = int(r["top_k"])
        self.oversample = int(r["oversample"])
        self.expand = r.get("expansion_depth", 0) > 0
        self.depth = int(r.get("expansion_depth", 0))
        self.max_nodes = int(r.get("max_expansion_nodes", 60))
        self.do_rerank = bool(r.get("rerank", True))
        self.w = (float(r["w_similarity"]), float(r["w_path"]), float(r["w_type"]))
        self.max_doc_chunks = int(r.get("max_doc_chunks", 20))
        self.channels = tuple(r.get("channels", CHANNELS))
        unknown = set(self.channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown retrieval channels {sorted(unknown)}; use {CHANNELS}")
        self.time_window_min = float(cfg.get("verification", {}).get("time_window_minutes", 30))

    def retrieve(self, parsed, query_text: str, initial_context: list[Event] | None = None,
                 filters: dict | None = None, top_k: int | None = None, expand: bool | None = None) -> RetrievalResult:
        k = top_k or self.top_k
        expand = self.expand if expand is None else expand
        store, index = self.memory.store, self.memory.index
        t0 = time.perf_counter()
        q = self.memory.embedder.encode([query_text])[0]
        t_embed = time.perf_counter() - t0
        allowed = store.allowed_vector_ids(filters) if filters else None
        hits = index.search(q, k * self.oversample, allowed) if "vector" in self.channels else []
        resolved = store.by_vector_ids([v for v, _ in hits])
        items: dict[str, RetrievedItem] = {}
        for vid, sim in hits:
            ev = resolved.get(vid)
            if ev is None:      # stale vector: never returned
                continue
            items[ev.event_id] = RetrievedItem(ev, sim, ["vector"], explanation=f"semantic match (cos={sim:.2f})")
        t_vec = time.perf_counter() - t0 - t_embed

        traversed: list[dict] = []
        n_exp = 0
        spec = parsed.spec if parsed is not None else None
        if expand and "relationship" in self.channels and parsed is not None and parsed.ids:
            start = {i: k_ for k_, ids in parsed.ids.items() for i in ids}
            graph = UnionGraph(SQLiteRelationshipGraph(store), InMemoryRelationshipGraph(initial_context or []))
            reached = traverse(graph, start, max_depth=self.depth, max_nodes=self.max_nodes)
            for r in reached.values():
                traversed += [p for p in r.to_dict()["path"] if p not in traversed]
            types = tuple(spec.event_types + spec.hypothesis_types) if spec else None
            for ev in store.events_mentioning(list(reached), types):
                via = next((r for e in ev.entity_ids() if (r := reached.get(e)) is not None), None)
                if via is None:
                    continue
                link_ids = [p["source_event_id"] for p in via.to_dict()["path"] if p["source_event_id"]]
                link_recs = store.get_events(link_ids)
                last = store.get_event(link_ids[-1]) if link_ids else None
                ref = last.timestamp if last is not None else None
                if spec is not None and ev.event_type in spec.event_types:
                    ok, _ = check_constraint(spec.constraint, ev, parsed.dates, ref)
                    if ok is False:
                        continue
                self._add_expansion(items, ev, via, q)
                n_exp += 1
                for lr in link_recs:      # the records that document the links are evidence too
                    self._add_expansion(items, lr, via, q, link=True)
            if spec is not None and spec.constraint == "latest_before_link":
                self._keep_latest(items, spec)
        # newer versions of superseded records
        newer_of: dict[str, str] = {}
        for it in list(items.values()):
            newer = store.superseded_by(it.event.event_id)
            if newer:
                newer_of[it.event.event_id] = newer
                if newer not in items:
                    nev = store.get_event(newer)
                    if nev is not None:
                        items[newer] = RetrievedItem(nev, self._sim(nev, q), ["supersedes"],
                                                     explanation=f"supersedes retrieved record {it.event.event_id}")
        # duplicates (identical content)
        seen, dups = {}, []
        for eid, it in list(items.items()):
            key = (it.event.text, it.event.timestamp)
            if key in seen:
                dups.append(eid)
                del items[eid]
            else:
                seen[key] = eid
        n_time = n_text = 0
        if parsed is not None and spec is None:
            for eid in [e for e, it in items.items() if not self._time_ok(it.event, parsed)]:
                del items[eid]
                n_time += 1
        if "vector" not in self.channels:
            for it in items.values():
                it.similarity = _lexical(it.event.text, parsed, query_text)
        answer_types = tuple(spec.event_types + spec.hypothesis_types) if spec else ()
        ranked = rerank(list(items.values()), *self.w, answer_types) if self.do_rerank else \
            sorted(items.values(), key=lambda it: -it.similarity)
        kept = ranked[:k]
        kept_ids = {it.event.event_id for it in kept}
        for it in list(kept):   # closure: linking records of kept items, and newer versions
            extra = [p.get("source_event_id") for p in it.path] + [newer_of.get(it.event.event_id)]
            for lid in extra:
                if lid and lid not in kept_ids and lid in items:
                    kept.append(items[lid])
                    kept_ids.add(lid)
        # Free-text document chunks carry no structured identifiers, so neither link expansion nor
        # metadata filters can reach them; fetch chunks whose TEXT names a question identifier.
        # Without a typed spec, operational log rows (also untyped) are fetched the same way.
        if parsed is not None and "text" in self.channels:
            types = ("document_chunk",) if spec is not None else None
            day = parsed.dates[0] if (spec is None and parsed.dates) else None
            # untyped questions: scan more rows (logs are dense), keep the most similar max_doc_chunks
            limit = self.max_doc_chunks if spec is not None else 50 * self.max_doc_chunks
            for ident in parsed.anchor_map():
                found = []
                for ev in store.text_mentions(ident, event_types=types, limit=limit, day=day):
                    if ev.event_id in kept_ids or not re.search(
                            r"(?<![A-Za-z0-9-])" + re.escape(ident) + r"(?![A-Za-z0-9-])", ev.text, re.I):
                        continue
                    if spec is None and (ev.event_type in TYPED_TYPES or not self._time_ok(ev, parsed)):
                        n_time += 1
                        continue
                    sim = self._sim(ev, q) if "vector" in self.channels else _lexical(ev.text, parsed, query_text)
                    found.append(RetrievedItem(ev, sim, ["text_mention"], explanation=f"record text names {ident}"))
                found.sort(key=lambda it: (-it.similarity, it.event.event_id))
                for it in found[: self.max_doc_chunks]:
                    kept.append(it)
                    kept_ids.add(it.event.event_id)
                    n_text += 1
        t_total = time.perf_counter() - t0
        return RetrievalResult(kept, traversed, {"embed_s": t_embed, "vector_s": t_vec, "total_s": t_total},
                               len(hits), n_exp, dups, n_time, n_text)

    def _time_ok(self, ev: Event, parsed) -> bool:
        """Metadata filter for untyped questions: a stated date (and clock time) must match the
        record's timestamp; undated records pass (verification checks their text)."""
        if not parsed.dates or not ev.timestamp:
            return True
        if ev.timestamp[:10] != parsed.dates[0]:
            return False
        if parsed.times and len(ev.timestamp) >= 16:
            from datetime import datetime

            try:
                d = datetime.fromisoformat(ev.timestamp[:19]) - datetime.fromisoformat(f"{parsed.dates[0]}T{parsed.times[0]}")
            except ValueError:
                return True
            return abs(d.total_seconds()) / 60 <= self.time_window_min
        return True

    def _sim(self, ev: Event, q: np.ndarray) -> float:
        row = self.memory.store.row(ev.event_id)
        if row and row["vector_id"] is not None:
            try:
                return float(self.memory.index.index.reconstruct(int(row["vector_id"])) @ q)
            except RuntimeError:
                pass
        return float(self.memory.embedder.encode([ev.text])[0] @ q)

    def _add_expansion(self, items, ev, via, q, link: bool = False):
        path = via.to_dict()["path"]
        expl = ("documents link " if link else "linked via ") + " -> ".join(
            f"{p['src']} {p['rel_type']} {p['dst']}" for p in path) if path else f"mentions {via.node_id}"
        if ev.event_id in items:
            it = items[ev.event_id]
            if "relationship" not in it.sources:
                it.sources.append("relationship")
            if it.depth is None or via.depth < it.depth:
                it.depth, it.path, it.explanation = via.depth, path, expl
            return
        items[ev.event_id] = RetrievedItem(ev, self._sim(ev, q), ["relationship"], via.depth, path, explanation=expl)

    @staticmethod
    def _keep_latest(items, spec):
        cands = [it for it in items.values() if it.event.event_type in spec.event_types and "relationship" in it.sources]
        if len(cands) > 1:
            latest = max(it.event.timestamp or "" for it in cands)
            for it in cands:
                if (it.event.timestamp or "") != latest and it.sources == ["relationship"]:
                    del items[it.event.event_id]


def _lexical(text: str, parsed, query_text: str) -> float:
    """Fraction of the question's content terms (or words, if it has none) present in the text."""
    from agentmamba.retrieval.query_processor import stem

    terms = parsed.focus_terms() if parsed is not None else []
    if not terms:
        terms = [stem(w) for w in re.findall(r"[a-zA-Z]{3,}", query_text.lower())]
    words = {stem(w) for w in re.findall(r"[a-zA-Z]+", text.lower())}
    return sum(t in words for t in terms) / len(terms) if terms else 0.0
