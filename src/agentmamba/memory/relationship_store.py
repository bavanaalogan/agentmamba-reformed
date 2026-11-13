"""Relationship graphs over documented links.

Two interchangeable neighbour sources:
* ``SQLiteRelationshipGraph`` - the persistent relationship table (retained records only).
* ``InMemoryRelationshipGraph`` - links derived from an explicit list of records (e.g. the
  initial context), used when the caller must NOT touch the database.
"""
from __future__ import annotations

from collections import defaultdict

from agentmamba.data.event_linker import link_events
from agentmamba.memory.schema import Event, Relationship


class InMemoryRelationshipGraph:
    def __init__(self, records: list[Event] | None = None, rels: list[Relationship] | None = None):
        self._adj: dict[str, list[Relationship]] = defaultdict(list)
        for r in (rels or []) + (link_events(records or [])):
            self.add(r)

    def add(self, r: Relationship) -> None:
        if all(x.rel_id != r.rel_id for x in self._adj[r.src_id]):
            self._adj[r.src_id].append(r)
            self._adj[r.dst_id].append(r)

    def neighbors(self, node_id: str) -> list[Relationship]:
        return list(self._adj.get(node_id, []))


class SQLiteRelationshipGraph:
    def __init__(self, store):
        self.store = store

    def neighbors(self, node_id: str) -> list[Relationship]:
        return self.store.relationships_of(node_id)


class UnionGraph:
    def __init__(self, *graphs):
        self.graphs = [g for g in graphs if g is not None]

    def neighbors(self, node_id: str) -> list[Relationship]:
        seen, out = set(), []
        for g in self.graphs:
            for r in g.neighbors(node_id):
                if r.rel_id not in seen:
                    seen.add(r.rel_id)
                    out.append(r)
        return out
