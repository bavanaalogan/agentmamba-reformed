"""Bounded breadth-first traversal over entity--entity relationships.

Only edges whose status is in ``allowed_statuses`` (documented / observed / synthetic
ground truth by default) are followed; ``inferred`` and ``hypothesis`` edges are never used
to justify evidence. ``mentions`` edges (event--entity) are not traversed here; callers fetch
the events that mention reached entities separately. Hub / group kinds (asset, document, and
the shared groupings ladle, cast sequence, hot-metal batch) can be *reached* but are not
expanded *through* (except as the start node): otherwise heat A -> cast sequence -> heat B
would make a sibling heat's records look linked to heat A.
Cycle protection: each node is visited once (shortest path kept). ``max_nodes`` caps work.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from agentmamba.memory.schema import Relationship

TRUSTED_STATUSES = ("documented", "observed", "synthetic_ground_truth")
HUB_KINDS = ("asset", "document", "ladle", "cast_seq", "hm_batch")


@dataclass
class Reached:
    node_id: str
    kind: str
    depth: int
    path: list[Relationship] = field(default_factory=list)

    def link_records(self) -> list[str]:
        return [r.source_event_id for r in self.path if r.source_event_id]

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "kind": self.kind, "depth": self.depth,
                "path": [{"rel_id": r.rel_id, "src": r.src_id, "rel_type": r.rel_type, "dst": r.dst_id,
                          "status": r.status, "source_event_id": r.source_event_id} for r in self.path]}


def traverse(graph, start: dict[str, str], max_depth: int = 3, max_nodes: int = 60,
             allowed_statuses: tuple = TRUSTED_STATUSES, hub_kinds: tuple = HUB_KINDS) -> dict[str, Reached]:
    """``start`` maps entity_id -> kind. Returns {entity_id: Reached} including the starts."""
    reached = {nid: Reached(nid, kind, 0, []) for nid, kind in start.items()}
    queue = deque(reached.values())
    while queue and len(reached) < max_nodes:
        cur = queue.popleft()
        if cur.depth >= max_depth or (cur.kind in hub_kinds and cur.depth > 0):
            continue
        for rel in sorted(graph.neighbors(cur.node_id), key=lambda r: r.rel_id):
            if rel.rel_type == "mentions" or rel.src_kind == "event" or rel.dst_kind == "event":
                continue
            if rel.status not in allowed_statuses:
                continue
            other, kind = (rel.dst_id, rel.dst_kind) if rel.src_id == cur.node_id else (rel.src_id, rel.src_kind)
            if other in reached:
                continue
            nxt = Reached(other, kind, cur.depth + 1, cur.path + [rel])
            reached[other] = nxt
            queue.append(nxt)
            if len(reached) >= max_nodes:
                break
    return reached
