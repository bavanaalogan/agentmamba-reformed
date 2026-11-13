"""Identity check: is a candidate's source record connected to the question's anchor
identifier by documented relationships? Similar text on another asset is NOT a link."""
from __future__ import annotations

from dataclasses import dataclass, field

from agentmamba.memory.schema import Event, Relationship
from agentmamba.retrieval.relationship_traversal import traverse


@dataclass
class LinkResult:
    linked: bool
    depth: int | None = None
    path: list[Relationship] = field(default_factory=list)
    via_entity: str | None = None

    def link_record_ids(self) -> list[str]:
        return [r.source_event_id for r in self.path if r.source_event_id]


def link_candidate(anchors: dict[str, str], record: Event, graph, max_depth: int = 4, max_nodes: int = 200
                   ) -> LinkResult:
    ents = record.entity_ids()
    for a in anchors:
        if a in ents:
            return LinkResult(True, 0, [], a)
    reached = traverse(graph, anchors, max_depth=max_depth, max_nodes=max_nodes)
    best = None
    for ent, kind in ents.items():
        if kind in ("asset", "document"):
            continue
        r = reached.get(ent)
        if r is not None and (best is None or r.depth < best.depth) and not identity_conflict(record, anchors, r):
            best = r
    if best is None:
        return LinkResult(False)
    return LinkResult(True, best.depth, best.path, best.node_id)


# Kinds that identify ONE unit of material / work. A record naming a different heat (slab, coil,
# work order) than the documented chain is about something else, even if it shares a ladle or
# cast sequence with the chain.
IDENTITY_KINDS = ("heat", "slab", "coil", "work_order")


def identity_conflict(record: Event, anchors: dict[str, str], reached) -> bool:
    chain: dict[str, set] = {}
    for nid, kind in anchors.items():
        chain.setdefault(kind, set()).add(nid)
    for rel in reached.path:
        chain.setdefault(rel.src_kind, set()).add(rel.src_id)
        chain.setdefault(rel.dst_kind, set()).add(rel.dst_id)
    for ent, kind in record.entity_ids().items():
        if kind in IDENTITY_KINDS and kind in chain and ent not in chain[kind]:
            return True
    return False
