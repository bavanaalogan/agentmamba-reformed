"""Evidence-based relationship construction.

Rules (the ONLY ways a relationship is created):
1. ``mentions``: a record carries a structured identifier -> event--entity edge.
2. Typed entity--entity edges when a single record documents two identifiers together
   (e.g. a ladle-assignment record names both the heat and the ladle). The record is kept
   as ``source_event_id``; if that record is not retained, the link is not retained either.
3. ``supersedes`` / ``references`` event--event edges from explicit source fields.

No relationship is ever created from timestamp proximity or embedding similarity.
Asset identifiers are hubs, so they only participate in ``mentions`` and ``work_on`` edges.
"""
from __future__ import annotations

from collections.abc import Iterable

from agentmamba.memory.schema import ID_FIELDS, Event, Relationship

# (event_type, src_field, dst_field) -> rel_type
TYPED_LINKS = {
    ("ladle_assignment", "heat_id", "ladle_id"): "carried_by",
    ("ladle_treatment", "heat_id", "ladle_id"): "treated_in",
    ("cast_record", "heat_id", "cast_seq_id"): "cast_in_sequence",
    ("cast_record", "slab_id", "heat_id"): "cast_from_heat",
    ("rolling_record", "coil_id", "slab_id"): "rolled_from",
    ("bof_heat_record", "heat_id", "batch_id"): "charged_with",
    ("hot_metal_tap", "batch_id", "asset_id"): "tapped_from",
    ("work_order", "work_order_id", "asset_id"): "work_on",
    ("maintenance_followup", "work_order_id", "asset_id"): "work_on",
    ("ladle_incident", "ladle_id", "heat_id"): "incident_during",
}
HUB_FIELDS = {"asset_id", "doc_id"}


def link_event(ev: Event) -> list[Relationship]:
    rels: list[Relationship] = []
    for val, kind in ev.entity_ids().items():
        rels.append(Relationship(ev.event_id, "event", val, kind, "mentions", "documented", ev.event_id))
    present = [f for f in ID_FIELDS if getattr(ev, f)]
    typed = set()
    for (etype, a, b), rel_type in TYPED_LINKS.items():
        if etype == ev.event_type and getattr(ev, a) and getattr(ev, b):
            rels.append(Relationship(getattr(ev, a), ID_FIELDS[a], getattr(ev, b), ID_FIELDS[b], rel_type,
                                     "documented", ev.event_id))
            typed.update({a, b})
    # Remaining non-hub co-recorded identifiers: generic documented association.
    rest = [f for f in present if f not in HUB_FIELDS]
    for i, a in enumerate(rest):
        for b in rest[i + 1:]:
            if {a, b} <= typed or getattr(ev, a) == getattr(ev, b):
                continue
            rels.append(Relationship(getattr(ev, a), ID_FIELDS[a], getattr(ev, b), ID_FIELDS[b], "co_recorded",
                                     "documented", ev.event_id))
    if ev.supersedes:
        rels.append(Relationship(ev.event_id, "event", ev.supersedes, "event", "supersedes", "documented", ev.event_id))
    for rid in ev.related_event_ids:
        if rid != ev.event_id:
            rels.append(Relationship(ev.event_id, "event", rid, "event", "references", "documented", ev.event_id))
    return rels


def link_events(events: Iterable[Event]) -> list[Relationship]:
    seen, out = set(), []
    for ev in events:
        for r in link_event(ev):
            if r.rel_id not in seen:
                seen.add(r.rel_id)
                out.append(r)
    return out
