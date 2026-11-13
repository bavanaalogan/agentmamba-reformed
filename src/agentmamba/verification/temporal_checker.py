"""Temporal constraints and process-sequence plausibility.

Constraints (declared per AnswerSpec):
  on_date             record date == date named in the question
  before_date         record timestamp <= end of the date named in the question
  latest_before_link  record is strictly before the record that links its entity into the
                      anchor chain (e.g. ladle inspection before the heat->ladle assignment);
                      among passing records only the latest is kept (selection rule)
  latest_version      handled by conflict_resolution (supersession / version numbers)
"""
from __future__ import annotations

from agentmamba.memory.schema import Event

# Nominal process order of record types for the same material flow (heat -> slab -> coil).
PROCESS_ORDER = {"hot_metal_tap": 0, "bof_heat_record": 1, "ladle_assignment": 2, "ladle_treatment": 3,
                 "cast_record": 4, "rolling_record": 5, "coil_inspection": 6, "rca_report": 7}


def check_constraint(constraint: str | None, record: Event, dates: list[str], link_ref_time: str | None
                     ) -> tuple[bool | None, str]:
    """Returns (ok, note). ok=None means the constraint could not be evaluated."""
    if constraint is None or constraint == "latest_version":
        return True, ""
    ts = record.timestamp
    if ts is None:
        return None, "record has no timestamp"
    if constraint == "on_date":
        if not dates:
            return None, "question names no date"
        return ts[:10] == dates[0], f"record date {ts[:10]} vs requested {dates[0]}"
    if constraint == "before_date":
        if not dates:
            return None, "question names no date"
        return ts <= f"{dates[0]}T23:59:59", f"record {ts} vs before end of {dates[0]}"
    if constraint == "latest_before_link":
        if link_ref_time is None:
            return None, "no linking record time available"
        return ts < link_ref_time, f"record {ts} vs linking record {link_ref_time}"
    return None, f"unknown constraint {constraint}"


def process_sequence_violations(records: list[Event]) -> list[str]:
    """Flags pairs of records on the same heat/slab/coil chain whose timestamps contradict the
    nominal process order (e.g. a coil rolled before its slab was cast)."""
    chain = [r for r in records if r.event_type in PROCESS_ORDER and r.timestamp]
    out = []
    for a in chain:
        for b in chain:
            if PROCESS_ORDER[a.event_type] < PROCESS_ORDER[b.event_type] and a.timestamp > b.timestamp:
                shared = set(a.entity_ids()) & set(b.entity_ids())
                linked = shared or (a.slab_id and a.slab_id == b.slab_id) or (a.heat_id and a.heat_id == b.heat_id)
                if linked:
                    out.append(f"{a.event_id} ({a.event_type}, {a.timestamp}) is after {b.event_id} "
                               f"({b.event_type}, {b.timestamp})")
    return out
