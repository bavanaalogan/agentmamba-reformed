"""Evidence entries with provenance and evidence-type classification.

evidence_type:
  observed    the answer is a recorded measurement (an Observation with a unit)
  documented  the record states the event / finding / outcome / relationship
  inferred    the record is connected to the answer only through an association the system
              surfaced (e.g. an upstream abnormality offered as a causal *hypothesis*)
The record's ``source_type`` (public | authorized | synthetic) is reported separately.
"""
from __future__ import annotations

from agentmamba.memory.schema import Event


def evidence_entry(ev: Event, role: str, evidence_type: str, answer_field: str | None = None) -> dict:
    excerpt = ev.text if len(ev.text) <= 300 else ev.text[:297] + "..."
    return {"record_id": ev.event_id, "source": ev.source_ref, "source_type": ev.source_type,
            "asset_id": ev.asset_id, "stage": ev.stage, "event_type": ev.event_type, "timestamp": ev.timestamp,
            "excerpt": excerpt, "evidence_type": evidence_type, "role": role, "answer_field": answer_field}


def classify(ev: Event, answer_field: str | None, hypothesis: bool) -> str:
    if hypothesis:
        return "inferred"
    o = ev.obs(answer_field) if answer_field else None
    if o is not None and o.unit:
        return "observed"
    return "documented"


def citations_valid(evidence: list[dict], resolve) -> list[str]:
    """Return record ids that do not resolve to a real stored/provided record."""
    return [e["record_id"] for e in evidence if resolve(e["record_id"]) is None]
