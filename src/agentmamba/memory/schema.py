"""Core record types: events, relationships, questions and (separately stored) gold
annotations. Everything is a plain dataclass so it serializes to JSON deterministically."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

STAGES = ("IRONMAKING", "STEELMAKING", "LADLE", "CASTING", "ROLLING", "MAINTENANCE", "QUALITY", "UTILITIES")
SOURCE_TYPES = ("public", "authorized", "synthetic")
SEVERITIES = (None, "info", "minor", "major", "critical")

# Structured identifier fields an event may carry, mapped to their entity kind.
ID_FIELDS = {
    "heat_id": "heat",
    "ladle_id": "ladle",
    "cast_seq_id": "cast_seq",
    "slab_id": "slab",
    "coil_id": "coil",
    "batch_id": "hm_batch",
    "work_order_id": "work_order",
    "doc_id": "document",
    "asset_id": "asset",
}
RAW_ID_FIELDS = {"alarm_id": "alarm", "trip_id": "trip", "fault_id": "fault", "incident_id": "incident"}
RAW_ID_PLACEHOLDERS = {"", "NA", "N/A", "NONE", "NULL", "-", "?", "UNKNOWN"}

# Relationship provenance. Only "documented"/"observed"/"synthetic_ground_truth" edges are
# used to justify cross-stage evidence; "inferred"/"hypothesis" are kept but flagged.
REL_STATUSES = ("documented", "observed", "inferred", "hypothesis", "synthetic_ground_truth")


def content_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}-{hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:12]}"


@dataclass
class Observation:
    name: str
    value: Any
    unit: str | None = None          # unit exactly as recorded at source; never silently converted
    raw: Any = None                  # original source value (string), if different


@dataclass
class Event:
    event_id: str
    stage: str
    event_type: str
    timestamp: str | None            # ISO-8601 as normalized; None if unknown
    text: str                        # human-readable record text (the evidence excerpt source)
    source_ref: str
    source_type: str                 # public | authorized | synthetic
    campaign_id: str | None = None   # synthetic scenario group / public dataset id
    timestamp_raw: str | None = None
    timezone: str = "unknown"
    asset_id: str | None = None
    heat_id: str | None = None
    ladle_id: str | None = None
    cast_seq_id: str | None = None
    slab_id: str | None = None
    coil_id: str | None = None
    batch_id: str | None = None
    work_order_id: str | None = None
    doc_id: str | None = None
    severity: str | None = None
    version: int | None = None
    effective_date: str | None = None
    supersedes: str | None = None    # event_id of the record this one supersedes
    related_event_ids: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    raw: dict | None = None          # original source record, untouched
    content_hash: str = ""
    ingested_at: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}")
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"unknown source_type {self.source_type!r}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")
        if not self.event_id or not self.text:
            raise ValueError("event_id and text are required")
        self.observations = [o if isinstance(o, Observation) else Observation(**o) for o in self.observations]
        if not self.content_hash:
            self.content_hash = content_hash(self.content_payload())

    def content_payload(self) -> dict:
        d = self.to_dict()
        for k in ("content_hash", "ingested_at", "raw"):
            d.pop(k, None)
        return d

    def entity_ids(self) -> dict[str, str]:
        """{entity_id: kind} for every structured identifier on this record."""
        out = {}
        for fld, kind in ID_FIELDS.items():
            val = getattr(self, fld)
            if val:
                out[val] = kind
        if self.raw:
            raw_fields = {str(name).strip().lower().replace("-", "_").replace(" ", "_"): value
                          for name, value in self.raw.items()}
            for fld, kind in RAW_ID_FIELDS.items():
                val = raw_fields.get(fld)
                if val is not None and str(val).strip().upper() not in RAW_ID_PLACEHOLDERS:
                    out.setdefault(str(val).strip().upper(), kind)
        return out

    def obs(self, name: str) -> Observation | None:
        return next((o for o in self.observations if o.name == name), None)

    def size_bytes(self) -> int:
        """Budget unit: UTF-8 bytes of the text plus serialized structured observations."""
        return len(self.text.encode("utf-8")) + len(
            json.dumps([asdict(o) for o in self.observations], sort_keys=True).encode("utf-8")
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        d = dict(d)
        d["observations"] = [Observation(**o) for o in d.get("observations") or []]
        return cls(**d)


@dataclass
class Relationship:
    src_id: str
    src_kind: str
    dst_id: str
    dst_kind: str
    rel_type: str
    status: str
    source_event_id: str | None = None   # the record that documents this relationship
    rel_id: str = ""

    def __post_init__(self) -> None:
        if self.status not in REL_STATUSES:
            raise ValueError(f"unknown relationship status {self.status!r}")
        if self.src_id == self.dst_id:
            raise ValueError("self-relationships are not allowed")
        if not self.rel_id:
            self.rel_id = stable_id("REL", self.src_id, self.dst_id, self.rel_type, self.source_event_id)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Question:
    """What the live system is allowed to see."""
    question_id: str
    campaign_id: str
    split: str
    text: str
    initial_context_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Annotation:
    """Gold labels. Stored in a separate file and never passed to the live pipeline."""
    question_id: str
    family: str
    question_type: str
    stages: list[str]
    answer_kind: str                    # value | abstain | clarify | conflict | hypothesis
    expected_answer: str | None
    answerable: bool
    required_evidence_ids: list[str]
    relationship_ids: list[str]
    expected_sources: list[str]
    initial_sufficient: bool
    retrieval_required: bool
    cross_stage: bool
    temporal_sensitive: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
