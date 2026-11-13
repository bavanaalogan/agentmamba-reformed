"""Request/response validation models for the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ContextRecord(BaseModel):
    """An initial-context record supplied by the caller (already in AgentMamba event layout)."""
    event_id: str
    stage: str
    event_type: str
    timestamp: str | None = None
    text: str = Field(min_length=1, max_length=5000)
    source_ref: str = "api://context"
    source_type: Literal["public", "authorized", "synthetic"] = "authorized"
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
    supersedes: str | None = None
    observations: list[dict] = Field(default_factory=list)


class QueryFilters(BaseModel):
    asset_id: list[str] | None = None
    heat_id: list[str] | None = None
    stage: list[str] | None = None
    source_type: list[str] | None = None
    start: str | None = None
    end: str | None = None


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    initial_context: list[ContextRecord] = Field(default_factory=list, max_length=50)
    initial_context_ids: list[str] = Field(default_factory=list, max_length=50,
                                           description="ids of stored records the engineer is looking at")
    mode: Literal["conditional", "always", "never"] = "conditional"
    verification: bool = True
    filters: QueryFilters | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class EvalCase(BaseModel):
    """Either a log-benchmark case (split, question_id, stage, run) evaluated on that run's memory
    snapshot, or a protocol case (protocol, question_id) evaluated on the CURRENT memory."""
    split: Literal["dev", "test"] | None = None
    question_id: str
    stage: int | None = Field(default=None, ge=0, le=4)
    run: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")
    protocol: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")


class CompareIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    initial_context_ids: list[str] = Field(default_factory=list, max_length=50)
    window_tokens: int = Field(default=768, ge=0, le=4096,
                               description="recent-arrival window shared by both systems (0 = selected records only)")
    mode: Literal["conditional", "always", "never"] = "conditional"
    verification: bool = True
    case: EvalCase | None = None


class EventsIn(BaseModel):
    events: list[dict] = Field(min_length=1, max_length=5000)
    budget_bytes: int | None = Field(default=None, ge=1)
