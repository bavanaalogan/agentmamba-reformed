"""Answer extraction.

Primary track ("mamba"): candidate answer spans are extracted from the supplied records
(typed by the parsed question's AnswerSpec), and Mamba ranks them by the mean
log-likelihood of the candidate continuing a prompt that contains the records and the
question. Mamba never sees records that were not explicitly passed to it.

Ablation track ("first_candidate"): picks candidates in record order. It is NOT a Mamba
track and is reported separately; it exists to measure what Mamba's ranking contributes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from agentmamba.memory.schema import Event
from agentmamba.retrieval.query_processor import AnswerSpec


@dataclass
class Candidate:
    value: str
    record: Event
    field: str
    hypothesis: bool = False
    score: float | None = None

    def key(self) -> str:
        return normalize_answer(self.value)

    def to_dict(self) -> dict:
        return {"value": self.value, "record_id": self.record.event_id, "field": self.field,
                "hypothesis": self.hypothesis, "score": None if self.score is None else round(self.score, 4)}


def normalize_answer(s: str | None) -> str:
    if s is None:
        return ""
    return " ".join(str(s).lower().replace(",", " ").split())


def field_value(ev: Event, fld: str) -> str | None:
    if hasattr(ev, fld) and fld not in ("observations", "raw", "text"):
        v = getattr(ev, fld)
        return str(v) if v not in (None, "") else None
    o = ev.obs(fld)
    if o is None or o.value in (None, ""):
        return None
    return f"{o.value} {o.unit}" if o.unit else str(o.value)


def extract_candidates(spec: AnswerSpec, records: list[Event]) -> list[Candidate]:
    out, seen = [], set()
    for ev in records:
        if ev.event_type in spec.event_types:
            v = field_value(ev, spec.answer_field)
            if v is not None and (ev.event_id, v) not in seen:
                seen.add((ev.event_id, v))
                out.append(Candidate(v, ev, spec.answer_field))
    if spec.hypothesis_field:
        for ev in records:
            if ev.event_type in spec.hypothesis_types:
                v = field_value(ev, spec.hypothesis_field)
                if v is not None and (ev.event_id, v) not in seen:
                    seen.add((ev.event_id, v))
                    out.append(Candidate(v, ev, spec.hypothesis_field, hypothesis=True))
    return out


def format_record(ev: Event) -> str:
    if ev.timestamp_raw and ev.text.startswith(ev.timestamp_raw):
        # a verbatim log line already carries its timestamp and source: don't repeat them
        return f"[{ev.event_id}] {ev.text}"
    ts = ev.timestamp or "time-unknown"
    return f"[{ev.event_id}] {ts} {ev.stage} {ev.event_type}: {ev.text}"


def build_prompt(question: str, records: list[Event]) -> str:
    recs = sorted(records, key=lambda e: (e.timestamp or "", e.event_id))
    body = "\n".join(format_record(e) for e in recs)
    return f"Plant records:\n{body}\nQuestion: {question}\nAnswer:"


@dataclass
class RankResult:
    candidates: list[Candidate]
    model: str
    seconds: float
    prompt_chars: int = 0
    notes: list[str] = field(default_factory=list)
    prompt_record_ids: list[str] = field(default_factory=list)   # records actually placed in the prompt
    model_call: dict = field(default_factory=dict)               # cache hit, token counts (if measured)


class MambaAnswerer:
    """``max_prompt_records`` caps the records placed in the prompt (None = no cap); ``max_tokens``
    overrides the backbone's prompt token budget (None = backbone default, 768)."""
    name = "mamba"

    def __init__(self, backbone, max_prompt_records: int | None = 16, max_tokens: int | None = None):
        self.backbone = backbone
        self.max_prompt_records = max_prompt_records
        self.max_tokens = max_tokens

    def rank(self, question: str, records: list[Event], candidates: list[Candidate]) -> RankResult:
        t = time.perf_counter()
        if not candidates:
            return RankResult([], self.backbone.model_id, 0.0)
        # keep candidate-bearing records first, then others, within the record cap
        bearing = list(dict.fromkeys(c.record.event_id for c in candidates))
        by_id = {e.event_id: e for e in records}
        chosen = [by_id[i] for i in bearing if i in by_id]
        chosen += [e for e in records if e.event_id not in set(bearing)]
        if self.max_prompt_records:
            chosen = chosen[: self.max_prompt_records]
        prompt = build_prompt(question, chosen)
        values = list(dict.fromkeys(c.value for c in candidates))
        if self.max_tokens:
            raw = self.backbone.score_candidates(prompt, values, self.max_tokens)
        else:
            raw = self.backbone.score_candidates(prompt, values)
        scores = dict(zip(values, raw))
        for c in candidates:
            c.score = scores[c.value]
        ranked = sorted(candidates, key=lambda c: (-c.score, c.record.timestamp or "", c.record.event_id))
        return RankResult(ranked, self.backbone.model_id, time.perf_counter() - t, len(prompt),
                          prompt_record_ids=[e.event_id for e in chosen],
                          model_call=dict(getattr(self.backbone, "last_call", {}) or {}))


class FirstCandidateAnswerer:
    """Ablation, not Mamba: candidates in chronological record order."""
    name = "first_candidate"

    def rank(self, question: str, records: list[Event], candidates: list[Candidate]) -> RankResult:
        ranked = sorted(candidates, key=lambda c: (c.record.timestamp or "", c.record.event_id))
        for n, c in enumerate(ranked):
            c.score = -float(n)
        return RankResult(ranked, "first_candidate(non-model ablation)", 0.0,
                          prompt_record_ids=[e.event_id for e in records])
