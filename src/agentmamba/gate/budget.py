"""Deterministic budget enforcement and the non-learned selection policies.

Budget unit: bytes (``Event.size_bytes``: UTF-8 text + serialized observations).
Streaming rule: each offered record is admitted; while the retained bytes exceed the budget,
the lowest-scored retained record is evicted (ties: later event_id evicted first). An
evicted record never returns. The offered record itself may be evicted immediately (SKIP).
"""
from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field

from agentmamba.memory.schema import Event

TYPE_PRIORITY = {  # heuristic policy: category priority (documented, not learned)
    "ladle_incident": 1.0, "rca_report": 1.0, "maintenance_followup": 1.0, "work_order": 1.0,
    "ladle_inspection": 1.0, "coil_inspection": 0.8, "procedure_doc": 0.8, "caster_alarm": 0.6,
    "cast_record": 0.5, "bof_heat_record": 0.5, "rolling_record": 0.5, "ladle_assignment": 0.5,
    "ladle_treatment": 0.4, "hot_metal_tap": 0.4,
}
SEV = {None: 0, "info": 1, "minor": 2, "major": 3, "critical": 4}


def heuristic_score(ev: Event) -> float:
    """Severity first, then category priority: severity_ord + type_priority (0..5)."""
    return SEV.get(ev.severity, 0) + TYPE_PRIORITY.get(ev.event_type, 0.0)


def random_score(ev: Event, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{ev.event_id}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


@dataclass(order=True)
class _Item:
    score: float
    neg_key: tuple = field(compare=True)
    event_id: str = field(compare=False)
    size: int = field(compare=False)


class BudgetedSelector:
    def __init__(self, budget_bytes: int | None):
        self.budget = budget_bytes   # None == unlimited (store-all)
        self.used = 0
        self._heap: list[_Item] = []
        self._alive: dict[str, _Item] = {}

    def offer(self, event_id: str, size: int, score: float) -> tuple[str, list[str]]:
        """Returns (decision, evicted_ids) where decision is WRITE or SKIP."""
        # neg_key makes later event_ids lose ties (evicted first): compare by inverted chars
        item = _Item(float(score), tuple(-ord(c) for c in event_id), event_id, size)
        if self.budget is None:
            self._alive[event_id] = item
            self.used += size
            return "WRITE", []
        if size > self.budget:
            return "SKIP", []
        heapq.heappush(self._heap, item)
        self._alive[event_id] = item
        self.used += size
        evicted = []
        while self.used > self.budget:
            low = heapq.heappop(self._heap)
            if low.event_id not in self._alive:
                continue
            del self._alive[low.event_id]
            self.used -= low.size
            evicted.append(low.event_id)
        if event_id in evicted:
            evicted.remove(event_id)
            return "SKIP", evicted
        return "WRITE", evicted

    def retained(self) -> set[str]:
        return set(self._alive)
