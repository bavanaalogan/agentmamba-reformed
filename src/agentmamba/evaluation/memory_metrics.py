"""Memory-selection metrics. Test annotations are used here ONLY to score what was kept."""
from __future__ import annotations

from collections import Counter

import numpy as np

from agentmamba.memory.schema import Event

KEY_OBS = ("finding", "abnormality", "outcome", "rca_status", "defect")


def category(ev: Event) -> str:
    for name in KEY_OBS:
        o = ev.obs(name)
        if o is not None:
            return f"{ev.event_type}:{o.value}"
    return f"{ev.event_type}:v{ev.version}" if ev.version else ev.event_type


def rare_categories(train_events: list[Event], max_freq: float = 0.01) -> set[str]:
    """Categories (event type + key observation) with frequency below ``max_freq`` in the
    TRAINING stream. Computed without labels."""
    c = Counter(category(e) for e in train_events)
    n = sum(c.values())
    return {k for k, v in c.items() if v / n < max_freq}


def memory_metrics(stream: list[Event], retained: set[str], useful: set[str], rare: set[str]) -> dict:
    total_bytes = sum(e.size_bytes() for e in stream)
    kept = [e for e in stream if e.event_id in retained]
    kept_bytes = sum(e.size_bytes() for e in kept)
    useful_in_stream = [e for e in stream if e.event_id in useful]
    rare_ev = [e for e in stream if category(e) in rare]
    rare_useful = [e for e in rare_ev if e.event_id in useful]
    by_type = Counter(e.event_type for e in kept)
    return {
        "candidates": len(stream), "retained_records": len(kept), "write_rate": len(kept) / max(1, len(stream)),
        "candidate_bytes": total_bytes, "retained_bytes": kept_bytes, "retention_ratio_bytes": kept_bytes / max(1, total_bytes),
        "relevant_evidence_retention": _r(sum(e.event_id in retained for e in useful_in_stream), len(useful_in_stream)),
        "n_relevant_evidence": len(useful_in_stream),
        "rare_event_retention": _r(sum(e.event_id in retained for e in rare_ev), len(rare_ev)), "n_rare": len(rare_ev),
        "useful_rare_retention": _r(sum(e.event_id in retained for e in rare_useful), len(rare_useful)),
        "n_useful_rare": len(rare_useful),
        "irrelevant_retained_fraction": _r(sum(e.event_id not in useful for e in kept), len(kept)),
        "retained_by_type": dict(by_type.most_common()),
    }


def gate_test_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    from agentmamba.gate.train import classification_report

    return classification_report(y, p, threshold)


def _r(a, b):
    return None if b == 0 else a / b
