"""Retrieval and evidence metrics (evaluation only)."""
from __future__ import annotations


def recall_at_k(retrieved: list[str], required: list[str], k: int | None = None) -> float | None:
    if not required:
        return None
    r = retrieved[:k] if k else retrieved
    return len(set(r) & set(required)) / len(set(required))


def precision_at_k(retrieved: list[str], required: list[str], k: int | None = None) -> float | None:
    r = retrieved[:k] if k else retrieved
    if not r or not required:
        return None
    return len(set(r) & set(required)) / len(r)


def reciprocal_rank(retrieved: list[str], required: list[str]) -> float | None:
    if not required:
        return None
    req = set(required)
    for i, rid in enumerate(retrieved, start=1):
        if rid in req:
            return 1.0 / i
    return 0.0


def prf(pred: list[str], gold: list[str]) -> tuple[float | None, float | None, float | None]:
    if not gold and not pred:
        return None, None, None
    tp = len(set(pred) & set(gold))
    p = tp / len(set(pred)) if pred else None
    r = tp / len(set(gold)) if gold else None
    f = None if p is None or r is None or (p + r) == 0 else 2 * p * r / (p + r)
    if p is not None and r is not None and p + r == 0:
        f = 0.0
    return p, r, f
