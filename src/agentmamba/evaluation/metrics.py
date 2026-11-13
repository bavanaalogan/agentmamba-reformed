"""Answer-quality metrics and statistical helpers. Definitions are in docs/EXPERIMENTS.md.

Per-question correctness by gold ``answer_kind``:
  value       answered and normalized answer_value == expected (Exact Match)
  abstain     answer_status == abstained
  clarify     answer_status == needs_clarification
  conflict    verification_status == conflict (system reports the disagreement)
  hypothesis  causal_claim_status == hypothesis_only, or abstained (no causal claim made)
"""
from __future__ import annotations

import math
import random
from collections import Counter

from agentmamba.models.answerer import normalize_answer


def exact_match(pred: str | None, gold: str | None) -> float:
    return float(pred is not None and gold is not None and normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str | None, gold: str | None) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def is_correct(out: dict, ann) -> bool:
    k = ann.answer_kind
    if k == "value":
        return out["answer_status"] == "answered" and bool(exact_match(out.get("answer_value"), ann.expected_answer))
    if k == "abstain":
        return out["answer_status"] == "abstained"
    if k == "clarify":
        return out["answer_status"] == "needs_clarification"
    if k == "conflict":
        return out.get("verification_status") == "conflict"
    if k == "hypothesis":
        return out.get("causal_claim_status") == "hypothesis_only" or out["answer_status"] == "abstained"
    raise ValueError(k)


def rate(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return [round(means[int(alpha / 2 * n_boot)], 4), round(means[int((1 - alpha / 2) * n_boot) - 1], 4)]


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired per-question correctness."""
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"a_only": b01, "b_only": b10, "p_value": min(1.0, 2 * p)}
