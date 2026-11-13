"""Scoring and comparison for the Baseline-Mamba vs AgentMamba experiments (evaluation only).

Answer matching (documented normalization):
* text is lower-cased, commas removed, whitespace collapsed (``normalize_answer``); numbers are
  compared numerically ("7.0 mm/s" == "7 mm/s");
* ``exact``    - the normalized answer equals the normalized gold answer;
* ``contains`` - the normalized gold phrase occurs in the normalized answer (free-text quotes).
  Containment can over-credit a long answer; quoted answers are capped at 400 characters, and
  token F1 and grounded accuracy are reported alongside so this is visible.

Per-question correctness by gold ``answer_kind``:
  value    answered and matching;           grounded = also cites a required evidence record
  abstain  answer_status == abstained       (baseline "no candidate" counts, but is flagged)
  conflict verification_status == conflict  (reports the disagreement instead of a value)
"""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict

from agentmamba.evaluation.evidence_metrics import precision_at_k, recall_at_k, reciprocal_rank
from agentmamba.evaluation.metrics import bootstrap_ci, mcnemar_exact, token_f1
from agentmamba.models.answerer import normalize_answer

_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")


def norm(s: str | None) -> str:
    toks = normalize_answer(s).split()
    return " ".join(f"{float(t):g}" if _NUM.match(t) else t for t in toks)


def matches(pred: str | None, gold: str | None, how: str | None) -> bool:
    if pred is None or gold is None:
        return False
    p, g = norm(pred), norm(gold)
    return p == g if how == "exact" else (g in p if g else False)


def score_reference(out: dict | None, expected: str, required_evidence_ids: set[str] | None = None) -> dict | None:
    """Score one output against a user-supplied reference, without changing inference."""
    if out is None:
        return None
    required = required_evidence_ids or set()
    answer = out.get("answer_value")
    answered = out.get("answer_status") == "answered"
    exact = answered and matches(answer, expected, "exact")
    cited = {e.get("record_id") for e in out.get("evidence", []) if e.get("record_id")}
    available = (set(out.get("context", {}).get("record_ids", []))
                 | set(out.get("initial_context_ids", []))
                 | set(out.get("retrieved_record_ids", [])))
    return {
        "correct": bool(exact),
        "grounded_correct": bool(exact and required <= cited) if required else None,
        "evidence_covered": bool(required <= available) if required else None,
        "unsupported": bool(answered and not required <= cited) if required else None,
        "em": float(exact),
        "contains": float(answered and matches(answer, expected, "contains")),
        "f1": token_f1(answer, expected),
    }


def score_output(out: dict, ann, *, system: str, stage: int, context_ids: list[str], n_available: int) -> dict:
    """One scored row. ``out`` is a finished system output; ``ann`` is read only here."""
    ev = out.get("evidence", [])
    answer_rec = next((e["record_id"] for e in ev if e.get("role") == "answer"), None)
    cited = [e["record_id"] for e in ev]
    req = list(ann.required_evidence_ids)
    answered = out["answer_status"] == "answered"
    abstained = out["answer_status"] == "abstained"
    kind = ann.answer_kind
    if kind == "value":
        correct = answered and matches(out.get("answer_value"), ann.expected_answer, ann.match)
        grounded = correct and answer_rec in req
    elif kind == "abstain":
        correct = grounded = abstained
    else:
        correct = grounded = out.get("verification_status") == "conflict"
    retrieved = out.get("retrieved_record_ids", [])
    seen = set(context_ids) | set(retrieved)
    conf = out.get("confidence") or {}
    calls = out.get("model_calls") or ([out["model_call"]] if out.get("model_call") else [])
    t = out.get("timings", {})
    return {
        "question_id": ann.question_id, "scenario": ann.scenario, "split": ann.split, "stage": stage,
        "family": ann.family, "answer_kind": kind, "system": system,
        "answer_status": out["answer_status"], "answer_value": out.get("answer_value"),
        "expected": ann.expected_answer, "verification_status": out.get("verification_status"),
        "no_candidate": bool(out.get("no_candidate")), "correct": bool(correct), "grounded_correct": bool(grounded),
        "em": float(matches(out.get("answer_value"), ann.expected_answer, "exact")) if kind == "value" else None,
        "contains": float(matches(out.get("answer_value"), ann.expected_answer, "contains")) if kind == "value" else None,
        "f1": token_f1(out.get("answer_value"), ann.expected_answer) if kind == "value" else None,
        "answered": answered, "abstained": abstained,
        "unsupported": answered and (kind != "value" or answer_rec not in req),
        "answer_record": answer_rec, "cited": cited, "required": req,
        "evidence_in_context": all(r in context_ids for r in req) if req else None,
        "evidence_covered": all(r in seen for r in req) if req else None,
        "retrieval_triggered": bool(out.get("retrieval_triggered")),
        "retrieved": retrieved,
        "recall_at_k": recall_at_k(retrieved, req) if out.get("retrieval_triggered") and req else None,
        "precision_at_k": precision_at_k(retrieved, req) if out.get("retrieval_triggered") and req else None,
        "rr": reciprocal_rank(retrieved, req) if out.get("retrieval_triggered") and req else None,
        "candidate_share": conf.get("candidate_share"),
        "retrieval_max_similarity": conf.get("retrieval_max_similarity"),
        "n_context_records": len(context_ids), "n_records_available": n_available,
        "total_s": t.get("total_s"), "generation_s": t.get("generation_s", (t.get("initial_answer_s") or 0)
                                                           + (t.get("final_answer_s") or 0)),
        "retrieval_s": t.get("retrieval_s"), "verification_s": t.get("verification_s"),
        "model_cache_hit": any(c.get("cache_hit") for c in calls) if calls else None,
        "all_model_calls_cold": all(c.get("cache_hit") is False for c in calls) if calls else None,
        "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in calls if c.get("prompt_tokens")) or None,
        "cpu_s": (out.get("resources") or {}).get("cpu_s"), "rss_mb": (out.get("resources") or {}).get("rss_mb"),
    }


def _rate(n, d):
    return None if d == 0 else n / d


def _mean(vals):
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _median(vals):
    vals = sorted(float(v) for v in vals if v is not None)
    if not vals:
        return None
    m = len(vals) // 2
    return vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2


def summarize_rows(rows: list[dict]) -> dict:
    n = len(rows)
    val = [r for r in rows if r["answer_kind"] == "value"]
    unans = [r for r in rows if r["answer_kind"] == "abstain"]
    ans = [r for r in rows if r["answered"]]
    abst = [r for r in rows if r["abstained"]]
    trig = [r for r in rows if r["retrieval_triggered"] and r["required"]]
    cold = [r for r in rows if r["all_model_calls_cold"]]
    corr = [float(r["correct"]) for r in rows]
    return {
        "n": n, "n_value": len(val), "n_abstain_gold": len(unans),
        "n_conflict_gold": sum(r["answer_kind"] == "conflict" for r in rows),
        "accuracy": _rate(sum(corr), n), "accuracy_ci95": bootstrap_ci(corr),
        "grounded_accuracy": _rate(sum(r["grounded_correct"] for r in rows), n),
        "answerable_accuracy": _mean([r["correct"] for r in val]),
        "answerable_grounded_accuracy": _mean([r["grounded_correct"] for r in val]),
        "exact_match": _mean([r["em"] for r in val]), "contains_match": _mean([r["contains"] for r in val]),
        "token_f1": _mean([r["f1"] for r in val]),
        "correct_answer_rate": _rate(sum(r["answered"] and r["correct"] for r in rows), n),
        "incorrect_answer_rate": _rate(sum(r["answered"] and not r["correct"] for r in rows), n),
        "correct_abstention_rate": _rate(sum(r["abstained"] for r in unans), len(unans)),
        "abstention_precision": _rate(sum(r["answer_kind"] != "value" for r in abst), len(abst)),
        "no_candidate_rate": _rate(sum(r["no_candidate"] for r in rows), n),
        "conflict_detection_rate": _mean([r["verification_status"] == "conflict" for r in rows
                                          if r["answer_kind"] == "conflict"]),
        "unsupported_claim_rate": _rate(sum(r["unsupported"] for r in ans), len(ans)),
        "unsupported_claims_per_question": _rate(sum(r["unsupported"] for r in rows), n),
        "evidence_in_context_rate": _mean([r["evidence_in_context"] for r in rows if r["required"]]),
        "required_evidence_coverage": _mean([r["evidence_covered"] for r in rows if r["required"]]),
        "retrieval_trigger_rate": _rate(sum(r["retrieval_triggered"] for r in rows), n),
        "recall_at_k": _mean([r["recall_at_k"] for r in trig]), "precision_at_k": _mean([r["precision_at_k"] for r in trig]),
        "mrr": _mean([r["rr"] for r in trig]),
        "latency_total_s_median_cold": _median([r["total_s"] for r in cold]),
        "latency_total_s_mean_cold": _mean([r["total_s"] for r in cold]), "n_cold": len(cold),
        "latency_total_s_median_all": _median([r["total_s"] for r in rows]),
        "generation_s_median_cold": _median([r["generation_s"] for r in cold]),
        "retrieval_s_median": _median([r["retrieval_s"] for r in rows if r["retrieval_s"] is not None]),
        "verification_s_median": _median([r["verification_s"] for r in rows if r["verification_s"] is not None]),
        "prompt_tokens_median": _median([r["prompt_tokens"] for r in rows]),
        "cpu_s_median": _median([r["cpu_s"] for r in rows]),
        "rss_mb_max": max((r["rss_mb"] for r in rows if r["rss_mb"] is not None), default=None),
        "records_available_mean": _mean([r["n_records_available"] for r in rows]),
        "context_records_mean": _mean([r["n_context_records"] for r in rows]),
    }


# ---------------------------------------------------------------- paired comparison
LOWER_IS_BETTER = {"incorrect_answer_rate", "unsupported_claim_rate", "unsupported_claims_per_question",
                   "latency_total_s_median_cold"}


def paired(rows_a: list[dict], rows_b: list[dict], key: str = "correct", n_boot: int = 2000, seed: int = 0) -> dict:
    """B minus A on per-question binary outcome ``key`` (same question/stage pairs only):
    difference, bootstrap 95% CI over pairs, and exact McNemar p-value."""
    ia = {(r["question_id"], r["stage"]): r for r in rows_a}
    pairs = [(bool(ia[k][key]), bool(r[key])) for r in rows_b if (k := (r["question_id"], r["stage"])) in ia]
    if not pairs:
        return {"n_pairs": 0}
    d = [float(b) - float(a) for a, b in pairs]
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    mc = mcnemar_exact([a for a, _ in pairs], [b for _, b in pairs])
    a_rate, b_rate = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    return {"n_pairs": n, "a": round(a_rate, 4), "b": round(b_rate, 4), "diff_b_minus_a": round(b_rate - a_rate, 4),
            "relative_change": round((b_rate - a_rate) / a_rate, 4) if a_rate > 0 else None,
            "diff_ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot) - 1], 4)],
            "mcnemar_p": round(mc["p_value"], 6), "a_only": mc["a_only"], "b_only": mc["b_only"],
            "note": "questions within a scenario are not independent; CIs/p-values treat pairs as independent"}


# ---------------------------------------------------------------- calibration
def brier(conf: list[float], y: list[bool]) -> float | None:
    if not conf:
        return None
    return sum((c - float(t)) ** 2 for c, t in zip(conf, y)) / len(conf)


def reliability(conf: list[float], y: list[bool], n_bins: int = 10) -> list[dict]:
    bins = defaultdict(list)
    for c, t in zip(conf, y):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, float(t)))
    return [{"bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}", "n": len(v),
             "mean_confidence": round(sum(c for c, _ in v) / len(v), 4),
             "accuracy": round(sum(t for _, t in v) / len(v), 4)} for b, v in sorted(bins.items())]


def ece(conf: list[float], y: list[bool], n_bins: int = 10) -> float | None:
    if not conf:
        return None
    return sum(b["n"] / len(conf) * abs(b["mean_confidence"] - b["accuracy"]) for b in reliability(conf, y, n_bins))


def calib_features(row: dict) -> list[float]:
    """Features available at answer time (no gold): Mamba candidate share, verification outcome,
    whether retrieval ran. Documented; identical for every system."""
    s = row.get("candidate_share")
    return [s if s is not None else 0.5, 1.0 if s is None else 0.0,
            1.0 if row.get("verification_status") == "supported" else 0.0,
            1.0 if row.get("verification_status") == "partial" else 0.0,
            1.0 if row.get("retrieval_triggered") else 0.0]


def fit_calibrator(rows: list[dict]) -> dict | None:
    """Logistic regression P(correct | answered, features) on DEV rows of one system."""
    rows = [r for r in rows if r["answered"]]
    y = [int(r["correct"]) for r in rows]
    if len(rows) < 10 or len(set(y)) < 2:
        return {"available": False, "reason": f"{len(rows)} answered dev rows, classes={sorted(set(y))}"}
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(max_iter=2000, C=1.0).fit([calib_features(r) for r in rows], y)
    return {"available": True, "coef": [float(c) for c in m.coef_[0]], "intercept": float(m.intercept_[0]),
            "n_dev": len(rows), "dev_positive_rate": round(sum(y) / len(y), 4),
            "features": ["candidate_share", "share_missing", "verified_supported", "verified_partial",
                         "retrieval_triggered"], "method": "logistic regression on dev split (Platt-style)"}


def apply_calibrator(cal: dict | None, row: dict) -> float | None:
    if not cal or not cal.get("available") or not row.get("answered"):
        return None
    z = cal["intercept"] + sum(c * x for c, x in zip(cal["coef"], calib_features(row)))
    return 1 / (1 + math.exp(-z))


def output_features(out: dict) -> dict:
    """The answer-time fields ``calib_features`` needs, taken from a live system output."""
    return {"candidate_share": (out.get("confidence") or {}).get("candidate_share"),
            "verification_status": out.get("verification_status"),
            "retrieval_triggered": bool(out.get("retrieval_triggered")),
            "answered": out.get("answer_status") == "answered"}


def calibration_report(rows: list[dict], key: str) -> dict:
    pts = [(r[key], r["correct"]) for r in rows if r["answered"] and r.get(key) is not None]
    conf, y = [p[0] for p in pts], [p[1] for p in pts]
    return {"confidence": key, "n_answered_with_confidence": len(pts),
            "brier": None if not pts else round(brier(conf, y), 4), "ece": None if not pts else round(ece(conf, y), 4),
            "mean_confidence": None if not pts else round(sum(conf) / len(conf), 4),
            "accuracy_when_answered": None if not pts else round(sum(y) / len(y), 4),
            "reliability": reliability(conf, y) if pts else []}
