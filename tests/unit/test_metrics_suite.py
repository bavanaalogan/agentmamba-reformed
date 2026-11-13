"""Metric definitions: text overlap, ranking, denominators, N/A handling and the composite score."""
from __future__ import annotations

import math

import pytest

from agentmamba.evaluation.metrics_suite import (ap_at_k, bleu1, comparison_table, composite, hit_at, metric_suite,
                                                 ndcg_at_k, recall_at, rouge1_f, rougeL_f)


def test_text_overlap_metrics_on_known_examples():
    assert rouge1_f("8.4 mm/s", "8.4 mm/s") == rougeL_f("8.4 mm/s", "8.4 mm/s") == bleu1("8.4 mm/s", "8.4 mm/s") == 1.0
    assert rouge1_f("the value 8.4", "8.4 mm/s") == pytest.approx(0.4)          # P=1/3, R=1/2
    assert rougeL_f("mm/s 8.4", "8.4 mm/s") == pytest.approx(0.5)              # LCS = 1 of 2
    assert bleu1("the value 8.4", "8.4 mm/s") == pytest.approx(1 / 3)          # longer candidate: no penalty
    assert bleu1("8.4", "8.4 mm/s") == pytest.approx(math.exp(1 - 2 / 1))      # brevity penalty
    assert rouge1_f("7.0 mm/s", "7 mm/s") == 1.0                               # numbers compared numerically
    assert rouge1_f("", "8.4 mm/s") == 0.0


def test_ranking_metrics_on_known_examples():
    ret, rel = ["a", "b", "c"], ["b"]
    assert ndcg_at_k(ret, rel, 10) == pytest.approx(1 / math.log2(3))
    assert ap_at_k(ret, rel, 10) == pytest.approx(0.5)
    assert hit_at(ret, rel, 1) == 0.0 and hit_at(ret, rel, 3) == 1.0
    assert recall_at(ret, rel, 1) == 0.0 and recall_at(ret, ["b", "z"], 10) == 0.5


def _row(kind, answered, correct, *, required=("e1",), retrieved=(), triggered=False, in_ctx=False, total=1.0,
         value="x", expected="x"):
    return {"answer_kind": kind, "answered": answered, "abstained": not answered, "correct": correct,
            "grounded_correct": correct and kind == "value", "unsupported": answered and not correct,
            "em": float(correct) if kind == "value" else None, "answer_value": value if answered else None,
            "expected": expected if kind == "value" else None, "required": list(required) if kind == "value" else [],
            "retrieved": list(retrieved), "retrieval_triggered": triggered,
            "evidence_in_context": in_ctx if kind == "value" else None,
            "evidence_covered": (in_ctx or set(required) <= set(retrieved)) if kind == "value" else None,
            "total_s": total, "generation_s": total, "retrieval_s": 0.1 if triggered else None,
            "verification_s": 0.01 if triggered else None}


def test_suite_denominators_and_not_applicable_metrics():
    rows = [_row("value", True, True, retrieved=["e1"], triggered=True),
            _row("value", True, False, value="y", retrieved=["zz"], triggered=True),
            _row("value", False, False, retrieved=[], triggered=True),
            _row("abstain", False, True)]
    s = metric_suite(rows, retrieves=True)
    assert s["errors"]["answer_error_rate"] == {**s["errors"]["answer_error_rate"], "value": round(2 / 3, 4), "n": 3}
    assert s["errors"]["false_answer_rate"]["value"] == round(1 / 3, 4)
    assert s["errors"]["abstention_error_rate"]["value"] == round(1 / 3, 4)
    assert s["errors"]["retrieval_failure_rate"]["value"] == round(2 / 3, 4)
    assert s["answer"]["correct_abstention_rate"] == {**s["answer"]["correct_abstention_rate"], "value": 1.0, "n": 1}
    assert s["retrieval"]["hit@1"]["value"] == round(1 / 3, 4)
    assert s["errors"]["brier"]["value"] is None                      # no calibrated confidence -> N/A, not 0
    assert s["efficiency"]["p95_s"]["value"] is None                  # fewer than 20 observations
    b = metric_suite(rows, retrieves=False)
    assert all(b["retrieval"][k]["value"] is None for k in ("ndcg@10", "hit@1", "retrieval_trigger_rate"))
    assert b["errors"]["retrieval_failure_rate"]["value"] is None
    assert b["efficiency"]["retrieval_mean_s"]["value"] is None


def test_composite_drops_missing_components_instead_of_scoring_zero():
    rows = [_row("value", True, True, retrieved=["e1"], triggered=True), _row("abstain", False, True)]
    base = metric_suite(rows, retrieves=False)
    enh = metric_suite(rows, retrieves=True)
    cb, ce = composite(base), composite(enh)
    assert cb["components"]["retrieval"] is None and "retrieval" not in cb["components_used_full"]
    assert cb["composite_full"] == cb["composite_shared"] > 0.5        # baseline not penalized for not retrieving
    assert ce["components_used_shared"] == ["answer_quality", "grounding", "efficiency"]
    assert 0 <= ce["composite_full"] <= 1
    table = comparison_table(base, enh)
    r = next(t for t in table if t["metric"] == "ndcg@10")
    assert r["baseline"] is None and r["difference"] is None
