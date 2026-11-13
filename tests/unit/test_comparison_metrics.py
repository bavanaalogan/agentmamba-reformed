from __future__ import annotations

from app.comparison_views import _comparison_metrics


def test_comparison_metrics_score_answers_and_retrieval_without_filling_unavailable_values():
    baseline = {
        "answer_status": "answered",
        "answer_value": "alarm cleared at 14:15",
        "retrieval_triggered": False,
        "retrieved_record_ids": [],
        "confidence": {"mean_component_candidate_share": 0.82},
        "verification_status": "not_verified",
        "timings": {"total_s": 1.2, "generation_s": 1.0},
        "resources": {"cpu_s": 0.9},
    }
    enhanced = {
        "answer_status": "answered",
        "answer_value": "alarm cleared at 14:15",
        "retrieval_triggered": True,
        "retrieved_record_ids": ["E1", "N1"],
        "confidence": {"calibrated": 0.91},
        "verification_status": "supported",
        "verification": {"candidate_checks": [
            {"question_part": "alarm state", "part_status": "supported"},
            {"question_part": "clearance time", "part_status": "supported"},
        ]},
        "timings": {"total_s": 2.2, "generation_s": 1.1, "retrieval_s": 0.4},
        "resources": {"cpu_s": 1.9},
    }

    metrics = _comparison_metrics({"baseline": baseline, "enhanced": enhanced},
                                  "alarm cleared at 14:15", {"E1"})
    values = metrics.set_index("Metric")

    assert values.loc["Exact match", "Baseline Mamba"] == "1.0000"
    assert values.loc["Answer error (1 - exact match)", "AgentMamba"] == "0.0000"
    assert values.loc["ROUGE-L F", "AgentMamba"] == "1.0000"
    assert values.loc["NDCG@10", "Baseline Mamba"] == "N/A (no retrieval)"
    assert values.loc["Recall@10", "AgentMamba"] == "1.0000"
    assert values.loc["Answer confidence score", "Baseline Mamba"] == "0.8200"
    assert values.loc["Answer confidence score", "AgentMamba"] == "0.9100"
    assert values.loc["Verified answer-component coverage", "AgentMamba"] == "2/2 (1.0000)"


def test_comparison_metrics_mark_reference_and_retrieval_metrics_unavailable():
    output = {"answer_status": "answered", "answer_value": "an answer", "retrieval_triggered": False,
              "confidence": {}, "verification_status": "not_verified", "timings": {}, "resources": {}}

    metrics = _comparison_metrics({"baseline": output, "enhanced": output}, None, set())

    answer_metrics = metrics[metrics["Category"] == "Answer quality"]
    retrieval_metrics = metrics[metrics["Category"] == "Retrieval quality"]
    assert set(answer_metrics["Baseline Mamba"]) == {"N/A"}
    assert set(retrieval_metrics["AgentMamba"]) == {"N/A"}
