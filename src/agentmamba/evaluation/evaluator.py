"""Scores pipeline outputs against gold annotations (evaluation only)."""
from __future__ import annotations

from collections import defaultdict

from agentmamba.evaluation.evidence_metrics import precision_at_k, prf, recall_at_k, reciprocal_rank
from agentmamba.evaluation.latency_metrics import summarize_timings
from agentmamba.evaluation.metrics import bootstrap_ci, exact_match, is_correct, rate, token_f1


def score_row(out: dict, ann, q, retained: set[str]) -> dict:
    required = ann.required_evidence_ids
    available = set(retained) | set(q.initial_context_ids)
    cited = [e["record_id"] for e in out.get("evidence", [])]
    answer_rec = next((e["record_id"] for e in out.get("evidence", []) if e.get("role") == "answer"), None)
    rels = {r["rel_id"] for r in (out.get("verification") or {}).get("relationships_used", [])} | \
           {r["rel_id"] for r in out.get("relationships_traversed", [])}
    answered = out["answer_status"] == "answered"
    correct = is_correct(out, ann)
    init_em = exact_match(out.get("initial_answer"), ann.expected_answer) if ann.answer_kind == "value" else None
    return {
        "question_id": q.question_id, "family": ann.family, "answer_kind": ann.answer_kind,
        "cross_stage": ann.cross_stage, "temporal_sensitive": ann.temporal_sensitive,
        "expected": ann.expected_answer, "answer_value": out.get("answer_value"), "answer_status": out["answer_status"],
        "verification_status": out.get("verification_status"), "causal_claim_status": out.get("causal_claim_status"),
        "correct": correct, "em": exact_match(out.get("answer_value"), ann.expected_answer) if ann.answer_kind == "value" else None,
        "f1": token_f1(out.get("answer_value"), ann.expected_answer) if ann.answer_kind == "value" else None,
        "initial_em": init_em, "initial_status": out.get("initial_status"),
        "controller_action": out["controller"]["action"], "retrieval_triggered": out["retrieval_triggered"],
        "initial_sufficient": ann.initial_sufficient, "retrieval_required": ann.retrieval_required,
        "required_evidence": required, "retrieved": out.get("retrieved_record_ids", []), "cited": cited,
        "answer_record": answer_rec,
        "evidence_available": all(r in available for r in required) if required else None,
        "recall_at_k": recall_at_k(out.get("retrieved_record_ids", []), required) if out["retrieval_triggered"] else None,
        "precision_at_k": precision_at_k(out.get("retrieved_record_ids", []), required) if out["retrieval_triggered"] else None,
        "rr": reciprocal_rank(out.get("retrieved_record_ids", []), required) if out["retrieval_triggered"] else None,
        "relationship_recall": (len(rels & set(ann.relationship_ids)) / len(ann.relationship_ids)) if ann.relationship_ids else None,
        "answered": answered, "invalid_citations": len(out.get("invalid_citations", [])),
        "timings": out.get("timings", {}),
    }


def _mean(rows, key, cond=lambda r: True):
    vals = [float(r[key]) for r in rows if cond(r) and r[key] is not None]
    return (sum(vals) / len(vals)) if vals else None


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    val = [r for r in rows if r["answer_kind"] == "value"]
    unans = [r for r in rows if r["answer_kind"] in ("abstain", "conflict")]
    answered = [r for r in rows if r["answered"]]
    abst = [r for r in rows if r["answer_status"] == "abstained"]
    wrong_answer = [r for r in answered if not (r["answer_kind"] == "value" and r["em"]) and r["answer_kind"] != "hypothesis"]
    unsupported = [r for r in answered if r["answer_kind"] != "value" and r["answer_kind"] != "hypothesis"
                   or (r["answer_kind"] == "value" and r["answer_record"] not in r["required_evidence"])]
    triggered = [r for r in rows if r["retrieval_triggered"]]
    suff = [r for r in rows if r["initial_sufficient"]]
    req = [r for r in rows if r["retrieval_required"]]
    hyp = [r for r in rows if r["answer_kind"] == "hypothesis"]
    gold_conf = [r for r in rows if r["answer_kind"] == "conflict"]
    pred_conf = [r for r in rows if r["verification_status"] == "conflict"]
    lost = [r for r in val if r["evidence_available"] is False]
    kept = [r for r in val if r["evidence_available"]]
    correct = [float(r["correct"]) for r in rows]
    s = {
        "n": n,
        "answer": {
            "overall_accuracy": rate(sum(correct), n), "overall_accuracy_ci95": bootstrap_ci(correct),
            "exact_match": _mean(val, "em"), "exact_match_ci95": bootstrap_ci([r["em"] for r in val]),
            "token_f1": _mean(val, "f1"), "n_value": len(val),
            "answerable_accuracy": _mean(val, "correct"),
            "unanswerable_abstention_accuracy": rate(sum(r["answer_status"] == "abstained" for r in unans), len(unans)),
            "false_answer_rate": rate(len(wrong_answer), n),
            "unsupported_answer_rate": rate(len(unsupported), len(answered)),
            "correct_abstention_rate": rate(sum(r["correct"] for r in abst), len(abst)),
            "abstention_rate": rate(len(abst), n),
            "partial_answer_rate": rate(sum(r["verification_status"] == "partial" for r in answered), len(answered)),
            "cross_stage_accuracy": _mean(rows, "correct", lambda r: r["cross_stage"]),
            "n_cross_stage": sum(r["cross_stage"] for r in rows),
            "temporal_version_accuracy": _mean(rows, "correct", lambda r: r["temporal_sensitive"]),
            "clarification_accuracy": _mean(rows, "correct", lambda r: r["answer_kind"] == "clarify"),
        },
        "evidence": {
            "recall_at_k": _mean(triggered, "recall_at_k"), "precision_at_k": _mean(triggered, "precision_at_k"),
            "mrr": _mean(triggered, "rr"), "required_relationship_recall": _mean(rows, "relationship_recall"),
            **dict(zip(("evidence_precision", "evidence_recall", "evidence_f1"), _evidence_prf(answered))),
            "invalid_citations": sum(r["invalid_citations"] for r in rows),
        },
        "controller": {
            "retrieval_trigger_rate": rate(len(triggered), n),
            "unnecessary_retrieval_rate": rate(sum(r["retrieval_triggered"] for r in suff), len(suff)),
            "correctly_avoided_retrieval_rate": rate(sum(not r["retrieval_triggered"] for r in suff), len(suff)),
            "missed_retrieval_rate": rate(sum(not r["retrieval_triggered"] and r["controller_action"] != "ABSTAIN_OR_CLARIFY"
                                              for r in req), len(req)),
            "initial_support_rate": rate(sum(r["initial_status"] == "supported" for r in rows), n),
            "final_improvement_after_retrieval": _improvement(triggered),
            "n_initial_sufficient": len(suff), "n_retrieval_required": len(req),
        },
        "verification": {
            "evidence_grounding_rate": rate(sum(r["answer_record"] in r["required_evidence"] for r in answered
                                                if r["answer_kind"] == "value"), sum(r["answer_kind"] == "value" for r in answered)),
            "unjustified_causal_claim_rate": rate(sum(r["causal_claim_status"] in ("documented_by_source", "claimed_unverified")
                                                      for r in hyp), len(hyp)),
            "incorrect_association_rate": rate(sum(r["answer_kind"] == "value" and not r["em"] and r["answer_record"] is not None
                                                   and r["answer_record"] not in r["required_evidence"] for r in answered),
                                               len(answered)),
            "conflict_detection_precision": rate(sum(r["answer_kind"] == "conflict" for r in pred_conf), len(pred_conf)),
            "conflict_detection_recall": rate(sum(r["verification_status"] == "conflict" for r in gold_conf), len(gold_conf)),
        },
        "discarded_evidence": {
            "n_value_questions_evidence_lost": len(lost),
            "abstain_rate_when_evidence_lost": rate(sum(r["answer_status"] == "abstained" for r in lost), len(lost)),
            "wrong_answer_rate_when_evidence_lost": rate(sum(r["answered"] and not r["em"] for r in lost), len(lost)),
            "em_when_evidence_available": _mean(kept, "em"),
        },
        "latency": summarize_timings([r["timings"] for r in rows]),
        "by_family": _by(rows, "family"),
    }
    return s


def _evidence_prf(answered):
    ps, rs, fs = [], [], []
    for r in answered:
        if r["answer_kind"] != "value":
            continue
        p, rc, f = prf(r["cited"], r["required_evidence"])
        for lst, v in ((ps, p), (rs, rc), (fs, f)):
            if v is not None:
                lst.append(v)
    return [sum(x) / len(x) if x else None for x in (ps, rs, fs)]


def _improvement(triggered):
    vals = [r for r in triggered if r["answer_kind"] == "value"]
    if not vals:
        return None
    return sum(float(bool(r["em"])) - float(bool(r["initial_em"])) for r in vals) / len(vals)


def _by(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {k: {"n": len(v), "accuracy": round(sum(r["correct"] for r in v) / len(v), 4),
                "retrieval_rate": round(sum(r["retrieval_triggered"] for r in v) / len(v), 4)}
            for k, v in sorted(groups.items())}
