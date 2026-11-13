"""Aggregate experiment runs into experiments/results/comparison.{json,md}.
Uses the most recent run directory of each method. Paired comparisons use exact McNemar
tests on per-question correctness (same test questions in every run).

    python scripts/evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import load_config, resolve_path
from agentmamba.evaluation.metrics import mcnemar_exact
from agentmamba.evaluation.reporting import markdown_table

PAIRS = [("E_agentmamba", "B_storeall_rag"), ("E_agentmamba", "D_learned_always"), ("D_learned_always", "C_heuristic_rag"),
         ("D_learned_always", "F_random_rag"), ("E_agentmamba", "A_no_retrieval"), ("E_agentmamba", "E_no_verification"),
         ("G_learned_nomamba", "H_learned_mamba"), ("E_agentmamba", "E_first_candidate"),
         ("B_storeall_rag", "B_storeall_vector_only"), ("B_storeall_rag", "B_storeall_no_verify"),
         ("E_agentmamba", "E_mlp_gate")]


def latest_runs(results: Path) -> dict[str, Path]:
    out = {}
    for d in sorted(results.glob("*_*")):
        s = d / "summary.json"
        if not s.exists():
            continue
        data = json.loads(s.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "method" in data and ":" not in data["method"]:   # test-split runs only
            out[data["method"]] = d          # sorted by timestamp prefix -> last wins
    return out


def row(name, s):
    a, e, c, v, m = s["answer"], s["evidence"], s["controller"], s["verification"], s["memory"]
    return {"method": name, "policy": s["spec"]["policy"], "budget": s["spec"]["budget"], "mode": s["spec"]["mode"],
            "verify": s["spec"]["verify"], "n": s["n"], "overall_accuracy": a["overall_accuracy"],
            "acc_ci95": a["overall_accuracy_ci95"], "exact_match": a["exact_match"], "token_f1": a["token_f1"],
            "cross_stage_accuracy": a["cross_stage_accuracy"], "temporal_version_accuracy": a["temporal_version_accuracy"],
            "false_answer_rate": a["false_answer_rate"], "unsupported_answer_rate": a["unsupported_answer_rate"],
            "abstention_accuracy": a["unanswerable_abstention_accuracy"], "retrieval_trigger_rate": c["retrieval_trigger_rate"],
            "unnecessary_retrieval": c["unnecessary_retrieval_rate"], "missed_retrieval": c["missed_retrieval_rate"],
            "recall_at_k": e["recall_at_k"], "mrr": e["mrr"], "evidence_f1": e["evidence_f1"],
            "relationship_recall": e["required_relationship_recall"],
            "unjustified_causal": v["unjustified_causal_claim_rate"], "incorrect_association": v["incorrect_association_rate"],
            "conflict_recall": v["conflict_detection_recall"], "retention_bytes": m["retention_ratio_bytes"],
            "relevant_evidence_retention": m["relevant_evidence_retention"], "rare_retention": m["rare_event_retention"],
            "useful_rare_retention": m["useful_rare_retention"], "retained_records": m["retained_records"],
            "sqlite_bytes": m["sqlite_file_bytes"], "faiss_bytes": m["faiss_file_bytes"],
            "abstain_when_lost": s["discarded_evidence"]["abstain_rate_when_evidence_lost"],
            "wrong_when_lost": s["discarded_evidence"]["wrong_answer_rate_when_evidence_lost"],
            "n_lost": s["discarded_evidence"]["n_value_questions_evidence_lost"],
            "latency_mean_s": s["latency"].get("total_s", {}).get("mean"),
            "latency_p95_s": s["latency"].get("total_s", {}).get("p95"),
            "gate_test_auc": (m.get("gate_test") or {}).get("roc_auc"),
            "gate_test_ap": (m.get("gate_test") or {}).get("average_precision")}


def main() -> None:
    cfg = load_config()
    results = resolve_path(cfg, "paths.results_dir")
    runs = latest_runs(results)
    summaries = {k: json.loads((d / "summary.json").read_text(encoding="utf-8")) for k, d in runs.items()}
    public = [row(k, s) for k, s in summaries.items() if k.startswith("public_")]
    summaries = {k: s for k, s in summaries.items() if not k.startswith("public_")}
    runs = {k: v for k, v in runs.items() if not k.startswith("public_")}
    methods = [row(k, s) for k, s in summaries.items() if not k.startswith("sweep_")]
    sweep = [row(k, s) | {"policy": k.split("_", 1)[1].rsplit("_", 1)[0]} for k, s in summaries.items() if k.startswith("sweep_")]
    correct = {}
    for k, d in runs.items():
        rows = [json.loads(line) for line in open(d / "rows.jsonl", encoding="utf-8")]
        correct[k] = {r["question_id"]: r["correct"] for r in rows}
    tests = []
    for a, b in PAIRS:
        if a in correct and b in correct:
            qids = sorted(set(correct[a]) & set(correct[b]))
            t = mcnemar_exact([correct[a][q] for q in qids], [correct[b][q] for q in qids])
            tests.append({"a": a, "b": b, "n": len(qids), **t})
    fam = {k: s["by_family"] for k, s in summaries.items() if not k.startswith("sweep_")}
    out = {"runs": {k: str(v) for k, v in runs.items()}, "methods": sorted(methods, key=lambda r: r["method"]),
           "sweep": sorted(sweep, key=lambda r: (r["policy"], r["budget"])), "paired_tests": tests, "by_family": fam,
           "public": public}
    (results / "comparison.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    cols = [("method", "method"), ("n", "n"), ("overall_accuracy", "acc"), ("acc_ci95", "acc 95% CI"),
            ("exact_match", "EM"), ("cross_stage_accuracy", "cross-stage"), ("false_answer_rate", "false ans"),
            ("unsupported_answer_rate", "unsupported"), ("retrieval_trigger_rate", "retrieve rate"),
            ("unnecessary_retrieval", "unnecessary"), ("missed_retrieval", "missed"), ("recall_at_k", "R@K"),
            ("unjustified_causal", "unjust. causal"), ("retention_bytes", "kept bytes"),
            ("relevant_evidence_retention", "evid. kept"), ("latency_mean_s", "lat mean s")]
    md = ["# Experiment comparison (synthetic benchmark, test split)", "",
          "All numbers are computed from saved predictions; synthetic results do not establish real-plant performance.", "",
          "## Methods", "", markdown_table(out["methods"], cols)]
    if sweep:
        md += ["## Memory-budget sweep (conditional mode, verification on)", "",
               markdown_table(out["sweep"], [("policy", "policy"), ("budget", "budget"), ("overall_accuracy", "acc"),
                                             ("exact_match", "EM"), ("relevant_evidence_retention", "evid. kept"),
                                             ("useful_rare_retention", "useful rare kept"), ("rare_retention", "rare kept"),
                                             ("abstain_when_lost", "abstain|lost"), ("wrong_when_lost", "wrong|lost"),
                                             ("retained_records", "records"), ("gate_test_auc", "gate AUC")])]
    if public:
        md += ["## Category A: real public records (UCI #851, templated date questions, test months Nov-Dec 2018)", "",
               markdown_table(public, [("method", "method"), ("n", "n"), ("overall_accuracy", "acc"),
                                       ("exact_match", "EM"), ("abstention_accuracy", "abstention acc"),
                                       ("recall_at_k", "R@K"), ("retrieval_trigger_rate", "retrieve rate")])]
    if tests:
        md += ["## Paired exact McNemar tests (per-question correctness)", "",
               markdown_table(tests, [("a", "A"), ("b", "B"), ("n", "n"), ("a_only", "A right, B wrong"),
                                      ("b_only", "B right, A wrong"), ("p_value", "p")])]
    (results / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
