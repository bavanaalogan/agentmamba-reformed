"""Category A experiment on REAL public data (UCI #851). Builds the public benchmark if
missing, stores all daily records (explicit store_all policy: the learned gate was trained on
synthetic event types and is not applied out of distribution), then answers the TEST-month
questions in conditional and no-retrieval modes.

    python scripts/run_public_experiment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import PROJECT_ROOT, load_config, resolve_path
from agentmamba.data.dataset_loader import load_dataset
from agentmamba.data.evidence_annotations import load_annotations
from agentmamba.data.public_benchmark import build
from agentmamba.evaluation.evaluator import score_row, summarize
from agentmamba.evaluation.reporting import new_run_dir, write_jsonl
from agentmamba.models.embedding_model import SentenceEmbedder
from agentmamba.pipeline.agent import QueryRequest
from agentmamba.pipeline.builder import build_agent, build_memory

RAW = PROJECT_ROOT / "data" / "raw" / "uci_851_steel_energy" / "Steel_industry_data.csv"
OUT = PROJECT_ROOT / "data" / "public" / "uci851"


def main() -> None:
    cfg = load_config()
    if not RAW.exists():
        sys.exit(f"missing {RAW}; download UCI #851 first (see docs/DATA_AUDIT.md)")
    if not (OUT / "split_manifest.json").exists():
        build(RAW, OUT, cfg["seed"])
    ds = load_dataset(OUT)
    qs = {q.question_id: q.split for q in ds.questions}
    anns = load_annotations(OUT, "test", "evaluation", qs)
    emb = SentenceEmbedder(cfg["models"]["embedding_id"])
    mem, rep = build_memory(cfg, PROJECT_ROOT / "storage" / "experiments" / "public_uci851", ds.events,
                            policy="store_all", budget_fraction=1.0, embedder=emb, feature_store=None,
                            run_tag="public:uci851")
    retained = {e.event_id for e in ds.events}
    agent = build_agent(cfg, mem)
    for mode in ("conditional", "never"):
        rows, preds = [], []
        for q in ds.questions_for("test"):
            out = agent.query(QueryRequest(q.text, [], mode=mode, query_id=q.question_id))
            preds.append(out)
            rows.append(score_row(out, anns[q.question_id], q, retained))
        s = summarize(rows)
        s.update({"method": f"public_uci851_{mode}", "spec": {"policy": "store_all", "budget": 1.0, "mode": mode,
                                                              "verify": True},
                  "memory": {"retention_ratio_bytes": 1.0, "relevant_evidence_retention": 1.0, "rare_event_retention": None,
                             "useful_rare_retention": None, "retained_records": len(retained),
                             "sqlite_file_bytes": mem.store.path.stat().st_size,
                             "faiss_file_bytes": mem.index_path.with_suffix(".faiss").stat().st_size},
                  "dataset": ds.manifest})
        d = new_run_dir(resolve_path(cfg, "paths.results_dir"), f"public_uci851_{mode}")
        write_jsonl(d / "predictions.jsonl", preds)
        write_jsonl(d / "rows.jsonl", rows)
        (d / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
        print(mode, json.dumps({"n": s["n"], "accuracy": s["answer"]["overall_accuracy"], "EM": s["answer"]["exact_match"],
                                "abstention_acc": s["answer"]["unanswerable_abstention_accuracy"],
                                "recall_at_k": s["evidence"]["recall_at_k"], "dir": str(d)}))


if __name__ == "__main__":
    main()
