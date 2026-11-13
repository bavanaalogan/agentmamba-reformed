"""Cold-cache latency benchmark. Experiment runs memoize Mamba scores, so their latencies
mostly measure cache hits; this script disables the score cache and times a fixed, seeded
sample of test questions per configuration on the already-built experiment memories.

    python scripts/benchmark_latency.py --n 40
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psutil

from agentmamba.config import PROJECT_ROOT, load_config, resolve_path
from agentmamba.data.dataset_loader import load_dataset
from agentmamba.evaluation.latency_metrics import summarize_timings
from agentmamba.evaluation.reporting import new_run_dir
from agentmamba.memory.episodic_memory import EpisodicMemory
from agentmamba.models.answerer import MambaAnswerer
from agentmamba.models.embedding_model import SentenceEmbedder
from agentmamba.models.mamba_backbone import MambaBackbone
from agentmamba.pipeline.agent import QueryRequest
from agentmamba.pipeline.builder import build_agent

CONFIGS = [("E_conditional_learned10", "test_learned_logreg_mamba_0.10", "conditional"),
           ("D_always_learned10", "test_learned_logreg_mamba_0.10", "always"),
           ("B_always_storeall", "test_store_all_none_1.00", "always"),
           ("B_conditional_storeall", "test_store_all_none_1.00", "conditional")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data" / "synthetic" / "main"))
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    cfg = load_config()
    ds = load_dataset(args.data)
    ev = ds.event_index()
    qs = random.Random(cfg["seed"]).sample(ds.questions_for("test"), args.n)
    emb = SentenceEmbedder(cfg["models"]["embedding_id"])
    mb = MambaBackbone(cfg["models"]["mamba_id"], cfg["models"]["mamba_threads"], cfg["models"]["mamba_max_tokens"],
                       cache_path=None).load()            # no score cache: every call recomputed
    proc = psutil.Process()
    out = {"n_questions": args.n, "mamba_load_s": round(mb.load_seconds, 2), "configs": {}}
    for name, mem_dir, mode in CONFIGS:
        mem = EpisodicMemory(PROJECT_ROOT / "storage" / "experiments" / mem_dir, emb)
        agent = build_agent(cfg, mem, MambaAnswerer(mb))
        timings, per_q, t0 = [], [], time.perf_counter()
        for q in qs:
            r = agent.query(QueryRequest(q.text, [ev[i] for i in q.initial_context_ids], mode=mode))
            timings.append(r["timings"])
            per_q.append({"question_id": q.question_id, "total_s": r["timings"]["total_s"],
                          "retrieved": r["retrieval_triggered"], "n_candidates": len(r.get("final_candidates", []))})
        out["configs"][name] = {"mode": mode, "memory": mem_dir, "wall_s": round(time.perf_counter() - t0, 1),
                                "rss_mb": round(proc.memory_info().rss / 2**20), "latency": summarize_timings(timings),
                                "per_question": per_q}
        print(name, json.dumps(out["configs"][name]["latency"].get("total_s")))
    d = new_run_dir(resolve_path(cfg, "paths.results_dir"), "latency_benchmark")
    (d / "summary.json").write_text(json.dumps({"kind": "latency_benchmark", **out}, indent=2), encoding="utf-8")
    print(d)


if __name__ == "__main__":
    main()
