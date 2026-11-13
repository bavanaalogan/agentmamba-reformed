"""Cold-cache latency of Baseline Mamba vs AgentMamba (no Mamba score cache, sequential, one process).

    python scripts/benchmark_compare_latency.py --run <context_growth_test_run_dir> --stage 4 --n 15

Uses the per-stage memory snapshots written by run_context_growth.py, the same shared 768-token
initial context, and the first ``--n`` test questions in file order (not selected by result).
Every Mamba call is computed (no cache), so latencies are real CPU costs on this machine.
Writes <run_dir>/latency_cold.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import PROJECT_ROOT, load_config  # noqa: E402
from agentmamba.data.document_loader import load_any  # noqa: E402
from agentmamba.data.log_benchmark import load_questions, stage_files  # noqa: E402
from agentmamba.memory.episodic_memory import EpisodicMemory  # noqa: E402
from agentmamba.models.answerer import MambaAnswerer  # noqa: E402
from agentmamba.models.embedding_model import SentenceEmbedder  # noqa: E402
from agentmamba.models.mamba_backbone import MambaBackbone  # noqa: E402
from agentmamba.pipeline.agent import QueryRequest  # noqa: E402
from agentmamba.pipeline.baseline import BaselineMamba, build_window  # noqa: E402
from agentmamba.pipeline.builder import build_agent  # noqa: E402
from agentmamba.reproducibility import disable_background_throttling, environment_info  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    print("process:", disable_background_throttling(), flush=True)
    run = Path(args.run)
    cfg = load_config(overrides={"models.mamba_threads": args.threads})
    rc = json.loads((run / "config.json").read_text(encoding="utf-8"))
    bench = PROJECT_ROOT / "data" / "log_benchmark" / "v1"
    m = cfg["models"]
    backbone = MambaBackbone(m["mamba_id"], m["mamba_threads"], m["mamba_max_tokens"], cache_path=None)
    t = time.perf_counter()
    backbone.load()
    load_s = time.perf_counter() - t
    emb = SentenceEmbedder(m["embedding_id"])
    qs = load_questions(bench, rc["split"])[: args.n]
    store_root = PROJECT_ROOT / "storage" / "experiments" / "context_growth" / run.name
    rows, cache = [], {}
    for q in qs:
        if q.scenario not in cache:
            mem = EpisodicMemory(store_root / q.scenario / f"stage{args.stage}", emb)
            agent = build_agent(cfg, mem, MambaAnswerer(backbone))
            hist, seen = [], set()
            for s in range(1, args.stage + 1):
                for f in stage_files(bench / rc["split"] / q.scenario, s):
                    for e in load_any(bench / rc["split"] / q.scenario / f["file"], stage=f["stage"]):
                        if e.event_id not in seen:
                            seen.add(e.event_id)
                            hist.append(e)
            cache[q.scenario] = (agent, hist)
        agent, hist = cache[q.scenario]
        win = build_window(q.text, [], hist, backbone.count_tokens, 768)
        for name, budget in (("baseline_mamba_768", 768), ("baseline_mamba_2048", 2048)):
            w = win if budget == 768 else build_window(q.text, [], hist, backbone.count_tokens, budget)
            out = BaselineMamba(agent.qp, MambaAnswerer(backbone, None, budget), backbone.count_tokens,
                                budget).answer(q.text, w, q.question_id)
            rows.append({"system": name, "question_id": q.question_id, **out["timings"], **out["resources"],
                         "prompt_tokens": out["model_call"].get("prompt_tokens")})
        out = agent.query(QueryRequest(q.text, win.records, "conditional", verification=True))
        rows.append({"system": "agentmamba", "question_id": q.question_id, **out["timings"], **out["resources"],
                     "retrieval_triggered": out["retrieval_triggered"],
                     "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in out.get("model_calls", []))})
        print(q.question_id, [(r["system"], r["total_s"]) for r in rows[-3:]], flush=True)
    summ = {}
    for name in ("baseline_mamba_768", "baseline_mamba_2048", "agentmamba"):
        rs = [r for r in rows if r["system"] == name]
        tot = [r["total_s"] for r in rs]
        summ[name] = {"n": len(rs), "total_s_median": round(statistics.median(tot), 3),
                      "total_s_mean": round(statistics.mean(tot), 3), "total_s_max": round(max(tot), 3),
                      "cpu_s_median": round(statistics.median(r["cpu_s"] for r in rs), 3),
                      "rss_mb_max": max(r["rss_mb"] or 0 for r in rs)}
        for k in ("retrieval_s", "verification_s", "final_answer_s", "initial_answer_s", "generation_s"):
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals:
                summ[name][f"{k}_median"] = round(statistics.median(vals), 4)
    res = {"stage": args.stage, "n_questions": len(qs), "mamba_load_s": round(load_s, 2), "threads": args.threads,
           "score_cache": "disabled", "environment": environment_info(), "summary": summ, "rows": rows,
           "note": "sequential, no other experiment running; first n questions of the split in file order"}
    (run / "latency_cold.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
