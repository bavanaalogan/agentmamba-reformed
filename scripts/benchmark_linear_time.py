"""Does AgentMamba keep Mamba's linear-time behaviour?

    python scripts/benchmark_linear_time.py --threads 8 [--scenario SMS1] [--max-len 8192]

Part A - Mamba itself: time one forward pass of the frozen mamba-130m over L tokens of ladle-log
         text, L = 256 ... max-len (3 repetitions, median). Fit t = a*L + b and report R^2.
         A selective state-space model processes a sequence in O(L) time; this measures it on CPU.
Part B - AgentMamba as the ladle history grows: the long-dataset memory snapshots of one scenario
         (stage 1 ... 4, ~20 -> ~3,000 records). For each stage, the 15 questions are asked with the
         Mamba score cache DISABLED and we record total query time, retrieval time and the number of
         tokens Mamba actually processed. "Mamba reads the whole history" is reported as the history
         token count and its time PROJECTED from the Part A fit (clearly labelled; not run).
Output: experiments/results/linear_time/linear_time.json (read by the dashboard).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from agentmamba.data import ladle_benchmark as lb  # noqa: E402
from agentmamba.memory.episodic_memory import EpisodicMemory  # noqa: E402
from agentmamba.models.answerer import MambaAnswerer, format_record  # noqa: E402
from agentmamba.models.embedding_model import SentenceEmbedder  # noqa: E402
from agentmamba.models.mamba_backbone import MambaBackbone  # noqa: E402
from agentmamba.pipeline.agent import QueryRequest  # noqa: E402
from agentmamba.pipeline.baseline import build_window  # noqa: E402
from agentmamba.pipeline.builder import build_agent  # noqa: E402


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys) or 1e-12
    return a, b, 1 - ss_res / ss_tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--scenario", default="SMS1")
    # 8192 tokens exhausted 16 GB RAM with the reference (non-CUDA) selective scan; 4096 is safe
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--window", type=int, default=2048)
    args = ap.parse_args()
    import torch

    cfg = load_config(overrides={"models.mamba_threads": args.threads})
    m = cfg["models"]
    bb = MambaBackbone(m["mamba_id"], args.threads, m["mamba_max_tokens"], None).load()   # no score cache
    tok = bb.tokenizer()

    # ---------------- Part A: Mamba forward time vs sequence length
    text = "\n".join(format_record(e) for s in (1, 2, 3, 4) for e in lb.stage_events("long", args.scenario, s))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    lengths = [L for L in (256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 16384)
               if L <= min(args.max_len, len(ids))]
    with torch.no_grad():
        bb._model(input_ids=torch.tensor([ids[:128]]), logits_to_keep=1)          # warm-up
    part_a = []
    for L in lengths:
        ts = []
        for _ in range(args.reps):
            t = time.perf_counter()
            with torch.no_grad():
                bb._model(input_ids=torch.tensor([ids[:L]]), logits_to_keep=1)
            ts.append(time.perf_counter() - t)
        part_a.append({"tokens": L, "seconds": round(statistics.median(ts), 4), "runs": [round(x, 4) for x in ts]})
        print(f"A  L={L:6d}  {statistics.median(ts):.3f}s", flush=True)
    a, b, r2 = linfit([p["tokens"] for p in part_a], [p["seconds"] for p in part_a])

    # ---------------- Part B: AgentMamba query cost as the ladle history grows
    embedder = SentenceEmbedder(m["embedding_id"])
    count = bb.count_tokens
    qs = [q for q in lb.load_questions("long") if q.scenario == args.scenario]
    history, seen, part_b = [], set(), []
    for stage in lb.stages_of("long", args.scenario):
        for e in lb.stage_events("long", args.scenario, stage):
            if e.event_id not in seen:
                seen.add(e.event_id)
                history.append(e)
        d = PROJECT_ROOT / "storage" / "ladle" / "long" / args.scenario / f"stage{stage}"
        if not (d / "memory.sqlite3").exists():
            print(f"B  stage {stage}: no snapshot at {d} (run scripts/run_ladle_eval.py first); skipped")
            continue
        mem = EpisodicMemory(d, embedder)
        agent = build_agent(cfg, mem, MambaAnswerer(bb))
        h_tok = sum(count(format_record(e)) + 1 for e in history)
        rows = []
        for q in qs:
            win = build_window(q.text, [], history, count, args.window)
            out = agent.query(QueryRequest(q.text, win.records, mode="conditional", verification=True))
            calls = out.get("model_calls") or []
            rows.append({"total_s": out["timings"].get("total_s", 0.0),
                         "retrieval_s": out["timings"].get("retrieval_s", 0.0),
                         "mamba_tokens": sum(c.get("prompt_tokens", 0) or 0 for c in calls),
                         "retrieved": bool(out.get("retrieval_triggered"))})
        n = mem.store.conn.execute("SELECT COUNT(*) FROM events WHERE status='active'").fetchone()[0]
        mem.close()
        rec = {"stage": stage, "records": n, "history_tokens": h_tok,
               "agent_total_s_median": round(statistics.median(r["total_s"] for r in rows), 3),
               "agent_retrieval_s_median": round(statistics.median(r["retrieval_s"] for r in rows), 4),
               "agent_mamba_tokens_median": statistics.median(r["mamba_tokens"] for r in rows),
               "agent_mamba_tokens_max": max(r["mamba_tokens"] for r in rows),
               "retrieval_rate": round(sum(r["retrieved"] for r in rows) / len(rows), 3),
               "full_history_mamba_s_projected": round(a * h_tok + b, 2)}
        part_b.append(rec)
        print("B ", rec, flush=True)

    out = {"model": m["mamba_id"], "threads": args.threads, "device": "CPU (reference PyTorch selective scan)",
           "part_a": {"points": part_a, "fit": {"seconds_per_token": a, "intercept_s": b, "r2": r2},
                      "note": "one forward pass of frozen mamba-130m over L tokens of ladle-log text"},
           "part_b": {"scenario": args.scenario, "window_tokens": args.window, "stages": part_b,
                      "note": ("AgentMamba timings are measured with the score cache disabled. "
                               "'full_history_mamba_s_projected' is NOT measured: it applies the Part A linear "
                               "fit to the token count of the whole history.")}}
    dest = resolve_path(cfg, "paths.results_dir") / "linear_time"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "linear_time.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"fit: t = {a * 1000:.4f} ms/token * L + {b:.3f}s   R^2 = {r2:.4f}  -> {dest / 'linear_time.json'}")


if __name__ == "__main__":
    main()
