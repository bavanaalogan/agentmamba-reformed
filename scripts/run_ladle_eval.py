"""Baseline Mamba vs AgentMamba on the two ladle datasets (short context, long context).

    python scripts/generate_ladle_datasets.py                       # writes data/ladle_benchmark/{short,long}
    python scripts/run_ladle_eval.py --threads 8                     # full run (Mamba answerer)
    python scripts/run_ladle_eval.py --answerer first_candidate      # fast smoke run, no Mamba

Protocol (identical for both systems):
* one fresh persistent memory per scenario (SQLite + FAISS, store-all policy); files are ingested
  in arrival order and the memory is snapshotted after every stage (storage/ladle/<dataset>/...);
* both systems get the SAME initial context: the most recent arrivals that fit a ``--window``
  token budget of the Mamba tokenizer (question-independent);
* Baseline Mamba answers from that window only (no retrieval, no verification);
  AgentMamba starts from it, retrieves from episodic memory only when the evidence check says the
  window is insufficient, and verifies before answering;
* short dataset: evaluated once (stage 1). Long dataset: evaluated after stages 2, 3 and 4
  (stage 1 of the long dataset IS the short dataset, so the short result is its stage-1 point);
* gold annotations are read only by the scoring pass, after all outputs are written.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from agentmamba.data import ladle_benchmark as lb  # noqa: E402
from agentmamba.evaluation.comparison import score_output, summarize_rows, paired  # noqa: E402
from agentmamba.evaluation.reporting import new_run_dir  # noqa: E402
from agentmamba.gate.predictor import build_scorer  # noqa: E402
from agentmamba.logging_utils import get_logger  # noqa: E402
from agentmamba.memory.episodic_memory import EpisodicMemory  # noqa: E402
from agentmamba.models.answerer import FirstCandidateAnswerer, MambaAnswerer  # noqa: E402
from agentmamba.models.embedding_model import SentenceEmbedder  # noqa: E402
from agentmamba.models.mamba_backbone import MambaBackbone  # noqa: E402
from agentmamba.pipeline.agent import QueryRequest  # noqa: E402
from agentmamba.pipeline.baseline import BaselineMamba, build_window  # noqa: E402
from agentmamba.pipeline.builder import build_agent  # noqa: E402
from agentmamba.pipeline.ingestion import FeatureProvider, IngestionPipeline  # noqa: E402
from agentmamba.reproducibility import environment_info, seed_everything  # noqa: E402

log = get_logger("ladle_eval")
SYSTEMS = ("baseline_mamba", "agentmamba")
# The metrics this project reports (everything else was dropped from the dashboard):
KEEP = ("n", "n_value", "n_abstain_gold", "n_conflict_gold", "accuracy", "accuracy_ci95", "answerable_accuracy",
        "correct_abstention_rate", "conflict_detection_rate", "incorrect_answer_rate", "unsupported_claim_rate",
        "required_evidence_coverage", "evidence_in_context_rate", "retrieval_trigger_rate",
        "latency_total_s_median_all", "latency_total_s_median_cold", "n_cold", "prompt_tokens_median",
        "records_available_mean")


def snapshot(mem: EpisodicMemory, src: Path, dest: Path) -> None:
    mem.save()
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".sqlite3":
            d = sqlite3.connect(dest / f.name)
            mem.store.conn.backup(d)
            d.close()
        elif f.is_file():
            shutil.copy2(f, dest / f.name)


def trim(out: dict) -> dict:
    out = dict(out)
    v = out.get("verification")
    if isinstance(v, dict) and len(v.get("candidate_checks", [])) > 20:
        out["verification"] = {**v, "candidate_checks": v["candidate_checks"][:20], "candidate_checks_truncated": True}
    return out


def history_tokens(history, count) -> int:
    from agentmamba.models.answerer import format_record

    return sum(count(format_record(e)) + 1 for e in history)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="short,long")
    ap.add_argument("--scenarios", default="", help="comma list (default: all)")
    ap.add_argument("--answerer", default="mamba", choices=["mamba", "first_candidate"])
    ap.add_argument("--window", type=int, default=2048, help="shared initial-context window (Mamba tokens)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true", help="disable the Mamba score cache (honest latency)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(overrides={"models.mamba_threads": args.threads} if args.threads else None)
    seed_everything(cfg["seed"])
    run_dir = new_run_dir(resolve_path(cfg, "paths.results_dir"), f"ladle_eval{'_' + args.tag if args.tag else ''}")
    embedder = SentenceEmbedder(cfg["models"]["embedding_id"])
    m = cfg["models"]
    backbone = MambaBackbone(m["mamba_id"], m["mamba_threads"], m["mamba_max_tokens"],
                             None if args.no_cache else
                             resolve_path(cfg, "paths.artifacts_dir") / "models" / "mamba_score_cache.sqlite3")
    count = backbone.count_tokens

    def answerer(budget=None, cap=16):
        if args.answerer == "first_candidate":
            return FirstCandidateAnswerer()
        return MambaAnswerer(backbone, max_prompt_records=cap, max_tokens=budget)

    datasets = [d for d in args.datasets.split(",") if d]
    (run_dir / "config.json").write_text(json.dumps({
        "experiment": "ladle_eval", "datasets": datasets, "answerer": args.answerer, "window_tokens": args.window,
        "score_cache": not args.no_cache, "store_policy": "store_all", "models": {"mamba": m["mamba_id"],
        "embedding": embedder.model_id}, "environment": environment_info(), "config": cfg}, indent=2, default=str),
        encoding="utf-8")
    pred_path = run_dir / "predictions.jsonl"
    t_run = time.perf_counter()
    for ds in datasets:
        questions = lb.load_questions(ds)
        scen = sorted({q.scenario for q in questions})
        if args.scenarios:
            scen = [s for s in scen if s in args.scenarios.split(",")]
        eval_stages = [1] if ds == "short" else [2, 3, 4]
        for sc in scen:
            store = PROJECT_ROOT / "storage" / "ladle" / ds / sc
            live = store / "live"
            shutil.rmtree(live, ignore_errors=True)
            mem = EpisodicMemory(live, embedder)
            pipe = IngestionPipeline(mem, FeatureProvider(embedder, None), build_scorer("store_all"), "store_all",
                                     False, run_tag=f"ladle:{ds}:{sc}")
            agent = build_agent(cfg, mem, answerer(), versions={"dataset": ds})
            baseline = BaselineMamba(agent.qp, answerer(args.window, None), count, args.window)
            history, qs = [], [q for q in questions if q.scenario == sc]
            for stage in lb.stages_of(ds, sc):
                evs = lb.stage_events(ds, sc, stage)
                t = time.perf_counter()
                rep = pipe.ingest(evs, None)
                known = {h.event_id for h in history}
                history += [e for e in evs if e.event_id not in known and not known.add(e.event_id)]
                snapshot(mem, live, store / f"stage{stage}")
                n_avail = mem.store.conn.execute("SELECT COUNT(*) FROM events WHERE status='active'").fetchone()[0]
                h_tok = history_tokens(history, count)
                with open(run_dir / "ingestion.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"dataset": ds, "scenario": sc, "stage": stage, "received": len(evs),
                                         "written": rep.written, "duplicates": rep.duplicates,
                                         "records_available": n_avail, "history_tokens": h_tok,
                                         "ingest_s": round(time.perf_counter() - t, 3)}) + "\n")
                log.info("%s/%s stage %d: %d records in memory, %d history tokens", ds, sc, stage, n_avail, h_tok)
                if stage not in eval_stages:
                    continue
                for q in qs:
                    win = build_window(q.text, [], history, count, args.window)
                    outs = {"baseline_mamba": baseline.answer(q.text, win, q.question_id),
                            "agentmamba": agent.query(QueryRequest(q.text, win.records, mode="conditional",
                                                                   verification=True, query_id=q.question_id))}
                    with open(pred_path, "a", encoding="utf-8") as fh:
                        for name, out in outs.items():
                            fh.write(json.dumps({"dataset": ds, "system": name, "stage": stage, "scenario": sc,
                                                 "question_id": q.question_id, "records_available": n_avail,
                                                 "history_tokens": h_tok,
                                                 "context_ids": [e.event_id for e in win.records],
                                                 "context_tokens": win.prompt_tokens, "output": trim(out)},
                                                default=str) + "\n")
                log.info("%s/%s stage %d: %d questions done (%.0fs)", ds, sc, stage, len(qs), time.perf_counter() - t_run)
            mem.close()
            shutil.rmtree(live, ignore_errors=True)

    # ------------------------------------------------------------ scoring (reads gold only now)
    preds = [json.loads(x) for x in pred_path.read_text(encoding="utf-8").splitlines() if x]
    anns = {ds: lb.load_annotations(ds) for ds in datasets}
    rows = []
    for p in preds:
        r = score_output(p["output"], anns[p["dataset"]][p["question_id"]], system=p["system"], stage=p["stage"],
                         context_ids=p["context_ids"], n_available=p["records_available"])
        rows.append(r | {"dataset": p["dataset"], "context_tokens": p["context_tokens"],
                         "history_tokens": p["history_tokens"], "question": p["output"].get("question")})
    with open(run_dir / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")

    def summ(rs):
        s = summarize_rows(rs)
        return {k: s[k] for k in KEEP}

    summary = {"experiment": "ladle_eval", "answerer": args.answerer, "window_tokens": args.window,
               "datasets": {}, "wall_s": round(time.perf_counter() - t_run, 1)}
    for ds in datasets:
        drows = [r for r in rows if r["dataset"] == ds]
        final = max((r["stage"] for r in drows), default=1)
        fin = [r for r in drows if r["stage"] == final]
        d = {"final_stage": final,
             "by_system": {s: summ([r for r in fin if r["system"] == s]) for s in SYSTEMS},
             "by_stage": {s: {str(st): summ([r for r in drows if r["system"] == s and r["stage"] == st])
                              for st in sorted({r["stage"] for r in drows})} for s in SYSTEMS},
             "by_family": {s: {f: summarize_rows([r for r in fin if r["system"] == s and r["family"] == f])["accuracy"]
                               for f in sorted({r["family"] for r in fin})} for s in SYSTEMS},
             "paired_correct": paired([r for r in fin if r["system"] == "baseline_mamba"],
                                      [r for r in fin if r["system"] == "agentmamba"], "correct")}
        summary["datasets"][ds] = d
    summary["ingestion"] = [json.loads(x) for x in (run_dir / "ingestion.jsonl").read_text(encoding="utf-8").splitlines()
                            if x]
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_md(summary), encoding="utf-8")
    log.info("done -> %s", run_dir)
    print(summary_md(summary))


def _f(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def summary_md(s: dict) -> str:
    L = [f"# Ladle evaluation: Baseline Mamba vs AgentMamba (answerer={s['answerer']}, window={s['window_tokens']} "
         f"tokens)", ""]
    for ds, d in s["datasets"].items():
        L += [f"## {ds} context (stage {d['final_stage']})", "",
              "| system | n | answerable accuracy | correct refusals | conflicts flagged | wrong answers | "
              "evidence seen | median latency (s) | median Mamba tokens |", "|---|---|---|---|---|---|---|---|---|"]
        for sys_, v in d["by_system"].items():
            L.append(f"| {sys_} | {v['n']} | {_f(v['answerable_accuracy'])} | {_f(v['correct_abstention_rate'])} | "
                     f"{_f(v['conflict_detection_rate'])} | {_f(v['incorrect_answer_rate'])} | "
                     f"{_f(v['required_evidence_coverage'])} | {_f(v['latency_total_s_median_all'], 2)} | "
                     f"{_f(v['prompt_tokens_median'], 0)} |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
