"""Context-growth experiment: Baseline Mamba vs AgentMamba as ingested history grows.

    python scripts/run_context_growth.py --split dev  --tag dev      # development + calibration fit
    python scripts/run_context_growth.py --split test --tag final    # final report (uses dev calibrators)
    python scripts/run_context_growth.py --split dev --answerer first_candidate --scenarios P4   # smoke

Protocol (fixed before running; see docs/EXPERIMENTS.md, "Context-growth experiment"):
* per scenario, one fresh persistent memory (SQLite + FAISS, store-all policy, no memory gate);
  the stage files are ingested IN ARRIVAL ORDER through the normal ingestion pipeline, and the
  memory is snapshotted after every stage (storage/experiments/context_growth/<run>/...);
* after each stage every question of the scenario is asked to every condition;
* every condition receives the SAME initial context: the most recent arrivals that fit a 768-token
  window (Mamba tokenizer). ``baseline_mamba_2048`` additionally gets a 2048-token window;
* raw outputs are written first; gold annotations are read only by the scoring pass at the end.
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

import yaml  # noqa: E402

from agentmamba.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from agentmamba.data.document_loader import load_any  # noqa: E402
from agentmamba.data.log_benchmark import load_annotations, load_questions, stage_files  # noqa: E402
from agentmamba.evaluation.comparison import (LOWER_IS_BETTER, apply_calibrator, calibration_report,  # noqa: E402
                                              fit_calibrator, paired, score_output, summarize_rows)
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
from agentmamba.reproducibility import disable_background_throttling, environment_info, seed_everything  # noqa: E402

log = get_logger("context_growth")
COMPARISONS = [("baseline_mamba_768", "agentmamba"), ("baseline_mamba_2048", "agentmamba"),
               ("agentmamba_no_retrieval", "agentmamba"), ("agentmamba_no_verification", "agentmamba"),
               ("retrieval_always", "agentmamba"), ("baseline_mamba_768", "agentmamba_no_verification"),
               ("baseline_mamba_768", "agentmamba_no_retrieval"), ("baseline_mamba_768", "baseline_mamba_2048"),
               ("agentmamba_vector_only", "agentmamba"), ("agentmamba_lexical_only", "agentmamba")]


def snapshot(mem: EpisodicMemory, src: Path, dest: Path) -> None:
    mem.save()
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/log_benchmark/v1")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--scenarios", default="", help="comma list (default: all of the split)")
    ap.add_argument("--conditions", default="", help="comma list (default: all in config)")
    ap.add_argument("--answerer", default="mamba", choices=["mamba", "first_candidate"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", default="", help="existing run directory to continue")
    args = ap.parse_args()
    print("process:", disable_background_throttling(), flush=True)
    cfg = load_config(overrides={"models.mamba_threads": args.threads} if args.threads else None)
    seed_everything(cfg["seed"])
    exp = yaml.safe_load((PROJECT_ROOT / "config" / "experiments.yaml").read_text(encoding="utf-8"))["context_growth"]
    conds = {k: v for k, v in exp["conditions"].items() if not args.conditions or k in args.conditions.split(",")}
    bench = PROJECT_ROOT / args.bench
    questions = load_questions(bench, args.split)
    scen = sorted({q.scenario for q in questions})
    if args.scenarios:
        scen = [s for s in scen if s in args.scenarios.split(",")]
    run_dir = Path(args.resume) if args.resume else new_run_dir(
        resolve_path(cfg, "paths.results_dir"), f"context_growth_{args.split}{'_' + args.tag if args.tag else ''}")
    store_root = PROJECT_ROOT / "storage" / "experiments" / "context_growth" / run_dir.name
    pred_path = run_dir / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["condition"], r["stage"], r["question_id"]))

    embedder = SentenceEmbedder(cfg["models"]["embedding_id"])
    m = cfg["models"]
    backbone = MambaBackbone(m["mamba_id"], m["mamba_threads"], m["mamba_max_tokens"],
                             resolve_path(cfg, "paths.artifacts_dir") / "models" / "mamba_score_cache.sqlite3")
    count = backbone.count_tokens        # Mamba tokenizer (always the real one, also for the smoke answerer)

    def answerer(budget=None, cap=16):
        if args.answerer == "first_candidate":
            return FirstCandidateAnswerer()
        return MambaAnswerer(backbone, max_prompt_records=cap, max_tokens=budget)

    run_cfg = {"experiment": "context_growth", "split": args.split, "scenarios": scen, "conditions": conds,
               "answerer": args.answerer, "benchmark": json.loads((bench / "benchmark_manifest.json").read_text()),
               "initial_context": exp["initial_context"], "config": cfg, "environment": environment_info(),
               "models": {"mamba": m["mamba_id"], "embedding": embedder.model_id, "precision": "float32 (CPU)",
                          "generation": "none: candidate ranking by mean token log-prob (no sampling)"},
               "store_policy": "store_all (memory gate not applied; it is not validated on log records)"}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():          # resumed run: keep the original config, record each continuation
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        old.setdefault("resumes", []).append({"scenarios": scen, "environment": run_cfg["environment"],
                                              "conditions": list(conds)})
        run_cfg = old
    cfg_path.write_text(json.dumps(run_cfg, indent=2, default=str), encoding="utf-8")
    ctx_budget = int(exp["initial_context"]["token_budget"])
    t_run = time.perf_counter()
    for sc in scen:
        root = bench / args.split / sc
        live = store_root / sc / "live"
        shutil.rmtree(live, ignore_errors=True)
        mem = EpisodicMemory(live, embedder)
        pipe = IngestionPipeline(mem, FeatureProvider(embedder, None), build_scorer("store_all"), "store_all", False,
                                 run_tag=f"context_growth:{sc}")
        agents = {}
        for name, c in conds.items():
            if c["kind"] == "agent":
                ccfg = json.loads(json.dumps(cfg))
                if c.get("channels"):
                    ccfg["retrieval"]["channels"] = c["channels"]
                agents[name] = build_agent(ccfg, mem, answerer(), versions={"condition": name})
        baselines = {name: BaselineMamba(agents[next(iter(agents))].qp, answerer(c["budget"], None), count,
                                         c["budget"]) for name, c in conds.items() if c["kind"] == "baseline"}
        history = []
        qs = [q for q in questions if q.scenario == sc]
        for stage in exp["stages"]:
            evs = []
            for f in stage_files(root, stage):
                evs += load_any(root / f["file"], stage=f["stage"])
            t = time.perf_counter()
            rep = pipe.ingest(evs, None)
            ingest = {k: v for k, v in rep.to_dict(0).items() if k != "decisions"}
            ingest.update({"received": len(evs), "chunks_created": sum(e.event_type == "document_chunk" for e in evs),
                           "files": len(stage_files(root, stage)), "wall_s": round(time.perf_counter() - t, 3)})
            known = {h.event_id for h in history}
            history += [e for e in evs if e.event_id not in known and not known.add(e.event_id)]
            snapshot(mem, live, store_root / sc / f"stage{stage}")
            n_avail = mem.store.conn.execute("SELECT COUNT(*) FROM events WHERE status='active'").fetchone()[0]
            with open(run_dir / "ingestion.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"scenario": sc, "stage": stage, "records_available": n_avail, **ingest}) + "\n")
            log.info("%s stage %d: %d received, %d written, %d duplicates, %d available", sc, stage, len(evs),
                     rep.written, rep.duplicates, n_avail)
            windows = {}
            for q in qs:
                for name, c in conds.items():
                    if (name, stage, q.question_id) in done or stage not in c.get("stages", exp["stages"]):
                        continue
                    if c["kind"] == "baseline":
                        win = windows.setdefault((q.question_id, c["budget"]),
                                                 build_window(q.text, [], history, count, c["budget"]))
                        out = baselines[name].answer(q.text, win, q.question_id)
                    else:
                        win = windows.setdefault((q.question_id, ctx_budget),
                                                 build_window(q.text, [], history, count, ctx_budget))
                        out = agents[name].query(QueryRequest(q.text, win.records, mode=c["mode"],
                                                              verification=c["verify"], query_id=q.question_id))
                    rec = {"condition": name, "stage": stage, "question_id": q.question_id, "scenario": sc,
                           "context_ids": [e.event_id for e in win.records], "context_tokens": win.prompt_tokens,
                           "records_available": n_avail, "output": trim(out)}
                    with open(pred_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, default=str) + "\n")
                log.info("%s stage %d %s done (%.0fs elapsed)", sc, stage, q.question_id, time.perf_counter() - t_run)
        mem.close()

    # ------------------------------------------------------------ scoring (reads gold only now)
    anns = load_annotations(bench, args.split)
    preds = [json.loads(x) for x in pred_path.read_text(encoding="utf-8").splitlines() if x]
    rows = [score_output(p["output"], anns[p["question_id"]], system=p["condition"], stage=p["stage"],
                         context_ids=p["context_ids"], n_available=p["records_available"]) | {
                "context_tokens": p["context_tokens"]} for p in preds]
    cal_dir = resolve_path(cfg, "paths.artifacts_dir") / "calibration" / "logbench"
    calibration = {}
    by_sys = {}
    for r in rows:
        by_sys.setdefault(r["system"], []).append(r)
    for name, rs in by_sys.items():
        if args.split == "dev" and args.answerer == "mamba":
            cal = fit_calibrator(rs)
            cal_dir.mkdir(parents=True, exist_ok=True)
            (cal_dir / f"{name}.json").write_text(json.dumps({**cal, "fitted_on": str(run_dir.name)}, indent=2),
                                                  encoding="utf-8")
        p = cal_dir / f"{name}.json"
        cal = json.loads(p.read_text(encoding="utf-8")) if p.exists() and args.answerer == "mamba" else None
        for r in rs:
            r["calibrated_confidence"] = apply_calibrator(cal, r)
        calibration[name] = {"calibrator": cal,
                             "raw_candidate_share": calibration_report(rs, "candidate_share"),
                             "calibrated": calibration_report(rs, "calibrated_confidence"),
                             "note": ("calibrator fitted on THESE dev rows: in-sample, optimistic" if args.split == "dev"
                                      else "calibrator fitted on the dev split, applied to test")}
    with open(run_dir / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    stages = sorted({r["stage"] for r in rows})
    summary = {
        "experiment": "context_growth", "split": args.split, "answerer": args.answerer, "n_rows": len(rows),
        "n_questions": len({r["question_id"] for r in rows}), "scenarios": sorted({p["scenario"] for p in preds}),
        "by_condition": {k: summarize_rows(v) for k, v in by_sys.items()},
        "by_condition_stage": {k: {str(s): summarize_rows([r for r in v if r["stage"] == s]) for s in stages
                                   if any(r["stage"] == s for r in v)} for k, v in by_sys.items()},
        "by_condition_family": {k: {f: summarize_rows([r for r in v if r["family"] == f])["accuracy"]
                                    for f in sorted({r["family"] for r in v})} for k, v in by_sys.items()},
        "comparisons": {}, "calibration": calibration,
        "ingestion": [json.loads(x) for x in (run_dir / "ingestion.jsonl").read_text(encoding="utf-8").splitlines() if x],
        "wall_s": round(time.perf_counter() - t_run, 1),
    }
    for a, b in COMPARISONS:
        if a in by_sys and b in by_sys:
            summary["comparisons"][f"{b}__vs__{a}"] = {
                "correct": paired(by_sys[a], by_sys[b], "correct"),
                "grounded_correct": paired(by_sys[a], by_sys[b], "grounded_correct"),
                "unsupported_claim": paired(by_sys[a], by_sys[b], "unsupported") | {"direction": "lower is better"},
                "incorrect_answer": paired([r | {"wrong": r["answered"] and not r["correct"]} for r in by_sys[a]],
                                           [r | {"wrong": r["answered"] and not r["correct"]} for r in by_sys[b]],
                                           "wrong") | {"direction": "lower is better"}}
    summary["lower_is_better"] = sorted(LOWER_IS_BETTER)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_md(summary), encoding="utf-8")
    log.info("done -> %s", run_dir)


def _f(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def summary_md(s: dict) -> str:
    L = [f"# Context-growth experiment ({s['split']} split, answerer={s['answerer']})", "",
         f"{s['n_questions']} questions x stages; scenarios {', '.join(s['scenarios'])}. Synthetic log benchmark.", "",
         "## Overall (all stages pooled)", "",
         "| condition | n | accuracy [95% CI] | grounded | answerable acc. | correct abstention | conflict det. | "
         "incorrect answers | unsupported claims / answered | evidence in context | Recall@k | median latency cold (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in s["by_condition"].items():
        L.append(f"| {k} | {v['n']} | {_f(v['accuracy'])} {v['accuracy_ci95']} | {_f(v['grounded_accuracy'])} | "
                 f"{_f(v['answerable_accuracy'])} | {_f(v['correct_abstention_rate'])} | {_f(v['conflict_detection_rate'])} | "
                 f"{_f(v['incorrect_answer_rate'])} | {_f(v['unsupported_claim_rate'])} | {_f(v['evidence_in_context_rate'])} | "
                 f"{_f(v['recall_at_k'])} | {_f(v['latency_total_s_median_cold'], 2)} |")
    L += ["", "## Accuracy by stage", "", "| condition | " + " | ".join(f"stage {x}" for x in
                                                                        sorted({st for v in s['by_condition_stage'].values() for st in v})) + " |"]
    sts = sorted({st for v in s["by_condition_stage"].values() for st in v})
    L.append("|---|" + "---|" * len(sts))
    for k, v in s["by_condition_stage"].items():
        L.append(f"| {k} | " + " | ".join(_f(v[x]["accuracy"]) if x in v else "-" for x in sts) + " |")
    L += ["", "## Evidence inside the 768/2048-token context window, by stage", "",
          "| condition | " + " | ".join(f"stage {x}" for x in sts) + " |", "|---|" + "---|" * len(sts)]
    for k, v in s["by_condition_stage"].items():
        L.append(f"| {k} | " + " | ".join(_f(v[x]["evidence_in_context_rate"]) if x in v else "-" for x in sts) + " |")
    L += ["", "## Paired comparisons (B vs A, same question and stage)", "",
          "| B vs A | outcome | A | B | B-A [95% CI] | McNemar p |", "|---|---|---|---|---|---|"]
    for k, v in s["comparisons"].items():
        for m, r in v.items():
            if r.get("n_pairs"):
                L.append(f"| {k.replace('__vs__', ' vs ')} | {m}{' (lower better)' if r.get('direction') else ''} | "
                         f"{r['a']} | {r['b']} | {r['diff_b_minus_a']} {r['diff_ci95']} | {r['mcnemar_p']} |")
    L += ["", "## Calibration (answered questions)", "", "| condition | confidence | n | Brier | ECE | mean conf. | acc. |",
          "|---|---|---|---|---|---|---|"]
    for k, v in s["calibration"].items():
        for key in ("raw_candidate_share", "calibrated"):
            c = v[key]
            L.append(f"| {k} | {key} | {c['n_answered_with_confidence']} | {_f(c['brier'])} | {_f(c['ece'])} | "
                     f"{_f(c['mean_confidence'])} | {_f(c['accuracy_when_answered'])} |")
    L += ["", "Pairs within a scenario are not independent; CIs and p-values are indicative only. "
              "Latency 'cold' = rows whose Mamba calls were all score-cache misses."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
