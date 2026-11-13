"""Stage-1 / stage-2 context-growth protocol: Baseline Mamba vs AgentMamba on ONE persistent profile.

    python scripts/run_protocol.py                                   # data/protocols/cp17_context_growth
    python scripts/run_protocol.py --answerer first_candidate        # quick smoke run (no Mamba scoring)

Steps (fixed in the protocol file before running):
1. Create a CLEAN profile (storage/profiles/<name>); refuse to reuse one unless --reuse is given.
2. Load Mamba and MiniLM once; their load times are reported separately (cold start). The Mamba
   score cache is DISABLED so every query's latency is a real computation.
3. For each stage: ingest the stage file into the SAME profile, then ask every question to both
   systems with the same shared initial context (most recent arrivals within the token budget).
   If AgentMamba skipped retrieval, the same query is re-run with retrieval forced ("always") to
   measure the time the skip saved (the forced run is reported, never used as the answer).
4. Only after all outputs are written: read gold answers, resolve gold evidence lines to record IDs,
   score, compute the metric suite, and write everything to the run directory and to the separate
   evaluation-results store.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from agentmamba.data.document_loader import load_any  # noqa: E402
from agentmamba.evaluation.comparison import score_output  # noqa: E402
from agentmamba.evaluation.metrics_suite import comparison_table, composite, metric_suite  # noqa: E402
from agentmamba.evaluation.reporting import new_run_dir  # noqa: E402
from agentmamba.evaluation.results_store import ResultsStore  # noqa: E402
from agentmamba.models.answerer import FirstCandidateAnswerer, MambaAnswerer  # noqa: E402
from agentmamba.models.mamba_backbone import MambaBackbone  # noqa: E402
from agentmamba.pipeline.agent import QueryRequest  # noqa: E402
from agentmamba.pipeline.service import AgentService  # noqa: E402
from agentmamba.reproducibility import disable_background_throttling, environment_info, utc_now  # noqa: E402


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="data/protocols/cp17_context_growth")
    ap.add_argument("--profile", default="")
    ap.add_argument("--reuse", action="store_true", help="allow an existing profile (default: refuse)")
    ap.add_argument("--answerer", choices=["mamba", "first_candidate"], default="mamba")
    ap.add_argument("--policy", choices=["configured", "store_all"], default="configured")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    print("process:", disable_background_throttling(), flush=True)
    pdir = PROJECT_ROOT / args.protocol
    proto = json.loads((pdir / "protocol.json").read_text(encoding="utf-8"))
    for st in proto["stages"]:
        if sha256(pdir / st["file"]) != st["sha256"]:
            sys.exit(f"checksum mismatch for {st['file']}: the protocol data changed")
    profile = args.profile or f"protocol_{proto['protocol']}_{args.answerer}"
    pdir_mem = PROJECT_ROOT / "storage" / "profiles" / profile
    if pdir_mem.exists():
        if not args.reuse:
            shutil.rmtree(pdir_mem)          # clean test profile: only ever under storage/profiles/
    cfg = load_config(overrides={"models.mamba_threads": args.threads} if args.threads else None)
    m = cfg["models"]
    backbone = MambaBackbone(m["mamba_id"], m["mamba_threads"], m["mamba_max_tokens"], cache_path=None)
    t = time.perf_counter()
    if args.answerer == "mamba":
        backbone.load()
    mamba_load_s = time.perf_counter() - t if args.answerer == "mamba" else None
    answerer = MambaAnswerer(backbone) if args.answerer == "mamba" else FirstCandidateAnswerer()
    svc = AgentService(cfg, pdir_mem, backbone=backbone, answerer=answerer, token_counter=backbone.count_tokens)
    t = time.perf_counter()
    _ = svc.embedder.encode(["warm-up"])
    embed_load_s = time.perf_counter() - t
    window = int(proto["shared_initial_context"]["token_budget"])
    run_dir = new_run_dir(resolve_path(cfg, "paths.results_dir"), f"protocol_{proto['protocol']}_{args.answerer}")
    preds, ingests = [], []
    for st in proto["stages"]:
        t = time.perf_counter()
        rep = svc.ingest_file(pdir / st["file"], st["plant_stage"], "authorized", None,
                              None if args.policy == "configured" else "store_all")
        ing = {k: v for k, v in rep.items() if k not in ("decisions", "normalized_preview")}
        ing.update({"stage": st["stage"], "file": st["file"], "wall_s": round(time.perf_counter() - t, 3),
                    "records_in_profile": svc.memory.stats()["records"]})
        ingests.append(ing)
        print(f"stage {st['stage']}: received {rep['received']}, written {rep['written']}, duplicates "
              f"{rep['duplicates']}, profile now {ing['records_in_profile']} records", flush=True)
        for q in proto["questions"]:
            res = svc.compare(q["text"], [], window, proto["agentmamba"]["mode"], proto["agentmamba"]["verification"])
            enh = res["enhanced"]
            forced = None
            if enh is not None and not enh["retrieval_triggered"]:
                ctx = svc.memory.store.get_events(res["shared_context"]["record_ids"])
                fo = svc.agent.query(QueryRequest(q["text"], ctx, "always", verification=True))
                forced = {"total_s": fo["timings"]["total_s"], "retrieval_s": fo["timings"].get("retrieval_s"),
                          "answer_value": fo["answer_value"], "answer_status": fo["answer_status"]}
            preds.append({"stage": st["stage"], "question_id": q["question_id"], "question": q["text"],
                          "result": res, "forced_retrieval_counterfactual": forced})
            b = res["baseline"] or {}
            print(f"  {q['question_id']}: baseline={b.get('answer_value')!r} | agentmamba={enh and enh['answer_value']!r} "
                  f"retrieval={enh and enh['retrieval_triggered']} ({enh and enh['sufficiency']['decision']})",
                  flush=True)
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, default=str) + "\n")

    # ------------------------------------------------ scoring: gold is read only now
    gold = json.loads((pdir / "annotations" / "gold.json").read_text(encoding="utf-8"))
    line_ids = {}
    for st in proto["stages"]:
        for e in load_any(pdir / st["file"], stage=st["plant_stage"]):
            line_ids[(st["file"], int(e.source_ref.rsplit("=", 1)[1]))] = e.event_id
    rows = []
    for p in preds:
        g = gold[p["question_id"]]
        ann = SimpleNamespace(question_id=p["question_id"], scenario=proto["protocol"], split="protocol",
                              family=next(q["kind"] for q in proto["questions"] if q["question_id"] == p["question_id"]),
                              answer_kind=g["answer_kind"], expected_answer=g["expected_answer"], match=g["match"],
                              required_evidence_ids=[line_ids[(f, n)] for f, n in g["required_evidence"]])
        res = p["result"]
        ctx_ids = res["shared_context"]["record_ids"]
        for system, out in (("baseline_mamba", res["baseline"]), ("agentmamba", res["enhanced"])):
            if out is None:
                continue
            r = score_output(out, ann, system=system, stage=p["stage"], context_ids=ctx_ids,
                             n_available=next(i["records_in_profile"] for i in ingests if i["stage"] == p["stage"]))
            r.update({"question": p["question"], "context_tokens": res["shared_context"]["prompt_tokens"],
                      "calibrated_confidence": (out.get("confidence") or {}).get("calibrated"),
                      "sufficiency": (out.get("sufficiency") or {}).get("decision"),
                      "sufficiency_reason": (out.get("sufficiency") or {}).get("reason"),
                      "retrieval_stats": out.get("retrieval_stats"),
                      "abstention_reason": out.get("abstention_reason"),
                      "evidence_ids": [e["record_id"] for e in out.get("evidence", [])],
                      "evidence_sources": [e["source"] for e in out.get("evidence", [])]})
            if system == "agentmamba" and p["forced_retrieval_counterfactual"]:
                f = p["forced_retrieval_counterfactual"]
                r["forced_retrieval_total_s"] = f["total_s"]
                r["time_saved_by_skipping_s"] = round(f["total_s"] - (r["total_s"] or 0), 4)
            rows.append(r)
    with open(run_dir / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    comp_cfg = cfg.get("evaluation", {}).get("composite")
    suites, by_stage = {}, {}
    for system, retrieves in (("baseline_mamba", False), ("agentmamba", True)):
        rs = [r for r in rows if r["system"] == system]
        suites[system] = metric_suite(rs, retrieves=retrieves, model_load_s=mamba_load_s)
        suites[system]["composite"] = composite(suites[system], comp_cfg)
        by_stage[system] = {}
        for s in sorted({r["stage"] for r in rs}):
            su = metric_suite([r for r in rs if r["stage"] == s], retrieves=retrieves)
            su["composite"] = composite(su, comp_cfg)
            by_stage[system][str(s)] = su
    saved = [r["time_saved_by_skipping_s"] for r in rows if r.get("time_saved_by_skipping_s") is not None]
    summary = {
        "kind": "protocol", "protocol": proto["protocol"], "answerer": args.answerer, "profile": str(pdir_mem),
        "diagnostic_note": f"{len(proto['questions'])} questions x {len(proto['stages'])} stages on one scenario: a "
                           "protocol demonstration, NOT a benchmark. Benchmark results: context-growth test run.",
        "cold_start": {"mamba_load_s": None if mamba_load_s is None else round(mamba_load_s, 2),
                       "embedding_load_s": round(embed_load_s, 2),
                       "note": "model loading is excluded from every query latency"},
        "ingestion": ingests, "suites": suites, "suites_by_stage": by_stage,
        "table": comparison_table(suites["baseline_mamba"], suites["agentmamba"]),
        "retrieval_skip_savings": {"n_skipped": len(saved),
                                   "mean_time_saved_s": round(sum(saved) / len(saved), 4) if saved else None,
                                   "definition": "forced-retrieval latency minus conditional latency, same query and "
                                                 "context, for queries where AgentMamba skipped retrieval"},
        "per_question": [{k: r.get(k) for k in ("stage", "question_id", "system", "answer_status", "answer_value",
                                                 "expected", "correct", "grounded_correct", "unsupported",
                                                 "sufficiency", "retrieval_triggered", "evidence_ids",
                                                 "evidence_in_context", "evidence_covered", "context_tokens",
                                                 "total_s", "generation_s", "retrieval_s", "verification_s",
                                                 "candidate_share", "time_saved_by_skipping_s")} for r in rows],
    }
    run_cfg = {"protocol": proto, "args": vars(args), "config": cfg, "environment": environment_info(),
               "models": {"mamba": m["mamba_id"], "score_cache": "disabled", "embedding": m["embedding_id"]}}
    (run_dir / "config.json").write_text(json.dumps(run_cfg, indent=2, default=str), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_md(summary), encoding="utf-8")
    store = ResultsStore(PROJECT_ROOT / "experiments" / "evaluation_results.sqlite3")
    store.put_run(run_dir.name, "protocol", utc_now(), str(run_dir), run_cfg)
    store.put_rows(run_dir.name, rows)
    for s, su in suites.items():
        store.put_suite(run_dir.name, s, su)
    store.close()
    print(f"done -> {run_dir}", flush=True)


def _f(v):
    return "N/A" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def summary_md(s: dict) -> str:
    L = [f"# Protocol {s['protocol']} (answerer={s['answerer']})", "", f"_{s['diagnostic_note']}_", "",
         f"Cold start: Mamba load {s['cold_start']['mamba_load_s']} s, embedding load "
         f"{s['cold_start']['embedding_load_s']} s (excluded from query latency).", "",
         "## Per question", "",
         "| stage | question | system | status | answer | expected | correct | sufficiency | retrieval | evidence IDs | "
         "context tok | total s |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["per_question"]:
        L.append(f"| {r['stage']} | {r['question_id']} | {r['system']} | {r['answer_status']} | "
                 f"{str(r['answer_value'])[:60]} | {r['expected']} | {r['correct']} | {r['sufficiency'] or '-'} | "
                 f"{r['retrieval_triggered']} | {', '.join(r['evidence_ids'] or [])} | {r['context_tokens']} | "
                 f"{_f(r['total_s'])} |")
    L += ["", "## Metrics (both stages pooled)", "",
          "| category | metric | Baseline Mamba | AgentMamba | difference | n (B / A) |", "|---|---|---|---|---|---|"]
    for t in s["table"]:
        L.append(f"| {t['category']} | {t['metric']} ({t['better']}) | {_f(t['baseline'])} | {_f(t['agentmamba'])} | "
                 f"{_f(t['difference'])} | {t['n_baseline']} / {t['n_agentmamba']} |")
    cb, ca = s["suites"]["baseline_mamba"]["composite"], s["suites"]["agentmamba"]["composite"]
    L += ["", f"Composite (shared components): Baseline {cb['composite_shared']} | AgentMamba {ca['composite_shared']}",
          f"Composite (all available components): Baseline {cb['composite_full']} {cb['components_used_full']} | "
          f"AgentMamba {ca['composite_full']} {ca['components_used_full']}",
          "", f"Retrieval skip savings: {s['retrieval_skip_savings']}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
