"""Compute the full metric suite (answer quality, errors, evidence, retrieval, efficiency,
calibration, composite) for an existing context-growth run, from its stored scored rows.

    python scripts/compute_metric_suite.py --run experiments/results/<..._context_growth_test_final>

No model is re-run: metrics are recomputed from rows.jsonl (outputs scored against the held-out
annotations). Writes <run>/metric_suite.json and registers the run in the evaluation-results store.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import PROJECT_ROOT, load_config  # noqa: E402
from agentmamba.evaluation.metrics_suite import comparison_table, composite, efficiency_metrics, metric_suite  # noqa: E402
from agentmamba.evaluation.results_store import ResultsStore  # noqa: E402
from agentmamba.reproducibility import utc_now  # noqa: E402

RETRIEVES = {"baseline_mamba_768": False, "baseline_mamba_2048": False, "agentmamba_no_retrieval": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", default="baseline_mamba_2048")
    ap.add_argument("--enhanced", default="agentmamba")
    ap.add_argument("--produced-by", default="", help="code version that produced the predictions (for the record)")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = load_config()
    rows = [json.loads(x) for x in (run / "rows.jsonl").read_text(encoding="utf-8").splitlines() if x]
    lat = json.loads((run / "latency_cold.json").read_text(encoding="utf-8")) if (run / "latency_cold.json").exists() \
        else None
    comp_cfg = cfg.get("evaluation", {}).get("composite")
    systems = sorted({r["system"] for r in rows})
    suites, by_stage = {}, {}
    for s in systems:
        rs = [r for r in rows if r["system"] == s]
        retr = RETRIEVES.get(s, True)
        suites[s] = metric_suite(rs, retrieves=retr, model_load_s=lat["mamba_load_s"] if lat else None)
        suites[s]["composite"] = composite(suites[s], comp_cfg)
        by_stage[s] = {}
        for st in sorted({r["stage"] for r in rs}):
            su = metric_suite([r for r in rs if r["stage"] == st], retrieves=retr)
            su["efficiency"]["source"] = "in-run (affected by score-cache hits and host stalls; not comparable)"
            su["composite"] = composite(su, comp_cfg)
            su["composite"]["note"] = "efficiency component uses in-run latency; compare answer/grounding only"
            by_stage[s][str(st)] = su
    if lat:   # replace cache-affected in-run latencies by the cold-cache benchmark for the systems it measured
        for s in systems:
            cold = [r for r in lat["rows"] if r["system"] == s]
            if cold:
                suites[s]["efficiency"] = efficiency_metrics(
                    cold, retrieves=RETRIEVES.get(s, True), model_load_s=lat["mamba_load_s"],
                    source=f"cold-cache benchmark latency_cold.json (n={len(cold)}, stage {lat['stage']}, score cache "
                           f"disabled, nothing else running)")
            else:
                suites[s]["efficiency"]["source"] = ("in-run (affected by score-cache hits and host stalls; "
                                                     "not comparable)")
            suites[s]["composite"] = composite(suites[s], comp_cfg)
    out = {"run": run.name, "baseline": args.baseline, "enhanced": args.enhanced, "systems": systems,
           "suites": suites, "suites_by_stage": by_stage,
           "table": comparison_table(suites[args.baseline], suites[args.enhanced]),
           "cold_latency_benchmark": lat["summary"] if lat else None,
           "latency_note": ("In-run latencies include Mamba score-cache hits (conditions after the first reuse "
                            "identical prompts) and host throttling/sleep stalls (docs/EXPERIMENTS.md), so they are "
                            "not comparable across systems. Efficiency for the systems in the cold-cache benchmark "
                            "is taken from latency_cold.json instead; see each suite's efficiency.source."),
           "produced_by": args.produced_by or "see run config.json", "computed_at": utc_now()}
    (run / "metric_suite.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    store = ResultsStore(PROJECT_ROOT / "experiments" / "evaluation_results.sqlite3")
    store.put_run(run.name, "context_growth", utc_now(), str(run), {"produced_by": out["produced_by"]})
    store.put_rows(run.name, rows)
    for s, su in suites.items():
        store.put_suite(run.name, s, su)
    store.close()
    for t in out["table"]:
        print(f"{t['category'][:12]:12s} {t['metric']:28s} {str(t['baseline']):>8s} {str(t['agentmamba']):>8s} "
              f"{str(t['difference']):>8s}  n={t['n_baseline']}/{t['n_agentmamba']}")
    for s in (args.baseline, args.enhanced):
        print(s, suites[s]["composite"])


if __name__ == "__main__":
    main()
