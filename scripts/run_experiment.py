"""Run experiments on the TEST split.

    python scripts/run_experiment.py --data data/synthetic/main --methods all
    python scripts/run_experiment.py --data data/synthetic/main --methods E_agentmamba,B_storeall_rag
    python scripts/run_experiment.py --data data/synthetic/main --sweep
    python scripts/run_experiment.py --data data/synthetic/main --memory-only --sweep   # memory metrics only

Each method writes experiments/results/<timestamp>_<method>/ with config.json,
predictions.jsonl, summary.json and summary.md. Memories are cached per (policy, gate, budget)
under storage/experiments/ and rebuilt only with --rebuild.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml

from agentmamba.config import PROJECT_ROOT, load_config, resolve_path
from agentmamba.data.dataset_loader import load_dataset
from agentmamba.data.evidence_annotations import load_annotations
from agentmamba.evaluation.evaluator import score_row, summarize
from agentmamba.evaluation.memory_metrics import gate_test_metrics, memory_metrics, rare_categories
from agentmamba.evaluation.reporting import new_run_dir, write_jsonl
from agentmamba.gate.features import FeatureStore, stream_novelty
from agentmamba.gate.label_builder import build_labels
from agentmamba.gate.train import GateArtifact
from agentmamba.logging_utils import get_logger
from agentmamba.memory.episodic_memory import EpisodicMemory
from agentmamba.models.embedding_model import SentenceEmbedder
from agentmamba.models.mamba_backbone import MambaBackbone
from agentmamba.models.answerer import FirstCandidateAnswerer, MambaAnswerer
from agentmamba.pipeline.agent import QueryRequest
from agentmamba.pipeline.builder import build_agent, build_memory, gate_path
from agentmamba.reproducibility import environment_info, seed_everything

log = get_logger("run_experiment")
GATES = {"logreg_mamba": ("logreg", True), "logreg_nomamba": ("logreg", False), "mlp_mamba": ("mlp", True)}


class Runner:
    def __init__(self, cfg, data, rebuild: bool, split: str = "test"):
        self.cfg, self.data, self.rebuild, self.split = cfg, Path(data), rebuild, split
        self.ds = load_dataset(data)
        qsplit = {q.question_id: q.split for q in self.ds.questions}
        # val = development runs (error analysis allowed); test = final reporting only
        self.anns = load_annotations(data, split, "evaluation", qsplit)
        self.questions = self.ds.questions_for(split)
        self.stream = self.ds.events_for(split)
        self.ev = self.ds.event_index()
        self.useful = {e for a in self.anns.values() for e in a.required_evidence_ids}
        self.rare = rare_categories(self.ds.events_for("train"))
        self.embedder = SentenceEmbedder(cfg["models"]["embedding_id"])
        self.fs = FeatureStore(resolve_path(cfg, "paths.feature_cache"))
        self.backbone = MambaBackbone(cfg["models"]["mamba_id"], cfg["models"]["mamba_threads"],
                                      cfg["models"]["mamba_max_tokens"],
                                      resolve_path(cfg, "paths.artifacts_dir") / "models" / "mamba_score_cache.sqlite3")
        self.gates: dict[str, GateArtifact] = {}
        self.memories: dict[str, tuple] = {}

    def gate(self, name):
        if name not in self.gates:
            kind, um = GATES[name]
            self.gates[name] = GateArtifact.load(gate_path(self.cfg, um, kind))
        return self.gates[name]

    def memory(self, policy, gate_name, budget):
        key = f"{self.split}_{policy}_{gate_name or 'none'}_{budget:.2f}"
        if key in self.memories:
            return self.memories[key]
        d = PROJECT_ROOT / "storage" / "experiments" / key
        gate = self.gate(gate_name) if policy == "learned" else None
        t = time.perf_counter()
        if d.exists() and not self.rebuild and (d / "memory.sqlite3").exists():
            mem = EpisodicMemory(d, self.embedder)
            rep = mem.store.get_meta("ingest_report")
        else:
            mem, r = build_memory(self.cfg, d, self.stream, policy=policy, budget_fraction=budget, embedder=self.embedder,
                                  feature_store=self.fs, backbone=self.backbone, gate=gate, run_tag=key)
            rep = r.to_dict(0)
            mem.store.set_meta("ingest_report", rep)
        retained = {r[0] for r in mem.store.conn.execute("SELECT event_id FROM events WHERE status='active'")}
        mm = memory_metrics(self.stream, retained, self.useful, self.rare)
        mm.update({"sqlite_file_bytes": mem.store.path.stat().st_size,
                   "faiss_file_bytes": mem.index_path.with_suffix(".faiss").stat().st_size,
                   "ingest_report": rep, "build_or_open_s": round(time.perf_counter() - t, 2)})
        if gate is not None:
            mm["gate_test"] = self._gate_test(gate)
        self.memories[key] = (mem, retained, mm)
        return self.memories[key]

    def _gate_test(self, gate):
        emb = self.fs.get("emb", self.embedder.model_id, self.stream, self.embedder.encode)
        mamba = None
        if gate.featurizer.use_mamba:
            mt = self.cfg["models"]["mamba_feature_max_tokens"]
            mamba = self.fs.get("mamba", self.cfg["models"]["mamba_id"] + f"@{mt}", self.stream,
                                lambda t: self.backbone.record_features(t, max_tokens=mt))
        X = gate.featurizer.transform(self.stream, emb, mamba, stream_novelty(emb, gate.featurizer.window))
        y = build_labels(self.stream, self.anns)   # evaluation-only use of test labels
        return gate_test_metrics(y, gate.predict_proba(X), gate.threshold)

    def run_method(self, name, spec, results_dir, memory_only=False):
        policy, gname, budget = spec["policy"], spec.get("gate"), float(spec["budget"])
        mem, retained, mm = self.memory(policy, gname, budget)
        run_cfg = {"method": name, "spec": spec, "config": self.cfg, "seed": self.cfg["seed"],
                   "dataset": {"root": str(self.data), "manifest": self.ds.manifest, "split": self.split},
                   "models": {"mamba": self.cfg["models"]["mamba_id"], "embedding": self.embedder.model_id,
                              "answerer": spec.get("answerer", self.cfg["models"]["answerer"])},
                   "gate_version": self.gate(gname).meta["gate_version"] if policy == "learned" else None,
                   "environment": environment_info()}
        if memory_only:
            return {"method": name, "memory": mm}
        cfg = json.loads(json.dumps(self.cfg))
        if spec.get("expand") is False:
            cfg["retrieval"]["expansion_depth"] = 0
        answerer = FirstCandidateAnswerer() if spec.get("answerer") == "first_candidate" else MambaAnswerer(self.backbone)
        agent = build_agent(cfg, mem, answerer, versions={"gate": run_cfg["gate_version"], "method": name})
        rows, preds = [], []
        t0 = time.perf_counter()
        for n, q in enumerate(self.questions, 1):
            ctx = [self.ev[i] for i in q.initial_context_ids]
            out = agent.query(QueryRequest(q.text, ctx, mode=spec["mode"], verification=spec["verify"],
                                           query_id=q.question_id))
            preds.append(out)
            rows.append(score_row(out, self.anns[q.question_id], q, retained))
            if n % 25 == 0:
                log.info("%s: %d/%d questions (%.0fs)", name, n, len(self.questions), time.perf_counter() - t0)
        summary = summarize(rows)
        summary["memory"] = mm
        summary["method"] = name if self.split == "test" else f"{self.split}:{name}"
        summary["split"] = self.split
        summary["spec"] = spec
        summary["wall_s"] = round(time.perf_counter() - t0, 1)
        d = new_run_dir(results_dir, name if self.split == "test" else f"{self.split}_{name}")
        (d / "config.json").write_text(json.dumps(run_cfg, indent=2, default=str), encoding="utf-8")
        write_jsonl(d / "predictions.jsonl", preds)
        write_jsonl(d / "rows.jsonl", rows)
        (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (d / "summary.md").write_text(_method_md(summary), encoding="utf-8")
        log.info("%s done: acc=%.3f EM=%s -> %s", name, summary["answer"]["overall_accuracy"],
                 summary["answer"]["exact_match"], d)
        return summary


def _method_md(s) -> str:
    a, c, m = s["answer"], s["controller"], s["memory"]
    lines = [f"# {s['method']}", "", f"Spec: `{json.dumps(s['spec'])}`", "", f"n = {s['n']} test questions (synthetic benchmark)", "",
             "| metric | value |", "|---|---|"]
    for k, v in [("overall accuracy", a["overall_accuracy"]), ("95% CI", a["overall_accuracy_ci95"]),
                 ("exact match (value qs)", a["exact_match"]), ("false-answer rate", a["false_answer_rate"]),
                 ("unsupported-answer rate", a["unsupported_answer_rate"]), ("cross-stage accuracy", a["cross_stage_accuracy"]),
                 ("retrieval trigger rate", c["retrieval_trigger_rate"]),
                 ("unnecessary retrieval", c["unnecessary_retrieval_rate"]), ("missed retrieval", c["missed_retrieval_rate"]),
                 ("Recall@K", s["evidence"]["recall_at_k"]), ("MRR", s["evidence"]["mrr"]),
                 ("unjustified causal claims", s["verification"]["unjustified_causal_claim_rate"]),
                 ("memory retention (bytes)", m["retention_ratio_bytes"]),
                 ("relevant-evidence retention", m["relevant_evidence_retention"]),
                 ("mean end-to-end latency (s)", s["latency"].get("total_s", {}).get("mean"))]:
        lines.append(f"| {k} | {v if not isinstance(v, float) else round(v, 4)} |")
    lines += ["", "## By family", "", "| family | n | accuracy | retrieval rate |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {v['accuracy']} | {v['retrieval_rate']} |" for k, v in s["by_family"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--methods", default="")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--memory-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--sweep-policies", default="", help="comma list of sweep policy names (default: all)")
    ap.add_argument("--sweep-budgets", default="", help="comma list of budgets (default: config)")
    ap.add_argument("--threads", type=int, default=None, help="torch threads (does not change results)")
    args = ap.parse_args()
    cfg = load_config(overrides={"models.mamba_threads": args.threads} if args.threads else None)
    seed_everything(cfg["seed"])
    exp = yaml.safe_load((PROJECT_ROOT / "config" / "experiments.yaml").read_text(encoding="utf-8"))
    runner = Runner(cfg, args.data, args.rebuild, args.split)
    results_dir = resolve_path(cfg, "paths.results_dir")
    todo = []
    if args.methods:
        names = list(exp["methods"]) if args.methods == "all" else args.methods.split(",")
        todo += [(n, exp["methods"][n]) for n in names]
    if args.sweep:
        sw = exp["sweep"]
        only = set(args.sweep_policies.split(",")) if args.sweep_policies else None
        budgets = [float(x) for x in args.sweep_budgets.split(",")] if args.sweep_budgets else sw["budgets"]
        for b in budgets:
            for pname, p in sw["policies"].items():
                if only and pname not in only:
                    continue
                todo.append((f"sweep_{pname}_{b:.2f}", {**p, "budget": b, "mode": sw["mode"], "verify": sw["verify"]}))
        if not only or "store_all" in only:
            todo.append(("sweep_store_all_1.00", {"policy": "store_all", "budget": 1.0, "mode": sw["mode"],
                                                  "verify": sw["verify"]}))
    out = []
    for name, spec in todo:
        out.append(runner.run_method(name, spec, results_dir, args.memory_only))
    if args.memory_only:
        d = new_run_dir(results_dir, f"memory_only{('_' + args.tag) if args.tag else ''}")
        (d / "summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(d)


if __name__ == "__main__":
    main()
