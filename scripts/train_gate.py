"""Train the learned memory gate variants, the intent classifier and the retrieval controller.

Uses ONLY train (fit) and val (selection / thresholds) annotations.

    python scripts/train_gate.py --data data/synthetic/main            # gates + intent + controller
    python scripts/train_gate.py --data ... --skip-controller
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import load_config, resolve_path
from agentmamba.data.dataset_loader import load_dataset
from agentmamba.data.evidence_annotations import load_annotations
from agentmamba.gate.features import FeatureStore, stream_novelty  # noqa: F401
from agentmamba.gate.label_builder import build_labels
from agentmamba.gate.train import train_gate
from agentmamba.logging_utils import get_logger
from agentmamba.models.embedding_model import SentenceEmbedder
from agentmamba.pipeline.agent import AgentMamba
from agentmamba.pipeline.builder import build_answerer, gate_path, train_controller, train_intent
from agentmamba.reproducibility import seed_everything
from agentmamba.retrieval.query_processor import QueryProcessor

log = get_logger("train_gate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--skip-controller", action="store_true")
    ap.add_argument("--public", default=str(Path(__file__).resolve().parents[1] / "data" / "public" / "uci851"),
                    help="Category A benchmark dir (train/val months also train the intent classifier/controller)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    ds = load_dataset(args.data)
    qsplit = {q.question_id: q.split for q in ds.questions}
    tr = load_annotations(args.data, "train", "training", qsplit)
    va = load_annotations(args.data, "val", "tuning", qsplit)
    summary = {}
    if not args.skip_gate:
        fs = FeatureStore(resolve_path(cfg, "paths.feature_cache"))
        emb_model = SentenceEmbedder(cfg["models"]["embedding_id"])
        mt = cfg["models"]["mamba_feature_max_tokens"]
        mamba_key = cfg["models"]["mamba_id"] + f"@{mt}"
        ev_tr, ev_va = ds.events_for("train"), ds.events_for("val")
        e_tr = fs.get("emb", emb_model.model_id, ev_tr, emb_model.encode)
        e_va = fs.get("emb", emb_model.model_id, ev_va, emb_model.encode)

        def no_compute(texts):
            raise RuntimeError("Mamba features missing from cache; run scripts/extract_features.py first")

        m_tr = fs.get("mamba", mamba_key, ev_tr, no_compute)
        m_va = fs.get("mamba", mamba_key, ev_va, no_compute)
        y_tr, y_va = build_labels(ev_tr, tr), build_labels(ev_va, va)
        log.info("train events %d (pos %d), val events %d (pos %d)", len(ev_tr), y_tr.sum(), len(ev_va), y_va.sum())
        for kind, use_mamba in (("logreg", True), ("logreg", False), ("mlp", True)):
            art = train_gate(ev_tr, e_tr, m_tr, y_tr, ev_va, e_va, m_va, y_va, kind=kind, use_mamba=use_mamba,
                             emb_pca_dims=cfg["gate"]["embedding_pca_dims"], mamba_pca_dims=cfg["gate"]["mamba_pca_dims"],
                             window=cfg["gate"]["novelty_window"], seed=cfg["seed"], data_manifest=ds.manifest)
            path = art.save(gate_path(cfg, use_mamba, kind))
            summary[path.name] = {"val": art.meta["val"], "C": art.meta["C"], "gate_version": art.meta["gate_version"]}
            log.info("%s val=%s", path.name, json.dumps(art.meta["val"]))
    if not args.skip_controller:
        paths = AgentMamba.artifact_paths(cfg)
        q_tr, q_va, ev_index = ds.questions_for("train"), ds.questions_for("val"), ds.event_index()
        if args.public and Path(args.public).exists():      # Category A train/val questions (by month)
            pub = load_dataset(args.public)
            pq = {q.question_id: q.split for q in pub.questions}
            tr = {**tr, **load_annotations(args.public, "train", "training", pq)}
            va = {**va, **load_annotations(args.public, "val", "tuning", pq)}
            q_tr = q_tr + pub.questions_for("train")
            q_va = q_va + pub.questions_for("val")
            ev_index = {**ev_index, **pub.event_index()}
        clf = train_intent(q_tr, {k: a.question_type for k, a in tr.items()})
        clf.save(paths["intent"])
        qp = QueryProcessor(clf)
        val_intent_acc = sum(qp.parse(q.text).intent == va[q.question_id].question_type for q in q_va) / len(q_va)
        answerer = build_answerer(cfg)
        ctl = train_controller(qp, answerer, q_tr, tr, q_va, va, ev_index)
        ctl.meta["val_intent_accuracy"] = round(val_intent_acc, 4)
        ctl.save(paths["controller"])
        summary["controller"] = ctl.meta
        log.info("controller %s", json.dumps(ctl.meta))
    out = resolve_path(cfg, "paths.artifacts_dir") / "training_summary.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    out.write_text(json.dumps({**prev, **summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
