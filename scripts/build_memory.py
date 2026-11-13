"""Build the application memory (storage/app) from a dataset split through the configured
gate and budget, or rebuild/validate its FAISS index.

    python scripts/build_memory.py --data data/synthetic/main --split test --budget-fraction 0.10
    python scripts/build_memory.py --rebuild-index
    python scripts/build_memory.py --public-uci851   # adds the public UCI #851 daily summaries (store-all)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import PROJECT_ROOT, load_config, resolve_path
from agentmamba.data.dataset_loader import load_dataset
from agentmamba.data.document_loader import load_uci_851
from agentmamba.gate.features import FeatureStore
from agentmamba.gate.train import GateArtifact
from agentmamba.memory.episodic_memory import EpisodicMemory
from agentmamba.models.embedding_model import SentenceEmbedder
from agentmamba.models.mamba_backbone import MambaBackbone
from agentmamba.pipeline.builder import build_memory, gate_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--budget-fraction", type=float, default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--storage", default=None)
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--public-uci851", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    storage = Path(args.storage) if args.storage else resolve_path(cfg, "paths.storage_dir") / "app"
    emb = SentenceEmbedder(cfg["models"]["embedding_id"])
    if args.rebuild_index:
        mem = EpisodicMemory(storage, emb, auto_recover=False)
        print(json.dumps(mem.rebuild_index(), indent=2))
        return
    if args.public_uci851:
        # The gate was trained on synthetic event types; applying it to these out-of-distribution
        # public records would be unjustified, so they go through the logged pipeline under an
        # EXPLICIT store_all policy (every decision is recorded as such).
        from agentmamba.gate.predictor import build_scorer
        from agentmamba.pipeline.ingestion import FeatureProvider, IngestionPipeline

        mem = EpisodicMemory(storage, emb)
        evs = load_uci_851(PROJECT_ROOT / "data" / "raw" / "uci_851_steel_energy" / "Steel_industry_data.csv")
        pipe = IngestionPipeline(mem, FeatureProvider(emb, None), build_scorer("store_all"), "store_all(public)",
                                 False, run_tag="public:uci851")
        rep = pipe.ingest(evs, None)
        print(json.dumps({"written": rep.written, "duplicates": rep.duplicates,
                          "source": "UCI #851 Steel Industry Energy Consumption (CC BY 4.0)"}, indent=2))
        return
    ds = load_dataset(args.data)
    policy = args.policy or cfg["memory"]["policy"]
    frac = args.budget_fraction if args.budget_fraction is not None else cfg["memory"]["budget_fraction"]
    gate = GateArtifact.load(gate_path(cfg)) if policy == "learned" else None
    mb = MambaBackbone(cfg["models"]["mamba_id"], cfg["models"]["mamba_threads"])
    mem, rep = build_memory(cfg, storage, ds.events_for(args.split), policy=policy, budget_fraction=frac,
                            embedder=emb, feature_store=FeatureStore(resolve_path(cfg, "paths.feature_cache")),
                            backbone=mb, gate=gate, run_tag=f"app:{args.split}")
    out = {k: v for k, v in rep.to_dict(0).items() if k != "decisions"}
    out["stats"] = {k: v for k, v in mem.stats().items() if k in ("records", "relationships", "vectors", "record_bytes")}
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
