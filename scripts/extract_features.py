"""Compute and cache per-record sentence embeddings and Mamba features for a dataset.
Cached by content hash in artifacts/features, so later steps never recompute them.

    python scripts/extract_features.py --data data/synthetic/main [--limit 256] [--no-mamba]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import load_config, resolve_path
from agentmamba.data.dataset_loader import load_dataset
from agentmamba.gate.features import FeatureStore
from agentmamba.models.embedding_model import SentenceEmbedder
from agentmamba.models.mamba_backbone import MambaBackbone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-mamba", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    cfg = load_config()
    ds = load_dataset(args.data)
    events = ds.events[: args.limit] if args.limit else ds.events
    fs = FeatureStore(resolve_path(cfg, "paths.feature_cache"))
    report = {"n_events": len(events)}
    t = time.perf_counter()
    emb = SentenceEmbedder(cfg["models"]["embedding_id"])
    fs.get("emb", emb.model_id, events, emb.encode)
    report["embedding_seconds"] = round(time.perf_counter() - t, 2)
    if not args.no_mamba:
        mb = MambaBackbone(cfg["models"]["mamba_id"], cfg["models"]["mamba_threads"])
        t = time.perf_counter()
        mt = cfg["models"]["mamba_feature_max_tokens"]
        fs.get("mamba", mb.model_id + f"@{mt}", events,
               lambda texts: mb.record_features(texts, max_tokens=mt, batch_size=args.batch))
        report["mamba_seconds"] = round(time.perf_counter() - t, 2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
