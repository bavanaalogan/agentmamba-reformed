"""Generate the synthetic plant-wide benchmark.

    python scripts/generate_data.py                     # full benchmark -> data/synthetic/main
    python scripts/generate_data.py --smoke             # tiny dataset  -> data/synthetic/smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import PROJECT_ROOT, load_config
from agentmamba.data.synthetic_generator import GenConfig, write_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    s = cfg["synthetic"]
    if args.smoke:
        gen = GenConfig(n_campaigns=5, heats_per_campaign=24, ladles_per_campaign=4,
                        noise_logs_per_heat=2, splits={"train": 3, "val": 1, "test": 1}, seed=cfg["seed"])
        out = args.out or PROJECT_ROOT / "data" / "synthetic" / "smoke"
    else:
        gen = GenConfig(n_campaigns=s["n_campaigns"], heats_per_campaign=s["heats_per_campaign"],
                        ladles_per_campaign=s["ladles_per_campaign"], noise_logs_per_heat=s["noise_logs_per_heat"],
                        splits=dict(s["splits"]), seed=cfg["seed"])
        out = args.out or PROJECT_ROOT / "data" / "synthetic" / "main"
    manifest = write_dataset(gen, out)
    print(json.dumps({"out": str(out), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
