"""Generate the synthetic industrial-log benchmark (fictional data) used by run_context_growth.py.

    python scripts/generate_log_benchmark.py                  # -> data/log_benchmark/v1 (seed 2026)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import PROJECT_ROOT  # noqa: E402
from agentmamba.data.log_benchmark import write_benchmark  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/log_benchmark/v1")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true", help="delete and regenerate an existing benchmark")
    args = ap.parse_args()
    out = PROJECT_ROOT / args.out
    if out.exists():
        if not args.force:
            sys.exit(f"{out} exists; pass --force to regenerate (the output is deterministic for a seed)")
        shutil.rmtree(out)
    print(json.dumps(write_benchmark(out, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
