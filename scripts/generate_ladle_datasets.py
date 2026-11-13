"""Write the two synthetic ladle datasets (fictional data):

    python scripts/generate_ladle_datasets.py [--scenarios 4] [--seed 2027]

-> data/ladle_benchmark/short   incident-shift logs only (fits in Mamba's context window)
-> data/ladle_benchmark/long    the same shift + thousands of later/older records (4 stages)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import PROJECT_ROOT  # noqa: E402
from agentmamba.data.ladle_benchmark import write_datasets  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "ladle_benchmark"))
    ap.add_argument("--scenarios", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args()
    print(json.dumps(write_datasets(a.out, a.seed, a.scenarios), indent=2))
