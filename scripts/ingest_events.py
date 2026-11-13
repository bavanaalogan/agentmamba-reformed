"""Ingest a file into the application memory through the configured gate + budget.

    python scripts/ingest_events.py path\\to\\events.jsonl
    python scripts/ingest_events.py path\\to\\procedure.txt --stage MAINTENANCE --source-type authorized
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import load_config
from agentmamba.data.document_loader import load_any
from agentmamba.pipeline.service import AgentService


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--stage", default="MAINTENANCE")
    ap.add_argument("--source-type", default="authorized")
    ap.add_argument("--budget-bytes", type=int, default=None)
    args = ap.parse_args()
    p = Path(args.path)
    kw = {"stage": args.stage, "source_type": args.source_type} if p.suffix.lower() in (".txt", ".md") else {}
    events = load_any(p, **kw)
    rep = AgentService(load_config()).ingest(events, args.budget_bytes)
    rep.pop("decisions", None)
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
