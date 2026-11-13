"""Ask the application memory a question from the command line.

    python scripts/run_query.py "Which ladle carried heat C03-H0012?" --context C03-E00123 --mode conditional
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import load_config
from agentmamba.pipeline.agent import QueryRequest
from agentmamba.pipeline.service import AgentService


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--context", nargs="*", default=[], help="ids of stored records used as initial context")
    ap.add_argument("--mode", default="conditional", choices=["conditional", "always", "never"])
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--full", action="store_true", help="print the whole decision trace")
    args = ap.parse_args()
    svc = AgentService(load_config())
    ctx = [e for i in args.context if (e := svc.memory.store.get_event(i)) is not None]
    out = svc.query(QueryRequest(args.question, ctx, args.mode, verification=not args.no_verify))
    keys = None if args.full else ["query_id", "initial_answer", "initial_status", "controller", "retrieval_triggered",
                                   "retrieved_record_ids", "verification_status", "causal_claim_status",
                                   "answer_status", "final_answer", "abstention_reason", "evidence", "timings"]
    print(json.dumps(out if keys is None else {k: out.get(k) for k in keys}, indent=2, default=str))


if __name__ == "__main__":
    main()
