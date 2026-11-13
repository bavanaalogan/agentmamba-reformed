"""Validate a generated dataset (split integrity, evidence ids, label consistency) and print
family / split statistics.  python scripts/validate_data.py --data data/synthetic/main"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import PROJECT_ROOT
from agentmamba.data.dataset_loader import load_dataset
from agentmamba.data.dataset_splits import validate_dataset
from agentmamba.data.evidence_annotations import load_annotations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data" / "synthetic" / "main"))
    args = ap.parse_args()
    ds = load_dataset(args.data)
    qsplit = {q.question_id: q.split for q in ds.questions}
    anns = {}
    for split in ("train", "val", "test"):
        anns.update(load_annotations(args.data, split, "evaluation", qsplit))
    problems = validate_dataset(ds, anns)
    stats = {
        "events": len(ds.events), "withheld": len(ds.withheld), "questions": len(ds.questions),
        "events_by_split": Counter(ds.split_of(e.campaign_id) for e in ds.events),
        "questions_by_split": Counter(q.split for q in ds.questions),
        "families": Counter(a.family for a in anns.values()),
        "answer_kinds": Counter(a.answer_kind for a in anns.values()),
        "event_types": Counter(e.event_type for e in ds.events),
        "problems": problems[:50], "n_problems": len(problems),
    }
    print(json.dumps(stats, indent=2, default=dict))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
