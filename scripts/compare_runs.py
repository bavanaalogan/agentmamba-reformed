"""Regression check: compare two experiment runs question by question.

    python scripts/compare_runs.py experiments/results/<old_run> experiments/results/<new_run>

Compared per question: answer status and value, verification status, cited records,
retrieval decision and retrieved records. Exit code 1 if anything differs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FIELDS = ("answer_status", "answer_value", "verification_status", "retrieval_triggered", "retrieved_record_ids")


def load(run: Path) -> dict:
    out = {}
    for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        p = json.loads(line)
        out[p["query_id"]] = {**{k: p.get(k) for k in FIELDS},
                              "cited": [e["record_id"] for e in p.get("evidence", [])]}
    return out


def main() -> None:
    a, b = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    diffs = []
    for q in sorted(set(a) | set(b)):
        if q not in a or q not in b:
            diffs.append((q, "missing in one run", None, None))
            continue
        for k in (*FIELDS, "cited"):
            if a[q][k] != b[q][k]:
                diffs.append((q, k, a[q][k], b[q][k]))
    print(f"{len(set(a) & set(b))} common questions; {len({d[0] for d in diffs})} differ")
    for d in diffs[:40]:
        print(" ", d)
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
