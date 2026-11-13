"""Run directories: machine-readable JSON/JSONL plus a Markdown summary. Never overwrites."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def new_run_dir(results_dir: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = Path(results_dir) / f"{stamp}_{name}"
    n = 1
    while d.exists():
        d = Path(results_dir) / f"{stamp}_{name}_{n}"
        n += 1
    d.mkdir(parents=True)
    return d


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return f"[{v[0]:.3f}, {v[1]:.3f}]"
    return str(v)


def markdown_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(fmt(_get(r, c[0])) for c in cols) + " |\n" for r in rows)
    return head + body


def _get(d: dict, dotted: str):
    for p in dotted.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d
