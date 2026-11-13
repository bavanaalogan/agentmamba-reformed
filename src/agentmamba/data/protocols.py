"""Staged evaluation protocols (data/protocols/<name>/): stage files, questions, and SEPARATE gold
annotations whose evidence is given as (file, line number) and resolved to record IDs only when
scoring. Questions carry no answers; gold is read only by evaluation code."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agentmamba.config import PROJECT_ROOT
from agentmamba.data.document_loader import load_any

ROOT = PROJECT_ROOT / "data" / "protocols"


def list_protocols() -> list[str]:
    return sorted(p.parent.name for p in ROOT.glob("*/protocol.json")) if ROOT.exists() else []


def load_protocol(name: str) -> dict:
    return json.loads((ROOT / name / "protocol.json").read_text(encoding="utf-8"))


def line_ids(name: str) -> dict[tuple[str, int], str]:
    """(file, line number) -> record ID, using the same loader as ingestion."""
    proto, out = load_protocol(name), {}
    for st in proto["stages"]:
        for e in load_any(ROOT / name / st["file"], stage=st["plant_stage"]):
            if "#line=" in e.source_ref:
                out[(st["file"], int(e.source_ref.rsplit("=", 1)[1]))] = e.event_id
    return out


def annotation(name: str, question_id: str, ids: dict | None = None) -> SimpleNamespace:
    """Gold for one question in the shape ``comparison.score_output`` expects (evaluation only)."""
    proto = load_protocol(name)
    g = json.loads((ROOT / name / "annotations" / "gold.json").read_text(encoding="utf-8"))[question_id]
    ids = ids if ids is not None else line_ids(name)
    return SimpleNamespace(question_id=question_id, scenario=name, split="protocol",
                           family=next(q["kind"] for q in proto["questions"] if q["question_id"] == question_id),
                           answer_kind=g["answer_kind"], expected_answer=g["expected_answer"], match=g["match"],
                           required_evidence_ids=[ids[(f, n)] for f, n in g["required_evidence"]])
