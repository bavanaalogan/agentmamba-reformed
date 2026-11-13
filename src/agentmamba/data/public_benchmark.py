"""Category A benchmark on REAL public data (UCI #851, DAEWOO Steel energy readings, CC BY 4.0).

Records are real (daily aggregates of the published 15-minute rows, each keeping its row
range). Questions are TEMPLATED by us; expected answers are read directly from the data.
This tests ingestion, date-constrained retrieval, provenance and abstention on real records.
It does NOT test cross-stage reasoning (the dataset has no cross-stage identifiers).

Split by month: train = Jan-Aug, val = Sep-Oct, test = Nov-Dec 2018.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from agentmamba.data.document_loader import load_uci_851
from agentmamba.memory.schema import Annotation, Question

ASSET = "UCI851-PLANT"
TEMPLATES = {
    "energy_peak_on_date": ("peak_usage", [f"What peak 15-minute energy usage was recorded at {ASSET} on {{d}}?",
                                           f"On {{d}}, what was the highest 15-minute usage reading at {ASSET}?"]),
    "energy_total_on_date": ("total_usage", [f"What total energy usage was recorded at {ASSET} on {{d}}?",
                                             f"How much energy did {ASSET} use in total on {{d}}?"]),
    "energy_load_type_on_date": ("dominant_load_type", [f"What was the dominant load type at {ASSET} on {{d}}?"]),
}


def split_of(day: str) -> str:
    m = int(day[5:7])
    return "train" if m <= 8 else ("val" if m <= 10 else "test")


def build(csv_path: str | Path, out_dir: str | Path, seed: int = 13) -> dict:
    rng = random.Random(seed)
    events = load_uci_851(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qs, anns = [], []
    for ev in events:
        day = ev.timestamp[:10]
        qtype = rng.choice(sorted(TEMPLATES))
        fld, temps = TEMPLATES[qtype]
        o = ev.obs(fld)
        qid = f"UCI851-Q{len(qs) + 1:04d}"
        qs.append(Question(qid, "UCI851", split_of(day), rng.choice(temps).format(d=day), []))
        anns.append(Annotation(qid, "P01_public_date_lookup", qtype, ["UTILITIES"], "value",
                               f"{o.value} {o.unit}" if o.unit else str(o.value), True, [ev.event_id], [],
                               [ev.source_ref], False, True, False, True, "real public record; templated question"))
    for n, day in enumerate(["2019-03-01", "2019-06-15", "2017-12-31", "2019-01-20"]):   # outside coverage
        qid = f"UCI851-Q{len(qs) + 1:04d}"
        sp = ["train", "val", "test", "test"][n]
        fld, temps = TEMPLATES["energy_peak_on_date"]
        qs.append(Question(qid, "UCI851", sp, temps[0].format(d=day), []))
        anns.append(Annotation(qid, "P02_public_out_of_coverage", "energy_peak_on_date", ["UTILITIES"], "abstain", None,
                               False, [], [], [], False, True, False, True, "date outside the dataset coverage"))
    _dump(out / "events.jsonl", [e.to_dict() for e in events])
    _dump(out / "withheld.jsonl", [])
    _dump(out / "questions.jsonl", [q.to_dict() for q in qs])
    _dump(out / "annotations.jsonl", [a.to_dict() for a in anns])
    manifest = {"source": "UCI #851 Steel Industry Energy Consumption (CC BY 4.0), DOI 10.24432/C52G8C",
                "split_unit": "calendar month (train Jan-Aug, val Sep-Oct, test Nov-Dec 2018)",
                "campaign_splits": {"UCI851": "mixed-by-month"}, "synthetic": False,
                "questions_templated": True, "seed": seed,
                "counts": {"events": len(events), "questions": len(qs)}}
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _dump(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
