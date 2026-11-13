"""Loading of a generated dataset directory (events, questions, manifest).
Gold annotations are deliberately NOT loaded here; see ``evidence_annotations``."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentmamba.memory.schema import Event, Question


@dataclass
class Dataset:
    root: Path
    events: list[Event]
    withheld: list[Event]
    questions: list[Question]
    manifest: dict

    def split_of(self, campaign_id: str | None) -> str | None:
        return self.manifest["campaign_splits"].get(campaign_id)

    def events_for(self, split: str) -> list[Event]:
        return [e for e in self.events if self.split_of(e.campaign_id) == split]

    def questions_for(self, split: str) -> list[Question]:
        return [q for q in self.questions if q.split == split]

    def event_index(self) -> dict[str, Event]:
        return {e.event_id: e for e in self.events}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_dataset(root: str | Path) -> Dataset:
    root = Path(root)
    manifest_path = root / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; run scripts/generate_data.py first")
    return Dataset(
        root=root,
        events=[Event.from_dict(d) for d in _jsonl(root / "events.jsonl")],
        withheld=[Event.from_dict(d) for d in _jsonl(root / "withheld.jsonl")],
        questions=[Question(**d) for d in _jsonl(root / "questions.jsonl")],
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
    )
