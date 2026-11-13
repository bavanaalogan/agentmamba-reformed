"""Gold annotations, isolated from live inference.

Access is split-scoped: training code must ask for ``train``/``val`` explicitly, and test
annotations can only be loaded with ``purpose="evaluation"``. The live pipeline
(``pipeline/``, ``retrieval/``, ``verification/``) never imports this module; a unit test
enforces that.
"""
from __future__ import annotations

import json
from pathlib import Path

from agentmamba.memory.schema import Annotation

ALLOWED = {"train": {"training", "evaluation"}, "val": {"training", "tuning", "evaluation"}, "test": {"evaluation"}}


class AnnotationAccessError(PermissionError):
    pass


def load_annotations(root: str | Path, split: str, purpose: str, questions_split: dict[str, str]) -> dict[str, Annotation]:
    """Return {question_id: Annotation} for one split. ``questions_split`` maps question_id->split
    (from questions.jsonl) so annotations from other splits are never returned."""
    if purpose not in ALLOWED.get(split, set()):
        raise AnnotationAccessError(f"annotations of split {split!r} may not be used for {purpose!r}")
    out = {}
    with open(Path(root) / "annotations.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            a = Annotation(**json.loads(line))
            if questions_split.get(a.question_id) == split:
                out[a.question_id] = a
    return out
