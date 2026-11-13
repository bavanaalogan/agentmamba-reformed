"""Gate training labels.

A record is labelled useful (1) if it appears in the ``required_evidence_ids`` of at least
one question of the SAME split (train for fitting, val for threshold/model selection). That
covers records that answer a question and records needed to establish a documented
relationship chain (both are listed as required evidence by the generator).

Limitations (documented in RESEARCH_METHODOLOGY.md): future utility is not observable at
write time; these labels encode which records the benchmark's *training* questions happened
to need, so records that would be useful for unasked questions are labelled 0 (label noise).
Test annotations are never accepted here (``load_annotations`` enforces purpose).
"""
from __future__ import annotations

import numpy as np

from agentmamba.memory.schema import Annotation, Event


def build_labels(events: list[Event], annotations: dict[str, Annotation]) -> np.ndarray:
    useful = {eid for a in annotations.values() for eid in a.required_evidence_ids}
    return np.array([1 if e.event_id in useful else 0 for e in events], dtype="int64")
