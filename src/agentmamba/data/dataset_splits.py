"""Split-integrity and annotation validation for generated datasets."""
from __future__ import annotations

from collections import defaultdict

from agentmamba.data.dataset_loader import Dataset
from agentmamba.memory.schema import Annotation


def validate_dataset(ds: Dataset, annotations: dict[str, Annotation]) -> list[str]:
    """Return a list of problems (empty == valid). Checks:
    * every event/question belongs to exactly one split via its campaign;
    * no entity identifier appears in more than one split (leakage);
    * evidence and initial-context ids exist and stay inside the question's campaign;
    * withheld events are never part of the delivered stream;
    * every annotation's answer_kind is consistent with its other labels."""
    problems: list[str] = []
    split_of = ds.manifest["campaign_splits"]
    ev = ds.event_index()
    entity_splits: dict[str, set] = defaultdict(set)
    for e in ds.events:
        s = split_of.get(e.campaign_id)
        if s is None:
            problems.append(f"event {e.event_id} has no split")
            continue
        for ent in e.entity_ids():
            entity_splits[ent].add(s)
    for ent, ss in entity_splits.items():
        if len(ss) > 1:
            problems.append(f"entity {ent} appears in splits {sorted(ss)}")
    withheld_ids = {e.event_id for e in ds.withheld}
    if withheld_ids & set(ev):
        problems.append("withheld events present in delivered stream")
    for q in ds.questions:
        if split_of.get(q.campaign_id) != q.split:
            problems.append(f"question {q.question_id} split mismatch")
        for cid in q.initial_context_ids:
            if cid not in ev or ev[cid].campaign_id != q.campaign_id:
                problems.append(f"question {q.question_id} context {cid} invalid")
        a = annotations.get(q.question_id)
        if a is None:
            continue
        for eid in a.required_evidence_ids:
            if eid not in ev or ev[eid].campaign_id != q.campaign_id:
                problems.append(f"question {q.question_id} evidence {eid} missing or cross-campaign")
        if a.answer_kind == "value" and not a.expected_answer:
            problems.append(f"question {q.question_id} value answer missing")
        if a.answer_kind in ("abstain", "clarify", "conflict") and a.answerable:
            problems.append(f"question {q.question_id} {a.answer_kind} marked answerable")
        if a.initial_sufficient and not set(a.required_evidence_ids) <= set(q.initial_context_ids):
            problems.append(f"question {q.question_id} initial_sufficient but evidence not in context")
    return problems
