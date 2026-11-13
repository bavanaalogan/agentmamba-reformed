"""Version/supersession handling and conflict detection among verified candidates."""
from __future__ import annotations

from collections.abc import Callable


def drop_superseded(cands: list, superseded_by: Callable[[str], str | None]) -> tuple[list, list[dict]]:
    """Remove candidates whose record is superseded by another available record; then keep only
    the highest version per document."""
    kept, notes = [], []
    for c in cands:
        newer = superseded_by(c.record.event_id)
        if newer:
            notes.append({"record_id": c.record.event_id, "superseded_by": newer})
        else:
            kept.append(c)
    best: dict = {}
    for c in kept:
        doc = c.record.doc_id or c.record.event_id
        if doc not in best or (c.record.version or 0) > (best[doc].record.version or 0):
            best[doc] = c
    dropped = [c for c in kept if c not in best.values()]
    for c in dropped:
        notes.append({"record_id": c.record.event_id, "superseded_by": "higher version available"})
    return [c for c in kept if c in best.values()], notes


def detect_conflict(cands: list, single_valued: bool) -> dict | None:
    """Distinct values asserted by distinct records for a single-valued slot == conflict."""
    if not single_valued:
        return None
    by_value: dict[str, list] = {}
    for c in cands:
        by_value.setdefault(c.key(), []).append(c)
    if len(by_value) <= 1:
        return None
    return {"values": [{"value": v[0].value, "record_ids": sorted({c.record.event_id for c in v})}
                       for v in by_value.values()]}
