"""Record lifecycle operations on top of EpisodicMemory: supersession, in-place updates,
deletion, and backup/restore. Every operation is audited."""
from __future__ import annotations

from pathlib import Path

from agentmamba.data.event_linker import link_event
from agentmamba.memory.episodic_memory import EpisodicMemory
from agentmamba.memory.schema import Event


def supersede(mem: EpisodicMemory, new_event: Event, old_event_id: str) -> int:
    """Store ``new_event`` as the replacement of ``old_event_id``. The old record stays stored
    (for history and audit) but every retrieval/verification path treats it as superseded."""
    if mem.store.get_event(old_event_id) is None:
        raise KeyError(f"cannot supersede unknown record {old_event_id}")
    if new_event.supersedes != old_event_id:
        new_event.supersedes = old_event_id
        new_event.content_hash = ""
        new_event.__post_init__()
    [vid] = mem.write([new_event])
    mem.store.audit("supersede", {"new": new_event.event_id, "old": old_event_id})
    return vid


def update(mem: EpisodicMemory, ev: Event) -> None:
    """Correct a record in place; re-embed it so FAISS never serves stale text."""
    row = mem.store.row(ev.event_id)
    if row is None:
        raise KeyError(ev.event_id)
    ev.content_hash = ""
    ev.__post_init__()
    mem.store.update_event(ev)
    with mem.store.tx() as c:
        c.execute("DELETE FROM relationships WHERE source_event_id=?", (ev.event_id,))
        mem.store.insert_relationships(link_event(ev), conn=c)
    if row["vector_id"] is not None:
        mem.index.remove([row["vector_id"]])
        mem.index.add(mem.embedder.encode([ev.text]), [row["vector_id"]])
        mem.index.save()
    mem.store.audit("update", {"event_id": ev.event_id})


def delete(mem: EpisodicMemory, event_ids: list[str], reason: str = "user_delete") -> int:
    n = mem.delete(event_ids, reason)
    mem.store.audit("delete", {"event_ids": event_ids, "reason": reason})
    return n


def backup(mem: EpisodicMemory, dest_dir: str | Path) -> dict:
    dest = Path(dest_dir)
    db = mem.store.backup(dest / "memory.sqlite3")
    idx = mem.index.save(dest / "memory")
    return {"sqlite": str(db), "faiss": str(idx)}


def restore(backup_dir: str | Path, storage_dir: str | Path, embedder) -> EpisodicMemory:
    import shutil

    src, dst = Path(backup_dir), Path(storage_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("memory.sqlite3", "memory.faiss", "memory.meta.json"):
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)
    return EpisodicMemory(dst, embedder)
