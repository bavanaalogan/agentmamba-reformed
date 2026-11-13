"""SQLite store, FAISS index, episodic memory persistence/recovery, lifecycle, traversal."""
from __future__ import annotations

import numpy as np
import pytest

from agentmamba.memory import lifecycle
from agentmamba.memory.episodic_memory import EpisodicMemory
from agentmamba.memory.relationship_store import InMemoryRelationshipGraph, SQLiteRelationshipGraph
from agentmamba.memory.schema import Relationship
from agentmamba.memory.sqlite_store import SQLiteStore
from agentmamba.memory.vector_index import VectorIndex
from agentmamba.retrieval.relationship_traversal import traverse
from tests.conftest import make_event


def _events():
    return [make_event("A1", "ladle_assignment", text="Heat H1 tapped into ladle L1.", heat_id="H1", ladle_id="L1"),
            make_event("I1", text="Refractory inspection of ladle L1: hairline crack at slag line.", ladle_id="L1",
                       observations=[{"name": "finding", "value": "hairline crack at slag line"}]),
            make_event("I2", text="Refractory inspection of ladle L2: lining condition normal.", ladle_id="L2",
                       ts="2024-01-02T10:00:00"),
            make_event("C1", "cast_record", "CASTING", text="Caster cast heat H1 in sequence S1 producing slab B1.",
                       heat_id="H1", cast_seq_id="S1", slab_id="B1")]


# ---------------------------------------------------------------- SQLite
def test_sqlite_crud_filters_versions_and_injection_safety(tmp_path):
    st = SQLiteStore(tmp_path / "m.sqlite3")
    for i, e in enumerate(_events(), 1):
        st.insert_event(e, i)
    assert st.get_event("I1").text.startswith("Refractory")
    assert [e.event_id for e in st.query_events(ladle_id="L1")] == ["A1", "I1"]
    assert [e.event_id for e in st.query_events(start="2024-01-02")] == ["I2"]
    assert st.query_events(text_like="'; DROP TABLE events; --") == []
    assert st.stats()["records"] == 4
    upd = make_event("I2", text="Corrected: lining condition normal, 150 mm.", ladle_id="L2", ts="2024-01-02T10:00:00")
    st.update_event(upd)
    assert st.get_event("I2").text.startswith("Corrected")
    assert [h["change_type"] for h in st.version_history("I2")] == ["insert", "update"]
    st.hard_delete("I2", "test")
    assert st.get_event("I2") is None and st.version_history("I2")[-1]["change_type"] == "delete:test"


def test_sqlite_backup_restore(tmp_path):
    st = SQLiteStore(tmp_path / "m.sqlite3")
    st.insert_event(_events()[0], 1)
    st.backup(tmp_path / "bk" / "b.sqlite3")
    restored = SQLiteStore.restore(tmp_path / "bk" / "b.sqlite3", tmp_path / "r.sqlite3")
    assert restored.get_event("A1") is not None


# ---------------------------------------------------------------- FAISS
def test_vector_index_search_persist_remove_filter(tmp_path):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(5, 8)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    idx = VectorIndex(8, "m", tmp_path / "ix")
    assert idx.search(v[0], 3) == []                      # empty index is safe
    idx.add(v, [10, 11, 12, 13, 14])
    assert idx.search(v[2], 1)[0][0] == 12 and idx.search(v[2], 1)[0][1] == pytest.approx(1.0, abs=1e-5)
    assert [i for i, _ in idx.search(v[2], 5, allowed_ids=[10, 14])] in ([10, 14], [14, 10])
    assert idx.search(v[2], 5, allowed_ids=[]) == []
    idx.save()
    again = VectorIndex.load(tmp_path / "ix", "m", 8)
    assert again.ids() == [10, 11, 12, 13, 14]
    assert again.remove([12]) == 1 and 12 not in [i for i, _ in again.search(v[2], 5)]
    with pytest.raises(ValueError):
        VectorIndex.load(tmp_path / "ix", "other-model")
    with pytest.raises(ValueError):
        idx.add(np.zeros((1, 3), dtype="float32"), [99])
    (tmp_path / "ix.meta.json").write_text((tmp_path / "ix.meta.json").read_text().replace('"count": 5', '"count": 4'))
    with pytest.raises(ValueError):
        VectorIndex.load(tmp_path / "ix", "m")
    with pytest.raises(FileNotFoundError):
        VectorIndex.load(tmp_path / "missing", "m")


# ---------------------------------------------------------------- episodic memory
def test_memory_survives_restart_and_ids_resolve(tmp_path, embedder):
    mem = EpisodicMemory(tmp_path, embedder)
    vids = mem.write(_events())
    mem.close()
    mem2 = EpisodicMemory(tmp_path, embedder)                 # "restart"
    assert mem2.recovery_note is None and mem2.validate()["ok"]
    q = embedder.encode(["hairline crack at slag line ladle L1"])[0]
    top_vid, _ = mem2.index.search(q, 1)[0]
    assert mem2.store.by_vector_ids([top_vid])[top_vid].event_id == "I1"
    assert set(vids) == set(mem2.index.ids())
    assert any(r.rel_type == "carried_by" for r in mem2.store.relationships_of("H1"))


def test_delete_never_returns_stale_records(tmp_path, embedder):
    mem = EpisodicMemory(tmp_path, embedder)
    mem.write(_events())
    lifecycle.delete(mem, ["I1"])
    q = embedder.encode(["hairline crack at slag line"])[0]
    hits = mem.store.by_vector_ids([v for v, _ in mem.index.search(q, 10)])
    assert "I1" not in {e.event_id for e in hits.values()} and mem.validate()["ok"]
    assert mem.store.relationships_of("I1") == []


def test_missing_or_inconsistent_index_is_rebuilt_from_sqlite(tmp_path, embedder):
    mem = EpisodicMemory(tmp_path, embedder)
    mem.write(_events())
    mem.close()
    (tmp_path / "memory.faiss").unlink()
    mem2 = EpisodicMemory(tmp_path, embedder)
    assert "rebuilt" in mem2.recovery_note and mem2.validate()["ok"] and mem2.index.size == 4
    mem2.index.remove([mem2.index.ids()[0]])
    mem2.index.save()
    mem2.close()
    mem3 = EpisodicMemory(tmp_path, embedder)
    assert mem3.validate()["ok"] and mem3.index.size == 4
    mem4 = EpisodicMemory(tmp_path, embedder, auto_recover=False)
    assert mem4.validate()["ok"]


def test_supersession_and_update_reembed(tmp_path, embedder):
    mem = EpisodicMemory(tmp_path, embedder)
    old = make_event("D1", "procedure_doc", "MAINTENANCE", text="Procedure P rev 1: limit 90 min.", doc_id="P", version=1)
    mem.write([old])
    new = make_event("D2", "procedure_doc", "MAINTENANCE", text="Procedure P rev 2: limit 60 min.", doc_id="P", version=2)
    lifecycle.supersede(mem, new, "D1")
    assert mem.store.superseded_by("D1") == "D2"
    with pytest.raises(KeyError):
        lifecycle.supersede(mem, make_event("D3"), "NOPE")
    fix = make_event("D2", "procedure_doc", "MAINTENANCE", text="Procedure P rev 2: limit 65 min.", doc_id="P",
                     version=2, supersedes="D1")
    lifecycle.update(mem, fix)
    q = embedder.encode(["Procedure P rev 2: limit 65 min."])[0]
    vid, sim = mem.index.search(q, 1)[0]
    assert mem.store.by_vector_ids([vid])[vid].text.endswith("65 min.") and sim == pytest.approx(1.0, abs=1e-5)


def test_lifecycle_backup_restore(tmp_path, embedder):
    mem = EpisodicMemory(tmp_path / "live", embedder)
    mem.write(_events())
    lifecycle.backup(mem, tmp_path / "bk")
    restored = lifecycle.restore(tmp_path / "bk", tmp_path / "restored", embedder)
    assert restored.store.stats()["records"] == 4 and restored.validate()["ok"]


# ---------------------------------------------------------------- traversal
def test_traversal_depth_cycles_hubs_and_status():
    rels = [Relationship("H1", "heat", "L1", "ladle", "carried_by", "documented", "A1"),
            Relationship("H2", "heat", "L1", "ladle", "carried_by", "documented", "A2"),
            Relationship("B1", "slab", "H1", "heat", "cast_from_heat", "documented", "C1"),
            Relationship("K1", "coil", "B1", "slab", "rolled_from", "documented", "R1"),
            Relationship("K1", "coil", "X9", "slab", "rolled_from", "inferred", None),
            Relationship("B1", "slab", "K1", "coil", "loop", "documented", "Z")]          # cycle
    g = InMemoryRelationshipGraph(rels=rels)
    r = traverse(g, {"K1": "coil"}, max_depth=3)
    assert set(r) == {"K1", "B1", "H1", "L1"}             # H2 not reached through the ladle hub
    assert "X9" not in r                                    # inferred edges are not followed
    assert r["L1"].depth == 3 and [p.rel_type for p in r["L1"].path][1:] == ["cast_from_heat", "carried_by"]
    assert set(traverse(g, {"K1": "coil"}, max_depth=1)) == {"K1", "B1"}
    assert len(traverse(g, {"K1": "coil"}, max_depth=9, max_nodes=2)) == 2
    assert {"H1", "H2"} <= set(traverse(g, {"L1": "ladle"}, max_depth=1))   # start node may be a hub


def test_sqlite_graph_matches_in_memory(tmp_path, embedder):
    mem = EpisodicMemory(tmp_path, embedder)
    mem.write(_events())
    a = traverse(SQLiteRelationshipGraph(mem.store), {"B1": "slab"}, max_depth=2)
    b = traverse(InMemoryRelationshipGraph(_events()), {"B1": "slab"}, max_depth=2)
    assert set(a) == set(b) == {"B1", "H1", "S1", "L1"}
