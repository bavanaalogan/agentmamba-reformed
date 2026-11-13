"""SQLite: the authoritative episodic store (records, relationships, versions, gate
decisions, audit trail). All statements are parameterized; writes use transactions."""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from agentmamba.memory.schema import ID_FIELDS, RAW_ID_FIELDS, Event, Relationship
from agentmamba.reproducibility import utc_now

SCHEMA_VERSION = 1
EVENT_COLUMNS = ("event_id", "campaign_id", "stage", "event_type", "timestamp", "asset_id", "heat_id", "ladle_id",
                 "cast_seq_id", "slab_id", "coil_id", "batch_id", "work_order_id", "doc_id", "severity", "version",
                 "supersedes", "source_type", "source_ref", "text", "content_hash")

DDL = f"""
CREATE TABLE IF NOT EXISTS events (
    {", ".join(c + (" TEXT PRIMARY KEY" if c == "event_id" else (" INTEGER" if c == "version" else " TEXT"))
               for c in EVENT_COLUMNS)},
    payload TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    vector_id INTEGER UNIQUE,
    gate_score REAL,
    ingested_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_heat ON events(heat_id);
CREATE INDEX IF NOT EXISTS ix_events_ladle ON events(ladle_id);
CREATE INDEX IF NOT EXISTS ix_events_coil ON events(coil_id);
CREATE INDEX IF NOT EXISTS ix_events_asset ON events(asset_id);
CREATE INDEX IF NOT EXISTS ix_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS ix_events_ts ON events(timestamp);
CREATE INDEX IF NOT EXISTS ix_events_sup ON events(supersedes);
CREATE TABLE IF NOT EXISTS relationships (
    rel_id TEXT PRIMARY KEY, src_id TEXT NOT NULL, src_kind TEXT NOT NULL, dst_id TEXT NOT NULL,
    dst_kind TEXT NOT NULL, rel_type TEXT NOT NULL, status TEXT NOT NULL, source_event_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_rel_src ON relationships(src_id);
CREATE INDEX IF NOT EXISTS ix_rel_dst ON relationships(dst_id);
CREATE INDEX IF NOT EXISTS ix_rel_source ON relationships(source_event_id);
CREATE TABLE IF NOT EXISTS event_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL, content_hash TEXT, payload TEXT,
    change_type TEXT NOT NULL, changed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gate_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL, decision TEXT NOT NULL, score REAL,
    policy TEXT, budget_bytes INTEGER, used_bytes INTEGER, reason TEXT, run_tag TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, action TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(DDL)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", SCHEMA_VERSION)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self):
        with self._lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    # ---------------------------------------------------------------- meta / audit
    def set_meta(self, key: str, value) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def get_meta(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def audit(self, action: str, detail: dict) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO audit (ts, action, detail) VALUES (?, ?, ?)", (utc_now(), action, json.dumps(detail)))

    def next_vector_id(self, n: int = 1) -> int:
        with self._lock:
            start = int(self.get_meta("next_vector_id", 1))
            self.set_meta("next_vector_id", start + n)
            return start

    # ---------------------------------------------------------------- events
    def insert_event(self, ev: Event, vector_id: int | None, gate_score: float | None = None, conn=None) -> None:
        row = {c: getattr(ev, c) for c in EVENT_COLUMNS}
        payload = json.dumps(ev.to_dict(), ensure_ascii=False)
        args = (*row.values(), payload, ev.size_bytes(), vector_id, gate_score, ev.ingested_at or utc_now())
        sql = (f"INSERT INTO events ({', '.join(EVENT_COLUMNS)}, payload, size_bytes, vector_id, gate_score, ingested_at) "
               f"VALUES ({', '.join('?' * (len(EVENT_COLUMNS) + 5))})")
        if conn is not None:
            conn.execute(sql, args)
            conn.execute("INSERT INTO event_versions (event_id, content_hash, payload, change_type, changed_at) "
                         "VALUES (?, ?, ?, 'insert', ?)", (ev.event_id, ev.content_hash, payload, utc_now()))
            return
        with self.tx() as c:
            self.insert_event(ev, vector_id, gate_score, conn=c)

    def get_event(self, event_id: str, include_inactive: bool = False) -> Event | None:
        sql = "SELECT payload FROM events WHERE event_id=?" + ("" if include_inactive else " AND status='active'")
        row = self.conn.execute(sql, (event_id,)).fetchone()
        return Event.from_dict(json.loads(row[0])) if row else None

    def get_events(self, ids: list[str]) -> list[Event]:
        if not ids:
            return []
        out = {}
        for chunk in _chunks(list(ids), 500):
            q = f"SELECT event_id, payload FROM events WHERE status='active' AND event_id IN ({','.join('?' * len(chunk))})"
            for r in self.conn.execute(q, chunk):
                out[r[0]] = Event.from_dict(json.loads(r[1]))
        return [out[i] for i in ids if i in out]

    def row(self, event_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        return dict(r) if r else None

    def by_vector_ids(self, vids: list[int]) -> dict[int, Event]:
        if not vids:
            return {}
        q = f"SELECT vector_id, payload FROM events WHERE status='active' AND vector_id IN ({','.join('?' * len(vids))})"
        return {r[0]: Event.from_dict(json.loads(r[1])) for r in self.conn.execute(q, [int(v) for v in vids])}

    def query_events(self, *, stage=None, event_type=None, asset_id=None, heat_id=None, ladle_id=None, coil_id=None,
                     batch_id=None, source_type=None, start=None, end=None, text_like=None, limit=200,
                     offset=0) -> list[Event]:
        clauses, args = ["status='active'"], []
        for col, val in (("stage", stage), ("event_type", event_type), ("asset_id", asset_id), ("heat_id", heat_id),
                         ("ladle_id", ladle_id), ("coil_id", coil_id), ("batch_id", batch_id),
                         ("source_type", source_type)):
            if val:
                clauses.append(f"{col}=?")
                args.append(val)
        if start:
            clauses.append("timestamp>=?")
            args.append(start)
        if end:
            clauses.append("timestamp<=?")
            args.append(end)
        if text_like:
            clauses.append("text LIKE ?")
            args.append(f"%{text_like}%")
        q = (f"SELECT payload FROM events WHERE {' AND '.join(clauses)} ORDER BY timestamp, event_id "
             f"LIMIT ? OFFSET ?")
        return [Event.from_dict(json.loads(r[0])) for r in self.conn.execute(q, (*args, int(limit), int(offset)))]

    def allowed_vector_ids(self, filters: dict) -> list[int]:
        clauses, args = ["status='active'", "vector_id IS NOT NULL"], []
        for col in ("stage", "asset_id", "heat_id", "ladle_id", "coil_id", "source_type", "event_type"):
            vals = filters.get(col)
            if vals:
                vals = [vals] if isinstance(vals, str) else list(vals)
                clauses.append(f"{col} IN ({','.join('?' * len(vals))})")
                args += vals
        if filters.get("start"):
            clauses.append("timestamp>=?")
            args.append(filters["start"])
        if filters.get("end"):
            clauses.append("timestamp<=?")
            args.append(filters["end"])
        return [r[0] for r in self.conn.execute(f"SELECT vector_id FROM events WHERE {' AND '.join(clauses)}", args)]

    def events_mentioning(self, entity_ids: list[str], event_types: tuple | list | None = None) -> list[Event]:
        if not entity_ids:
            return []
        ids = list(dict.fromkeys(entity_ids))
        cols = list(ID_FIELDS)
        ors = [f"{c} IN ({','.join('?' * len(ids))})" for c in cols]
        args = ids * len(cols)
        ors.extend(
            f"EXISTS (SELECT 1 FROM json_each(events.payload, '$.raw') AS raw_field "
            f"WHERE lower(replace(replace(raw_field.key, '-', '_'), ' ', '_'))=? "
            f"AND upper(CAST(raw_field.value AS TEXT)) IN ({','.join('?' * len(ids))}))"
            for _ in RAW_ID_FIELDS)
        args.extend(value for field in RAW_ID_FIELDS for value in (field, *(ident.upper() for ident in ids)))
        q = f"SELECT payload FROM events WHERE status='active' AND ({' OR '.join(ors)})"
        if event_types:
            q += f" AND event_type IN ({','.join('?' * len(event_types))})"
            args += list(event_types)
        return [Event.from_dict(json.loads(r[0])) for r in self.conn.execute(q + " ORDER BY timestamp", args)]

    def events_by_doc(self, doc_id: str, limit: int = 200) -> list[Event]:
        return [Event.from_dict(json.loads(r[0])) for r in self.conn.execute(
            "SELECT payload FROM events WHERE status='active' AND doc_id=? ORDER BY event_id LIMIT ?", (doc_id, limit))]

    def text_mentions(self, ident: str, event_types: tuple | None = ("document_chunk",), limit: int = 20,
                      day: str | None = None) -> list[Event]:
        """Records whose TEXT contains an identifier (used for free-text records whose structured
        identifier fields are empty). Parameterized LIKE; exact-token check is done by the caller.
        ``event_types=None`` searches every type; ``day`` keeps records timestamped that day or undated."""
        q = "SELECT payload FROM events WHERE status='active' AND text LIKE ?"
        args: list = [f"%{ident}%"]
        if event_types:
            q += f" AND event_type IN ({','.join('?' * len(event_types))})"
            args += list(event_types)
        if day:
            q += " AND (timestamp IS NULL OR substr(timestamp, 1, 10) = ?)"
            args.append(day)
        q += " ORDER BY event_id LIMIT ?"
        return [Event.from_dict(json.loads(r[0])) for r in self.conn.execute(q, (*args, limit))]

    def recent_events(self, limit: int = 400) -> list[Event]:
        """The ``limit`` most recently stored active records, returned oldest-first (arrival order =
        insertion order; within one ingestion batch records are inserted in timestamp order)."""
        rows = self.conn.execute("SELECT payload FROM events WHERE status='active' ORDER BY rowid DESC LIMIT ?",
                                 (limit,)).fetchall()
        return [Event.from_dict(json.loads(r[0])) for r in reversed(rows)]

    def superseded_by(self, event_id: str) -> str | None:
        r = self.conn.execute("SELECT event_id FROM events WHERE status='active' AND supersedes=?", (event_id,)).fetchone()
        return r[0] if r else None

    def active_vectors(self) -> list[tuple[int, str, str]]:
        return [(r[0], r[1], r[2]) for r in self.conn.execute(
            "SELECT vector_id, event_id, text FROM events WHERE status='active' AND vector_id IS NOT NULL ORDER BY vector_id")]

    def set_vector_id(self, event_id: str, vector_id: int | None) -> None:
        with self.tx() as c:
            c.execute("UPDATE events SET vector_id=? WHERE event_id=?", (vector_id, event_id))

    def hard_delete(self, event_id: str, reason: str, conn=None) -> int | None:
        """Remove a record and the relationships it documents (used for eviction / deletion)."""
        def _do(c):
            r = c.execute("SELECT vector_id, payload, content_hash FROM events WHERE event_id=?", (event_id,)).fetchone()
            if r is None:
                return None
            c.execute("DELETE FROM relationships WHERE source_event_id=?", (event_id,))
            c.execute("DELETE FROM events WHERE event_id=?", (event_id,))
            c.execute("INSERT INTO event_versions (event_id, content_hash, payload, change_type, changed_at) "
                      "VALUES (?, ?, NULL, ?, ?)", (event_id, r[2], f"delete:{reason}", utc_now()))
            return r[0]
        if conn is not None:
            return _do(conn)
        with self.tx() as c:
            return _do(c)

    def update_event(self, ev: Event) -> None:
        """Replace a record's content in place (same event_id). Version history keeps the old payload."""
        payload = json.dumps(ev.to_dict(), ensure_ascii=False)
        with self.tx() as c:
            sets = ", ".join(f"{col}=?" for col in EVENT_COLUMNS if col != "event_id")
            vals = [getattr(ev, col) for col in EVENT_COLUMNS if col != "event_id"]
            n = c.execute(f"UPDATE events SET {sets}, payload=?, size_bytes=? WHERE event_id=?",
                          (*vals, payload, ev.size_bytes(), ev.event_id)).rowcount
            if n == 0:
                raise KeyError(ev.event_id)
            c.execute("INSERT INTO event_versions (event_id, content_hash, payload, change_type, changed_at) "
                      "VALUES (?, ?, ?, 'update', ?)", (ev.event_id, ev.content_hash, payload, utc_now()))

    def version_history(self, event_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT event_id, content_hash, change_type, changed_at FROM event_versions WHERE event_id=? ORDER BY id",
            (event_id,))]

    # ---------------------------------------------------------------- relationships
    def insert_relationships(self, rels: list[Relationship], conn=None) -> None:
        rows = [(r.rel_id, r.src_id, r.src_kind, r.dst_id, r.dst_kind, r.rel_type, r.status, r.source_event_id)
                for r in rels]
        sql = "INSERT OR IGNORE INTO relationships VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        if conn is not None:
            conn.executemany(sql, rows)
            return
        with self.tx() as c:
            c.executemany(sql, rows)

    def relationships_of(self, node_id: str, exclude_types: tuple = ()) -> list[Relationship]:
        q = "SELECT * FROM relationships WHERE (src_id=? OR dst_id=?)"
        args: list = [node_id, node_id]
        if exclude_types:
            q += f" AND rel_type NOT IN ({','.join('?' * len(exclude_types))})"
            args += list(exclude_types)
        return [Relationship(**{k: r[k] for k in r.keys()}) for r in self.conn.execute(q, args)]

    def relationship(self, rel_id: str) -> Relationship | None:
        r = self.conn.execute("SELECT * FROM relationships WHERE rel_id=?", (rel_id,)).fetchone()
        return Relationship(**{k: r[k] for k in r.keys()}) if r else None

    # ---------------------------------------------------------------- gate log
    def log_decisions(self, rows: list[tuple]) -> None:
        with self.tx() as c:
            c.executemany("INSERT INTO gate_decisions (event_id, decision, score, policy, budget_bytes, used_bytes, "
                          "reason, run_tag, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def decisions(self, limit: int = 500, event_id: str | None = None) -> list[dict]:
        if event_id:
            rows = self.conn.execute("SELECT * FROM gate_decisions WHERE event_id=? ORDER BY id", (event_id,))
        else:
            rows = self.conn.execute("SELECT * FROM gate_decisions ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- stats / backup
    def stats(self) -> dict:
        c = self.conn
        n = c.execute("SELECT COUNT(*), COALESCE(SUM(size_bytes),0) FROM events WHERE status='active'").fetchone()
        by_stage = dict(c.execute("SELECT stage, COUNT(*) FROM events WHERE status='active' GROUP BY stage").fetchall())
        by_src = dict(c.execute("SELECT source_type, COUNT(*) FROM events WHERE status='active' GROUP BY source_type").fetchall())
        rng = c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events WHERE status='active'").fetchone()
        rels = c.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        dec = dict(c.execute("SELECT decision, COUNT(*) FROM gate_decisions GROUP BY decision").fetchall())
        return {"records": n[0], "record_bytes": n[1], "relationships": rels, "by_stage": by_stage,
                "by_source_type": by_src, "time_coverage": [rng[0], rng[1]], "gate_decisions": dec,
                "db_file_bytes": self.path.stat().st_size if self.path.exists() else 0}

    def backup(self, dest: str | Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            target = sqlite3.connect(str(dest))
            self.conn.backup(target)
            target.close()
        return dest

    @classmethod
    def restore(cls, backup_path: str | Path, dest: str | Path) -> "SQLiteStore":
        shutil.copyfile(backup_path, dest)
        return cls(dest)


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
