"""Evaluation-results store: a SQLite file SEPARATE from every operational memory
(default ``experiments/evaluation_results.sqlite3``). It holds per-question predictions,
reference answers, evidence IDs, retrieval decisions, verification outcomes, confidence values and
latencies for each evaluation run, plus each run's metric suite. Systems never read from it."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, kind TEXT, created TEXT, source TEXT, config TEXT);
CREATE TABLE IF NOT EXISTS rows (run_id TEXT, system TEXT, stage INTEGER, question_id TEXT, payload TEXT,
                                 PRIMARY KEY (run_id, system, stage, question_id));
CREATE TABLE IF NOT EXISTS suites (run_id TEXT, system TEXT, payload TEXT, PRIMARY KEY (run_id, system));
"""


class ResultsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.executescript(DDL)

    def put_run(self, run_id: str, kind: str, created: str, source: str, config: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                          (run_id, kind, created, source, json.dumps(config, default=str)))
        self.conn.commit()

    def put_rows(self, run_id: str, rows: list[dict]) -> None:
        self.conn.executemany("INSERT OR REPLACE INTO rows VALUES (?,?,?,?,?)",
                              [(run_id, r["system"], int(r["stage"]), r["question_id"], json.dumps(r, default=str))
                               for r in rows])
        self.conn.commit()

    def put_suite(self, run_id: str, system: str, suite: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO suites VALUES (?,?,?)", (run_id, system, json.dumps(suite)))
        self.conn.commit()

    def runs(self) -> list[dict]:
        return [{"run_id": r[0], "kind": r[1], "created": r[2], "source": r[3]}
                for r in self.conn.execute("SELECT run_id, kind, created, source FROM runs ORDER BY created DESC")]

    def rows(self, run_id: str, system: str | None = None) -> list[dict]:
        q, a = "SELECT payload FROM rows WHERE run_id=?", [run_id]
        if system:
            q, a = q + " AND system=?", a + [system]
        return [json.loads(r[0]) for r in self.conn.execute(q + " ORDER BY stage, question_id", a)]

    def suites(self, run_id: str) -> dict:
        return {s: json.loads(p) for s, p in self.conn.execute("SELECT system, payload FROM suites WHERE run_id=?",
                                                                (run_id,))}

    def close(self) -> None:
        self.conn.close()
