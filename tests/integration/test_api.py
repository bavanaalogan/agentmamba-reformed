"""FastAPI endpoints against the real pipeline (fixture embedder + non-model answerer)."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from agentmamba.api.main import create_app
from agentmamba.models.answerer import FirstCandidateAnswerer
from agentmamba.pipeline.agent import AgentMamba
from agentmamba.pipeline.service import AgentService


@pytest.fixture
def client(cfg, embedder, trained_qp_controller, tmp_path):
    qp, ctl = trained_qp_controller
    paths = AgentMamba.artifact_paths(cfg)
    qp.classifier.save(paths["intent"])
    ctl.save(paths["controller"])
    cfg["memory"]["policy"] = "heuristic"     # explicit baseline policy for this test (no gate artifact)
    svc = AgentService(cfg, tmp_path / "app", embedder=embedder, answerer=FirstCandidateAnswerer())
    return TestClient(create_app(svc)), svc


def test_api_flow(client, dataset, cfg, tmp_path):
    c, svc = client
    assert c.get("/health").json() == {"status": "ok"}
    evs = [e.to_dict() for e in dataset.events_for("test")]
    r = c.post("/events/ingest", json={"events": evs, "budget_bytes": 10**9})
    assert r.status_code == 200 and r.json()["written"] == len(evs)
    st = c.get("/status").json()
    assert st["sqlite"]["records"] == len(evs) and st["faiss"]["ok"] and st["ready"]
    assert c.get("/memory/stats").json()["vectors"] == len(evs)
    some = c.get("/events", params={"stage": "LADLE", "limit": 5}).json()
    assert some["count"] == 5 and all(e["stage"] == "LADLE" for e in some["events"])
    one = c.get(f"/events/{some['events'][0]['event_id']}").json()
    assert one["event"]["event_id"] == some["events"][0]["event_id"] and one["gate_decisions"]
    assert c.get("/events/NOPE").status_code == 404
    heat = next(e for e in evs if e["event_type"] == "cast_record")
    q = c.post("/query", json={"question": f"Which ladle carried heat {heat['heat_id']}?",
                               "initial_context_ids": [heat["event_id"]]}).json()
    assert q["retrieval_triggered"] and q["answer_status"] == "answered" and q["evidence"]
    assert c.post("/query", json={"question": "x"}).status_code == 422
    assert c.post("/query", json={"question": "What happened?", "mode": "bogus"}).status_code == 422
    assert c.post("/memory/rebuild-index").json()["ok"]
    assert c.get("/experiments").json() == []
    assert c.get("/experiments/..%2F..%2Fetc").status_code == 404


def test_document_upload_and_validation(client, tmp_path):
    c, _ = client
    r = c.post("/documents/ingest", files={"file": ("../../evil SOP.txt", b"Step one.\n\nStep two.", "text/plain")},
               data={"stage": "MAINTENANCE", "budget_bytes": "100000"})
    assert r.status_code == 200 and r.json()["written"] == 2
    assert r.json()["normalized_events"][0]["source_ref"].startswith("file://evil SOP.txt")
    assert c.post("/documents/ingest", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 415
    bad = c.post("/events/ingest", json={"events": [{"event_id": "x", "stage": "MOON"}]})
    assert bad.status_code == 422


def test_learned_policy_without_gate_refuses(cfg, embedder, tmp_path):
    cfg["memory"]["policy"] = "learned"
    svc = AgentService(cfg, tmp_path / "app2", embedder=embedder, answerer=FirstCandidateAnswerer())
    c = TestClient(create_app(svc))
    r = c.post("/events/ingest", json={"events": [{"event_id": "E1", "stage": "LADLE", "event_type": "x",
                                                   "timestamp": None, "text": "t", "source_ref": "s",
                                                   "source_type": "synthetic"}]})
    assert r.status_code == 503 and "no heuristic fallback" in r.json()["detail"]


def test_log_upload_and_side_by_side_compare(cfg, embedder, trained_qp_controller, tmp_path):
    """Generic log CSV upload (explicit store-all), then /compare: both systems get the same
    initial context; the baseline never retrieves; AgentMamba retrieves and cites the row."""
    qp, ctl = trained_qp_controller
    paths = AgentMamba.artifact_paths(cfg)
    qp.classifier.save(paths["intent"])
    ctl.save(paths["controller"])
    cfg["memory"]["policy"] = "heuristic"
    svc = AgentService(cfg, tmp_path / "app2", embedder=embedder, answerer=FirstCandidateAnswerer(),
                       token_counter=lambda s: len(s.split()))
    c = TestClient(create_app(svc))
    rows = "timestamp,equipment,work_order,action,technician\n2026-05-04T10:15:00,K9-PMP-01,WO-1,replaced mechanical seal,A\n"
    noise = "".join(f"2026-06-{d:02d}T08:00:00,K9-FAN-0{d % 3},WO-{d},replaced worn V-belts,B\n" for d in range(1, 29))
    r = c.post("/documents/ingest", files={"file": ("maintenance_log_202605.csv", (rows + noise.replace(
        "timestamp,equipment,work_order,action,technician\n", "")).encode(), "text/csv")},
               data={"stage": "MAINTENANCE", "policy": "store_all"})
    body = r.json()
    assert r.status_code == 200 and body["received"] == 29 and body["written"] == 29 and body["chunks_created"] == 0
    assert body["normalized_events"][0]["event_type"] == "maintenance_log"
    again = c.post("/documents/ingest", files={"file": ("maintenance_log_202605.csv", (rows + noise).encode(),
                                                        "text/csv")}, data={"policy": "store_all"}).json()
    assert again["written"] == 0 and again["duplicates"] == 29
    assert c.post("/documents/ingest", files={"file": ("x.csv", rows.encode(), "text/csv")},
                  data={"policy": "random"}).status_code == 422
    out = c.post("/compare", json={"question": "What maintenance action was recorded for K9-PMP-01 on 2026-05-04?",
                                   "window_tokens": 120}).json()
    base, enh = out["baseline"], out["enhanced"]
    assert out["errors"] == {} and out["diagnostic_only"]
    assert base["context"]["record_ids"] == enh["initial_context_ids"] == out["shared_context"]["record_ids"]
    assert base["retrieval_triggered"] is False and base["verification_status"] == "not_verified"
    assert enh["retrieval_triggered"] and enh["answer_value"] == "replaced mechanical seal"
    assert enh["confidence"]["candidate_share"] is None          # non-model answerer: no model confidence
    assert c.post("/compare", json={"question": "abc?", "case": {"split": "x", "question_id": "q", "stage": 1,
                                                                 "run": "r"}}).status_code == 422


def test_compare_can_run_raw_mamba_generation_from_shared_context(cfg, embedder, trained_qp_controller, tmp_path):
    class GenerationBackbone:
        model_id = "fake-mamba"
        last_call = {"cache_hit": False, "prompt_tokens": 20, "generated_tokens": 4}

        def generate(self, prompt, max_new_tokens, max_prompt_tokens):
            assert "What is in memory?" in prompt
            assert max_new_tokens == 64 and max_prompt_tokens == 2048
            return "Raw Mamba response"

    class GenerationAnswerer(FirstCandidateAnswerer):
        name = "mamba"
        backbone = GenerationBackbone()

    qp, ctl = trained_qp_controller
    paths = AgentMamba.artifact_paths(cfg)
    qp.classifier.save(paths["intent"])
    ctl.save(paths["controller"])
    cfg["memory"]["policy"] = "store_all"
    svc = AgentService(cfg, tmp_path / "raw-app", embedder=embedder, answerer=GenerationAnswerer(),
                       token_counter=lambda s: len(s.split()))

    result = svc.compare("What is in memory?", [], 2048, baseline_mode="raw_generation")

    assert result["errors"] == {}
    assert result["baseline"]["generation_mode"] == "raw_mamba_generation"
    assert result["baseline"]["answer_value"] == "Raw Mamba response"
    assert result["baseline"]["confidence"]["candidate_share"] is None
