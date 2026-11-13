"""Controlled synthetic steel-plant scenario, end to end (deterministic fixtures, no model weights):
generate -> split -> train gate on train labels -> budgeted ingestion -> persist -> reload ->
conditional answering -> verification -> abstention -> metrics + traces saved."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from agentmamba.evaluation.evaluator import score_row, summarize
from agentmamba.evaluation.reporting import new_run_dir, write_jsonl
from agentmamba.gate.label_builder import build_labels
from agentmamba.gate.train import train_gate
from agentmamba.memory.episodic_memory import EpisodicMemory
from agentmamba.models.answerer import FirstCandidateAnswerer
from agentmamba.models.embedding_model import HashingEmbedder
from agentmamba.pipeline.agent import AgentMamba, QueryRequest
from agentmamba.pipeline.builder import build_memory

EMB = HashingEmbedder()


@pytest.fixture(scope="module")
def gate(dataset, annotations):
    tr, va = dataset.events_for("train"), dataset.events_for("val")
    e_tr, e_va = EMB.encode([e.text for e in tr]), EMB.encode([e.text for e in va])
    return train_gate(tr, e_tr, None, build_labels(tr, annotations["train"]), va, e_va, None,
                      build_labels(va, annotations["val"]), use_mamba=False, emb_pca_dims=8, c_grid=(0.1, 1.0))


def _agent(cfg, mem, qp_ctl):
    qp, ctl = qp_ctl
    return AgentMamba(cfg, mem, FirstCandidateAnswerer(), qp, ctl)


def _ask(agent, dataset, q, **kw):
    ev = dataset.event_index()
    return agent.query(QueryRequest(q.text, [ev[i] for i in q.initial_context_ids], query_id=q.question_id, **kw))


def test_gate_is_trained_on_train_only_and_beats_chance(gate, dataset, annotations):
    te = dataset.events_for("test")
    from agentmamba.gate.predictor import GatePredictor

    p = GatePredictor(gate).score(te, EMB.encode([e.text for e in te]), None)
    y = build_labels(te, annotations["test"])           # evaluation-only use of test labels
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, p) > 0.7
    assert gate.meta["train"]["n"] == len(dataset.events_for("train"))


def test_end_to_end_budgeted_memory_persist_reload_and_answer(tmp_path, cfg, dataset, annotations, gate,
                                                              trained_qp_controller):
    stream = dataset.events_for("test")
    mem, rep = build_memory(cfg, tmp_path / "mem", stream, policy="learned", budget_fraction=0.25, embedder=EMB,
                            feature_store=None, gate=gate)
    total = sum(e.size_bytes() for e in stream)
    assert rep.used_bytes <= int(total * 0.25) and 0 < rep.written < len(stream)
    assert Counter(d["decision"] for d in mem.store.decisions(limit=100000))["SKIP"] > 0
    retained = {r[0] for r in mem.store.conn.execute("SELECT event_id FROM events")}
    mem.close()
    mem = EpisodicMemory(tmp_path / "mem", EMB)                        # reload after "restart"
    assert mem.validate()["ok"] and mem.store.stats()["records"] == len(retained)
    agent = _agent(cfg, mem, trained_qp_controller)
    qs = dataset.questions_for("test")
    rows, outs = [], []
    for q in qs:
        out = _ask(agent, dataset, q)
        json.dumps(out, default=str)                                   # serializable
        assert out["invalid_citations"] == [] if "invalid_citations" in out else True
        for e in out.get("evidence", []):                              # every citation is a real record
            assert mem.store.get_event(e["record_id"]) or e["record_id"] in q.initial_context_ids
        outs.append(out)
        rows.append(score_row(out, annotations["test"][q.question_id], q, retained))
    s = summarize(rows)
    d = new_run_dir(tmp_path / "results", "it")
    write_jsonl(d / "predictions.jsonl", outs)
    (d / "summary.json").write_text(json.dumps(s, default=str))
    assert (d / "predictions.jsonl").stat().st_size > 0 and s["n"] == len(qs)
    lost = [r for r in rows if r["evidence_available"] is False]
    # discarded evidence is never magically recovered: no correct value answer without its evidence
    assert all(not (r["em"] and r["answer_record"] in r["required_evidence"]) for r in lost)


def test_store_all_family_behaviours(tmp_path, cfg, dataset, annotations, trained_qp_controller):
    mem, _ = build_memory(cfg, tmp_path / "all", dataset.events_for("test"), policy="store_all", budget_fraction=1.0,
                          embedder=EMB, feature_store=None)
    agent = _agent(cfg, mem, trained_qp_controller)
    by_fam = {}
    for q in dataset.questions_for("test"):
        by_fam.setdefault(annotations["test"][q.question_id].family, []).append((q, _ask(agent, dataset, q)))
    a = annotations["test"]
    # 1 sufficient initial evidence bypasses retrieval
    direct = [o for q, o in by_fam["F01_direct_context"] if a[q.question_id].question_type == "measurement"]
    assert direct and all(not o["retrieval_triggered"] and o["answer_status"] == "answered" for o in direct)
    # 2 cross-stage question with missing initial evidence triggers retrieval; IDs resolve to original records
    for q, o in by_fam["F02_earlier_stage"] + by_fam["F03_relationship_chain"]:
        assert o["retrieval_triggered"] and o["answer_value"] == a[q.question_id].expected_answer
        ans = next(e for e in o["evidence"] if e["role"] == "answer")
        assert mem.store.get_event(ans["record_id"]).text == ans["excerpt"]
    # 3 causal: association reported as hypothesis, never as an established cause
    for q, o in by_fam.get("F15_correlation_not_causation", []):
        assert o["causal_claim_status"] == "hypothesis_only" and "not an established cause" in o["final_answer"]
    for q, o in by_fam.get("F15b_documented_cause", []):
        assert o["causal_claim_status"] == "documented_by_source"
    # 4 insufficient evidence / missing link / never stored -> abstain; ambiguous -> clarify; conflict reported
    for fam in ("F11_missing_identifier", "F12_evidence_never_stored", "F13_unanswerable"):
        assert all(o["answer_status"] == "abstained" for _, o in by_fam[fam]), fam
    assert all(o["answer_status"] == "needs_clarification" for _, o in by_fam["F14_ambiguous"])
    conf = [o for q, o in by_fam["F09_conflicting_records"] if a[q.question_id].answer_kind == "conflict"]
    assert conf and all(o["verification_status"] == "conflict" for o in conf)
    # 5 outdated documentation: current revision wins after retrieval
    for q, o in by_fam["F10_outdated_document"]:
        assert o["retrieval_triggered"] and o["answer_value"] == a[q.question_id].expected_answer
    # 6 decision traces carry controller output, timings and versions
    o = by_fam["F02_earlier_stage"][0][1]
    assert o["controller"]["action"] == "RETRIEVE" and o["retrieval_reason"] and "total_s" in o["timings"]
    assert o["versions"]["config"]


def test_discarded_evidence_and_absent_relationship(tmp_path, cfg, dataset, annotations, trained_qp_controller):
    """Delete the required records for a cross-stage question: the system must abstain, not guess."""
    mem, _ = build_memory(cfg, tmp_path / "m", dataset.events_for("test"), policy="store_all", budget_fraction=1.0,
                          embedder=EMB, feature_store=None)
    agent = _agent(cfg, mem, trained_qp_controller)
    q = next(q for q in dataset.questions_for("test") if annotations["test"][q.question_id].family == "F02_earlier_stage")
    assert _ask(agent, dataset, q)["answer_status"] == "answered"
    mem.delete(annotations["test"][q.question_id].required_evidence_ids, reason="simulated gate discard")
    out = _ask(agent, dataset, q)
    assert out["answer_status"] == "abstained" and out["abstention_reason"]
    assert out["answer_value"] is None


def test_modes(tmp_path, cfg, dataset, annotations, trained_qp_controller):
    mem, _ = build_memory(cfg, tmp_path / "m", dataset.events_for("test"), policy="store_all", budget_fraction=1.0,
                          embedder=EMB, feature_store=None)
    agent = _agent(cfg, mem, trained_qp_controller)
    q = next(q for q in dataset.questions_for("test") if annotations["test"][q.question_id].family == "F02_earlier_stage")
    assert not _ask(agent, dataset, q, mode="never")["retrieval_triggered"]
    assert _ask(agent, dataset, q, mode="never")["answer_status"] == "abstained"
    assert _ask(agent, dataset, q, mode="always")["retrieval_triggered"]
    with pytest.raises(ValueError):
        _ask(agent, dataset, q, mode="sometimes")
