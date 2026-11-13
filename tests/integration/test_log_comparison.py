"""Baseline-vs-AgentMamba on operational logs: ingestion, retrieval, grounding, abstention,
baseline isolation, persistence and evaluation. Deterministic stand-ins only (HashingEmbedder,
first-candidate ranking, a whitespace token counter); real-model behaviour is measured by
scripts/run_context_growth.py, not asserted here."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentmamba.config import load_config
from agentmamba.data.document_loader import load_any, load_log_records
from agentmamba.data.log_benchmark import Scenario, build_scenario, stage_files
from agentmamba.evaluation.comparison import brier, ece, matches, paired, score_output, score_reference, summarize_rows
from agentmamba.gate.predictor import build_scorer
from agentmamba.memory.episodic_memory import EpisodicMemory
from agentmamba.models.answerer import FirstCandidateAnswerer
from agentmamba.models.embedding_model import HashingEmbedder
from agentmamba.pipeline.agent import AgentMamba, QueryRequest
from agentmamba.pipeline.baseline import BaselineMamba, build_window
from agentmamba.pipeline.ingestion import FeatureProvider, IngestionPipeline
from agentmamba.retrieval.retriever import Retriever

EMB = HashingEmbedder()
COUNT = lambda s: len(s.split())  # noqa: E731  deterministic stand-in for the Mamba tokenizer


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    root = tmp_path_factory.mktemp("logbench") / "T1"
    qs, anns = build_scenario(Scenario("T1", "dev", "2026-05-04"), root, seed=11)
    return root, {q.question_id: q for q in qs}, {a.question_id: a for a in anns}


def _ingest(mem, root, stages):
    pipe = IngestionPipeline(mem, FeatureProvider(EMB, None), build_scorer("store_all"), "store_all", False)
    history, reports = [], []
    for s in stages:
        evs = []
        for f in stage_files(root, s):
            evs += load_any(root / f["file"], stage=f["stage"])
        reports.append(pipe.ingest(evs, None))
        known = {h.event_id for h in history}
        history += [e for e in evs if e.event_id not in known and not known.add(e.event_id)]
    return history, reports


@pytest.fixture(scope="module")
def world(scenario, tmp_path_factory, trained_qp_controller):
    root, qs, anns = scenario
    d = tmp_path_factory.mktemp("mem")
    cfg = load_config(overrides={"models.answerer": "first_candidate"})
    mem = EpisodicMemory(d / "store", EMB)
    history, reports = _ingest(mem, root, (1, 2, 3, 4))
    qp, ctl = trained_qp_controller
    agent = AgentMamba(cfg, mem, FirstCandidateAnswerer(), qp, ctl)
    return {"root": root, "qs": qs, "anns": anns, "mem": mem, "history": history, "reports": reports,
            "agent": agent, "cfg": cfg, "qp": qp, "ctl": ctl, "dir": d / "store"}


def q_of(world, family):
    a = next(a for a in world["anns"].values() if a.family == family)
    return world["qs"][a.question_id], a


def ask(world, family, ctx=(), **kw):
    q, a = q_of(world, family)
    return world["agent"].query(QueryRequest(q.text, list(ctx), **kw)), a


# ---------------------------------------------------------------- ingestion
def test_generic_log_loader_preserves_rows_units_and_identity(tmp_path):
    p = tmp_path / "historian_20260504.csv"
    p.write_text("timestamp,equipment,signal,value,unit,operator\n2026-05-04T09:30:00,X1-PMP-01,vibration,7.0,mm/s,A\n",
                 encoding="utf-8")
    (ev,) = load_log_records(p, stage="UTILITIES")
    assert ev.event_type == "historian" and ev.asset_id == "X1-PMP-01" and ev.timestamp == "2026-05-04T09:30:00"
    assert ev.raw == {"timestamp": "2026-05-04T09:30:00", "equipment": "X1-PMP-01", "signal": "vibration",
                      "value": "7.0", "unit": "mm/s", "operator": "A"}          # original row untouched
    assert ev.obs("vibration").unit == "mm/s" and ev.obs("vibration").raw == "7.0"
    assert load_log_records(p, stage="UTILITIES")[0].event_id == ev.event_id     # deterministic ID


def test_event_layout_files_still_load_as_events(tmp_path, dataset):
    p = tmp_path / "events.jsonl"
    p.write_text("".join(json.dumps(e.to_dict()) + "\n" for e in dataset.events[:3]), encoding="utf-8")
    assert [e.event_id for e in load_any(p)] == [e.event_id for e in dataset.events[:3]]


def test_duplicate_and_repeated_ingestion(world, tmp_path):
    rep2 = world["reports"][1]
    assert rep2.duplicates == 3 and rep2.written == rep2.offered            # the re-delivered alarm file
    mem = EpisodicMemory(tmp_path / "again", EMB)
    _, (r1,) = _ingest(mem, world["root"], (1,))
    _, (r2,) = _ingest(mem, world["root"], (1,))
    assert r1.written == r1.offered > 0 and r2.written == 0 and r2.duplicates == r1.offered
    assert mem.validate()["ok"]


def test_long_document_is_chunked_and_reachable_by_identifier(world):
    manual = next(f for f in stage_files(world["root"], 3) if "manual" in f["file"])
    chunks = load_any(world["root"] / manual["file"], stage="MAINTENANCE")
    assert len(chunks) > 5 and all(len(c.text) <= 800 for c in chunks)
    q, _ = q_of(world, "maint_action")
    eq = q.text.split(" for ")[1].split(" on ")[0]
    parsed = world["qp"].parse(f"What does the care manual say about lubrication for {eq}?")
    res = Retriever(world["mem"], world["cfg"]).retrieve(parsed, parsed.text)
    assert any(it.event.event_id in {c.event_id for c in chunks} and "text_mention" in it.sources for it in res.items)


# ---------------------------------------------------------------- answering and grounding
def test_exact_equipment_identifier_answer_is_grounded(world):
    out, a = ask(world, "maint_action")
    assert out["answer_status"] == "answered" and out["answer_value"] == a.expected_answer
    ans = next(e for e in out["evidence"] if e["role"] == "answer")
    assert ans["record_id"] in a.required_evidence_ids and out["retrieval_triggered"]
    assert out["invalid_citations"] == []


def test_timestamp_questions_pick_the_raised_alarm_and_nearest_reading(world):
    out, a = ask(world, "alarm_at_time")
    assert out["answer_value"] == a.expected_answer and out["evidence"][0]["record_id"] in a.required_evidence_ids
    out, a = ask(world, "sensor_at_time")
    assert matches(out["answer_value"], a.expected_answer, "exact")
    assert next(e for e in out["evidence"] if e["role"] == "answer")["record_id"] in a.required_evidence_ids


def test_multiple_relevant_chunks_date_from_sibling_chunk(world, tmp_path):
    p = tmp_path / "split_report.md"
    p.write_text("# Report\n\nInspection report for T1-ZZZ-07, 4 May 2026.\n\nInspection finding for T1-ZZZ-07: "
                 "worn coupling insert with visible cracking.\n", encoding="utf-8")
    chunks = load_any(p, stage="MAINTENANCE")
    out = world["agent"].query(QueryRequest("What inspection finding was recorded for T1-ZZZ-07 on 2026-05-04?",
                                            chunks, mode="never"))
    assert out["answer_status"] == "answered" and "worn coupling insert" in out["answer_value"]
    roles = {e["role"] for e in out["evidence"]}
    assert roles == {"answer", "link"}                    # the date came from the sibling chunk


def test_similarity_retrieval_channel_finds_semantic_match(world):
    q, a = q_of(world, "inspection_finding")
    target = world["mem"].store.get_event(a.required_evidence_ids[0])
    cfg = json.loads(json.dumps(world["cfg"]))
    cfg["retrieval"]["channels"] = ["vector"]
    res = Retriever(world["mem"], cfg).retrieve(world["qp"].parse(target.text), target.text)
    assert res.items[0].event.event_id == target.event_id and res.items[0].sources == ["vector"]


def test_lexical_only_channel_and_invalid_channel(world):
    cfg = json.loads(json.dumps(world["cfg"]))
    cfg["retrieval"]["channels"] = ["relationship", "text"]
    out, a = ask(world, "maint_action")
    lex = AgentMamba(cfg, world["mem"], FirstCandidateAnswerer(), world["qp"], world["ctl"])
    q, _ = q_of(world, "maint_action")
    out2 = lex.query(QueryRequest(q.text, []))
    assert out2["answer_value"] == a.expected_answer
    assert all("vector" not in r["sources"] for r in out2["retrieval"])
    cfg["retrieval"]["channels"] = ["bm25"]
    with pytest.raises(ValueError):
        Retriever(world["mem"], cfg)


@pytest.mark.parametrize("family,reason", [("unknown_equipment", "no record names"),
                                           ("wrong_date", "do not state"),
                                           ("clearance_undocumented", "does not establish"),
                                           ("rca_undocumented", "does not establish")])
def test_missing_or_unestablished_evidence_abstains(world, family, reason):
    out, _ = ask(world, family)
    assert out["answer_status"] == "abstained" and out["answer_value"] is None
    assert reason in out["abstention_reason"]


def test_contradictory_records_are_reported_not_resolved(world):
    out, a = ask(world, "conflicting_readings")
    assert out["verification_status"] == "conflict" and out["answer_status"] == "abstained"
    assert {e["record_id"] for e in out["evidence"] if e["role"] == "conflicting"} == set(a.required_evidence_ids)


def test_later_confirmed_cause_for_another_date_is_not_used(world):
    out, _ = ask(world, "rca_undocumented")
    assert "confirmed" not in (out.get("answer_value") or "")
    out, a = ask(world, "rca_documented")
    assert out["answer_status"] == "answered" and a.expected_answer in out["answer_value"]
    assert out["causal_claim_status"] == "documented_by_source"


def test_timeline_question_lists_records_in_time_order(world):
    q, a = q_of(world, "alarm_at_time")
    eq = q.text.split(" on ")[1].split(" at ")[0]
    day = q.text.split(" at ")[1][:10]
    out = world["agent"].query(QueryRequest(f"Reconstruct the timeline of alarm events for {eq} on {day}.", []))
    times = [e["timestamp"] for e in out["evidence"] if e["role"] == "answer"]
    assert out["answer_status"] == "answered" and len(times) >= 2 and times == sorted(times)


def test_paraphrase_without_record_vocabulary_abstains_rather_than_guessing(world):
    out, _ = ask(world, "maint_action_paraphrase")      # documented limitation of lexical coverage
    assert out["answer_status"] == "abstained"


def test_persistent_retrieval_after_restart(world, trained_qp_controller):
    out1, a = ask(world, "inspection_finding")
    reopened = EpisodicMemory(world["dir"], EMB)       # a second process opening the same files
    assert reopened.validate()["ok"] and reopened.stats()["records"] == world["mem"].stats()["records"]
    agent2 = AgentMamba(world["cfg"], reopened, FirstCandidateAnswerer(), *trained_qp_controller)
    q, _ = q_of(world, "inspection_finding")
    out2 = agent2.query(QueryRequest(q.text, []))
    assert out2["answer_value"] == out1["answer_value"] and a.expected_answer in out2["answer_value"]
    reopened.close()


# ---------------------------------------------------------------- baseline isolation
def test_baseline_window_is_recent_question_independent_and_within_budget(world):
    hist = world["history"]
    w1 = build_window("Which alarm was raised?", [], hist, COUNT, 300)
    w2 = build_window("What maintenance action was recorded?", [], hist, COUNT, 300)
    assert [e.event_id for e in w1.records] == [e.event_id for e in w2.records]     # no question-dependent lookup
    assert w1.records[0].event_id == hist[-1].event_id and w1.prompt_tokens <= 300 - 128
    init = hist[0]
    w3 = build_window("q?", [init], hist, COUNT, 300)
    assert w3.records[0].event_id == init.event_id


def test_baseline_answers_only_from_its_window_and_has_no_memory_access(world):
    q, a = q_of(world, "rca_undocumented")
    base = BaselineMamba(world["qp"], FirstCandidateAnswerer(), COUNT, token_budget=100000)
    assert not hasattr(base, "memory") and not hasattr(base, "retriever")
    win = build_window(q.text, [], world["history"][:40], COUNT, 100000)         # stage-1 records only
    out = base.answer(q.text, win)
    assert out["retrieval_triggered"] is False and out["verification_status"] == "not_verified"
    assert all(e["record_id"] in {r.event_id for r in win.records} for e in out["evidence"])
    # it answers the undocumented-root-cause question from the hedged note: confidently wrong
    assert out["answer_status"] == "answered" and out["audit"]["status"] == "unsupported"
    row = score_output(out, a, system="baseline", stage=1, context_ids=win.to_dict()["record_ids"], n_available=40)
    assert row["correct"] is False and row["unsupported"] is True


def test_gold_answers_are_not_visible_to_systems(world):
    root = world["root"]
    files = [p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"]
    for a in world["anns"].values():
        assert not any(a.question_id in p.read_text(encoding="utf-8") for p in files)
    q = next(iter(world["qs"].values()))
    assert set(q.__dict__) == {"question_id", "scenario", "split", "text"}        # no answer fields
    out, _ = ask(world, "maint_action")
    assert "expected_answer" not in json.dumps(out, default=str)


# ---------------------------------------------------------------- evaluation
def test_answer_matching_and_metrics():
    assert matches("7 mm/s", "7.0 mm/s", "exact") and not matches("7.1 mm/s", "7.0 mm/s", "exact")
    assert matches("The inspector noted: worn insert, cracked.", "worn insert", "contains")
    baseline = {"answer_status": "answered", "answer_value": "7.0 mm/s", "evidence": [{"record_id": "E1"}],
                "context": {"record_ids": ["E1"]}}
    assert score_reference(baseline, "7 mm/s", {"E1"}) == {
        "correct": True, "grounded_correct": True, "evidence_covered": True, "unsupported": False,
        "em": 1.0, "contains": 1.0, "f1": pytest.approx(0.5)}
    ungrounded = score_reference({**baseline, "evidence": []}, "7 mm/s", {"E1"})
    assert ungrounded["correct"] and not ungrounded["grounded_correct"] and ungrounded["evidence_covered"]
    assert score_reference(baseline, "7 mm/s")["grounded_correct"] is None
    assert brier([1.0, 1.0], [False, True]) == 0.5 and ece([0.9] * 10, [True] * 9 + [False]) < 1e-9
    a = [{"question_id": f"q{i}", "stage": 1, "correct": i < 2} for i in range(10)]
    b = [{"question_id": f"q{i}", "stage": 1, "correct": i < 8} for i in range(10)]
    r = paired(a, b)
    assert r["diff_b_minus_a"] == 0.6 and r["b_only"] == 6 and r["mcnemar_p"] < 0.05


def test_summary_separates_abstention_kinds(world):
    _, a_abs = q_of(world, "unknown_equipment")
    rows = [score_output({"answer_status": "abstained", "no_candidate": True, "evidence": []}, a_abs, system="b",
                         stage=1, context_ids=[], n_available=1)]
    s = summarize_rows(rows)
    assert s["correct_abstention_rate"] == 1.0 and s["no_candidate_rate"] == 1.0
