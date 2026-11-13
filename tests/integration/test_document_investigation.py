"""Regression tests: investigations grounded in uploaded free-text documents.

Bug: a question that explicitly named a non-benchmark ladle ID and a date was answered with a
clarification request, because (a) only benchmark ID formats were extracted, so the anchor was
"missing", and (b) document_chunk records (null structured identifiers) could never support an
answer. Nothing here is specific to one document: the IDs below are arbitrary test values.
"""
from __future__ import annotations

import pytest

from agentmamba.data.document_loader import load_any, load_text_document, load_text_log, parse_log_line
from agentmamba.gate.predictor import build_scorer
from agentmamba.models.answerer import FirstCandidateAnswerer
from agentmamba.models.embedding_model import HashingEmbedder
from agentmamba.pipeline.agent import AgentMamba, QueryRequest
from agentmamba.pipeline.builder import build_memory
from agentmamba.pipeline.ingestion import FeatureProvider, IngestionPipeline
from agentmamba.retrieval.query_processor import extract_ids
from agentmamba.verification.document_evidence import dates_in, declarative_sentences

EMB = HashingEmbedder()

DOC_A = """# Ingestion Test Record - Fictional Data

> **TEST DATA ONLY:** invented record.

## Record metadata
- Asset ID: QA-LADLE-417
- Heat ID: QA-HEAT-2210
- Timestamp: 2026-10-20T09:30:00+05:30

## Inspection finding
A fictional refractory inspection was recorded for QA-LADLE-417 on 20 October 2026 at 09:30 local time. The inspector noted **minor wear near the slag line**. No crack was documented in this test record.

## Questions for testing
1. What finding was recorded for QA-LADLE-417?
"""

DOC_B = """## Inspection finding
A fictional refractory inspection was recorded for QA-LADLE-418 on 20 October 2026. The inspector noted deep erosion at the tap hole.
"""


@pytest.fixture
def agent(tmp_path, cfg, dataset, trained_qp_controller):
    mem, _ = build_memory(cfg, tmp_path / "mem", dataset.events_for("test"), policy="store_all", budget_fraction=1.0,
                          embedder=EMB, feature_store=None)
    pipe = IngestionPipeline(mem, FeatureProvider(EMB, None), build_scorer("store_all"), "store_all", False)
    for name, text in (("qa_record_a.md", DOC_A), ("qa_record_b.md", DOC_B)):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        chunks = load_text_document(p, stage="MAINTENANCE", source_type="synthetic")
        assert all(c.ladle_id is None and c.asset_id is None for c in chunks)   # structured fields stay null
        pipe.ingest(chunks, None)
    qp, ctl = trained_qp_controller
    return AgentMamba(cfg, mem, FirstCandidateAnswerer(), qp, ctl)


def ask(agent, q, ctx=(), **kw):
    return agent.query(QueryRequest(q, list(ctx), **kw))


# ---------------------------------------------------------------- required regressions
@pytest.mark.parametrize("question", [
    "What finding was recorded for QA-LADLE-417 on 2026-10-20?",
    "What refractory inspection finding was recorded for ladle QA-LADLE-417 on 2026-10-20?",
])
def test_explicit_ladle_and_date_in_ingested_document_is_answered(agent, question):
    out = ask(agent, question)
    assert out["answer_status"] == "answered", out["final_answer"]
    assert "minor wear near the slag line" in out["answer_value"]
    assert "QA-LADLE-418" not in out["answer_value"] and "erosion" not in out["answer_value"]
    ans = next(e for e in out["evidence"] if e["role"] == "answer")
    assert ans["event_type"] == "document_chunk" and ans["source"].startswith("file://qa_record_a.md")
    assert out["verification_status"] == "supported" and out["invalid_citations"] == []
    assert out["retrieval_triggered"]                      # nothing in the initial context -> retrieved


def test_exact_sentence_is_quoted_for_a_specific_question(agent):
    out = ask(agent, "Was a crack documented for QA-LADLE-417?")
    assert out["answer_status"] == "answered" and out["answer_value"] == "No crack was documented in this test record."


def test_no_ladle_identifier_with_multiple_candidate_ladles_requests_clarification(agent, dataset):
    out = ask(agent, "What finding was recorded for the ladle on 2026-10-20?")
    assert out["answer_status"] == "needs_clarification"
    # two different ladles in the initial context are still ambiguous for a typed ladle question
    two = [e for e in dataset.events_for("test") if e.event_type == "ladle_inspection"]
    two = [two[0], next(e for e in two if e.ladle_id != two[0].ladle_id)]
    out = ask(agent, "What lining thickness was recorded for the ladle at the inspection on 2024-01-03?", two)
    assert out["answer_status"] == "needs_clarification"


def test_single_candidate_ladle_in_context_is_resolved_not_asked(agent, dataset):
    ev = next(e for e in dataset.events_for("test") if e.event_type == "ladle_inspection")
    out = ask(agent, f"What lining thickness was recorded for the ladle at the inspection on {ev.timestamp[:10]}?", [ev])
    assert out["answer_status"] != "needs_clarification"
    assert out["parsed"]["resolved_from_context"] == {ev.ladle_id: "ladle"}


@pytest.mark.parametrize("question,why", [
    ("What lining thickness was recorded for QA-LADLE-417 on 2026-10-20?", "do not state lining"),  # not stated
    ("What finding was recorded for QA-LADLE-417 on 2026-10-21?", "2026-10-21"),           # other date
    ("What finding was recorded for QA-LADLE-999 on 2026-10-20?", "QA-LADLE-999"),         # unknown ladle
])
def test_answer_absent_from_records_abstains(agent, question, why):
    out = ask(agent, question)
    assert out["answer_status"] == "abstained" and out["answer_value"] is None
    assert why in out["abstention_reason"]


def test_question_sentences_inside_a_document_are_not_evidence(agent):
    # DOC_A literally contains the question "What finding was recorded for QA-LADLE-417?"; the answer
    # must come from the declarative finding statement, not from that question line.
    out = ask(agent, "What finding was recorded for QA-LADLE-417?")
    assert "?" not in out["answer_value"] and "minor wear" in out["answer_value"]


def test_existing_heat_to_ladle_investigation_still_succeeds(agent, dataset, annotations):
    q = next(q for q in dataset.questions_for("test") if annotations["test"][q.question_id].family == "F02_earlier_stage")
    ev = dataset.event_index()
    out = ask(agent, q.text, [ev[i] for i in q.initial_context_ids])
    assert out["answer_status"] == "answered" and out["answer_value"] == annotations["test"][q.question_id].expected_answer
    assert next(e for e in out["evidence"] if e["role"] == "answer")["event_type"] == "ladle_assignment"


def test_existing_ambiguous_refractory_question_still_asks_for_clarification(agent):
    out = ask(agent, "What did the last refractory inspection of the ladle find?")
    assert out["answer_status"] == "needs_clarification"


def test_no_retrieval_mode_does_not_reach_documents(agent):
    out = ask(agent, "What finding was recorded for QA-LADLE-417 on 2026-10-20?", mode="never")
    assert not out["retrieval_triggered"] and out["answer_status"] == "abstained"


# ---------------------------------------------------------------- unit-level pieces
def test_generic_identifier_extraction_and_kinds():
    ids = extract_ids("Was QA-LADLE-417 used for heat PLANT2-H-88 and coil X9-777 by WO-55 on 2026-10-20?")
    assert ids["ladle"] == ["QA-LADLE-417"] and ids["work_order"] == ["WO-55"]
    assert "PLANT2-H-88" in ids.get("heat", []) or "PLANT2-H-88" in ids.get("identifier", [])
    assert ids["coil"] == ["X9-777"]                          # kind from the preceding word
    assert "2026-10-20" not in sum(ids.values(), [])
    assert extract_ids("Which ladle carried heat C05-H0002?") == {"heat": ["C05-H0002"]}   # benchmark IDs unchanged


def test_date_formats_and_declarative_sentences():
    assert dates_in("on 20 October 2026 at 09:30") == {"2026-10-20"}
    assert dates_in("October 20, 2026; 2026-10-21T09:30; 21.10.2026") == {"2026-10-20", "2026-10-21"}
    assert declarative_sentences("## Finding\nIt was worn. Was it cracked? No.") == ["Finding", "It was worn.", "No."]


def test_positional_log_columns_are_extracted_as_event_and_action(tmp_path):
    line = ("2026-10-12 08:24:00.000 +05:30 | INFO | CMMS-BRIDGE | COOLING-PUMP-17 | "
            "WORK_ORDER_OPENED | work_order=WO-CP17-211; inspect pump vibration and coupling alignment | "
            "quality=GOOD")
    parsed = parse_log_line(line)
    assert parsed is not None
    assert parsed["equipment"] == "COOLING-PUMP-17"
    assert parsed["event_type"] == "WORK_ORDER_OPENED"
    assert parsed["work_order"] == "WO-CP17-211"
    assert parsed["action"] == "inspect pump vibration and coupling alignment"
    alarm_line = ("2026-10-12 08:19:13.000 +05:30 | INFO | ALARM-MANAGER | COOLING-PUMP-17 | "
                  "ALARM_STATUS | alarm_id=ALM-CP17-884; status=CLEARED; "
                  "reason=vibration below configured limit | quality=GOOD")
    alarm = parse_log_line(alarm_line)
    assert alarm is not None
    assert alarm["reason"] == "vibration below configured limit"

    path = tmp_path / "cp17.txt"
    path.write_text(line, encoding="utf-8")
    event, = load_text_log(path, stage="UTILITIES")
    assert event.asset_id == "COOLING-PUMP-17"
    assert event.event_type == "work_order_opened"
    assert event.raw["action"] == "inspect pump vibration and coupling alignment"


def test_uploaded_cp17_logs_answer_compound_question_from_distinct_records(agent, tmp_path):
    content = """# SYNTHETIC FICTIONAL INDUSTRIAL LOGS — NOT REAL PLANT DATA
# Format: timestamp | level | source | equipment | event_type | details | quality
# STAGE 1: ingest this file first.
2026-10-12 08:14:02.100 +05:30 | INFO | PLC-CP17 | COOLING-PUMP-17 | PUMP_START | state=RUNNING | quality=GOOD
2026-10-12 08:14:05.300 +05:30 | INFO | HISTORIAN | COOLING-PUMP-17 | SENSOR_READING | tag=CP17_DISCHARGE_PRESSURE; value=5.8; unit=bar | quality=GOOD
2026-10-12 08:14:08.000 +05:30 | WARN | PLC-CP17 | COOLING-PUMP-17 | HIGH_VIBRATION_ALARM | tag=CP17_VIBRATION; value=8.4; unit=mm/s; limit=7.0; alarm_id=ALM-CP17-884 | quality=GOOD
2026-10-12 08:14:09.200 +05:30 | INFO | ALARM-MANAGER | COOLING-PUMP-17 | ALARM_STATUS | alarm_id=ALM-CP17-884; status=ACTIVE | quality=GOOD
2026-10-12 08:16:40.000 +05:30 | INFO | OPERATOR-LOG | COOLING-PUMP-17 | OBSERVATION | Visible vibration at pump casing; no fluid leak observed | quality=GOOD
2026-10-12 08:19:12.500 +05:30 | INFO | HISTORIAN | COOLING-PUMP-17 | SENSOR_READING | tag=CP17_VIBRATION; value=4.1; unit=mm/s | quality=GOOD
2026-10-12 08:19:13.000 +05:30 | INFO | ALARM-MANAGER | COOLING-PUMP-17 | ALARM_STATUS | alarm_id=ALM-CP17-884; status=CLEARED; reason=vibration below configured limit | quality=GOOD
2026-10-12 08:24:00.000 +05:30 | INFO | CMMS-BRIDGE | COOLING-PUMP-17 | WORK_ORDER_OPENED | work_order=WO-CP17-211; inspect pump vibration and coupling alignment | quality=GOOD
2026-10-12 09:02:11.000 +05:30 | INFO | CMMS-BRIDGE | COOLING-PUMP-17 | WORK_ORDER_ASSIGNED | work_order=WO-CP17-211; team=MECHANICAL-MAINTENANCE | quality=GOOD
2026-10-12 10:35:45.000 +05:30 | INFO | CMMS-BRIDGE | COOLING-PUMP-17 | WORK_NOTE_ADDED | work_order=WO-CP17-211; inspection scheduled; no repair completion recorded | quality=GOOD
"""
    path = tmp_path / "cp17.txt"
    path.write_text(content, encoding="utf-8")
    records = load_any(path, stage="UTILITIES")
    assert len(records) == 10
    assert all(record.timestamp is not None for record in records)
    assert [r.event_type for r in records] == [
        "pump_start", "sensor_reading", "high_vibration_alarm", "alarm_status", "observation",
        "sensor_reading", "alarm_status", "work_order_opened", "work_order_assigned", "work_note_added",
    ]
    pipe = IngestionPipeline(agent.memory, FeatureProvider(EMB, None), build_scorer("store_all"), "store_all", False)
    pipe.ingest(records, None)

    out = ask(agent, "What happened to COOLING-PUMP-17 on 2026-10-12, when did the alarm clear, "
                     "and what maintenance action was recorded?")
    assert out["answer_status"] == "answered", out["final_answer"]
    assert out["verification_status"] == "supported", out["verification"].get("abstention_reason")
    assert "HIGH_VIBRATION_ALARM" in out["answer_value"]
    assert "CP17_VIBRATION=8.4 mm/s" in out["answer_value"]
    assert "7.0" in out["answer_value"]
    assert "2026-10-12T08:19:13" in out["answer_value"]
    assert "CP17_VIBRATION=4.1 mm/s" in out["answer_value"]
    assert "reason=vibration below configured limit" in out["answer_value"]
    assert "WO-CP17-211" in out["answer_value"]
    assert "inspect pump vibration and coupling alignment" in out["answer_value"]
    assert "insufficient evidence" not in out["answer_value"].lower()
    assert len([e for e in out["evidence"] if e["role"] == "answer"]) >= 3

    with_context = ask(agent, out["question"], records)
    assert with_context["answer_status"] == "answered", with_context["final_answer"]
    assert with_context["verification_status"] == "supported"
    assert not with_context["retrieval_triggered"]
    assert "HIGH_VIBRATION_ALARM" in with_context["answer_value"]
    assert "CP17_VIBRATION=8.4 mm/s" in with_context["answer_value"]
    assert "7.0" in with_context["answer_value"]
    assert "2026-10-12T08:19:13" in with_context["answer_value"]
    assert "CP17_VIBRATION=4.1 mm/s" in with_context["answer_value"]
    assert "WO-CP17-211" in with_context["answer_value"]
    assert "inspect pump vibration and coupling alignment" in with_context["answer_value"]
