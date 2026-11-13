"""General end-to-end regressions for log/document evidence and conditional retrieval."""
from __future__ import annotations

import json

import pytest

from agentmamba.data.document_loader import load_any, load_text_log
from agentmamba.memory.schema import Event
from agentmamba.models.answerer import FirstCandidateAnswerer
from agentmamba.pipeline.agent import AgentMamba, QueryRequest
from agentmamba.pipeline.builder import build_memory
from agentmamba.pipeline.ingestion import FeatureProvider, IngestionPipeline
from agentmamba.gate.predictor import build_scorer
from agentmamba.models.embedding_model import HashingEmbedder

EMB = HashingEmbedder()


@pytest.fixture
def investigator(cfg, dataset, trained_qp_controller, tmp_path):
    mem, _ = build_memory(cfg, tmp_path / "memory", dataset.events_for("test"), policy="store_all",
                          budget_fraction=1.0, embedder=EMB, feature_store=None)
    qp, controller = trained_qp_controller
    return AgentMamba(cfg, mem, FirstCandidateAnswerer(), qp, controller)


def _log(event_id, timestamp, asset, event_type, **fields):
    raw = {"timestamp": timestamp, "equipment": asset, "event_type": event_type.upper(), **fields}
    details = "; ".join(f"{key}={value}" for key, value in fields.items())
    text = f"{timestamp} | INFO | {asset} | {event_type.upper()} | {details}"
    return Event(event_id=event_id, stage="UTILITIES", event_type=event_type, timestamp=timestamp, text=text,
                 source_ref=f"file://plant.log#{event_id}", source_type="authorized", asset_id=asset, raw=raw)


def _ingest(agent, records):
    pipe = IngestionPipeline(agent.memory, FeatureProvider(EMB, None), build_scorer("store_all"), "store_all", False)
    pipe.ingest(records, None)


def _ask(agent, question, context=()):
    return agent.query(QueryRequest(question, list(context)))


def test_complete_initial_context_returns_verified_event_time_without_retrieval(investigator):
    asset = "ZX-PUMP-41"
    acknowledged = _log("ACK-41", "2026-10-12T08:15:00+05:30", asset, "alarm_status",
                        alarm_id="ALM-41", status="ACKNOWLEDGED")
    cleared = _log("CLEAR-41", "2026-10-12T08:19:00+05:30", asset, "alarm_status",
                   alarm_id="ALM-41", status="CLEARED")

    out = _ask(investigator, "When was the alert acknowledged for pump ZX-PUMP-41 on 2026-10-12?",
               [acknowledged, cleared])

    assert out["sufficiency"]["decision"] == "sufficient"
    assert not out["retrieval_triggered"]
    assert out["verification_status"] == "supported"
    assert out["answer_value"] == acknowledged.timestamp
    assert {e["record_id"] for e in out["evidence"] if e["role"] == "answer"} == {"ACK-41"}


def test_compound_incident_is_supported_initially_then_retrieved_when_outside_context(investigator, tmp_path):
    asset = "COOLING-WATER-PUMP-03"
    question = (f"For {asset} on 2026-10-03, what alarm occurred, what did the inspection find, "
                "what maintenance action was completed, and when was the alarm cleared?")
    source = tmp_path / "stage1.log"
    source.write_text(
        "# Fictional industrial log\n"
        "2026-10-03 09:12:04 +05:30 | CRITICAL | ALARM-MANAGER | COOLING-WATER-PUMP-03 | "
        "ALARM_ASSERTED | alarm_id=ALM-CWP03-771; discharge_pressure=2.1 bar; low_limit=2.8 bar\n"
        "2026-10-03 09:12:09 +05:30 | WARN | HISTORIAN | COOLING-WATER-PUMP-03 | PROCESS_READING | "
        "suction_pressure=4.6 bar; motor_current=31.8 A\n"
        "2026-10-03 09:14:20 +05:30 | WARN | OPERATOR-LOG | COOLING-WATER-PUMP-03 | OPERATOR_OBSERVATION | "
        "intermittent_rattling=non-drive-end_bearing_housing; visible_leak=no\n"
        "2026-10-03 09:18:00 +05:30 | WARN | PLC-CWP03 | COOLING-WATER-PUMP-03 | STATE_CHANGE | "
        "standby_pump=CWP-04_started; CWP-03=isolated_from_process_header\n"
        "2026-10-03 10:06:31 +05:30 | INFO | CMMS | COOLING-WATER-PUMP-03 | WORK_ORDER_OPENED | "
        "work_order=WO-MECH-8831; scope=inspect_coupling,suction_strainer,non-drive-end_bearing\n"
        "2026-10-03 11:42:10 +05:30 | CRITICAL | MAINTENANCE-LOG | COOLING-WATER-PUMP-03 | "
        "INSPECTION_FINDING | work_order=WO-MECH-8831; suction_strainer=70_percent_obstructed_by_scale; "
        "coupling_alignment=0.42_mm_outside_tolerance\n"
        "2026-10-03 13:25:00 +05:30 | WARN | MAINTENANCE-LOG | COOLING-WATER-PUMP-03 | MAINTENANCE_ACTION | "
        "work_order=WO-MECH-8831; suction_strainer=cleaned; coupling=realigned; bearing=checked; "
        "bearing_replacement=not_recorded\n"
        "2026-10-03 14:15:00 +05:30 | INFO | ALARM-MANAGER | COOLING-WATER-PUMP-03 | ALARM_CLEARED | "
        "alarm_id=ALM-CWP03-771; clear_condition=pressure_above_recovery_threshold_for_5_minutes\n",
        encoding="utf-8")
    history = load_text_log(source, stage="UTILITIES")
    _ingest(investigator, history)
    context_extras = [
        _log(f"CONTEXT-CWP03-{n}", f"2026-10-03T13:{n:02}:00", asset, "sensor_reading",
             tag=f"FLOW-{n}", value=str(100 + n), unit="m3/h")
        for n in (10, 20, 30, 40)
    ]
    initial = _ask(investigator, question, history + context_extras)

    assert initial["controller"]["action"] == "RETURN_INITIAL"
    assert not initial["retrieval_triggered"]
    assert initial["verification_status"] == "supported"
    assert len(initial["answer_components"]) == 4
    assert initial["answer_components"][0]["question"] == "what alarm occurred"
    assert initial["answer_components"][0]["record_ids"]
    for expected in ("ALM-CWP03-771", "discharge pressure=2.1 bar", "70 percent obstructed by scale",
                     "0.42 mm outside tolerance", "suction strainer=cleaned", "coupling=realigned",
                     "bearing=checked", "bearing replacement=not recorded", "14:15:00"):
        assert expected.lower() in initial["answer_value"].lower()
    alarm_answer = initial["answer_value"].split("; what did the inspection find", 1)[0].lower()
    assert "alarm cleared" not in alarm_answer

    expanded_context = [
        _log(f"RECENT-CWP03-{n}", f"2026-10-03T15:{n:02}:00", asset, "sensor_reading",
             tag=f"FLOW-{n}", value=str(110 + n), unit="m3/h")
        for n in range(24)
    ]
    retrieved = _ask(investigator, question, expanded_context)

    assert retrieved["retrieval_triggered"]
    assert retrieved["verification_status"] == "supported"
    for expected in ("ALM-CWP03-771", "discharge pressure=2.1 bar", "70 percent obstructed by scale",
                     "0.42 mm outside tolerance", "suction strainer=cleaned", "coupling=realigned",
                     "bearing=checked", "bearing replacement=not recorded", "14:15:00"):
        assert expected.lower() in retrieved["answer_value"].lower()
    answer_record_ids = {item["record_id"] for item in retrieved["evidence"] if item["role"] == "answer"}
    expected_source_ids = {record.event_id for record in history
                           if record.event_type in {"alarm_asserted", "inspection_finding",
                                                    "maintenance_action", "alarm_cleared"}}
    assert expected_source_ids <= answer_record_ids


def test_jsonl_sensor_record_is_normalized_and_answered_from_initial_context(investigator, tmp_path):
    source = tmp_path / "historian.jsonl"
    source.write_text(json.dumps({
        "time": "12/10/2026 08:15:00",
        "equipment_id": "ZX-PUMP-40",
        "record_type": "sensor_reading",
        "signal": "vibration",
        "value": "7.2",
        "unit": "mm/s",
        "severity": "warning",
    }) + "\n", encoding="utf-8")
    record, = load_any(source, stage="UTILITIES")

    out = _ask(investigator, "What vibration reading was recorded for pump ZX-PUMP-40 on 2026-10-12?",
               [record])

    assert record.timestamp == "2026-10-12T08:15:00"
    assert record.severity == "major"
    assert record.obs("vibration").raw == "7.2"
    assert out["verification_status"] == "supported"
    assert "7.2 mm/s" in out["answer_value"]
    assert not out["retrieval_triggered"]


def test_partial_initial_context_retrieves_and_completes_missing_question_part(investigator):
    asset = "ZX-PUMP-42"
    acknowledged = _log("ACK-42", "2026-10-12T08:15:00+05:30", asset, "alarm_status",
                        alarm_id="ALM-42", status="ACKNOWLEDGED")
    action = _log("ACTION-42", "2026-10-12T08:24:00+05:30", asset, "work_order_opened",
                  work_order="WO-42", action="inspect coupling alignment")
    _ingest(investigator, [acknowledged, action])

    out = _ask(investigator, "When was the alarm acknowledged for pump ZX-PUMP-42 on 2026-10-12, "
                              "and what maintenance action was recorded?", [acknowledged])

    assert out["retrieval_triggered"]
    assert out["verification_status"] == "supported"
    assert acknowledged.timestamp in out["answer_value"]
    assert "inspect coupling alignment" in out["answer_value"]
    assert any(e["record_id"] == "ACTION-42" for e in out["evidence"])


def test_evidence_only_in_persistent_memory_is_retrieved_and_verified(investigator):
    asset = "ZX-PUMP-43"
    action = _log("ACTION-43", "2026-10-12T08:24:00+05:30", asset, "work_order_opened",
                  work_order="WO-43", action="replace the worn seal")
    _ingest(investigator, [action])

    out = _ask(investigator, "What maintenance action was recorded for pump ZX-PUMP-43 on 2026-10-12?")

    assert out["retrieval_triggered"]
    assert out["verification_status"] == "supported"
    assert out["answer_value"] == "replace the worn seal"
    assert any(e["record_id"] == "ACTION-43" for e in out["evidence"])
    assert out["invalid_citations"] == []


def test_raw_alarm_identifier_retrieves_related_rows(investigator):
    raised = _log("RAISE-52", "2026-10-12T08:14:00+05:30", "ZX-PUMP-52", "alarm_status",
                  alarm_id="ALM-52", status="ACTIVE")
    cleared = _log("CLEAR-52", "2026-10-12T08:19:00+05:30", "ZX-PUMP-52", "alarm_status",
                   alarm_id="ALM-52", status="CLEARED")
    for record in (raised, cleared):
        record.raw["Alarm ID"] = record.raw.pop("alarm_id")
    _ingest(investigator, [raised, cleared])

    out = _ask(investigator, "When did alarm ALM-52 clear on 2026-10-12?")

    assert out["verification_status"] == "supported"
    assert out["answer_value"] == cleared.timestamp
    assert out["retrieval_stats"]["n_expansion_hits"] >= 1
    assert {e["record_id"] for e in out["evidence"] if e["role"] == "answer"} == {"CLEAR-52"}


def test_timestamps_are_bound_to_matching_alarm_state_not_other_events(investigator):
    asset = "ZX-PUMP-44"
    records = [
        _log("RAISE-44", "2026-10-12T08:14:00+05:30", asset, "alarm_status",
             alarm_id="ALM-44", status="ACTIVE"),
        _log("ACK-44", "2026-10-12T08:15:00+05:30", asset, "alarm_status",
             alarm_id="ALM-44", status="ACKNOWLEDGED"),
        _log("CLEAR-44", "2026-10-12T08:19:00+05:30", asset, "alarm_status",
             alarm_id="ALM-44", status="CLEARED"),
        _log("READ-44", "2026-10-12T08:22:00+05:30", asset, "sensor_reading", pressure="5.8 bar"),
    ]

    out = _ask(investigator, "When did the alarm clear for pump ZX-PUMP-44 on 2026-10-12?", records)

    assert out["verification_status"] == "supported"
    assert out["answer_value"] == records[2].timestamp
    assert {e["record_id"] for e in out["evidence"] if e["role"] == "answer"} == {"CLEAR-44"}


def test_conflicting_measurements_are_flagged_and_not_chosen(investigator):
    asset = "ZX-PUMP-45"
    records = [
        _log("VIB-A", "2026-10-12T08:14:00+05:30", asset, "sensor_reading", vibration="8.2 mm/s"),
        _log("VIB-B", "2026-10-12T08:14:00+05:30", asset, "sensor_reading", vibration="9.1 mm/s"),
    ]

    out = _ask(investigator, "What vibration reading was recorded for pump ZX-PUMP-45 on 2026-10-12?", records)

    assert out["verification_status"] == "conflict"
    assert out["answer_status"] == "abstained"
    assert {v["value"] for v in out["verification"]["conflict"]["values"]} == {"8.2 mm/s", "9.1 mm/s"}


def test_conflicting_component_in_compound_question_remains_visible(investigator):
    asset = "ZX-PUMP-54"
    records = [
        _log("VIB-A-54", "2026-10-12T08:14:00+05:30", asset, "sensor_reading", vibration="8.2 mm/s"),
        _log("VIB-B-54", "2026-10-12T08:14:00+05:30", asset, "sensor_reading", vibration="9.1 mm/s"),
        _log("ACTION-54", "2026-10-12T08:24:00+05:30", asset, "maintenance_completed",
             action="coupling alignment inspected", status="completed"),
    ]

    out = _ask(investigator,
               "What vibration reading was recorded for pump ZX-PUMP-54 on 2026-10-12, "
               "and what maintenance action was completed?",
               records)

    assert out["verification_status"] == "conflict"
    assert out["answer_status"] == "abstained"
    assert out["verification"]["conflict"]["question_part"].startswith("What vibration reading")
    assert {v["value"] for v in out["verification"]["conflict"]["values"]} == {"8.2 mm/s", "9.1 mm/s"}
    assert {item["record_id"] for item in out["evidence"] if item["role"] == "conflicting"} == {
        "VIB-A-54", "VIB-B-54"}


def test_unstructured_document_text_without_metadata_can_support_an_answer(investigator):
    asset = "ZX-PUMP-46"
    document = Event(event_id="DOC-46", stage="MAINTENANCE", event_type="document_chunk", timestamp=None,
                     text="Field note for pump ZX-PUMP-46 on 2026-10-12: discharge pressure was 6.4 bar.",
                     source_ref="file://field-note.md#chunk=0", source_type="authorized",
                     raw={"chunk_index": 0, "file": "field-note.md"})

    out = _ask(investigator, "What discharge pressure was recorded for pump ZX-PUMP-46 on 2026-10-12?",
               [document])

    assert out["verification_status"] == "supported"
    assert "6.4 bar" in out["answer_value"]
    assert not out["retrieval_triggered"]
    assert out["evidence"][0]["record_id"] == "DOC-46"


def test_alternate_alarm_wording_matches_the_correct_status_value(investigator):
    asset = "ZX-PUMP-47"
    acknowledged = _log("ACK-47", "2026-10-12T08:15:00+05:30", asset, "alarm_status",
                        alarm_id="ALM-47", status="ACKNOWLEDGED")

    out = _ask(investigator, "At what time was the alert acknowledged for pump ZX-PUMP-47 on 2026-10-12?",
               [acknowledged])

    assert out["verification_status"] == "supported"
    assert out["answer_value"] == acknowledged.timestamp
    assert out["evidence"][0]["record_id"] == "ACK-47"


@pytest.mark.parametrize("question", [
    "What was logged for pump ZX-PUMP-48 and pump ZX-PUMP-49?",
    "What was logged for pump ZX-PUMP-48 on 2026-10-12 and 2026-10-13?",
])
def test_multiple_candidate_entities_or_dates_require_clarification(investigator, question):
    out = _ask(investigator, question)

    assert out["answer_status"] == "needs_clarification"
    assert not out["retrieval_triggered"]
    assert "multiple" in out["abstention_reason"]


def test_asset_mention_without_requested_fact_does_not_count_as_evidence(investigator):
    asset = "ZX-PUMP-50"
    mention = _log("MENTION-50", "2026-10-12T08:14:00+05:30", asset, "state_change", state="RUNNING")
    _ingest(investigator, [mention])

    out = _ask(investigator, "What maintenance action was recorded for pump ZX-PUMP-50 on 2026-10-12?",
               [mention])

    assert out["retrieval_triggered"]
    assert out["answer_status"] == "abstained"
    assert out["verification_status"] == "insufficient"
    assert "action" in out["abstention_reason"]


def test_cause_is_not_inferred_from_temporal_association(investigator):
    asset = "ZX-PUMP-51"
    record = Event(event_id="OBS-51", stage="UTILITIES", event_type="observation",
                   timestamp="2026-10-12T08:14:00+05:30",
                   text="Pump ZX-PUMP-51 vibration increased after the inlet valve was closed.",
                   source_ref="file://observations.log#OBS-51", source_type="authorized", asset_id=asset,
                   raw={"timestamp": "2026-10-12T08:14:00+05:30", "equipment": asset,
                        "event_type": "observation"})
    _ingest(investigator, [record])

    out = _ask(investigator, "What caused the vibration on pump ZX-PUMP-51 on 2026-10-12?")

    assert out["answer_status"] == "abstained"
    assert out["answer_value"] is None
    assert "cause" in out["abstention_reason"].lower() or "caus" in out["abstention_reason"].lower()
