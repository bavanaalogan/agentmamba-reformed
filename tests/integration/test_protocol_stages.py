"""Stage-1 / stage-2 protocol on the user's synthetic log files (data/protocols/cp17_context_growth),
through the real service with deterministic stand-ins (HashingEmbedder, first-candidate ranking,
whitespace token counter). Real-model behaviour is measured by scripts/run_protocol.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentmamba.config import PROJECT_ROOT
from agentmamba.data.document_loader import is_log_text, load_any, parse_log_line
from agentmamba.models.answerer import FirstCandidateAnswerer, format_record
from agentmamba.pipeline.agent import AgentMamba
from agentmamba.pipeline.service import AgentService

PDIR = PROJECT_ROOT / "data" / "protocols" / "cp17_context_growth"
S1, S2 = PDIR / "agentmamba_stage1_small_context.txt", PDIR / "agentmamba_stage2_additional_logs.txt"
Q1 = "What vibration value triggered the high vibration alarm on COOLING-PUMP-17 on 2026-10-12?"
Q4 = "What was the confirmed root cause of the high vibration alarm on COOLING-PUMP-17 on 2026-10-12?"
COUNT = lambda s: len(s.split()) * 3  # noqa: E731  conservative stand-in for the Mamba tokenizer

pytestmark = pytest.mark.skipif(not S1.exists(), reason="protocol data not present")


def test_text_log_lines_become_one_record_each_with_citations():
    evs = load_any(S1, stage="UTILITIES")
    assert len(evs) == 10 and all(e.event_type != "document_chunk" for e in evs)
    alarm = next(e for e in evs if "HIGH_VIBRATION_ALARM" in e.text)
    assert alarm.source_ref.endswith("#line=6") and alarm.asset_id == "COOLING-PUMP-17"
    assert alarm.timestamp == "2026-10-12T08:14:08+05:30" and alarm.timezone == "+05:30"
    assert alarm.obs("CP17_VIBRATION").raw == "8.4" and alarm.obs("CP17_VIBRATION").unit == "mm/s"
    assert alarm.text.startswith("2026-10-12 08:14:08.000 +05:30 | WARN")          # verbatim evidence line
    assert format_record(alarm) == f"[{alarm.event_id}] {alarm.text}"              # no duplicated prefix
    wo = next(e for e in evs if "status=OPENED" in e.text)
    assert wo.work_order_id == "WO-CP17-211"


def test_log_line_parser_keeps_values_and_avoids_key_collisions():
    row = parse_log_line('2026-10-12 08:14:09.200 +05:30 | INFO | ALARM-MANAGER | alarm_id=ALM-1 | source=CP17_VIB '
                         '| note="no fluid leak; visible vibration"')
    assert row["log_source"] == "ALARM-MANAGER" and row["source"] == "CP17_VIB" and row["log_level"] == "INFO"
    assert row["note"] == "no fluid leak; visible vibration"
    assert is_log_text(S1.read_text(encoding="utf-8")) and not is_log_text("# Report\n\nThe pump was inspected.\n\n"
                                                                             "Nothing else happened today.\n\nEnd.")


@pytest.fixture(scope="module")
def svc(tmp_path_factory, trained_qp_controller):
    from agentmamba.config import load_config
    from agentmamba.models.embedding_model import HashingEmbedder

    d = tmp_path_factory.mktemp("proto")
    cfg = load_config(overrides={"paths.artifacts_dir": str(d / "artifacts"), "models.answerer": "first_candidate",
                                 "memory.policy": "store_all"})
    qp, ctl = trained_qp_controller
    paths = AgentMamba.artifact_paths(cfg)
    qp.classifier.save(paths["intent"])
    ctl.save(paths["controller"])
    return AgentService(cfg, d / "profile", embedder=HashingEmbedder(), answerer=FirstCandidateAnswerer(),
                        token_counter=COUNT)


def test_stage1_context_is_sufficient_and_retrieval_is_skipped(svc):
    rep = svc.ingest_file(S1, "UTILITIES", "authorized", None, "store_all")
    assert rep["received"] == rep["written"] == 10 and rep["chunks_created"] == 0
    res = svc.compare(Q1, [], 2048)
    b, a = res["baseline"], res["enhanced"]
    assert res["shared_context"]["n_records"] == 10                # the whole incident, not one stray record
    assert b["retrieval_triggered"] is False and b["context"]["record_ids"] == res["shared_context"]["record_ids"]
    assert a["sufficiency"]["decision"] == "sufficient" and a["retrieval_triggered"] is False
    assert a["answer_value"] == "8.4 mm/s"
    assert next(e for e in a["evidence"] if e["role"] == "answer")["source"].endswith("#line=6")


def test_compound_log_question_combines_supported_facts_from_separate_rows(svc):
    svc.ingest_file(S1, "UTILITIES", "authorized", None, "store_all")
    question = ("What happened to COOLING-PUMP-17 on 2026-10-12, when did the alarm clear, "
                "and what maintenance action was recorded?")

    result = svc.compare(question, [], 2048)
    assert result["errors"] == {}
    out = result["enhanced"]

    assert out["answer_status"] == "answered", out["final_answer"]
    assert "HIGH_VIBRATION_ALARM" in out["answer_value"]
    assert "8.4 mm/s" in out["answer_value"]
    assert "2026-10-12T08:19:13" in out["answer_value"]
    assert "Inspect pump vibration and coupling alignment" in out["answer_value"]
    assert out["verification_status"] == "supported"
    assert len([e for e in out["evidence"] if e["role"] == "answer"]) >= 3


def test_stage2_same_question_triggers_retrieval_and_recovers_the_cited_line(svc):
    rep = svc.ingest_file(S2, "UTILITIES", "authorized", None, "store_all")
    assert rep["written"] == 300 and svc.memory.stats()["records"] == 310
    res = svc.compare(Q1, [], 2048)
    b, a = res["baseline"], res["enhanced"]
    ctx = set(res["shared_context"]["record_ids"])
    target = next(e for e in load_any(S1, stage="UTILITIES") if e.source_ref.endswith("#line=6")).event_id
    assert target not in ctx                                       # the incident left the shared window
    assert all(e["record_id"] in ctx for e in b["evidence"])       # the baseline never sees retrieved records
    assert a["sufficiency"]["decision"] == "insufficient" and a["retrieval_triggered"]
    st = a["retrieval_stats"]
    assert st["n_selected"] > 0 and st["n_removed_by_date_time_filter"] > 0
    assert target in a["retrieved_record_ids"] and a["answer_value"] == "8.4 mm/s"
    assert a["verification_status"] == "supported" and a["invalid_citations"] == []


def test_unanswerable_root_cause_is_not_invented(svc):
    a = svc.compare(Q4, [], 2048)["enhanced"]
    assert a["answer_status"] == "abstained" and a["answer_value"] is None
    assert "root" in a["abstention_reason"] or "cause" in a["abstention_reason"]


def test_gold_file_is_separate_from_what_systems_see():
    proto = json.loads((PDIR / "protocol.json").read_text(encoding="utf-8"))
    assert all(set(q) == {"question_id", "kind", "text"} for q in proto["questions"])
    gold = (PDIR / "annotations" / "gold.json").read_text(encoding="utf-8")
    assert "8.4 mm/s" in gold and "8.4 mm/s" not in json.dumps(proto)
