"""Ladle datasets: short and long share gold; a value that restates the question is never a candidate."""
from __future__ import annotations

from agentmamba.data import ladle_benchmark as lb
from agentmamba.data.document_loader import load_any
from agentmamba.config import load_config
from agentmamba.pipeline.agent import AgentMamba
from agentmamba.retrieval.query_processor import IntentClassifier, QueryProcessor
from agentmamba.verification.document_evidence import _restates_question, text_candidates


def test_short_and_long_share_questions_and_gold(tmp_path):
    lb.write_datasets(tmp_path, seed=5, n_scenarios=1)
    for name in ("questions.jsonl", "annotations.jsonl"):
        s = (tmp_path / "short" / name).read_text(encoding="utf-8").replace('"short"', '"X"')
        lo = (tmp_path / "long" / name).read_text(encoding="utf-8").replace('"long"', '"X"')
        assert s == lo
    short = lb.stages_of("short", "SMS1", tmp_path / "short")
    long_ = lb.stages_of("long", "SMS1", tmp_path / "long")
    assert short == [1] and long_ == [1, 2, 3, 4]
    n_short = len(lb.stage_events("short", "SMS1", 1, tmp_path / "short"))
    n_long = sum(len(lb.stage_events("long", "SMS1", k, tmp_path / "long")) for k in long_)
    assert n_short < 40 < 2000 < n_long


def test_signal_name_is_not_an_answer(tmp_path):
    p = tmp_path / "ladle_historian_20260302.csv"
    p.write_text("timestamp,equipment,signal,value,unit\n2026-03-02T11:30:00,LADLE-11,shell_temperature,368,degC\n",
                 encoding="utf-8")
    q = "What was the shell temperature reading for LADLE-11 at 2026-03-02 11:30?"
    qp = QueryProcessor(IntentClassifier.load(AgentMamba.artifact_paths(load_config())["intent"]))
    vals = [c.value for c in text_candidates(qp.parse(q), load_any(p, stage="LADLE"))]
    assert "368 degC" in vals and "shell_temperature" not in vals
    assert _restates_question("shell_temperature", q)
    assert not _restates_question("368 degC", q)
