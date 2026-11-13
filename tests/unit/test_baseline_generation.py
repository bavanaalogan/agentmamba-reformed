from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentmamba.data.document_loader import load_text_log
from agentmamba.models.mamba_backbone import MambaBackbone
from agentmamba.models.answerer import FirstCandidateAnswerer
from agentmamba.pipeline.baseline import BaselineMamba, ContextWindow
from agentmamba.retrieval.query_processor import QueryProcessor
from tests.conftest import make_event


class FakeBackbone:
    model_id = "fake-mamba"

    def __init__(self, response="generated answer"):
        self.response = response
        self.last_call = {"cache_hit": False, "prompt_tokens": 42, "generated_tokens": 3}
        self.call = None

    def generate(self, prompt, max_new_tokens, max_prompt_tokens):
        self.call = (prompt, max_new_tokens, max_prompt_tokens)
        return self.response


class FakeQueryProcessor:
    def parse(self, _question):
        return SimpleNamespace(causal_language=False)


def _baseline(backbone):
    return BaselineMamba(FakeQueryProcessor(), SimpleNamespace(backbone=backbone), lambda text: len(text.split()),
                         token_budget=256)


def test_raw_generation_uses_only_the_supplied_context_and_reports_no_confidence():
    event = make_event("E1", text="Pump alarm cleared at 08:19.")
    window = ContextWindow([event], 20, 256, 1, 1, [])
    backbone = FakeBackbone()

    out = _baseline(backbone).generate("When did the alarm clear?", window)

    prompt, max_new, max_prompt = backbone.call
    assert "[E1]" in prompt and "When did the alarm clear?" in prompt
    assert max_new == 64 and max_prompt == 256
    assert out["generation_mode"] == "raw_mamba_generation"
    assert out["answer_value"] == "generated answer"
    assert out["confidence"]["candidate_share"] is None
    assert out["evidence"] == [] and out["verification_status"] == "not_verified"
    assert out["model_call"]["generated_tokens"] == 3


def test_raw_generation_reports_empty_output_as_abstained():
    out = _baseline(FakeBackbone("  ")).generate("question?", ContextWindow([], 0, 256, 0, 0, []))

    assert out["answer_status"] == "abstained"
    assert out["answer_value"] is None
    assert out["no_candidate"]


def test_candidate_baseline_answers_each_compound_part_from_initial_context(tmp_path):
    source = tmp_path / "incident.txt"
    source.write_text(
        "2026-10-03 09:12:04 +05:30 | CRITICAL | ALARM-MANAGER | COOLING-WATER-PUMP-03 | "
        "ALARM_ASSERTED | alarm_id=ALM-CWP03-771; discharge_pressure=2.1 bar; low_limit=2.8 bar\n"
        "2026-10-03 11:42:10 +05:30 | CRITICAL | MAINTENANCE-LOG | COOLING-WATER-PUMP-03 | "
        "INSPECTION_FINDING | suction_strainer=70_percent_obstructed_by_scale; "
        "coupling_alignment=0.42_mm_outside_tolerance\n"
        "2026-10-03 13:25:00 +05:30 | WARN | MAINTENANCE-LOG | COOLING-WATER-PUMP-03 | "
        "MAINTENANCE_ACTION | suction_strainer=cleaned; coupling=realigned; bearing=checked; "
        "bearing_replacement=not_recorded\n"
        "2026-10-03 14:15:00 +05:30 | INFO | ALARM-MANAGER | COOLING-WATER-PUMP-03 | "
        "ALARM_CLEARED | alarm_id=ALM-CWP03-771\n",
        encoding="utf-8")
    records = load_text_log(source, stage="UTILITIES")

    class FreeformClassifier:
        def predict(self, _text):
            return "freeform", 1.0

    question = ("For COOLING-WATER-PUMP-03 on 2026-10-03, what alarm occurred, what did the inspection find, "
                "what maintenance action was completed, and when was the alarm cleared?")
    window = ContextWindow(records, 400, 2048, 0, 0, [])
    baseline = BaselineMamba(QueryProcessor(FreeformClassifier()), FirstCandidateAnswerer(),
                             lambda text: len(text.split()), token_budget=2048)

    out = baseline.answer(question, window)

    assert out["answer_status"] == "answered"
    assert not out["retrieval_triggered"]
    assert out["verification_status"] == "not_verified"
    assert "ALARM ASSERTED" in out["answer_value"]
    assert "ALARM CLEARED" not in out["answer_value"].split("; what did the inspection find", 1)[0]
    assert "70 percent obstructed by scale" in out["answer_value"]
    assert "suction strainer=cleaned" in out["answer_value"]
    assert "2026-10-03T14:15:00+05:30" in out["answer_value"]
    assert out["model_call"]["component_calls"] == 4
    assert len(out["answer_components"]) == 4
    assert out["answer_components"][0]["question"] == "what alarm occurred"
    assert out["answer_components"][0]["record_ids"]
    assert out["confidence"]["candidate_share"] is None


def test_raw_generation_requires_a_generation_capable_mamba():
    baseline = _baseline(SimpleNamespace())

    with pytest.raises(RuntimeError, match="requires an answerer backed by Mamba"):
        baseline.generate("question?", ContextWindow([], 0, 256, 0, 0, []))


def test_backbone_prompt_truncation_preserves_the_question_at_the_end():
    import torch

    class Tokenizer:
        def __call__(self, _prompt, return_tensors):
            assert return_tensors == "pt"
            ids = torch.tensor([[1, 2, 3, 4, 5]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def decode(self, tokens, skip_special_tokens):
            assert skip_special_tokens
            return "answer"

    class Model:
        def generate(self, input_ids, attention_mask, max_new_tokens, do_sample):
            assert input_ids.tolist() == [[4, 5]]
            assert attention_mask.tolist() == [[1, 1]]
            assert max_new_tokens == 2 and not do_sample
            return torch.cat([input_ids, torch.tensor([[6]])], dim=1)

    backbone = MambaBackbone(max_tokens=8)
    backbone._tok = Tokenizer()
    backbone._model = Model()

    assert backbone.generate("prompt", max_new_tokens=2, max_prompt_tokens=2) == "answer"
    assert backbone.last_call["prompt_tokens"] == 2
    assert backbone.last_call["generated_tokens"] == 1
