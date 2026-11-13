"""Tests that load REAL weights (state-spaces/mamba-130m-hf, all-MiniLM-L6-v2).
Run with:  pytest -m model   (they are slow on CPU and need the HF cache or network)."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def mamba(tmp_path_factory):
    from agentmamba.models.mamba_backbone import MambaBackbone

    return MambaBackbone(cache_path=tmp_path_factory.mktemp("mc") / "c.sqlite3", max_tokens=256).load()


def test_mamba_loads_and_is_the_configured_model(mamba):
    assert mamba.loaded and mamba.model_id == "state-spaces/mamba-130m-hf"
    assert type(mamba._model).__name__ == "MambaForCausalLM"


def test_mamba_record_features(mamba):
    f = mamba.record_features(["Ladle L1 refractory check: hairline crack at slag line.", "Routine shift log."],
                              max_tokens=32, batch_size=2)
    assert f.shape == (2, 2 + 768) and np.isfinite(f).all() and (f[:, 0] > 0).all() and (f[:, 1] >= f[:, 0]).all()


def test_mamba_candidate_scoring_is_deterministic_and_padding_invariant(mamba):
    prompt = "Plant records:\n[A] Heat H12 tapped into ladle LD04.\nQuestion: Which ladle carried heat H12?\nAnswer:"
    s2 = mamba.score_candidates(prompt, ["LD04", "LD02 with a much longer tail"])
    s1 = mamba.score_candidates(prompt + " ", ["LD04"])          # different key -> recomputed alone
    again = mamba.score_candidates(prompt, ["LD04", "LD02 with a much longer tail"])
    assert len(s2) == 2 and all(math.isfinite(x) for x in s2) and s2 == again
    alone = mamba.score_candidates(prompt, ["LD04"])
    assert alone[0] == pytest.approx(s2[0], abs=1e-4)              # candidates scored independently
    assert isinstance(s1[0], float)
    ref = mamba.score_candidates_full(prompt, ["LD04", "LD02 with a much longer tail"])
    assert s2 == pytest.approx(ref, abs=1e-3)                      # cached decoding == full forward (padded batch)


def test_sentence_embedder_dims_and_normalization():
    from agentmamba.models.embedding_model import SentenceEmbedder

    e = SentenceEmbedder()
    v = e.encode(["ladle refractory crack", "coil surface defect"])
    assert v.shape == (2, 384) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)
    assert e.encode([]).shape == (0, 384)


def test_baseline_window_token_count_matches_what_mamba_reads(mamba, tmp_path):
    """The baseline's window is built with the Mamba tokenizer; the prompt Mamba then scores must
    not be left-truncated (the budget is respected, so every window record is actually read)."""
    from agentmamba.models.answerer import MambaAnswerer
    from agentmamba.pipeline.baseline import build_window
    from tests.conftest import make_event

    hist = [make_event(f"E{i}", etype="maintenance_log", stage="MAINTENANCE", ts=f"2026-05-0{1 + i % 9}T10:00:00",
                       text=f"equipment=K1-PMP-0{i % 5}; action=replaced seal number {i}") for i in range(60)]
    win = build_window("What action was recorded?", [], hist, mamba.count_tokens, 256, reserve_tokens=64)
    assert 0 < len(win.records) < 60 and win.prompt_tokens <= 256 - 64
    ans = MambaAnswerer(mamba, max_prompt_records=None, max_tokens=256)
    from agentmamba.models.answerer import Candidate

    r = ans.rank("What action was recorded?", win.records, [Candidate("replaced seal", win.records[0], "action")])
    from agentmamba.models.answerer import build_prompt

    assert r.model_call["cache_hit"] is False          # fresh cache: Mamba really ran
    assert r.model_call["prompt_tokens"] == mamba.count_tokens(build_prompt("What action was recorded?", win.records))
    assert len(r.prompt_record_ids) == len(win.records)
