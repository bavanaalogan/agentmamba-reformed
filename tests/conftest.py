"""Shared fixtures. Fast tests use deterministic fixtures (HashingEmbedder, the non-model
first-candidate answerer). Tests that load real weights are marked ``@pytest.mark.model``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import load_config  # noqa: E402
from agentmamba.data.dataset_loader import load_dataset  # noqa: E402
from agentmamba.data.evidence_annotations import load_annotations  # noqa: E402
from agentmamba.data.synthetic_generator import GenConfig, write_dataset  # noqa: E402
from agentmamba.memory.schema import Event, Observation  # noqa: E402
from agentmamba.models.answerer import FirstCandidateAnswerer  # noqa: E402
from agentmamba.models.embedding_model import HashingEmbedder  # noqa: E402

SMALL = GenConfig(n_campaigns=5, heats_per_campaign=24, ladles_per_campaign=4, noise_logs_per_heat=2,
                  splits={"train": 3, "val": 1, "test": 1}, seed=7)


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    write_dataset(SMALL, d)
    return d


@pytest.fixture(scope="session")
def dataset(dataset_dir):
    return load_dataset(dataset_dir)


@pytest.fixture(scope="session")
def annotations(dataset_dir, dataset):
    qs = {q.question_id: q.split for q in dataset.questions}
    return {s: load_annotations(dataset_dir, s, "evaluation", qs) for s in ("train", "val", "test")}


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"paths.artifacts_dir": str(tmp_path / "artifacts"),
                                  "paths.storage_dir": str(tmp_path / "storage"),
                                  "paths.results_dir": str(tmp_path / "results"),
                                  "paths.feature_cache": str(tmp_path / "features"),
                                  "gate.artifact_dir": str(tmp_path / "artifacts" / "gate"),
                                  "models.answerer": "first_candidate"})


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture(scope="session")
def trained_qp_controller(dataset, annotations):
    from agentmamba.pipeline.builder import train_controller, train_intent
    from agentmamba.retrieval.query_processor import QueryProcessor

    clf = train_intent(dataset.questions_for("train"), {k: a.question_type for k, a in annotations["train"].items()})
    qp = QueryProcessor(clf)
    ctl = train_controller(qp, FirstCandidateAnswerer(), dataset.questions_for("train"), annotations["train"],
                           dataset.questions_for("val"), annotations["val"], dataset.event_index())
    return qp, ctl


def make_event(eid, etype="ladle_inspection", stage="LADLE", ts="2024-01-01T10:00:00", text=None, **kw):
    obs = kw.pop("observations", [])
    return Event(event_id=eid, stage=stage, event_type=etype, timestamp=ts, text=text or f"record {eid}",
                 source_ref=f"test://{eid}", source_type="synthetic",
                 observations=[o if isinstance(o, Observation) else Observation(**o) for o in obs], **kw)
