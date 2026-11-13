"""Config, schema, normalization, loaders, linking, generator and split integrity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentmamba.config import PROJECT_ROOT, load_config
from agentmamba.data import document_loader as dl
from agentmamba.data.dataset_splits import validate_dataset
from agentmamba.data.event_linker import link_event, link_events
from agentmamba.data.event_normalizer import normalize_identifier, normalize_timestamp, normalize_unit
from agentmamba.data.evidence_annotations import AnnotationAccessError, load_annotations
from agentmamba.data.synthetic_generator import GenConfig, generate
from agentmamba.memory.episodic_memory import EpisodicMemory
from agentmamba.memory.schema import Event, Observation, Relationship, stable_id
from agentmamba.models.embedding_model import HashingEmbedder
from tests.conftest import SMALL, make_event


# ---------------------------------------------------------------- config
def test_config_loads_and_overrides():
    cfg = load_config(overrides={"memory.budget_fraction": 0.05})
    assert cfg["memory"]["budget_fraction"] == 0.05
    assert cfg["models"]["mamba_id"] == "state-spaces/mamba-130m-hf"


@pytest.mark.parametrize("key,val", [("memory.budget_fraction", 0), ("memory.budget_fraction", 1.5),
                                     ("memory.policy", "magic"), ("retrieval.top_k", 0), ("gate.model", "xgb")])
def test_config_validation_rejects(key, val):
    with pytest.raises(ValueError):
        load_config(overrides={key: val})


# ---------------------------------------------------------------- schema
def test_event_validation_and_roundtrip():
    ev = make_event("E1", observations=[{"name": "lining_thickness", "value": 140, "unit": "mm", "raw": "140"}],
                    ladle_id="L1", severity="info")
    again = Event.from_dict(json.loads(json.dumps(ev.to_dict())))
    assert again == ev and again.content_hash == ev.content_hash
    assert ev.obs("lining_thickness").unit == "mm"
    with pytest.raises(ValueError):
        make_event("E2", stage="MOON")
    with pytest.raises(ValueError):
        make_event("E3", severity="apocalyptic")
    with pytest.raises(ValueError):
        Event(event_id="E4", stage="LADLE", event_type="x", timestamp=None, text="t", source_ref="s", source_type="leaked")


def test_stable_ids_and_hash_change_with_content():
    assert stable_id("X", 1, "a") == stable_id("X", 1, "a") != stable_id("X", 1, "b")
    a, b = make_event("E1", text="alpha"), make_event("E1", text="beta")
    assert a.content_hash != b.content_hash
    assert a.size_bytes() == len("alpha") + len("[]")


def test_relationship_rejects_self_loop_and_bad_status():
    with pytest.raises(ValueError):
        Relationship("A", "heat", "A", "heat", "x", "documented")
    with pytest.raises(ValueError):
        Relationship("A", "heat", "B", "ladle", "x", "rumour")


# ---------------------------------------------------------------- normalization
def test_timestamp_normalization_formats_and_unknowns():
    assert normalize_timestamp("05.03.2024 14:20") == "2024-03-05T14:20:00"
    assert normalize_timestamp("22/02/2018 02:15") == "2018-02-22T02:15:00"   # day-first source
    assert normalize_timestamp("2024-03-05") == "2024-03-05T00:00:00"
    assert normalize_timestamp("2024-03-05T10:00:00+02:00").endswith("+02:00")  # offset preserved
    assert normalize_timestamp("yesterday") is None and normalize_timestamp(None) is None
    assert normalize_timestamp("12/10/2026 08:14:08,250 +05:30") == "2026-10-12T08:14:08+05:30"
    assert normalize_timestamp("12 Oct 2026 08:19:13") == "2026-10-12T08:19:13"


def test_units_are_spelled_not_converted():
    assert normalize_unit("°C") == "degC" and normalize_unit("furlongs") == "furlongs"
    assert normalize_identifier(" ld-03 ") == "LD-03" and normalize_identifier("N/A") is None


# ---------------------------------------------------------------- loaders
def test_csv_loader_preserves_raw_and_units(tmp_path):
    p = tmp_path / "recs.csv"
    p.write_text("id,ts,heat,temp\nR1,05.03.2024 14:20,h-1,1650\nR2,bad-date,,1600\n", encoding="utf-8")
    evs = dl.load_records(p, {"event_id": "id", "timestamp": "ts", "heat_id": "heat"}, stage="STEELMAKING",
                          event_type="bof_heat_record", observation_units={"temp": "°C"},
                          text_template="Heat {heat} temp {temp}")
    assert [e.event_id for e in evs] == ["R1", "R2"]
    assert evs[0].timestamp == "2024-03-05T14:20:00" and evs[0].timestamp_raw == "05.03.2024 14:20"
    assert evs[0].heat_id == "H-1" and evs[0].raw["heat"] == "h-1"
    o = evs[0].obs("temp")
    assert (o.value, o.unit, o.raw) == (1650, "degC", "1650")
    assert evs[1].timestamp is None and evs[1].heat_id is None and evs[0].timezone == "unknown"


def test_text_logs_accept_common_timestamp_forms_and_keep_original_lines(tmp_path):
    p = tmp_path / "alarm-journal.txt"
    lines = [
        "12/10/2026 08:14:08,250 +05:30 | WARNING | PLC-1 | TEST-PUMP-1 | "
        "ALARM-RAISED | alarm_id=ALM-1; status=ACTIVE",
        "12 Oct 2026 08:19:13 +05:30 | INFO | ALARM-MANAGER | TEST-PUMP-1 | "
        "ALARM-STATUS | alarm_id=ALM-1; status=CLEARED",
        "2026-10-12T08:24:00Z | INFO | CMMS | TEST-PUMP-1 | "
        "WORK-ORDER-OPENED | action=Inspect coupling",
        "2026-10-12 08:25:00 +05:30 | MAJOR | MAINTENANCE-LOG | TEST-PUMP-1 | "
        "INSPECTION-FINDING | finding=seal_worn",
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    events = dl.load_any(p, stage="UTILITIES")
    assert len(events) == 4
    assert [e.timestamp for e in events] == [
        "2026-10-12T08:14:08+05:30", "2026-10-12T08:19:13+05:30", "2026-10-12T08:24:00+00:00",
        "2026-10-12T08:25:00+05:30",
    ]
    assert [e.event_type for e in events] == ["alarm_raised", "alarm_status", "work_order_opened", "inspection_finding"]
    assert [e.severity for e in events] == ["major", "info", "info", "major"]
    assert [e.text for e in events] == lines
    assert events[0].raw["alarm_id"] == events[1].raw["alarm_id"] == "ALM-1"


def test_jsonl_event_loader_and_text_chunking(tmp_path):
    ev = make_event("E1")
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps(ev.to_dict()) + "\n", encoding="utf-8")
    assert dl.load_any(p)[0] == ev
    t = tmp_path / "SOP-1.txt"
    t.write_text("Para one. " * 120 + "\n\nPara two.", encoding="utf-8")
    a, b = dl.load_text_document(t, max_chars=300), dl.load_text_document(t, max_chars=300)
    assert [x.event_id for x in a] == [x.event_id for x in b]          # deterministic
    assert all(len(x.text) <= 300 for x in a) and a[-1].text == "Para two."
    assert a[0].doc_id == "SOP-1" and a[0].source_ref.endswith("#chunk=0")
    with pytest.raises(ValueError):
        dl.load_any(tmp_path / "x.pdf")


RAW = PROJECT_ROOT / "data" / "raw"


@pytest.mark.skipif(not (RAW / "uci_851_steel_energy" / "Steel_industry_data.csv").exists(), reason="UCI 851 not downloaded")
def test_uci_851_loader_daily_windows(tmp_path):
    src = (RAW / "uci_851_steel_energy" / "Steel_industry_data.csv").read_text(encoding="utf-8").splitlines()
    p = tmp_path / "Steel_industry_data.csv"
    p.write_text("\n".join(src[:200]) + "\n", encoding="utf-8")       # ~2 days of 15-min rows
    evs = dl.load_uci_851(p)
    assert len(evs) >= 2 and all(e.source_type == "public" and e.timezone == "unknown" for e in evs)
    assert evs[0].obs("total_usage").unit == "kWh" and "rows=" in evs[0].source_ref


@pytest.mark.skipif(not (RAW / "uci_198_steel_plates" / "Faults.NNA").exists(), reason="UCI 198 not downloaded")
def test_uci_198_loader():
    evs = dl.load_uci_198(RAW / "uci_198_steel_plates" / "Faults.NNA", RAW / "uci_198_steel_plates" / "Faults27x7_var")
    assert len(evs) == 1941 and evs[0].timestamp is None and evs[0].source_type == "public"
    assert link_events(evs[:50]) == []      # no identifiers -> no relationships invented


# ---------------------------------------------------------------- linking
def test_linker_creates_documented_links_only_from_identifiers():
    a = make_event("A", "ladle_assignment", heat_id="H1", ladle_id="L1")
    rels = link_event(a)
    carried = [r for r in rels if r.rel_type == "carried_by"]
    assert len(carried) == 1 and carried[0].status == "documented" and carried[0].source_event_id == "A"
    # same timestamp, no shared identifiers -> no relationship
    b, c = make_event("B", heat_id="H2"), make_event("C", ladle_id="L9")
    assert [r for r in link_events([b, c]) if r.rel_type != "mentions"] == []
    s = make_event("S", "procedure_doc", doc_id="D", supersedes="OLD", version=2)
    assert any(r.rel_type == "supersedes" and r.dst_id == "OLD" for r in link_event(s))


# ---------------------------------------------------------------- generator & splits
def test_raw_alarm_identifier_reaches_all_linked_records_without_schema_changes(tmp_path):
    mem = EpisodicMemory(tmp_path / "memory", HashingEmbedder())
    first = make_event("AL1", "alarm_status", stage="UTILITIES", raw={"alarm_id": "alm-17"})
    second = make_event("AL2", "alarm_clearance", stage="UTILITIES", raw={"alarm_id": "ALM-17"})
    mem.write([first, second])
    assert {e.event_id for e in mem.store.events_mentioning(["ALM-17"])} == {"AL1", "AL2"}
    assert mem.validate()["ok"]


def test_generator_is_deterministic():
    a = generate(SMALL)["builders"][0]
    b = generate(SMALL)["builders"][0]
    assert [e.content_hash for e in a.events] == [e.content_hash for e in b.events]
    other = generate(GenConfig(**{**SMALL.__dict__, "seed": 99}))["builders"][0]
    assert [e.content_hash for e in a.events] != [e.content_hash for e in other.events]


def test_dataset_valid_no_leakage(dataset, annotations):
    allann = {k: v for s in annotations.values() for k, v in s.items()}
    assert validate_dataset(dataset, allann) == []
    assert all(e.source_type == "synthetic" for e in dataset.events)
    assert dataset.manifest["synthetic"] is True
    assert set(dataset.manifest["campaign_splits"].values()) == {"train", "val", "test"}


def test_every_question_has_annotation_fields(dataset, annotations):
    allann = {k: v for s in annotations.values() for k, v in s.items()}
    assert set(allann) == {q.question_id for q in dataset.questions}
    fams = {a.family for a in allann.values()}
    assert {"F01_direct_context", "F09_conflicting_records", "F14_ambiguous", "F15_correlation_not_causation"} <= fams
    for a in allann.values():
        assert a.answer_kind in ("value", "abstain", "clarify", "conflict", "hypothesis")


def test_annotation_access_is_split_scoped(dataset_dir, dataset):
    qs = {q.question_id: q.split for q in dataset.questions}
    with pytest.raises(AnnotationAccessError):
        load_annotations(dataset_dir, "test", "training", qs)
    with pytest.raises(AnnotationAccessError):
        load_annotations(dataset_dir, "test", "tuning", qs)
    tr = load_annotations(dataset_dir, "train", "training", qs)
    assert all(qs[k] == "train" for k in tr)


def test_live_pipeline_never_imports_annotations():
    src = PROJECT_ROOT / "src" / "agentmamba"
    for sub in ("pipeline", "retrieval", "verification", "models", "memory", "api"):
        for f in (src / sub).glob("*.py"):
            assert "evidence_annotations" not in f.read_text(encoding="utf-8"), f
