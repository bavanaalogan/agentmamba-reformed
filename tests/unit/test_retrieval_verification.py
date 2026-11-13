"""Query parsing, candidate extraction, retrieval, verification, controller, metrics."""
from __future__ import annotations

import numpy as np
import pytest

from agentmamba.evaluation.evidence_metrics import precision_at_k, prf, recall_at_k, reciprocal_rank
from agentmamba.evaluation.metrics import bootstrap_ci, exact_match, mcnemar_exact, token_f1
from agentmamba.memory.episodic_memory import EpisodicMemory
from agentmamba.memory.relationship_store import InMemoryRelationshipGraph, SQLiteRelationshipGraph, UnionGraph
from agentmamba.models.answerer import FirstCandidateAnswerer, extract_candidates, field_value
from agentmamba.retrieval.answerability import FEATURES, Assessment, assess
from agentmamba.retrieval.controller import CLARIFY, RETRIEVE, RETURN, Controller
from agentmamba.retrieval.query_processor import IntentClassifier, ParsedQuery, extract_ids
from agentmamba.retrieval.retriever import Retriever
from agentmamba.verification.evidence_checker import EvidenceVerifier
from agentmamba.verification.temporal_checker import process_sequence_violations
from tests.conftest import make_event

V = EvidenceVerifier()


def P(intent, text="q", **ids):
    return ParsedQuery(text=text, intent=intent, intent_confidence=1.0, ids=ids.pop("ids", {}), **ids)


def run(parsed, records, store=None, graph=None, enabled=True):
    cands = FirstCandidateAnswerer().rank("q", records, extract_candidates(parsed.spec, records)).candidates
    g = graph or InMemoryRelationshipGraph(records)
    return V.verify(parsed, cands, {r.event_id: r for r in records}, g, store=store, enabled=enabled)


def insp(eid, ladle, ts, finding, th=150, **kw):
    return make_event(eid, ts=ts, ladle_id=ladle, text=f"Ladle {ladle}: {finding}, {th} mm.",
                      observations=[{"name": "finding", "value": finding}, {"name": "lining_thickness", "value": th, "unit": "mm"}], **kw)


# ---------------------------------------------------------------- parsing
def test_id_and_date_extraction():
    ids = extract_ids("Coil C01-CL0004 and heat C01-H0012 on ladle C01-LD03, WO C01-WO002 at C01-CC1, doc C01-SOP-LP")
    assert ids == {"heat": ["C01-H0012"], "ladle": ["C01-LD03"], "coil": ["C01-CL0004"], "work_order": ["C01-WO002"],
                   "document": ["C01-SOP-LP"], "asset": ["C01-CC1"]}


def test_intent_classifier_masks_ids_and_roundtrips(tmp_path):
    clf = IntentClassifier().fit(["Which ladle carried heat C01-H0001?", "Which ladle carried heat C02-H0002?",
                                  "What was the documented outcome of work order C01-WO001?",
                                  "What was the documented outcome of work order C03-WO004?"],
                                 ["ladle_of_heat", "ladle_of_heat", "wo_outcome", "wo_outcome"])
    assert clf.predict("Which ladle carried heat Z9-H9999?")[0] == "ladle_of_heat"
    clf.save(tmp_path / "i.pkl")
    assert IntentClassifier.load(tmp_path / "i.pkl").predict("documented outcome of work order X1-WO777")[0] == "wo_outcome"


# ---------------------------------------------------------------- candidates
def test_typed_candidate_extraction():
    a = make_event("A", "ladle_assignment", heat_id="H1", ladle_id="L1")
    b = make_event("B", "cast_record", heat_id="H1", cast_seq_id="S1")
    p = P("ladle_of_heat", ids={"heat": ["H1"]})
    c = extract_candidates(p.spec, [a, b])
    assert [(x.value, x.record.event_id) for x in c] == [("L1", "A")]
    assert field_value(insp("I", "L", "2024-01-01T00:00:00", "ok"), "lining_thickness") == "150 mm"


# ---------------------------------------------------------------- verification cases
def test_work_order_on_a_ladle_is_answerable():
    # regression: ladle IDs parse as kind "ladle"; work orders on ladles must still be found
    wo = make_event("W", "work_order", "MAINTENANCE", ts="2024-03-06T08:00:00", asset_id="C1-LD06", ladle_id="C1-LD06",
                    work_order_id="C1-WO006")
    p = P("wo_opened_on", ids=extract_ids("Which work order was opened on C1-LD06 on 2024-03-06?"), dates=["2024-03-06"])
    assert p.anchors() == ["C1-LD06"]
    r = run(p, [wo])
    assert r.status == "supported" and r.chosen.value == "C1-WO006"


def test_supported_when_record_names_anchor():
    r = run(P("ladle_of_heat", ids={"heat": ["H1"]}), [make_event("A", "ladle_assignment", heat_id="H1", ladle_id="L1")])
    assert r.status == "supported" and r.chosen.value == "L1" and r.evidence[0]["record_id"] == "A"


def test_similar_record_on_other_asset_is_rejected():
    # same finding text, different ladle: must not be attributed to L1
    r = run(P("ladle_warning", ids={"ladle": ["L1"]}, dates=["2024-01-05"]),
            [insp("X", "L2", "2024-01-02T00:00:00", "hairline crack at slag line")])
    assert r.status == "insufficient" and r.chosen is None and "linked" in r.abstention_reason


def test_missing_identifier_link_leads_to_abstention():
    broken = make_event("A", "ladle_assignment", heat_id="H1", text="ladle number not recorded")
    other = make_event("B", "ladle_assignment", heat_id="H2", ladle_id="L7")
    r = run(P("ladle_of_heat", ids={"heat": ["H1"]}), [broken, other])
    assert r.chosen is None and r.status == "insufficient"


def test_sibling_heat_via_shared_group_is_not_linked():
    mine = make_event("C1", "cast_record", "CASTING", heat_id="H1", cast_seq_id="S1", slab_id="B1")
    sib = make_event("C2", "cast_record", "CASTING", heat_id="H2", cast_seq_id="S1", slab_id="B2",
                     observations=[{"name": "abnormality", "value": "nozzle clogging"}])
    r = run(P("caster_abnormality", ids={"heat": ["H1"]}), [mine, sib])
    assert r.chosen is None      # H1's own record has no abnormality; H2's must not be borrowed


def test_conflicting_records_reported():
    a = insp("I1", "L1", "2024-01-03T05:41:00", "normal", 171)
    b = insp("I2", "L1", "2024-01-03T05:53:00", "normal", 154)
    r = run(P("ladle_thickness_at", ids={"ladle": ["L1"]}, dates=["2024-01-03"]), [a, b])
    assert r.status == "conflict" and {v["value"] for v in r.conflict["values"]} == {"171 mm", "154 mm"}
    one = run(P("ladle_thickness_at", ids={"ladle": ["L1"]}, dates=["2024-01-04"]), [a, b])
    assert one.status == "insufficient"           # temporal constraint excludes both


def test_outdated_document_superseded(tmp_path, embedder):
    v1 = make_event("D1", "procedure_doc", "MAINTENANCE", doc_id="P", version=1,
                    observations=[{"name": "limit_value", "value": 90, "unit": "min"}])
    v2 = make_event("D2", "procedure_doc", "MAINTENANCE", doc_id="P", version=2, supersedes="D1",
                    observations=[{"name": "limit_value", "value": 60, "unit": "min"}])
    p = P("procedure_current_value", ids={"document": ["P"]})
    init = run(p, [v1])                              # no memory: currency unverifiable
    assert init.status == "partial" and init.currency_verified is False
    mem = EpisodicMemory(tmp_path, embedder)
    mem.write([v1, v2])
    post = run(p, [v1, v2], store=mem.store, graph=SQLiteRelationshipGraph(mem.store))
    assert post.status == "supported" and post.chosen.value == "60 min" and post.superseded[0]["record_id"] == "D1"


def test_causal_discipline_hypothesis_vs_documented():
    coil = make_event("Q", "coil_inspection", "QUALITY", coil_id="K1")
    roll = make_event("R", "rolling_record", "ROLLING", slab_id="B1", coil_id="K1", ts="2024-01-01T12:00:00")
    cast = make_event("C", "cast_record", "CASTING", heat_id="H1", slab_id="B1", ts="2024-01-01T08:00:00",
                      observations=[{"name": "abnormality", "value": "mould level fluctuation"}])
    p = P("coil_defect_cause", ids={"coil": ["K1"]}, causal_language=True)
    hyp = run(p, [coil, roll, cast])
    assert hyp.causal_claim_status == "hypothesis_only" and hyp.status == "partial"
    assert hyp.evidence[0]["evidence_type"] == "inferred"
    unconfirmed = make_event("U", "rca_report", "QUALITY", coil_id="K1",
                             observations=[{"name": "rca_status", "value": "inconclusive"},
                                           {"name": "cause", "value": "mould level fluctuation"}])
    assert run(p, [coil, roll, cast, unconfirmed]).causal_claim_status == "hypothesis_only"
    rca = make_event("V", "rca_report", "QUALITY", coil_id="K1",
                     observations=[{"name": "rca_status", "value": "confirmed"}, {"name": "cause", "value": "worn rollers"}])
    doc = run(p, [coil, roll, cast, rca])
    assert doc.causal_claim_status == "documented_by_source" and doc.chosen.value == "worn rollers"
    off = run(p, [coil, roll, cast], enabled=False)
    assert off.causal_claim_status == "claimed_unverified" and off.status == "unverified"


def test_latest_inspection_before_linking_record():
    roll = make_event("R", "rolling_record", "ROLLING", slab_id="B1", coil_id="K1", ts="2024-01-02T00:00:00")
    cast = make_event("C", "cast_record", "CASTING", heat_id="H1", slab_id="B1", ts="2024-01-01T12:00:00")
    asg = make_event("A", "ladle_assignment", heat_id="H1", ladle_id="L1", ts="2024-01-01T10:00:00")
    olds = [insp("I1", "L1", "2024-01-01T06:00:00", "erosion"), insp("I2", "L1", "2024-01-01T09:00:00", "normal"),
            insp("I3", "L1", "2024-01-01T11:00:00", "after-the-fact")]
    r = run(P("upstream_ladle_condition", ids={"coil": ["K1"]}), [roll, cast, asg] + olds)
    assert r.status == "supported" and r.chosen.record.event_id == "I2"
    assert {e["record_id"] for e in r.evidence} == {"I2", "R", "C", "A"}


def test_process_sequence_violation_detected():
    cast = make_event("C", "cast_record", "CASTING", heat_id="H1", slab_id="B1", ts="2024-01-02T00:00:00")
    roll = make_event("R", "rolling_record", "ROLLING", slab_id="B1", coil_id="K1", ts="2024-01-01T00:00:00")
    assert process_sequence_violations([cast, roll])


# ---------------------------------------------------------------- retrieval
def test_retriever_vector_filters_expansion_and_supersession(tmp_path, embedder, cfg):
    mem = EpisodicMemory(tmp_path, embedder)
    evs = [make_event("A", "ladle_assignment", text="Heat H1 tapped into ladle L1.", heat_id="H1", ladle_id="L1"),
           make_event("A2", "ladle_assignment", text="Heat H2 tapped into ladle L2.", heat_id="H2", ladle_id="L2"),
           make_event("N", "shift_log", "STEELMAKING", text="routine shift no remarks", asset_id="BOF1"),
           make_event("D1", "procedure_doc", "MAINTENANCE", text="Procedure P rev 1", doc_id="P", version=1),
           make_event("D2", "procedure_doc", "MAINTENANCE", text="Procedure P rev 2 supersedes rev 1", doc_id="P",
                      version=2, supersedes="D1")]
    mem.write(evs)
    r = Retriever(mem, cfg)
    p = P("ladle_of_heat", text="Which ladle carried heat H1?", ids={"heat": ["H1"]})
    res = r.retrieve(p, p.text)
    ids = [it.event.event_id for it in res.items]
    assert "A" in ids and ids.index("A") == 0
    a_item = next(it for it in res.items if it.event.event_id == "A")
    assert "relationship" in a_item.sources and a_item.explanation
    only = r.retrieve(None, "routine shift", filters={"stage": ["STEELMAKING"]})
    assert [it.event.event_id for it in only.items] == ["N"]
    sup = r.retrieve(None, "Procedure P rev 1", filters={"event_type": ["procedure_doc"]}, top_k=1)
    got = [it.event.event_id for it in sup.items]
    assert got[0] == "D2" or {"D1", "D2"} <= set(got)    # a superseded hit always brings its newer version
    empty = Retriever(EpisodicMemory(tmp_path / "empty", embedder), cfg).retrieve(p, p.text)
    assert empty.items == []


# ---------------------------------------------------------------- controller
def _assessment(evidence_status=None, **over):
    f = {k: 0.0 for k in FEATURES}
    f.update(over)
    return Assessment(f, missing_anchor=False, ambiguous=False, reasons=["test"],
                      evidence_status=evidence_status)


def test_controller_modes_and_learned_decision(tmp_path):
    rng = np.random.default_rng(0)
    X = np.array([_assessment(any_supported=s, top_supported=s, n_candidates_log=s).vector() for s in rng.integers(0, 2, 60)])
    y = X[:, FEATURES.index("any_supported")].astype(int)
    ctl = Controller().fit(X, y)
    ctl.tune_threshold(X, y)
    assert ctl.decide(_assessment(any_supported=1, top_supported=1, n_candidates_log=1)).action == RETURN
    assert ctl.decide(_assessment()).action == RETRIEVE
    assert ctl.decide(_assessment("insufficient", any_supported=1, top_supported=1,
                                 n_candidates_log=1)).action == RETRIEVE
    assert ctl.decide(_assessment("partial", any_supported=1, top_supported=1,
                                 n_candidates_log=1)).action == RETRIEVE
    no_probability_model = type("LowProbability", (), {
        "predict_proba": lambda self, values: np.array([[0.99, 0.01]]),
    })()
    evidence_first = Controller(model=no_probability_model)
    supported = evidence_first.decide(_assessment("supported"))
    assert supported.action == RETURN and supported.p_sufficient is None
    assert ctl.decide(_assessment(), "always").action == RETRIEVE
    assert ctl.decide(_assessment(), "never").action == RETURN
    amb = Assessment(_assessment().features, missing_anchor=True, ambiguous=False)
    assert ctl.decide(amb).action == CLARIFY
    ctl.save(tmp_path / "c.pkl")
    assert Controller.load(tmp_path / "c.pkl").threshold == ctl.threshold
    with pytest.raises(FileNotFoundError):
        Controller.load(tmp_path / "none.pkl")
    with pytest.raises(RuntimeError):
        Controller().decide(_assessment())


def test_assess_features_reflect_evidence():
    p = P("ladle_of_heat", ids={"heat": ["H1"]})
    recs = [make_event("A", "ladle_assignment", heat_id="H1", ladle_id="L1")]
    cands = FirstCandidateAnswerer().rank("q", recs, extract_candidates(p.spec, recs)).candidates
    ver = V.verify(p, cands, {r.event_id: r for r in recs}, InMemoryRelationshipGraph(recs))
    a = assess(p, cands, ver)
    assert a.features["any_supported"] == 1.0 and a.features["top_supported"] == 1.0 and not a.missing_anchor
    a2 = assess(P("ladle_of_heat"), [], V.verify(P("ladle_of_heat"), [], {}, UnionGraph()))
    assert a2.missing_anchor


# ---------------------------------------------------------------- metrics
def test_metrics():
    assert exact_match("161 MM", "161 mm") == 1.0 and exact_match(None, "x") == 0.0
    assert token_f1("slag line crack", "hairline crack at slag line") == pytest.approx(2 * 1 * 0.6 / 1.6)
    assert recall_at_k(["a", "b", "c"], ["c", "d"]) == 0.5 and recall_at_k(["a"], []) is None
    assert precision_at_k(["a", "b"], ["b"]) == 0.5 and reciprocal_rank(["x", "b"], ["b"]) == 0.5
    assert prf(["a", "b"], ["b"]) == (0.5, 1.0, pytest.approx(2 / 3))
    assert mcnemar_exact([True] * 10, [False] * 10)["p_value"] < 0.01
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
    lo, hi = bootstrap_ci([1.0] * 50 + [0.0] * 50)
    assert lo < 0.5 < hi
