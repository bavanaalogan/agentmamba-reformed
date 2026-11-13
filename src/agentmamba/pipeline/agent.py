"""AgentMamba: Mamba-first conditional RAG over persistent plant memory.

Question -> parse -> initial answer from the permitted initial context ONLY (Mamba ranking)
-> assessment -> controller (RETURN_INITIAL | RETRIEVE | ABSTAIN_OR_CLARIFY)
-> [retrieval from SQLite+FAISS+relationships] -> verification -> grounded answer / abstention.

Research decision support only: answers never authorize production or safety actions.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from agentmamba.config import config_hash, resolve_path
from agentmamba.memory.relationship_store import InMemoryRelationshipGraph, SQLiteRelationshipGraph, UnionGraph
from agentmamba.memory.schema import Event
from agentmamba.models.answerer import extract_candidates
from agentmamba.pipeline.answering import compose, initial_status
from agentmamba.retrieval.answerability import assess
from agentmamba.retrieval.controller import CLARIFY, RETRIEVE, RETURN, Controller, Decision
from agentmamba.retrieval.query_processor import IntentClassifier, QueryProcessor, clarification_reason
from agentmamba.retrieval.retriever import Retriever
from agentmamba.verification.document_evidence import (candidate_share, n_distinct, share_note,
                                                       question_parts, supported_record_ids, text_candidates, verify_text)
from agentmamba.verification.evidence_checker import EvidenceVerifier, VerificationResult
from agentmamba.verification.provenance import citations_valid

MODES = {"conditional", "always", "never"}
DISCLAIMER = "Research decision support; not a safety-certified system. Verify before acting."


@dataclass
class QueryRequest:
    question: str
    initial_context: list[Event] = field(default_factory=list)
    mode: str = "conditional"
    filters: dict | None = None
    verification: bool = True
    query_id: str | None = None
    top_k: int | None = None


class AgentMamba:
    def __init__(self, cfg: dict, memory, answerer, query_processor: QueryProcessor, controller: Controller,
                 verifier: EvidenceVerifier | None = None, versions: dict | None = None):
        self.cfg = cfg
        self.memory = memory
        self.answerer = answerer
        self.qp = query_processor
        self.controller = controller
        self.verifier = verifier or EvidenceVerifier()
        self.retriever = Retriever(memory, cfg)
        self.time_window_min = float(cfg.get("verification", {}).get("time_window_minutes", 30))
        self.versions = {"config": config_hash(cfg), **(versions or {})}

    @staticmethod
    def artifact_paths(cfg: dict) -> dict[str, Path]:
        root = resolve_path(cfg, "paths.artifacts_dir")
        return {"intent": root / "controller" / "intent.pkl", "controller": root / "controller" / "controller.pkl"}

    def query(self, req: QueryRequest) -> dict:
        if req.mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        qid = req.query_id or uuid.uuid4().hex[:12]
        T = {}
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        calls: list[dict] = []
        parsed = self.qp.parse(req.question)
        ctx = list(req.initial_context)
        ctx_ids = {e.event_id: e for e in ctx}
        _resolve_from_context(parsed, ctx)
        spec = parsed.spec
        ambiguity = clarification_reason(parsed)
        if ambiguity:
            T["parse_s"] = time.perf_counter() - t0
            decision = Decision(CLARIFY, None, None, [ambiguity])
            out = {
                "query_id": qid, "question": req.question, "mode": req.mode, "parsed": parsed.to_dict(),
                "initial_model": None, "initial_answer": None, "initial_candidates": [], "initial_status": "unavailable",
                "assessment": None, "controller": decision.to_dict(), "retrieval_triggered": False,
                "retrieval_reason": None, "retrieved_record_ids": [], "retrieval": [],
                "relationships_traversed": [], "verification_enabled": req.verification,
                "initial_context_ids": list(ctx_ids),
                "sufficiency": _sufficiency(decision, req.mode, spec is not None, spec is None and bool(parsed.anchor_map())),
                "evidence": [], "verification_status": "insufficient", "causal_claim_status": "not_claimed",
                "answer_status": "needs_clarification", "answer_value": None, "abstention_reason": ambiguity,
                "final_answer": f"Please clarify: {ambiguity}.",
            }
            return self._finish(out, T, t0, cpu0, calls, None, None)
        # No trusted typed intent, but the question names identifiers: answer only from
        # text records that verifiably mention them (never ask for an identifier already given).
        doc_mode = spec is None and bool(parsed.anchor_map())
        T["parse_s"] = time.perf_counter() - t0
        # ---- initial answer: initial context only, no memory access
        t = time.perf_counter()
        cands0 = extract_candidates(spec, ctx) if spec else []
        rank0 = self.answerer.rank(req.question, ctx, cands0)
        if cands0:
            calls.append(_call("initial", rank0))
        T["initial_answer_s"] = time.perf_counter() - t
        t = time.perf_counter()
        a0 = None
        trank0 = None
        if spec:
            ver0 = self.verifier.verify(parsed, rank0.candidates, ctx_ids, InMemoryRelationshipGraph(ctx), store=None,
                                        enabled=req.verification)
            ver0, trank0 = self._with_text(parsed, ver0, req.question, ctx, None, req.verification, calls, "initial_text")
            a0 = assess(parsed, rank0.candidates, ver0)
            decision = self.controller.decide(a0, req.mode)
        elif doc_mode:
            ver0, trank0 = self._with_text(parsed, None, req.question, ctx, None, req.verification, calls,
                                           "initial_text")
            decision = _document_decision(ver0, req.mode)
        else:
            ver0 = None
            decision = self.controller.decide(_ambiguous(), req.mode)
        T["controller_s"] = time.perf_counter() - t
        if trank0 is not None:
            T["initial_answer_s"] += trank0.seconds
            T["controller_s"] -= trank0.seconds
        shown0 = trank0 or rank0
        final_rank = trank0 or rank0
        out = {
            "query_id": qid, "question": req.question, "mode": req.mode, "parsed": parsed.to_dict(),
            "initial_model": shown0.model, "initial_answer": shown0.candidates[0].value if shown0.candidates else None,
            "initial_candidates": [c.to_dict() for c in shown0.candidates[:10]],
            "initial_status": initial_status(shown0.candidates, ver0) if ver0 else "unavailable",
            "assessment": a0.features if a0 else None, "controller": decision.to_dict(),
            "retrieval_triggered": False, "retrieval_reason": None, "retrieved_record_ids": [], "retrieval": [],
            "relationships_traversed": [], "verification_enabled": req.verification,
            "initial_context_ids": list(ctx_ids),
            "sufficiency": _sufficiency(decision, req.mode, spec is not None, doc_mode),
        }
        if decision.action == CLARIFY:
            kinds = ", ".join(spec.anchor_kinds) if spec else "asset / heat / ladle / coil / work order"
            out.update({"evidence": [], "verification_status": "insufficient", "causal_claim_status": "not_claimed",
                        "answer_status": "needs_clarification", "answer_value": None,
                        "abstention_reason": decision.reasons[0],
                        "final_answer": f"Please specify which {kinds} you mean (identifier and, if relevant, date)."})
            return self._finish(out, T, t0, cpu0, calls, None, None)
        ver = ver0
        if decision.action == RETRIEVE:
            t = time.perf_counter()
            res = self.retriever.retrieve(parsed, req.question, ctx, req.filters, top_k=req.top_k)
            T["retrieval_s"] = time.perf_counter() - t
            out["retrieval_triggered"] = True
            out["retrieval_reason"] = "; ".join(decision.reasons)
            out["retrieved_record_ids"] = [it.event.event_id for it in res.items]
            out["retrieval"] = [it.to_dict() for it in res.items]
            out["retrieval_stats"] = {"n_vector_hits": res.n_vector_hits, "n_expansion_hits": res.n_expansion_hits,
                                      "n_text_hits": res.n_text_hits,
                                      "n_removed_by_date_time_filter": res.n_filtered_by_time,
                                      "n_selected": len(res.items), "duplicates_dropped": len(res.duplicates_dropped),
                                      "channels": list(self.retriever.channels)}
            out["relationships_traversed"] = res.relationships_traversed[:100]
            pool = ctx + [it.event for it in res.items if it.event.event_id not in ctx_ids]
            pool_ids = {e.event_id: e for e in pool}
            t = time.perf_counter()
            cands1 = extract_candidates(spec, pool) if spec else []
            rank1 = self.answerer.rank(req.question, pool, cands1)
            if cands1:
                calls.append(_call("final", rank1))
            final_rank = rank1
            T["final_answer_s"] = time.perf_counter() - t
            t = time.perf_counter()
            ver = None
            if spec:
                graph = UnionGraph(SQLiteRelationshipGraph(self.memory.store), InMemoryRelationshipGraph(ctx))
                ver = self.verifier.verify(parsed, rank1.candidates, pool_ids, graph, store=self.memory.store,
                                           enabled=req.verification)
            ver, trank = self._with_text(parsed, ver, req.question, pool, self.memory.store, req.verification, calls,
                                         "final_text")
            T["verification_s"] = time.perf_counter() - t
            if trank is not None:
                final_rank = trank
                T["final_answer_s"] += trank.seconds
                T["verification_s"] -= trank.seconds
            out["final_candidates"] = [c.to_dict() for c in final_rank.candidates[:10]]
        else:
            out["retrieval_reason"] = None
            out["no_retrieval_reason"] = "; ".join(decision.reasons)
        out.update(compose(parsed, ver, req.verification))
        out.update({"evidence": ver.evidence, "verification_status": ver.status,
                    "causal_claim_status": ver.causal_claim_status, "verification": ver.to_dict()})
        resolve = lambda i: ctx_ids.get(i) or self.memory.store.get_event(i)  # noqa: E731
        out["invalid_citations"] = citations_valid(ver.evidence, resolve)
        return self._finish(out, T, t0, cpu0, calls, ver, final_rank)

    def _with_text(self, parsed, ver, question, records, store, enabled, calls, label):
        """Use untyped records when typed evidence is insufficient, and decompose supported
        multi-part text questions into independently checked clauses. Returns (result, rank)."""
        compound = len(question_parts(parsed)) > 1
        if ver is not None and ver.status in {"supported", "conflict", "unverified"} and not compound:
            return ver, None
        cands = text_candidates(parsed, records)
        if enabled and not compound:   # compound parts are checked independently below
            ok = supported_record_ids(parsed, records, store)
            cands = [c for c in cands if c.record.event_id in ok]
        rank = self.answerer.rank(question, records, cands) if cands else None
        if rank is not None:
            calls.append(_call(label, rank))
        doc = verify_text(parsed, rank.candidates if rank else [], records, store, enabled, self.time_window_min)
        if doc is not None and (doc.status != "insufficient" or ver is None):
            return doc, rank
        if doc is not None and ver is not None and doc.abstention_reason:
            ver.abstention_reason = f"{ver.abstention_reason}; {doc.abstention_reason}"
        return (ver if ver is not None else VerificationResult(
            "insufficient", None, "not_claimed", abstention_reason="no verifiable record mentions the identifiers")), None

    def _finish(self, out: dict, T: dict, t0: float, cpu0: float, calls: list[dict], ver, rank) -> dict:
        T["total_s"] = time.perf_counter() - t0
        out["timings"] = {k: round(v, 4) for k, v in T.items()}
        out["model_calls"] = calls
        out["resources"] = resource_usage(cpu0)
        out["confidence"] = confidence_block(self.answerer, rank, ver, out)
        out["versions"] = self.versions
        out["disclaimer"] = DISCLAIMER
        return out


def _call(stage: str, rank) -> dict:
    return {"stage": stage, "model": rank.model, "seconds": round(rank.seconds, 4),
            "n_candidates": len(rank.candidates), "prompt_records": len(rank.prompt_record_ids),
            "prompt_chars": rank.prompt_chars, **rank.model_call}


def resource_usage(cpu0: float) -> dict:
    """Process CPU seconds used by this query and resident memory after it (psutil, if present)."""
    out = {"cpu_s": round(time.process_time() - cpu0, 3)}
    try:
        import psutil

        out["rss_mb"] = round(psutil.Process().memory_info().rss / 2 ** 20, 1)
    except ImportError:
        out["rss_mb"] = None
    return out


def confidence_block(answerer, rank, ver, out: dict) -> dict:
    """Confidence-related numbers, kept SEPARATE because they are not interchangeable:
    - candidate_share: Mamba's softmax share of the answer value among the extracted candidate
      values (model-derived, uncalibrated). None for non-model answerers.
    - evidence_support: what verification established (categorical, not a probability).
    - retrieval_max_similarity: cosine similarity of the best retrieved record (retrieval quality,
      NOT answer confidence).
    - calibrated: filled in by the service only if a calibrator fitted on the development split exists."""
    value = out.get("answer_value")
    share = None
    if getattr(answerer, "name", "") == "mamba" and rank is not None and value is not None:
        share = candidate_share(rank.candidates, value)
    sims = [r["similarity"] for r in out.get("retrieval", [])]
    return {
        "candidate_share": None if share is None else round(share, 4),
        "candidate_share_definition": ("softmax over Mamba mean token log-probs of distinct candidate values; "
                                       "uncalibrated, model-derived") if share is not None else
                                      "not available (no Mamba-ranked answer)",
        "evidence_support": None if ver is None else {"status": ver.status, "n_supporting": ver.n_supported,
                                                      "causal_claim_status": ver.causal_claim_status},
        "n_distinct_candidates": n_distinct(rank.candidates) if rank is not None else 0,
        "candidate_share_note": share_note(n_distinct(rank.candidates)) if share is not None else None,
        "retrieval_max_similarity": round(max(sims), 4) if sims else None,
        "calibrated": None,
    }


def _resolve_from_context(parsed, ctx: list[Event]) -> None:
    """A typed question that names no identifier of the required kind can be resolved from the
    initial context only if that context contains exactly ONE identifier of that kind;
    with several candidates the question stays ambiguous (clarification)."""
    spec = parsed.spec
    if spec is None or parsed.anchor_map():
        return
    found = {i: k for e in ctx for i, k in e.entity_ids().items() if k in spec.anchor_kinds}
    if len(found) == 1:
        (ident, kind), = found.items()
        parsed.ids.setdefault(kind, []).append(ident)
        parsed.resolved_from_context[ident] = kind


def _sufficiency(decision: Decision, mode: str, typed: bool, doc_mode: bool) -> dict:
    """Plain statement of the initial-context sufficiency decision and how it was made. Both
    methods reuse computations already done for the initial answer (no extra model call)."""
    label = {RETURN: "sufficient", RETRIEVE: "insufficient", CLARIFY: "ambiguous / missing identifier"}[decision.action]
    if mode != "conditional":
        method = f"bypassed: mode={mode}"
    elif decision.action == RETURN and any("verified evidence" in reason for reason in decision.reasons):
        method = "rule: evidence verification found complete support in the initial context"
    elif typed:
        method = ("trained controller: logistic regression over initial-assessment features "
                  "(candidates, verification outcome, Mamba score margin, ...) vs a validation-tuned threshold")
    elif doc_mode:
        method = ("rule: sufficient only if verification of the initial context alone (identity, date/time, "
                  "term coverage, hedging, conflicts) yields a supported answer")
    else:
        method = "no typed intent and no identifier: clarification requested"
    return {"decision": label, "retrieval_needed": decision.action == RETRIEVE, "method": method,
            "reason": "; ".join(decision.reasons), "p_sufficient": decision.p_sufficient,
            "threshold": decision.threshold}


def _document_decision(ver, mode: str) -> Decision:
    """Documented rule for document-grounded questions (the trained controller covers typed
    intents only): return when the initial context already holds a verified passage."""
    if mode == "always":
        return Decision(RETRIEVE, None, None, ["mode=always_retrieve"])
    if mode == "never" or ver.status == "supported":
        return Decision(RETURN, None, None, ["mode=no_retrieval"] if mode == "never"
                        else ["initial context contains verified evidence for the answer"])
    return Decision(RETRIEVE, None, None, ["no verified evidence in the initial context; "
                                           + (ver.abstention_reason or "")])


def _ambiguous():
    from agentmamba.retrieval.answerability import Assessment

    return Assessment({}, missing_anchor=True, ambiguous=True, reasons=["question is ambiguous"])


def load_query_processor(cfg: dict) -> QueryProcessor:
    return QueryProcessor(IntentClassifier.load(AgentMamba.artifact_paths(cfg)["intent"]))
