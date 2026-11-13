"""Baseline Mamba: the frozen backbone answering from its allowed context only.

Information access (fixed in advance, identical for every question):
* The context is the engineer-supplied initial-context records, then the most recently ARRIVED
  records of the ingested history (newest first), as whole records, until the prompt reaches
  ``token_budget`` tokens of the Mamba tokenizer (minus ``reserve_tokens`` for the question
  template and candidate continuations). This is question-independent: no similarity search,
  no identifier lookup, no relationship expansion, no memory gate.
* Mamba-130m has no positional context limit; ``token_budget`` is a declared CPU-cost policy.
  Whatever the baseline cannot see is therefore a consequence of THIS policy, and the experiment
  reports, per question, whether the required evidence was inside the window.

Answer mechanism (the same one AgentMamba uses, so the comparison isolates memory, retrieval
and verification): typed candidates if the question has a trusted AnswerSpec and the context
carries them, otherwise ``text_candidates``; Mamba ranks them by mean token log-probability
after a prompt containing the context records and the question; the top candidate is the answer.
There is no verification and no abstention mechanism: with no candidate at all, the baseline
gives no answer (reported as ``no_candidate``, not as a verified abstention).

A post-hoc audit runs the verifier on the baseline's answer against the baseline's own context
for DISPLAY ONLY; it never changes the baseline's answer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from agentmamba.memory.relationship_store import InMemoryRelationshipGraph
from agentmamba.memory.schema import Event
from agentmamba.models.answerer import Candidate, build_prompt, extract_candidates, format_record
from agentmamba.pipeline.agent import resource_usage
from agentmamba.verification.document_evidence import (EVENT_QUESTION_RE, _event_details, candidate_share,
                                                       n_distinct, question_parts, share_note, text_candidates,
                                                       verify_text)
from agentmamba.verification.evidence_checker import EvidenceVerifier
from agentmamba.verification.provenance import classify, evidence_entry

BASELINE_VERSION = "baseline-mamba-v1"


@dataclass
class ContextWindow:
    records: list[Event]
    prompt_tokens: int
    token_budget: int
    n_history: int                 # history records offered (newest first)
    n_history_included: int
    initial_excluded: list[str]    # initial-context records that did not fit

    def to_dict(self) -> dict:
        return {"record_ids": [e.event_id for e in self.records], "n_records": len(self.records),
                "prompt_tokens": self.prompt_tokens, "token_budget": self.token_budget,
                "n_history_offered": self.n_history, "n_history_included": self.n_history_included,
                "initial_excluded": self.initial_excluded}


def build_window(question: str, initial: list[Event], history: list[Event], count_tokens, token_budget: int,
                 reserve_tokens: int = 128) -> ContextWindow:
    """Initial-context records first, then ``history`` from its END (most recent arrival) backwards,
    as whole records, while the prompt stays within ``token_budget - reserve_tokens``. The window is
    contiguous: it stops at the first history record that does not fit."""
    limit = token_budget - reserve_tokens
    used = count_tokens(build_prompt(question, []))
    chosen, seen, excluded = [], set(), []
    for ev in initial:
        t = count_tokens(format_record(ev)) + 1
        if used + t <= limit:
            chosen.append(ev)
            seen.add(ev.event_id)
            used += t
        else:
            excluded.append(ev.event_id)
    n_hist = 0
    for ev in reversed(history):
        if ev.event_id in seen:
            continue
        t = count_tokens(format_record(ev)) + 1
        if used + t > limit:
            break
        chosen.append(ev)
        seen.add(ev.event_id)
        used += t
        n_hist += 1
    return ContextWindow(chosen, used, token_budget, len(history), n_hist, excluded)


class BaselineMamba:
    name = "baseline_mamba"

    def __init__(self, query_processor, answerer, count_tokens, token_budget: int = 768, reserve_tokens: int = 128,
                 time_window_min: float = 30.0):
        self.qp = query_processor
        self.answerer = answerer
        self.count_tokens = count_tokens
        self.token_budget = token_budget
        self.reserve_tokens = reserve_tokens
        self.time_window_min = time_window_min

    def config(self) -> dict:
        return {"version": BASELINE_VERSION, "token_budget": self.token_budget, "reserve_tokens": self.reserve_tokens,
                "answerer": getattr(self.answerer, "name", "?"),
                "model": getattr(getattr(self.answerer, "backbone", None), "model_id", None),
                "context_policy": "initial context + most recent arrivals (whole records) within token budget",
                "retrieval": False, "verification": False}

    def window(self, question: str, initial: list[Event], history: list[Event]) -> ContextWindow:
        return build_window(question, initial, history, self.count_tokens, self.token_budget, self.reserve_tokens)

    def answer(self, question: str, window: ContextWindow, query_id: str | None = None) -> dict:
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        parsed = self.qp.parse(question)
        ctx = window.records
        spec = parsed.spec
        parts = question_parts(parsed)
        if len(parts) > 1:
            return self._answer_compound(question, parsed, parts, window, query_id, t0, cpu0)
        cands = extract_candidates(spec, ctx) if spec else []
        source = "typed" if cands else "text"
        if not cands:
            cands = text_candidates(parsed, ctx)
        t = time.perf_counter()
        rank = self.answerer.rank(question, ctx, cands)
        gen_s = time.perf_counter() - t
        top = rank.candidates[0] if rank.candidates else None
        out = {
            "system": self.name, "query_id": query_id, "question": question, "parsed": parsed.to_dict(),
            "context": window.to_dict(), "candidate_source": source if top else None,
            "n_candidates": len(rank.candidates), "candidates": [c.to_dict() for c in rank.candidates[:10]],
            "model": rank.model, "model_call": {"prompt_records": len(rank.prompt_record_ids), **rank.model_call},
            "retrieval_triggered": False, "retrieved_record_ids": [], "verification_enabled": False,
        }
        if top is None:
            out.update({"answer_status": "abstained", "answer_value": None, "no_candidate": True,
                        "abstention_reason": "no candidate span in the context (the baseline has no abstention "
                                             "mechanism; this is not a verified abstention)",
                        "final_answer": "No answer: the context contains no candidate span for this question.",
                        "evidence": [], "verification_status": "not_verified", "causal_claim_status": "not_claimed"})
        else:
            out.update({"answer_status": "answered", "answer_value": top.value, "no_candidate": False,
                        "abstention_reason": None,
                        "final_answer": f"{top.value} [Baseline Mamba, unverified; top-ranked span from record "
                                        f"{top.record.event_id}]",
                        "evidence": [evidence_entry(top.record, "answer", classify(top.record, top.field, False),
                                                    top.field)],
                        "verification_status": "not_verified",
                        "causal_claim_status": "claimed_unverified" if parsed.causal_language else "not_claimed"})
        share = candidate_share(rank.candidates) if getattr(self.answerer, "name", "") == "mamba" and top else None
        out["confidence"] = {
            "candidate_share": None if share is None else round(share, 4),
            "candidate_share_definition": ("softmax over Mamba mean token log-probs of distinct candidate values; "
                                           "uncalibrated, model-derived") if share is not None else
                                          "not available (no Mamba-ranked answer)",
            "n_distinct_candidates": n_distinct(rank.candidates),
            "candidate_share_note": share_note(n_distinct(rank.candidates)) if share is not None else None,
            "evidence_support": None, "retrieval_max_similarity": None, "calibrated": None}
        out["audit"] = self.audit(parsed, top, ctx) if top else None
        out["timings"] = {"generation_s": round(gen_s, 4), "total_s": round(time.perf_counter() - t0, 4)}
        out["resources"] = resource_usage(cpu0)
        return out

    def _answer_compound(self, question, parsed, parts, window, query_id, t0, cpu0) -> dict:
        """Rank an initial-context candidate independently for each requested component."""
        answers, all_candidates, evidence, audits, calls, component_shares = [], [], [], [], [], []
        seconds = 0.0
        model = getattr(getattr(self.answerer, "backbone", None), "model_id", "unknown")
        for part in parts:
            candidates = text_candidates(part, window.records)
            if not candidates and part.spec:
                candidates = extract_candidates(part.spec, window.records)
            if EVENT_QUESTION_RE.search(part.text):
                focus = set(part.focus_terms())
                if focus & {"alarm", "alert", "fault", "trip", "incident"}:
                    asks_clearance = bool({"clear", "acknowledg", "resolv"} & focus)
                    summaries = []
                    for candidate in candidates:
                        event_type = candidate.record.event_type.replace("_", " ")
                        state = " ".join((event_type,
                                          str((candidate.record.raw or {}).get("status", "")),
                                          str((candidate.record.raw or {}).get("state", "")))).replace("_", " ")
                        if not asks_clearance and any(word in state.lower()
                                                      for word in ("clear", "acknowledg", "resolv", "cancel")):
                            continue
                        details = _event_details(candidate.record)
                        timestamp = candidate.record.timestamp
                        prefix = f"{timestamp}: " if timestamp else ""
                        value = f"{prefix}{event_type.upper()}"
                        if details:
                            value += f" ({'; '.join(details)})"
                        summaries.append(Candidate(value, candidate.record, "event_summary"))
                    if summaries:
                        candidates = summaries
            if not candidates:
                answers.append((part.text, None))
                audits.append({"question_part": part.text, "status": "no_candidate"})
                continue
            ranked = self.answerer.rank(part.text, window.records, candidates)
            seconds += ranked.seconds
            calls.append(ranked)
            all_candidates.extend(ranked.candidates)
            top = ranked.candidates[0] if ranked.candidates else None
            answers.append((part.text, top))
            share = candidate_share(ranked.candidates, top.value) if top else None
            component_shares.append({"question_part": part.text, "candidate_share": share,
                                     "n_distinct_candidates": n_distinct(ranked.candidates)})
            if top:
                evidence.append(evidence_entry(top.record, "answer", classify(top.record, top.field, False),
                                               top.field))
                audits.append({"question_part": part.text, **self.audit(part, top, window.records)})
            else:
                audits.append({"question_part": part.text, "status": "no_candidate"})

        supported = [(part, candidate) for part, candidate in answers if candidate is not None]
        answer_value = "; ".join(
            f"{part}: {candidate.value}" if candidate else f"{part}: no candidate in initial context"
            for part, candidate in answers)
        answer_components = [
            {"question": part, "answer": candidate.value if candidate else "No candidate in initial context",
             "status": "unverified" if candidate else "insufficient",
             "record_ids": [candidate.record.event_id] if candidate else [],
             "reason": None if candidate else "No candidate span was found in the initial context."}
            for part, candidate in answers]
        primary = supported[0][1] if supported else None
        answered = bool(supported)
        model_call = {
            "prompt_records": len(window.records),
            "component_calls": len(calls),
            "prompt_tokens": sum(int(c.model_call.get("prompt_tokens", 0)) for c in calls),
            "candidate_tokens": sum(int(c.model_call.get("candidate_tokens", 0)) for c in calls),
            "cache_hit": bool(calls) and all(c.model_call.get("cache_hit", False) for c in calls),
        }
        out = {
            "system": self.name, "query_id": query_id, "question": question, "parsed": parsed.to_dict(),
            "context": window.to_dict(), "candidate_source": "text_components" if answered else None,
            "n_candidates": len(all_candidates), "candidates": [c.to_dict() for c in all_candidates[:20]],
            "model": calls[0].model if calls else model, "model_call": model_call,
            "retrieval_triggered": False, "retrieved_record_ids": [], "verification_enabled": False,
            "answer_components": answer_components,
            "answer_status": "answered" if answered else "abstained",
            "answer_value": answer_value if answered else None, "no_candidate": not answered,
            "abstention_reason": None if answered else
                "no candidate spans in the initial context (the baseline has no verification or retrieval)",
            "final_answer": (f"{answer_value} [Baseline Mamba, unverified; ranked independently per question part]"
                             if answered else "No answer: the initial context contains no candidate spans."),
            "evidence": evidence, "verification_status": "not_verified",
            "causal_claim_status": "claimed_unverified" if parsed.causal_language else "not_claimed",
            "confidence": {
                "candidate_share": None,
                "candidate_share_definition": "not aggregated: scores from separate question-part prompts are incomparable",
                "mean_component_candidate_share": (
                    sum(item["candidate_share"] for item in component_shares if item["candidate_share"] is not None)
                    / sum(item["candidate_share"] is not None for item in component_shares)
                    if any(item["candidate_share"] is not None for item in component_shares) else None),
                "component_candidate_shares": component_shares,
                "mean_component_candidate_share_definition": (
                    "mean of per-part Mamba softmax shares; uncalibrated and not a probability of correctness"),
                "n_distinct_candidates": n_distinct(all_candidates),
                "candidate_share_note": None, "evidence_support": None,
                "retrieval_max_similarity": None, "calibrated": None,
            },
            "audit": {"status": "component_audits", "used_for_answer": False, "parts": audits},
            "timings": {"generation_s": round(seconds, 4), "total_s": round(time.perf_counter() - t0, 4)},
            "resources": resource_usage(cpu0),
        }
        return out

    def generate(self, question: str, window: ContextWindow, query_id: str | None = None,
                 max_new_tokens: int = 64) -> dict:
        """Answer by raw Mamba generation from the supplied context, without candidate extraction."""
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        backbone = getattr(self.answerer, "backbone", None)
        if backbone is None or not callable(getattr(backbone, "generate", None)):
            raise RuntimeError("raw Mamba generation requires an answerer backed by Mamba")
        prompt = build_prompt(question, window.records)
        generated = backbone.generate(prompt, max_new_tokens=max_new_tokens,
                                      max_prompt_tokens=self.token_budget).strip()
        call = dict(getattr(backbone, "last_call", {}) or {})
        answered = bool(generated)
        final = generated if answered else "Mamba generated no answer."
        out = {
            "system": self.name, "query_id": query_id, "question": question,
            "context": window.to_dict(), "candidate_source": "raw_generation" if answered else None,
            "n_candidates": 0, "candidates": [], "model": getattr(backbone, "model_id", "mamba"),
            "model_call": {"prompt_records": len(window.records), **call},
            "retrieval_triggered": False, "retrieved_record_ids": [], "verification_enabled": False,
            "generation_mode": "raw_mamba_generation",
            "answer_status": "answered" if answered else "abstained",
            "answer_value": generated or None, "no_candidate": not answered,
            "abstention_reason": None if answered else "Mamba generated an empty response",
            "final_answer": final, "evidence": [], "verification_status": "not_verified",
            "causal_claim_status": "claimed_unverified" if self.qp.parse(question).causal_language else "not_claimed",
            "confidence": {
                "candidate_share": None,
                "candidate_share_definition": "not available for raw free generation",
                "n_distinct_candidates": 0,
                "candidate_share_note": None,
                "evidence_support": None,
                "retrieval_max_similarity": None,
                "calibrated": None,
            },
        }
        out["timings"] = {"generation_s": round(time.perf_counter() - t0, 4)}
        out["resources"] = resource_usage(cpu0)
        out["timings"]["total_s"] = round(time.perf_counter() - t0, 4)
        return out

    def audit(self, parsed, top, ctx: list[Event]) -> dict:
        """Post-hoc, display-only: would the verifier accept this answer from this context?"""
        spec = parsed.spec
        if spec and top.field in (spec.answer_field, spec.hypothesis_field):      # a typed candidate
            ver = EvidenceVerifier().verify(parsed, [top], {e.event_id: e for e in ctx}, InMemoryRelationshipGraph(ctx))
        else:
            ver = verify_text(parsed, [top], ctx, None, True, self.time_window_min)
        if ver is None:
            return {"status": "unverifiable", "reason": "question names no identifier", "used_for_answer": False}
        return {"status": ver.status if ver.chosen is not None or ver.status == "conflict" else "unsupported",
                "reason": ver.abstention_reason, "used_for_answer": False}
