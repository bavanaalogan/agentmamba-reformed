"""Compose user-facing answers from a verification result. Wording keeps observed facts,
documented statements and hypotheses visibly separate."""
from __future__ import annotations


def initial_status(ranked: list, ver) -> str:
    if not ranked:
        return "unavailable"
    if ver.status == "supported":
        return "supported"
    if ver.status in ("partial", "conflict", "unverified"):
        return "partial"
    return "unsupported"


def compose(parsed, ver, verification_enabled: bool) -> dict:
    """Returns answer_status, answer_value, final_answer, abstention_reason."""
    c = ver.chosen
    components = getattr(ver, "answer_components", [])
    if ver.status == "conflict":
        vals = "; ".join(f"{v['value']} ({', '.join(v['record_ids'])})" for v in ver.conflict["values"])
        return {"answer_status": "abstained", "answer_value": None, "abstention_reason": ver.abstention_reason,
                "answer_components": components,
                "final_answer": f"Conflicting records, not reconciled: {vals}. Please confirm against the source "
                                f"documents before acting."}
    if c is None:
        return {"answer_status": "abstained", "answer_value": None,
                "answer_components": components,
                "abstention_reason": ver.abstention_reason or "insufficient evidence",
                "final_answer": f"Insufficient evidence to answer: {ver.abstention_reason or 'no supported record'}."}
    cite = f"record {c.record.event_id} ({c.record.timestamp or 'time unknown'}, {c.record.stage})"
    if not verification_enabled:
        return {"answer_status": "answered", "answer_value": c.value, "abstention_reason": None,
                "answer_components": components,
                "final_answer": f"{c.value} [unverified; from {cite}]"}
    if ver.causal_claim_status == "hypothesis_only":
        return {"answer_status": "answered", "answer_value": c.value, "abstention_reason": None,
                "answer_components": components,
                "final_answer": (f"No source-documented root cause was found. Associated observation (hypothesis "
                                 f"for investigation, not an established cause): '{c.value}' from {cite}.")}
    prefix = "Documented root cause" if ver.causal_claim_status == "documented_by_source" else "Answer"
    note = ""
    if ver.status == "partial":
        if ver.currency_verified is False:
            note = " Note: revision currency not verified against memory."
        elif ver.sequence_violations:
            note = " Note: process-sequence timestamps look inconsistent; check the linked records."
    return {"answer_status": "answered", "answer_value": c.value, "abstention_reason": None,
            "answer_components": components,
            "final_answer": f"{prefix}: {c.value} (source: {cite}).{note}"}
