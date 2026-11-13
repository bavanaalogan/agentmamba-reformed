"""Initial-answer assessment from the permitted initial context ONLY (no memory access).

Produces explicit, documented features; the controller turns them into a decision. Mamba's
score margin is one feature among several - the decision never rests on model confidence alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

FEATURES = ("has_anchor", "n_candidates_log", "any_supported", "top_supported", "currency_unverified",
            "causal_question", "causal_documented", "hypothesis_only", "conflict", "mamba_margin",
            "temporal_constraint", "intent_confidence")


@dataclass
class Assessment:
    features: dict
    missing_anchor: bool
    ambiguous: bool
    reasons: list[str] = field(default_factory=list)
    evidence_status: str | None = None

    def vector(self) -> list[float]:
        return [float(self.features[k]) for k in FEATURES]


def assess(parsed, ranked: list, verification) -> Assessment:
    spec = parsed.spec
    anchors = parsed.anchors()
    top = ranked[0] if ranked else None
    top_supported = bool(top is not None and verification.chosen is not None
                         and verification.chosen.record.event_id == top.record.event_id
                         and verification.chosen.value == top.value)
    margin = 0.0
    if len(ranked) >= 2 and ranked[0].score is not None and ranked[1].score is not None:
        margin = max(0.0, min(5.0, ranked[0].score - ranked[1].score))
    f = {
        "has_anchor": 1.0 if anchors else 0.0,
        "n_candidates_log": math.log1p(len(ranked)),
        "any_supported": 1.0 if verification.status in ("supported",) else 0.0,
        "top_supported": 1.0 if top_supported else 0.0,
        "currency_unverified": 1.0 if (spec and spec.constraint == "latest_version") else 0.0,
        "causal_question": 1.0 if (spec and spec.causal) or parsed.causal_language else 0.0,
        "causal_documented": 1.0 if verification.causal_claim_status == "documented_by_source" else 0.0,
        "hypothesis_only": 1.0 if verification.causal_claim_status == "hypothesis_only" else 0.0,
        "conflict": 1.0 if verification.status == "conflict" else 0.0,
        "mamba_margin": margin,
        "temporal_constraint": 1.0 if (spec and spec.constraint) else 0.0,
        "intent_confidence": parsed.intent_confidence,
    }
    reasons = []
    if not ranked:
        reasons.append("no candidate answer in the initial context")
    elif verification.status == "insufficient":
        reasons.append(verification.abstention_reason or "initial candidates are not supported")
    elif verification.status == "partial":
        reasons.append("initial context provides only partially verified evidence")
    if f["currency_unverified"]:
        reasons.append("currency of the document revision cannot be verified without memory")
    if f["hypothesis_only"]:
        reasons.append("causal question: initial context offers only an association, no documented cause")
    if f["conflict"]:
        reasons.append("conflicting values in the initial context")
    return Assessment(f, missing_anchor=spec is not None and not anchors,
                      ambiguous=parsed.intent == "ambiguous" or spec is None, reasons=reasons,
                      evidence_status=verification.status)
