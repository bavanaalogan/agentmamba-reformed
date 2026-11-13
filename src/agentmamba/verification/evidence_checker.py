"""Evidence verification: decides which (if any) ranked candidate is supported.

For each candidate, in Mamba rank order:
  1. identity   - source record linked to the question anchor by documented relationships
  2. temporal   - the spec's temporal constraint holds
  3. currency   - record not superseded (needs the memory store; unverifiable without it)
  4. causal     - for causal questions, only a source record that itself states a confirmed
                  cause can support a causal answer; otherwise the best linked association is
                  returned as ``hypothesis_only``
Then version selection, the latest-before selection rule, and conflict detection.

``enabled=False`` (ablation) skips all checks and returns the top-ranked candidate as is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agentmamba.memory.schema import Event
from agentmamba.retrieval.query_processor import ParsedQuery
from agentmamba.verification.conflict_resolution import detect_conflict, drop_superseded
from agentmamba.verification.provenance import classify, evidence_entry
from agentmamba.verification.relationship_checker import link_candidate
from agentmamba.verification.temporal_checker import check_constraint, process_sequence_violations


@dataclass
class VerificationResult:
    status: str                       # supported | partial | conflict | insufficient | unverified
    chosen: object | None             # Candidate
    causal_claim_status: str          # not_claimed | hypothesis_only | documented_by_source | claimed_unverified
    evidence: list[dict] = field(default_factory=list)
    candidate_checks: list[dict] = field(default_factory=list)
    conflict: dict | None = None
    superseded: list[dict] = field(default_factory=list)
    sequence_violations: list[str] = field(default_factory=list)
    currency_verified: bool | None = None
    abstention_reason: str | None = None
    relationships_used: list[dict] = field(default_factory=list)
    n_supported: int = 0
    answer_components: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "chosen"}
        d["chosen"] = self.chosen.to_dict() if self.chosen is not None else None
        return d


def _closest(passing: list) -> list:
    if not passing:
        return passing
    d = min(p[1].depth for p in passing)
    return [p for p in passing if p[1].depth == d]


class EvidenceVerifier:
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def verify(self, parsed: ParsedQuery, ranked: list, records: dict[str, Event], graph, store=None,
               enabled: bool = True) -> VerificationResult:
        spec = parsed.spec
        if not enabled:
            return self._unverified(parsed, ranked)
        anchors = parsed.anchor_map() if spec else {}
        if not anchors:
            return VerificationResult("insufficient", None, "not_claimed",
                                      abstention_reason="question names no identifier of the required kind")

        def fetch(eid):
            return records.get(eid) or (store.get_event(eid) if store is not None else None)

        def superseded_by(eid):
            for r in records.values():
                if r.supersedes == eid:
                    return r.event_id
            return store.superseded_by(eid) if store is not None else None

        checks, passing = [], []
        for c in ranked:
            link = link_candidate(anchors, c.record, graph, self.max_depth)
            link_recs = [fetch(i) for i in link.link_record_ids()]
            # reference time = the record documenting the LAST edge (the one that reached the
            # candidate's entity, e.g. the heat->ladle assignment)
            ref = link_recs[-1].timestamp if link_recs and link_recs[-1] is not None else None
            t_ok, t_note = check_constraint(spec.constraint, c.record, parsed.dates, ref)
            confirmed = True
            if spec.causal and not c.hypothesis and spec.confirm_field:
                o = c.record.obs(spec.confirm_field[0])
                confirmed = o is not None and str(o.value) == spec.confirm_field[1]
            ok = link.linked and t_ok is True and confirmed and None not in link_recs
            checks.append({**c.to_dict(), "linked": link.linked, "link_depth": link.depth,
                           "link_records": link.link_record_ids(), "temporal_ok": t_ok, "temporal_note": t_note,
                           "confirmed_by_source": confirmed if spec.causal else None, "passed": ok})
            if ok:
                passing.append((c, link, link_recs))

        # Most-specific association wins: keep only candidates at the minimal documented link
        # distance (a record naming the heat beats one that merely shares its ladle).
        primary = _closest([p for p in passing if not p[0].hypothesis])
        hyp = _closest([p for p in passing if p[0].hypothesis])
        # currency / supersession (latest_version questions)
        currency = None
        superseded_notes: list[dict] = []
        if spec.constraint == "latest_version":
            kept, superseded_notes = drop_superseded([p[0] for p in primary], superseded_by)
            primary = [p for p in primary if p[0] in kept]
            currency = store is not None
        if spec.constraint == "latest_before_link" and primary:
            latest = max(p[0].record.timestamp for p in primary)
            primary = [p for p in primary if p[0].record.timestamp == latest]

        res = VerificationResult("insufficient", None, "not_claimed", candidate_checks=checks,
                                 superseded=superseded_notes, currency_verified=currency,
                                 n_supported=len(primary))
        if primary:
            conflict = detect_conflict([p[0] for p in primary], spec.single_valued)
            if conflict:
                res.status, res.conflict = "conflict", conflict
                res.abstention_reason = "records disagree and no supersession or version resolves the conflict"
                seen = set()
                for p in primary:
                    if p[0].record.event_id not in seen:
                        seen.add(p[0].record.event_id)
                        res.evidence.append(evidence_entry(p[0].record, "conflicting", classify(
                            p[0].record, spec.answer_field, False), spec.answer_field))
                return res
            c, link, link_recs = primary[0]          # highest Mamba rank among supported
            res.chosen = c
            res.status = "supported" if currency is not False else "partial"
            if spec.causal:
                res.causal_claim_status = "documented_by_source"
            self._fill_evidence(res, c, link, link_recs, spec.answer_field)
            if currency is False:
                res.abstention_reason = None
            return res
        if spec.causal and hyp:
            c, link, link_recs = hyp[0]
            res.chosen, res.status, res.causal_claim_status = c, "partial", "hypothesis_only"
            self._fill_evidence(res, c, link, link_recs, spec.hypothesis_field, hypothesis=True)
            for r in records.values():   # surface inconclusive RCA records for the same anchor
                if r.event_type == "rca_report" and set(r.entity_ids()) & set(anchors):
                    res.evidence.append(evidence_entry(r, "context", "documented"))
            return res
        if not ranked:
            res.abstention_reason = (f"no available record carries '{spec.answer_field}' for "
                                     f"{', '.join(anchors)} (never recorded, not retained, or not retrieved)")
        elif not any(ch["linked"] for ch in checks):
            res.abstention_reason = (f"records with '{spec.answer_field}' exist but none is linked to "
                                     f"{', '.join(anchors)} by a documented identifier")
        elif spec.causal:
            res.abstention_reason = "no source-documented cause and no linked association"
        else:
            res.abstention_reason = "linked records fail the temporal/version constraint"
        return res

    def _fill_evidence(self, res, c, link, link_recs, answer_field, hypothesis=False):
        res.evidence.append(evidence_entry(c.record, "answer", classify(c.record, answer_field, hypothesis), answer_field))
        for r in link_recs:
            if r is not None and r.event_id != c.record.event_id:
                res.evidence.append(evidence_entry(r, "link", "documented"))
        res.relationships_used = [{"rel_id": r.rel_id, "src": r.src_id, "rel_type": r.rel_type, "dst": r.dst_id,
                                   "status": r.status, "source_event_id": r.source_event_id} for r in link.path]
        res.sequence_violations = process_sequence_violations([c.record] + [r for r in link_recs if r is not None])
        if res.sequence_violations and res.status == "supported":
            res.status = "partial"

    def _unverified(self, parsed: ParsedQuery, ranked: list) -> VerificationResult:
        spec = parsed.spec
        if not ranked:
            return VerificationResult("insufficient", None, "not_claimed", abstention_reason="no candidate answer")
        primary = [c for c in ranked if not c.hypothesis] or ranked
        c = primary[0]
        causal = "claimed_unverified" if (spec and spec.causal) else "not_claimed"
        return VerificationResult("unverified", c, causal,
                                  evidence=[evidence_entry(c.record, "answer", classify(c.record, c.field, False), c.field)])
