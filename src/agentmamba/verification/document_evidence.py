"""Text-grounded candidates and verification for *untyped* records.

Untyped records are records whose event type no ``AnswerSpec`` models: uploaded document chunks
and operational log rows (historian / alarm / CMMS exports loaded by ``load_log_records``), plus
free-text logs. The typed verifier links candidates through structured identifiers and typed
fields, so it cannot use them. This module provides, for such records:

* ``text_candidates`` - answer candidates, shared by every system that answers from text
  (Baseline Mamba and AgentMamba use the SAME extractor; only verification differs):
    - log rows: values of fields / observations whose NAME matches a question content term
      (e.g. "maintenance action" -> column ``action``; "vibration reading" -> observation
      ``vibration`` with its recorded unit);
    - free text: one quotation per passage that contains a question content term.
* ``verify_text`` - explicit, conservative checks on Mamba-ranked candidates:
    1. Identity  - the record names a question identifier (structured field, or whole token in text).
    2. Date      - the record's timestamp date or a date stated in the passage equals the question
                   date; or another chunk of the SAME document states both identifier and date.
                   A record dated differently fails.
    3. Time      - if the question states a clock time, timestamped records must lie within
                   ``time_window_min`` minutes, and only the nearest such records are kept.
    4. Coverage  - every question content term occurs in the record (text, field names, type).
                   Sentences ending in "?" are not evidence.
    5. Qualified - for causal or clearance/approval questions, a statement that is hedged or
                   negated ("suspected", "under investigation", "awaiting", "not cleared", ...)
                   does not support an answer; the system abstains and quotes it.
    6. Conflict  - distinct values of the same field from distinct records at the same time
                   -> ``conflict`` (no answer). Passages from several documents -> ``partial``.
  The chosen answer is the highest-ranked passing candidate; nothing is paraphrased.
"""
from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, datetime

from agentmamba.memory.schema import Event
from agentmamba.models.answerer import Candidate, normalize_answer
from agentmamba.retrieval.query_processor import SPECS, ParsedQuery, stem
from agentmamba.verification.evidence_checker import VerificationResult
from agentmamba.verification.provenance import classify, evidence_entry

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
_ISO = re.compile(r"(?<![\d-])(\d{4})-(\d{2})-(\d{2})(?![\d-])")
_DMY = re.compile(r"\b(\d{1,2})\s+(" + "|".join(m[:3] for m in MONTHS) + r")[a-z]*\.?\s+(\d{4})", re.I)
_MDY = re.compile(r"\b(" + "|".join(m[:3] for m in MONTHS) + r")[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})", re.I)
_NUM = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
TIME_QUESTION_RE = re.compile(r"\bwhen\b|\bwhat\s+time\b|\bat\s+what\s+time\b", re.I)
EVENT_QUESTION_RE = re.compile(
    r"\bwhat\s+(?:(?:alarm|alert|event|incident|fault|trip)\s+)?"
    r"(?:happened|occurred|(?:was|were)\s+(?:reported|raised|triggered|recorded))\b", re.I)
QUESTION_START_RE = re.compile(
    r"^\s*(?:what|which|who|whom|whose|when|where|why|how|is|are|was|were|do|does|did|can|could|"
    r"has|have|had|will|would)\b", re.I)
TYPED_TYPES = frozenset(t for s in SPECS.values() for t in s.event_types + s.hypothesis_types)
# Statements that do not establish a cause / clearance (checked only for such questions).
QUALIFIED_RE = re.compile(
    r"\b(not\s+(?:yet\s+)?(?:been\s+)?(?:determined|confirmed|established|identified|cleared|released|approved|"
    r"completed|verified)|under\s+investigation|pending|awaiting|suspected|possibl[ey]|probabl[ey]|likely|"
    r"may\s+(?:have|be)|might|unconfirmed|hypothes\w*|to\s+be\s+determined|tbd|unknown|inconclusive|"
    r"no\s+root\s+cause|not\s+safe)\b", re.I)
CLAIM_STEMS = {"clear", "safe", "releas", "release", "approv", "approve", "return", "certif", "authoriz", "authorize"}
ACTION_FIELDS = {"action", "description", "work_note", "activity", "task", "outcome", "maintenance_action", "note"}
ACTION_STATUS_FIELDS = {"status", "action_status", "completion", "outcome"}
COMPLETED_STATES = {"complet", "done", "finish", "clos", "execut", "perform", "verifi"}
LOG_METADATA_FIELDS = {
    "timestamp", "ts", "time", "datetime", "date_time", "event_time", "time_stamp", "date",
    "log_level", "level", "severity", "log_source", "source", "equipment", "equipment_id", "asset", "asset_id",
}
PENDING_ACTION_RE = re.compile(
    r"\b(?:planned|pending|scheduled|awaiting|not[ _]+(?:done|complete\w*|finished|performed|executed|recorded))\b",
    re.I)
PAST_ACTION_RE = re.compile(r"\b[a-z]{3,}(?:ed|en)\b", re.I)
QUESTION_PART_RE = re.compile(
    r";\s*|\n+|,\s*(?=(?:when|what|which|who|where|why|how)\b)|"
    r"\s+(?:and|also|additionally|furthermore)\s+(?=(?:when|what|which|who|where|why|how)\b)", re.I)
TERM_EQUIVALENTS = (
    {"alarm", "alert"},
    {"ack", "acknowledg"},
    {"vibration", "vib"},
    {"pressure", "press"},
    {"temperature", "temp"},
    {"inspection", "inspect"},
    {"observation", "observ"},
)


def dates_in(text: str) -> set[str]:
    """ISO dates stated in text (ISO, '20 October 2026', 'October 20, 2026', '20.10.2026' day-first)."""
    out = set()
    for y, m, d in _ISO.findall(text):
        out.add(_iso(int(y), int(m), int(d)))
    for d, mon, y in _DMY.findall(text):
        out.add(_iso(int(y), _month(mon), int(d)))
    for mon, d, y in _MDY.findall(text):
        out.add(_iso(int(y), _month(mon), int(d)))
    for d, m, y in _NUM.findall(text):
        out.add(_iso(int(y), int(m), int(d)))
    out.discard("")
    return out


def _month(s: str) -> int:
    return [m[:3] for m in MONTHS].index(s[:3].lower()) + 1


def _iso(y: int, m: int, d: int) -> str:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return ""


def _clean(text: str) -> str:
    """Drop Markdown emphasis/heading/quote markup without altering the words."""
    text = re.sub(r"[*`]+", "", text)          # '_' is kept: identifiers may contain it
    text = re.sub(r"^[ \t]*[#>]+[ \t]*", "", text, flags=re.M)
    return re.sub(r"[ \t]+", " ", text).strip()


def declarative_sentences(text: str) -> list[str]:
    sents = [s.strip() for s in _SENT.split(_clean(text)) if s.strip()]
    return [s for s in sents if not s.rstrip(")").rstrip().endswith("?")]


def _mentions(text: str, ident: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9-])" + re.escape(ident) + r"(?![A-Za-z0-9-])", text, re.I) is not None


def _stems(text: str) -> set[str]:
    return {stem(w) for w in re.findall(r"[a-zA-Z]+", text.lower())}


# ---------------------------------------------------------------- record views
def is_untyped(ev: Event) -> bool:
    return ev.event_type not in TYPED_TYPES or is_structured_log(ev)


def is_structured_log(ev: Event) -> bool:
    return (isinstance(ev.raw, dict) and "chunk_index" not in ev.raw and ev.event_type != "document_chunk"
            and bool({"timestamp", "ts", "time", "datetime", "date_time", "event_time", "time_stamp", "date",
                      "event_type", "record_type", "log_type", "category", "type", "log_level", "equipment",
                      "equipment_id", "machine", "device", "severity", "alarm_id"} & {
                          str(key).strip().lower().replace("-", "_").replace(" ", "_") for key in ev.raw
                      }))


def log_fields(ev: Event, exclude: set[str] | None = None) -> list[tuple[str, str]]:
    """(field name, value) pairs of a log row: observations (with recorded unit), then the
    remaining raw columns except the timestamp and ``exclude`` (identifier values that must not be
    offered as answers; default: every identifier on the record)."""
    def shown(o):                       # the value as recorded at source, with its recorded unit
        v = o.raw if o.raw not in (None, "") else o.value
        return f"{v} {o.unit}" if o.unit else str(v)

    out = [(o.name, shown(o)) for o in ev.observations]
    ids = {i.upper() for i in ev.entity_ids()} if exclude is None else exclude
    obs_raw = {str(o.raw) for o in ev.observations if o.raw is not None}
    for c, v in (ev.raw or {}).items():
        sv = str(v).strip() if v is not None else ""
        if not sv or sv.upper() in ids or sv == ev.timestamp_raw or sv in obs_raw:
            continue
        if any(c == o.name for o in ev.observations):
            continue
        out.append((str(c), sv))
    return out


def _body(ev: Event) -> tuple[str, list[str]]:
    sents = declarative_sentences(ev.text)
    return " ".join(sents), sents


def _record_stems(ev: Event, body: str) -> set[str]:
    if is_structured_log(ev):
        fields = [(name, value) for name, value in log_fields(ev, set())
                  if name.lower() not in LOG_METADATA_FIELDS]
        return _stems(ev.event_type.replace("_", " ")) | {
            token for name, value in fields
            for token in (_stems(name.replace("_", " ")) | _stems(value))
        }
    return _stems(body) | _stems(ev.event_type.replace("_", " "))


# ---------------------------------------------------------------- candidates (shared)
def _restates_question(value: str, question: str) -> bool:
    v = " ".join(re.findall(r"[a-z0-9]+", str(value).replace("_", " ").lower()))
    q = " " + " ".join(re.findall(r"[a-z0-9]+", question.replace("_", " ").lower())) + " "
    return bool(v) and not any(ch.isdigit() for ch in v) and f" {v} " in q


def text_candidates(parsed: ParsedQuery, records: list[Event]) -> list[Candidate]:
    """Answer candidates from untyped records. Used identically by every system."""
    parts = question_parts(parsed)
    if len(parts) > 1:
        combined, seen = [], set()
        for part in parts:
            for candidate in text_candidates(part, records):
                key = (candidate.record.event_id, normalize_answer(candidate.value), candidate.field)
                if key not in seen:
                    seen.add(key)
                    combined.append(candidate)
        return combined
    focus = set(parsed.focus_terms())
    asked = {a.upper() for a in parsed.anchor_map()}      # never answer with the identifier asked about
    broad_event_question = bool(EVENT_QUESTION_RE.search(parsed.text))
    asks_when = bool(TIME_QUESTION_RE.search(parsed.text))
    out, seen = [], set()
    for ev in records:
        if not is_untyped(ev):
            continue
        if asks_when and parsed.anchor_map():
            check = _record_check(ev, parsed, parsed.anchor_map(), parsed.focus_terms(), records, None)
            if not check["passed"]:
                continue
        if is_structured_log(ev):
            fields = log_fields(ev, asked)
            start = len(out)
            for name, val in fields:
                generic_type = name.lower() in {"event_type", "record_type", "log_type", "category", "type"}
                if name.lower() in LOG_METADATA_FIELDS and not (
                        broad_event_question and not (set(focus) & {"alarm", "alert", "fault", "trip", "incident"})
                        and generic_type):
                    continue
                name_stems = _stems(name.replace("_", " "))
                value_stems = _stems(val)
                relevant = any(_term_is_covered(term, name_stems | value_stems) for term in focus)
                relevant |= bool(focus & {"action", "maintenance"} and name.lower() in ACTION_FIELDS)
                named_event = bool(set(focus) & {"alarm", "alert", "fault", "trip", "incident"})
                relevant |= broad_event_question and not named_event and name.lower() in {
                    "event", "event_name", "alarm", "event_type"}
                if asks_when:
                    relevant = False
                # A value that only restates the question's own words (e.g. the signal name
                # "shell_temperature" for "what was the shell temperature reading ...") is the
                # subject of the question, never its answer.
                # (Type fields are exempt: a matching type is what pulls in the record's details below.)
                if relevant and not generic_type and _restates_question(val, parsed.text):
                    relevant = False
                if relevant and (ev.event_id, val) not in seen:
                    seen.add((ev.event_id, val))
                    out.append(Candidate(val, ev, name))
            matching_fields = out[start:]
            has_type_candidate = any(c.field.lower() in {"event_type", "record_type", "log_type", "category", "type"}
                                     for c in matching_fields)
            has_detail_candidate = any(c.field.lower() not in {
                "event_type", "record_type", "log_type", "category", "type", "timestamp",
            } for c in matching_fields)
            if has_type_candidate and not has_detail_candidate and not asks_when and not broad_event_question:
                details = [(name, val) for name, val in fields if _is_record_detail(name)]
                if details:
                    value = "; ".join(f"{name.replace('_', ' ')}={val.replace('_', ' ')}" for name, val in details)
                    if (ev.event_id, value) not in seen:
                        seen.add((ev.event_id, value))
                        out.append(Candidate(value, ev, "event_details"))
            if asks_when and ev.timestamp:
                out.append(Candidate(ev.timestamp, ev, "timestamp"))
            continue
        body, sents = _body(ev)
        if sents and (not focus or focus & _stems(body)):
            val = _answer_text(sents, sorted(focus))
            if (ev.event_id, val) not in seen:
                seen.add((ev.event_id, val))
                out.append(Candidate(val, ev, "document_text"))
    generic_type_fields = {"event_type", "record_type", "log_type", "category", "type"}
    specific_records = {candidate.record.event_id for candidate in out
                        if candidate.field.lower() not in generic_type_fields | {"timestamp"}}
    out = [candidate for candidate in out
           if candidate.field.lower() not in generic_type_fields or candidate.record.event_id not in specific_records]
    return out


# ---------------------------------------------------------------- verification
def _is_record_detail(name: str) -> bool:
    return name.lower() not in {
        "timestamp", "ts", "time", "datetime", "date_time", "event_time", "time_stamp", "date",
        "event_type", "record_type", "log_type", "category", "type", "log_level", "level", "severity",
        "log_source", "source", "equipment", "equipment_id", "asset", "asset_id",
        "work_order", "work_order_id", "wo", "wo_id", "heat", "heat_id", "ladle", "ladle_id",
        "coil", "coil_id", "doc_id", "document_id",
    }


def _ts(ev: Event) -> datetime | None:
    if not ev.timestamp or len(ev.timestamp) < 16:
        return None
    try:
        return datetime.fromisoformat(ev.timestamp[:19])
    except ValueError:
        return None


def _record_check(ev: Event, parsed: ParsedQuery, anchors: dict, focus: list[str], records, store) -> dict:
    body, _ = _body(ev)
    ident = next((a for a in anchors if a.upper() in {i.upper() for i in ev.entity_ids()} or _mentions(body, a)), None)
    stems = _record_stems(ev, body)
    if is_structured_log(ev):
        stems |= {s for name, _ in log_fields(ev) if name.lower() not in LOG_METADATA_FIELDS
                  for s in _stems(name.replace("_", " "))}
    missing = [t for t in focus if not _term_is_covered(t, stems)]
    if "complet" in missing and _completed_action_is_documented(ev):
        missing.remove("complet")
    want = parsed.dates[0] if parsed.dates else None
    date_ok, date_src = True, None
    if want:
        stated = dates_in(body) | ({ev.timestamp[:10]} if ev.timestamp else set())
        if want in stated:
            date_src = ev
        elif stated:
            date_ok = False                      # record is about a different date
        else:
            date_src = _same_document_date(ev, ident, want, records, store)
            date_ok = date_src is not None
    return {"record_id": ev.event_id, "names_identifier": ident, "missing_terms": missing,
            "date_ok": date_ok if want else None, "date_src": date_src,
            "passed": ident is not None and not missing and date_ok}


def _completed_action_is_documented(ev: Event) -> bool:
    if not is_structured_log(ev):
        return False
    type_stems = _stems(ev.event_type.replace("_", " "))
    if not {"maintenance", "action"} <= type_stems:
        return False
    fields = log_fields(ev)
    status_values = [value for name, value in fields if name.lower() in ACTION_STATUS_FIELDS]
    if status_values:
        status_stems = set().union(*(_stems(value) for value in status_values))
        return bool(status_stems & COMPLETED_STATES)
    for name, value in fields:
        if not _is_record_detail(name) or PENDING_ACTION_RE.search(value):
            continue
        if PAST_ACTION_RE.search(value):
            return True
    return False


def _term_is_covered(term: str, stems: set[str]) -> bool:
    if term == "action":
        return bool(stems & {"action", "description", "activity", "task", "outcome", "work_note"})
    if term == "maintenance":
        return bool(stems & {"maintenance", "description", "action", "activity", "task", "outcome", "work_note",
                             "note"})
    for equivalent in TERM_EQUIVALENTS:
        if term in equivalent:
            return bool(stems & equivalent)
    return term in stems


def question_parts(parsed: ParsedQuery) -> list[ParsedQuery]:
    clauses = [c.strip(" ,.?") for c in QUESTION_PART_RE.split(parsed.text) if c and c.strip(" ,.?")]
    questions = [clause for clause in clauses if QUESTION_START_RE.match(clause)]
    if questions:
        clauses = questions
    return [replace(parsed, text=clause) for clause in clauses] if len(clauses) > 1 else [parsed]


def _verify_question_parts(parts: list[ParsedQuery], ranked: list[Candidate], records: list[Event], store, enabled: bool,
                           time_window_min: float) -> VerificationResult:
    answers, evidence, checks, missing, conflicts, components = [], [], [], [], [], []
    chosen_record = None
    n_supported = 0
    unverified = False
    score_by_candidate = {(c.record.event_id, normalize_answer(c.value), c.field): c.score for c in ranked}
    for part in parts:
        candidates = text_candidates(part, records)
        if enabled:
            supported = supported_record_ids(part, records, store)
            candidates = [c for c in candidates if c.record.event_id in supported]
        for candidate in candidates:
            candidate.score = score_by_candidate.get(
                (candidate.record.event_id, normalize_answer(candidate.value), candidate.field))
        if any(c.score is not None for c in candidates):
            candidates.sort(key=lambda c: (c.score is None, -(c.score or 0.0), c.record.timestamp or "",
                                           c.record.event_id))
        if enabled and EVENT_QUESTION_RE.search(part.text):
            event_candidates = [c for c in candidates if c.field.lower() in {
                "event", "event_name", "alarm", "alarm_name", "alarm_id", "alarm_code", "alert", "fault", "trip",
                "incident", "message", "event_type"}]
            specific = [c for c in event_candidates if c.field.lower() != "event_type"]
            if specific:
                event_candidates = specific
            if set(part.focus_terms()) & {"alarm", "alert"}:
                raised = []
                for candidate in event_candidates:
                    state = " ".join((candidate.record.event_type,
                                      str((candidate.record.raw or {}).get("status", "")),
                                      str((candidate.record.raw or {}).get("state", "")))).replace("_", " ")
                    if not re.search(r"\b(?:clear(?:ed|ance)?|acknowledg\w*|resolv\w*|cancel\w*)\b", state, re.I):
                        raised.append(candidate)
                if raised:
                    event_candidates = raised
            result = None
            if event_candidates:
                event_candidates.sort(key=lambda c: (c.record.timestamp or "", c.record.event_id))
                summaries = []
                for candidate in event_candidates:
                    value = candidate.value
                    details = _event_details(candidate.record)
                    if candidate.field.lower() in {"alarm_id", "alarm_code"}:
                        value = f"{candidate.record.event_type.replace('_', ' ')} {value}"
                        details = [detail for detail in details
                                   if not detail.lower().startswith(f"{candidate.field.lower().replace('_', ' ')}=")]
                    timestamp = candidate.record.timestamp
                    prefix = f"{timestamp}: " if timestamp else ""
                    summaries.append(f"{prefix}{value}" + (f" ({', '.join(details)})" if details else ""))
                summary = "; ".join(dict.fromkeys(summaries))
                result = VerificationResult(
                    "supported", Candidate(summary, event_candidates[0].record, "event_summary"),
                    "not_claimed",
                    evidence=[evidence_entry(c.record, "answer", classify(c.record, c.field, False), c.field)
                              for c in event_candidates],
                    n_supported=len(event_candidates))
        else:
            result = verify_text(part, candidates, records, store, enabled, time_window_min)
        if result is not None and result.candidate_checks:
            checks.extend({**check, "question_part": part.text, "part_status": result.status}
                          for check in result.candidate_checks)
        else:
            checks.append({"question_part": part.text,
                           "part_status": result.status if result else "insufficient",
                           "passed": bool(result and result.chosen is not None)})
        if result is not None and result.status == "conflict":
            conflicts.append({"question_part": part.text, **(result.conflict or {})})
            evidence.extend(item for item in result.evidence if item not in evidence)
        if result is None or result.chosen is None or result.status not in {"supported", "partial", "unverified"}:
            missing.append(part.text)
            reason = result.abstention_reason if result else None
            answers.append(f"{part.text}: insufficient evidence" + (f" ({reason})" if reason else ""))
            components.append({"question": part.text, "answer": "Insufficient evidence",
                               "status": result.status if result else "insufficient",
                               "record_ids": [item["record_id"] for item in (result.evidence if result else [])
                                              if item.get("record_id")],
                               "reason": reason})
            continue
        n_supported += 1
        unverified |= result.status == "unverified"
        chosen_record = chosen_record or result.chosen.record
        citation_ids = [e["record_id"] for e in result.evidence if e.get("record_id")]
        answers.append(f"{part.text}: {result.chosen.value} (records: {', '.join(citation_ids)})")
        components.append({"question": part.text, "answer": result.chosen.value, "status": result.status,
                           "record_ids": citation_ids, "reason": result.abstention_reason})
        for item in result.evidence:
            if item not in evidence:
                evidence.append(item)
    if conflicts:
        conflict = conflicts[0]
        if len(conflicts) > 1:
            conflict["conflicting_parts"] = conflicts
        return VerificationResult(
            "conflict", None, "not_claimed", evidence=evidence, candidate_checks=checks, conflict=conflict,
            abstention_reason="one or more question parts have conflicting source values",
            answer_components=components)
    if not n_supported:
        return VerificationResult("insufficient", None, "not_claimed", evidence=evidence,
                                  candidate_checks=checks,
                                  abstention_reason="no question part had verifiable supporting evidence",
                                  answer_components=components)
    answer = Candidate("; ".join(answers), chosen_record, "multi_part")
    reason = f"verified {n_supported} of {len(parts)} question parts" if missing else None
    status = "partial" if missing else "unverified" if unverified else "supported"
    return VerificationResult(status, answer, "not_claimed",
                              evidence=evidence, candidate_checks=checks,
                              abstention_reason=reason, n_supported=n_supported, answer_components=components)


def _event_details(record: Event) -> list[str]:
    """Include source-record details that make a log event answerable, not just its type."""
    details = [
        f"{o.name if o.name.isupper() else o.name.replace('_', ' ')}="
        f"{(o.raw if o.raw not in (None, '') else o.value)}"
        + (f" {o.unit}" if o.unit else "")
        for o in record.observations
    ]
    excluded = {"timestamp", "event_type", "log_level", "log_source", "equipment", "quality"}
    observation_names = {o.name.lower() for o in record.observations}
    if record.observations:
        excluded |= {"tag", "tag_name", "signal", "value", "reading", "measurement", "unit", "units", "uom"}
    details.extend(
        f"{name.replace('_', ' ')}={value.strip().replace('_', ' ') if isinstance(value, str) else value}"
        for name, value in (record.raw or {}).items()
        if name.lower() not in excluded and name.lower() not in observation_names and value not in (None, "")
    )
    return details


def verify_text(parsed: ParsedQuery, ranked: list[Candidate], records: list[Event], store=None,
                enabled: bool = True, time_window_min: float = 30.0) -> VerificationResult | None:
    """Verify Mamba-ranked text candidates. Returns None when the question names no identifier
    or no untyped record is available (the caller then reports why it cannot answer)."""
    anchors = parsed.anchor_map()
    pool = [r for r in records if is_untyped(r)]
    if not anchors or not pool:
        return None
    parts = question_parts(parsed)
    if len(parts) > 1:
        return _verify_question_parts(parts, ranked, records, store, enabled, time_window_min)
    if not enabled:
        if not ranked:
            return VerificationResult("insufficient", None, "not_claimed", abstention_reason="no candidate answer")
        c = ranked[0]
        return VerificationResult("unverified", c, "claimed_unverified" if parsed.causal_language else "not_claimed",
                                  evidence=[evidence_entry(c.record, "answer", classify(c.record, c.field, False), c.field)])
    focus = parsed.focus_terms()
    checks = {r.event_id: _record_check(r, parsed, anchors, focus, records, store) for r in pool}
    public = [{k: v for k, v in ch.items() if k != "date_src"} for ch in checks.values()]
    if parsed.listing:
        return _timeline(parsed, pool, checks, public, anchors, focus)
    claim_q = parsed.causal_language or bool(CLAIM_STEMS & set(focus))
    passing, qualified = [], []
    for c in ranked:
        ch = checks.get(c.record.event_id)
        if ch is None or not ch["passed"]:
            continue
        if claim_q and QUALIFIED_RE.search(c.value):
            qualified.append(c)
            continue
        passing.append(c)
    passing, time_note = _nearest_in_time(parsed, passing, time_window_min)
    if not passing:
        reason = _why(public, anchors, focus, parsed.dates[0] if parsed.dates else None)
        if qualified:
            reason = (f"the source does not establish it; it states: \"{qualified[0].value[:200]}\" "
                      f"(record {qualified[0].record.event_id})")
        elif time_note:
            reason = time_note
        elif any(ch["passed"] for ch in public):
            reason = (f"records naming {', '.join(anchors)} pass identity/date checks but carry no field or "
                      f"passage answering '{' '.join(focus)}'")
        res = VerificationResult("insufficient", None, "not_claimed", candidate_checks=public, abstention_reason=reason)
        for c in qualified[:3]:
            res.evidence.append(evidence_entry(c.record, "context", "documented"))
        return res
    chosen = passing[0]
    res = VerificationResult("supported", chosen, "documented_by_source" if parsed.causal_language else "not_claimed",
                             candidate_checks=public, n_supported=len(passing))
    if chosen.field != "document_text":
        rivals = [c for c in passing if c.field.lower() == chosen.field.lower()
                  and c.record.event_id != chosen.record.event_id and c.key() != chosen.key()]
        # same field, same time, different value -> conflict; at other times -> several entries
        same_time = [c for c in rivals if c.record.timestamp == chosen.record.timestamp]
        if rivals and not same_time:
            res.status = "partial"
            res.abstention_reason = None
            for c in rivals[:5]:
                res.evidence.append(evidence_entry(c.record, "context", "documented"))
            rivals = []
        rivals = same_time
        if rivals:
            vals: dict[str, list[str]] = {}
            for c in [chosen] + rivals:
                vals.setdefault(c.value, []).append(c.record.event_id)
            res.status, res.chosen, res.causal_claim_status = "conflict", None, "not_claimed"
            res.conflict = {"field": chosen.field, "values": [{"value": v, "record_ids": ids} for v, ids in vals.items()]}
            res.abstention_reason = "records disagree and no supersession or version resolves the conflict"
            for c in [chosen] + rivals:
                res.evidence.append(evidence_entry(c.record, "conflicting", classify(c.record, c.field, False), c.field))
            return res
    res.evidence.append(evidence_entry(chosen.record, "answer", classify(chosen.record, chosen.field, False),
                                       chosen.field))
    src = checks[chosen.record.event_id]["date_src"]
    if src is not None and src.event_id != chosen.record.event_id:
        res.evidence.append(evidence_entry(src, "link", "documented"))
    docs = {c.record.doc_id or c.record.event_id for c in passing if c.field == "document_text"}
    if chosen.field == "document_text" and len(docs) > 1:
        res.status = "partial"             # agreement between free-text documents is not verified
        for c in passing[1:4]:
            res.evidence.append(evidence_entry(c.record, "context", "documented"))
    return res


def _nearest_in_time(parsed: ParsedQuery, passing: list[Candidate], window: float):
    if not (parsed.times and parsed.dates) or not passing:
        return passing, None
    try:
        want = datetime.fromisoformat(f"{parsed.dates[0]}T{parsed.times[0]}")
    except ValueError:
        return passing, None
    timed = [(abs((t - want).total_seconds()) / 60, c) for c in passing if (t := _ts(c.record)) is not None]
    untimed = [c for c in passing if _ts(c.record) is None]
    near = [(d, c) for d, c in timed if d <= window]
    if timed and not near and not untimed:
        return [], f"no record within {window:g} min of {parsed.dates[0]} {parsed.times[0]}"
    best = min((d for d, _ in near), default=None)
    kept = [c for d, c in near if d == best] + untimed
    return [c for c in passing if c in kept], None


def _timeline(parsed, pool, checks, public, anchors, focus) -> VerificationResult:
    """Listing questions: every record that names the identifier, satisfies the date and contains
    at least one content term, in time order. Each item is quoted from its own record."""
    hits = [r for r in pool if checks[r.event_id]["names_identifier"] and checks[r.event_id]["date_ok"] is not False
            and (not focus or set(focus) - set(checks[r.event_id]["missing_terms"]))]
    if not hits:
        return VerificationResult("insufficient", None, "not_claimed", candidate_checks=public,
                                  abstention_reason=_why(public, anchors, focus, parsed.dates[0] if parsed.dates else None))
    hits.sort(key=lambda r: (r.timestamp or "", r.event_id))
    lines = [f"{r.timestamp or 'time unknown'}: {_body(r)[0][:160]}" for r in hits]
    res = VerificationResult("supported", Candidate(" | ".join(lines), hits[0], "timeline"), "not_claimed",
                             candidate_checks=public, n_supported=len(hits))
    for r in hits:
        res.evidence.append(evidence_entry(r, "answer", "documented", "timeline"))
    return res


def _same_document_date(ch: Event, ident: str | None, want: str, records: list[Event], store) -> Event | None:
    if ident is None or not ch.doc_id:
        return None
    siblings = [r for r in records if r.doc_id == ch.doc_id and r.event_id != ch.event_id]
    if store is not None:
        known = {r.event_id for r in siblings}
        siblings += [r for r in store.events_by_doc(ch.doc_id) if r.event_id not in known and r.event_id != ch.event_id]
    for s in siblings:
        body = " ".join(declarative_sentences(s.text))
        if _mentions(body, ident) and want in dates_in(body):
            return s
    return None


def _answer_text(sents: list[str], focus: list[str]) -> str:
    """Quote the statements that each contain ALL question terms; if no single statement does
    (e.g. a term only appears in the section heading), quote the passage's statements (bounded).
    Headings (<= 3 words) are not quoted."""
    body = [s for s in sents if len(s.split()) > 3] or sents
    full = [s for s in body if all(_term_is_covered(term, _stems(s)) for term in focus)] if focus else []
    text = " ".join(full or body)
    return text if len(text) <= 400 else text[:397] + "..."


def _why(checks: list[dict], anchors: dict, focus: list[str], want_date: str | None) -> str:
    named = [c for c in checks if c["names_identifier"]]
    if not named:
        return f"no record names {', '.join(anchors)}"
    if want_date and not any(c["date_ok"] for c in named):
        return f"records naming {', '.join(anchors)} do not state the date {want_date}"
    return (f"records naming {', '.join(anchors)} do not state "
            f"{', '.join(sorted({t for c in named for t in c['missing_terms']}))}")


def n_distinct(ranked: list[Candidate]) -> int:
    return len({normalize_answer(c.value) for c in ranked})


def share_note(n: int) -> str | None:
    return ("only one candidate value was ranked, so the share is trivially 1.0 and carries no information"
            if n == 1 else None)


def candidate_share(ranked: list[Candidate], value: str | None = None) -> float | None:
    """Softmax share of the top candidate VALUE among distinct candidate values, from Mamba's
    mean token log-probabilities. A model-derived, UNCALIBRATED relative score; it says how
    strongly Mamba prefers the top value over the other extracted values, not P(correct)."""
    import math

    best: dict[str, float] = {}
    for c in ranked:
        if c.score is None:
            return None
        k = normalize_answer(c.value)
        best[k] = max(best.get(k, -1e9), c.score)
    if not best:
        return None
    m = max(best.values())
    z = sum(math.exp(v - m) for v in best.values())
    k = normalize_answer(value) if value is not None else max(best, key=best.get)
    return math.exp(best[k] - m) / z if k in best else None


def supported_record_ids(parsed: ParsedQuery, records: list[Event], store=None) -> set[str]:
    """IDs of untyped records that pass identity, date and coverage (record-level checks).
    Used to skip ranking candidates that verification could never accept."""
    anchors = parsed.anchor_map()
    if not anchors:
        return set()
    focus = parsed.focus_terms()
    return {r.event_id for r in records if is_untyped(r)
            and _record_check(r, parsed, anchors, focus, records, store)["passed"]}
