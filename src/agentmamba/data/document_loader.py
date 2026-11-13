"""Loaders that turn source files into ``Event`` records with provenance.

Supported, tested formats:
* JSONL / JSON of AgentMamba events (``Event.to_dict`` layout)            -> ``load_event_file``
* Generic CSV / JSON / JSONL operational records with a column mapping      -> ``load_records``
* Plain text documents, chunked deterministically by paragraph              -> ``load_text_document``
* UCI #851 Steel Industry Energy Consumption (public, CC BY 4.0), daily windows -> ``load_uci_851``
* UCI #198 Steel Plates Faults (public, CC BY 4.0), one quality record per plate -> ``load_uci_198``
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from agentmamba.data.event_normalizer import normalize_identifier, normalize_timestamp, normalize_unit
from agentmamba.memory.schema import ID_FIELDS, Event, Observation, stable_id
from agentmamba.reproducibility import utc_now

SUPPORTED_SUFFIXES = {".jsonl", ".json", ".csv", ".txt", ".md"}


def _read_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    if suffix == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    raise ValueError(f"unsupported record format {suffix!r}")


def load_event_file(path: str | Path) -> list[Event]:
    """Load records already in AgentMamba event layout (validated by the Event dataclass)."""
    return [Event.from_dict(r) for r in _read_rows(Path(path))]


def load_records(path: str | Path, mapping: dict, *, source_type: str = "authorized", stage: str,
                 event_type: str, timezone: str = "unknown", text_template: str | None = None,
                 observation_units: dict[str, str] | None = None, ts_formats: tuple | None = None) -> list[Event]:
    """Generic loader. ``mapping`` maps Event fields (timestamp, heat_id, ...) to source columns.
    Columns listed in ``observation_units`` become Observations with the *recorded* unit."""
    path = Path(path)
    rows = _read_rows(path)
    out = []
    for n, row in enumerate(rows):
        fields = {}
        for fld in ID_FIELDS:
            if fld in mapping:
                fields[fld] = normalize_identifier(row.get(mapping[fld]))
        ts_raw = row.get(mapping["timestamp"]) if "timestamp" in mapping else None
        ts = normalize_timestamp(ts_raw, ts_formats) if ts_formats else normalize_timestamp(ts_raw)
        obs = [Observation(col, _num(row.get(col)), normalize_unit(unit), row.get(col))
               for col, unit in (observation_units or {}).items() if row.get(col) not in (None, "")]
        text = text_template.format(**row) if text_template else "; ".join(f"{k}={v}" for k, v in row.items())
        eid = normalize_identifier(row.get(mapping["event_id"])) if "event_id" in mapping else None
        out.append(Event(
            event_id=eid or stable_id("EV", path.name, n, json.dumps(row, sort_keys=True)),
            stage=stage, event_type=event_type, timestamp=ts, timestamp_raw=ts_raw, timezone=timezone, text=text,
            source_ref=f"file://{path.name}#row={n + 1}", source_type=source_type,
            severity=row.get(mapping["severity"]) if "severity" in mapping else None,
            observations=obs, raw=dict(row), ingested_at=utc_now(), **fields))
    return out


# Common unambiguous date prefixes used in timestamped operational logs. Slash and dotted
# numeric dates are interpreted day-first by event_normalizer.normalize_timestamp.
_LOG_DATE = r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{4}|" \
            r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})"
_LOG_TIME = r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?)?"
LOG_LINE_RE = re.compile(r"^\s*(" + _LOG_DATE + _LOG_TIME
                         + r"(?:\s*(?:Z|[+-]\d{2}:?\d{2}))?)\s*[|;,]?\s*(.*)$")
LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "NOTICE", "MINOR", "MAJOR", "LOW", "MEDIUM", "HIGH", "WARN", "WARNING",
              "ERROR", "ALARM", "CRITICAL", "SEVERE", "FATAL"}
_KV_RE = re.compile(r'([A-Za-z_][\w.-]*)=("(?:[^"\\]|\\.)*"|[^|;]+)')
_POSITIONAL_EVENT_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:[_-][A-Z0-9]+)*$")


def is_log_text(text: str, min_lines: int = 3, min_share: float = 0.8) -> bool:
    """True when most non-comment lines are timestamped log entries (not prose)."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    hits = sum(bool(LOG_LINE_RE.match(ln)) for ln in lines)
    return len(lines) >= min_lines and hits / len(lines) >= min_share


def _unquote(v: str) -> str:
    return v[1:-1] if len(v) >= 2 and v[0] == v[-1] == '"' else v


def parse_log_line(line: str) -> dict | None:
    """Split a timestamped log line into fields without changing its evidence text.

    Recognizes the positional ``level | source | equipment | event_type`` prefix when present,
    then parses KEY=value details (surrounding quotes removed) and keeps free text as an action
    for WORK_ORDER_OPENED records or as a generic message otherwise.
    """
    m = LOG_LINE_RE.match(line)
    if not m:
        return None
    row: dict = {"timestamp": m.group(1).strip()}
    rest = m.group(2)
    parts = [x.strip() for x in rest.split("|")] if "|" in rest else [rest.strip()]
    positional = (
        len(parts) >= 4
        and parts[0].upper() in LOG_LEVELS
        and not _KV_RE.search(parts[1])
        and not _KV_RE.search(parts[2])
        and _POSITIONAL_EVENT_RE.fullmatch(parts[3]) is not None
    )
    if positional:
        row.update({"log_level": parts[0], "log_source": parts[1], "equipment": parts[2], "event_type": parts[3]})
        parts = parts[4:]
    free = []
    for part in parts:
        kvs = _KV_RE.findall(part)
        leftover = _KV_RE.sub("", part).strip(" ;,")
        if kvs:
            for k, v in kvs:
                row[k] = _unquote(v)
        if not leftover:
            continue
        if not kvs and leftover.upper() in LOG_LEVELS and "log_level" not in row:
            row["log_level"] = leftover
        elif not kvs and "log_source" not in row and " " not in leftover:
            row["log_source"] = leftover
        else:
            free.append(leftover)
    if free:
        field = "action" if row.get("event_type") == "WORK_ORDER_OPENED" else "message"
        row.setdefault(field, " ".join(free))
    return row


def load_text_log(path: str | Path, *, stage: str = "UTILITIES", source_type: str = "authorized") -> list[Event]:
    """One record per timestamped line of a plain-text log. The record text is the original line
    (exact evidence) and the source reference points to its line number; parsed fields go through
    ``load_log_records``. Comment lines (#) and blank lines are skipped."""
    path = Path(path)
    rows, originals = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = parse_log_line(line)
        if row is None:
            continue
        ts = row["timestamp"]
        rows.append(row)
        originals.append((n, line, ts))
    return load_log_records(path, stage=stage, source_type=source_type, rows=rows, type_column="event_type",
                            originals=originals)


def load_text_document(path: str | Path, *, stage: str = "MAINTENANCE", source_type: str = "authorized",
                       max_chars: int = 800, doc_id: str | None = None) -> list[Event]:
    """Deterministic paragraph chunking; paragraphs longer than ``max_chars`` are split on
    sentence boundaries. Each chunk keeps a source reference with its chunk index.
    Plain-text files that are line-oriented logs are routed to ``load_text_log`` instead."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if is_log_text(text):
        return load_text_log(path, stage=stage, source_type=source_type)
    doc_id = doc_id or normalize_identifier(path.stem)
    chunks: list[str] = []
    for para in [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]:
        while len(para) > max_chars:
            cut = para.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > 0 else max_chars
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            chunks.append(para)
    return [Event(event_id=stable_id("DOC", path.name, i, c), stage=stage, event_type="document_chunk",
                  timestamp=None, text=c, source_ref=f"file://{path.name}#chunk={i}", source_type=source_type,
                  doc_id=doc_id, raw={"chunk_index": i, "file": path.name}, ingested_at=utc_now())
            for i, c in enumerate(chunks)]


# Column-name aliases for operational logs (historian exports, alarm journals, CMMS exports).
# Matching is case-insensitive on the column name with spaces/dashes treated as "_".
LOG_TIMESTAMP_COLS = ("timestamp", "ts", "time", "datetime", "date_time", "event_time", "time_stamp", "date")
LOG_ID_COLS = {
    "asset_id": ("equipment", "equipment_id", "asset", "asset_id", "machine", "device", "unit_id", "eq_id"),
    "heat_id": ("heat", "heat_id"), "ladle_id": ("ladle", "ladle_id"), "coil_id": ("coil", "coil_id"),
    "work_order_id": ("work_order", "work_order_id", "wo", "wo_id"), "doc_id": ("doc_id", "document_id"),
}
LOG_TYPE_COLS = ("event_type", "record_type", "log_type", "category", "type")
LOG_SEVERITY_COLS = ("severity", "level", "log_level", "priority")
LOG_SIGNAL_COLS = ("signal", "tag", "tag_name", "sensor", "variable", "parameter")
LOG_VALUE_COLS = ("value", "reading", "measurement")
EVENT_LAYOUT_KEYS = {"event_id", "stage", "event_type", "text", "source_ref", "source_type"}


def _key(col: str) -> str:
    return "_".join(str(col).strip().lower().replace("-", " ").split())


def _find(cols: dict[str, str], aliases) -> str | None:
    return next((cols[a] for a in aliases if a in cols), None)


def _severity(value) -> str | None:
    if value in (None, ""):
        return None
    level = str(value).strip().lower()
    return {
        "info": "info", "informational": "info",
        "minor": "minor", "low": "minor", "notice": "minor",
        "major": "major", "medium": "major", "warn": "major", "warning": "major",
        "error": "major", "alarm": "major", "high": "major",
        "critical": "critical", "fatal": "critical", "severe": "critical",
    }.get(level)


def load_log_records(path: str | Path, *, stage: str = "MAINTENANCE", source_type: str = "authorized",
                     timezone: str = "unknown", rows: list[dict] | None = None, type_column: str | None = None,
                     originals: list[tuple[int, str, str]] | None = None) -> list[Event]:
    """Generic loader for operational logs whose columns are NOT in AgentMamba event layout.

    * timestamp, equipment / heat / ladle / work-order columns are recognized by name
      (``LOG_*_COLS``); unrecognized columns are kept verbatim in ``raw`` and in ``text``.
    * long-format sensor rows (signal/tag + value [+ unit]) become one Observation named after
      the signal, with the recorded unit; other numeric columns become Observations without unit.
    * event_type comes from a type/category column if present, otherwise from the file name
      (e.g. ``alarm_log_20260314.csv`` -> ``alarm_log``).
    * the record ID is derived from the file name and the row content, so re-ingesting the same
      file yields duplicates (not new records) and identical rows inside one file collapse.
    The original row is stored untouched in ``raw``; nothing is converted or inferred."""
    path = Path(path)
    rows = _read_rows(path) if rows is None else rows
    # file-name type: trailing name parts containing digits (dates, sequence numbers) are dropped,
    # so alarm_log_20260314.jsonl and alarm_log_d2.jsonl are both of type "alarm_log"
    parts = _key(path.stem).split("_")
    while parts and any(ch.isdigit() for ch in parts[-1]):
        parts.pop()
    default_type = "_".join(parts) or "log_record"
    out = []
    for n, row in enumerate(rows):
        if not isinstance(row, dict) or not row:
            continue
        cols = {_key(c): c for c in row}
        ts_col = _find(cols, LOG_TIMESTAMP_COLS)
        ts_raw = row.get(ts_col) if ts_col else None
        ts = normalize_timestamp(str(ts_raw)) if ts_raw not in (None, "") else None
        ids = {fld: normalize_identifier(str(row[c])) for fld, al in LOG_ID_COLS.items()
               if (c := _find(cols, al)) and row.get(c) not in (None, "")}
        type_col = (type_column if type_column in row else None) or _find(cols, LOG_TYPE_COLS)
        etype = _key(row[type_col]) if type_col and row.get(type_col) else default_type
        severity_col = _find(cols, LOG_SEVERITY_COLS)
        sig_col, val_col = _find(cols, LOG_SIGNAL_COLS), _find(cols, LOG_VALUE_COLS)
        unit_col = cols.get("unit") or cols.get("units") or cols.get("uom")
        used = {c for c in (ts_col, type_col, sig_col, val_col, unit_col) if c} | \
               {cols[a] for al in LOG_ID_COLS.values() for a in al if a in cols}
        obs = []
        if sig_col and val_col and row.get(sig_col) not in (None, ""):
            unit = row.get(unit_col) if unit_col else None
            obs.append(Observation(str(row[sig_col]), _num(row.get(val_col)), normalize_unit(unit) if unit else None,
                                   row.get(val_col)))
        for c, v in row.items():
            if c not in used and isinstance(_num(v), (int, float)) and not isinstance(v, bool):
                obs.append(Observation(str(c), _num(v), None, v))
        text = "; ".join(f"{c}={v}" for c, v in row.items() if v not in (None, ""))
        ref, tz, raw_ts = f"file://{path.name}#row={n + 1}", timezone, ts_raw
        if originals is not None:          # plain-text log: keep the exact line and its line number
            line_no, text, raw_ts = originals[n]
            ref = f"file://{path.name}#line={line_no}"
            off = re.search(r"(Z|[+-]\d{2}:?\d{2})$", str(row.get(ts_col) or ""))
            tz = off.group(1) if off else timezone
        out.append(Event(
            event_id=stable_id("LOG", path.name, json.dumps(row, sort_keys=True, default=str)),
            stage=stage, event_type=etype, timestamp=ts, timestamp_raw=None if raw_ts is None else str(raw_ts),
            timezone=tz, text=text, source_ref=ref, source_type=source_type,
            severity=_severity(row.get(severity_col)) if severity_col else None,
            observations=obs, raw=dict(row), ingested_at=utc_now(), **ids))
    return out


def is_event_layout(rows: list[dict]) -> bool:
    return bool(rows) and all(isinstance(r, dict) and EVENT_LAYOUT_KEYS <= set(r) for r in rows)


def load_any(path: str | Path, **kw) -> list[Event]:
    """Route a file to its loader: text documents are chunked; CSV/JSON/JSONL in AgentMamba
    event layout are loaded as events; any other CSV/JSON/JSONL is loaded as an operational log."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported file type {suffix!r}; supported: {sorted(SUPPORTED_SUFFIXES)}")
    if suffix in (".txt", ".md"):
        return load_text_document(path, **kw)      # line-oriented logs are detected inside
    if "mapping" in kw:
        return load_records(path, **kw)
    rows = _read_rows(path)
    if is_event_layout(rows):
        return [Event.from_dict(r) for r in rows]
    return load_log_records(path, rows=rows, **kw)


# ---------------------------------------------------------------- public datasets
UCI_851_UNITS = {"Usage_kWh": "kWh", "Lagging_Current_Reactive.Power_kVarh": "kVarh",
                 "Leading_Current_Reactive_Power_kVarh": "kVarh", "CO2(tCO2)": "tCO2"}


def load_uci_851(path: str | Path) -> list[Event]:
    """UCI #851 (DAEWOO Steel, 15-min energy readings, CC BY 4.0). Rows are aggregated into
    one event per calendar day; the event keeps the raw row range it summarizes. Timestamps are
    day-first (dd/mm/YYYY HH:MM) and the timezone is not documented -> recorded as unknown."""
    import pandas as pd

    path = Path(path)
    df = pd.read_csv(path)
    df["_ts"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    df["_row"] = range(1, len(df) + 1)
    out = []
    for day, g in df.groupby(df["_ts"].dt.date):
        peak = g.loc[g["Usage_kWh"].idxmax()]
        total = round(float(g["Usage_kWh"].sum()), 2)
        dominant = g["Load_Type"].mode().iloc[0]
        text = (f"Daily energy summary {day} (public UCI #851 data): total usage {total} kWh over {len(g)} "
                f"15-minute readings; peak {peak['Usage_kWh']} kWh at {peak['_ts'].strftime('%H:%M')}; "
                f"dominant load type {dominant}.")
        out.append(Event(
            event_id=f"UCI851-{day}", stage="UTILITIES", event_type="energy_daily_summary",
            timestamp=f"{day}T00:00:00", timestamp_raw=str(g["date"].iloc[0]), timezone="unknown",
            text=text, source_ref=f"uci://851/{path.name}#rows={g['_row'].min()}-{g['_row'].max()}",
            source_type="public", campaign_id="UCI851", asset_id="UCI851-PLANT",
            observations=[Observation("total_usage", total, "kWh"),
                          Observation("peak_usage", float(peak["Usage_kWh"]), "kWh", str(peak["Usage_kWh"])),
                          Observation("peak_time", peak["_ts"].strftime("%H:%M"), None, str(peak["date"])),
                          Observation("dominant_load_type", dominant)],
            raw={"rows": [int(g["_row"].min()), int(g["_row"].max())], "n_readings": int(len(g))},
            ingested_at=utc_now()))
    return out


UCI_198_FAULTS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]


def load_uci_198(nna_path: str | Path, var_path: str | Path) -> list[Event]:
    """UCI #198 (Semeion, CC BY 4.0). No timestamps and no plant identifiers exist in the source,
    so these are standalone quality observations with no relationships."""
    names = [n.strip() for n in Path(var_path).read_text(encoding="utf-8").splitlines() if n.strip()]
    out = []
    for n, line in enumerate(Path(nna_path).read_text(encoding="utf-8").splitlines(), start=1):
        vals = line.split()
        if len(vals) != len(names):
            continue
        row = dict(zip(names, vals))
        fault = next((f for f in UCI_198_FAULTS if row.get(f) == "1"), "unknown")
        steel = "A300" if row["TypeOfSteel_A300"] == "1" else "A400"
        text = (f"Steel plate surface-fault record (public UCI #198 data): fault class {fault}; steel type {steel}; "
                f"plate thickness {row['Steel_Plate_Thickness']}; defect area {row['Pixels_Areas']} pixels.")
        out.append(Event(
            event_id=f"UCI198-{n:04d}", stage="QUALITY", event_type="plate_fault_record", timestamp=None,
            text=text, source_ref=f"uci://198/Faults.NNA#line={n}", source_type="public", campaign_id="UCI198",
            observations=[Observation("fault_class", fault), Observation("steel_type", steel),
                          Observation("plate_thickness", _num(row["Steel_Plate_Thickness"]), None,
                                      row["Steel_Plate_Thickness"])],
            raw=row, ingested_at=utc_now()))
    return out


def _num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return v
