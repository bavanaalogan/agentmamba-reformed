"""Normalization of timestamps, identifiers and units.

Policy:
* Timestamps are parsed only with an explicit format list; the raw string is always kept.
  Timezone is never assumed: it is recorded as given (default ``"unknown"``).
* Units are kept exactly as recorded. ``normalize_unit`` only canonicalizes the *spelling*
  of a small set of known unit tokens (e.g. "°C" -> "degC"); it never converts values.
* Identifiers are stripped and upper-cased; empty / placeholder values become None.
"""
from __future__ import annotations

import re
from datetime import datetime

TIMESTAMP_FORMATS = (
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
    "%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S.%f", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
    "%d %b %Y %H:%M:%S", "%d %b %Y %H:%M", "%d %B %Y %H:%M:%S", "%d %B %Y %H:%M",
    "%b %d %Y %H:%M:%S", "%b %d %Y %H:%M", "%B %d %Y %H:%M:%S", "%B %d %Y %H:%M",
    "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y",
)
_UNIT_SPELLINGS = {"°c": "degC", "degc": "degC", "deg c": "degC", "c": "degC", "mm": "mm", "kwh": "kWh",
                   "kvarh": "kVarh", "%": "%", "ppm": "ppm", "m/min": "m/min", "min": "min"}
_ID_PLACEHOLDERS = {"", "NA", "N/A", "NONE", "NULL", "-", "?", "UNKNOWN"}


def normalize_timestamp(raw: str | None, formats: tuple[str, ...] = TIMESTAMP_FORMATS) -> str | None:
    """Return ISO-8601 (no timezone conversion) or None if the value cannot be parsed
    with one of the declared formats. Ambiguous day/month order is resolved ONLY by the
    declared format order, which callers should set per source."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = re.sub(r"(?<=\d),(?=\d{1,6}(?:\s*(?:Z|[+-]\d{2}:?\d{2})?$))", ".", s, flags=re.I)
    s = re.sub(r"\s+(?=(?:Z|[+-]\d{2}:?\d{2})$)", "", s, flags=re.I)
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    # ISO strings with offsets (kept as-is, offset preserved)
    try:
        return datetime.fromisoformat(s).isoformat(timespec="seconds")
    except ValueError:
        pass
    zone = re.search(r"\s*(Z|[+-]\d{2}:?\d{2})$", s, re.I)
    if zone:
        base = s[:zone.start()].rstrip()
        offset = zone.group(1)
        if offset.upper() == "Z":
            offset = "+00:00"
        elif len(offset) == 5:
            offset = f"{offset[:3]}:{offset[3:]}"
        try:
            tzinfo = datetime.fromisoformat(f"2000-01-01T00:00:00{offset}").tzinfo
        except ValueError:
            return None
        for fmt in formats:
            try:
                return datetime.strptime(base, fmt).replace(tzinfo=tzinfo).isoformat(timespec="seconds")
            except ValueError:
                continue
    return None


def normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    key = str(unit).strip().lower()
    return _UNIT_SPELLINGS.get(key, str(unit).strip())


def normalize_identifier(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    return None if s in _ID_PLACEHOLDERS else s
