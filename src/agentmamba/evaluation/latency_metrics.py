"""Latency summaries from per-query timing dicts."""
from __future__ import annotations

import numpy as np


def summarize_timings(timings: list[dict]) -> dict:
    keys = sorted({k for t in timings for k in t})
    out = {}
    for k in keys:
        v = np.array([t[k] for t in timings if k in t], dtype="float64")
        if len(v):
            out[k] = {"n": int(len(v)), "mean": round(float(v.mean()), 4), "p50": round(float(np.median(v)), 4),
                      "p95": round(float(np.percentile(v, 95)), 4), "max": round(float(v.max()), 4)}
    return out
