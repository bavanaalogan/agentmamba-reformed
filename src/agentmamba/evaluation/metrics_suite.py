"""Comprehensive metrics for the Baseline-Mamba vs AgentMamba comparison (evaluation only).

Input: scored rows from ``comparison.score_output`` (one per question x stage x system). Every
metric is returned as ``{"value", "n", "definition", "better"}``; ``value`` is None (shown as N/A)
when its denominator is empty or the metric does not apply (e.g. retrieval metrics for a system
that never retrieves). Nothing unavailable is reported as zero.

Question classes:
  answerable    gold answer_kind == "value"
  unanswerable  gold answer_kind == "abstain"
  conflict      gold answer_kind == "conflict" (correct = reports the conflict); counted in the
                overall rates only
Text metrics compare the normalized answer with the normalized reference (lower-case, commas
removed, numbers compared numerically). An abstention on an answerable question is scored as an
empty answer (0), which is a real outcome, not a missing value.
"""
from __future__ import annotations

import math
from collections import Counter

from agentmamba.evaluation.comparison import brier, ece, norm
from agentmamba.evaluation.metrics import token_f1


# ---------------------------------------------------------------- text overlap
def _toks(s: str | None) -> list[str]:
    return norm(s).split() if s else []


def rouge1_f(pred: str | None, gold: str | None) -> float:
    p, g = _toks(pred), _toks(gold)
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rougeL_f(pred: str | None, gold: str | None) -> float:
    p, g = _toks(pred), _toks(gold)
    if not p or not g:
        return float(p == g)
    lcs = _lcs(p, g)
    if not lcs:
        return 0.0
    pr, rc = lcs / len(p), lcs / len(g)
    return 2 * pr * rc / (pr + rc)


def bleu1(pred: str | None, gold: str | None) -> float:
    """Clipped unigram precision x brevity penalty exp(1 - r/c) when the candidate is shorter."""
    p, g = _toks(pred), _toks(gold)
    if not p or not g:
        return float(p == g)
    clipped = sum((Counter(p) & Counter(g)).values()) / len(p)
    bp = 1.0 if len(p) > len(g) else math.exp(1 - len(g) / len(p))
    return clipped * bp


# ---------------------------------------------------------------- ranking
def _rel(retrieved: list[str], relevant: list[str], k: int) -> list[int]:
    rel = set(relevant)
    return [1 if r in rel else 0 for r in retrieved[:k]]


def ndcg_at_k(retrieved, relevant, k=10) -> float:
    g = _rel(retrieved, relevant, k)
    dcg = sum(x / math.log2(i + 2) for i, x in enumerate(g))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(set(relevant)), k)))
    return dcg / ideal if ideal else 0.0


def ap_at_k(retrieved, relevant, k=10) -> float:
    g = _rel(retrieved, relevant, k)
    hits, s = 0, 0.0
    for i, x in enumerate(g, 1):
        if x:
            hits += 1
            s += hits / i
    return s / min(len(set(relevant)), k) if relevant else 0.0


def recall_at(retrieved, relevant, k) -> float:
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def hit_at(retrieved, relevant, k) -> float:
    return float(bool(set(retrieved[:k]) & set(relevant)))


# ---------------------------------------------------------------- helpers
def _m(value, n, definition, better):
    return {"value": None if value is None else round(float(value), 4), "n": n, "definition": definition,
            "better": better}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
    return xs[i]


# ---------------------------------------------------------------- suite
def metric_suite(rows: list[dict], *, retrieves: bool, k: int = 10, model_load_s: float | None = None,
                 min_calibration_n: int = 20, min_p95_n: int = 20) -> dict:
    """All metrics for ONE system's rows. ``retrieves`` = the system can retrieve at all."""
    ans = [r for r in rows if r["answer_kind"] == "value"]
    unans = [r for r in rows if r["answer_kind"] == "abstain"]
    answered = [r for r in rows if r["answered"]]
    with_req = [r for r in rows if r["required"]]
    out: dict = {"n_rows": len(rows), "n_answerable": len(ans), "n_unanswerable": len(unans),
                 "n_conflict": sum(r["answer_kind"] == "conflict" for r in rows), "n_answered": len(answered)}

    def txt(fn):
        return _mean([fn(r["answer_value"] if r["answered"] else "", r["expected"]) for r in ans])

    A = {}
    v, n = _mean([float(r["em"]) for r in ans])
    A["exact_match"] = _m(v, n, "normalized answer == reference; answerable questions (abstention = miss)", "higher")
    for name, fn, d in (("f1", token_f1, "token-level F1"), ("rouge_1", rouge1_f, "unigram-overlap F-measure"),
                        ("rouge_l", rougeL_f, "longest-common-subsequence F-measure"),
                        ("bleu_1", bleu1, "clipped unigram precision x brevity penalty")):
        v, n = txt(fn)
        A[name] = _m(v, n, f"{d} vs reference; answerable questions (abstention scored as empty answer)", "higher")
    A["correct_answer_rate"] = _m(sum(r["correct"] for r in rows) / len(rows) if rows else None, len(rows),
                                  "correct outcomes / all questions (value match, abstain, or conflict as gold says)",
                                  "higher")
    A["incorrect_answer_rate"] = _m(sum(r["answered"] and not r["correct"] for r in rows) / len(rows) if rows else None,
                                    len(rows), "answered but not correct / all questions", "lower")
    A["unsupported_answer_rate"] = _m(sum(r["unsupported"] for r in answered) / len(answered) if answered else None,
                                      len(answered), "answers not grounded in required evidence (or that should "
                                                     "not have been given) / answers", "lower")
    A["correct_abstention_rate"] = _m(sum(r["abstained"] for r in unans) / len(unans) if unans else None, len(unans),
                                      "abstained / unanswerable questions", "higher")
    out["answer"] = A

    E = {}
    na = len(ans)
    E["answer_error_rate"] = _m(sum(not r["correct"] for r in ans) / na if na else None, na,
                                "answerable questions not answered correctly (wrong or abstained) / answerable", "lower")
    E["false_answer_rate"] = _m(sum(r["answered"] and not r["correct"] for r in ans) / na if na else None, na,
                                "answerable questions answered wrongly (non-abstaining) / answerable", "lower")
    E["abstention_error_rate"] = _m(sum(r["abstained"] for r in ans) / na if na else None, na,
                                    "answerable questions abstained on / answerable", "lower")
    E["unsupported_claim_rate"] = A["unsupported_answer_rate"] | {"definition": "answers containing an unsupported "
                                                                                "claim / answers evaluated for grounding"}
    need = [r for r in with_req if r["evidence_in_context"] is False]
    if retrieves:
        fail = [r for r in need if not set(r["required"]) <= set(r["retrieved"])]
        E["retrieval_failure_rate"] = _m(len(fail) / len(need) if need else None, len(need),
                                         "questions whose required evidence was NOT in the initial context and was "
                                         "not retrieved / such questions", "lower")
    else:
        E["retrieval_failure_rate"] = _m(None, 0, "not applicable: system does not retrieve", "lower")
    miss = [r for r in with_req if not r["evidence_covered"]]
    E["evidence_miss_rate"] = _m(len(miss) / len(with_req) if with_req else None, len(with_req),
                                 "questions whose required evidence was neither in the context nor retrieved / "
                                 "questions with required evidence", "lower")
    cal = [(r.get("calibrated_confidence"), r["correct"]) for r in answered if r.get("calibrated_confidence") is not None]
    ok = len(cal) >= min_calibration_n
    E["brier"] = _m(brier([c for c, _ in cal], [y for _, y in cal]) if ok else None, len(cal),
                    f"mean squared error of calibrated confidence vs correctness (answers; needs >= {min_calibration_n})",
                    "lower")
    E["ece"] = _m(ece([c for c, _ in cal], [y for _, y in cal]) if ok else None, len(cal),
                  "expected calibration error, 10 bins (answers with calibrated confidence)", "lower")
    out["errors"] = E

    G = {}
    G["grounded_answer_rate"] = _m(sum(r["grounded_correct"] for r in ans) / na if na else None, na,
                                   "answerable questions answered correctly AND citing a required evidence record / "
                                   "answerable", "higher")
    G["evidence_coverage"] = _m(1 - E["evidence_miss_rate"]["value"] if E["evidence_miss_rate"]["value"] is not None
                                else None, len(with_req), "required evidence in context or retrieved / questions with "
                                                          "required evidence", "higher")
    out["evidence"] = G

    R = {}
    trig = [r for r in rows if r["retrieval_triggered"]]
    scored = [r for r in trig if r["required"]]
    if retrieves:
        for name, fn, d in ((f"ndcg@{k}", lambda r: ndcg_at_k(r["retrieved"], r["required"], k), "binary-relevance NDCG"),
                            (f"map@{k}", lambda r: ap_at_k(r["retrieved"], r["required"], k), "average precision"),
                            (f"recall@{k}", lambda r: recall_at(r["retrieved"], r["required"], k), "recall"),
                            ("recall@1", lambda r: recall_at(r["retrieved"], r["required"], 1), "recall at rank 1"),
                            ("hit@1", lambda r: hit_at(r["retrieved"], r["required"], 1), "any required record at rank 1"),
                            ("hit@3", lambda r: hit_at(r["retrieved"], r["required"], 3), "... within top 3"),
                            ("hit@10", lambda r: hit_at(r["retrieved"], r["required"], 10), "... within top 10")):
            v, n = _mean([fn(r) for r in scored])
            R[name] = _m(v, n, f"{d} of gold evidence IDs in the retrieved list; questions where retrieval ran and "
                               f"gold evidence exists", "higher")
        R["retrieval_trigger_rate"] = _m(len(trig) / len(rows) if rows else None, len(rows),
                                         "questions where retrieval ran / all questions", "info")
        suff = [r for r in with_req if r["evidence_in_context"]]
        R["unnecessary_retrieval_rate"] = _m(sum(r["retrieval_triggered"] for r in suff) / len(suff) if suff else None,
                                             len(suff), "retrieved although all required evidence was already in the "
                                                        "initial context / such questions", "lower")
    else:
        for name in (f"ndcg@{k}", f"map@{k}", f"recall@{k}", "recall@1", "hit@1", "hit@3", "hit@10",
                     "retrieval_trigger_rate", "unnecessary_retrieval_rate"):
            R[name] = _m(None, 0, "not applicable: system does not retrieve", "higher")
    R["required_evidence_coverage"] = G["evidence_coverage"]
    out["retrieval"] = R

    out["efficiency"] = efficiency_metrics(rows, retrieves=retrieves, model_load_s=model_load_s, min_p95_n=min_p95_n)
    return out


def efficiency_metrics(rows: list[dict], *, retrieves: bool, model_load_s: float | None = None,
                       min_p95_n: int = 20, source: str = "measured in this run") -> dict:
    """Latency statistics from per-question rows with total_s / generation_s / retrieval_s /
    verification_s (seconds, warm model). Model loading is reported separately."""
    T = {}
    tot = [r["total_s"] for r in rows if r.get("total_s") is not None]
    T["total_s"] = _m(sum(tot) if tot else None, len(tot), "sum of end-to-end query latency (warm model)", "lower")
    v, n = _mean(tot)
    T["mean_s"] = _m(v, n, "mean end-to-end latency per question (warm)", "lower")
    T["median_s"] = _m(_pct(tot, 0.5), len(tot), "median end-to-end latency (warm)", "lower")
    T["p95_s"] = _m(_pct(tot, 0.95) if len(tot) >= min_p95_n else None, len(tot),
                    f"95th percentile latency (needs >= {min_p95_n} observations)", "lower")
    for r in rows:      # AgentMamba timings split Mamba scoring into initial + final answer stages
        if r.get("generation_s") is None and (r.get("initial_answer_s") is not None or r.get("final_answer_s") is not None):
            r["generation_s"] = (r.get("initial_answer_s") or 0.0) + (r.get("final_answer_s") or 0.0)
    for key, name, d in (("generation_s", "generation_mean_s", "Mamba answer scoring"),
                         ("retrieval_s", "retrieval_mean_s", "retrieval (SQLite + FAISS + channels)"),
                         ("verification_s", "verification_mean_s", "evidence verification")):
        v, n = _mean([r.get(key) for r in rows])
        if name != "generation_mean_s" and not retrieves and key == "retrieval_s":
            T[name] = _m(None, 0, "not applicable: system does not retrieve", "lower")
        else:
            T[name] = _m(v, n, f"mean {d} time per question where it ran", "lower")
    T["model_load_s"] = _m(model_load_s, 1 if model_load_s is not None else 0,
                           "one-time model loading (cold start); excluded from query latency", "lower")
    T["source"] = source
    return T


# ---------------------------------------------------------------- composite
DEFAULT_COMPOSITE = {"weights": {"answer_quality": 0.4, "grounding": 0.3, "retrieval": 0.15, "efficiency": 0.15},
                     "latency_reference_s": 10.0}


def composite(suite: dict, cfg: dict | None = None) -> dict:
    """Weighted mean of normalized components in [0, 1] (higher = better):
      answer_quality = mean(correct_answer_rate, F1, ROUGE-L)
      grounding      = mean(grounded_answer_rate, 1 - unsupported_answer_rate)
      retrieval      = mean(NDCG@10, Recall@10)                       (only if the system retrieves)
      efficiency     = ref / (ref + median latency)                    (ref = latency_reference_s)
    Missing components are DROPPED and the remaining weights renormalized (never scored as 0).
    ``shared`` uses only the components every system has (answer_quality, grounding, efficiency),
    which is the like-for-like comparison."""
    cfg = cfg or DEFAULT_COMPOSITE
    w, ref = cfg["weights"], float(cfg["latency_reference_s"])

    def val(*path):
        node = suite
        for p in path:
            node = node[p]
        return node["value"]

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    ua = val("answer", "unsupported_answer_rate")
    med = val("efficiency", "median_s")
    comp = {
        "answer_quality": avg([val("answer", "correct_answer_rate"), val("answer", "f1"), val("answer", "rouge_l")]),
        "grounding": avg([val("evidence", "grounded_answer_rate"), None if ua is None else 1 - ua]),
        "retrieval": avg([suite["retrieval"].get("ndcg@10", {}).get("value"),
                          suite["retrieval"].get("recall@10", {}).get("value")]),
        "efficiency": None if med is None else ref / (ref + med),
    }

    def combine(keys):
        used = [k for k in keys if comp[k] is not None]
        tw = sum(w[k] for k in used)
        return (None if not used else round(sum(w[k] * comp[k] for k in used) / tw, 4)), used

    full, used_full = combine(list(w))
    shared, used_shared = combine(["answer_quality", "grounding", "efficiency"])
    return {"components": {k: None if v is None else round(v, 4) for k, v in comp.items()},
            "composite_full": full, "components_used_full": used_full,
            "composite_shared": shared, "components_used_shared": used_shared,
            "weights": w, "latency_reference_s": ref}


def comparison_table(base: dict, enh: dict) -> list[dict]:
    """Rows: category, metric, baseline, agentmamba, difference (AgentMamba - Baseline, only where
    both are defined), better, n_baseline, n_agentmamba, definition."""
    rows = []
    for cat_key, cat in (("answer", "Answer generation"), ("errors", "Error analysis"), ("evidence", "Evidence"),
                         ("retrieval", "Retrieval"), ("efficiency", "Efficiency")):
        for name, m in enh[cat_key].items():
            if not isinstance(m, dict):
                continue
            b = base[cat_key].get(name, {"value": None, "n": 0})
            diff = None if b["value"] is None or m["value"] is None else round(m["value"] - b["value"], 4)
            rows.append({"category": cat, "metric": name, "baseline": b["value"], "agentmamba": m["value"],
                         "difference": diff, "better": m["better"], "n_baseline": b["n"], "n_agentmamba": m["n"],
                         "definition": m["definition"]})
    return rows
