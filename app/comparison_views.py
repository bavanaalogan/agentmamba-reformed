"""Dashboard views for the Baseline-Mamba vs AgentMamba comparison. Everything rendered here comes
from a live service call or from result files written by scripts/run_context_growth.py."""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from agentmamba.evaluation.comparison import matches
from agentmamba.evaluation.evidence_metrics import precision_at_k
from agentmamba.evaluation.metrics import token_f1
from agentmamba.evaluation.metrics_suite import ap_at_k, bleu1, hit_at, ndcg_at_k, recall_at, rouge1_f, rougeL_f

# Fixed condition order -> fixed categorical slot (colour follows the condition, never its rank).
# Validated (dataviz validate_palette.js): all checks pass in both modes.
CONDITIONS = ["baseline_mamba_768", "baseline_mamba_2048", "retrieval_always", "agentmamba",
              "agentmamba_no_verification", "agentmamba_no_retrieval", "agentmamba_vector_only",
              "agentmamba_lexical_only"]
PALETTE = {"dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
           "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]}


def _palette():
    try:
        mode = "dark" if (st.context.theme.type or "light") == "dark" else "light"
    except Exception:  # noqa: BLE001 - theme detection is a convenience only
        mode = "dark"
    return PALETTE[mode]


def _color(present: list[str]):
    dom = [c for c in CONDITIONS if c in present] + sorted(set(present) - set(CONDITIONS))
    pal = _palette()
    rng = [pal[CONDITIONS.index(c)] if c in CONDITIONS else "#888888" for c in dom]
    return alt.Color("condition:N", scale=alt.Scale(domain=dom, range=rng), legend=alt.Legend(title="condition",
                                                                                               orient="bottom",
                                                                                               columns=3))


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _render_structured_answer(out: dict) -> None:
    components = out.get("answer_components") or (out.get("verification") or {}).get("answer_components") or []
    if components:
        rows = [{
            "Requested information": item.get("question", "Answer"),
            "Answer": item.get("answer") or "Insufficient evidence",
            "Evidence records": ", ".join(item.get("record_ids") or []) or "None",
            "Evidence status": item.get("status", "n/a").replace("_", " ").title(),
        } for item in components]
    else:
        rows = [{
            "Requested information": "Response",
            "Answer": out.get("answer_value") or out.get("final_answer") or "No answer",
            "Evidence records": ", ".join(
                item.get("record_id", "") for item in out.get("evidence", []) if item.get("record_id")) or "None",
            "Evidence status": out.get("verification_status", "n/a").replace("_", " ").title(),
        }]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _comparison_metrics(outputs: dict, reference: str | None, required_ids: set[str]) -> pd.DataFrame:
    rows = []

    def add(category, metric, baseline, enhanced, notes):
        rows.append({"Category": category, "Metric": metric,
                     "Baseline Mamba": baseline, "AgentMamba": enhanced, "Interpretation": notes})

    answer_metrics = [
        ("Exact match", lambda pred, gold: float(matches(pred, gold, "exact")),
         "1 = normalized answer matches reference."),
        ("Answer error (1 - exact match)", lambda pred, gold: 1.0 - float(matches(pred, gold, "exact")),
         "0 = exact match; lower is better."),
        ("Token F1", token_f1, "Token overlap precision/recall F1; higher is better."),
        ("Token error (1 - F1)", lambda pred, gold: 1.0 - token_f1(pred, gold),
         "1 - token F1; lower is better."),
        ("ROUGE-1 F", rouge1_f, "Unigram overlap F-measure; higher is better."),
        ("ROUGE-L F", rougeL_f, "Longest-common-subsequence F-measure; higher is better."),
        ("BLEU-1", bleu1, "Clipped unigram precision with brevity penalty; higher is better."),
    ]
    for label, fn, note in answer_metrics:
        vals = {}
        for key in ("baseline", "enhanced"):
            out = outputs.get(key)
            answer = out.get("answer_value") if out and out.get("answer_status") == "answered" else ""
            vals[key] = "N/A" if reference is None or out is None else _fmt(fn(answer, reference), 4)
        add("Answer quality", label, vals["baseline"], vals["enhanced"],
            note if reference is not None else "N/A: provide/select an expected answer to score this output.")

    retrieval_metrics = [
        ("NDCG@10", lambda found, relevant: ndcg_at_k(found, relevant, 10)),
        ("MAP@10", lambda found, relevant: ap_at_k(found, relevant, 10)),
        ("Precision@10", lambda found, relevant: precision_at_k(found, relevant, 10)),
        ("Recall@10", lambda found, relevant: recall_at(found, relevant, 10)),
        ("Hit@1", lambda found, relevant: hit_at(found, relevant, 1)),
        ("Hit@3", lambda found, relevant: hit_at(found, relevant, 3)),
        ("Hit@10", lambda found, relevant: hit_at(found, relevant, 10)),
    ]
    for label, fn in retrieval_metrics:
        vals = {}
        for key in ("baseline", "enhanced"):
            out = outputs.get(key)
            retrieved = out.get("retrieved_record_ids", []) if out else []
            if not out or not required_ids:
                vals[key] = "N/A"
            elif not out.get("retrieval_triggered"):
                vals[key] = "N/A (no retrieval)"
            else:
                value = fn(retrieved, required_ids)
                vals[key] = "N/A" if value is None else _fmt(value, 4)
        add("Retrieval quality", label, vals["baseline"], vals["enhanced"],
            "Requires expected evidence record IDs; retrieval metrics are N/A when retrieval was not run.")

    for key in ("baseline", "enhanced"):
        out = outputs.get(key)
        conf = (out or {}).get("confidence") or {}
        if conf.get("calibrated") is not None:
            confidence, source = _fmt(conf["calibrated"], 4), "Calibrated confidence"
        elif conf.get("candidate_share") is not None:
            confidence, source = _fmt(conf["candidate_share"], 4), "Uncalibrated Mamba candidate share"
        elif conf.get("mean_component_candidate_share") is not None:
            confidence = _fmt(conf["mean_component_candidate_share"], 4)
            source = "Mean per-part candidate share (uncalibrated)"
        else:
            confidence, source = "N/A", "No numeric confidence score available"
        if not out:
            confidence, source = "N/A", "System did not return an answer"
        if key == "baseline":
            baseline_confidence, baseline_source = confidence, source
        else:
            enhanced_confidence, enhanced_source = confidence, source
    add("Confidence", "Answer confidence score", baseline_confidence, enhanced_confidence,
        "Calibrated when available; otherwise candidate share is uncalibrated, not probability of correctness "
        "(a single candidate makes the share trivially 1.0).")
    add("Confidence", "Score type", baseline_source, enhanced_source,
        "Do not compare calibrated and uncalibrated values as though they shared a scale.")
    for key in ("baseline", "enhanced"):
        out = outputs.get(key)
        if not out:
            value = "unavailable"
        elif key == "baseline" or out.get("verification_status") == "not_verified":
            value = "not verified"
        else:
            verification = out.get("verification") or {}
            part_statuses = {}
            for check in verification.get("candidate_checks", []):
                part = check.get("question_part")
                if part:
                    part_statuses[part] = check.get("part_status")
            if part_statuses:
                supported = sum(status == "supported" for status in part_statuses.values())
                value = f"{supported}/{len(part_statuses)} ({supported / len(part_statuses):.4f})"
            else:
                value = out.get("verification_status", "unavailable")
        if key == "baseline":
            base_status = value
        else:
            enhanced_status = value
    add("Confidence", "Verified answer-component coverage", base_status, enhanced_status,
        "Supported answer parts / total parts; a coverage ratio, not a probability of correctness.")

    timing_metrics = [
        ("Total time (s)", lambda out: (out.get("timings") or {}).get("total_s")),
        ("Answer generation time (s)", lambda out: (out.get("timings") or {}).get(
            "generation_s", (out.get("timings") or {}).get("initial_answer_s", 0)
            + (out.get("timings") or {}).get("final_answer_s", 0))),
        ("Retrieval time (s)", lambda out: (out.get("timings") or {}).get("retrieval_s")),
        ("CPU time (s)", lambda out: (out.get("resources") or {}).get("cpu_s")),
    ]
    for label, get_value in timing_metrics:
        vals = {}
        for key in ("baseline", "enhanced"):
            out = outputs.get(key)
            value = get_value(out) if out else None
            vals[key] = "N/A" if value is None else _fmt(value, 4)
        add("Efficiency", label, vals["baseline"], vals["enhanced"],
            "Single-query measurement; use repeated benchmark runs for averages or statistical comparison.")
    return pd.DataFrame(rows)


# ====================================================================== single question
def render_compare(svc) -> None:
    st.caption("Both systems receive the same question and initial context. Baseline Mamba ranks candidate answers "
               "from that context without retrieval or verification. AgentMamba may retrieve missing evidence and "
               "verifies its answer. Results here are a single-question diagnostic, not a benchmark.")
    cases = []
    try:
        cases = svc.protocol_cases() + svc.benchmark_cases()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Evaluation cases unavailable: {exc}")
    st.caption(f"Memory profile: `{svc.storage_dir}`")
    c = st.columns([3, 1])
    labels = ["(none - free question, no gold)"] + [
        (f"protocol {x['protocol']} | {x['question_id']} | {x['text']}" if x.get("protocol") else
         f"log benchmark {x['split']} | {x['question_id']} | {x['text']}") for x in cases]
    pick = c[0].selectbox("Optional evaluation case (protocol cases use THIS profile's memory; log-benchmark cases "
                          "use that scenario's stored snapshot)", labels)
    case = cases[labels.index(pick) - 1] if pick != labels[0] else None
    is_bench = bool(case and not case.get("protocol"))
    stage = c[1].selectbox("Stage" if is_bench else "Stage label (protocol: which stage you ingested)",
                           [1, 2, 3, 4], index=3 if is_bench else 0, disabled=case is None)
    q = st.text_input("Question", value=case["text"] if case else "",
                      placeholder="What vibration value triggered the high vibration alarm on COOLING-PUMP-17 on 2026-10-12?",
                      key=f"cmp_q_{case['question_id'] if case else 'free'}")   # per-case key: selecting a case loads its text
    c = st.columns(2)
    reference = c[0].text_input(
        "Expected answer (optional; scoring only, not sent to either system)",
        disabled=case is not None, key="cmp_reference")
    evidence_ids = c[1].text_input(
        "Expected evidence record IDs (optional, comma separated; scoring only)",
        disabled=case is not None, key="cmp_reference_evidence")
    c = st.columns(4)
    ctx_ids = c[0].text_input("Initial-context record IDs (optional)", key="cmp_ctx")
    window = c[1].selectbox("Shared recent-arrival window (Mamba tokens)", [2048, 768, 0], index=0,
                            help="Both systems get the same window. 2048 holds about 20 log lines (the stage-1 incident "
                                 "needs about 1,050 tokens). 0 = only the selected records.")
    mode = c[2].radio("AgentMamba retrieval mode", ["conditional", "always", "never"], horizontal=True, key="cmp_mode")
    verify = c[3].checkbox("Evidence verification (AgentMamba)", value=True, key="cmp_verify")
    baseline_mode = st.radio("Baseline Mamba answer mode",
                             ["candidate_ranking", "raw_generation"], index=0, horizontal=True,
                             format_func=lambda x: ("Candidate-span ranking (Mamba baseline)" if x == "candidate_ranking"
                                                    else "Raw Mamba generation (diagnostic)"),
                             help="The default Mamba baseline ranks answer spans independently for each question "
                                  "part using only the shared context, without retrieval or verification. Raw "
                                  "generation is available as a diagnostic but may be unreliable because the frozen "
                                  "base checkpoint is not instruction-tuned.",
                             key="cmp_baseline_mode_v2")
    if baseline_mode == "raw_generation":
        st.info("Mamba is frozen: uploaded records are included in the inference context, not learned into model "
                "weights. This base checkpoint is not instruction-tuned, so its free-form answer can be poor.")
    if not st.button("Run both systems") or not q:
        return
    ev_case = None
    if case and case.get("protocol"):
        ev_case = {"protocol": case["protocol"], "question_id": case["question_id"], "stage": stage}
    elif case:
        ev_case = {"split": case["split"], "question_id": case["question_id"], "stage": stage, "run": case["run"]}
    with st.spinner("Running Baseline Mamba and AgentMamba (CPU)..."):
        try:
            res = svc.compare(q, [i.strip() for i in ctx_ids.split(",") if i.strip()], window, mode, verify, ev_case,
                              baseline_mode=baseline_mode)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")
            return
    manual_evidence = {i.strip() for i in evidence_ids.split(",") if i.strip()}
    sc = res["shared_context"]
    st.write(f"**Shared initial context:** {sc['n_records']} records, {sc['prompt_tokens']} prompt tokens "
             f"(budget {sc['token_budget']}); policy: {res['context_policy']}.")
    m = res["models"]
    st.write(f"**Model:** `{m['mamba_id']}`, answerer `{m['answerer']}`, loaded: {m['mamba_loaded']}"
             + (f", load error: {m['mamba_load_error']}" if m["mamba_load_error"] else "")
             + (" (scores may come from the score cache without loading weights)" if not m["mamba_loaded"] else ""))
    ev = res.get("evaluation")
    left, right = st.columns(2)
    with left:
        st.subheader("A. Baseline Mamba")
        _baseline_card(res["baseline"], res["errors"].get("baseline"))
    with right:
        st.subheader("B. AgentMamba (enhanced)")
        _enhanced_card(svc, res["enhanced"], res["errors"].get("enhanced"))
    st.divider()
    st.subheader("Evaluation summary")
    st.caption("Single-question diagnostic. Answer metrics require an expected answer; retrieval metrics require "
               "relevant evidence IDs. N/A means a metric cannot be calculated for this run, not zero.")
    reference_value = (ev["gold"].get("expected_answer") if ev else reference.strip() or None)
    required_ids = (set(ev["gold"].get("required_evidence_ids") or []) if ev else manual_evidence)
    st.dataframe(_comparison_metrics({"baseline": res["baseline"], "enhanced": res["enhanced"]},
                                     reference_value, required_ids),
                 hide_index=True, width="stretch")
    if ev:
        st.caption("Gold reference is loaded for scoring only after both systems have run; it is not passed to either.")
    elif reference_value or required_ids:
        with st.expander("Scoring reference (not sent to either system)"):
            if reference_value:
                st.write("**Expected answer:**", reference_value)
            if required_ids:
                st.write("**Expected evidence record IDs:**", sorted(required_ids))
    st.caption("Use the staged benchmark for aggregate averages, calibration, confidence intervals, and statistical tests.")


def _baseline_card(out, err):
    if err:
        st.error(f"Baseline did not run: {err}")
        return
    with st.container(border=True):
        st.markdown("**Answer**")
        status = {"answered": st.success, "abstained": st.warning}.get(out["answer_status"], st.info)
        status(out.get("answer_status", "unknown").replace("_", " ").title())
        _render_structured_answer(out)
        st.caption("Baseline uses the shared initial context only; retrieval and evidence verification are disabled.")
        cols = st.columns(3)
        cols[0].metric("Context records", out["context"]["n_records"])
        cols[1].metric("Total time", f"{out['timings'].get('total_s', 0):.3f} s")
        cols[2].metric("CPU time", f"{out['resources'].get('cpu_s', 0):.3f} s")
        with st.expander("Baseline run details"):
            call = out.get("model_call", {})
            st.write(f"**Model:** {out.get('model')} · **Mode:** "
                     f"{'raw generation' if out.get('generation_mode') else 'candidate-span ranking'} · "
                     f"**Prompt records:** {call.get('prompt_records', 0)} · "
                     f"**Prompt tokens:** {call.get('prompt_tokens', 'n/a')}")
            st.write(f"**Generation time:** {out['timings'].get('generation_s', 0):.3f} s · "
                     f"**Post-answer audit (display only):** {(out.get('audit') or {}).get('status', 'n/a')}")
            if out.get("candidates"):
                st.markdown("**Ranked candidates**")
                st.dataframe(pd.DataFrame(out["candidates"]), hide_index=True, width="stretch")
            if out.get("evidence"):
                st.markdown("**Selected source records**")
                st.dataframe(pd.DataFrame(out["evidence"])[["record_id", "source", "timestamp", "excerpt"]],
                             hide_index=True, width="stretch")


def _enhanced_card(svc, out, err):
    if err:
        st.error(f"AgentMamba did not run: {err}")
        return
    with st.container(border=True):
        st.markdown("**Answer**")
        status = {"answered": st.success, "abstained": st.warning, "needs_clarification": st.info}[out["answer_status"]]
        status(out.get("answer_status", "unknown").replace("_", " ").title())
        _render_structured_answer(out)
        st.caption(f"Status: {out['answer_status']} · Evidence verification: {out.get('verification_status')} · "
                   f"Retrieval: {'used' if out.get('retrieval_triggered') else 'not used'}")
        suf = out.get("sufficiency") or {}
        cols = st.columns(4)
        cols[0].metric("Initial evidence", str(suf.get("decision", "n/a")).upper())
        cols[1].metric("Retrieval", "Used" if out.get("retrieval_triggered") else "Not used")
        cols[2].metric("Total time", f"{out['timings'].get('total_s', 0):.3f} s")
        cols[3].metric("CPU time", f"{out['resources'].get('cpu_s', 0):.3f} s")
        with st.expander("Investigation and evidence details"):
            st.write(f"**Initial answer:** {out.get('initial_answer') or 'No candidate answer'} "
                     f"({out.get('initial_status', 'n/a')})")
            st.write("**Retrieval decision:**",
                     suf.get("reason") or "; ".join((out.get("controller") or {}).get("reasons", []))
                     or out.get("no_retrieval_reason") or "n/a")
            st.write(f"**Causal claim status:** {out.get('causal_claim_status', 'n/a')}")
            if out.get("retrieval_triggered"):
                rs = out.get("retrieval_stats", {})
                st.write(f"**Selected records:** {rs.get('n_selected', len(out.get('retrieval', [])))} · "
                         f"**Channels:** {rs.get('channels', 'n/a')} · **Vector hits:** {rs.get('n_vector_hits', 'n/a')} · "
                         f"**Relationship hits:** {rs.get('n_expansion_hits', 'n/a')} · "
                         f"**Text hits:** {rs.get('n_text_hits', 'n/a')}")
                df = pd.DataFrame(out.get("retrieval", []))
                if len(df):
                    st.dataframe(df[["record_id", "similarity", "score", "sources", "explanation"]],
                                 hide_index=True, width="stretch")
            calls = out.get("model_calls", [])
            if calls:
                st.write("**Model calls:** " + "; ".join(
                    f"{c['stage']}: {c['n_candidates']} candidates ({c['seconds']} s)" for c in calls))
            if out.get("evidence"):
                st.markdown("**Cited source records**")
                evd = pd.DataFrame(out["evidence"])
                gate = []
                for rid in evd["record_id"]:
                    decisions = svc.memory.store.decisions(event_id=rid)
                    gate.append(f"{decisions[0]['decision']} ({decisions[0]['policy']})" if decisions
                                else "n/a (not in app memory)")
                evd["Memory gate"] = gate
                st.dataframe(evd[["record_id", "role", "source", "timestamp", "evidence_type", "excerpt", "Memory gate"]],
                             hide_index=True, width="stretch")
            elif out.get("abstention_reason"):
                st.write(f"**Abstention reason:** {out['abstention_reason']}")
            with st.expander("Full AgentMamba trace"):
                st.json(out)


# ====================================================================== experiments
def render_context_growth(results_dir: Path) -> None:
    runs = sorted([d for d in results_dir.glob("*context_growth*") if (d / "summary.json").exists()
                   and "protocol" not in d.name], reverse=True)       # protocol runs are shown in the metric suite
    if not runs:
        st.info("No context-growth results yet. Run scripts/run_context_growth.py --split dev, then --split test.")
        return
    run = st.selectbox("Context-growth run", runs, format_func=lambda p: p.name)
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    st.caption(f"Split **{s['split']}** ({'final reporting' if s['split'] == 'test' else 'development'}), answerer "
               f"**{s['answerer']}**, {s['n_questions']} questions x stages, scenarios {', '.join(s['scenarios'])}. "
               f"Synthetic log benchmark ({cfg['benchmark']['benchmark']}); fictional data. Each condition received "
               f"the same {cfg['initial_context']['token_budget']}-token recent-arrival window as initial context.")
    overall = pd.DataFrame([{"condition": k, **{m: v[m] for m in (
        "n", "accuracy", "grounded_accuracy", "answerable_accuracy", "correct_abstention_rate",
        "conflict_detection_rate", "incorrect_answer_rate", "unsupported_claim_rate", "evidence_in_context_rate",
        "required_evidence_coverage", "recall_at_k", "mrr", "latency_total_s_median_cold", "n_cold")},
                             "accuracy_ci95": v["accuracy_ci95"]} for k, v in s["by_condition"].items()])
    st.subheader("Overall (all stages pooled)")
    st.dataframe(overall, hide_index=True, width="stretch")
    ing = pd.DataFrame(s["ingestion"])
    avail = ing.groupby("stage")["records_available"].mean().round(0).to_dict() if len(ing) else {}
    long = []
    for cond, by_stage in s["by_condition_stage"].items():
        for st_, v in by_stage.items():
            long.append({"condition": cond, "stage": int(st_),
                         "stage_label": f"S{st_} ({int(avail.get(int(st_), 0))} rec.)", **v})
    df = pd.DataFrame(long)
    order = [f"S{k} ({int(avail.get(k, 0))} rec.)" for k in sorted(df["stage"].unique())]
    for metric, title, note in (
            ("accuracy", "Accuracy by stage", "higher is better"),
            ("evidence_in_context_rate", "Required evidence inside the initial-context window", "measured, per stage"),
            ("unsupported_claim_rate", "Unsupported claims among answers", "lower is better")):
        # In-run latencies are not charted: score-cache hits and host stalls distort them; see the
        # separate cold-cache benchmark table below.
        st.subheader(title)
        st.caption(f"{note}. x = ingestion stage (mean records available per scenario). Points are measured values; "
                   "lines only connect them.")
        sub = df[["condition", "stage_label", metric]].dropna()
        if sub.empty:
            st.write("no data")
            continue
        enc_y = alt.Y(f"{metric}:Q", title=metric.replace("_", " "),
                      scale=alt.Scale(domain=[0, 1]) if metric.endswith(("rate", "accuracy")) else alt.Undefined)
        ch = alt.Chart(sub).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
            x=alt.X("stage_label:N", sort=order, title="stage"), y=enc_y, color=_color(sub["condition"].unique().tolist()),
            tooltip=["condition", "stage_label", alt.Tooltip(f"{metric}:Q", format=".3f")])
        st.altair_chart(ch, width="stretch")
        with st.expander("Table view"):
            st.dataframe(sub.pivot(index="condition", columns="stage_label", values=metric)[order], width="stretch")
    st.subheader("Paired comparisons (B vs A, same question and stage)")
    comp = []
    for k, v in s["comparisons"].items():
        b, a = k.split("__vs__")
        for metric, r in v.items():
            if r.get("n_pairs"):
                comp.append({"B": b, "A": a, "outcome": metric + (" (lower better)" if r.get("direction") else ""),
                             "A rate": r["a"], "B rate": r["b"], "B-A": r["diff_b_minus_a"], "95% CI": r["diff_ci95"],
                             "McNemar p": r["mcnemar_p"], "pairs": r["n_pairs"]})
    st.dataframe(pd.DataFrame(comp), hide_index=True, width="stretch")
    st.caption("Questions within a scenario are correlated; CIs and p-values treat pairs as independent and are "
               "indicative only. No multiple-comparison correction.")
    st.subheader("Confidence calibration (answered questions)")
    cal_rows = []
    for k, v in s["calibration"].items():
        for key in ("raw_candidate_share", "calibrated"):
            c = v[key]
            cal_rows.append({"condition": k, "confidence": key, "n": c["n_answered_with_confidence"],
                             "Brier": c["brier"], "ECE": c["ece"], "mean conf.": c["mean_confidence"],
                             "accuracy": c["accuracy_when_answered"]})
    st.dataframe(pd.DataFrame(cal_rows), hide_index=True, width="stretch")
    st.caption(next(iter(s["calibration"].values()))["note"] if s["calibration"] else "")
    cond = st.selectbox("Reliability diagram for", list(s["calibration"]))
    rel = []
    for key in ("raw_candidate_share", "calibrated"):
        for b in s["calibration"][cond][key]["reliability"]:
            rel.append({"confidence": key, **b})
    if rel:
        rdf = pd.DataFrame(rel)
        pal = _palette()
        pts = alt.Chart(rdf).mark_circle(opacity=0.9).encode(
            x=alt.X("mean_confidence:Q", scale=alt.Scale(domain=[0, 1]), title="mean confidence in bin"),
            y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="accuracy in bin"),
            size=alt.Size("n:Q", title="answers in bin"),
            color=alt.Color("confidence:N", scale=alt.Scale(domain=["raw_candidate_share", "calibrated"],
                                                            range=pal[:2]), legend=alt.Legend(orient="bottom")),
            tooltip=["confidence", "bin", "n", "mean_confidence", "accuracy"])
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="#888888",
                                                                              strokeWidth=1).encode(x="x", y="y")
        st.altair_chart(diag + pts, width="stretch")
        st.dataframe(rdf, hide_index=True)
    else:
        st.write("No answered questions with a confidence value for this condition.")
    st.subheader("Ingestion per stage")
    st.dataframe(ing, hide_index=True, width="stretch")
    lat = run / "latency_cold.json"
    st.subheader("Cold-cache latency (separate sequential benchmark)")
    if lat.exists():
        L = json.loads(lat.read_text(encoding="utf-8"))
        st.caption(f"Stage {L['stage']} memory, {L['n_questions']} questions, Mamba score cache disabled, "
                   f"{L['threads']} threads, Mamba load {L['mamba_load_s']} s. {L['note']}.")
        st.dataframe(pd.DataFrame([{"system": k, **v} for k, v in L["summary"].items()]), hide_index=True,
                     width="stretch")
    else:
        st.write("Not run yet: scripts/benchmark_compare_latency.py --run " + str(run))
    rows_p = run / "rows.jsonl"
    if rows_p.exists():
        st.subheader("Per-question results")
        rows = pd.DataFrame([json.loads(x) for x in rows_p.read_text(encoding="utf-8").splitlines() if x])
        c = st.columns(3)
        f_cond = c[0].multiselect("condition", sorted(rows["system"].unique()))
        f_fam = c[1].multiselect("family", sorted(rows["family"].unique()))
        f_stage = c[2].multiselect("stage", sorted(rows["stage"].unique()))
        sel = rows
        if f_cond:
            sel = sel[sel["system"].isin(f_cond)]
        if f_fam:
            sel = sel[sel["family"].isin(f_fam)]
        if f_stage:
            sel = sel[sel["stage"].isin(f_stage)]
        st.dataframe(sel[["question_id", "stage", "system", "family", "answer_status", "answer_value", "expected",
                          "correct", "grounded_correct", "unsupported", "evidence_in_context", "retrieval_triggered",
                          "candidate_share", "calibrated_confidence", "total_s"]], hide_index=True, width="stretch")


# ====================================================================== comprehensive metrics
SYSTEM_COLORS = {"dark": ["#3987e5", "#d95926"], "light": ["#2a78d6", "#eb6834"]}   # validated slots 1-2


def _sys_color(names):
    try:
        mode = "dark" if (st.context.theme.type or "light") == "dark" else "light"
    except Exception:  # noqa: BLE001
        mode = "dark"
    return alt.Color("system:N", scale=alt.Scale(domain=list(names), range=SYSTEM_COLORS[mode][:len(names)]),
                     legend=alt.Legend(orient="bottom", title=None))


def _bars(df, title, names, y_title="value", domain01=True):
    df = df.dropna(subset=["value"]) if len(df) else df
    if df.empty:
        st.write(f"{title}: no defined values (N/A)")
        return
    st.markdown(f"**{title}**")
    ch = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("metric:N", title=None, sort=list(dict.fromkeys(df["metric"]))),
        xOffset="system:N",
        y=alt.Y("value:Q", title=y_title, scale=alt.Scale(domain=[0, 1]) if domain01 else alt.Undefined),
        color=_sys_color(names), tooltip=["system", "metric", alt.Tooltip("value:Q", format=".4f"), "n"])
    st.altair_chart(ch, width="stretch")


def render_metric_suite(results_dir: Path) -> None:
    from agentmamba.evaluation.metrics_suite import comparison_table

    cands = []
    for d in sorted(results_dir.glob("*"), reverse=True):
        if (d / "metric_suite.json").exists():
            cands.append((d, "benchmark"))
        elif "protocol" in d.name and (d / "summary.json").exists():
            cands.append((d, "protocol"))
    if not cands:
        st.info("No metric suites yet: run scripts/run_protocol.py, or scripts/compute_metric_suite.py --run <dir>.")
        return
    pick = st.selectbox("Run", cands, format_func=lambda c: f"{c[0].name}  [{c[1]}]", key="ms_run")
    d, kind = pick
    data = json.loads((d / ("metric_suite.json" if kind == "benchmark" else "summary.json")).read_text(encoding="utf-8"))
    suites, by_stage = data["suites"], data.get("suites_by_stage", {})
    if kind == "protocol":
        st.warning(data["diagnostic_note"])
        base_key, enh_key = "baseline_mamba", "agentmamba"
    else:
        st.caption(f"Benchmark run (held-out test split). Predictions produced by: {data.get('produced_by')}. "
                   f"{data.get('latency_note', '')}")
        bases = [s for s in data["systems"] if "baseline" in s]
        enhs = [s for s in data["systems"] if "baseline" not in s]
        c = st.columns(2)
        base_key = c[0].selectbox("Baseline system", bases, index=bases.index(data["baseline"]), key="ms_b")
        enh_key = c[1].selectbox("Enhanced system", enhs, index=enhs.index(data["enhanced"]), key="ms_e")
    b, a = suites[base_key], suites[enh_key]
    table = pd.DataFrame(comparison_table(b, a))
    comp = []
    for m, dfn in (("composite_shared", "like-for-like: answer quality, grounding, efficiency only"),
                   ("composite_full", "all components each system has; missing ones dropped, weights renormalized")):
        vb, va = b["composite"][m], a["composite"][m]
        comp.append({"category": "Composite score", "metric": m, "baseline": vb, "agentmamba": va,
                     "difference": None if vb is None or va is None else round(va - vb, 4), "better": "higher",
                     "n_baseline": None, "n_agentmamba": None, "definition": dfn})
    table = pd.concat([table, pd.DataFrame(comp)], ignore_index=True)
    shown = table.rename(columns={"baseline": "Baseline Mamba", "agentmamba": "AgentMamba",
                                  "difference": "Difference (A - B)"}).astype(object)
    for col in ("Baseline Mamba", "AgentMamba", "Difference (A - B)", "n_baseline", "n_agentmamba"):
        shown[col] = ["N/A" if v is None or v != v else (f"{v:.4f}" if isinstance(v, float) else str(v))
                      for v in shown[col]]
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption(f"Composite weights {a['composite']['weights']}, latency reference "
               f"{a['composite']['latency_reference_s']} s (efficiency = ref / (ref + median latency)). Components: "
               f"Baseline {b['composite']['components']}, AgentMamba {a['composite']['components']}. The composite "
               f"never replaces the individual metrics. Efficiency source: "
               f"{a['efficiency'].get('source', 'measured in this run')}.")
    names = ["Baseline Mamba", "AgentMamba"]

    def long(cat, metrics):
        rows = []
        for nm, s in zip(names, (b, a)):
            for m in metrics:
                x = s[cat].get(m)
                if isinstance(x, dict):
                    rows.append({"system": nm, "metric": m, "value": x["value"], "n": x["n"]})
        return pd.DataFrame(rows)

    c = st.columns(2)
    with c[0]:
        _bars(long("answer", ["exact_match", "f1", "rouge_1", "rouge_l", "bleu_1", "correct_answer_rate"]),
              "Answer quality (higher is better)", names)
    with c[1]:
        _bars(long("errors", ["answer_error_rate", "false_answer_rate", "abstention_error_rate",
                              "unsupported_claim_rate", "evidence_miss_rate"]), "Error rates (lower is better)", names)
    c = st.columns(2)
    with c[0]:
        _bars(pd.concat([long("evidence", ["grounded_answer_rate", "evidence_coverage"]),
                         long("retrieval", ["ndcg@10", "map@10", "recall@10", "recall@1", "hit@1", "hit@3",
                                            "hit@10"])]),
              "Evidence and retrieval (baseline retrieval is N/A and not plotted)", names)
    with c[1]:
        _bars(long("efficiency", ["median_s", "mean_s", "generation_mean_s", "retrieval_mean_s",
                                  "verification_mean_s"]),
              "Latency in seconds (lower is better; warm model, load time excluded)", names, "seconds", False)
    if by_stage.get(base_key) and by_stage.get(enh_key):
        rows = []
        for nm, key in zip(names, (base_key, enh_key)):
            for stg, su in by_stage[key].items():
                rows.append({"system": nm, "stage": f"stage {stg}",
                             "value": su["answer"]["correct_answer_rate"]["value"]})
        df = pd.DataFrame(rows).dropna()
        st.markdown("**Correct-answer rate as historical context grows (measured per stage)**")
        st.altair_chart(alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
            x=alt.X("stage:N", title=None),
            y=alt.Y("value:Q", title="correct-answer rate", scale=alt.Scale(domain=[0, 1])),
            color=_sys_color(names), tooltip=["system", "stage", alt.Tooltip("value:Q", format=".3f")]),
            width="stretch")
        st.dataframe(df.pivot(index="system", columns="stage", values="value"), width="stretch")
    st.markdown("**Confidence calibration**")
    st.write({nm: {"brier": s["errors"]["brier"]["value"], "ece": s["errors"]["ece"]["value"],
                   "n": s["errors"]["brier"]["n"]} for nm, s in zip(names, (b, a))})
    st.caption("N/A when no calibrated probabilities exist or fewer than 20 answered questions have them. "
               "Reliability diagrams for the benchmark are in the context-growth section below.")
    if kind == "protocol":
        st.markdown("**Per-question results (diagnostic)**")
        st.dataframe(pd.DataFrame(data["per_question"]), hide_index=True, width="stretch")
        st.write("**Time saved by skipping retrieval:**", data["retrieval_skip_savings"])
        st.write("**Cold start (excluded from query latency):**", data["cold_start"])
