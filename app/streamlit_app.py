"""AgentMamba - Ladle Incident Investigator (Streamlit dashboard).

    python scripts/launch_dashboard.py            # http://localhost:8501

Everything shown comes from the live service (SQLite/FAISS memory, trained artifacts) or from
result files on disk written by scripts/run_ladle_eval.py and scripts/benchmark_linear_time.py.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui_kit as ui  # noqa: E402
from agentmamba.config import load_config, resolve_path  # noqa: E402
from agentmamba.data import ladle_benchmark as lb  # noqa: E402
from agentmamba.data.document_loader import SUPPORTED_SUFFIXES  # noqa: E402
from agentmamba.pipeline.service import AgentService, ServiceError  # noqa: E402

st.set_page_config(page_title="AgentMamba · Ladle Investigator", page_icon="🔥", layout="wide")
ui.inject()

DATASET_LABEL = {"short": "Short history", "long": "Long history"}
STAGE_LABEL = {1: "Incident shift only", 2: "+ 3 days of operations", 3: "+ older history & manual",
               4: "+ 6 weeks of later cycles"}
TOPICS = {
    "Maintenance work done": ("What maintenance action was recorded for {L} on {D}?", False),
    "Alarm raised at a time": ("Which alarm was raised on {L} at {D} {T}?", True),
    "Shell temperature at a time": ("What was the shell temperature reading for {L} at {D} {T}?", True),
    "Inspection finding": ("What inspection finding was recorded for {L} on {D}?", False),
    "Confirmed root cause of a trip": ("What was the confirmed root cause of the trip of {L} on {D}?", False),
    "Cleared for service after repair?": ("Was {L} cleared for operation after the repair on {D}?", False),
}


@st.cache_resource
def service() -> AgentService:
    return AgentService(load_config(), os.environ.get("AGENTMAMBA_STORAGE") or None)


svc = service()
RESULTS = resolve_path(svc.cfg, "paths.results_dir")


# ====================================================================== shared helpers
def latest_eval() -> tuple[Path | None, dict | None]:
    runs = sorted((d for d in RESULTS.glob("*ladle_eval*") if (d / "summary.json").exists()), reverse=True)
    best = None
    for d in runs:
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        if s.get("answerer") == "mamba" and {"short", "long"} <= set(s["datasets"]):
            return d, s
        best = best or (d, s)
    return best if best else (None, None)


def linear_time() -> dict | None:
    p = RESULTS / "linear_time" / "linear_time.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def style(chart, height=260):
    return (chart.properties(height=height, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=ui.INK2, titleColor=ui.INK2, gridColor=ui.GRID, domainColor=ui.AXIS,
                            tickColor=ui.AXIS, labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(labelColor=ui.INK2, titleColor=ui.INK2, orient="bottom", labelFontSize=12)
            .configure_title(color=ui.INK, fontSize=14, anchor="start"))


SYS_SCALE = alt.Scale(domain=["Baseline Mamba", "AgentMamba"], range=[ui.BASE_C, ui.AGENT_C])


def source_picker() -> dict | None:
    """Sidebar: which memory to investigate. Returns a ladle source dict or None (= uploaded logs)."""
    st.sidebar.markdown("### 🗄️ Memory to investigate")
    sources = svc.ladle_sources()
    kinds = (["Ladle dataset"] if sources else []) + ["My uploaded logs"]
    kind = st.sidebar.radio("Memory", kinds, label_visibility="collapsed", key="src_kind")
    if kind == "My uploaded logs":
        st.sidebar.caption("Logs you added on the **Upload logs** page.")
        return None
    dsets = sorted({s["dataset"] for s in sources}, key=lambda d: lb.DATASETS.index(d))
    ds = st.sidebar.radio("History length", dsets, format_func=lambda d: DATASET_LABEL[d], horizontal=True,
                          key="src_ds")
    scen = [s for s in sources if s["dataset"] == ds]
    sc = st.sidebar.selectbox("Melt shop (scenario)", [s["scenario"] for s in scen], key="src_sc")
    stages = next(s["stages"] for s in scen if s["scenario"] == sc)
    stage = stages[-1]
    if len(stages) > 1:
        stage = st.sidebar.select_slider("How much history is loaded", stages, value=stages[-1],
                                         format_func=lambda k: f"{k}: {STAGE_LABEL.get(k, k)}", key="src_stage")
    try:
        n = svc.ladle_memory(ds, sc, stage).store.conn.execute(
            "SELECT COUNT(*) FROM events WHERE status='active'").fetchone()[0]
        st.sidebar.markdown(f'<span class="pill"><span class="dot" style="background:{ui.AGENT_C}"></span>'
                            f'{n:,} records in memory</span>', unsafe_allow_html=True)
    except ServiceError as exc:
        st.sidebar.error(str(exc))
    return {"dataset": ds, "scenario": sc, "stage": stage}


def memory_for(src):
    return svc.ladle_memory(src["dataset"], src["scenario"], src["stage"]) if src else svc.memory


# ====================================================================== pages
def page_home():
    ui.hero("Agentic Mamba · PSG iTech · AI & DS",
            "Ladle Incident Investigator",
            "Ask about a ladle's history in plain words. AgentMamba keeps the plant's logs in an external "
            "episodic memory, retrieves only the records that matter, lets the pretrained Mamba model pick the "
            "answer, and shows the source records so an engineer can check every finding.")
    ui.flow([("Operational logs", "Alarms, historian, maintenance, inspections, RCA, shift notes", False),
             ("Memory gate", "Scores each record; keeps the useful ones within a budget", True),
             ("Episodic memory", "SQLite records + FAISS vectors, persistent", True),
             ("Selective retrieval", "Only when the recent window can't answer", True),
             ("Mamba-130m", "Ranks candidate answers in linear time", False),
             ("Evidence-linked finding", "Verified answer with source records, or a safe refusal", True)])
    st.caption("Orange-bordered steps are what AgentMamba adds around the frozen Mamba model.")

    run_dir, s = latest_eval()
    st.markdown('<div class="section-title">What the experiments show</div>', unsafe_allow_html=True)
    if not s:
        st.info("No evaluation results yet. Run `python scripts/run_ladle_eval.py` to fill this page.")
    else:
        c = st.columns(3)
        for col, ds in zip(c[:2], ("short", "long")):
            d = s["datasets"].get(ds)
            if d:
                b, a = d["by_system"]["baseline_mamba"], d["by_system"]["agentmamba"]
                with col:
                    ui.vs_tile(f"{DATASET_LABEL[ds]}: answerable questions answered correctly",
                               ui.pct(b["answerable_accuracy"]), ui.pct(a["answerable_accuracy"]),
                               f"{int(b['records_available_mean'] or 0):,} records in memory · "
                               f"{b['n']} questions")
        lt = linear_time()
        with c[2]:
            if lt:
                f = lt["part_a"]["fit"]
                ui.tile("Mamba time grows linearly with input", f"R² = {f['r2']:.3f}",
                        f"{f['seconds_per_token'] * 1000:.2f} ms per extra token on this CPU; AgentMamba keeps "
                        f"Mamba's input bounded")
            else:
                ui.tile("Linear-time check", "not run", "python scripts/benchmark_linear_time.py")
    st.markdown('<div class="section-title">How to use this app</div>', unsafe_allow_html=True)
    c = st.columns(3)
    with c[0]:
        ui.tile("1 · Pick a memory", "🗄️", "In the sidebar, choose the short or long ladle dataset, or your own "
                "uploaded logs.")
    with c[1]:
        ui.tile("2 · Investigate", "🔍", "Pick a ladle, a date and what you want to know. Both systems answer "
                "side by side.")
    with c[2]:
        ui.tile("3 · Check the evidence", "📄", "Every AgentMamba answer cites the source records, or refuses "
                "when the logs don't support an answer.")
    st.caption("Research prototype for decision support. Not safety-certified; it never authorizes production or "
               "safety actions. All ladle datasets are fictional synthetic data.")


def _ladles(mem) -> list[str]:
    rows = mem.store.conn.execute("SELECT DISTINCT asset_id FROM events WHERE status='active' AND asset_id IS NOT "
                                  "NULL").fetchall()
    ids = sorted({r[0] for r in rows})
    return sorted(ids, key=lambda x: (not x.upper().startswith("LADLE"), x))


def page_investigate():
    src = source_picker()
    ui.hero("Investigation", "Ask about a ladle incident",
            "Both systems get the same question and the same most recent log records. Baseline Mamba answers "
            "from that window alone. AgentMamba can also search the episodic memory and verifies before it "
            "answers.")
    try:
        mem = memory_for(src)
    except ServiceError as exc:
        st.error(str(exc))
        return
    ladles = _ladles(mem)
    default_day = date.today()
    if src:
        default_day = date.fromisoformat(lb.scenario_manifest(src["dataset"], src["scenario"])["incident_day"])
    tabs = st.tabs(["🧭 Guided", "✍️ In your own words"] + (["📋 Sample questions"] if src else []))
    with tabs[0]:
        c = st.columns([1.2, 1, 0.8, 1.6])
        ladle = c[0].selectbox("Ladle / equipment ID", ladles or ["LADLE-07"], key="g_ladle")
        day = c[1].date_input("Date", value=default_day, key="g_day")
        hhmm = c[2].text_input("Time (HH:MM)", value="10:30", key="g_time")
        topic = c[3].selectbox("What do you want to know?", list(TOPICS), key="g_topic")
        tmpl, needs_time = TOPICS[topic]
        q_guided = tmpl.format(L=ladle, D=day.isoformat(), T=hhmm.strip())
        st.markdown(f'<div class="note">Question sent to both systems: <b style="color:{ui.INK}">{ui.esc(q_guided)}'
                    f'</b>{"" if needs_time else " (time not used for this topic)"}</div>', unsafe_allow_html=True)
        if st.button("🔍 Investigate", type="primary", key="go_g"):
            st.session_state["inv_q"] = q_guided
            st.session_state["inv_run"] = True
    with tabs[1]:
        q_free = st.text_input("Your question", placeholder=f"e.g. What maintenance action was recorded for "
                                                            f"{ladles[0] if ladles else 'LADLE-07'} on "
                                                            f"{default_day.isoformat()}?", key="free_q")
        st.caption("Name the ladle ID and the date (and a time for alarms or readings) so the answer can be "
                   "checked against the records.")
        if st.button("🔍 Investigate", type="primary", key="go_f", disabled=not q_free.strip()):
            st.session_state["inv_q"] = q_free.strip()
            st.session_state["inv_run"] = True
    if src:
        with tabs[2]:
            qs = svc.ladle_questions(src["dataset"], src["scenario"])
            pick = st.selectbox("Questions from the dataset (gold answers are used only to score the result)",
                                [q["text"] for q in qs], key="sample_q")
            if st.button("🔍 Investigate", type="primary", key="go_s"):
                st.session_state["inv_q"] = pick
                st.session_state["inv_run"] = True
    with st.expander("⚙️ Advanced settings"):
        c = st.columns(3)
        window = c[0].select_slider("Recent-log window shared by both systems (Mamba tokens)",
                                    [512, 768, 1024, 2048, 4096], value=2048, key="adv_win")
        mode = c[1].radio("AgentMamba retrieval", ["conditional", "always", "never"], horizontal=True,
                          format_func={"conditional": "only when needed", "always": "always",
                                       "never": "never"}.get, key="adv_mode")
        verify = c[2].toggle("Verify evidence before answering", value=True, key="adv_verify")
        ref = None
        if not src:
            r = st.text_input("Expected answer (optional - used only to mark the answers correct/incorrect)",
                              key="adv_ref")
            if r.strip():
                ref = {"answer_kind": "value", "expected_answer": r.strip(), "match": "contains"}

    if st.session_state.pop("inv_run", False):
        q = st.session_state["inv_q"]
        with st.spinner("Running Baseline Mamba and AgentMamba on CPU…"):
            try:
                st.session_state["inv_res"] = svc.compare(q, [], window, mode, verify, source=src, reference=ref)
                st.session_state["inv_src"] = src
            except Exception as exc:  # noqa: BLE001 - shown, never hidden
                st.error(f"{type(exc).__name__}: {exc}")
                return
    res = st.session_state.get("inv_res")
    if res and st.session_state.get("inv_src") == src:
        render_result(res, mem)


def _mamba_tokens(out: dict | None, baseline: bool) -> str:
    if not out:
        return "n/a"
    if baseline:
        return f"{out.get('context', {}).get('prompt_tokens', 0):,}"
    calls = out.get("model_calls") or []
    toks = [c.get("prompt_tokens") for c in calls if c.get("prompt_tokens")]
    if toks:
        return f"{max(toks):,}"
    return "cached" if calls else "0"


def _answer_html(out, baseline: bool) -> str:
    if not out:
        return '<div class="answer none">—</div>'
    comps = out.get("answer_components") or []
    if out.get("answer_status") == "answered" and out.get("answer_value"):
        val = out["answer_value"]
        if comps and len(comps) > 1:
            val = "<br>".join(f"{ui.esc(c.get('question'))}: <b>{ui.esc(c.get('answer'))}</b>" for c in comps)
            return f'<div class="answer">{val}</div>'
        return f'<div class="answer">{ui.esc(val)}</div>'
    if not baseline and out.get("verification_status") == "conflict":
        return '<div class="answer">Two sources report different values - see the evidence below.</div>'
    reason = out.get("abstention_reason") or out.get("final_answer") or "No answer"
    return f'<div class="answer none">{ui.esc(reason[:260])}</div>'


def render_result(res: dict, mem) -> None:
    base, enh = res.get("baseline"), res.get("enhanced")
    ev = res.get("evaluation") or {}
    sc = res["shared_context"]
    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="note">Question: <b style="color:{ui.INK}">{ui.esc(res["question"])}</b><br>'
                f'Memory holds <b>{res.get("memory_records", 0):,}</b> records. Both systems first saw the same '
                f'<b>{sc["n_records"]}</b> most recent records ({sc["prompt_tokens"]:,} tokens).</div>',
                unsafe_allow_html=True)
    if ev.get("gold"):
        g = ev["gold"]
        exp = ("no answer: the logs don't establish one (should refuse)" if g["answer_kind"] == "abstain"
               else "flag the conflict between sources" if g["answer_kind"] == "conflict" else g["expected_answer"])
        st.markdown(f'<div class="note">Expected ({ui.esc(g.get("source", "gold"))}): '
                    f'<b style="color:{ui.INK}">{ui.esc(exp)}</b></div>', unsafe_allow_html=True)
    c = st.columns(2, gap="medium")
    for col, out, is_base, key in ((c[0], base, True, "baseline"), (c[1], enh, False, "enhanced")):
        err = res.get("errors", {}).get(key)
        score = ev.get(key)
        t = (out or {}).get("timings", {}).get("total_s")
        calls = ((out or {}).get("model_calls") or
                 ([out["model_call"]] if (out or {}).get("model_call") else []))
        cached = bool(calls) and all(c.get("cache_hit") for c in calls)
        facts = [ui.badge(f"⏱ {t:.2f} s" if t is not None else "⏱ n/a")]
        facts.append(ui.badge("⚡ Asked before: Mamba scores reused from cache (time not representative)") if cached
                     else ui.badge(f"🧠 Mamba read {_mamba_tokens(out, is_base)} tokens"))
        if not is_base and out:
            facts.append(ui.badge("🔎 Searched memory: " + (f"{len(out.get('retrieved_record_ids', []))} records found"
                                                            if out.get("retrieval_triggered") else "not needed")))
        name = "Baseline Mamba" if is_base else "AgentMamba"
        sub = ("Recent window only · no retrieval · no verification" if is_base
               else "Recent window + episodic memory · verified")
        with col:
            st.markdown(
                f'<div class="sys {"base" if is_base else "agent"}"><div class="head"><div><div class="name">{name}'
                f'</div><div class="sub">{sub}</div></div><div>{ui.correctness_badge(score)}</div></div>'
                f'{_answer_html(out, is_base) if not err else f"<div class=answer>{ui.esc(err)}</div>"}'
                f'<div>{ui.status_badge(out, is_base)}</div><div class="facts">{"".join(facts)}</div></div>',
                unsafe_allow_html=True)
    tabs = st.tabs(["🧭 How AgentMamba got there", "📄 Evidence records", "🧾 Raw trace"])
    with tabs[0]:
        render_timeline(enh, sc, res.get("memory_records", 0))
    with tabs[1]:
        c = st.columns(2, gap="medium")
        for col, out, title in ((c[0], base, "Baseline Mamba picked its answer from"),
                                (c[1], enh, "AgentMamba cites")):
            with col:
                st.markdown(f"**{title}**")
                evid = (out or {}).get("evidence") or []
                if not evid:
                    st.caption("No source record.")
                html = []
                for e in evid[:8]:
                    full = mem.store.get_event(e["record_id"])
                    html.append(ui.record_card(e["record_id"], (e.get("source") or "").replace("file://", ""),
                                               e.get("timestamp"), full.text if full else e.get("excerpt", ""),
                                               e.get("role")))
                st.markdown("".join(html), unsafe_allow_html=True)
    with tabs[2]:
        st.json(res, expanded=False)


def render_timeline(out: dict | None, sc: dict, n_mem: int) -> None:
    if not out:
        st.caption("AgentMamba did not run.")
        return
    suf = out.get("sufficiency") or {}
    steps = [("Read the recent log window", f"{sc['n_records']} records, {sc['prompt_tokens']:,} tokens - the same "
              f"window Baseline Mamba used.", True)]
    init = out.get("initial_answer")
    steps.append(("Evidence check on the window",
                  (f"Initial candidate: “{init}”. " if init else "No candidate answer in the window. ")
                  + f"Decision: window is {suf.get('decision', 'n/a')}."
                  + (f" ({suf.get('reason')})" if suf.get("reason") else ""), True))
    if out.get("retrieval_triggered"):
        rs = out.get("retrieval_stats", {})
        rt = out.get("timings", {}).get("retrieval_s", 0)
        steps.append(("Selective retrieval from episodic memory",
                      f"Searched {n_mem:,} stored records in {rt * 1000:.0f} ms → kept {rs.get('n_selected', 0)} "
                      f"(semantic matches {rs.get('n_vector_hits', 0)}, ID/text matches {rs.get('n_text_hits', 0)}, "
                      f"linked records {rs.get('n_expansion_hits', 0)}).", True))
    else:
        steps.append(("Selective retrieval skipped", out.get("no_retrieval_reason") or
                      "The window already held verified evidence, so memory was not searched.", False))
    calls = out.get("model_calls") or []
    if calls:
        steps.append(("Mamba ranks candidate answers",
                      "; ".join(f"{c['stage'].replace('_', ' ')}: {c['n_candidates']} candidates over "
                                f"{c.get('prompt_records', '?')} records ({c['seconds']:.2f} s)" for c in calls), True))
    ver = out.get("verification_status")
    vtext = {"supported": "Identity, date/time and source checks passed - answer is backed by the cited records.",
             "conflict": "Two records disagree for the same ladle and time - both are shown instead of guessing.",
             "insufficient": "The records do not establish an answer - AgentMamba refuses rather than guess.",
             "unverified": "Answer returned without full verification.",
             "not_verified": "Verification was switched off."}.get(ver, str(ver))
    reason = out.get("abstention_reason")
    steps.append(("Verification", vtext + (f" Reason: {reason}." if reason and ver != "supported" else ""), True))
    html = "".join(f'<div class="ev{"" if on else " skip"}"><div class="t">{ui.esc(t)}</div>'
                   f'<div class="d">{ui.esc(d)}</div></div>' for t, d, on in steps)
    st.markdown(f'<div class="timeline">{html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------- results
def page_results():
    ui.hero("Results", "Short context vs long context",
            "Same ladle questions, same gold answers - only the amount of log history changes. Short: the incident "
            "shift fits inside Mamba's window. Long: thousands of later records push the evidence out of it.")
    run_dir, s = latest_eval()
    if not s:
        st.info("No results yet. Run `python scripts/generate_ladle_datasets.py` then "
                "`python scripts/run_ladle_eval.py`.")
    else:
        st.caption(f"Run `{run_dir.name}` · answerer: {s['answerer']} · shared recent window: {s['window_tokens']} "
                   f"Mamba tokens · synthetic, fictional data.")
        rows = []
        metrics = [("answerable_accuracy", "Correct answers (answerable questions)"),
                   ("correct_abstention_rate", "Correct refusals (unanswerable questions)"),
                   ("required_evidence_coverage", "Required evidence seen by the model"),
                   ("incorrect_answer_rate", "Wrong answers given (lower is better)")]
        for ds, d in s["datasets"].items():
            for key, nm in (("baseline_mamba", "Baseline Mamba"), ("agentmamba", "AgentMamba")):
                v = d["by_system"][key]
                for m, label in metrics:
                    rows.append({"dataset": DATASET_LABEL[ds], "system": nm, "metric": label, "value": v[m],
                                 "n": v["n"]})
        df = pd.DataFrame(rows)
        ui.legend()
        c = st.columns(2, gap="large")
        for i, (m, label) in enumerate(metrics):
            sub = df[df["metric"] == label].dropna(subset=["value"])
            enc = dict(x=alt.X("dataset:N", title=None, sort=["Short history", "Long history"],
                               axis=alt.Axis(labelAngle=0)),
                       xOffset=alt.XOffset("system:N", sort=["Baseline Mamba", "AgentMamba"]))
            bars = alt.Chart(sub, title=label).mark_bar(cornerRadiusEnd=4, size=26).encode(
                y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1.08]), axis=alt.Axis(format="%")),
                color=alt.Color("system:N", scale=SYS_SCALE, legend=None),
                tooltip=["dataset", "system", alt.Tooltip("value:Q", format=".0%", title=label), "n"], **enc)
            text = alt.Chart(sub).mark_text(dy=-8, color=ui.INK2, fontSize=11).encode(
                y="value:Q", text=alt.Text("value:Q", format=".0%"), **enc)
            with c[i % 2]:
                st.altair_chart(style(bars + text, 240), use_container_width=True)
        with st.expander("Table view"):
            st.dataframe(df.pivot_table(index=["dataset", "system"], columns="metric", values="value")
                         .map(lambda v: f"{100 * v:.0f}%"), use_container_width=True)

        st.markdown('<div class="section-title">Accuracy as the ladle history grows</div>', unsafe_allow_html=True)
        st.caption("Stage 1 is the short dataset; stages 2-4 add history to the long dataset. Points are measured; "
                   "lines only connect them.")
        g = []
        for ds, d in s["datasets"].items():
            for key, nm in (("baseline_mamba", "Baseline Mamba"), ("agentmamba", "AgentMamba")):
                for stg, v in d["by_stage"][key].items():
                    g.append({"system": nm, "stage": int(stg), "records": v["records_available_mean"],
                              "answerable accuracy": v["answerable_accuracy"],
                              "label": f"S{stg}: {int(v['records_available_mean'] or 0):,} records"})
        gd = pd.DataFrame(g).sort_values("stage").drop_duplicates(["system", "stage"])
        order = list(dict.fromkeys(gd["label"]))
        ch = alt.Chart(gd).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=80, filled=True)).encode(
            x=alt.X("label:N", sort=order, title="history loaded into memory", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("answerable accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                    title="correct answers"),
            color=alt.Color("system:N", scale=SYS_SCALE, legend=None),
            tooltip=["system", "label", alt.Tooltip("answerable accuracy:Q", format=".0%")])
        ui.legend()
        st.altair_chart(style(ch, 260), use_container_width=True)

        rows_p = run_dir / "rows.jsonl"
        if rows_p.exists():
            with st.expander("Every question, every system"):
                rr = pd.DataFrame([json.loads(x) for x in rows_p.read_text(encoding="utf-8").splitlines() if x])
                rr["system"] = rr["system"].map({"baseline_mamba": "Baseline Mamba", "agentmamba": "AgentMamba"})
                f = st.columns(3)
                fd = f[0].multiselect("Dataset", sorted(rr["dataset"].unique()))
                fs = f[1].multiselect("System", sorted(rr["system"].unique()))
                ff = f[2].multiselect("Question type", sorted(rr["family"].unique()))
                sel = rr
                for col, vals in (("dataset", fd), ("system", fs), ("family", ff)):
                    if vals:
                        sel = sel[sel[col].isin(vals)]
                st.dataframe(sel[["dataset", "stage", "question", "system", "answer_value", "expected", "correct",
                                  "retrieval_triggered", "total_s"]], hide_index=True, use_container_width=True)

    st.markdown('<div class="section-title">Is Mamba still linear-time inside AgentMamba?</div>',
                unsafe_allow_html=True)
    lt = linear_time()
    if not lt:
        st.info("Run `python scripts/benchmark_linear_time.py` to measure this.")
        return
    a = lt["part_a"]
    f = a["fit"]
    pa = pd.DataFrame(a["points"])
    pb = pd.DataFrame(lt["part_b"]["stages"])
    c = st.columns(3)
    with c[0]:
        ui.tile("Linear fit of Mamba time vs tokens", f"R² = {f['r2']:.3f}",
                f"time ≈ {f['seconds_per_token'] * 1000:.2f} ms × tokens {'+' if f['intercept_s'] >= 0 else '−'} "
                f"{abs(f['intercept_s']):.2f} s on {lt['threads']} CPU threads")
    if len(pb):
        last = pb.iloc[-1]
        with c[1]:
            ui.tile("Tokens Mamba reads per AgentMamba query", f"~{int(pb['agent_mamba_tokens_median'].median()):,}",
                    f"typical (max {int(pb['agent_mamba_tokens_max'].max()):,}); stays bounded while the history "
                    f"grows to {int(last['history_tokens']):,} tokens")
        with c[2]:
            ui.tile("Query time with the full long history", f"{last['agent_total_s_median']:.1f} s",
                    f"vs ~{last['full_history_mamba_s_projected']:.0f} s projected for Mamba reading all "
                    f"{int(last['records']):,} records")
    c = st.columns(2, gap="large")
    with c[0]:
        fit = pd.DataFrame({"tokens": [pa["tokens"].min(), pa["tokens"].max()]})
        fit["seconds"] = f["seconds_per_token"] * fit["tokens"] + f["intercept_s"]
        pts = alt.Chart(pa, title="Mamba forward pass: time vs input length").mark_circle(
            size=90, color=ui.BASE_C, opacity=1).encode(
            x=alt.X("tokens:Q", title="input tokens"), y=alt.Y("seconds:Q", title="seconds"),
            tooltip=["tokens", alt.Tooltip("seconds:Q", format=".3f")])
        line = alt.Chart(fit).mark_line(color=ui.MUTED, strokeDash=[5, 4], strokeWidth=2).encode(
            x="tokens:Q", y="seconds:Q")
        st.altair_chart(style(line + pts, 280), use_container_width=True)
        st.caption("Dots: measured (median of repeated runs). Dashed line: least-squares linear fit.")
    with c[1]:
        if len(pb):
            long_ = pd.concat([
                pd.DataFrame({"records": pb["records"], "seconds": pb["agent_total_s_median"],
                              "system": "AgentMamba (measured)"}),
                pd.DataFrame({"records": pb["records"], "seconds": pb["full_history_mamba_s_projected"],
                              "system": "Mamba reading the whole history (projected)"})])
            dom = ["Mamba reading the whole history (projected)", "AgentMamba (measured)"]
            ch = alt.Chart(long_, title="Time per question as history grows").mark_line(
                strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True)).encode(
                x=alt.X("records:Q", title="records in memory"), y=alt.Y("seconds:Q", title="seconds per question"),
                color=alt.Color("system:N", scale=alt.Scale(domain=dom, range=[ui.BASE_C, ui.AGENT_C]), title=None),
                strokeDash=alt.StrokeDash("system:N", scale=alt.Scale(domain=dom, range=[[5, 4], [1, 0]]),
                                          legend=None),
                tooltip=["system", "records", alt.Tooltip("seconds:Q", format=".2f")])
            st.altair_chart(style(ch, 280), use_container_width=True)
            st.caption("AgentMamba: measured, Mamba score cache off. Blue dashed line: NOT run - the time Mamba "
                       "would need to read the whole history, from the linear fit on the left.")
    with st.expander("Table view"):
        st.dataframe(pa, hide_index=True)
        if len(pb):
            st.dataframe(pb, hide_index=True)


# ---------------------------------------------------------------------- upload
def page_upload():
    ui.hero("Upload logs", "Add your own ladle logs to memory",
            "CSV, JSON or JSONL logs (alarm journals, historian exports, maintenance logs) and TXT / MD documents "
            "(inspection reports, shift notes). They go into the 'My uploaded logs' memory.")
    up = st.file_uploader("Drop a log file or document", type=sorted(s.strip(".") for s in SUPPORTED_SUFFIXES))
    c = st.columns(2)
    stage = c[0].selectbox("Plant area of this file", ["LADLE", "MAINTENANCE", "STEELMAKING", "CASTING", "QUALITY",
                                                        "UTILITIES"])
    policy = c[1].radio("What to keep", ["Keep every record (recommended)", "Let the memory gate decide"],
                        help="The learned memory gate was trained on the synthetic steel benchmark; it is not "
                             "validated on arbitrary uploaded logs.")
    st.caption("Columns named timestamp, equipment / ladle / heat / work_order, signal, value, unit are recognised "
               "automatically; every original column is kept and searchable.")
    if up is not None and st.button("📥 Add to memory", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / Path(up.name).name
            p.write_bytes(up.getvalue())
            try:
                with st.spinner("Parsing, embedding and storing…"):
                    rep = svc.ingest_file(p, stage, "authorized", None,
                                          "store_all" if policy.startswith("Keep") else None)
            except (ServiceError, ValueError, TypeError, KeyError) as exc:
                st.error(str(exc))
                return
        c = st.columns(4)
        with c[0]:
            ui.tile("Records read", f"{rep['received']:,}")
        with c[1]:
            ui.tile("Stored in memory", f"{rep['written']:,}")
        with c[2]:
            ui.tile("Already stored (duplicates)", f"{rep['duplicates']:,}")
        with c[3]:
            ui.tile("Time", f"{rep['timings'].get('total_s', 0):.1f} s")
        st.success("Done. Choose **My uploaded logs** in the sidebar on the Investigate page to ask about it.")
        prev = pd.DataFrame(rep["normalized_preview"])
        st.dataframe(prev[[c for c in ("event_id", "timestamp", "asset_id", "event_type", "text") if c in prev]],
                     hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------- explorer
def page_explorer():
    src = source_picker()
    ui.hero("Memory explorer", "What is stored in episodic memory",
            "Browse the records the investigator can retrieve. Each record keeps its source file and row.")
    try:
        mem = memory_for(src)
    except ServiceError as exc:
        st.error(str(exc))
        return
    s = mem.stats()
    c = st.columns(4)
    with c[0]:
        ui.tile("Records", f"{s['records']:,}")
    with c[1]:
        ui.tile("Vectors in FAISS", f"{s['vectors']:,}")
    with c[2]:
        ui.tile("Documented links", f"{s['relationships']:,}")
    with c[3]:
        ui.tile("Storage", f"{s['record_bytes'] / 1e6:.1f} MB")
    c = st.columns([1, 1, 2])
    lad = c[0].selectbox("Ladle / equipment", ["(all)"] + _ladles(mem))
    day = c[1].text_input("Date (YYYY-MM-DD)")
    txt = c[2].text_input("Text contains", placeholder="e.g. slide gate")
    evs = mem.store.query_events(asset_id=None if lad == "(all)" else lad, start=day or None,
                                 end=(day + "T23:59:59") if day else None, text_like=txt or None, limit=300)
    st.caption(f"Showing {len(evs)} records (max 300).")
    st.dataframe(pd.DataFrame([{"time": e.timestamp, "equipment": e.asset_id, "type": e.event_type,
                                "record": e.text, "source": (e.source_ref or "").replace("file://", ""),
                                "id": e.event_id} for e in evs]), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------- system
def page_system():
    ui.hero("System", "Components and health", "What is loaded, where memory lives, and the run history.")
    s = svc.status()
    c = st.columns(4)
    with c[0]:
        ui.tile("Uploaded-logs memory", f"{s.get('sqlite', {}).get('records', 0):,}", "records (SQLite)")
    with c[1]:
        ui.tile("Memory gate", "loaded" if s["gate"].get("present") else "missing",
                f"version {s['gate'].get('version')}" if s["gate"].get("present") else "run scripts/train_gate.py")
    with c[2]:
        ui.tile("Retrieval controller", "loaded" if s["controller"]["present"] else "missing")
    with c[3]:
        ui.tile("Mamba-130m", "loaded" if s["mamba"]["loaded"] else "on demand",
                "loads on the first question (~10 s)")
    if not s["mamba"]["loaded"] and st.button("Load Mamba now"):
        try:
            st.success(f"Mamba loaded: {svc.load_mamba()}")
        except ServiceError as exc:
            st.error(str(exc))
    st.write(f"**Models:** `{s['mamba']['model_id']}` (answer ranking), `{s['embedding_model']}` (embeddings)")
    st.write(f"**Uploaded-logs storage:** `{s['storage_dir']}`")
    hist = svc.comparison_history()
    st.markdown('<div class="section-title">Recent investigations</div>', unsafe_allow_html=True)
    if hist:
        st.dataframe(pd.DataFrame(hist[::-1]), hide_index=True, use_container_width=True)
        if st.button("Clear history"):
            svc.clear_comparison_history()
            st.rerun()
    else:
        st.caption("None yet.")


nav = st.navigation([
    st.Page(page_home, title="Overview", icon=":material/home:", default=True),
    st.Page(page_investigate, title="Investigate", icon=":material/search:", url_path="investigate"),
    st.Page(page_results, title="Results", icon=":material/insights:", url_path="results"),
    st.Page(page_upload, title="Upload logs", icon=":material/upload_file:", url_path="upload"),
    st.Page(page_explorer, title="Memory explorer", icon=":material/database:", url_path="memory"),
    st.Page(page_system, title="System", icon=":material/settings:", url_path="system"),
])
st.sidebar.markdown('<div style="font-weight:800;font-size:1.05rem;margin:-6px 0 2px">🔥 AgentMamba</div>'
                    '<div class="note">Ladle incident investigator</div>', unsafe_allow_html=True)
nav.run()
