"""Look-and-feel for the AgentMamba dashboard: one stylesheet and a few HTML components.

Colours: the two systems keep FIXED identities everywhere (charts, badges, cards):
Baseline Mamba = blue (#3987e5), AgentMamba = molten-steel orange (#d95926) - categorical slots
1 and 2 of the validated dark palette. Status colours (good / warning / critical) are reserved
for answer status and always come with an icon and a label.
"""
from __future__ import annotations

import html

import streamlit as st

BASE_C = "#3987e5"
AGENT_C = "#d95926"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
SURFACE = "#1a1b20"
INK, INK2, MUTED = "#eceae4", "#c3c2b7", "#898781"
GRID, AXIS = "#2c2c2a", "#383835"

CSS = f"""
<style>
:root {{ --ink:{INK}; --ink2:{INK2}; --muted:{MUTED}; --surface:{SURFACE}; --base:{BASE_C}; --agent:{AGENT_C};
        --good:{GOOD}; --warn:{WARN}; --crit:{CRIT}; --line:rgba(255,255,255,.08); }}
.block-container {{ padding-top: 3.2rem; max-width: 1280px; }}
@media (max-width: 900px) {{ .flow .arrow {{ display:none; }} .flow .step {{ flex:1 1 160px; }} }}
h1, h2, h3 {{ letter-spacing: -.01em; }}
.hero {{ position:relative; overflow:hidden; border-radius:18px; padding:28px 30px 26px;
        background: radial-gradient(120% 140% at 100% 0%, rgba(240,122,53,.35) 0%, rgba(240,122,53,0) 55%),
                    linear-gradient(135deg, #1d1f26 0%, #15161b 100%);
        border:1px solid var(--line); margin-bottom: 18px; }}
.hero:after {{ content:""; position:absolute; right:-60px; bottom:-80px; width:280px; height:280px; border-radius:50%;
        background: radial-gradient(circle, rgba(255,170,60,.25), rgba(255,120,40,0) 70%); }}
.hero .eyebrow {{ color:#f3a06d; font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }}
.hero h1 {{ margin:.25rem 0 .35rem; font-size:2.1rem; line-height:1.15; color:var(--ink); }}
.hero p {{ color:var(--ink2); font-size:1.02rem; max-width:780px; margin:0; }}
.pill {{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-size:.78rem;
        font-weight:600; border:1px solid var(--line); background:rgba(255,255,255,.04); color:var(--ink2); }}
.pill .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.flow {{ display:flex; flex-wrap:wrap; align-items:stretch; gap:6px; margin:6px 0 4px; }}
.flow .step {{ flex:1 1 0; min-width:118px; background:var(--surface); border:1px solid var(--line);
        border-radius:12px; padding:12px 12px 10px; }}
.flow .step .n {{ font-size:.7rem; color:var(--muted); font-weight:700; letter-spacing:.08em; }}
.flow .step .t {{ font-weight:700; color:var(--ink); margin:2px 0 3px; }}
.flow .step .d {{ font-size:.8rem; color:var(--ink2); line-height:1.35; }}
.flow .step.hot {{ border-color: rgba(240,122,53,.55); box-shadow: 0 0 0 1px rgba(240,122,53,.15) inset; }}
.flow .arrow {{ align-self:center; color:var(--muted); font-size:1.1rem; }}
.tile {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:16px 18px; height:100%; }}
.tile .k {{ color:var(--ink2); font-size:.82rem; font-weight:600; }}
.tile .v {{ font-size:2rem; font-weight:750; color:var(--ink); line-height:1.1; margin-top:4px; }}
.tile .s {{ color:var(--muted); font-size:.8rem; margin-top:4px; }}
.tile .vs {{ display:flex; gap:18px; align-items:baseline; margin-top:6px; }}
.tile .vs .lab {{ font-size:.72rem; color:var(--muted); display:flex; align-items:center; gap:5px; }}
.tile .vs .num {{ font-size:1.7rem; font-weight:750; color:var(--ink); }}
.sys {{ border-radius:16px; border:1px solid var(--line); background:var(--surface); padding:18px 20px; height:100%; }}
.sys.base {{ border-top:4px solid var(--base); }}
.sys.agent {{ border-top:4px solid var(--agent); }}
.sys .head {{ display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap; }}
.sys .name {{ font-weight:750; font-size:1.05rem; color:var(--ink); }}
.sys .sub {{ color:var(--muted); font-size:.8rem; margin-top:2px; }}
.sys .answer {{ font-size:1.25rem; font-weight:650; color:var(--ink); margin:14px 0 10px; line-height:1.35;
        word-break: break-word; }}
.sys .answer.none {{ color:var(--ink2); font-weight:500; font-size:1.05rem; }}
.sys .facts {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }}
.badge {{ display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:8px; font-size:.8rem;
        font-weight:700; color:var(--ink); background:rgba(255,255,255,.05); border:1px solid var(--line); }}
.badge.good {{ border-color:rgba(12,163,12,.6); background:rgba(12,163,12,.12); }}
.badge.warn {{ border-color:rgba(250,178,25,.6); background:rgba(250,178,25,.10); }}
.badge.crit {{ border-color:rgba(208,59,59,.7); background:rgba(208,59,59,.12); }}
.timeline {{ border-left:2px solid var(--line); margin:4px 0 0 8px; padding-left:18px; }}
.timeline .ev {{ position:relative; margin-bottom:14px; }}
.timeline .ev:before {{ content:""; position:absolute; left:-25px; top:4px; width:12px; height:12px; border-radius:50%;
        background:var(--surface); border:2px solid var(--agent); }}
.timeline .ev.skip:before {{ border-color:var(--muted); }}
.timeline .ev .t {{ font-weight:700; color:var(--ink); }}
.timeline .ev .d {{ color:var(--ink2); font-size:.86rem; }}
.rec {{ border:1px solid var(--line); border-radius:10px; padding:10px 12px; margin-bottom:8px; background:rgba(255,255,255,.02); }}
.rec .meta {{ font-size:.74rem; color:var(--muted); display:flex; gap:10px; flex-wrap:wrap; }}
.rec .meta b {{ color:var(--ink2); font-weight:600; }}
.rec .txt {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size:.8rem; color:var(--ink);
        margin-top:4px; word-break: break-word; }}
.note {{ color:var(--muted); font-size:.82rem; }}
.section-title {{ font-size:1.15rem; font-weight:750; margin:18px 0 6px; color:var(--ink); }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; font-size:.82rem; color:var(--ink2); margin:2px 0 6px; }}
.legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.legend i {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
div[data-testid="stSidebarNav"] {{ padding-top: .4rem; }}
@media (max-width: 640px) {{ .hero h1 {{ font-size:1.6rem; }} .tile .v {{ font-size:1.6rem; }} }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def hero(eyebrow: str, title: str, text: str) -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1>'
                f'<p>{esc(text)}</p></div>', unsafe_allow_html=True)


def flow(steps: list[tuple[str, str, bool]]) -> None:
    parts = []
    for i, (t, d, hot) in enumerate(steps):
        if i:
            parts.append('<div class="arrow">&rarr;</div>')
        parts.append(f'<div class="step{" hot" if hot else ""}"><div class="n">STEP {i + 1}</div>'
                     f'<div class="t">{esc(t)}</div><div class="d">{esc(d)}</div></div>')
    st.markdown(f'<div class="flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def tile(label: str, value: str, sub: str = "") -> None:
    st.markdown(f'<div class="tile"><div class="k">{esc(label)}</div><div class="v">{esc(value)}</div>'
                f'<div class="s">{esc(sub)}</div></div>', unsafe_allow_html=True)


def vs_tile(label: str, base: str, agent: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="tile"><div class="k">{esc(label)}</div><div class="vs">'
        f'<div><div class="lab"><span class="dot" style="width:8px;height:8px;border-radius:50%;background:{BASE_C};'
        f'display:inline-block"></span>Baseline Mamba</div><div class="num">{esc(base)}</div></div>'
        f'<div><div class="lab"><span style="width:8px;height:8px;border-radius:50%;background:{AGENT_C};'
        f'display:inline-block"></span>AgentMamba</div><div class="num">{esc(agent)}</div></div></div>'
        f'<div class="s">{esc(sub)}</div></div>', unsafe_allow_html=True)


def legend() -> None:
    st.markdown(f'<div class="legend"><span><i style="background:{BASE_C}"></i>Baseline Mamba</span>'
                f'<span><i style="background:{AGENT_C}"></i>AgentMamba</span></div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def pct(v, nd=0) -> str:
    return "n/a" if v is None else f"{100 * v:.{nd}f}%"


def status_badge(out: dict | None, is_baseline: bool) -> str:
    """Answer status -> (icon + label) badge. Never colour alone."""
    if not out:
        return badge("✖ Did not run", "crit")
    st_ = out.get("answer_status")
    ver = out.get("verification_status")
    if is_baseline:
        if st_ == "answered":
            return badge("⚠ Answered, unverified", "warn")
        return badge("○ No answer found in its window", "")
    if ver == "conflict":
        return badge("⚡ Sources disagree - flagged", "warn")
    if st_ == "answered" and ver == "supported":
        return badge("✔ Verified against source records", "good")
    if st_ == "answered":
        return badge("⚠ Answered, not fully verified", "warn")
    if st_ == "needs_clarification":
        return badge("? Needs a ladle ID / date", "")
    return badge("🛡 Refused: not enough evidence", "")


def correctness_badge(score: dict | None) -> str:
    if not score:
        return ""
    return badge("✔ Correct", "good") if score.get("correct") else badge("✖ Incorrect", "crit")


def record_card(rid: str, source: str | None, ts: str | None, text: str, role: str | None = None) -> str:
    meta = [f"<b>{esc(rid)}</b>"]
    if role:
        meta.append(esc(role))
    if ts:
        meta.append(esc(ts))
    if source:
        meta.append(esc(source))
    return (f'<div class="rec"><div class="meta">{" · ".join(meta)}</div>'
            f'<div class="txt">{esc(text[:420])}{"…" if len(text) > 420 else ""}</div></div>')
