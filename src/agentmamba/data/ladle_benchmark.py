"""Synthetic LADLE operational-log datasets: one SHORT-context and one LONG-context version.

ALL DATA HERE IS FICTIONAL. It imitates the formats of a steel-melt-shop ladle log book
(historian exports, alarm journals, CMMS maintenance exports, thermography rounds, inspection
reports, RCA and return-to-service records, ladle cycle logs).

Why two datasets
----------------
Both datasets ask the SAME questions about the SAME incident shift (same seeds, same record IDs,
same gold answers). Only the amount of surrounding history differs:

* ``short``: the incident-shift log only (about 20 records, ~1,100 Mamba tokens). Everything fits
  inside Mamba's context window, so plain Mamba should answer the answerable questions well.
* ``long``: the same incident shift followed by thousands of later and older records (about
  3,000 records, ~200,000 tokens): following-day operations with sibling ladles raising the
  SAME alarms, older maintenance history, a long care manual and six weeks of routine ladle
  cycles. A fixed context window can no longer hold the evidence; selective retrieval from
  episodic memory has to find it.

The long dataset arrives in four cumulative stages (1 = the short dataset), so accuracy and
latency can also be plotted against the amount of history.

Question families (15 per scenario)
    value:      maintenance action, alarm at a time, shell-temperature reading at a time,
                inspection finding, confirmed root cause, documented clearance
    abstain:    root cause still under investigation, clearance still pending, unknown ladle,
                a date with no maintenance record
    conflict:   thermography and the manual round disagree for the same ladle and time
    paraphrase: four answerable questions reworded without the records' vocabulary
Questions (what systems see) and annotations (gold answers, evidence IDs) are separate files.
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from agentmamba.data.document_loader import load_any
from agentmamba.data.log_benchmark import LogAnnotation, LogQuestion, _fmt_day, _val, _Writer

DATASET_VERSION = "ladle-bench-v1"
DATASETS = ("short", "long")
LADLES = [f"LADLE-{n:02d}" for n in (3, 5, 7, 9, 11, 12, 14, 16)]
AUX = ["LF-01", "LTC-02"]                                   # ladle furnace, ladle transfer car
SIGNALS = {
    "LADLE": [("shell_temperature", "degC", 180, 420, 0), ("steel_temperature", "degC", 1545, 1640, 0),
              ("argon_flow", "Nl/min", 60, 380, 0)],
    "LF": [("arc_current", "kA", 22, 45, 1), ("electrode_power", "MW", 8, 22, 1)],
    "LTC": [("travel_speed", "m/min", 10, 40, 1), ("drive_current", "A", 40, 160, 1)],
}
ACTIONS = ["replaced slide gate plates", "replaced porous plug", "relined slag line with MgO-C bricks",
           "gunned hot spot on the ladle barrel", "replaced well block", "re-seated the collector nozzle",
           "replaced trunnion bearing", "repaired argon line coupling", "replaced slide gate cylinder seal",
           "patched impact pad refractory"]
ALARMS = ["SHELL HOT SPOT", "SLIDE GATE HYDRAULIC LEAK", "POROUS PLUG BLOCKED", "LOW ARGON FLOW",
          "STEEL TEMPERATURE DROP", "TILT DRIVE OVERLOAD"]
FINDINGS = ["erosion of the slag line bricks to 40 mm residual thickness", "a crack in the well block refractory",
            "skull build-up around the collector nozzle", "shell discoloration near the north trunnion",
            "spalling of the impact pad", "a worn bore on the upper slide gate plate",
            "a loose trunnion pin retaining bolt", "open joints between the barrel bricks"]
CAUSES = ["refractory erosion at the slag line", "steel penetration through a cracked well block",
          "a porous plug blocked by slag infiltration", "a failed seal on the slide gate cylinder",
          "thermal shock after insufficient preheating", "a misaligned nozzle after the last gate change"]
TECHS = ["T. Varga", "M. Okafor", "S. Lindqvist", "R. Iyer", "J. Moreau", "A. Haddad"]
GRADES = ["S275JR", "S355J2", "DC04", "API-X65", "C45", "S460M"]
STAGE_NAMES = {1: "incident shift (the short dataset)", 2: "following three days of operations",
               3: "older maintenance history and the ladle care manual", 4: "six weeks of later ladle cycles"}


def _signal(eq: str, rng, which: int | None = None):
    sigs = SIGNALS[eq.split("-")[0]]
    return sigs[which % len(sigs)] if which is not None else rng.choice(sigs)


def build_scenario(area: str, day: str, root: Path, seed: int, dataset: str) -> tuple[list, list]:
    """Write one scenario. ``dataset='short'`` writes stage 1 only; ``'long'`` writes stages 1-4.
    The random stream for stage 1 and the questions is identical in both, so gold IDs match."""
    rng = random.Random(seed)
    long_rng = random.Random(seed + 7919)          # only the long-history stages draw from this one
    eqs = LADLES + AUX
    D = datetime.fromisoformat(day)
    day_s = D.strftime("%Y-%m-%d")
    words = _fmt_day(D)
    w = _Writer(root)
    roles = dict(zip(["maint", "alarm", "sensor", "insp", "rca", "hedge", "clear", "noclear"], rng.sample(LADLES, 8)))
    roles["conflict"] = rng.choice([e for e in LADLES if e != roles["sensor"]])
    roles["missing"] = "LADLE-29"
    ts = lambda d, h, m=0: d.replace(hour=h, minute=m).strftime("%Y-%m-%dT%H:%M:%S")  # noqa: E731
    gold_action = rng.choice(ACTIONS)
    gold_alarm = rng.choice(ALARMS)
    gold_finding = rng.choice(FINDINGS)
    gold_cause, later_cause = rng.sample(CAUSES, 2)
    t_alarm = (rng.randint(8, 11), rng.choice([5, 10, 20, 25, 35, 40, 50, 55]))
    t_sensor = (rng.randint(8, 12), 30)
    t_conf = (rng.choice([h for h in range(8, 13) if h != t_sensor[0]]), 30)
    s_sig = SIGNALS["LADLE"][0]                    # shell temperature: the ladle engineer's key reading
    c_sig = SIGNALS["LADLE"][0]
    s_val = _val(rng, *s_sig[2:])
    c_hist = _val(rng, *c_sig[2:])
    c_manual = c_hist
    while abs(c_manual - c_hist) < (c_sig[3] - c_sig[2]) * 0.15:
        c_manual = _val(rng, *c_sig[2:])
    wo = lambda r=rng: f"WO-{area}-{r.randint(10000, 99999)}"  # noqa: E731
    wo_maint, wo_clear, wo_noclear = wo(), wo(), wo()

    def hist(e, sig, t, v):
        return {"timestamp": t, "equipment": e, "signal": sig[0], "value": v, "unit": sig[1]}

    # ------------------------------------------------------------ stage 1: incident shift
    others = [e for e in LADLES if e not in (roles["sensor"], roles["conflict"])]
    h1 = [hist(roles["sensor"], s_sig, ts(D, *t_sensor), s_val),
          hist(roles["conflict"], c_sig, ts(D, *t_conf), c_hist)]
    for e in rng.sample(others, 2):
        sig = _signal(e, rng)
        h1.append(hist(e, sig, ts(D, rng.randint(8, 12), 30), _val(rng, *sig[2:])))
    h1.sort(key=lambda r: r["timestamp"])
    w.rows(1, f"ladle_historian_{D:%Y%m%d}.csv", h1, "LADLE")
    manual_row = {"timestamp": ts(D, *t_conf), "equipment": roles["conflict"], "signal": c_sig[0], "value": c_manual,
                  "unit": c_sig[1], "operator": rng.choice(TECHS)}
    w.rows(1, f"thermography_round_{D:%Y%m%d}.csv", [manual_row], "LADLE")
    t_clr = D.replace(hour=t_alarm[0], minute=t_alarm[1]) + timedelta(minutes=22)
    alarms1 = [{"ts": ts(D, *t_alarm), "equipment": roles["alarm"], "alarm": gold_alarm, "state": "RAISED",
                "priority": rng.choice([1, 2])},
               {"ts": t_clr.strftime("%Y-%m-%dT%H:%M:%S"), "equipment": roles["alarm"], "alarm": gold_alarm,
                "state": "CLEARED", "priority": 2},
               {"ts": ts(D, 14, 5), "equipment": roles["hedge"], "alarm": "LADLE REMOVED FROM SERVICE",
                "state": "RAISED", "priority": 1}]
    alarm_file = f"ladle_alarm_log_{D:%Y%m%d}.jsonl"
    w.rows(1, alarm_file, alarms1, "LADLE")
    m1 = [{"timestamp": ts(D, 10, 15), "equipment": roles["maint"], "work_order": wo_maint, "action": gold_action,
           "technician": rng.choice(TECHS), "status": "completed"},
          {"timestamp": ts(D, 12, 30), "equipment": roles["clear"], "work_order": wo_clear,
           "action": rng.choice([a for a in ACTIONS if a != gold_action]), "technician": rng.choice(TECHS),
           "status": "completed"},
          {"timestamp": ts(D, 13, 10), "equipment": roles["noclear"], "work_order": wo_noclear,
           "action": rng.choice([a for a in ACTIONS if a != gold_action]), "technician": rng.choice(TECHS),
           "status": "completed"}]
    maint_file = f"ladle_maintenance_log_{D:%Y%m%d}.csv"
    w.rows(1, maint_file, m1, "MAINTENANCE")
    p_insp = w.doc(1, f"inspection_{roles['insp']}_{D:%Y%m%d}.md",
                   f"# Ladle inspection report\n\nInspection report for {roles['insp']}, {words}: the inspector "
                   f"recorded the following finding: {gold_finding}.\n")
    p_rca = w.doc(1, f"rca_{roles['rca']}_{D:%Y%m%d}.md",
                  f"# Root cause analysis\n\nRoot cause analysis for the trip of {roles['rca']} on {words}: the root "
                  f"cause was confirmed as {gold_cause}.\n")
    p_notes = w.doc(1, f"shift_notes_{D:%Y%m%d}.md",
                    f"# Melt shop shift notes\n\n{roles['hedge']} tripped at 14:05 on {words} and was removed from "
                    f"service. A shell hot spot was observed before the trip; the root cause is under investigation "
                    f"and has not been determined.\n\nRepair work on {roles['noclear']} was completed on {words}. The "
                    f"functional test is pending and the ladle has not been cleared for operation.\n")
    p_rts = w.doc(1, f"return_to_service_{roles['clear']}_{D:%Y%m%d}.md",
                  f"# Return-to-service record\n\nReturn-to-service record: {roles['clear']} was cleared for "
                  f"operation on {words} at 16:40 by the shift supervisor after the repair under work order "
                  f"{wo_clear}.\n")

    if dataset == "long":
        _long_history(w, long_rng, area, D, eqs, roles, gold_alarm, gold_action, gold_finding, later_cause,
                      alarm_file, alarms1, ts, hist)

    (root / "manifest.json").write_text(json.dumps({
        "benchmark": DATASET_VERSION, "dataset": dataset, "scenario": area, "incident_day": day_s, "roles": roles,
        "stages": {str(k): {"name": STAGE_NAMES[k], "files": v} for k, v in w.manifest.items() if v},
        "synthetic": True}, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ questions + annotations
    def ids_of(rel_path: str, stage: str, pred) -> list[str]:
        return [e.event_id for e in load_any(root / rel_path, stage=stage) if pred(e)]

    def row_ids(name, row, stage="LADLE"):
        return ids_of(f"stage1/{name}", stage, lambda e: e.raw == {k: (str(v) if name.endswith(".csv") else v)
                                                                   for k, v in row.items()})

    def chunk_ids(p, needle):
        return ids_of(f"stage1/{p.name}", "MAINTENANCE", lambda e: needle in e.text)

    hh = lambda t: f"{t[0]:02d}:{t[1]:02d}"  # noqa: E731
    later = (D + timedelta(days=3)).strftime("%Y-%m-%d")
    hist_file = f"ladle_historian_{D:%Y%m%d}.csv"
    qs = [
        ("maint_action", f"What maintenance action was recorded for {roles['maint']} on {day_s}?", "value",
         gold_action, "exact", row_ids(maint_file, m1[0], "MAINTENANCE"), []),
        ("alarm_at_time", f"Which alarm was raised on {roles['alarm']} at {day_s} {hh(t_alarm)}?", "value",
         gold_alarm, "exact", row_ids(alarm_file, alarms1[0]), row_ids(alarm_file, alarms1[1])),
        ("sensor_at_time", f"What was the shell temperature reading for {roles['sensor']} at {day_s} "
                           f"{hh(t_sensor)}?", "value", f"{s_val} {s_sig[1]}", "exact",
         row_ids(hist_file, h1[[r['equipment'] for r in h1].index(roles['sensor'])]), []),
        ("inspection_finding", f"What inspection finding was recorded for {roles['insp']} on {day_s}?", "value",
         gold_finding, "contains", chunk_ids(p_insp, gold_finding), []),
        ("rca_documented", f"What was the confirmed root cause of the trip of {roles['rca']} on {day_s}?", "value",
         gold_cause, "contains", chunk_ids(p_rca, gold_cause), []),
        ("rca_undocumented", f"What was the root cause of the trip of {roles['hedge']} on {day_s}?", "abstain",
         None, None, [], chunk_ids(p_notes, "under investigation")),
        ("clearance_documented", f"Was {roles['clear']} cleared for operation after the repair on {day_s}?", "value",
         f"{roles['clear']} was cleared for operation", "contains", chunk_ids(p_rts, "cleared for operation"), []),
        ("clearance_undocumented", f"Was {roles['noclear']} cleared for operation after the repair on {day_s}?",
         "abstain", None, None, [], chunk_ids(p_notes, "functional test is pending")),
        ("conflicting_readings", f"What was the shell temperature reading for {roles['conflict']} at "
                                 f"{day_s} {hh(t_conf)}?", "conflict", None, None,
         row_ids(hist_file, h1[[r['equipment'] for r in h1].index(roles['conflict'])]) +
         row_ids(f"thermography_round_{D:%Y%m%d}.csv", manual_row), []),
        ("unknown_ladle", f"What maintenance action was recorded for {roles['missing']} on {day_s}?", "abstain",
         None, None, [], []),
        ("wrong_date", f"What maintenance action was recorded for {roles['maint']} on {later}?", "abstain",
         None, None, [], row_ids(maint_file, m1[0], "MAINTENANCE")),
        ("maint_action_paraphrase", f"What work was carried out on {roles['maint']} on {day_s}?", "value",
         gold_action, "exact", row_ids(maint_file, m1[0], "MAINTENANCE"), []),
        ("alarm_paraphrase", f"Which alarm did {roles['alarm']} trigger at {day_s} {hh(t_alarm)}?", "value",
         gold_alarm, "exact", row_ids(alarm_file, alarms1[0]), []),
        ("inspection_paraphrase", f"What did the inspection of {roles['insp']} on {day_s} find?", "value",
         gold_finding, "contains", chunk_ids(p_insp, gold_finding), []),
        ("rca_paraphrase", f"Why did {roles['rca']} trip on {day_s}?", "value", gold_cause, "contains",
         chunk_ids(p_rca, gold_cause), []),
    ]
    questions, anns = [], []
    for n, (fam, text, kind, ans, match, req, rel) in enumerate(qs, 1):
        qid = f"{area}-Q{n:02d}"
        questions.append(LogQuestion(qid, area, dataset, text))
        anns.append(LogAnnotation(qid, area, dataset, fam, kind, ans, match, req, rel))
    return questions, anns


def _long_history(w, rng, area, D, eqs, roles, gold_alarm, gold_action, gold_finding, later_cause,
                  alarm_file, alarms1, ts, hist) -> None:
    """Stages 2-4: everything a long-context model has to wade through."""
    wo = lambda: f"WO-{area}-{rng.randint(10000, 99999)}"  # noqa: E731
    # ---------------- stage 2: following three days (sibling ladles raise the SAME alarm)
    h2 = []
    for k in range(3):
        d = D + timedelta(days=k)
        for h in range(0, 24, 2):
            for e in eqs:
                sig = _signal(e, rng, 0)
                if k == 0 and h <= 14:          # never repeat the incident-shift readings
                    continue
                h2.append(hist(e, sig, ts(d, h), _val(rng, *sig[2:])))
    w.rows(2, f"ladle_historian_{D:%Y%m%d}_{(D + timedelta(days=2)):%Y%m%d}.csv", h2, "LADLE")
    w.rows(2, alarm_file, alarms1, "LADLE")                       # exact re-delivery -> duplicates
    sib = [e for e in LADLES if e != roles["alarm"]]
    a2 = []
    for k in range(3):
        d = D + timedelta(days=k)
        for _ in range(6):
            t = d.replace(hour=rng.randint(15 if k == 0 else 0, 23), minute=rng.choice([0, 15, 30, 45]))
            a2.append({"ts": t.strftime("%Y-%m-%dT%H:%M:%S"), "equipment": rng.choice(sib),
                       "alarm": gold_alarm if rng.random() < 0.6 else rng.choice(ALARMS), "state": "RAISED",
                       "priority": rng.choice([1, 2, 3])})
    a2.sort(key=lambda r: r["ts"])
    w.rows(2, f"ladle_alarm_log_{(D + timedelta(days=1)):%Y%m%d}_{(D + timedelta(days=2)):%Y%m%d}.jsonl", a2, "LADLE")
    m2 = []
    for k in (1, 2):
        d = D + timedelta(days=k)
        for e in rng.sample([x for x in LADLES if x != roles["maint"]], 4):
            m2.append({"timestamp": ts(d, rng.randint(7, 16), rng.choice([0, 15, 30, 45])), "equipment": e,
                       "work_order": wo(), "action": rng.choice([a for a in ACTIONS if a != gold_action]),
                       "technician": rng.choice(TECHS), "status": "completed"})
    m2.sort(key=lambda r: r["timestamp"])
    w.rows(2, f"ladle_maintenance_log_{(D + timedelta(days=1)):%Y%m%d}_{(D + timedelta(days=2)):%Y%m%d}.csv", m2,
           "MAINTENANCE")
    n2 = "\n\n".join(f"{e} cycling normally on {_fmt_day(D + timedelta(days=k))}; no abnormal shell readings "
                     f"reported during the round." for k in (1, 2) for e in rng.sample(LADLES, 3))
    w.doc(2, f"shift_notes_{(D + timedelta(days=2)):%Y%m%d}.md", "# Melt shop shift notes\n\n" + n2 + "\n")

    # ---------------- stage 3: older history + long care manual
    m3 = []
    for e in eqs:
        for _ in range(rng.randint(6, 10)):
            d = D - timedelta(days=rng.randint(2, 120))
            m3.append({"timestamp": ts(d, rng.randint(7, 16), rng.choice([0, 15, 30, 45])), "equipment": e,
                       "work_order": wo(), "action": rng.choice(ACTIONS), "technician": rng.choice(TECHS),
                       "status": "completed"})
    m3.sort(key=lambda r: r["timestamp"])
    w.rows(3, f"ladle_maintenance_history_{(D - timedelta(days=120)):%Y%m%d}_{(D - timedelta(days=1)):%Y%m%d}.csv",
           m3, "MAINTENANCE")
    a3 = []
    for _ in range(50):
        d = D - timedelta(days=rng.randint(1, 90), hours=rng.randint(0, 23))
        a3.append({"ts": d.replace(minute=rng.choice([0, 15, 30, 45]), second=0).strftime("%Y-%m-%dT%H:%M:%S"),
                   "equipment": rng.choice(LADLES), "alarm": rng.choice(ALARMS), "state": "RAISED",
                   "priority": rng.choice([1, 2, 3])})
    a3.sort(key=lambda r: r["ts"])
    w.rows(3, f"ladle_alarm_history_{(D - timedelta(days=90)):%Y%m%d}_{(D - timedelta(days=1)):%Y%m%d}.jsonl", a3,
           "LADLE")
    for back in (30, 60):
        d = D - timedelta(days=back)
        w.doc(3, f"inspection_{roles['insp']}_{d:%Y%m%d}.md",
              f"# Ladle inspection report\n\nInspection report for {roles['insp']}, {_fmt_day(d)}: the inspector "
              f"recorded the following finding: {rng.choice([f for f in FINDINGS if f != gold_finding])}.\n")
    manual = [f"# Ladle care manual, melt shop {area}",
              "This manual describes routine care for the steel ladles and their auxiliaries. It is general "
              "guidance and does not record any specific event."]
    for e in eqs:
        manual.append(f"{e}: measure the shell temperature by thermography every heat, check the slide gate "
                      f"plates every 3 heats and record refractory wear on the round sheet. Relining and gate "
                      f"changes are planned through the maintenance system with a work order.")
    manual += ["When a ladle trips or is removed from service, the root cause must be confirmed by refractory "
               "engineering before it is recorded as established. Until then, operators note observations only.",
               "Return to service after repair requires preheating, a functional test of the slide gate and a "
               "supervisor signature; a ladle is cleared for operation only by that signed record."] * 3
    w.doc(3, f"ladle_care_manual_{area}.md", "\n\n".join(manual) + "\n")

    # ---------------- stage 4: six weeks of later ladle cycles
    h4 = []
    for k in range(3, 46):
        d = D + timedelta(days=k)
        for h in range(0, 24, 6):
            for e in eqs:
                sig = _signal(e, rng, 0)
                h4.append(hist(e, sig, ts(d, h), _val(rng, *sig[2:])))
    w.rows(4, f"ladle_historian_{(D + timedelta(days=3)):%Y%m%d}_{(D + timedelta(days=45)):%Y%m%d}.csv", h4, "LADLE")
    cyc, heat = [], rng.randint(1000, 4000)
    for k in range(3, 46):
        d = D + timedelta(days=k)
        for _ in range(16):
            heat += 1
            t = d.replace(hour=rng.randint(0, 23), minute=rng.choice(range(0, 60, 5)))
            cyc.append({"timestamp": t.strftime("%Y-%m-%dT%H:%M:%S"), "ladle": rng.choice(LADLES),
                        "heat": f"{area}-H{heat:04d}", "grade": rng.choice(GRADES),
                        "tap_temperature": rng.randint(1600, 1680), "cycle_event": "TAPPED"})
    cyc.sort(key=lambda r: r["timestamp"])
    w.rows(4, f"ladle_cycle_log_{(D + timedelta(days=3)):%Y%m%d}_{(D + timedelta(days=45)):%Y%m%d}.csv", cyc, "LADLE")
    a4, m4 = [], []
    for _ in range(70):
        d = D + timedelta(days=rng.randint(3, 45), hours=rng.randint(0, 23))
        a4.append({"ts": d.replace(minute=rng.choice([0, 15, 30, 45]), second=0).strftime("%Y-%m-%dT%H:%M:%S"),
                   "equipment": rng.choice(LADLES), "alarm": rng.choice(ALARMS), "state": "RAISED",
                   "priority": rng.choice([1, 2, 3])})
    for _ in range(35):
        d = D + timedelta(days=rng.randint(4, 45))         # never on D+3 (the wrong-date question's date)
        m4.append({"timestamp": ts(d, rng.randint(7, 16), rng.choice([0, 15, 30, 45])), "equipment": rng.choice(eqs),
                   "work_order": wo(), "action": rng.choice(ACTIONS), "technician": rng.choice(TECHS),
                   "status": "completed"})
    a4.sort(key=lambda r: r["ts"])
    m4.sort(key=lambda r: r["timestamp"])
    w.rows(4, f"ladle_alarm_log_{(D + timedelta(days=3)):%Y%m%d}_{(D + timedelta(days=45)):%Y%m%d}.jsonl", a4, "LADLE")
    w.rows(4, f"ladle_maintenance_log_{(D + timedelta(days=4)):%Y%m%d}_{(D + timedelta(days=45)):%Y%m%d}.csv", m4,
           "MAINTENANCE")
    d20, d10 = D + timedelta(days=20), D + timedelta(days=10)
    w.doc(4, f"inspection_{roles['insp']}_{d20:%Y%m%d}.md",
          f"# Ladle inspection report\n\nInspection report for {roles['insp']}, {_fmt_day(d20)}: the inspector "
          f"recorded the following finding: {rng.choice([f for f in FINDINGS if f != gold_finding])}.\n")
    w.doc(4, f"rca_{roles['hedge']}_{d10:%Y%m%d}.md",
          f"# Root cause analysis\n\nRoot cause analysis for the trip of {roles['hedge']} on {_fmt_day(d10)}: the "
          f"root cause was confirmed as {later_cause}.\n")


def write_datasets(out_dir: str | Path, seed: int = 2027, n_scenarios: int = 4) -> dict:
    """Write ``<out>/short`` and ``<out>/long``. Scenario ``i`` uses the same seed in both."""
    out = Path(out_dir)
    base = datetime(2026, 3, 2)
    manifest = {"benchmark": DATASET_VERSION, "seed": seed, "synthetic": True, "datasets": {}}
    for dataset in DATASETS:
        droot = out / dataset
        shutil.rmtree(droot, ignore_errors=True)
        all_q, all_a = [], []
        for i in range(n_scenarios):
            area = f"SMS{i + 1}"
            day = (base + timedelta(days=9 * i)).strftime("%Y-%m-%d")
            qs, anns = build_scenario(area, day, droot / area, seed * 100 + i, dataset)
            missing = [x.question_id for x in anns if x.answer_kind in ("value", "conflict") and not x.required_evidence_ids]
            if missing:
                raise RuntimeError(f"evidence IDs not resolved for {missing}")
            all_q += qs
            all_a += anns
        (droot / "questions.jsonl").write_text("".join(json.dumps(asdict(q)) + "\n" for q in all_q), encoding="utf-8")
        (droot / "annotations.jsonl").write_text("".join(json.dumps(asdict(a)) + "\n" for a in all_a), encoding="utf-8")
        manifest["datasets"][dataset] = {"scenarios": [f"SMS{i + 1}" for i in range(n_scenarios)],
                                         "n_questions": len(all_q)}
    manifest["note"] = ("Fictional ladle logs. 'short' = incident shift only; 'long' = the same shift plus thousands "
                        "of later/older records. Same questions and gold answers in both.")
    (out / "datasets_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- loaders
def dataset_root(dataset: str, base: str | Path | None = None) -> Path:
    from agentmamba.config import PROJECT_ROOT

    return Path(base) if base else PROJECT_ROOT / "data" / "ladle_benchmark" / dataset


def load_questions(dataset: str, base: str | Path | None = None) -> list[LogQuestion]:
    p = dataset_root(dataset, base) / "questions.jsonl"
    return [LogQuestion(**json.loads(x)) for x in p.read_text(encoding="utf-8").splitlines() if x]


def load_annotations(dataset: str, base: str | Path | None = None) -> dict[str, LogAnnotation]:
    """Evaluation-only: read after a system has produced its output."""
    p = dataset_root(dataset, base) / "annotations.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]
    return {r["question_id"]: LogAnnotation(**r) for r in rows}


def scenario_manifest(dataset: str, scenario: str, base: str | Path | None = None) -> dict:
    return json.loads((dataset_root(dataset, base) / scenario / "manifest.json").read_text(encoding="utf-8"))


def stage_events(dataset: str, scenario: str, stage: int, base: str | Path | None = None) -> list:
    """Events of one stage, in file arrival order."""
    root = dataset_root(dataset, base) / scenario
    m = scenario_manifest(dataset, scenario, base)
    evs = []
    for f in m["stages"].get(str(stage), {}).get("files", []):
        evs += load_any(root / f["file"], stage=f["stage"])
    return evs


def stages_of(dataset: str, scenario: str, base: str | Path | None = None) -> list[int]:
    return sorted(int(k) for k in scenario_manifest(dataset, scenario, base)["stages"])
