"""Synthetic industrial-log benchmark for the Baseline-Mamba vs AgentMamba comparison.

ALL DATA HERE IS FICTIONAL. It imitates the *formats* of operational logs (historian exports in
long format, alarm journals, CMMS maintenance exports, manual round sheets, inspection reports,
shift notes, RCA and return-to-service records); its values, events and causes are generated.

Design
* A scenario is one plant area with its own equipment namespace (e.g. ``P3-PMP-01``). Splits are
  by scenario (dev / test), so no identifier, file or record is shared between splits. Dev is for
  development and calibration; test is for final reporting only.
* Each scenario's history arrives in four cumulative stages (files listed in arrival order):
    1 incident day  - the records that answer the questions (plus a little context)
    2 operations    - the next days' logs for all equipment: sibling equipment with the SAME alarm
                      text and similar maintenance actions, plus an exact re-delivery of a stage-1
                      file (duplicates)
    3 history       - older maintenance / alarm history, older inspections of the same equipment
                      with other findings, and a long equipment-care manual
    4 later months  - routine logs for weeks after the incident, a later inspection with another
                      finding, and a later trip whose root cause WAS confirmed (different date)
  Later stages add records that arrive AFTER the evidence, so a recency-window context gradually
  stops containing it; whether it does is measured per question, not assumed.
* Fifteen question families per scenario, written once as fixed templates:
    value:    maintenance action, alarm at a time, sensor reading at a time (exact match)
    phrase:   inspection finding, confirmed root cause, documented clearance (containment)
    abstain:  undocumented root cause (only a hedged note), clearance not documented (pending),
              unknown equipment ID, a date with no maintenance record
    conflict: two sources report different readings for the same signal and time
    paraphrase: four answerable questions reworded without the records' vocabulary (robustness)
* Questions (what systems see) and annotations (gold answers, required evidence IDs) are written
  to separate files; the runner reads annotations only to score finished outputs.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from agentmamba.data.document_loader import load_any

BENCHMARK_VERSION = "logbench-v1"
EQUIPMENT = ["PMP-01", "PMP-02", "CMP-01", "MTR-01", "MTR-02", "FAN-01", "CNV-01", "CNV-02"]
SIGNALS = {
    "PMP": [("vibration", "mm/s", 1.5, 8.5, 1), ("discharge_pressure", "bar", 4.0, 9.0, 2),
            ("bearing_temp", "degC", 40, 90, 1)],
    "CMP": [("discharge_temp", "degC", 80, 140, 1), ("pressure", "bar", 6.0, 10.0, 2)],
    "MTR": [("current", "A", 30, 120, 1), ("winding_temp", "degC", 50, 110, 1), ("vibration", "mm/s", 1.0, 7.5, 1)],
    "FAN": [("vibration", "mm/s", 1.0, 9.0, 1), ("speed", "rpm", 900, 1500, 0)],
    "CNV": [("belt_speed", "m/s", 1.0, 3.0, 2), ("motor_current", "A", 20, 80, 1)],
}
ACTIONS = ["replaced mechanical seal", "replaced drive-end bearing", "realigned motor coupling",
           "cleaned suction strainer", "tightened baseplate anchor bolts", "replaced worn V-belts",
           "recalibrated pressure transmitter", "replaced cooling fan blade", "re-greased non-drive-end bearing",
           "replaced damaged cable gland"]
ALARMS = ["HIGH DISCHARGE PRESSURE", "HIGH BEARING TEMPERATURE", "HIGH VIBRATION", "MOTOR OVERCURRENT",
          "LOW SUCTION PRESSURE", "BELT MISALIGNMENT"]
FINDINGS = ["minor pitting on the impeller vanes", "hairline crack at the motor mounting foot",
            "heavy scaling inside the cooling jacket", "loose terminal connection in the junction box",
            "worn coupling insert with visible cracking", "oil leak at the gearbox output seal",
            "corroded anchor bolts on the drive side", "blocked lubrication line to the outboard bearing"]
CAUSES = ["bearing failure caused by lubrication starvation", "a failed thermal overload relay",
          "a blocked suction line", "shaft misalignment after the previous coupling replacement",
          "a cracked impeller vane", "a loose phase connection at the motor terminal box"]
TECHS = ["T. Varga", "M. Okafor", "S. Lindqvist", "R. Iyer", "J. Moreau", "A. Haddad"]
STAGE_NAMES = {1: "incident day", 2: "following operations", 3: "older history and long manual", 4: "later months"}


@dataclass
class Scenario:
    area: str
    split: str
    day: str                      # incident date (ISO)
    roles: dict = field(default_factory=dict)


@dataclass
class LogQuestion:
    question_id: str
    scenario: str
    split: str
    text: str


@dataclass
class LogAnnotation:
    question_id: str
    scenario: str
    split: str
    family: str
    answer_kind: str               # value | abstain | conflict
    expected_answer: str | None
    match: str | None              # exact | contains
    required_evidence_ids: list[str]
    related_evidence_ids: list[str]
    note: str = ""


def _fmt_day(d: datetime) -> str:
    return f"{d.day} {d.strftime('%B %Y')}"


def _val(rng, lo, hi, nd):
    v = rng.uniform(lo, hi)
    return int(round(v)) if nd == 0 else round(v, nd)


class _Writer:
    def __init__(self, root: Path):
        self.root = root
        self.manifest: dict[int, list[dict]] = {1: [], 2: [], 3: [], 4: []}

    def rows(self, stage: int, name: str, rows: list[dict], plant_stage: str = "UTILITIES"):
        p = self.root / f"stage{stage}" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".csv"):
            with open(p, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        else:
            p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.manifest[stage].append({"file": f"stage{stage}/{name}", "stage": plant_stage})
        return p

    def doc(self, stage: int, name: str, text: str):
        p = self.root / f"stage{stage}" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        self.manifest[stage].append({"file": f"stage{stage}/{name}", "stage": "MAINTENANCE"})
        return p


def _signal(eq: str, rng, which: int | None = None):
    sigs = SIGNALS[eq.split("-")[1]]
    return sigs[which % len(sigs)] if which is not None else rng.choice(sigs)


def build_scenario(sc: Scenario, root: Path, seed: int) -> tuple[list[LogQuestion], list[LogAnnotation]]:
    rng = random.Random(seed)
    A = sc.area
    eqs = [f"{A}-{e}" for e in EQUIPMENT]
    D = datetime.fromisoformat(sc.day)
    day = D.strftime("%Y-%m-%d")
    words = _fmt_day(D)
    w = _Writer(root)
    roles = dict(zip(["maint", "alarm", "sensor", "insp", "rca", "hedge", "clear", "noclear"], rng.sample(eqs, 8)))
    roles["conflict"] = rng.choice([e for e in eqs if e != roles["sensor"]])
    roles["missing"] = f"{A}-PMP-09"
    sc.roles = roles
    ts = lambda d, h, m=0: (d.replace(hour=h, minute=m)).strftime("%Y-%m-%dT%H:%M:%S")  # noqa: E731
    gold_action = rng.choice(ACTIONS)
    gold_alarm = rng.choice(ALARMS)
    gold_finding = rng.choice(FINDINGS)
    gold_cause, later_cause = rng.sample(CAUSES, 2)
    t_alarm = (rng.randint(8, 11), rng.choice([5, 10, 20, 25, 35, 40, 50, 55]))
    t_sensor = (rng.randint(8, 12), 30)
    t_conf = (rng.choice([h for h in range(8, 13) if h != t_sensor[0]]), 30)
    s_sig = _signal(roles["sensor"], rng)
    c_sig = _signal(roles["conflict"], rng)
    s_val = _val(rng, *s_sig[2:])
    c_hist = _val(rng, *c_sig[2:])
    c_manual = c_hist
    while abs(c_manual - c_hist) < (c_sig[3] - c_sig[2]) * 0.15:
        c_manual = _val(rng, *c_sig[2:])
    wo = lambda: f"WO-{A}-{rng.randint(10000, 99999)}"  # noqa: E731
    wo_maint, wo_clear, wo_noclear = wo(), wo(), wo()

    def hist(e, sig, t, v):
        return {"timestamp": t, "equipment": e, "signal": sig[0], "value": v, "unit": sig[1]}

    # ------------------------------------------------------------ stage 1: incident day
    others = [e for e in eqs if e not in (roles["sensor"], roles["conflict"])]
    h1 = [hist(roles["sensor"], s_sig, ts(D, *t_sensor), s_val),
          hist(roles["conflict"], c_sig, ts(D, *t_conf), c_hist)]
    for e in rng.sample(others, 2):
        sig = _signal(e, rng)
        h1.append(hist(e, sig, ts(D, rng.randint(8, 12), 30), _val(rng, *sig[2:])))
    h1.sort(key=lambda r: r["timestamp"])
    w.rows(1, f"historian_{D:%Y%m%d}.csv", h1)
    manual_row = {"timestamp": ts(D, *t_conf), "equipment": roles["conflict"], "signal": c_sig[0], "value": c_manual,
                  "unit": c_sig[1], "operator": rng.choice(TECHS)}
    w.rows(1, f"manual_rounds_{D:%Y%m%d}.csv", [manual_row])
    t_clr = D.replace(hour=t_alarm[0], minute=t_alarm[1]) + timedelta(minutes=22)
    alarms1 = [{"ts": ts(D, *t_alarm), "equipment": roles["alarm"], "alarm": gold_alarm, "state": "RAISED",
                "priority": rng.choice([1, 2])},
               {"ts": t_clr.strftime("%Y-%m-%dT%H:%M:%S"), "equipment": roles["alarm"], "alarm": gold_alarm,
                "state": "CLEARED", "priority": 2},
               {"ts": ts(D, 14, 5), "equipment": roles["hedge"], "alarm": "UNIT TRIP", "state": "RAISED", "priority": 1}]
    alarm_file = f"alarm_log_{D:%Y%m%d}.jsonl"
    w.rows(1, alarm_file, alarms1)
    m1 = [{"timestamp": ts(D, 10, 15), "equipment": roles["maint"], "work_order": wo_maint, "action": gold_action,
           "technician": rng.choice(TECHS), "status": "completed"},
          {"timestamp": ts(D, 12, 30), "equipment": roles["clear"], "work_order": wo_clear,
           "action": rng.choice([a for a in ACTIONS if a != gold_action]), "technician": rng.choice(TECHS),
           "status": "completed"},
          {"timestamp": ts(D, 13, 10), "equipment": roles["noclear"], "work_order": wo_noclear,
           "action": rng.choice([a for a in ACTIONS if a != gold_action]), "technician": rng.choice(TECHS),
           "status": "completed"}]
    w.rows(1, f"maintenance_log_{D:%Y%m%d}.csv", m1, "MAINTENANCE")
    p_insp = w.doc(1, f"inspection_{roles['insp']}_{D:%Y%m%d}.md",
                   f"# Inspection report\n\nInspection report for {roles['insp']}, {words}: the inspector recorded the "
                   f"following finding: {gold_finding}.\n")
    p_rca = w.doc(1, f"rca_{roles['rca']}_{D:%Y%m%d}.md",
                  f"# Root cause analysis\n\nRoot cause analysis for the trip of {roles['rca']} on {words}: the root "
                  f"cause was confirmed as {gold_cause}.\n")
    p_notes = w.doc(1, f"shift_notes_{D:%Y%m%d}.md",
                    f"# Shift notes\n\n{roles['hedge']} tripped at 14:05 on {words}. High vibration was observed "
                    f"before the trip; the root cause is under investigation and has not been determined.\n\n"
                    f"Repair work on {roles['noclear']} was completed on {words}. The functional test is pending "
                    f"and the unit has not been cleared for operation.\n")
    p_rts = w.doc(1, f"return_to_service_{roles['clear']}_{D:%Y%m%d}.md",
                  f"# Return-to-service record\n\nReturn-to-service record: {roles['clear']} was cleared for "
                  f"operation on {words} at 16:40 by the shift supervisor after the repair under work order "
                  f"{wo_clear}.\n")

    # ------------------------------------------------------------ stage 2: following operations
    h2 = []
    for k in range(3):
        d = D + timedelta(days=k)
        for h in range(0, 24, 2):
            for e in eqs:
                sig = _signal(e, rng, 0)
                if k == 0 and abs(h - t_sensor[0] - 0.5) < 1 and e == roles["sensor"] and sig[0] == s_sig[0]:
                    continue
                h2.append(hist(e, sig, ts(d, h), _val(rng, *sig[2:])))
    w.rows(2, f"historian_{D:%Y%m%d}_{(D + timedelta(days=2)):%Y%m%d}.csv", h2)
    w.rows(2, alarm_file, alarms1)                                   # exact re-delivery -> duplicates
    sib = [e for e in eqs if e != roles["alarm"]]
    a2 = []
    for k in range(3):
        d = D + timedelta(days=k)
        for _ in range(5):
            e = rng.choice(sib)
            t = d.replace(hour=rng.randint(0, 23), minute=rng.choice([0, 15, 30, 45]))
            a2.append({"ts": t.strftime("%Y-%m-%dT%H:%M:%S"), "equipment": e,
                       "alarm": gold_alarm if rng.random() < 0.6 else rng.choice(ALARMS), "state": "RAISED",
                       "priority": rng.choice([1, 2, 3])})
    a2.sort(key=lambda r: r["ts"])
    w.rows(2, f"alarm_log_{(D + timedelta(days=1)):%Y%m%d}_{(D + timedelta(days=2)):%Y%m%d}.jsonl", a2)
    m2 = []
    for k in (1, 2):
        d = D + timedelta(days=k)
        for e in rng.sample([x for x in eqs if x != roles["maint"]], 4):
            m2.append({"timestamp": ts(d, rng.randint(7, 16), rng.choice([0, 15, 30, 45])), "equipment": e,
                       "work_order": wo(), "action": rng.choice([a for a in ACTIONS if a != gold_action]),
                       "technician": rng.choice(TECHS), "status": "completed"})
    m2.sort(key=lambda r: r["timestamp"])
    w.rows(2, f"maintenance_log_{(D + timedelta(days=1)):%Y%m%d}_{(D + timedelta(days=2)):%Y%m%d}.csv", m2, "MAINTENANCE")
    n2 = "\n\n".join(f"{e} running normally on {_fmt_day(D + timedelta(days=k))}; no abnormal noise reported "
                     f"during the round." for k in (1, 2) for e in rng.sample(eqs, 3))
    w.doc(2, f"shift_notes_{(D + timedelta(days=2)):%Y%m%d}.md", "# Shift notes\n\n" + n2 + "\n")

    # ------------------------------------------------------------ stage 3: older history + long manual
    m3 = []
    for e in eqs:
        for _ in range(rng.randint(5, 9)):
            d = D - timedelta(days=rng.randint(2, 120))
            m3.append({"timestamp": ts(d, rng.randint(7, 16), rng.choice([0, 15, 30, 45])), "equipment": e,
                       "work_order": wo(), "action": rng.choice(ACTIONS), "technician": rng.choice(TECHS),
                       "status": "completed"})
    m3.sort(key=lambda r: r["timestamp"])
    w.rows(3, f"maintenance_history_{(D - timedelta(days=120)):%Y%m%d}_{(D - timedelta(days=1)):%Y%m%d}.csv", m3,
           "MAINTENANCE")
    a3 = []
    for _ in range(40):
        d = D - timedelta(days=rng.randint(1, 90), hours=rng.randint(0, 23))
        a3.append({"ts": d.replace(minute=rng.choice([0, 15, 30, 45]), second=0).strftime("%Y-%m-%dT%H:%M:%S"),
                   "equipment": rng.choice(eqs), "alarm": rng.choice(ALARMS), "state": "RAISED",
                   "priority": rng.choice([1, 2, 3])})
    a3.sort(key=lambda r: r["ts"])
    w.rows(3, f"alarm_history_{(D - timedelta(days=90)):%Y%m%d}_{(D - timedelta(days=1)):%Y%m%d}.jsonl", a3)
    for back in (30, 60):
        d = D - timedelta(days=back)
        w.doc(3, f"inspection_{roles['insp']}_{d:%Y%m%d}.md",
              f"# Inspection report\n\nInspection report for {roles['insp']}, {_fmt_day(d)}: the inspector recorded "
              f"the following finding: {rng.choice([f for f in FINDINGS if f != gold_finding])}.\n")
    manual = [f"# Equipment care manual, area {A}", "This manual describes routine care for the rotating equipment "
              "of the area. It is general guidance and does not record any specific event."]
    for e in eqs:
        manual.append(f"{e}: check lubrication weekly, record vibration on the round sheet, and compare readings "
                      f"against the previous month. Seal and bearing replacements are planned through the "
                      f"maintenance system with a work order.")
    manual += ["When a unit trips, the root cause must be confirmed by reliability engineering before it is "
               "recorded as established. Until then, operators note observations only.",
               "Return to service after repair requires a functional test and a supervisor signature; a unit is "
               "cleared for operation only by that signed record."] * 3
    w.doc(3, f"equipment_care_manual_{A}.md", "\n\n".join(manual) + "\n")

    # ------------------------------------------------------------ stage 4: later months
    h4 = []
    for k in range(3, 46):
        d = D + timedelta(days=k)
        for h in range(0, 24, 6):
            for e in eqs:
                sig = _signal(e, rng, 0)
                h4.append(hist(e, sig, ts(d, h), _val(rng, *sig[2:])))
    w.rows(4, f"historian_{(D + timedelta(days=3)):%Y%m%d}_{(D + timedelta(days=45)):%Y%m%d}.csv", h4)
    a4, m4 = [], []
    for _ in range(60):
        d = D + timedelta(days=rng.randint(3, 45), hours=rng.randint(0, 23))
        a4.append({"ts": d.replace(minute=rng.choice([0, 15, 30, 45]), second=0).strftime("%Y-%m-%dT%H:%M:%S"),
                   "equipment": rng.choice(eqs), "alarm": rng.choice(ALARMS), "state": "RAISED",
                   "priority": rng.choice([1, 2, 3])})
    for _ in range(30):
        k = rng.randint(4, 45)                      # never on D+3 (the wrong-date question's date)
        d = D + timedelta(days=k)
        m4.append({"timestamp": ts(d, rng.randint(7, 16), rng.choice([0, 15, 30, 45])), "equipment": rng.choice(eqs),
                   "work_order": wo(), "action": rng.choice(ACTIONS), "technician": rng.choice(TECHS),
                   "status": "completed"})
    a4.sort(key=lambda r: r["ts"])
    m4.sort(key=lambda r: r["timestamp"])
    w.rows(4, f"alarm_log_{(D + timedelta(days=3)):%Y%m%d}_{(D + timedelta(days=45)):%Y%m%d}.jsonl", a4)
    w.rows(4, f"maintenance_log_{(D + timedelta(days=4)):%Y%m%d}_{(D + timedelta(days=45)):%Y%m%d}.csv", m4,
           "MAINTENANCE")
    d20, d10 = D + timedelta(days=20), D + timedelta(days=10)
    w.doc(4, f"inspection_{roles['insp']}_{d20:%Y%m%d}.md",
          f"# Inspection report\n\nInspection report for {roles['insp']}, {_fmt_day(d20)}: the inspector recorded the "
          f"following finding: {rng.choice([f for f in FINDINGS if f != gold_finding])}.\n")
    w.doc(4, f"rca_{roles['hedge']}_{d10:%Y%m%d}.md",
          f"# Root cause analysis\n\nRoot cause analysis for the trip of {roles['hedge']} on {_fmt_day(d10)}: the "
          f"root cause was confirmed as {later_cause}.\n")

    (root / "manifest.json").write_text(json.dumps({
        "benchmark": BENCHMARK_VERSION, "scenario": A, "split": sc.split, "incident_day": day, "roles": roles,
        "stages": {str(k): {"name": STAGE_NAMES[k], "files": v} for k, v in w.manifest.items()},
        "synthetic": True}, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ questions + annotations
    def ids_of(rel_path: str, stage: str, pred) -> list[str]:
        return [e.event_id for e in load_any(root / rel_path, stage=stage) if pred(e)]

    def row_ids(name, row, stage="UTILITIES"):
        return ids_of(f"stage1/{name}", stage, lambda e: e.raw == {k: (str(v) if name.endswith(".csv") else v)
                                                                   for k, v in row.items()})

    def chunk_ids(p, needle):
        return ids_of(f"stage1/{p.name}", "MAINTENANCE", lambda e: needle in e.text)

    hh = lambda t: f"{t[0]:02d}:{t[1]:02d}"  # noqa: E731
    later = (D + timedelta(days=3)).strftime("%Y-%m-%d")
    qs = [
        ("maint_action", f"What maintenance action was recorded for {roles['maint']} on {day}?", "value",
         gold_action, "exact", row_ids(f"maintenance_log_{D:%Y%m%d}.csv", m1[0], "MAINTENANCE"), []),
        ("alarm_at_time", f"Which alarm was raised on {roles['alarm']} at {day} {hh(t_alarm)}?", "value",
         gold_alarm, "exact", row_ids(alarm_file, alarms1[0]), row_ids(alarm_file, alarms1[1])),
        ("sensor_at_time", f"What was the {s_sig[0].replace('_', ' ')} reading for {roles['sensor']} at {day} "
                           f"{hh(t_sensor)}?", "value", f"{s_val} {s_sig[1]}", "exact",
         row_ids(f"historian_{D:%Y%m%d}.csv", h1[[r['equipment'] for r in h1].index(roles['sensor'])]), []),
        ("inspection_finding", f"What inspection finding was recorded for {roles['insp']} on {day}?", "value",
         gold_finding, "contains", chunk_ids(p_insp, gold_finding), []),
        ("rca_documented", f"What was the confirmed root cause of the trip of {roles['rca']} on {day}?", "value",
         gold_cause, "contains", chunk_ids(p_rca, gold_cause), []),
        ("rca_undocumented", f"What was the root cause of the trip of {roles['hedge']} on {day}?", "abstain",
         None, None, [], chunk_ids(p_notes, "under investigation")),
        ("clearance_documented", f"Was {roles['clear']} cleared for operation after the repair on {day}?", "value",
         f"{roles['clear']} was cleared for operation", "contains", chunk_ids(p_rts, "cleared for operation"), []),
        ("clearance_undocumented", f"Was {roles['noclear']} cleared for operation after the repair on {day}?",
         "abstain", None, None, [], chunk_ids(p_notes, "functional test is pending")),
        ("conflicting_readings", f"What was the {c_sig[0].replace('_', ' ')} reading for {roles['conflict']} at "
                                 f"{day} {hh(t_conf)}?", "conflict", None, None,
         row_ids(f"historian_{D:%Y%m%d}.csv", h1[[r['equipment'] for r in h1].index(roles['conflict'])]) +
         row_ids(f"manual_rounds_{D:%Y%m%d}.csv", manual_row), []),
        ("unknown_equipment", f"What maintenance action was recorded for {roles['missing']} on {day}?", "abstain",
         None, None, [], []),
        ("wrong_date", f"What maintenance action was recorded for {roles['maint']} on {later}?", "abstain",
         None, None, [], row_ids(f"maintenance_log_{D:%Y%m%d}.csv", m1[0], "MAINTENANCE")),
        # Paraphrases of answerable families, written once with the templates above and never tuned:
        # they probe robustness to wording that does not reuse the records' vocabulary.
        ("maint_action_paraphrase", f"What work was carried out on {roles['maint']} on {day}?", "value",
         gold_action, "exact", row_ids(f"maintenance_log_{D:%Y%m%d}.csv", m1[0], "MAINTENANCE"), []),
        ("alarm_paraphrase", f"Which alarm did {roles['alarm']} trigger at {day} {hh(t_alarm)}?", "value",
         gold_alarm, "exact", row_ids(alarm_file, alarms1[0]), []),
        ("inspection_paraphrase", f"What did the inspection of {roles['insp']} on {day} find?", "value",
         gold_finding, "contains", chunk_ids(p_insp, gold_finding), []),
        ("rca_paraphrase", f"Why did {roles['rca']} trip on {day}?", "value", gold_cause, "contains",
         chunk_ids(p_rca, gold_cause), []),
    ]
    questions, anns = [], []
    for n, (fam, text, kind, ans, match, req, rel) in enumerate(qs, 1):
        qid = f"{A}-Q{n:02d}"
        questions.append(LogQuestion(qid, A, sc.split, text))
        anns.append(LogAnnotation(qid, A, sc.split, fam, kind, ans, match, req, rel))
    return questions, anns


def write_benchmark(out_dir: str | Path, seed: int = 2026, n_dev: int = 3, n_test: int = 5) -> dict:
    out = Path(out_dir)
    rng = random.Random(seed)
    areas = [f"P{i}" for i in range(1, n_dev + n_test + 1)]
    rng.shuffle(areas)
    splits = {a: ("dev" if i < n_dev else "test") for i, a in enumerate(areas)}
    base = datetime(2026, 2, 2)
    all_q, all_a = [], []
    for i, a in enumerate(sorted(areas)):
        sc = Scenario(a, splits[a], (base + timedelta(days=11 * i)).strftime("%Y-%m-%d"))
        root = out / sc.split / a
        qs, anns = build_scenario(sc, root, seed * 100 + i)
        missing = [x.question_id for x in anns if x.answer_kind in ("value", "conflict") and not x.required_evidence_ids]
        if missing:
            raise RuntimeError(f"evidence IDs not resolved for {missing}")
        all_q += qs
        all_a += anns
    (out / "questions.jsonl").write_text("".join(json.dumps(asdict(q)) + "\n" for q in all_q), encoding="utf-8")
    (out / "annotations").mkdir(exist_ok=True)
    (out / "annotations" / "annotations.jsonl").write_text("".join(json.dumps(asdict(a)) + "\n" for a in all_a),
                                                          encoding="utf-8")
    manifest = {"benchmark": BENCHMARK_VERSION, "seed": seed, "synthetic": True,
                "splits": {s: sorted(a for a, v in splits.items() if v == s) for s in ("dev", "test")},
                "n_questions": {s: sum(q.split == s for q in all_q) for s in ("dev", "test")},
                "note": "Fictional data imitating operational log formats. Not plant data."}
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_questions(root: str | Path, split: str) -> list[LogQuestion]:
    rows = [json.loads(x) for x in (Path(root) / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return [LogQuestion(**r) for r in rows if r["split"] == split]


def load_annotations(root: str | Path, split: str) -> dict[str, LogAnnotation]:
    """Evaluation-only: called by the scorer after a system has produced its output."""
    p = Path(root) / "annotations" / "annotations.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]
    return {r["question_id"]: LogAnnotation(**r) for r in rows if r["split"] == split}


def stage_files(scenario_root: str | Path, stage: int) -> list[dict]:
    m = json.loads((Path(scenario_root) / "manifest.json").read_text(encoding="utf-8"))
    return m["stages"][str(stage)]["files"]
