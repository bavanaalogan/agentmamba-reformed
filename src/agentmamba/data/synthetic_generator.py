"""Reproducible generator of explicitly FICTIONAL plant-wide steel-plant event histories.

Each *campaign* is an independent fictional plant history (its own asset/heat/ladle IDs,
prefixed with the campaign code). Campaigns are the unit of train/val/test splitting, so
no event, entity or question is shared across splits.

All values are synthetic and every event carries ``source_type="synthetic"`` and a
``synthetic://`` source reference. Nothing here is a real industrial observation.

Outputs (see ``write_dataset``):
  events.jsonl        events delivered to ingestion, in stream (timestamp) order
  withheld.jsonl      events that exist in the fictional world but were never delivered
                      (scenario "evidence never stored"); never ingested
  questions.jsonl     what the live system may see (text + initial-context event ids)
  annotations.jsonl   gold labels; only the evaluator and gate/controller *training* read it
  split_manifest.json campaign -> split, seed, generator config
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from agentmamba.memory.schema import Annotation, Event, Observation, Question

TZ = "plant-local (synthetic)"
GENERATOR_VERSION = "synth-v1"

NORMAL_FINDINGS = ["lining condition normal", "no abnormality observed", "refractory wear within limits"]
WARNING_FINDINGS = ["hairline crack at slag line", "localized erosion near tap hole", "minor spalling on ladle bottom"]
CAST_ABNORMALITIES = [
    "intermittent mould level fluctuation",
    "suspected submerged entry nozzle clogging",
    "casting speed reduced after tundish temperature drop",
    "mould flux feeding interruption",
]
DEFECTS = ["edge cracks", "surface scale pits", "longitudinal crack", "sliver"]
CONFIRMED_CAUSES = ["mould flux feeding interruption", "worn segment rollers", "excessive descaling pressure loss"]
WO_ACTIONS = {
    "ladle": ["reline slag line bricks", "repair tap hole well block", "replace slide gate plate"],
    "caster": ["replace segment 4 rollers", "recalibrate mould level sensor", "clean spray nozzles in zone 2"],
    "mill": ["replace work rolls on stand F3", "repair descaler header valve", "realign edger guides"],
    "bof": ["repair oxygen lance tip", "reline taphole", "replace sublance probe holder"],
}
OUTCOMES = ["effective", "partially effective", "not effective"]
OUTCOME_NOTES = {
    "effective": "condition resolved; no recurrence observed during follow-up",
    "partially effective": "condition improved but symptom recurred at reduced frequency",
    "not effective": "symptom persisted after intervention; further investigation requested",
}


@dataclass
class GenConfig:
    n_campaigns: int = 30
    heats_per_campaign: int = 48
    ladles_per_campaign: int = 6
    noise_logs_per_heat: int = 4
    splits: dict = field(default_factory=lambda: {"train": 16, "val": 6, "test": 8})
    seed: int = 13


class CampaignBuilder:
    def __init__(self, idx: int, cfg: GenConfig, split: str):
        self.idx = idx
        self.cfg = cfg
        self.split = split
        self.cid = f"C{idx:02d}"
        self.rng = random.Random(cfg.seed * 1000 + idx)
        self.events: list[Event] = []
        self.withheld: list[Event] = []
        self.questions: list[tuple[Question, Annotation]] = []
        self._seq = 0
        self.t0 = datetime(2024, 1, 1) + timedelta(days=12 * idx)
        self.assets = {
            "bf": f"{self.cid}-BF1", "bof": [f"{self.cid}-BOF1", f"{self.cid}-BOF2"],
            "caster": f"{self.cid}-CC1", "mill": f"{self.cid}-HSM1",
        }
        self.ladles = [f"{self.cid}-LD{n:02d}" for n in range(1, cfg.ladles_per_campaign + 1)]
        self.heats: list[dict] = []
        self.inspections: dict[str, list[Event]] = {ld: [] for ld in self.ladles}
        self.work_orders: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def _ev(self, stage: str, etype: str, t: datetime, text: str, *, withheld: bool = False, **kw) -> Event:
        self._seq += 1
        eid = f"{self.cid}-E{self._seq:05d}"
        obs = kw.pop("observations", [])
        ev = Event(
            event_id=eid, stage=stage, event_type=etype, timestamp=t.isoformat(timespec="seconds"),
            timestamp_raw=t.strftime("%d.%m.%Y %H:%M"), timezone=TZ, text=text,
            source_ref=kw.pop("source_ref", f"synthetic://{self.cid}/{etype}/{eid}"),
            source_type="synthetic", campaign_id=self.cid, observations=obs, **kw,
        )
        (self.withheld if withheld else self.events).append(ev)
        return ev

    def _pick(self, options):
        return self.rng.choice(options)

    def _q(self, family, qtype, text, ctx, ann_kw):
        qid = f"{self.cid}-Q{len(self.questions) + 1:02d}"
        q = Question(question_id=qid, campaign_id=self.cid, split=self.split, text=text,
                     initial_context_ids=[e.event_id for e in ctx])
        ann = Annotation(question_id=qid, family=family, question_type=qtype, **ann_kw)
        self.questions.append((q, ann))

    def _noise_ctx(self, around: datetime, n: int = 2) -> list[Event]:
        pool = [e for e in self.events if e.event_type in ("shift_log", "sensor_summary", "ladle_preheat_log")
                and abs((datetime.fromisoformat(e.timestamp) - around).total_seconds()) < 6 * 3600]
        return self.rng.sample(pool, min(n, len(pool)))

    # ------------------------------------------------------------------ world
    def build(self) -> "CampaignBuilder":
        self._documents()
        self._production()
        self._ladle_incident()
        self._maintenance()
        self._conflict_inspection()
        self.events.sort(key=lambda e: (e.timestamp, e.event_id))
        self._questions()
        return self

    def _documents(self):
        c = self.cid
        self.docs = {}
        specs = [("SOP-LP", "maximum preheat-to-tap interval", "min", 90, 60, True),
                 ("SOP-LR", "maximum heats between ladle slag-line repairs", "heats", 40, 32, True),
                 ("SOP-TD", "minimum tundish superheat", "degC", 25, None, False)]
        specs = self.rng.sample(specs, 2) if self.rng.random() < 0.5 else specs[:2]
        for code, param, unit, v1, v2, revised in specs:
            doc_id = f"{c}-{code}"
            t1 = self.t0 + timedelta(hours=1)
            e1 = self._ev("MAINTENANCE", "procedure_doc", t1,
                          f"Procedure {doc_id} revision 1 (effective {t1.date()}): {param} is {v1} {unit}.",
                          doc_id=doc_id, version=1, effective_date=str(t1.date()),
                          observations=[Observation("limit_value", v1, unit, str(v1))])
            entry = {"doc_id": doc_id, "param": param, "unit": unit, "v1": e1, "v2": None}
            if revised:
                t2 = self.t0 + timedelta(days=3, hours=self.rng.randint(0, 20))
                e2 = self._ev("MAINTENANCE", "procedure_doc", t2,
                              f"Procedure {doc_id} revision 2 (effective {t2.date()}), supersedes revision 1: "
                              f"{param} is {v2} {unit}.",
                              doc_id=doc_id, version=2, effective_date=str(t2.date()), supersedes=e1.event_id,
                              observations=[Observation("limit_value", v2, unit, str(v2))])
                entry["v2"] = e2
            self.docs[doc_id] = entry

    def _production(self):
        c, rng = self.cid, self.rng
        n = self.cfg.heats_per_campaign
        self.broken_link_heat = rng.randrange(8, n - 8)
        t = self.t0 + timedelta(hours=2)
        ladle_cursor = 0
        last_insp = {ld: None for ld in self.ladles}
        for i in range(n):
            t = t + timedelta(minutes=rng.randint(130, 170))
            heat = f"{c}-H{i + 1:04d}"
            batch = f"{c}-HM{i // 2 + 1:03d}"
            seq = f"{c}-CS{i // 4 + 1:02d}"
            slab, coil = f"{c}-SL{i + 1:04d}", f"{c}-CL{i + 1:04d}"
            bof = self.assets["bof"][i % 2]
            if i % 2 == 0:
                hm_t = rng.randint(1440, 1510)
                self._ev("IRONMAKING", "hot_metal_tap", t - timedelta(minutes=70),
                         f"Blast furnace {self.assets['bf']} tapped hot metal batch {batch}: {hm_t} degC, "
                         f"Si {rng.uniform(0.3, 0.7):.2f} %.", asset_id=self.assets["bf"], batch_id=batch,
                         observations=[Observation("hot_metal_temp", hm_t, "degC")])
            tap = rng.randint(1615, 1685)
            carbon = round(rng.uniform(0.03, 0.09), 3)
            oxy = rng.randint(380, 820)
            bof_ev = self._ev("STEELMAKING", "bof_heat_record", t,
                              self._pick([f"Heat {heat} blown in {bof} using hot metal batch {batch}; tap temperature "
                                          f"{tap} degC, end-blow carbon {carbon} %, oxygen {oxy} ppm.",
                                          f"{bof} heat {heat} (charge batch {batch}) tapped at {tap} degC; "
                                          f"C {carbon} %, O {oxy} ppm at end of blow."]),
                              asset_id=bof, heat_id=heat, batch_id=batch,
                              observations=[Observation("tap_temperature", tap, "degC"),
                                            Observation("carbon", carbon, "%"), Observation("oxygen", oxy, "ppm")])
            # ladle assignment (one deliberately incomplete record: broken identifier link)
            ladle = self.ladles[ladle_cursor % len(self.ladles)]
            ladle_cursor += rng.choice([1, 1, 2])
            ta = t + timedelta(minutes=8)
            if i == self.broken_link_heat:
                assign = self._ev("LADLE", "ladle_assignment", ta,
                                  f"Ladle assignment entry for heat {heat}: ladle number not recorded (entry incomplete).",
                                  heat_id=heat, severity="minor")
                ladle_for_heat = None
            else:
                assign = self._ev("LADLE", "ladle_assignment", ta,
                                  self._pick([f"Heat {heat} tapped into ladle {ladle}.",
                                              f"Ladle {ladle} assigned to heat {heat} at tapping."]),
                                  heat_id=heat, ladle_id=ladle)
                ladle_for_heat = ladle
            # periodic refractory inspection of the ladle before this heat
            if ladle_for_heat and (last_insp[ladle] is None or i - last_insp[ladle] >= rng.randint(2, 4)):
                last_insp[ladle] = i
                finding = self._pick(NORMAL_FINDINGS)
                if rng.random() < 0.06:
                    finding = self._pick(WARNING_FINDINGS)  # distractor warning, no later incident
                self._inspection(ladle, ta - timedelta(minutes=rng.randint(25, 60)), finding)
            lf_t = rng.randint(1570, 1610)
            self._ev("LADLE", "ladle_treatment", t + timedelta(minutes=45),
                     f"Ladle furnace treatment of heat {heat}" + (f" in ladle {ladle_for_heat}" if ladle_for_heat else "")
                     + f": final temperature {lf_t} degC, argon stirring completed.",
                     heat_id=heat, ladle_id=ladle_for_heat,
                     observations=[Observation("lf_final_temperature", lf_t, "degC")])
            abn = self._pick(CAST_ABNORMALITIES) if rng.random() < 0.3 else None
            speed = round(rng.uniform(0.9, 1.3), 2)
            sh = rng.randint(18, 38)
            cast = self._ev("CASTING", "cast_record", t + timedelta(minutes=95),
                            f"Caster {self.assets['caster']} cast heat {heat} in sequence {seq} producing slab {slab}; "
                            f"speed {speed} m/min, superheat {sh} degC." + (f" Note: {abn}." if abn else " No abnormality noted."),
                            asset_id=self.assets["caster"], heat_id=heat, cast_seq_id=seq, slab_id=slab,
                            severity="minor" if abn else None,
                            observations=[Observation("casting_speed", speed, "m/min"), Observation("superheat", sh, "degC")]
                            + ([Observation("abnormality", abn)] if abn else []))
            th = round(rng.uniform(2.0, 6.0), 2)
            roll = self._ev("ROLLING", "rolling_record", t + timedelta(hours=6),
                            f"Slab {slab} rolled on {self.assets['mill']} into coil {coil}; finishing temperature "
                            f"{rng.randint(860, 910)} degC, gauge {th} mm.",
                            asset_id=self.assets["mill"], slab_id=slab, coil_id=coil,
                            observations=[Observation("gauge", th, "mm")])
            defect = self._pick(DEFECTS) if rng.random() < 0.3 else None
            insp = self._ev("QUALITY", "coil_inspection", t + timedelta(hours=8),
                            f"Coil {coil} inspection: " + (f"rejected, {defect} detected." if defect else "passed, no surface defects."),
                            coil_id=coil, severity="major" if defect else None,
                            observations=[Observation("result", "rejected" if defect else "passed")]
                            + ([Observation("defect", defect)] if defect else []))
            self.heats.append(dict(i=i, heat=heat, batch=batch, seq=seq, slab=slab, coil=coil, bof=bof_ev,
                                   assign=assign, ladle=ladle_for_heat, cast=cast, abn=abn, roll=roll,
                                   insp=insp, defect=defect, t=t))
            self._noise(t, heat, ladle_for_heat)

    def _inspection(self, ladle, t, finding, severity="info", **kw) -> Event:
        th = self.rng.randint(120, 175)
        ev = self._ev("LADLE", "ladle_inspection", t,
                      self._pick([f"Refractory inspection of ladle {ladle}: {finding}; lining thickness {th} mm.",
                                  f"Ladle {ladle} refractory check - {finding}, residual lining {th} mm."]),
                      ladle_id=ladle, severity=severity,
                      observations=[Observation("finding", finding), Observation("lining_thickness", th, "mm")], **kw)
        self.inspections[ladle].append(ev)
        return ev

    def _noise(self, t, heat, ladle):
        rng, a = self.rng, self.assets
        gens = [
            lambda tt: self._ev("STEELMAKING", "shift_log", tt, self._pick([
                f"Shift handover at {a['bof'][0]}: routine operation, no remarks.",
                f"Converter shop shift log: production on schedule, crane availability normal, no safety observations."]),
                asset_id=a["bof"][0]),
            lambda tt: self._ev("CASTING", "sensor_summary", tt,
                                f"Hourly summary for {a['caster']} mould: temperature, flow and level signals within nominal band.",
                                asset_id=a["caster"]),
            lambda tt: self._ev("CASTING", "caster_alarm", tt, self._pick([
                f"Mould level alarm on {a['caster']} triggered and auto-cleared within {rng.randint(2, 9)} s; no operator action required.",
                f"High cooling-water temperature alarm on {a['caster']} acknowledged; reading returned to normal, nuisance alarm."]),
                asset_id=a["caster"], severity="major"),
            lambda tt: self._ev("ROLLING", "sensor_summary", tt,
                                f"Hourly summary for {a['mill']}: roll force, speed and temperature trends nominal.",
                                asset_id=a["mill"]),
            lambda tt: self._ev("UTILITIES", "sensor_summary", tt,
                                f"Utilities log: oxygen plant and compressed air pressures nominal; no deviations.", asset_id=f"{self.cid}-UT1"),
        ]
        if ladle:
            gens.append(lambda tt: self._ev("LADLE", "ladle_preheat_log", tt,
                                            f"Ladle {ladle} preheated on station 2 to target lining temperature.",
                                            ladle_id=ladle))
        for _ in range(self.cfg.noise_logs_per_heat):
            self._pick(gens)(t + timedelta(minutes=rng.randint(-60, 120)))

    def _ladle_incident(self):
        rng = self.rng
        ladle = rng.choice(self.ladles)
        used = [h for h in self.heats if h["ladle"] == ladle]
        if len(used) < 6:
            self.incident = None
            return
        # incident on the ladle's last use, so it is never assigned again after removal
        warn_heat, inc_heat = used[1], used[-1]
        w_t = warn_heat["assign"].timestamp
        warning = self._pick(WARNING_FINDINGS)
        w_ev = self._inspection(ladle, datetime.fromisoformat(w_t) - timedelta(minutes=30), warning, severity="info")
        i_t = datetime.fromisoformat(inc_heat["assign"].timestamp) + timedelta(minutes=70)
        inc = self._ev("LADLE", "ladle_incident", i_t,
                       f"Ladle {ladle} removed from service after steel penetration observed at the shell during heat "
                       f"{inc_heat['heat']}; heat transferred safely.", ladle_id=ladle, heat_id=inc_heat["heat"],
                       severity="critical")
        self.incident = dict(ladle=ladle, warning=w_ev, warning_text=warning, incident=inc)

    def _maintenance(self):
        rng, a = self.rng, self.assets
        targets = [("caster", a["caster"]), ("caster", a["caster"]), ("mill", a["mill"]), ("bof", a["bof"][1]),
                   ("ladle", rng.choice(self.ladles)), ("ladle", rng.choice(self.ladles)), ("mill", a["mill"])]
        days = rng.sample(range(0, 6), len(targets) - 1) + [rng.randint(0, 5)]
        for n, ((kind, asset), day) in enumerate(zip(targets, days), start=1):
            wo = f"{self.cid}-WO{n:03d}"
            t = self.t0 + timedelta(days=day, hours=rng.randint(6, 18))
            action = self._pick(WO_ACTIONS[kind])
            wo_ev = self._ev("MAINTENANCE", "work_order", t, f"Work order {wo} opened on {asset}: {action}.",
                             asset_id=asset, work_order_id=wo, ladle_id=asset if kind == "ladle" else None,
                             severity="minor", observations=[Observation("action", action)])
            outcome = OUTCOMES[n % 3] if n <= 3 else rng.choice(OUTCOMES)
            ft = t + timedelta(days=1, hours=rng.randint(0, 12))
            withheld = n == 4  # this follow-up exists but was never delivered to ingestion
            fu = self._ev("MAINTENANCE", "maintenance_followup", ft,
                          f"Follow-up inspection for work order {wo} on {asset}: intervention {outcome}; {OUTCOME_NOTES[outcome]}.",
                          asset_id=asset, work_order_id=wo, ladle_id=asset if kind == "ladle" else None,
                          related_event_ids=[wo_ev.event_id], withheld=withheld,
                          severity="minor" if outcome != "effective" else "info",
                          observations=[Observation("outcome", outcome)])
            self.work_orders.append(dict(wo=wo, asset=asset, ev=wo_ev, fu=fu, outcome=outcome, withheld=withheld,
                                         date=str(t.date()), kind=kind))

    def _conflict_inspection(self):
        ladle = self.rng.choice([ld for ld in self.ladles if not self.incident or ld != self.incident["ladle"]])
        base = self.inspections[ladle][len(self.inspections[ladle]) // 2]
        t = datetime.fromisoformat(base.timestamp) + timedelta(minutes=12)
        other = base.obs("lining_thickness").value - self.rng.randint(12, 25)
        dup = self._ev("LADLE", "ladle_inspection", t,
                       f"Manual inspection sheet for ladle {ladle}: lining thickness {other} mm; "
                       f"{base.obs('finding').value}.", ladle_id=ladle, severity="info",
                       source_ref=f"synthetic://{self.cid}/manual_sheet/{self._seq + 1}",
                       observations=[Observation("finding", base.obs("finding").value),
                                     Observation("lining_thickness", other, "mm")])
        self.conflict = dict(ladle=ladle, a=base, b=dup)

    # ------------------------------------------------------------------ questions
    def _ann(self, *, stages, kind, answer, evidence, initial, cross, temporal=False, answerable=True, notes=""):
        return dict(stages=stages, answer_kind=kind, expected_answer=answer, answerable=answerable,
                    required_evidence_ids=[e.event_id for e in evidence], relationship_ids=[],
                    expected_sources=[e.source_ref for e in evidence], initial_sufficient=initial,
                    retrieval_required=not initial and kind not in ("clarify",), cross_stage=cross,
                    temporal_sensitive=temporal, notes=notes)

    def _questions(self):
        rng, H = self.rng, [h for h in self.heats if h["ladle"]]
        # F01 direct-context measurement
        for h in rng.sample(H, 2):
            name, label = rng.choice([("tap_temperature", "tap temperature"), ("carbon", "end-blow carbon content")])
            o = h["bof"].obs(name)
            self._q("F01_direct_context", "measurement",
                    rng.choice([f"What {label} was recorded for heat {h['heat']}?",
                                f"Report the {label} measured for heat {h['heat']}."]),
                    [h["bof"]] + self._noise_ctx(h["t"]),
                    self._ann(stages=["STEELMAKING"], kind="value", answer=f"{o.value} {o.unit}", evidence=[h["bof"]],
                              initial=True, cross=False))
        # F02 earlier stage (ladle of heat), plus a direct-context control
        for n, h in enumerate(rng.sample(H, 3)):
            direct = n == 2
            ctx = [h["assign"] if direct else h["cast"]] + self._noise_ctx(h["t"])
            self._q("F01_direct_context" if direct else "F02_earlier_stage", "ladle_of_heat",
                    rng.choice([f"Which ladle carried heat {h['heat']}?", f"Heat {h['heat']} was tapped into which ladle?"]),
                    ctx, self._ann(stages=["CASTING", "LADLE"], kind="value", answer=h["ladle"], evidence=[h["assign"]],
                                   initial=direct, cross=not direct))
        # F03 relationship chain coil -> slab -> cast sequence
        for h in rng.sample(H, 2):
            self._q("F03_relationship_chain", "cast_seq_of_coil",
                    rng.choice([f"Which casting sequence produced the slab that was rolled into coil {h['coil']}?",
                                f"Coil {h['coil']} came from a slab cast in which casting sequence?"]),
                    [h["insp"]], self._ann(stages=["QUALITY", "ROLLING", "CASTING"], kind="value", answer=h["seq"],
                                           evidence=[h["roll"], h["cast"]], initial=False, cross=True))
        # F04 rare early warning before an incident (longitudinal)
        if self.incident:
            inc = self.incident
            self._q("F04_rare_warning", "ladle_warning",
                    rng.choice([f"Ladle {inc['ladle']} was removed from service on {inc['incident'].timestamp[:10]}. "
                                f"What earlier refractory warning had been recorded for this ladle?",
                                f"Before ladle {inc['ladle']} was taken out of service on {inc['incident'].timestamp[:10]}, "
                                f"which refractory warning sign was noted for it?"]),
                    [inc["incident"]], self._ann(stages=["LADLE"], kind="value", answer=inc["warning_text"],
                                                 evidence=[inc["warning"]], initial=False, cross=False, temporal=True,
                                                 notes="warning recorded with low severity long before the incident"))
        # F07/F08 maintenance outcome (incl. failed / partial), F12 never stored, F18 precise date
        delivered = [w for w in self.work_orders if not w["withheld"]]
        for n, w in enumerate(rng.sample(delivered, 3)):
            direct = n == 2
            fam = "F01_direct_context" if direct else ("F08_failed_intervention" if w["outcome"] != "effective"
                                                       else "F07_maintenance_outcome")
            self._q(fam, "wo_outcome",
                    rng.choice([f"What was the documented outcome of work order {w['wo']}?",
                                f"Was work order {w['wo']} effective according to the follow-up record?"]),
                    [w["fu"] if direct else w["ev"]],
                    self._ann(stages=["MAINTENANCE"], kind="value", answer=w["outcome"], evidence=[w["fu"]],
                              initial=direct, cross=False))
        lost = next(w for w in self.work_orders if w["withheld"])
        self._q("F12_evidence_never_stored", "wo_outcome", f"What was the documented outcome of work order {lost['wo']}?",
                [lost["ev"]], self._ann(stages=["MAINTENANCE"], kind="abstain", answer=None, evidence=[], initial=False,
                                        cross=False, answerable=False, notes="follow-up exists in the fictional world but was never delivered"))
        by_day: dict[tuple, list] = {}
        for w in self.work_orders:
            by_day.setdefault((w["asset"], w["date"]), []).append(w)
        uniq = [v[0] for v in by_day.values() if len(v) == 1 and sum(1 for x in self.work_orders if x["asset"] == v[0]["asset"]) > 1]
        if uniq:
            w = rng.choice(uniq)
            self._q("F18_precise_time", "wo_opened_on", f"Which work order was opened on {w['asset']} on {w['date']}?",
                    [], self._ann(stages=["MAINTENANCE"], kind="value", answer=w["wo"], evidence=[w["ev"]], initial=False,
                                  cross=False, temporal=True))
        # F09 conflict + non-conflict control
        cf = self.conflict
        self._q("F09_conflicting_records", "ladle_thickness_at",
                f"What lining thickness was recorded for ladle {cf['ladle']} at the inspection on {cf['a'].timestamp[:10]}?",
                [], self._ann(stages=["LADLE"], kind="conflict", answer=None, evidence=[cf["a"], cf["b"]], initial=False,
                              cross=False, temporal=True, answerable=False))
        cands = [e for ld, evs in self.inspections.items() for e in evs
                 if ld != cf["ladle"] and sum(1 for x in evs if x.timestamp[:10] == e.timestamp[:10]) == 1]
        if cands:
            e = rng.choice(cands)
            o = e.obs("lining_thickness")
            self._q("F09_conflicting_records", "ladle_thickness_at",
                    f"What lining thickness was recorded for ladle {e.ladle_id} at the inspection on {e.timestamp[:10]}?",
                    [], self._ann(stages=["LADLE"], kind="value", answer=f"{o.value} {o.unit}", evidence=[e], initial=False,
                                  cross=False, temporal=True, notes="non-conflicting control"))
        # F10 outdated documentation (+ unrevised control)
        for d in self.docs.values():
            cur = d["v2"] or d["v1"]
            o = cur.obs("limit_value")
            self._q("F10_outdated_document", "procedure_current_value",
                    f"According to the current revision of procedure {d['doc_id']}, what is the {d['param']}?",
                    # Declared policy: "current revision" cannot be established from the context alone,
                    # so retrieval is required even when the revision on screen is in fact current.
                    [d["v1"]], self._ann(stages=["MAINTENANCE"], kind="value", answer=f"{o.value} {o.unit}",
                                         evidence=[cur] if cur is d["v1"] else [d["v1"], cur], initial=False,
                                         cross=False, temporal=True,
                                         notes="revised" if d["v2"] else "unrevised control"))
        # F11 broken identifier link
        bh = self.heats[self.broken_link_heat]
        self._q("F11_missing_identifier", "ladle_of_heat", f"Which ladle carried heat {bh['heat']}?", [bh["cast"]],
                self._ann(stages=["CASTING", "LADLE"], kind="abstain", answer=None, evidence=[bh["assign"]], initial=False,
                          cross=True, answerable=False, notes="assignment record lacks the ladle identifier"))
        # F13 unanswerable (measurement never recorded)
        h = rng.choice(H)
        self._q("F13_unanswerable", "measurement", f"What hydrogen content was measured for heat {h['heat']}?",
                [h["bof"]], self._ann(stages=["STEELMAKING"], kind="abstain", answer=None, evidence=[], initial=False,
                                      cross=False, answerable=False))
        # F14 ambiguous (no identifier)
        self._q("F14_ambiguous", "ambiguous", rng.choice([
            "What did the last refractory inspection of the ladle find?",
            "Which work order was opened on the caster last week?",
            "What tap temperature did the heat have?"]), [],
            self._ann(stages=[], kind="clarify", answer=None, evidence=[], initial=False, cross=False, answerable=False))
        # F15 correlation without established cause; F15b documented cause
        defect_heats = [h for h in H if h["defect"]]
        corr = [h for h in defect_heats if h["abn"]]
        chosen = None
        if corr:
            h = chosen = rng.choice(corr)
            rca = self._ev("QUALITY", "rca_report", datetime.fromisoformat(h["insp"].timestamp) + timedelta(hours=20),
                           f"Root-cause analysis for {h['defect']} on coil {h['coil']}: inconclusive. Casting note "
                           f"'{h['abn']}' recorded for the source heat is under review; causal link not confirmed.",
                           coil_id=h["coil"], severity="minor",
                           observations=[Observation("rca_status", "inconclusive")])
            self.events.sort(key=lambda e: (e.timestamp, e.event_id))
            self._q("F15_correlation_not_causation", "coil_defect_cause",
                    f"What caused the {h['defect']} on coil {h['coil']}?", [h["insp"]],
                    self._ann(stages=["QUALITY", "ROLLING", "CASTING"], kind="hypothesis", answer=h["abn"],
                              evidence=[h["roll"], h["cast"], rca], initial=False, cross=True, answerable=False,
                              notes="association exists, cause not established"))
        rest = [h for h in defect_heats if h is not chosen]
        if rest:
            h = rng.choice(rest)
            cause = rng.choice(CONFIRMED_CAUSES)
            rca = self._ev("QUALITY", "rca_report", datetime.fromisoformat(h["insp"].timestamp) + timedelta(hours=30),
                           f"Root-cause analysis for {h['defect']} on coil {h['coil']}: confirmed root cause is {cause} "
                           f"(verified by trial and inspection).", coil_id=h["coil"], severity="minor",
                           observations=[Observation("rca_status", "confirmed"), Observation("cause", cause)])
            self.events.sort(key=lambda e: (e.timestamp, e.event_id))
            self._q("F15b_documented_cause", "coil_defect_cause",
                    f"What caused the {h['defect']} on coil {h['coil']}?", [h["insp"]],
                    self._ann(stages=["QUALITY"], kind="value", answer=cause, evidence=[rca], initial=False, cross=False))
        # F16 multi-stage chain: coil -> slab -> heat -> ladle -> last inspection before the heat
        chain = [h for h in H if self._last_insp_before(h)]
        if chain:
            h = rng.choice(chain)
            ins = self._last_insp_before(h)
            self._q("F16_multi_stage", "upstream_ladle_condition",
                    f"Coil {h['coil']} is under review. What refractory condition was recorded for the "
                    f"ladle that carried its heat, at that ladle's last inspection before the heat?", [h["insp"]],
                    self._ann(stages=["QUALITY", "ROLLING", "CASTING", "LADLE"], kind="value",
                              answer=ins.obs("finding").value, evidence=[h["roll"], h["cast"], h["assign"], ins],
                              initial=False, cross=True, temporal=True))
        # F17 similar-but-irrelevant incidents: casting abnormality for a specific heat
        ab = [h for h in H if h["abn"]]
        if ab:
            h = rng.choice(ab)
            self._q("F17_similar_irrelevant", "caster_abnormality",
                    f"What abnormality was noted at the caster while casting heat {h['heat']}?", [h["bof"]],
                    self._ann(stages=["STEELMAKING", "CASTING"], kind="value", answer=h["abn"], evidence=[h["cast"]],
                              initial=False, cross=True, notes="same abnormality text occurs on other heats"))

    def _last_insp_before(self, h):
        if not h["ladle"]:
            return None
        before = [e for e in self.inspections[h["ladle"]] if e.timestamp < h["assign"].timestamp
                  and e.source_ref.find("manual_sheet") < 0]
        before.sort(key=lambda e: e.timestamp)
        if not before:
            return None
        last = before[-1]
        # avoid ambiguity: exactly one inspection record at that last timestamp-day
        if sum(1 for e in self.inspections[h["ladle"]] if e.timestamp[:13] == last.timestamp[:13]) > 1:
            return None
        return last


def assign_splits(cfg: GenConfig) -> dict[int, str]:
    if sum(cfg.splits.values()) != cfg.n_campaigns:
        raise ValueError("split sizes must sum to n_campaigns")
    order = list(range(cfg.n_campaigns))
    random.Random(cfg.seed).shuffle(order)
    out, pos = {}, 0
    for split in ("train", "val", "test"):
        for idx in order[pos: pos + cfg.splits[split]]:
            out[idx] = split
        pos += cfg.splits[split]
    return out


def generate(cfg: GenConfig) -> dict:
    splits = assign_splits(cfg)
    builders = [CampaignBuilder(i, cfg, splits[i]).build() for i in range(cfg.n_campaigns)]
    return {"builders": builders, "splits": splits}


def write_dataset(cfg: GenConfig, out_dir: str | Path) -> dict:
    from agentmamba.data.event_linker import link_events

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = generate(cfg)
    events, withheld, questions, anns = [], [], [], []
    for b in res["builders"]:
        events += b.events
        withheld += b.withheld
        for q, a in b.questions:
            questions.append(q)
            anns.append(a)
    # Gold relationship ids = the documented links the evidence records establish.
    by_id = {e.event_id: e for e in events}
    for a in anns:
        rels = link_events([by_id[i] for i in a.required_evidence_ids if i in by_id])
        a.relationship_ids = sorted({r.rel_id for r in rels if r.rel_type != "mentions"})
    _dump(out / "events.jsonl", [e.to_dict() for e in events])
    _dump(out / "withheld.jsonl", [e.to_dict() for e in withheld])
    _dump(out / "questions.jsonl", [q.to_dict() for q in questions])
    _dump(out / "annotations.jsonl", [a.to_dict() for a in anns])
    manifest = {
        "generator_version": GENERATOR_VERSION, "seed": cfg.seed, "config": cfg.__dict__,
        "split_unit": "campaign (independent fictional plant history)",
        "campaign_splits": {f"C{i:02d}": s for i, s in sorted(res["splits"].items())},
        "counts": {"events": len(events), "withheld": len(withheld), "questions": len(questions)},
        "synthetic": True,
        "disclaimer": "All records are fictional and generated; they are not industrial observations.",
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _dump(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
