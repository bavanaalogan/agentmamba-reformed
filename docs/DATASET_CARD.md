# Dataset card

## 1. Synthetic plant-wide benchmark (Category B): `data/synthetic/main`

**Nature.** These are fully fictional, generated event histories of an integrated steel plant. **No record,
value, incident or relationship is a real industrial observation.** Every event has `source_type="synthetic"` and a
`synthetic://` source reference, and the dashboard shows a warning whenever synthetic records are in memory.

**Generator.** `src/agentmamba/data/synthetic_generator.py`, version `synth-v1`, seed 13. Run
`python scripts/generate_data.py`, then `python scripts/validate_data.py`.

**Structure.** There are 30 *campaigns*. Each one is an independent fictional plant history of about 6 days with its own
ID namespace (`C07-H0012`, `C07-LD03`, ...). A campaign contains:
- **Ironmaking:** hot-metal taps (batch, temperature).
- **Steelmaking:** BOF heat records (tap temperature, end-blow carbon and oxygen).
- **Ladle:** heat-to-ladle assignments, ladle-furnace treatments, periodic refractory inspections (finding and
  lining thickness), one ladle incident (steel penetration) preceded by a low-severity warning, and occasional
  look-alike warnings on other ladles.
- **Casting:** cast records (heat, sequence, slab, speed, superheat, optional abnormality note).
- **Rolling:** slab → coil records (gauge, finishing temperature).
- **Quality:** coil inspections (pass or defect), root-cause-analysis reports (*inconclusive* or *confirmed*).
- **Maintenance:** 7 work orders, each with a follow-up whose outcome is effective, partially effective or not effective,
  and revised and unrevised procedure documents (revision 2 supersedes revision 1).
- **Noise:** shift logs, hourly sensor summaries, ladle preheat logs, and nuisance caster alarms recorded with
  *major* severity.

All values (temperatures, compositions, thicknesses) are random draws from plausible-looking ranges and carry
their unit as recorded. They are not taken from any plant.

**Size (seed 13).** 17,165 delivered events, 30 withheld events, and 719 questions.

| split | campaigns | events | questions |
|---|---|---|---|
| train | 16 | 9,153 | 383 |
| val | 6 | 3,432 | 144 |
| test | 8 | 4,580 | 192 |

**Question families** (per campaign, ~24 questions; mapping to the 18 scenario families requested):

| family | what it tests | expected behaviour |
|---|---|---|
| F01 direct context | answer is in the initial context (measurement, ladle assignment, WO follow-up) | answer without retrieval |
| F02 earlier stage | casting record on screen, ladle assignment needed | retrieve (LADLE) |
| F03 relationship chain | coil → slab → cast sequence | retrieve + 2-hop documented links |
| F04 rare warning | low-severity refractory warning days before an incident; look-alike warnings on other ladles | retrieve, reject other ladles |
| F07 / F08 maintenance outcome | follow-up outcome, incl. failed / partially effective interventions | retrieve |
| F09 conflicting records | two inspection sheets disagree on the same date (+ a non-conflicting control) | report the conflict |
| F10 outdated document | revision 1 on screen, revision 2 supersedes it (+ an unrevised control) | retrieve, answer the current revision |
| F11 missing identifier | assignment record lacks the ladle number | abstain |
| F12 evidence never stored | follow-up exists in the world but was never delivered to ingestion | abstain |
| F13 unanswerable | measurement that was never recorded | abstain |
| F14 ambiguous | no identifier ("the ladle") | ask for clarification |
| F15 correlation ≠ causation | defect + upstream casting abnormality + *inconclusive* RCA | hypothesis only, no causal claim |
| F15b documented cause | *confirmed* RCA record exists | documented cause, cited |
| F16 multi-stage | coil → slab → heat → ladle → last inspection before the heat (4 stages) | retrieve + temporal selection |
| F17 similar but irrelevant | same abnormality text appears on other heats | answer only for the named heat |
| F18 precise time | which work order on asset X on date D | date-constrained answer |

Families 5 (repeated event across assets) and 6 (similar symptoms, different explanations) appear as distractors in
F04 and F17 (identical warnings and abnormalities on other assets and heats), not as separate question families. Family 12
(evidence discarded by the gate) is not generated; it *emerges* whenever a memory budget drops required
evidence, and is measured as "evidence lost".

**Annotations** are stored separately (`annotations.jsonl`). Each holds the answer kind, expected answer, required
evidence IDs, gold relationship IDs, expected sources, `initial_sufficient`, `retrieval_required`, `cross_stage` and
`temporal_sensitive`. Declared policy: questions asking for the *current* revision need retrieval even when the
revision on screen happens to be current.

**Splits and leakage controls.** Splits are assigned per campaign (seeded shuffle). No entity ID or event crosses splits
(checked by `validate_dataset`). Test annotations can only be loaded with `purpose="evaluation"`. The live
pipeline modules never import the annotation loader (unit test). The split manifest records the seed and the
generator configuration.

**Known artifacts / limitations of the generator.**
- The question grammar is templated (1–2 phrasings per question type), so intent classification is close to trivial.
- Relevance is structural by construction. For example, warnings are always worded with one of three phrases and
  incidents always follow one warning. A learned gate can pick up these regularities, and that does not
  transfer to real plants.
- Each ladle-assignment record is individually unremarkable. Whether it is "useful" depends only on whether a
  question happens to ask about that heat, which no write-time feature can know. This is a deliberate and realistic
  source of label noise.

## 2. Public Category A benchmark: `data/public/uci851`
- **Records:** 365 real daily summaries of UCI #851 (CC BY 4.0). Each record keeps the raw row range it aggregates.
- **Questions:** 369 *templated* date-lookup questions (peak usage, total usage, dominant load type), plus 4
  out-of-coverage dates that should be abstained on. The answers are read from the data.
- **Split by month:** train Jan–Aug, val Sep–Oct, test Nov–Dec 2018.
- **Limitation:** single source, no cross-stage identifiers. This tests ingestion, date-constrained retrieval,
  provenance and abstention, not plant-wide reasoning.

## 3. Synthetic industrial-log benchmark: `data/log_benchmark/v1`
- **What:** fictional records in common operational-log *formats*:
  - historian exports in long format (`timestamp, equipment, signal, value, unit`);
  - alarm journals (JSONL: `ts, equipment, alarm, state, priority`);
  - CMMS maintenance exports and manual round sheets;
  - Markdown inspection reports, shift notes, RCA and return-to-service records, and an equipment-care manual.

  All values, events, causes and people are invented. It is **not plant data** and makes no claim about real
  equipment behaviour.
- **Generator:** `src/agentmamba/data/log_benchmark.py`, seed 2026. It is deterministic: regenerating reproduces
  every file byte for byte.
- **Structure:**
  - 8 scenarios (plant areas P1–P8, 8 machines each), split by scenario: dev P4/P7/P8, test P1/P2/P3/P5/P6.
  - 4 cumulative arrival stages per scenario, about 20 / 340 / 450 / 1,920 records.
  - 15 questions per scenario (45 dev, 75 test).
- **Files:** `<split>/<scenario>/stage<k>/*` holds the records; `manifest.json` gives arrival order and stages;
  `questions.jsonl` holds what systems see; `annotations/annotations.jsonl` holds gold answers, answer kinds,
  match rule and required/related evidence IDs, and is read by the scorer only.
- **Known regularities:** templated wording; one maintenance entry per machine-day for the target machine; exact
  same-alarm text on sibling equipment; a confirmed cause for a *later* trip of the machine whose earlier trip is
  undetermined. These are deliberate distractors, but they are generated, not observed.

## Intended use
Research on memory-retention policies and evidence-grounded, conditional retrieval. **Not** for training or
validating real plant-control, safety or quality-release decisions.
