# Data audit

Audit date: 2026-10-25. Method: web search plus reading the dataset landing pages, then downloading and
parsing the files where access was open. The machine-readable version is
`src/agentmamba/data/source_audit.py` (`python scripts/audit_data.py` prints it and re-checks the local files).

**Status legend.** *Verified + downloaded* means the licence was read on the source page, the files were
downloaded into `data/raw/`, and they parse with the loaders in `document_loader.py`. *Unverified* means the
source was only seen in search results or behind a login, and its terms still need a manual check.

## Summary

No openly licensed public dataset I could find covers more than one steel production stage with shared
identifiers (heat, ladle, cast sequence, coil). This rules out a real-data **cross-stage** benchmark. Two small
real steel datasets are usable for **single-source** experiments. Every cross-stage relationship in this
project therefore comes from the explicitly fictional synthetic generator (Category B).

| Source | Status | Licence | Stages / content | Identifiers | Cross-stage? | Used for |
|---|---|---|---|---|---|---|
| UCI #851 Steel Industry Energy Consumption (DAEWOO Steel, Gwangyang) | verified + downloaded | CC BY 4.0 (UCI page) | Plant electrical energy, 15-min, 2018 | none | no | Category A: real daily records + templated date questions |
| UCI #198 Steel Plates Faults (Semeion) | verified + downloaded | CC BY 4.0 (UCI page) | Plate surface-fault geometry/luminosity features, 7 fault classes | none, no timestamps | no | Ingestion/loader test only |
| Severstal Steel Defect Detection (Kaggle) | unverified (rules need login) | competition rules | Strip-surface images + masks | image ids | no | not used |
| HZDR RODARE 1463: tomography-controlled continuous casting (lab) | unverified (not downloaded) | CC BY 4.0 per search result | Laboratory model caster | experiment ids | no | not used (lab physics, size) |
| MaintNet maintenance logbooks | unverified | dataset licence not found | Aviation/automotive/facility logs | log ids | no | not used (not steel) |

## Details

### UCI #851, Steel Industry Energy Consumption
- Page: https://archive.ics.uci.edu/dataset/851 (DOI 10.24432/C52G8C). Licence: CC BY 4.0. Attribution:
  Sathishkumar V E, Changsun Shin, Yongyun Cho (2021).
- File: `Steel_industry_data.csv`, 35,040 rows, from `01/01/2018 00:15` to `31/12/2018 00:00`.
- Fields: `date` (day-first `dd/mm/YYYY HH:MM`, verified with rows such as `22/02/2018`), `Usage_kWh`, lagging and
  leading reactive power (kVarh), `CO2(tCO2)`, lagging and leading power factor (%), `NSM` (seconds from midnight),
  `WeekStatus`, `Day_of_week`, `Load_Type`.
- Gaps: no timezone is documented, so it is stored as `unknown`. The source doesn't say whether a timestamp
  marks the start or the end of its interval, so we group rows by the calendar date of the recorded timestamp. There are
  no asset, heat or process-stage identifiers and no events or incidents. The UCI page lists the CO2 unit as ppm while the
  column name says tCO2; we keep the column name's unit as recorded and never convert it.
- Use: Category A experiment (`scripts/run_public_experiment.py`). The 365 daily-summary records are real; the
  questions are templated by us, and the answers are read directly from the records.

### UCI #198, Steel Plates Faults
- Page: https://archive.ics.uci.edu/dataset/198 (DOI 10.24432/C5J88N). Licence: CC BY 4.0. Source: Semeion
  Research Center.
- Files: `Faults.NNA` (1,941 rows × 34 columns), `Faults27x7_var` (column names).
- Gaps: no time, no plant or asset identifiers, and the unit of plate thickness isn't documented (we store none).
- Use: loader and ingestion test only. Without time or identifiers, it can't link to any other record.

### Not used
- **Severstal (Kaggle):** the competition rules weren't readable without a login, and redistribution terms are
  unconfirmed. It is image data in any case.
- **HZDR RODARE 1463:** a laboratory physical model, not plant operations. The record pages disagree on size
  (4 GB vs 6 TB), and it was not downloaded.
- **MaintNet:** not steel, and the dataset licence couldn't be found (the CC BY 4.0 notice applies to the paper).

## Consequences for the research claims
- Cross-stage results (RQ2) and memory-gate results (RQ1) rest on **synthetic** data only. They show how the
  method behaves on the declared generator, not how it performs in a real plant.
- The Category A experiment shows that the same pipeline ingests real records with provenance, answers
  date-constrained questions from them, and abstains outside the data's coverage. It is a single-source,
  low-difficulty check.
- Datasets from different plants were **not** joined. Nothing in the sources supports a join on identifiers, units,
  timestamps or process definitions.
