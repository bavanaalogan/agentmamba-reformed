"""Structured record of the public-data audit (see docs/DATA_AUDIT.md for the narrative).

``verified`` means: the licence/fields were read on the source page during this project AND
(where marked ``downloaded``) the files were downloaded and parsed by our loaders. Entries
marked ``unverified`` were only seen in search results and are listed for manual follow-up.
"""
from __future__ import annotations

AUDIT_DATE = "2026-10-25"

SOURCES = [
    {"name": "Steel Industry Energy Consumption", "id": "UCI #851", "url": "https://archive.ics.uci.edu/dataset/851",
     "doi": "10.24432/C52G8C", "licence": "CC BY 4.0 (stated on UCI page)", "access": "public zip download",
     "status": "verified, downloaded, loader implemented (load_uci_851)",
     "fields": "date (dd/mm/YYYY HH:MM, 15-min), Usage_kWh, lagging/leading reactive power (kVarh), CO2(tCO2), "
               "power factors (%), NSM (s), WeekStatus, Day_of_week, Load_Type",
     "coverage": "35,040 rows, 2018-01-01 to 2018-12-31 (observed in file)", "identifiers": "none (single plant, DAEWOO Steel, Gwangyang)",
     "missing": "no heat/ladle/asset/process-stage identifiers, no events, no timezone stated",
     "labels": "Load_Type (3 classes)", "cross_stage": False,
     "use": "Category A: real public records ingested as daily summaries; date-specific retrieval sanity check only"},
    {"name": "Steel Plates Faults", "id": "UCI #198", "url": "https://archive.ics.uci.edu/dataset/198", "doi": "10.24432/C5J88N",
     "licence": "CC BY 4.0 (stated on UCI page)", "access": "public zip download",
     "status": "verified, downloaded, loader implemented (load_uci_198)",
     "fields": "27 geometric/luminosity features + 7 binary fault classes", "coverage": "1,941 plates, no timestamps",
     "identifiers": "none", "missing": "no time, no plant/asset/heat identifiers, units of thickness not documented",
     "labels": "7 fault classes", "cross_stage": False,
     "use": "Category A: standalone quality observations; cannot be linked to any other stage"},
    {"name": "Severstal: Steel Defect Detection (Kaggle)", "id": "kaggle-severstal",
     "url": "https://www.kaggle.com/competitions/severstal-steel-defect-detection", "licence": "competition rules; not verified",
     "access": "Kaggle login + rule acceptance", "status": "unverified (rules page not readable without login)",
     "fields": "flat-steel surface images + RLE defect masks", "cross_stage": False,
     "use": "not used (images; redistribution terms unverified)"},
    {"name": "Laboratory tomography-controlled continuous steel casting (HZDR RODARE)", "id": "rodare.1463",
     "url": "https://rodare.hzdr.de/record/1463", "doi": "10.14278/rodare.1463", "licence": "CC BY 4.0 (per record page in search results)",
     "access": "large download (record pages disagree: 4 GB zip vs 6 TB stats)", "status": "unverified (not downloaded)",
     "fields": "laboratory model-caster measurement data", "cross_stage": False,
     "use": "not used: laboratory physical model, not plant operations; size"},
    {"name": "MaintNet maintenance logbooks", "id": "maintnet", "url": "https://arxiv.org/abs/2005.12443",
     "licence": "paper CC BY 4.0; dataset licence not found", "status": "unverified",
     "fields": "aviation/automotive/facility maintenance logbook text", "cross_stage": False,
     "use": "not used: not steel; dataset licence unconfirmed"},
]
