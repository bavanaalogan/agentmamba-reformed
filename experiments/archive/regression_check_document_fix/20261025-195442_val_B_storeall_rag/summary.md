# val:B_storeall_rag

Spec: `{"policy": "store_all", "budget": 1.0, "mode": "always", "verify": true, "note": "no selective writing"}`

n = 144 test questions (synthetic benchmark)

| metric | value |
|---|---|
| overall accuracy | 0.9722 |
| 95% CI | [0.9444, 0.9931] |
| exact match (value qs) | 0.963 |
| false-answer rate | 0.0278 |
| unsupported-answer rate | 0.037 |
| cross-stage accuracy | 1.0 |
| retrieval trigger rate | 0.9583 |
| unnecessary retrieval | 1.0 |
| missed retrieval | 0.0 |
| Recall@K | 0.9385 |
| MRR | 0.9489 |
| unjustified causal claims | 0.0 |
| memory retention (bytes) | 1.0 |
| relevant-evidence retention | 1.0 |
| mean end-to-end latency (s) | 0.0448 |

## By family

| family | n | accuracy | retrieval rate |
|---|---|---|---|
| F01_direct_context | 24 | 1.0 | 1.0 |
| F02_earlier_stage | 12 | 1.0 | 1.0 |
| F03_relationship_chain | 12 | 1.0 | 1.0 |
| F04_rare_warning | 6 | 0.3333 | 1.0 |
| F07_maintenance_outcome | 7 | 1.0 | 1.0 |
| F08_failed_intervention | 5 | 1.0 | 1.0 |
| F09_conflicting_records | 12 | 1.0 | 1.0 |
| F10_outdated_document | 12 | 1.0 | 1.0 |
| F11_missing_identifier | 6 | 1.0 | 1.0 |
| F12_evidence_never_stored | 6 | 1.0 | 1.0 |
| F13_unanswerable | 6 | 1.0 | 1.0 |
| F14_ambiguous | 6 | 1.0 | 0.0 |
| F15_correlation_not_causation | 6 | 1.0 | 1.0 |
| F15b_documented_cause | 6 | 1.0 | 1.0 |
| F16_multi_stage | 6 | 1.0 | 1.0 |
| F17_similar_irrelevant | 6 | 1.0 | 1.0 |
| F18_precise_time | 6 | 1.0 | 1.0 |
