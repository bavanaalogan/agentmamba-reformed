# B_storeall_conditional

Spec: `{"policy": "store_all", "budget": 1.0, "mode": "conditional", "verify": true}`

n = 192 test questions (synthetic benchmark)

| metric | value |
|---|---|
| overall accuracy | 0.9896 |
| 95% CI | [0.974, 1.0] |
| exact match (value qs) | 0.9861 |
| false-answer rate | 0.0104 |
| unsupported-answer rate | 0.0138 |
| cross-stage accuracy | 1.0 |
| retrieval trigger rate | 0.7917 |
| unnecessary retrieval | 0.0 |
| missed retrieval | 0.0 |
| Recall@K | 0.9602 |
| MRR | 0.94 |
| unjustified causal claims | 0.0 |
| memory retention (bytes) | 1.0 |
| relevant-evidence retention | 1.0 |
| mean end-to-end latency (s) | 0.0437 |

## By family

| family | n | accuracy | retrieval rate |
|---|---|---|---|
| F01_direct_context | 32 | 1.0 | 0.0 |
| F02_earlier_stage | 16 | 1.0 | 1.0 |
| F03_relationship_chain | 16 | 1.0 | 1.0 |
| F04_rare_warning | 8 | 0.75 | 1.0 |
| F07_maintenance_outcome | 7 | 1.0 | 1.0 |
| F08_failed_intervention | 9 | 1.0 | 1.0 |
| F09_conflicting_records | 16 | 1.0 | 1.0 |
| F10_outdated_document | 16 | 1.0 | 1.0 |
| F11_missing_identifier | 8 | 1.0 | 1.0 |
| F12_evidence_never_stored | 8 | 1.0 | 1.0 |
| F13_unanswerable | 8 | 1.0 | 1.0 |
| F14_ambiguous | 8 | 1.0 | 0.0 |
| F15_correlation_not_causation | 8 | 1.0 | 1.0 |
| F15b_documented_cause | 8 | 1.0 | 1.0 |
| F16_multi_stage | 8 | 1.0 | 1.0 |
| F17_similar_irrelevant | 8 | 1.0 | 1.0 |
| F18_precise_time | 8 | 1.0 | 1.0 |
