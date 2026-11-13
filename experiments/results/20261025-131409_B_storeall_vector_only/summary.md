# B_storeall_vector_only

Spec: `{"policy": "store_all", "budget": 1.0, "mode": "always", "verify": true, "expand": false, "note": "conventional vector RAG, no relationship expansion"}`

n = 192 test questions (synthetic benchmark)

| metric | value |
|---|---|
| overall accuracy | 0.6354 |
| 95% CI | [0.5677, 0.7031] |
| exact match (value qs) | 0.5486 |
| false-answer rate | 0.0469 |
| unsupported-answer rate | 0.1023 |
| cross-stage accuracy | 0.375 |
| retrieval trigger rate | 0.9583 |
| unnecessary retrieval | 1.0 |
| missed retrieval | 0.0 |
| Recall@K | 0.4494 |
| MRR | 0.3605 |
| unjustified causal claims | 0.0 |
| memory retention (bytes) | 1.0 |
| relevant-evidence retention | 1.0 |
| mean end-to-end latency (s) | 3.4427 |

## By family

| family | n | accuracy | retrieval rate |
|---|---|---|---|
| F01_direct_context | 32 | 1.0 | 1.0 |
| F02_earlier_stage | 16 | 0.4375 | 1.0 |
| F03_relationship_chain | 16 | 0.0625 | 1.0 |
| F04_rare_warning | 8 | 0.0 | 1.0 |
| F07_maintenance_outcome | 7 | 1.0 | 1.0 |
| F08_failed_intervention | 9 | 0.6667 | 1.0 |
| F09_conflicting_records | 16 | 0.3125 | 1.0 |
| F10_outdated_document | 16 | 1.0 | 1.0 |
| F11_missing_identifier | 8 | 1.0 | 1.0 |
| F12_evidence_never_stored | 8 | 1.0 | 1.0 |
| F13_unanswerable | 8 | 1.0 | 1.0 |
| F14_ambiguous | 8 | 1.0 | 0.0 |
| F15_correlation_not_causation | 8 | 1.0 | 1.0 |
| F15b_documented_cause | 8 | 0.25 | 1.0 |
| F16_multi_stage | 8 | 0.0 | 1.0 |
| F17_similar_irrelevant | 8 | 0.0 | 1.0 |
| F18_precise_time | 8 | 0.75 | 1.0 |
