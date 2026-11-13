# F_random_rag

Spec: `{"policy": "random", "budget": 0.1, "mode": "always", "verify": true}`

n = 192 test questions (synthetic benchmark)

| metric | value |
|---|---|
| overall accuracy | 0.4375 |
| 95% CI | [0.3698, 0.5104] |
| exact match (value qs) | 0.2986 |
| false-answer rate | 0.099 |
| unsupported-answer rate | 0.0968 |
| cross-stage accuracy | 0.2656 |
| retrieval trigger rate | 0.9583 |
| unnecessary retrieval | 1.0 |
| missed retrieval | 0.0 |
| Recall@K | 0.1052 |
| MRR | 0.1235 |
| unjustified causal claims | 0.0 |
| memory retention (bytes) | 0.1 |
| relevant-evidence retention | 0.1079 |
| mean end-to-end latency (s) | 6.398 |

## By family

| family | n | accuracy | retrieval rate |
|---|---|---|---|
| F01_direct_context | 32 | 1.0 | 1.0 |
| F02_earlier_stage | 16 | 0.0625 | 1.0 |
| F03_relationship_chain | 16 | 0.0 | 1.0 |
| F04_rare_warning | 8 | 0.125 | 1.0 |
| F07_maintenance_outcome | 7 | 0.1429 | 1.0 |
| F08_failed_intervention | 9 | 0.0 | 1.0 |
| F09_conflicting_records | 16 | 0.125 | 1.0 |
| F10_outdated_document | 16 | 0.1875 | 1.0 |
| F11_missing_identifier | 8 | 1.0 | 1.0 |
| F12_evidence_never_stored | 8 | 1.0 | 1.0 |
| F13_unanswerable | 8 | 1.0 | 1.0 |
| F14_ambiguous | 8 | 1.0 | 0.0 |
| F15_correlation_not_causation | 8 | 1.0 | 1.0 |
| F15b_documented_cause | 8 | 0.125 | 1.0 |
| F16_multi_stage | 8 | 0.0 | 1.0 |
| F17_similar_irrelevant | 8 | 0.0 | 1.0 |
| F18_precise_time | 8 | 0.375 | 1.0 |
