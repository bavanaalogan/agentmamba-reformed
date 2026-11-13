# sweep_random_0.25

Spec: `{"policy": "random", "budget": 0.25, "mode": "conditional", "verify": true}`

n = 192 test questions (synthetic benchmark)

| metric | value |
|---|---|
| overall accuracy | 0.5208 |
| 95% CI | [0.4531, 0.5938] |
| exact match (value qs) | 0.4028 |
| false-answer rate | 0.0833 |
| unsupported-answer rate | 0.0811 |
| cross-stage accuracy | 0.3125 |
| retrieval trigger rate | 0.7917 |
| unnecessary retrieval | 0.0 |
| missed retrieval | 0.0 |
| Recall@K | 0.2341 |
| MRR | 0.2647 |
| unjustified causal claims | 0.0 |
| memory retention (bytes) | 0.25 |
| relevant-evidence retention | 0.2614 |
| mean end-to-end latency (s) | 6.898 |

## By family

| family | n | accuracy | retrieval rate |
|---|---|---|---|
| F01_direct_context | 32 | 1.0 | 0.0 |
| F02_earlier_stage | 16 | 0.0625 | 1.0 |
| F03_relationship_chain | 16 | 0.0 | 1.0 |
| F04_rare_warning | 8 | 0.375 | 1.0 |
| F07_maintenance_outcome | 7 | 0.1429 | 1.0 |
| F08_failed_intervention | 9 | 0.2222 | 1.0 |
| F09_conflicting_records | 16 | 0.3125 | 1.0 |
| F10_outdated_document | 16 | 0.375 | 1.0 |
| F11_missing_identifier | 8 | 1.0 | 1.0 |
| F12_evidence_never_stored | 8 | 1.0 | 1.0 |
| F13_unanswerable | 8 | 1.0 | 1.0 |
| F14_ambiguous | 8 | 1.0 | 0.0 |
| F15_correlation_not_causation | 8 | 1.0 | 1.0 |
| F15b_documented_cause | 8 | 0.25 | 1.0 |
| F16_multi_stage | 8 | 0.0 | 1.0 |
| F17_similar_irrelevant | 8 | 0.375 | 1.0 |
| F18_precise_time | 8 | 0.625 | 1.0 |
