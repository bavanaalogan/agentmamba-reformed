# sweep_random_0.50

Spec: `{"policy": "random", "budget": 0.5, "mode": "conditional", "verify": true}`

n = 192 test questions (synthetic benchmark)

| metric | value |
|---|---|
| overall accuracy | 0.6615 |
| 95% CI | [0.599, 0.7292] |
| exact match (value qs) | 0.5694 |
| false-answer rate | 0.0729 |
| unsupported-answer rate | 0.0729 |
| cross-stage accuracy | 0.4844 |
| retrieval trigger rate | 0.7917 |
| unnecessary retrieval | 0.0 |
| missed retrieval | 0.0 |
| Recall@K | 0.4675 |
| MRR | 0.5325 |
| unjustified causal claims | 0.0 |
| memory retention (bytes) | 0.5 |
| relevant-evidence retention | 0.5477 |
| mean end-to-end latency (s) | 6.898 |

## By family

| family | n | accuracy | retrieval rate |
|---|---|---|---|
| F01_direct_context | 32 | 1.0 | 0.0 |
| F02_earlier_stage | 16 | 0.25 | 1.0 |
| F03_relationship_chain | 16 | 0.3125 | 1.0 |
| F04_rare_warning | 8 | 0.375 | 1.0 |
| F07_maintenance_outcome | 7 | 0.5714 | 1.0 |
| F08_failed_intervention | 9 | 0.5556 | 1.0 |
| F09_conflicting_records | 16 | 0.6875 | 1.0 |
| F10_outdated_document | 16 | 0.5625 | 1.0 |
| F11_missing_identifier | 8 | 1.0 | 1.0 |
| F12_evidence_never_stored | 8 | 1.0 | 1.0 |
| F13_unanswerable | 8 | 1.0 | 1.0 |
| F14_ambiguous | 8 | 1.0 | 0.0 |
| F15_correlation_not_causation | 8 | 1.0 | 1.0 |
| F15b_documented_cause | 8 | 0.375 | 1.0 |
| F16_multi_stage | 8 | 0.25 | 1.0 |
| F17_similar_irrelevant | 8 | 0.5 | 1.0 |
| F18_precise_time | 8 | 0.625 | 1.0 |
