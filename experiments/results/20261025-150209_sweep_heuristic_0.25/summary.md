# sweep_heuristic_0.25

Spec: `{"policy": "heuristic", "budget": 0.25, "mode": "conditional", "verify": true}`

n = 192 test questions (synthetic benchmark)

| metric | value |
|---|---|
| overall accuracy | 0.6562 |
| 95% CI | [0.5938, 0.724] |
| exact match (value qs) | 0.5625 |
| false-answer rate | 0.0833 |
| unsupported-answer rate | 0.0206 |
| cross-stage accuracy | 0.375 |
| retrieval trigger rate | 0.7917 |
| unnecessary retrieval | 0.0 |
| missed retrieval | 0.0 |
| Recall@K | 0.4755 |
| MRR | 0.4851 |
| unjustified causal claims | 0.0 |
| memory retention (bytes) | 0.25 |
| relevant-evidence retention | 0.3942 |
| mean end-to-end latency (s) | 1.6666 |

## By family

| family | n | accuracy | retrieval rate |
|---|---|---|---|
| F01_direct_context | 32 | 1.0 | 0.0 |
| F02_earlier_stage | 16 | 0.0 | 1.0 |
| F03_relationship_chain | 16 | 0.0 | 1.0 |
| F04_rare_warning | 8 | 0.5 | 1.0 |
| F07_maintenance_outcome | 7 | 0.5714 | 1.0 |
| F08_failed_intervention | 9 | 1.0 | 1.0 |
| F09_conflicting_records | 16 | 0.6875 | 1.0 |
| F10_outdated_document | 16 | 0.125 | 1.0 |
| F11_missing_identifier | 8 | 1.0 | 1.0 |
| F12_evidence_never_stored | 8 | 1.0 | 1.0 |
| F13_unanswerable | 8 | 1.0 | 1.0 |
| F14_ambiguous | 8 | 1.0 | 0.0 |
| F15_correlation_not_causation | 8 | 1.0 | 1.0 |
| F15b_documented_cause | 8 | 1.0 | 1.0 |
| F16_multi_stage | 8 | 0.0 | 1.0 |
| F17_similar_irrelevant | 8 | 1.0 | 1.0 |
| F18_precise_time | 8 | 1.0 | 1.0 |
