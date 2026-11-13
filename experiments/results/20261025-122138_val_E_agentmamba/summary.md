# val:E_agentmamba

Spec: `{"policy": "learned", "gate": "logreg_mamba", "budget": 0.1, "mode": "conditional", "verify": true, "note": "full system (== H)"}`

n = 144 test questions (synthetic benchmark)

| metric | value |
|---|---|
| overall accuracy | 0.6875 |
| 95% CI | [0.6111, 0.7639] |
| exact match (value qs) | 0.6296 |
| false-answer rate | 0.0417 |
| unsupported-answer rate | 0.0811 |
| cross-stage accuracy | 0.375 |
| retrieval trigger rate | 0.7917 |
| unnecessary retrieval | 0.0 |
| missed retrieval | 0.0 |
| Recall@K | 0.549 |
| MRR | 0.6078 |
| unjustified causal claims | 0.0 |
| memory retention (bytes) | 0.0996 |
| relevant-evidence retention | 0.4833 |
| mean end-to-end latency (s) | 6.9335 |

## By family

| family | n | accuracy | retrieval rate |
|---|---|---|---|
| F01_direct_context | 24 | 1.0 | 0.0 |
| F02_earlier_stage | 12 | 0.1667 | 1.0 |
| F03_relationship_chain | 12 | 0.0 | 1.0 |
| F04_rare_warning | 6 | 0.8333 | 1.0 |
| F07_maintenance_outcome | 7 | 1.0 | 1.0 |
| F08_failed_intervention | 5 | 1.0 | 1.0 |
| F09_conflicting_records | 12 | 0.0833 | 1.0 |
| F10_outdated_document | 12 | 1.0 | 1.0 |
| F11_missing_identifier | 6 | 1.0 | 1.0 |
| F12_evidence_never_stored | 6 | 1.0 | 1.0 |
| F13_unanswerable | 6 | 1.0 | 1.0 |
| F14_ambiguous | 6 | 1.0 | 0.0 |
| F15_correlation_not_causation | 6 | 1.0 | 1.0 |
| F15b_documented_cause | 6 | 1.0 | 1.0 |
| F16_multi_stage | 6 | 0.0 | 1.0 |
| F17_similar_irrelevant | 6 | 0.6667 | 1.0 |
| F18_precise_time | 6 | 0.5 | 1.0 |
