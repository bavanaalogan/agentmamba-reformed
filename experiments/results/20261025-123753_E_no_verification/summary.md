# E_no_verification

Spec: `{"policy": "learned", "gate": "logreg_mamba", "budget": 0.1, "mode": "conditional", "verify": false}`

n = 192 test questions (synthetic benchmark)

| metric | value |
|---|---|
| overall accuracy | 0.3958 |
| 95% CI | [0.3281, 0.4635] |
| exact match (value qs) | 0.3611 |
| false-answer rate | 0.526 |
| unsupported-answer rate | 0.7019 |
| cross-stage accuracy | 0.1562 |
| retrieval trigger rate | 0.9583 |
| unnecessary retrieval | 1.0 |
| missed retrieval | 0.0 |
| Recall@K | 0.4891 |
| MRR | 0.5402 |
| unjustified causal claims | 1.0 |
| memory retention (bytes) | 0.0999 |
| relevant-evidence retention | 0.4647 |
| mean end-to-end latency (s) | 0.031 |

## By family

| family | n | accuracy | retrieval rate |
|---|---|---|---|
| F01_direct_context | 32 | 0.8438 | 1.0 |
| F02_earlier_stage | 16 | 0.0625 | 1.0 |
| F03_relationship_chain | 16 | 0.0 | 1.0 |
| F04_rare_warning | 8 | 0.25 | 1.0 |
| F07_maintenance_outcome | 7 | 0.0 | 1.0 |
| F08_failed_intervention | 9 | 0.7778 | 1.0 |
| F09_conflicting_records | 16 | 0.0 | 1.0 |
| F10_outdated_document | 16 | 0.5 | 1.0 |
| F11_missing_identifier | 8 | 1.0 | 1.0 |
| F12_evidence_never_stored | 8 | 0.0 | 1.0 |
| F13_unanswerable | 8 | 1.0 | 1.0 |
| F14_ambiguous | 8 | 1.0 | 0.0 |
| F15_correlation_not_causation | 8 | 0.0 | 1.0 |
| F15b_documented_cause | 8 | 0.5 | 1.0 |
| F16_multi_stage | 8 | 0.0 | 1.0 |
| F17_similar_irrelevant | 8 | 0.125 | 1.0 |
| F18_precise_time | 8 | 0.25 | 1.0 |
