# A_no_retrieval

Spec: `{"policy": "learned", "gate": "logreg_mamba", "budget": 0.1, "mode": "never", "verify": true, "note": "initial Mamba track only (memory unused)"}`

n = 192 test questions (synthetic benchmark)

| metric | value |
|---|---|
| overall accuracy | 0.3854 |
| 95% CI | [0.3177, 0.4583] |
| exact match (value qs) | 0.2361 |
| false-answer rate | 0.0729 |
| unsupported-answer rate | 0.0 |
| cross-stage accuracy | 0.25 |
| retrieval trigger rate | 0.0 |
| unnecessary retrieval | 0.0 |
| missed retrieval | 1.0 |
| Recall@K | None |
| MRR | None |
| unjustified causal claims | 0.0 |
| memory retention (bytes) | 0.0999 |
| relevant-evidence retention | 0.4647 |
| mean end-to-end latency (s) | 0.0016 |

## By family

| family | n | accuracy | retrieval rate |
|---|---|---|---|
| F01_direct_context | 32 | 1.0 | 0.0 |
| F02_earlier_stage | 16 | 0.0 | 0.0 |
| F03_relationship_chain | 16 | 0.0 | 0.0 |
| F04_rare_warning | 8 | 0.0 | 0.0 |
| F07_maintenance_outcome | 7 | 0.0 | 0.0 |
| F08_failed_intervention | 9 | 0.0 | 0.0 |
| F09_conflicting_records | 16 | 0.0 | 0.0 |
| F10_outdated_document | 16 | 0.125 | 0.0 |
| F11_missing_identifier | 8 | 1.0 | 0.0 |
| F12_evidence_never_stored | 8 | 1.0 | 0.0 |
| F13_unanswerable | 8 | 1.0 | 0.0 |
| F14_ambiguous | 8 | 1.0 | 0.0 |
| F15_correlation_not_causation | 8 | 1.0 | 0.0 |
| F15b_documented_cause | 8 | 0.0 | 0.0 |
| F16_multi_stage | 8 | 0.0 | 0.0 |
| F17_similar_irrelevant | 8 | 0.0 | 0.0 |
| F18_precise_time | 8 | 0.0 | 0.0 |
