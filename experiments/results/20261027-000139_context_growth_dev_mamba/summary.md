# Context-growth experiment (dev split, answerer=mamba)

45 questions x stages; scenarios P4, P7, P8. Synthetic log benchmark.

## Overall (all stages pooled)

| condition | n | accuracy [95% CI] | grounded | answerable acc. | correct abstention | conflict det. | incorrect answers | unsupported claims / answered | evidence in context | Recall@k | median latency cold (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_mamba_768 | 180 | 0.122 [0.0778, 0.1722] | 0.122 | 0.108 | 0.188 | 0.000 | 0.478 | 0.869 | 0.114 | n/a | 10.18 |
| baseline_mamba_2048 | 180 | 0.239 [0.1778, 0.3056] | 0.178 | 0.283 | 0.188 | 0.000 | 0.511 | 0.817 | 0.250 | n/a | 21.72 |
| retrieval_always | 180 | 0.867 [0.8167, 0.9111] | 0.867 | 0.800 | 1.000 | 1.000 | 0.000 | 0.000 | 0.114 | 1.000 | 11.03 |
| agentmamba | 180 | 0.867 [0.8167, 0.9111] | 0.867 | 0.800 | 1.000 | 1.000 | 0.000 | 0.000 | 0.114 | 1.000 | n/a |
| agentmamba_no_verification | 180 | 0.406 [0.3333, 0.4778] | 0.389 | 0.608 | 0.000 | 0.000 | 0.578 | 0.605 | 0.114 | 1.000 | 12.93 |
| agentmamba_no_retrieval | 180 | 0.350 [0.2778, 0.4167] | 0.350 | 0.125 | 1.000 | 0.000 | 0.000 | 0.000 | 0.114 | n/a | n/a |
| agentmamba_vector_only | 45 | 0.800 [0.6667, 0.9111] | 0.800 | 0.700 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.818 | 9.51 |
| agentmamba_lexical_only | 45 | 0.867 [0.7556, 0.9556] | 0.867 | 0.800 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 9.89 |

## Accuracy by stage

| condition | stage 1 | stage 2 | stage 3 | stage 4 |
|---|---|---|---|---|
| baseline_mamba_768 | 0.289 | 0.133 | 0.000 | 0.067 |
| baseline_mamba_2048 | 0.511 | 0.267 | 0.089 | 0.089 |
| retrieval_always | 0.867 | 0.867 | 0.867 | 0.867 |
| agentmamba | 0.867 | 0.867 | 0.867 | 0.867 |
| agentmamba_no_verification | 0.511 | 0.511 | 0.267 | 0.333 |
| agentmamba_no_retrieval | 0.600 | 0.267 | 0.267 | 0.267 |
| agentmamba_vector_only | - | - | - | 0.800 |
| agentmamba_lexical_only | - | - | - | 0.867 |

## Evidence inside the 768/2048-token context window, by stage

| condition | stage 1 | stage 2 | stage 3 | stage 4 |
|---|---|---|---|---|
| baseline_mamba_768 | 0.455 | 0.000 | 0.000 | 0.000 |
| baseline_mamba_2048 | 1.000 | 0.000 | 0.000 | 0.000 |
| retrieval_always | 0.455 | 0.000 | 0.000 | 0.000 |
| agentmamba | 0.455 | 0.000 | 0.000 | 0.000 |
| agentmamba_no_verification | 0.455 | 0.000 | 0.000 | 0.000 |
| agentmamba_no_retrieval | 0.455 | 0.000 | 0.000 | 0.000 |
| agentmamba_vector_only | - | - | - | 0.000 |
| agentmamba_lexical_only | - | - | - | 0.000 |

## Paired comparisons (B vs A, same question and stage)

| B vs A | outcome | A | B | B-A [95% CI] | McNemar p |
|---|---|---|---|---|---|
| agentmamba vs baseline_mamba_768 | correct | 0.1222 | 0.8667 | 0.7444 [0.6778, 0.8056] | 0.0 |
| agentmamba vs baseline_mamba_768 | grounded_correct | 0.1222 | 0.8667 | 0.7444 [0.6778, 0.8056] | 0.0 |
| agentmamba vs baseline_mamba_768 | unsupported_claim (lower better) | 0.4778 | 0.0 | -0.4778 [-0.5444, -0.4056] | 0.0 |
| agentmamba vs baseline_mamba_768 | incorrect_answer (lower better) | 0.4778 | 0.0 | -0.4778 [-0.5444, -0.4056] | 0.0 |
| agentmamba vs baseline_mamba_2048 | correct | 0.2389 | 0.8667 | 0.6278 [0.5389, 0.7056] | 0.0 |
| agentmamba vs baseline_mamba_2048 | grounded_correct | 0.1778 | 0.8667 | 0.6889 [0.6111, 0.7611] | 0.0 |
| agentmamba vs baseline_mamba_2048 | unsupported_claim (lower better) | 0.5722 | 0.0 | -0.5722 [-0.6444, -0.5] | 0.0 |
| agentmamba vs baseline_mamba_2048 | incorrect_answer (lower better) | 0.5111 | 0.0 | -0.5111 [-0.5833, -0.4389] | 0.0 |
| agentmamba vs agentmamba_no_retrieval | correct | 0.35 | 0.8667 | 0.5167 [0.4444, 0.5889] | 0.0 |
| agentmamba vs agentmamba_no_retrieval | grounded_correct | 0.35 | 0.8667 | 0.5167 [0.4444, 0.5889] | 0.0 |
| agentmamba vs agentmamba_no_retrieval | unsupported_claim (lower better) | 0.0 | 0.0 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs agentmamba_no_retrieval | incorrect_answer (lower better) | 0.0 | 0.0 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs agentmamba_no_verification | correct | 0.4056 | 0.8667 | 0.4611 [0.3667, 0.5444] | 0.0 |
| agentmamba vs agentmamba_no_verification | grounded_correct | 0.3889 | 0.8667 | 0.4778 [0.3833, 0.5611] | 0.0 |
| agentmamba vs agentmamba_no_verification | unsupported_claim (lower better) | 0.5944 | 0.0 | -0.5944 [-0.6611, -0.5167] | 0.0 |
| agentmamba vs agentmamba_no_verification | incorrect_answer (lower better) | 0.5778 | 0.0 | -0.5778 [-0.6444, -0.5] | 0.0 |
| agentmamba vs retrieval_always | correct | 0.8667 | 0.8667 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs retrieval_always | grounded_correct | 0.8667 | 0.8667 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs retrieval_always | unsupported_claim (lower better) | 0.0 | 0.0 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs retrieval_always | incorrect_answer (lower better) | 0.0 | 0.0 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba_no_verification vs baseline_mamba_768 | correct | 0.1222 | 0.4056 | 0.2833 [0.2056, 0.3611] | 0.0 |
| agentmamba_no_verification vs baseline_mamba_768 | grounded_correct | 0.1222 | 0.3889 | 0.2667 [0.1889, 0.3444] | 0.0 |
| agentmamba_no_verification vs baseline_mamba_768 | unsupported_claim (lower better) | 0.4778 | 0.5944 | 0.1167 [0.05, 0.1833] | 0.001914 |
| agentmamba_no_verification vs baseline_mamba_768 | incorrect_answer (lower better) | 0.4778 | 0.5778 | 0.1 [0.0333, 0.1667] | 0.006427 |
| agentmamba_no_retrieval vs baseline_mamba_768 | correct | 0.1222 | 0.35 | 0.2278 [0.1667, 0.2889] | 0.0 |
| agentmamba_no_retrieval vs baseline_mamba_768 | grounded_correct | 0.1222 | 0.35 | 0.2278 [0.1667, 0.2889] | 0.0 |
| agentmamba_no_retrieval vs baseline_mamba_768 | unsupported_claim (lower better) | 0.4778 | 0.0 | -0.4778 [-0.5444, -0.4056] | 0.0 |
| agentmamba_no_retrieval vs baseline_mamba_768 | incorrect_answer (lower better) | 0.4778 | 0.0 | -0.4778 [-0.5444, -0.4056] | 0.0 |
| baseline_mamba_2048 vs baseline_mamba_768 | correct | 0.1222 | 0.2389 | 0.1167 [0.0722, 0.1667] | 1e-06 |
| baseline_mamba_2048 vs baseline_mamba_768 | grounded_correct | 0.1222 | 0.1778 | 0.0556 [0.0222, 0.0889] | 0.001953 |
| baseline_mamba_2048 vs baseline_mamba_768 | unsupported_claim (lower better) | 0.4778 | 0.5722 | 0.0944 [0.0444, 0.1444] | 0.000488 |
| baseline_mamba_2048 vs baseline_mamba_768 | incorrect_answer (lower better) | 0.4778 | 0.5111 | 0.0333 [-0.0056, 0.0778] | 0.179565 |
| agentmamba vs agentmamba_vector_only | correct | 0.8 | 0.8667 | 0.0667 [0.0, 0.1556] | 0.25 |
| agentmamba vs agentmamba_vector_only | grounded_correct | 0.8 | 0.8667 | 0.0667 [0.0, 0.1556] | 0.25 |
| agentmamba vs agentmamba_vector_only | unsupported_claim (lower better) | 0.0 | 0.0 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs agentmamba_vector_only | incorrect_answer (lower better) | 0.0 | 0.0 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs agentmamba_lexical_only | correct | 0.8667 | 0.8667 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs agentmamba_lexical_only | grounded_correct | 0.8667 | 0.8667 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs agentmamba_lexical_only | unsupported_claim (lower better) | 0.0 | 0.0 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs agentmamba_lexical_only | incorrect_answer (lower better) | 0.0 | 0.0 | 0.0 [0.0, 0.0] | 1.0 |

## Calibration (answered questions)

| condition | confidence | n | Brier | ECE | mean conf. | acc. |
|---|---|---|---|---|---|---|
| baseline_mamba_768 | raw_candidate_share | 99 | 0.504 | 0.566 | 0.697 | 0.131 |
| baseline_mamba_768 | calibrated | 99 | 0.114 | 0.000 | 0.131 | 0.131 |
| baseline_mamba_2048 | raw_candidate_share | 126 | 0.370 | 0.344 | 0.594 | 0.270 |
| baseline_mamba_2048 | calibrated | 126 | 0.197 | 0.000 | 0.270 | 0.270 |
| retrieval_always | raw_candidate_share | 96 | 0.000 | 0.000 | 1.000 | 1.000 |
| retrieval_always | calibrated | 0 | n/a | n/a | n/a | n/a |
| agentmamba | raw_candidate_share | 96 | 0.000 | 0.000 | 1.000 | 1.000 |
| agentmamba | calibrated | 0 | n/a | n/a | n/a | n/a |
| agentmamba_no_verification | raw_candidate_share | 177 | 0.257 | 0.257 | 0.670 | 0.412 |
| agentmamba_no_verification | calibrated | 177 | 0.192 | 0.096 | 0.412 | 0.412 |
| agentmamba_no_retrieval | raw_candidate_share | 15 | 0.000 | 0.000 | 1.000 | 1.000 |
| agentmamba_no_retrieval | calibrated | 0 | n/a | n/a | n/a | n/a |
| agentmamba_vector_only | raw_candidate_share | 21 | 0.000 | 0.000 | 1.000 | 1.000 |
| agentmamba_vector_only | calibrated | 0 | n/a | n/a | n/a | n/a |
| agentmamba_lexical_only | raw_candidate_share | 24 | 0.000 | 0.000 | 1.000 | 1.000 |
| agentmamba_lexical_only | calibrated | 0 | n/a | n/a | n/a | n/a |

Pairs within a scenario are not independent; CIs and p-values are indicative only. Latency 'cold' = rows whose Mamba calls were all score-cache misses.
