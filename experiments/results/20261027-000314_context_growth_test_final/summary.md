# Context-growth experiment (test split, answerer=mamba)

75 questions x stages; scenarios P1, P2, P3, P5, P6. Synthetic log benchmark.

## Overall (all stages pooled)

| condition | n | accuracy [95% CI] | grounded | answerable acc. | correct abstention | conflict det. | incorrect answers | unsupported claims / answered | evidence in context | Recall@k | median latency cold (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_mamba_768 | 300 | 0.123 [0.0867, 0.1633] | 0.123 | 0.110 | 0.188 | 0.000 | 0.503 | 0.873 | 0.114 | n/a | 9.06 |
| baseline_mamba_2048 | 300 | 0.197 [0.1533, 0.2433] | 0.170 | 0.220 | 0.188 | 0.000 | 0.570 | 0.833 | 0.250 | n/a | 19.67 |
| retrieval_always | 300 | 0.853 [0.8133, 0.8933] | 0.853 | 0.800 | 1.000 | 0.800 | 0.013 | 0.024 | 0.114 | 1.000 | 9.92 |
| agentmamba | 300 | 0.853 [0.8133, 0.8933] | 0.853 | 0.800 | 1.000 | 0.800 | 0.013 | 0.024 | 0.114 | 1.000 | n/a |
| agentmamba_no_verification | 300 | 0.410 [0.3567, 0.4667] | 0.410 | 0.615 | 0.000 | 0.000 | 0.573 | 0.583 | 0.114 | 1.000 | 12.13 |
| agentmamba_no_retrieval | 300 | 0.350 [0.2967, 0.4067] | 0.350 | 0.125 | 1.000 | 0.000 | 0.003 | 0.038 | 0.114 | n/a | n/a |
| agentmamba_vector_only | 75 | 0.733 [0.64, 0.84] | 0.733 | 0.680 | 1.000 | 0.200 | 0.013 | 0.029 | 0.000 | 0.745 | 8.33 |
| agentmamba_lexical_only | 75 | 0.853 [0.7733, 0.92] | 0.853 | 0.800 | 1.000 | 0.800 | 0.013 | 0.024 | 0.000 | 1.000 | 9.13 |

## Accuracy by stage

| condition | stage 1 | stage 2 | stage 3 | stage 4 |
|---|---|---|---|---|
| baseline_mamba_768 | 0.293 | 0.133 | 0.000 | 0.067 |
| baseline_mamba_2048 | 0.480 | 0.213 | 0.027 | 0.067 |
| retrieval_always | 0.853 | 0.853 | 0.853 | 0.853 |
| agentmamba | 0.853 | 0.853 | 0.853 | 0.853 |
| agentmamba_no_verification | 0.493 | 0.493 | 0.253 | 0.400 |
| agentmamba_no_retrieval | 0.600 | 0.267 | 0.267 | 0.267 |
| agentmamba_vector_only | - | - | - | 0.733 |
| agentmamba_lexical_only | - | - | - | 0.853 |

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
| agentmamba vs baseline_mamba_768 | correct | 0.1233 | 0.8533 | 0.73 [0.68, 0.78] | 0.0 |
| agentmamba vs baseline_mamba_768 | grounded_correct | 0.1233 | 0.8533 | 0.73 [0.68, 0.78] | 0.0 |
| agentmamba vs baseline_mamba_768 | unsupported_claim (lower better) | 0.5033 | 0.0133 | -0.49 [-0.55, -0.43] | 0.0 |
| agentmamba vs baseline_mamba_768 | incorrect_answer (lower better) | 0.5033 | 0.0133 | -0.49 [-0.55, -0.43] | 0.0 |
| agentmamba vs baseline_mamba_2048 | correct | 0.1967 | 0.8533 | 0.6567 [0.5967, 0.7167] | 0.0 |
| agentmamba vs baseline_mamba_2048 | grounded_correct | 0.17 | 0.8533 | 0.6833 [0.6267, 0.74] | 0.0 |
| agentmamba vs baseline_mamba_2048 | unsupported_claim (lower better) | 0.5967 | 0.0133 | -0.5833 [-0.64, -0.5267] | 0.0 |
| agentmamba vs baseline_mamba_2048 | incorrect_answer (lower better) | 0.57 | 0.0133 | -0.5567 [-0.61, -0.5] | 0.0 |
| agentmamba vs agentmamba_no_retrieval | correct | 0.35 | 0.8533 | 0.5033 [0.4467, 0.56] | 0.0 |
| agentmamba vs agentmamba_no_retrieval | grounded_correct | 0.35 | 0.8533 | 0.5033 [0.4467, 0.56] | 0.0 |
| agentmamba vs agentmamba_no_retrieval | unsupported_claim (lower better) | 0.0033 | 0.0133 | 0.01 [0.0, 0.0233] | 0.25 |
| agentmamba vs agentmamba_no_retrieval | incorrect_answer (lower better) | 0.0033 | 0.0133 | 0.01 [0.0, 0.0233] | 0.25 |
| agentmamba vs agentmamba_no_verification | correct | 0.41 | 0.8533 | 0.4433 [0.37, 0.5133] | 0.0 |
| agentmamba vs agentmamba_no_verification | grounded_correct | 0.41 | 0.8533 | 0.4433 [0.37, 0.5133] | 0.0 |
| agentmamba vs agentmamba_no_verification | unsupported_claim (lower better) | 0.5733 | 0.0133 | -0.56 [-0.6167, -0.5033] | 0.0 |
| agentmamba vs agentmamba_no_verification | incorrect_answer (lower better) | 0.5733 | 0.0133 | -0.56 [-0.6167, -0.5033] | 0.0 |
| agentmamba vs retrieval_always | correct | 0.8533 | 0.8533 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs retrieval_always | grounded_correct | 0.8533 | 0.8533 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs retrieval_always | unsupported_claim (lower better) | 0.0133 | 0.0133 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs retrieval_always | incorrect_answer (lower better) | 0.0133 | 0.0133 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba_no_verification vs baseline_mamba_768 | correct | 0.1233 | 0.41 | 0.2867 [0.2233, 0.3533] | 0.0 |
| agentmamba_no_verification vs baseline_mamba_768 | grounded_correct | 0.1233 | 0.41 | 0.2867 [0.2233, 0.3533] | 0.0 |
| agentmamba_no_verification vs baseline_mamba_768 | unsupported_claim (lower better) | 0.5033 | 0.5733 | 0.07 [0.0133, 0.1267] | 0.02382 |
| agentmamba_no_verification vs baseline_mamba_768 | incorrect_answer (lower better) | 0.5033 | 0.5733 | 0.07 [0.0133, 0.1267] | 0.02382 |
| agentmamba_no_retrieval vs baseline_mamba_768 | correct | 0.1233 | 0.35 | 0.2267 [0.1833, 0.2733] | 0.0 |
| agentmamba_no_retrieval vs baseline_mamba_768 | grounded_correct | 0.1233 | 0.35 | 0.2267 [0.1833, 0.2733] | 0.0 |
| agentmamba_no_retrieval vs baseline_mamba_768 | unsupported_claim (lower better) | 0.5033 | 0.0033 | -0.5 [-0.5567, -0.44] | 0.0 |
| agentmamba_no_retrieval vs baseline_mamba_768 | incorrect_answer (lower better) | 0.5033 | 0.0033 | -0.5 [-0.5567, -0.44] | 0.0 |
| baseline_mamba_2048 vs baseline_mamba_768 | correct | 0.1233 | 0.1967 | 0.0733 [0.0433, 0.1033] | 0.0 |
| baseline_mamba_2048 vs baseline_mamba_768 | grounded_correct | 0.1233 | 0.17 | 0.0467 [0.0233, 0.07] | 0.000122 |
| baseline_mamba_2048 vs baseline_mamba_768 | unsupported_claim (lower better) | 0.5033 | 0.5967 | 0.0933 [0.0567, 0.1333] | 4e-06 |
| baseline_mamba_2048 vs baseline_mamba_768 | incorrect_answer (lower better) | 0.5033 | 0.57 | 0.0667 [0.0333, 0.1] | 0.000325 |
| agentmamba vs agentmamba_vector_only | correct | 0.7333 | 0.8533 | 0.12 [0.0533, 0.1867] | 0.003906 |
| agentmamba vs agentmamba_vector_only | grounded_correct | 0.7333 | 0.8533 | 0.12 [0.0533, 0.1867] | 0.003906 |
| agentmamba vs agentmamba_vector_only | unsupported_claim (lower better) | 0.0133 | 0.0133 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs agentmamba_vector_only | incorrect_answer (lower better) | 0.0133 | 0.0133 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs agentmamba_lexical_only | correct | 0.8533 | 0.8533 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs agentmamba_lexical_only | grounded_correct | 0.8533 | 0.8533 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs agentmamba_lexical_only | unsupported_claim (lower better) | 0.0133 | 0.0133 | 0.0 [0.0, 0.0] | 1.0 |
| agentmamba vs agentmamba_lexical_only | incorrect_answer (lower better) | 0.0133 | 0.0133 | 0.0 [0.0, 0.0] | 1.0 |

## Calibration (answered questions)

| condition | confidence | n | Brier | ECE | mean conf. | acc. |
|---|---|---|---|---|---|---|
| baseline_mamba_768 | raw_candidate_share | 173 | 0.516 | 0.569 | 0.696 | 0.127 |
| baseline_mamba_768 | calibrated | 173 | 0.111 | 0.004 | 0.131 | 0.127 |
| baseline_mamba_2048 | raw_candidate_share | 215 | 0.359 | 0.385 | 0.589 | 0.205 |
| baseline_mamba_2048 | calibrated | 215 | 0.167 | 0.065 | 0.270 | 0.205 |
| retrieval_always | raw_candidate_share | 164 | 0.007 | 0.013 | 0.989 | 0.976 |
| retrieval_always | calibrated | 0 | n/a | n/a | n/a | n/a |
| agentmamba | raw_candidate_share | 164 | 0.011 | 0.015 | 0.991 | 0.976 |
| agentmamba | calibrated | 0 | n/a | n/a | n/a | n/a |
| agentmamba_no_verification | raw_candidate_share | 295 | 0.252 | 0.236 | 0.653 | 0.417 |
| agentmamba_no_verification | calibrated | 295 | 0.196 | 0.055 | 0.402 | 0.417 |
| agentmamba_no_retrieval | raw_candidate_share | 26 | 0.038 | 0.038 | 1.000 | 0.962 |
| agentmamba_no_retrieval | calibrated | 0 | n/a | n/a | n/a | n/a |
| agentmamba_vector_only | raw_candidate_share | 35 | 0.029 | 0.029 | 1.000 | 0.971 |
| agentmamba_vector_only | calibrated | 0 | n/a | n/a | n/a | n/a |
| agentmamba_lexical_only | raw_candidate_share | 41 | 0.004 | 0.010 | 0.986 | 0.976 |
| agentmamba_lexical_only | calibrated | 0 | n/a | n/a | n/a | n/a |

Pairs within a scenario are not independent; CIs and p-values are indicative only. Latency 'cold' = rows whose Mamba calls were all score-cache misses.
