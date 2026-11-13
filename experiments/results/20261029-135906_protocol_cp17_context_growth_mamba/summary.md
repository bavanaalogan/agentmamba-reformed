# Protocol cp17_context_growth (answerer=mamba)

_5 questions x 2 stages on one scenario: a protocol demonstration, NOT a benchmark. Benchmark results: context-growth test run._

Cold start: Mamba load 205.66 s, embedding load 5.86 s (excluded from query latency).

## Per question

| stage | question | system | status | answer | expected | correct | sufficiency | retrieval | evidence IDs | context tok | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CP17-Q1 | baseline_mamba | answered | ALM-CP17-884 | 8.4 mm/s | False | - | False | LOG-3410c4981b10 | 844 | 6.8953 |
| 1 | CP17-Q1 | agentmamba | answered | 8.4 mm/s | 8.4 mm/s | True | sufficient | False | LOG-1f01efbc1f1d | 844 | 6.1862 |
| 1 | CP17-Q2 | baseline_mamba | answered | WO-CP17-211 | WO-CP17-211 | True | - | False | LOG-19b3554e3d46 | 841 | 6.3294 |
| 1 | CP17-Q2 | agentmamba | answered | WO-CP17-211 | WO-CP17-211 | True | sufficient | False | LOG-19b3554e3d46 | 841 | 5.6162 |
| 1 | CP17-Q3 | baseline_mamba | answered | 8.4 mm/s | 4.1 mm/s | False | - | False | LOG-1f01efbc1f1d | 844 | 5.1760 |
| 1 | CP17-Q3 | agentmamba | answered | 4.1 mm/s | 4.1 mm/s | True | sufficient | False | LOG-3139f23ccc55 | 844 | 5.4310 |
| 1 | CP17-Q4 | baseline_mamba | answered | ALM-CP17-884 | None | False | - | False | LOG-3410c4981b10 | 847 | 6.4501 |
| 1 | CP17-Q4 | agentmamba | abstained | None | None | True | insufficient | True |  | 847 | 0.0805 |
| 1 | CP17-Q5 | baseline_mamba | abstained | None | None | True | - | False |  | 840 | 0.0024 |
| 1 | CP17-Q5 | agentmamba | abstained | None | None | True | insufficient | True |  | 840 | 0.0358 |
| 2 | CP17-Q1 | baseline_mamba | answered | ALM-HIST-0276 | 8.4 mm/s | False | - | False | LOG-192b902f16f3 | 1864 | 12.2580 |
| 2 | CP17-Q1 | agentmamba | answered | 8.4 mm/s | 8.4 mm/s | True | insufficient | True | LOG-1f01efbc1f1d | 1864 | 4.5526 |
| 2 | CP17-Q2 | baseline_mamba | answered | WO-HIST-0285 | WO-CP17-211 | False | - | False | LOG-edf195d95487 | 1861 | 11.8931 |
| 2 | CP17-Q2 | agentmamba | answered | WO-CP17-211 | WO-CP17-211 | True | insufficient | True | LOG-19b3554e3d46 | 1861 | 4.1680 |
| 2 | CP17-Q3 | baseline_mamba | answered | 3.2 mm/s | 4.1 mm/s | False | - | False | LOG-3f7e569996bc | 1864 | 11.2398 |
| 2 | CP17-Q3 | agentmamba | answered | 4.1 mm/s | 4.1 mm/s | True | insufficient | True | LOG-3139f23ccc55 | 1864 | 4.7657 |
| 2 | CP17-Q4 | baseline_mamba | answered | ALM-HIST-0276 | None | False | - | False | LOG-192b902f16f3 | 1867 | 12.6202 |
| 2 | CP17-Q4 | agentmamba | abstained | None | None | True | insufficient | True |  | 1867 | 0.0991 |
| 2 | CP17-Q5 | baseline_mamba | abstained | None | None | True | - | False |  | 1860 | 0.0030 |
| 2 | CP17-Q5 | agentmamba | abstained | None | None | True | insufficient | True |  | 1860 | 0.0516 |

## Metrics (both stages pooled)

| category | metric | Baseline Mamba | AgentMamba | difference | n (B / A) |
|---|---|---|---|---|---|
| Answer generation | exact_match (higher) | 0.1667 | 1.0000 | 0.8333 | 6 / 6 |
| Answer generation | f1 (higher) | 0.3333 | 1.0000 | 0.6667 | 6 / 6 |
| Answer generation | rouge_1 (higher) | 0.3333 | 1.0000 | 0.6667 | 6 / 6 |
| Answer generation | rouge_l (higher) | 0.3333 | 1.0000 | 0.6667 | 6 / 6 |
| Answer generation | bleu_1 (higher) | 0.3333 | 1.0000 | 0.6667 | 6 / 6 |
| Answer generation | correct_answer_rate (higher) | 0.3000 | 1.0000 | 0.7000 | 10 / 10 |
| Answer generation | incorrect_answer_rate (lower) | 0.7000 | 0.0000 | -0.7000 | 10 / 10 |
| Answer generation | unsupported_answer_rate (lower) | 0.8750 | 0.0000 | -0.8750 | 8 / 6 |
| Answer generation | correct_abstention_rate (higher) | 0.5000 | 1.0000 | 0.5000 | 4 / 4 |
| Error analysis | answer_error_rate (lower) | 0.8333 | 0.0000 | -0.8333 | 6 / 6 |
| Error analysis | false_answer_rate (lower) | 0.8333 | 0.0000 | -0.8333 | 6 / 6 |
| Error analysis | abstention_error_rate (lower) | 0.0000 | 0.0000 | 0.0000 | 6 / 6 |
| Error analysis | unsupported_claim_rate (lower) | 0.8750 | 0.0000 | -0.8750 | 8 / 6 |
| Error analysis | retrieval_failure_rate (lower) | N/A | 0.0000 | N/A | 0 / 3 |
| Error analysis | evidence_miss_rate (lower) | 0.5000 | 0.0000 | -0.5000 | 6 / 6 |
| Error analysis | brier (lower) | N/A | N/A | N/A | 8 / 0 |
| Error analysis | ece (lower) | N/A | N/A | N/A | 8 / 0 |
| Evidence | grounded_answer_rate (higher) | 0.1667 | 1.0000 | 0.8333 | 6 / 6 |
| Evidence | evidence_coverage (higher) | 0.5000 | 1.0000 | 0.5000 | 6 / 6 |
| Retrieval | ndcg@10 (higher) | N/A | 0.8102 | N/A | 0 / 3 |
| Retrieval | map@10 (higher) | N/A | 0.7500 | N/A | 0 / 3 |
| Retrieval | recall@10 (higher) | N/A | 1.0000 | N/A | 0 / 3 |
| Retrieval | recall@1 (higher) | N/A | 0.6667 | N/A | 0 / 3 |
| Retrieval | hit@1 (higher) | N/A | 0.6667 | N/A | 0 / 3 |
| Retrieval | hit@3 (higher) | N/A | 0.6667 | N/A | 0 / 3 |
| Retrieval | hit@10 (higher) | N/A | 1.0000 | N/A | 0 / 3 |
| Retrieval | retrieval_trigger_rate (info) | N/A | 0.7000 | N/A | 0 / 10 |
| Retrieval | unnecessary_retrieval_rate (lower) | N/A | 0.0000 | N/A | 0 / 3 |
| Retrieval | required_evidence_coverage (higher) | 0.5000 | 1.0000 | 0.5000 | 6 / 6 |
| Efficiency | total_s (lower) | 72.8673 | 30.9867 | -41.8806 | 10 / 10 |
| Efficiency | mean_s (lower) | 7.2867 | 3.0987 | -4.1880 | 10 / 10 |
| Efficiency | median_s (lower) | 6.4501 | 4.1680 | -2.2821 | 10 / 10 |
| Efficiency | p95_s (lower) | N/A | N/A | N/A | 10 / 10 |
| Efficiency | generation_mean_s (lower) | 7.2818 | 3.0337 | -4.2481 | 10 / 10 |
| Efficiency | retrieval_mean_s (lower) | N/A | 0.0728 | N/A | 0 / 7 |
| Efficiency | verification_mean_s (lower) | N/A | 0.0087 | N/A | 0 / 7 |
| Efficiency | model_load_s (lower) | 205.6586 | 205.6586 | 0.0000 | 1 / 1 |

Composite (shared components): Baseline 0.3104 | AgentMamba 0.9481
Composite (all available components): Baseline 0.3104 ['answer_quality', 'grounding', 'efficiency'] | AgentMamba 0.9416 ['answer_quality', 'grounding', 'retrieval', 'efficiency']

Retrieval skip savings: {'n_skipped': 3, 'mean_time_saved_s': 5.3913, 'definition': 'forced-retrieval latency minus conditional latency, same query and context, for queries where AgentMamba skipped retrieval'}
