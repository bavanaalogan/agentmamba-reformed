# Ladle evaluation: Baseline Mamba vs AgentMamba (answerer=mamba, window=2048 tokens)

## short context (stage 1)

| system | n | answerable accuracy | correct refusals | conflicts flagged | wrong answers | evidence seen | median latency (s) | median Mamba tokens |
|---|---|---|---|---|---|---|---|---|
| baseline_mamba | 60 | 0.550 | 0.000 | 0.000 | 0.633 | 1.000 | 15.36 | 1343 |
| agentmamba | 60 | 0.900 | 1.000 | 1.000 | 0.000 | 1.000 | 8.59 | 761 |

## long context (stage 4)

| system | n | answerable accuracy | correct refusals | conflicts flagged | wrong answers | evidence seen | median latency (s) | median Mamba tokens |
|---|---|---|---|---|---|---|---|---|
| baseline_mamba | 60 | 0.000 | 0.000 | 0.000 | 0.733 | 0.000 | 17.11 | 1902 |
| agentmamba | 60 | 0.900 | 1.000 | 1.000 | 0.000 | 1.000 | 8.14 | 742 |

