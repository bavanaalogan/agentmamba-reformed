# Ladle evaluation: Baseline Mamba vs AgentMamba (answerer=first_candidate, window=2048 tokens)

## short context (stage 1)

| system | n | answerable accuracy | correct refusals | conflicts flagged | wrong answers | evidence seen | median latency (s) | median Mamba tokens |
|---|---|---|---|---|---|---|---|---|
| baseline_mamba | 60 | 0.675 | 0.000 | 0.000 | 0.550 | 1.000 | 0.01 | n/a |
| agentmamba | 60 | 0.900 | 1.000 | 1.000 | 0.000 | 1.000 | 0.01 | n/a |

## long context (stage 4)

| system | n | answerable accuracy | correct refusals | conflicts flagged | wrong answers | evidence seen | median latency (s) | median Mamba tokens |
|---|---|---|---|---|---|---|---|---|
| baseline_mamba | 60 | 0.000 | 0.000 | 0.000 | 0.733 | 0.000 | 0.00 | n/a |
| agentmamba | 60 | 0.900 | 1.000 | 1.000 | 0.000 | 1.000 | 0.23 | n/a |

