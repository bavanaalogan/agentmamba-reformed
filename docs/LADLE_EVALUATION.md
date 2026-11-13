# Ladle evaluation: short vs long context, and the linear-time check

All numbers are read from `experiments/results/20261113-120248_ladle_eval_mamba/summary.json` and
`experiments/results/linear_time/linear_time.json` (CPU, i9-13900H, 12 threads, Mamba score cache off).
**All data is fictional synthetic ladle logs.**

## Datasets (`scripts/generate_ladle_datasets.py`, `src/agentmamba/data/ladle_benchmark.py`)
4 melt shops (SMS1-SMS4), 10 pieces of equipment each (8 ladles, a ladle furnace, a transfer car), 15 questions
per shop (60 total). Question types: maintenance action, alarm at a time, shell-temperature reading at a time,
inspection finding, confirmed root cause, clearance after repair (answerable); root cause still under
investigation, clearance still pending, unknown ladle, wrong date (must refuse); thermography vs historian
disagreement (must flag the conflict); four paraphrases.

| dataset | records / shop | Mamba tokens of history | gold |
|---|---|---|---|
| short | ~20 (incident shift only) | ~1,300 | identical |
| long | ~3,000 (shift + 3 days ops + older history & manual + 6 weeks of cycles) | ~228,000 | identical |

## Protocol (`scripts/run_ladle_eval.py`)
Both systems get the same question and the same 2,048-token window of the most recent records. Baseline Mamba
answers from the window (candidate spans ranked by mamba-130m; no retrieval, no verification). AgentMamba starts
from the same window, retrieves from episodic memory (SQLite + FAISS, store-all) only when its evidence check
says the window is insufficient, and verifies before answering. Gold is read only after all answers exist.

## Results (60 questions per dataset; long = after all 4 stages)

| metric | short: Baseline | short: AgentMamba | long: Baseline | long: AgentMamba |
|---|---|---|---|---|
| correct answers, answerable questions (n=40) | 55% | **90%** | 0% | **90%** |
| correct refusals, unanswerable (n=16) | 0% | **100%** | 0% | **100%** |
| conflicts flagged (n=4) | 0% | **100%** | 0% | **100%** |
| wrong answers given (all 60) | 63% | **0%** | 73% | **0%** |
| required evidence seen by the model | 100% | 100% | 0% | 100% |
| median seconds / question | 15.4 | 8.6 | 17.1 | 8.1 |
| median tokens Mamba processed | 1,343 | 761 | 1,902 | 742 |

Baseline answerable accuracy as history grows: 55% (20 records) → 15% (333) → 5% (488) → 0% (3,005).
AgentMamba: 90% at every stage. Paired exact McNemar test on per-question correctness, AgentMamba vs Baseline:
p < 0.001 on both datasets.

**Where plain Mamba does well on short context:** inspection findings, confirmed root causes and clearance
records (100% each). **Where it fails even on short context:** alarm, temperature and maintenance rows (0-50%):
several ladles have similar rows in the same shift log and mamba-130m often ranks another ladle's row first. It
also never refuses and cannot detect conflicting sources.

**AgentMamba's remaining error:** the paraphrase "What work was carried out on LADLE-xx on <date>?" (0/4 on both
datasets): it refuses instead of answering. It never gives a wrong answer for it.

## Linear time (`scripts/benchmark_linear_time.py`)
* One mamba-130m forward pass over L tokens (256 ... 4,096): time = 7.9 ms × L − 0.70 s, **R² = 0.998**.
  (8,192 tokens exhausted 16 GB RAM with the reference CPU scan, so the sweep stops at 4,096.)
* AgentMamba on SMS1 as memory grows from 20 to 3,012 records: Mamba processes a median of **739 tokens per
  question at every size** (max 1,530); median query time 8.0 / 6.2 / 5.8 / 6.8 s; FAISS + SQLite retrieval
  0.05-0.20 s. Reading the whole 228,273-token history with Mamba would take ~1,807 s **(projected from the
  fit, not run)**.

So the expensive part (Mamba) stays bounded and linear; the memory adds a cheap retrieval step that grows with
the number of stored records (exact FAISS search is a linear scan of 384-d vectors).

## Fix made during this work
`text_candidates` offered the signal NAME ("shell_temperature") as an answer to "what was the shell temperature
reading ...", and Mamba preferred it. Values that only restate the question's own words are now excluded (both
systems use this extractor). Regression test: `tests/unit/test_ladle_benchmark.py`.
