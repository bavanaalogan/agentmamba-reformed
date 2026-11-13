# AgentMamba

*Learning plant-wide episodic memory retention and conditional evidence retrieval for industrial troubleshooting*
(provisional title; no novelty claim, see `docs/RESEARCH_METHODOLOGY.md`).

AgentMamba is a research prototype that keeps a **budgeted, persistent memory of steel-plant operational events**.
A **trained write gate** decides what to keep. SQLite is the authoritative store and FAISS provides similarity search. To answer an engineer's
question it **first tries the permitted initial context** (answers ranked by `state-spaces/mamba-130m-hf`). It **retrieves
cross-stage evidence only when the evidence check says that context is insufficient**, then **verifies identity,
time, version and causal claims** before answering or abstaining.

> **Not a safety system.** This is research decision support. It never authorizes production changes or
> safety-critical actions. All cross-stage scenarios are **fictional synthetic data** (see `docs/DATASET_CARD.md`).

## Quick start: Ladle Incident Investigator (agentmamba-v2)
This merged copy adds the ladle use case from the project review (enter a ladle ID, a date / time and what you
want to know, then review evidence-linked findings) and two synthetic ladle datasets:

| dataset | what it contains | purpose |
|---|---|---|
| `data/ladle_benchmark/short` | the incident-shift log only, ~20 records (~1,300 Mamba tokens) per melt shop | everything fits in Mamba's window: shows how plain Mamba does on short context |
| `data/ladle_benchmark/long` | the SAME shift + ~3,000 later/older records (~228,000 tokens), in 4 cumulative stages | the evidence falls out of any fixed window: shows what episodic memory + selective retrieval add |

Both datasets ask the same 15 questions per melt shop (4 shops, 60 questions) with identical gold answers and
evidence record IDs; only the amount of history differs. All data is fictional.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts\generate_ladle_datasets.py                      # writes both datasets
& $py scripts\run_ladle_eval.py --threads 12 --no-cache        # Baseline Mamba vs AgentMamba, both datasets
& $py scripts\benchmark_linear_time.py --threads 12           # is Mamba still linear-time inside AgentMamba?
& $py scripts\launch_dashboard.py                             # http://localhost:8501
```
`run_ladle_eval.py` also stores the memory snapshots (`storage/ladle/<dataset>/<shop>/stage<k>`) that the
dashboard's **Investigate** page uses. Add `--answerer first_candidate` for a 2-minute smoke run without Mamba.

Dashboard pages: **Overview** (pipeline + headline results), **Investigate** (guided ladle question, own words, or
dataset samples; side-by-side answers with a verification badge, a step-by-step retrieval timeline and the cited
source records), **Results** (short vs long context, accuracy as history grows, linear-time check), **Upload logs**,
**Memory explorer**, **System**.

Metrics kept (each maps to the objective): correct answers on answerable questions, correct refusals on
unanswerable questions, required evidence seen by the model, wrong answers given, seconds per question, tokens
Mamba reads per question, and the linear fit (R²) of Mamba time vs input length. BLEU/ROUGE/NDCG/MAP/ECE/Brier and
the composite score were removed from the dashboard (the older evaluation code is still in `src/` and `scripts/`).

## Contents
- `src/agentmamba/`: library (data, models, gate, memory, retrieval, verification, pipeline, evaluation, api)
- `scripts/`: CLI entry points (below)
- `app/streamlit_app.py`: dashboard
- `tests/`: unit, integration and real-model tests
- `docs/`: architecture, methodology, data audit, dataset card, experiments, limitations, `REPORT.md`
- `experiments/results/`: one directory per run (config, per-question predictions, summary JSON/MD) + `comparison.md`

## Installation (Windows PowerShell, CPU only)
```powershell
cd agentmamba
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
`transformers==5.20.2` refuses `torch<2.6` (observed: it silently disables PyTorch with 2.5.1). The model weights
(`state-spaces/mamba-130m-hf`, `sentence-transformers/all-MiniLM-L6-v2`) download from Hugging Face on first use;
after that you can set `$env:HF_HUB_OFFLINE = "1"`.

## Reproduce everything
```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts\probe_mamba.py                                # Mamba CPU feasibility -> artifacts/models/mamba_probe.json
& $py scripts\audit_data.py                                 # data audit + local public files
& $py scripts\generate_data.py                              # synthetic benchmark -> data/synthetic/main
& $py scripts\validate_data.py --data data\synthetic\main   # split integrity / annotation checks
& $py scripts\run_public_experiment.py                      # builds data/public/uci851 on first run (Category A)
& $py scripts\extract_features.py --data data\synthetic\main  # MiniLM + Mamba features, cached (~1 h on CPU)
& $py scripts\train_gate.py --data data\synthetic\main      # gates (3 variants), intent classifier, controller
& $py scripts\run_experiment.py --data data\synthetic\main --methods all
& $py scripts\run_experiment.py --data data\synthetic\main --sweep
& $py scripts\evaluate.py                                   # -> experiments/results/comparison.{json,md}
```
Note: `run_public_experiment.py` needs the trained controller. On a fresh clone, run it once to build the public
benchmark (it stops at the missing controller), then again after `train_gate.py`.

## Baseline Mamba vs AgentMamba (operational logs, growing history)
```powershell
& $py scripts\generate_log_benchmark.py                                  # synthetic log benchmark (fictional)
& $py scriptsun_context_growth.py --split dev  --tag mamba --threads 6  # dev run + calibrators (~3 h CPU)
& $py scriptsun_context_growth.py --split test --tag final --threads 6  # final test run (~5 h CPU)
```
In the dashboard: **Ingestion** accepts CSV/JSON/JSONL logs or TXT/MD documents and defaults to `store_all` so
uploaded evidence remains searchable; common timestamp formats, available severity/event fields, and source text are
preserved for evidence checks, while unknown fields remain searchable. The configured learned gate is available as an
explicit alternative, but was not validated for arbitrary uploaded logs. **Investigation → Baseline
Mamba vs AgentMamba** runs both systems on the same question and initial context. Its default baseline mode uses
Mamba to rank candidate spans independently for each question part, using only the shared context and no retrieval or
verification. Raw greedy generation remains available as a diagnostic, but the base checkpoint is not instruction-tuned
and may return incomplete answers. The checkpoint remains frozen: uploaded records are stored in memory and
placed in the inference context, not learned into the weights. This base checkpoint is not instruction-tuned, so
raw-generation answers can be poor and have no candidate-share confidence or evidence verification. The staged
context-growth results under **Experiments** retain the candidate-ranking baseline used by that benchmark. The method
is in `docs/EXPERIMENTS.md` and the findings are in `docs/REPORT.md`.

For an interactive growing-history demonstration, upload a small log file in **Ingestion**, then ask its question
under **Investigation → Baseline Mamba vs AgentMamba**. The baseline ranks an answer span for each question part from
the initial context; AgentMamba
uses the same initial context and can retrieve from the persistent memory profile. AgentMamba checks evidence for each
part of a compound question independently, binds time answers to the supporting event record, returns complete verified
initial evidence without retrieval, and retrieves when required evidence is incomplete. Answers include source
citations. Optionally enter the expected answer and
evidence record IDs for post-answer scoring; these references are not sent to either system. Upload additional logs
into the same memory profile and rerun the question to observe whether the original evidence leaves the baseline's
recent-context window and whether conditional retrieval recovers it. A free-question comparison is only a
single-question diagnostic; use the staged context-growth benchmark under **Experiments** for aggregate accuracy,
calibration, and statistical comparisons. Confidence values are not interchangeable with correctness: candidate
share is uncalibrated unless the applicable calibrator is available, and retrieval similarity is not answer confidence.
The comparison result ends with a metric table for answer overlap (exact match, token F1/error, ROUGE-1/L, BLEU-1),
retrieval ranking (when expected evidence IDs are supplied and retrieval ran), and per-query efficiency. Answer
metrics require an expected answer; retrieval metrics require relevant evidence IDs, otherwise the table shows N/A.
Confidence is shown as calibrated confidence when available, otherwise as an explicitly uncalibrated Mamba candidate
share; verified answer-part coverage is shown separately and is not a probability of correctness.
Answers are displayed as structured rows by requested information, answer, evidence record IDs, and verification
status instead of a single compound-answer paragraph.

## Configuration
- `config/default.yaml`: models, gate, memory policy and budget, retrieval (top-k, expansion depth, reranker weights).
- `config/experiments.yaml`: methods A–H, ablations, budget sweep.
- Override in code with `load_config(overrides={"memory.budget_fraction": 0.05})`.

## Application memory, querying, API, dashboard
```powershell
& $py scripts\build_memory.py --data data\synthetic\main --split test --budget-fraction 0.10   # learned gate + budget
& $py scripts\build_memory.py --public-uci851        # add the real public daily records (explicit store_all policy)
& $py scripts\run_query.py "Which ladle carried heat C05-H0012?" --mode conditional
& $py scripts\ingest_events.py path\to\records.jsonl
& $py -m uvicorn agentmamba.api.main:app --app-dir src --port 8000      # http://127.0.0.1:8000/docs
& $py scripts\launch_dashboard.py                                      # http://localhost:8501
```

## Tests
```powershell
& $py -m pytest -m "not model"     # fast deterministic suite (fixture embedder, non-model answerer)
& $py -m pytest -m model           # loads the real Mamba and MiniLM weights (slow on CPU)
```

## Troubleshooting
- **"PyTorch was not found" from transformers:** install `torch>=2.6` (CPU wheel above).
- **`trained gate artifact not found`:** run `scripts/train_gate.py`. The primary pipeline never substitutes a heuristic.
- **Index inconsistent / missing:** the memory rebuilds FAISS from SQLite on open (see `recovery_note` in
  `/status`), or run `scripts/build_memory.py --rebuild-index`.
- **Slow queries:** Mamba runs the reference PyTorch selective scan on CPU (no `mamba_ssm`/`causal_conv1d` kernels).
  A query that retrieves takes a few seconds. Scores are memoized in `artifacts/models/mamba_score_cache.sqlite3`.

## CPU limitations
No fine-tuning. Mamba is frozen and used only for candidate likelihood ranking and optional gate features. Mamba
feature extraction for 17k records took about an hour on an i7-1360P. Answering latency is reported per run in
`summary.json`.
