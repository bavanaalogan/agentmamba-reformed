# AgentMamba: Learned Episodic-Memory Retention and Conditional Evidence Retrieval for Plant-Wide Steel Troubleshooting

*Research report on a final-year prototype. Every number below is read from files in `experiments/results/` (see
`comparison.md`). All cross-stage results are measured on a **fictional synthetic benchmark** and do not show
real-plant performance.*

## 1. Abstract
An integrated steel plant spreads the evidence for an incident across stages: steelmaking, ladle, casting,
rolling, quality and maintenance. We built AgentMamba, a CPU-only prototype that (i) keeps a byte-budgeted persistent
memory of operational records, with a trained write gate deciding what to keep, (ii) answers an engineer's question
first from the permitted initial context, with `state-spaces/mamba-130m-hf` ranking candidate answers extracted from
records, (iii) retrieves cross-stage evidence (FAISS plus documented-identifier relationship traversal over SQLite)
only when an evidence-support check says the initial context is insufficient, and (iv) verifies identity, time,
version and causal claims before answering or abstaining.

On a 192-question held-out synthetic test set (8 independent fictional plant campaigns) at a 10% memory budget, the
learned gate kept 46.5% of the test-relevant evidence and 95.8% of the useful rare records. A severity-based heuristic
kept 16.6% and 33.3%, and random selection 10.8% and 12.5%. Accuracy was 0.677 (95% CI 0.609–0.745) for the learned
gate, versus 0.521 (heuristic) and 0.438 (random), with paired exact McNemar p < 0.001 for both. Store-all memory
reached 0.990, and the learned gate matched it at a 50% budget. The controller avoided all 32 unnecessary retrievals
and missed none, without changing any answer (no measurable latency gain on CPU, where Mamba scoring dominates at about
4–5 s per query). Verification was the largest single factor: removing it cut accuracy
from 0.677 to 0.396 and raised unjustified causal claims on the correlation-only questions from 0% to 100%. Mamba-derived gate features and
Mamba answer ranking added no measurable benefit over a feature set without Mamba and a non-model ranking. Every
error of the full system at 10% came from evidence the gate had not retained; the system abstained in all of those
cases. At a 25% budget, however, partial retention hid conflicts and caused 6 confidently wrong answers.

## 2. Problem statement
Can a learned memory-writing gate keep the most useful historical operational evidence under a limited memory
budget, and can a conditional retrieval system recover relevant cross-stage evidence when the current context cannot
answer the question, without fabricating evidence or overclaiming causes?

## 3. Motivation
Keeping every record in a model context is impossible, and keeping every record forever costs storage, index size and
retrieval noise. Selective memory risks losing weak early signals (a hairline crack logged as "info" days before a
ladle breakout). Retrieving for every question adds latency and distractors. Industrial answers must cite their
sources and must not turn correlation into a "root cause".

## 4. Research questions
RQ1 retention under budget; RQ2 cross-stage retrieval; RQ3 conditional retrieval; RQ4 verification; RQ5
quality–efficiency trade-offs (full statements in `RESEARCH_METHODOLOGY.md`).

## 5. Related work
Adaptive retrieval decides *when* to retrieve: Self-RAG (reflection tokens), FLARE (low-confidence tokens) and
Adaptive-RAG (a question-complexity classifier). Our controller is closest to Adaptive-RAG, but its features are
evidence-support checks on the initial context rather than question complexity or token confidence. Learned or
gated memory *writing* for LLM agents is an active area. Examples include "Learn to Memorize" (arXiv 2508.16629), SAGE novelty gating
(arXiv 2605.30711), ConsistencyGate (arXiv 2607.22962), gated-memory routing in multi-agent systems (arXiv 2609.00237) and
AdaMem (arXiv 2606.21144). Learned write gates and novelty features are therefore not new. Industrial RAG work
includes a steel-plant maintenance copilot over CMMS data and manuals (Danieli Automation/ABS, AISTech listing), a
maintenance work-order retriever (SEASONED, Procedia CIRP 2025) and graph-RAG for maintenance chatbots (Fraunhofer).
The search was limited, and several entries were read only as abstracts or listings. **We make no novelty claim.** The
contribution here is an integrated, tested implementation and a controlled evaluation of these ideas on plant-wide
event records.

## 6. Background: Mamba and episodic memory
Mamba is a selective state-space sequence model. Its recurrence decides what to propagate *inside* one forward
pass, which is unrelated to deciding which records to store in an external, persistent memory. We use the 129M-
parameter base model frozen on CPU. Its free generation does not answer record questions: asked which ladle carried
heat H0012, it continued with "Caster C01-CC1 cast heat C01-H0012." (`artifacts/models/mamba_probe.json`). We
therefore use it to score candidate spans by mean token log-likelihood, and optionally as a feature extractor
(token surprisal and pooled hidden state) for the gate.

## 7. Plant-wide operational-memory representation
Every record is an `Event` with a stable ID, stage, type, timestamp (raw value kept, timezone as given), structured
identifiers (asset, heat, ladle, cast sequence, slab, coil, hot-metal batch, work order, document), observations with
the units as recorded, severity, version and supersession, source reference and source type
(public/authorized/synthetic), and the untouched raw record. Relationships are created **only** from identifiers
recorded together in one record (e.g. an assignment record naming heat and ladle → `carried_by`, documented by that
record) or from explicit `supersedes`/reference fields. None are created from time proximity or embedding similarity.
A link survives only if the record that documents it survives.

## 8. Data sources and limitations
The audit (`DATA_AUDIT.md`) found no openly licensed dataset linking several steel stages by identifiers. Two real
CC BY 4.0 datasets were downloaded and parsed: UCI #851 (DAEWOO Steel energy, 35,040 15-minute rows in 2018) and UCI #198
(1,941 plate-fault records). Neither has cross-stage identifiers. Cross-stage experiments therefore use a seeded synthetic
generator (`DATASET_CARD.md`): 30 independent campaigns, 17,165 events and 719 questions in 16 families, split by
campaign into 16/6/8 for train/val/test (test: 4,580 events, 192 questions). UCI #851 supports a separate real-data check:
365 real daily records and 63 templated test-month questions.

## 9. Learned memory gate
- **Features (write-time only):** type and stage one-hot, recorded severity, structure counts, IDF rarity (trained on train
  text, identifiers masked), streaming novelty and redundancy over the last 256 records, PCA-16 of the MiniLM embedding,
  and optionally Mamba NLL plus PCA-16 of the pooled state.
- **Labels:** a record is labelled 1 if it is required evidence for any question of the same split (train for fitting,
  val for selecting C and the threshold).
- **Model:** balanced logistic regression, with an MLP ablation.
- **Budget:** bytes. A streaming selector admits each record and evicts the lowest score while the budget is exceeded.
  Every WRITE/SKIP/EVICT is logged.

Gate quality on held-out data:

| gate | val AUC | test AUC | test AP | test ECE |
|---|---|---|---|---|
| LR + Mamba features | 0.884 | 0.888 | 0.513 | 0.221 |
| LR without Mamba features | 0.890 | 0.888 | 0.511 | 0.219 |
| MLP + Mamba features | 0.891 | 0.892 | 0.512 | 0.012 |

The LR gates are poorly calibrated, as expected with balanced class weights. Only the ranking matters under a budget.

## 10. Persistent storage and retrieval
- **SQLite** is authoritative: records, relationships, version history, gate decisions and an audit trail, with parameterized
  queries and transactions.
- **FAISS** (`IndexIDMap2(IndexFlatIP)`, cosine on normalized MiniLM vectors) stores stable integer IDs equal to the SQLite
  `vector_id`. Its consistency with SQLite is validated on open, and it is rebuilt from SQLite when missing or inconsistent.
- **Retrieval** combines FAISS search filtered through SQLite metadata with bounded BFS over documented entity links from
  the identifiers in the question (depth ≤ 3, ≤ 60 nodes). Shared groupings such as ladle, cast sequence and asset are
  never traversed *through*. It then applies answer-type and temporal metadata filters, pulls in newer versions of
  superseded hits, removes duplicates, reranks linearly and adds the linking records that justify each item.

## 11. Conditional answering and verification
- **Initial stage:** candidates come from the initial context only, Mamba ranks them, and the verifier runs with no store
  handle. Twelve assessment features feed a logistic regression trained on 611 train questions (P(initial context
  sufficient)). The threshold of 0.95 was chosen on 200 val questions (balanced accuracy 1.0). Questions without the
  required identifier are sent to clarification.
- **Verifier:** checks identity (documented link to the anchor, plus rejection of records that name a *different*
  heat/slab/coil/work order), temporal constraints (on-date, before-date, latest-before-link), currency (supersession and
  version), conflicts, process order and causal discipline. A cause is reported only from a record that itself states a
  confirmed cause; otherwise the system returns an association marked `hypothesis_only`.

## 12. Experimental setup
- **Test split:** 192 questions. The answerer (Mamba-130m ranking), embeddings (MiniLM) and evaluation are the same across
  methods unless a row names the change. The primary budget of 10% was fixed in advance.
- **Development:** done on the validation split. One bug (work orders on ladles were sent to clarification) was found while
  reading the first test outputs. It was fixed with a regression test, the controller was retrained, and every test run
  was regenerated. The superseded runs are kept in `experiments/archive/`.
- **Statistics:** bootstrap 95% CIs over questions; exact McNemar tests on paired correctness (no multiple-comparison
  correction).

## 13. Baselines and ablations
A (no retrieval), B (store-all + RAG), C (heuristic + RAG), D (learned + always retrieve), E = H (full system), F (random
+ RAG), G (learned without Mamba features). Ablations: no verification (E, B), vector-only RAG without relationship
expansion (B), first-candidate answerer (E), MLP gate (E), store-all conditional. Budget sweep over 1–50% for four
policies.

## 14. Results (synthetic test split, n = 192)

### 14.1 Main comparison

| method | memory | mode | verify | accuracy [95% CI] | EM | cross-stage acc | false-answer rate | Recall@K |
|---|---|---|---|---|---|---|---|---|
| A no retrieval | (learned 10%) | never | on | 0.385 [0.318, 0.458] | 0.236 | 0.250 | 0.073 | – |
| B store-all RAG | 100% | always | on | **0.990** [0.974, 1.000] | 0.986 | 1.000 | 0.010 | 0.968 |
| B store-all, conditional | 100% | conditional | on | 0.990 [0.974, 1.000] | 0.986 | 1.000 | 0.010 | 0.960 |
| C heuristic RAG | heuristic 10% | always | on | 0.521 [0.453, 0.594] | 0.417 | 0.266 | 0.073 | 0.203 |
| D learned, always | learned 10% | always | on | 0.677 [0.609, 0.745] | 0.625 | 0.359 | 0.042 | 0.489 |
| **E AgentMamba (= H)** | learned 10% | conditional | on | **0.677** [0.609, 0.745] | 0.625 | 0.359 | 0.042 | 0.538 |
| F random RAG | random 10% | always | on | 0.438 [0.370, 0.510] | 0.299 | 0.266 | 0.099 | 0.105 |
| G learned, no Mamba features | learned 10% | conditional | on | 0.667 [0.599, 0.729] | 0.611 | 0.359 | 0.042 | 0.523 |
| E without verification | learned 10% | conditional* | **off** | 0.396 [0.328, 0.464] | 0.361 | 0.156 | 0.526 | 0.489 |
| B without verification | 100% | always | **off** | 0.557 [0.484, 0.630] | 0.576 | 0.688 | 0.432 | 0.968 |
| B vector-only (no link expansion) | 100% | always | on | 0.635 [0.568, 0.703] | 0.549 | 0.375 | 0.047 | 0.449 |
| E first-candidate answerer (not Mamba) | learned 10% | conditional | on | 0.672 [0.604, 0.740] | 0.618 | 0.359 | 0.047 | 0.538 |
| E MLP gate | learned-MLP 10% | conditional | on | 0.677 [0.609, 0.745] | 0.625 | 0.359 | 0.042 | 0.538 |

\*Without verification the initial check cannot mark anything as supported, so the controller retrieved for 95.8% of questions.

Paired tests (exact McNemar; "a/b" = questions only A got right / only B got right):
- D vs C: 35/5, p < 0.001. D vs F: 50/4, p < 0.001.
- E vs B: 2/62, p < 0.001. E vs A: 56/0, p < 0.001. E vs no verification: 55/1, p < 0.001.
- B vs vector-only: 68/0, p < 0.001. B vs B without verification: 83/0, p < 0.001.
- G vs H: 1/3, p = 0.63. E vs first-candidate: 1/0, p = 1.0. E vs D: 0/0. E vs MLP: 5/5, p = 1.0.

### 14.2 RQ1: retention under a budget (conditional mode, verification on)

| budget | learned + Mamba features: acc / evidence kept / useful rare kept | learned, no Mamba features | heuristic | random |
|---|---|---|---|---|
| 1% | 0.479 / 0.149 / 0.375 | 0.484 / 0.149 / 0.375 | 0.385 / 0.000 / 0.000 | 0.396 / 0.021 / 0.021 |
| 5% | 0.630 / 0.402 / 0.844 | 0.630 / 0.402 / 0.844 | 0.385 / 0.000 / 0.000 | 0.411 / 0.050 / 0.062 |
| 10% | 0.677 / 0.465 / 0.958 | 0.667 / 0.452 / 0.958 | 0.521 / 0.166 / 0.333 | 0.438 / 0.108 / 0.125 |
| 25% | 0.792 / 0.705 / 1.000 | 0.797 / 0.718 / 1.000 | 0.656 / 0.394 / 0.615 | 0.521 / 0.261 / 0.271 |
| 50% | 0.990 / 0.959 / 1.000 | 0.990 / 0.954 / 1.000 | 0.859 / 0.755 / 1.000 | 0.661 / 0.548 / 0.552 |
| 100% (store-all) | 0.990 / 1.000 / 1.000 | | | |

The learned gate dominates the heuristic and random baselines at every budget. The heuristic is worst at small budgets
because it spends them on nuisance caster alarms recorded as "major": 246 of its 485 records at 10%. Mamba features
make no consistent difference.

### 14.3 RQ2: cross-stage retrieval
With full memory, documented-link expansion is what makes cross-stage questions answerable. Store-all with expansion
reaches cross-stage accuracy 1.000 and Recall@K 0.968. Vector-only retrieval reaches 0.375 and 0.449; for example,
coil→slab→sequence chains (F03) score 0.06 and four-stage chains (F16) 0.0. Under a 10% learned budget, cross-stage
accuracy falls to 0.359. Individually unremarkable linking records (rolling records, most heat→ladle assignments) were
not retained, so F02, F03 and F16 fail. The system abstains on them instead of answering.

### 14.4 RQ3: conditional retrieval
Retrieval decisions:
- **Retrieved for 79.2% of questions**, against 95.8% for always-retrieve; the 4.2% remainder are the ambiguous questions,
  which go to clarification in both modes.
- **Unnecessary retrievals: 0 of 32**, where the answer was in the context.
- **Missed retrievals: 0 of 152**, where retrieval was needed.

Answers were identical to always-retrieve: E and D differ on 0 questions, and B-conditional and B-always match in accuracy.
The controller result is in-distribution and template-driven. The validation balanced accuracy of 1.0 shows the task
is easy on this benchmark, not that the controller is robust. The Mamba score-margin feature had a coefficient of 0
because initial contexts never contained competing candidates.

### 14.5 RQ4: verification
Removing verification (same memory and retrieval):
- **Learned 10% memory (E):** accuracy fell from 0.677 to 0.396 and the false-answer rate rose from 0.042 to 0.526.
  Unjustified causal claims on correlation-only questions went from 0% to 100%.
- **Store-all memory (B):** accuracy fell from 0.990 to 0.557, and unjustified causal claims rose from 0% to 25%.

With verification, every test run produced 0 invalid citations, and the store-all runs detected all conflicting-record
pairs (conflict recall 1.0, precision 1.0).

### 14.6 Answering model
Replacing Mamba's ranking with "first candidate in time order" changed 1 of 192 answers (p = 1.0). Mamba's
ranking also caused store-all's only two errors: on F04 it preferred "refractory wear within limits" over the true
warning. On this benchmark, typed candidate extraction plus verification does nearly all the work. The Mamba-130m
contribution was not measurable, and we do not claim one.

### 14.7 Real public data (Category A)
On 63 test-month questions over the 365 real UCI #851 daily records (61 templated date lookups and 2 dates outside
coverage), conditional mode answered 61/61 correctly with exact row-range provenance and abstained on both
out-of-coverage dates. The no-retrieval mode abstained on everything (2/63 correct). This is a low-difficulty, single-source
check of ingestion, date filtering, provenance and abstention on real records. It is **not** evidence about cross-stage
reasoning.

### 14.8 Baseline Mamba vs AgentMamba as ingested history grows (synthetic log benchmark)

**Setup.** The method is in `docs/EXPERIMENTS.md` ("Context-growth experiment").
- **Data:** 5 test scenarios × 15 questions × 4 cumulative stages, so 300 (question, stage) pairs per condition.
  Each scenario has about 20 / 337 / 455 / 1,925 records after stages 1–4. All data is fictional and in log formats.
- **Same for every condition:** the frozen `mamba-130m-hf`, the same candidate extractors and the same 768-token
  recent-arrival window as initial context.
- **Development and calibration:** dev scenarios P4/P7/P8 only.
- **Run:** `experiments/results/20261027-000314_context_growth_test_final/`.

**The steel benchmark also gains `A_baseline_mamba`** (the question's designed initial context, no retrieval, no
verification): 0.344 [0.276, 0.411] accuracy, against 0.385 for A (no retrieval, verification on) and 0.677 for E.
Its false-answer rate is 0.198 (A 0.073), and 0.300 of its answers are unsupported (A 0.000).

**Overall (test, all stages pooled, n = 300 per condition)**

| condition | accuracy [95% CI] | answerable acc. (n=200) | correct abstention (n=80) | conflict detected (n=20) | incorrect answers / all | unsupported / answered |
|---|---|---|---|---|---|---|
| Baseline Mamba, 768-token window | 0.123 [0.087, 0.163] | 0.110 | 0.188* | 0.000 | 0.503 | 0.873 |
| Baseline Mamba, 2048-token window | 0.197 [0.153, 0.243] | 0.220 | 0.188* | 0.000 | 0.570 | 0.833 |
| AgentMamba, retrieval disabled | 0.350 [0.297, 0.407] | 0.125 | 1.000 | 0.000 | 0.003 | 0.038 |
| AgentMamba, verification disabled | 0.410 [0.357, 0.467] | 0.615 | 0.000 | 0.000 | 0.573 | 0.583 |
| Retrieval always + verification | 0.853 [0.813, 0.893] | 0.800 | 1.000 | 0.800 | 0.013 | 0.024 |
| **AgentMamba (conditional + verification)** | **0.853** [0.813, 0.893] | 0.800 | 1.000 | 0.800 | 0.013 | 0.024 |
| AgentMamba, vector channel only (stage 4, n=75) | 0.733 [0.640, 0.840] | 0.680 | 1.000 | 0.200 | 0.013 | 0.029 |
| AgentMamba, identifier/text + relationship only (stage 4, n=75) | 0.853 [0.773, 0.920] | 0.800 | 1.000 | 0.800 | 0.013 | 0.024 |

\*The baseline cannot abstain. These are cases where its window held no candidate span at all ("no candidate":
42% of the 768-token rows, 28% of the 2048-token rows), not verified abstentions.

**Accuracy by stage (test)**

| condition | stage 1 (~20 rec.) | stage 2 (~337) | stage 3 (~455) | stage 4 (~1,925) |
|---|---|---|---|---|
| Baseline Mamba 768 | 0.293 | 0.133 | 0.000 | 0.067 |
| Baseline Mamba 2048 | 0.480 | 0.213 | 0.027 | 0.067 |
| AgentMamba, retrieval disabled | 0.600 | 0.267 | 0.267 | 0.267 |
| AgentMamba, verification disabled | 0.493 | 0.493 | 0.253 | 0.400 |
| AgentMamba | 0.853 | 0.853 | 0.853 | 0.853 |
| Required evidence inside the 768-token window | 0.455 | 0.000 | 0.000 | 0.000 |
| Required evidence inside the 2048-token window | 1.000 | 0.000 | 0.000 | 0.000 |

**What causes the baseline's decline.** The decline was measured, not assumed. At stage 1 the 2048-token window
contains all required evidence, yet the baseline answers 0.480 correctly. The 768-token window contains 45.5% of
the evidence, and the baseline answers 0.293.

After stage 1 the newer arrivals push every piece of required evidence out of both windows (0% inside). The
baseline's remaining correct answers are then mostly "no candidate" outcomes on should-abstain questions, plus
chance matches.

So there are two separate effects:
1. **Even with the evidence in view,** frozen Mamba-130m picks the right span among extracted candidates only about
   half the time.
2. **The context policy drops the evidence** once newer records arrive.

This is a property of the declared window policy (most recent arrivals within N tokens). It is not Mamba
"forgetting": Mamba has no hard context limit, and a question-dependent selection is exactly what retrieval adds.

**Where AgentMamba's gain comes from (paired, same question and stage, n = 300)**

| B vs A | correct: A → B | B − A [95% CI] | McNemar p |
|---|---|---|---|
| AgentMamba vs Baseline 768 | 0.123 → 0.853 | +0.730 [0.680, 0.780] | < 1e-6 |
| AgentMamba vs Baseline 2048 | 0.197 → 0.853 | +0.657 [0.597, 0.717] | < 1e-6 |
| Retrieval only (no-verification vs Baseline 768) | 0.123 → 0.410 | +0.287 [0.223, 0.353] | < 1e-6 |
| Verification only (no-retrieval vs Baseline 768) | 0.123 → 0.350 | +0.227 [0.183, 0.273] | < 1e-6 |
| Verification on top of retrieval (AgentMamba vs no-verification) | 0.410 → 0.853 | +0.443 [0.370, 0.513] | < 1e-6 |
| Retrieval on top of verification (AgentMamba vs no-retrieval) | 0.350 → 0.853 | +0.503 [0.447, 0.560] | < 1e-6 |
| Conditional vs always retrieve | 0.853 → 0.853 | 0.000 | 1.0 |
| All channels vs vector only (stage 4) | 0.733 → 0.853 | +0.120 [0.053, 0.187] | 0.004 |

How to read the paired results:
- **Unsupported claims:** AgentMamba makes 0.013 per question, against 0.503 (768-token baseline) and 0.597
  (2048-token baseline). The difference vs 768 is −0.49 [−0.55, −0.43].
- **Larger baseline window:** going from 768 to 2048 tokens raises accuracy by 0.073, but it also raises
  unsupported claims by 0.093 (more candidates in view, no verification).
- **Statistics:** the four stages of a question and the questions of a scenario are correlated. CIs and p-values
  treat pairs as independent, so they overstate certainty. There is no multiple-comparison correction.

**Per-family accuracy (test, all stages)**

| family | Baseline 768 | Baseline 2048 | no retrieval | no verification | AgentMamba |
|---|---|---|---|---|---|
| maintenance action | 0.000 | 0.100 | 0.000 | 0.200 | 1.000 |
| alarm at a time | 0.000 | 0.400 | 0.000 | 1.000 | 1.000 |
| sensor reading at a time | 0.000 | 0.050 | 0.000 | 0.400 | 1.000 |
| inspection finding | 0.250 | 0.250 | 0.250 | 0.700 | 1.000 |
| confirmed root cause | 0.150 | 0.250 | 0.250 | 0.650 | 1.000 |
| documented clearance | 0.200 | 0.250 | 0.250 | 0.650 | 1.000 |
| undocumented root cause (abstain) | 0.250* | 0.250* | 1.000 | 0.000 | 1.000 |
| clearance not documented (abstain) | 0.500* | 0.500* | 1.000 | 0.000 | 1.000 |
| unknown equipment (abstain) | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| wrong date (abstain) | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| conflicting readings | 0.000 | 0.000 | 0.000 | 0.000 | 0.800 |
| paraphrase: "what work was carried out" | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| paraphrase: "which alarm did X trigger" | 0.000 | **0.400** | 0.000 | **1.000** | **0.000** |
| paraphrase: "what did the inspection find" | 0.250 | 0.250 | 0.250 | 0.700 | 1.000 |
| paraphrase: "why did X trip" | 0.250 | 0.250 | 0.250 | 0.850 | 1.000 |

**Negative results and failures (test)**
- **Paraphrase brittleness.** Two of the four paraphrase families score 0% for AgentMamba: it abstains every time,
  because "work", "carried" and "trigger" do not occur in the records and the coverage rule requires every content
  term. On "which alarm did X trigger", the unverified variants answer correctly (2048-token baseline 0.40,
  no-verification 1.00) while AgentMamba abstains. These abstentions are safe but wrong, and they account for all
  of AgentMamba's errors except four. Verification trades recall for precision, and its lexical form is the main
  limitation on this benchmark.
- **One verifier bug: an untimed passage beat timed readings.** In one scenario (P5), the conflicting-readings
  question at all four stages was answered with a shift-note sentence: "High vibration was observed before the
  trip; the root cause is under investigation…". That machine was also the one whose trip was undocumented. The
  sentence names the machine and the date and contains "vibration". It has no timestamp, so the nearest-time rule
  kept it alongside the two timed readings, and the conflict check compares only same-named fields. These are
  AgentMamba's only 4 wrong answers (incorrect-answer rate 0.013).
  - **Proposed fix (not applied):** when the question states a clock time, keep untimed passages only if they
    themselves state that time.
  - **Not applied because** it was found on test outputs; evaluating it needs new data or a fresh split.
- **Conditional retrieval saved little.** It skipped retrieval for 35% of questions at stage 1 (where the window
  held the verified evidence) and never afterwards. With identical accuracy to always retrieving, its saving here
  is 8.7% of retrievals overall.
- **The vector channel alone is weakest.** Stage-4 Recall@k is 0.745 for vector-only, against 1.000 when the
  identifier/text channel is included. Template-like log rows are hard to separate by embedding similarity alone.

**Confidence (test, answered questions only)**

| condition | confidence | n | Brier | ECE | mean confidence | accuracy |
|---|---|---|---|---|---|---|
| Baseline 768 | raw candidate share | 173 | 0.516 | 0.569 | 0.696 | 0.127 |
| Baseline 768 | calibrated on dev | 173 | 0.111 | 0.004 | 0.131 | 0.127 |
| Baseline 2048 | raw candidate share | 215 | 0.359 | 0.385 | 0.589 | 0.205 |
| Baseline 2048 | calibrated on dev | 215 | 0.167 | 0.065 | 0.270 | 0.205 |
| AgentMamba, no verification | raw candidate share | 295 | 0.252 | 0.236 | 0.653 | 0.417 |
| AgentMamba, no verification | calibrated on dev | 295 | 0.196 | 0.055 | 0.402 | 0.417 |
| AgentMamba | raw candidate share | 164 | 0.011 | 0.015 | 0.991 | 0.976 |

- **The baseline's raw share is badly overconfident:** mean 0.70 against 0.13 accuracy. The dev calibrator mostly
  learns the base rate; it fixes calibration but cannot discriminate right from wrong answers.
- **AgentMamba could not be calibrated.** All 96 of its answered dev questions were correct, so a calibrator cannot
  be fitted. Its candidate share is trivially 1.0 when one candidate survives verification, which was nearly every
  answer. Its low Brier score therefore reflects verified precision (0.976 of answers correct), not an informative
  confidence score. **No calibrated confidence is claimed for AgentMamba.**

**Latency.** A separate sequential benchmark ran on stage-4 memory with the Mamba score cache disabled and nothing else
running: 12 threads, Intel CPU, Mamba load 16.3 s (`latency_cold.json`), first 15 test questions.

| system | median end-to-end (s) | median when a candidate was scored (s) | median CPU (s) | peak RSS (MB) |
|---|---|---|---|---|
| Baseline 768 | 5.0 | 7.4 | 12.1 | 3,056 |
| Baseline 2048 | 16.9 | 17.4 | 51.7 | 3,012 |
| AgentMamba | 7.5 | 7.5 | 19.4 | 3,065 |

For AgentMamba, retrieval (SQLite + FAISS + text channel) took a median 0.15 s and verification 0.02 s; the
rest is Mamba scoring of the verified candidates. Baseline medians include 7 of 15 questions where the window held
no candidate and no model call was made.

Latencies recorded inside the main runs are not used: score-cache hits, the concurrent first phase, and host
throttling/sleep all distort them (see `docs/EXPERIMENTS.md`).

**Conclusion for this experiment.** On this synthetic log benchmark, AgentMamba (persistent memory, retrieval and
verification around the same frozen Mamba) is substantially more accurate than Baseline Mamba reading a recent-history
window. The gap is measured: 0.853 vs 0.123 / 0.197, a paired difference of +0.66 to +0.73, at every stage after the
first. AgentMamba also makes far fewer unsupported claims (0.013 vs 0.50–0.60 per question).

The gain comes from two components that matter roughly equally and work together:
- **Retrieval** recovers evidence that left the context window.
- **Verification** turns candidate spans into grounded answers or justified abstentions.

Mamba's own contribution is unchanged across conditions, so this is not evidence that the Mamba network improved.

Four things limit the claim:
- the data is synthetic, and the templates were written alongside the verifier's lexical rules;
- paraphrases expose a large recall loss (two families at 0%);
- one verifier bug produced the only confident errors;
- the baseline's window policy is one reasonable choice among several.

### 14.9 Stage-1 / stage-2 protocol on the user's CP-17 logs (diagnostic)

**Run:** `experiments/results/20261029-135906_protocol_cp17_context_growth_mamba/`. The method is in
`docs/EXPERIMENTS.md`.
- **Setup:** one clean profile; the frozen Mamba with its score cache disabled; the configured learned memory
  gate (every record fit the budget).
- **Shared context:** a 2048-token recent-arrival window, identical for both systems.
- **Scale:** five questions after each of the two stages. This demonstrates the protocol; it is not a
  benchmark.

| stage | question | Baseline Mamba | AgentMamba | AgentMamba sufficiency / retrieval |
|---|---|---|---|---|
| 1 (10 records) | vibration value that triggered the alarm | `ALM-CP17-884` ✗ | `8.4 mm/s` ✓ | sufficient / skipped |
| 1 | work order opened | `WO-CP17-211` ✓ | `WO-CP17-211` ✓ | sufficient / skipped |
| 1 | reading at 08:19 | `8.4 mm/s` ✗ | `4.1 mm/s` ✓ | sufficient / skipped |
| 1 | confirmed root cause (unanswerable) | `ALM-CP17-884` ✗ (unsupported) | abstained ✓ | insufficient / retrieved, then abstained |
| 1 | repair completed (unanswerable) | no candidate (abstained) | abstained ✓ | insufficient / retrieved, then abstained |
| 2 (+300 records) | vibration value | `ALM-HIST-0276` ✗ | `8.4 mm/s` ✓ (line 6) | insufficient / retrieved |
| 2 | work order opened | `WO-HIST-0285` ✗ | `WO-CP17-211` ✓ (line 11) | insufficient / retrieved |
| 2 | reading at 08:19 | `3.2 mm/s` ✗ | `4.1 mm/s` ✓ (line 9) | insufficient / retrieved |
| 2 | confirmed root cause | `ALM-HIST-0276` ✗ (unsupported) | abstained ✓ | insufficient / retrieved, then abstained |
| 2 | repair completed | no candidate | abstained ✓ | insufficient / retrieved, then abstained |

**What happened at each stage.**
- **Stage 1:** the shared window held all 10 incident lines (844 tokens). Even so, the baseline picked the
  right span in only 1 of 3 answerable questions.
- **Stage 2:** the 300 newer arrivals filled the window (24 records, 1,864 tokens) and none of the incident
  lines remained.
- **AgentMamba's retrieval:** 32 vector, 68 relationship and 2 identifier-text candidates; 58 removed by the
  date filter; 10 selected. Every answer cites the original source line.

**Metrics (10 questions, both stages pooled)**

| metric | Baseline Mamba | AgentMamba |
|---|---|---|
| correct-answer rate | 0.30 | 1.00 |
| unsupported claims per answer | 0.875 | 0.00 |
| exact match (answerable) | 0.167 | 1.00 |
| NDCG@10 / Recall@10 | N/A | 0.81 / 1.00 |
| unnecessary retrieval | N/A | 0.00 |
| median latency (warm) | 6.5 s | 4.2 s |

**Latency and cold start.**
- **Time saved by skipping retrieval:** a mean of 5.4 s over the 3 skipped queries. That is the forced-retrieval
  re-run minus the conditional run; most of it is the extra Mamba scoring over the retrieved pool.
- **Mamba load time:** 205.7 s in this run, reported separately (cold start). It was 3–16 s in earlier runs;
  the difference is probably Hugging Face Hub network checks. Setting `HF_HUB_OFFLINE=1` avoids them once the
  model is cached.

**Calibration.** Brier and ECE are N/A: there are fewer than 20 answers with calibrated probabilities, and
AgentMamba has no calibrator.

**Changes made before this run, and their effect elsewhere.** These are line-based parsing of text logs, the
word "trigger" as question vocabulary, answer candidates that may be other identifiers, and visible
retrieval-filter counts (see `docs/EXPERIMENTS.md`). Effect on other results:
- **Steel benchmark:** 0 of 144 validation questions changed.
- **Log benchmark:** its stored 2026-10-27 predictions were produced before these changes, and its test split
  can no longer evaluate the "trigger" change cleanly.

## 15. Error analysis
- **E at 10%:** all 62 errors are questions whose gold evidence was not retained. On value questions with available
  evidence, EM was 1.000. On value questions with lost evidence (54), the system abstained in 100% and answered wrongly in 0%.
  Heuristic and random memories, by contrast, answered 17% of their lost-evidence questions wrongly. A distractor
  record survived and passed the checks, for example the only remaining inspection sheet of a pair.
- **Selective memory hides conflicts (learned 25%).** Six conflicting-record questions were answered confidently, because
  only one of the two disagreeing sheets was kept. The verifier cannot detect the absence of a record it never saw.
- **"Latest before" over incomplete memory (learned 25%).** One four-stage question returned an older inspection because
  the true last one had been dropped.
- **Store-all:** 2 errors, both on F04 rare warnings (Mamba ranking, §14.6). One validation case is a generator ambiguity:
  two genuine warnings on the same ladle.
- **Generator artifacts:** templated phrasing (intent accuracy 1.0) and recognizable warning and RCA wording make the
  gate's and controller's jobs easier than they would be on real text.

## 16. Efficiency analysis
**Query latency, cold cache** (`scripts/benchmark_latency.py`). The Mamba score cache was disabled, and the same 40 seeded
test questions were run per configuration, in two replicates (`*_latency_benchmark` result directories):

| configuration | median s (rep 1 / rep 2) | p95 s (rep 1 / rep 2) | max s (rep 1 / rep 2) |
|---|---|---|---|
| E conditional, learned 10% | 5.42 / 4.31 | 10.53 / 7.19 | 231.6 / 8.1 |
| D always, learned 10% | 4.90 / 4.54 | 8.26 / 15.93 | 10.5 / 22.1 |
| B always, store-all | 4.43 / 4.22 | 183.1 / 7.53 | 1000.5 / 8.4 |
| B conditional, store-all | 4.02 / 4.35 | 7.03 / 7.16 | 7.8 / 419.0 |

The extreme maxima (231 s, 1000 s, 419 s) hit different configurations and questions in each replicate, while the same
questions took 4–9 s elsewhere. We attribute them to the laptop host (throttling or suspension), not to the method, and we
report medians.

Per-query cost breakdown:
- **Mamba candidate scoring dominates:** about 4–5 s per retrieving query, consisting of one prompt prefill of up to 768
  tokens on the CPU reference scan plus token-by-token decoding of each candidate.
- **Retrieval** (FAISS, SQLite and link expansion) takes about 0.05 s, **verification** about 2–3 ms, and the initial context-only
  stage about 0.2 s on average.
- **Conditional vs always:** conditional retrieval skips the final scoring pass for the ~17% of questions answerable from context.
  On this sample the median differences between configurations are within run-to-run noise, so we claim **no**
  measurable latency gain.
- **Memory:** process RSS was about 2.1 GB with Mamba and MiniLM loaded.

**Storage** on the 4,580-record test stream:
- **Store-all:** SQLite 14.1 MB, FAISS 7.1 MB, 4,580 vectors.
- **Learned 10%:** SQLite 2.0 MB, FAISS 0.42 MB, 273 records (9.99% of candidate bytes).

Write path: the gate scored 4,580 records in 0.25 s and the full budgeted ingestion took 2.7 s, given cached features.
Computing the Mamba features for the 17,165 records took 2,872 s and the MiniLM embeddings 106 s, on an
i7-1360P (CPU reference scan path). Latencies recorded in the experiment runs are dominated by the Mamba score cache and
are **not** used for efficiency claims.

## 17. Limitations
See `LIMITATIONS.md`. The main ones:
- synthetic cross-stage data from a single generator, with templated questions;
- a rule-based question grammar (AnswerSpecs);
- no measurable Mamba contribution;
- budgeted memory can hide conflicts;
- small test set;
- no real-plant validation and no safety certification.

## 18. Conclusion and future work

**Baseline Mamba vs AgentMamba (§14.8).**
- **Accuracy:** on the synthetic log benchmark, as history grows from about 20 to about 1,925 records per
  scenario, Baseline Mamba (frozen Mamba ranking candidates from a 768- or 2048-token recent-history window) falls
  from 0.29 / 0.48 accuracy to 0.07. AgentMamba stays at 0.853 at every stage. The decline is explained by measured
  evidence loss from the window plus Mamba's limited span selection; it is not explained by Mamba "forgetting".
- **Where the gain comes from:** retrieval and verification both contribute, and together they account for the
  gain.
- **Limits:**
  - the data is synthetic;
  - paraphrased questions expose a large lexical-recall weakness (two families at 0%);
  - one verifier bug, found on test and not tuned away, caused the only confident errors;
  - AgentMamba's confidence could not be calibrated (no wrong answers on dev).
- **Next steps:** semantic rather than lexical coverage, and the untimed-passage fix, evaluated on new data.
On a controlled cross-stage benchmark, a learned write gate retained far more useful evidence than heuristic or
random selection under the same byte budget, and that translated into significantly higher accuracy. Documented-identifier
relationship expansion was needed for cross-stage retrieval. Verification was the component that kept answers grounded
and prevented causal overclaiming. Conditional retrieval removed unnecessary retrieval without any loss. The binding
constraint is retention of individually unremarkable *structural* records: heat→ladle assignments, rolling records and
paired inspection sheets. Future work:
1. Retention constraints by record class, e.g. keep every linking record, which is cheap in bytes.
2. Explicit "memory incomplete" signals at answer time.
3. Evaluation on real multi-stage plant records under a data agreement.
4. Free-text question understanding.
5. An instruction-tuned or fine-tuned state-space model, to test whether a stronger answerer changes §14.6.
