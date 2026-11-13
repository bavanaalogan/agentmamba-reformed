# Research methodology

## Research questions (hypotheses, not results)
- **RQ1 Memory retention:** does a learned write gate keep more *useful* evidence under a byte budget than
  store-all, random or heuristic selection?
- **RQ2 Cross-stage investigation:** does persistent memory with documented-link expansion recover evidence from
  other stages when the initial context is insufficient?
- **RQ3 Conditional retrieval:** can the controller skip retrieval when the initial context supports the answer,
  and trigger it when it does not?
- **RQ4 Verification:** do identity, temporal, version and causal checks reduce unsupported answers, wrong-asset
  attributions and unjustified causal claims?
- **RQ5 Trade-offs:** how do budget, retrieval policy, verification and the answerer affect quality, latency and
  storage?

## Related work (from the literature search; descriptions are taken from search results and abstracts and were not
## independently verified in full text)
- **Adaptive retrieval:** Self-RAG (Asai et al.) trains reflection tokens such as Retrieve and IsSupported. FLARE (Jiang et al.)
  retrieves when generation confidence is low. Adaptive-RAG (Jeong et al.) routes queries by a predicted complexity class.
  AgentMamba's controller sits closest to Adaptive-RAG (a classifier decides before the final answer), but its
  features are evidence-support checks on the *initial context*, not question complexity or token
  confidence.
- **Learned memory writing for LLM agents (2025–2026 preprints):** "Learn to Memorize" (arXiv 2508.16629), SAGE
  novelty gating (arXiv 2605.30711), ConsistencyGate admission control (arXiv 2607.22962), gated-memory routing in
  multi-agent systems (arXiv 2609.00237), and AdaMem (arXiv 2606.21144). These train or design *write* policies for
  conversational or agent memory. The ideas of a learned write gate, novelty and redundancy features, and budgeted
  admission are therefore **not new**.
- **Industrial RAG:** a steel-plant maintenance copilot with RAG over CMMS data and manuals (Danieli Automation / ABS, AISTech
  listing), a maintenance work-order retriever (SEASONED, Procedia CIRP 2025), and graph-RAG for maintenance (Fraunhofer, 2025).
- **Positioning.** What this project combines is a budgeted, *learned* write gate over plant-wide *event records*,
  documented-identifier relationship traversal across production stages, an initial-context-only evidence check
  before retrieval, and causal-claim discipline, all evaluated against a controlled cross-stage benchmark.
  I did not establish that this combination is unprecedented. The search was limited, and several cited items are
  preprints or conference listings whose full text I didn't review. No novelty claim is made.

## Data
- Category B synthetic benchmark (all cross-stage experiments) and Category A UCI #851 records (single-source
  sanity experiment). See `DATASET_CARD.md` and `DATA_AUDIT.md`.
- Splits are by campaign (synthetic) or by month (public). Test annotations are loaded only with `purpose="evaluation"`.

## Gate training labels (`gate/label_builder.py`)
- A record is labelled 1 if it is required evidence for at least one question of the same split. Fitting uses train
  only; regularization C and the F1 threshold are selected on val.
- **Limitation:** this is a proxy for future utility. Records that would answer questions nobody asked are
  labelled 0. A record's label is not observable at write time; the gate learns P(useful | write-time features).

## Gate features (all available at write time; no labels, questions or future records)
- Event type and stage (one-hot), recorded severity, length and structure counts, the number of structured identifiers
  and observations, version and supersession flags.
- Token rarity: IDF fitted on training text, with identifiers masked so a heat number is not a feature.
- Streaming novelty and redundancy against the last 256 records considered.
- PCA-16 of the MiniLM embedding.
- *Optional:* Mamba mean and max token NLL plus PCA-16 of the pooled final hidden state.

## Baselines and ablations (`config/experiments.yaml`)

| ID | Memory policy | Retrieval mode | Verification | Purpose |
|---|---|---|---|---|
| A | learned (unused) | never | on | initial Mamba track only |
| B | store-all | always | on | no selective writing |
| C | heuristic (severity + category priority) | always | on | non-learned selection |
| D | learned (Mamba features) | always | on | isolates retention |
| E = H | learned (Mamba features) | conditional | on | full AgentMamba |
| F | random (seeded) | always | on | selection baseline |
| G | learned (no Mamba features) | conditional | on | feature ablation |
| E_no_verification, B_storeall_no_verify | | | off | RQ4 |
| B_storeall_vector_only | store-all | always | on, but no relationship expansion | conventional vector RAG |
| E_first_candidate | as E, but chronological first candidate instead of Mamba ranking | | | what the Mamba ranking contributes |
| E_mlp_gate | MLP gate | conditional | on | gate model ablation |
| sweep | {learned, learned-no-Mamba, heuristic, random} × budgets {1, 5, 10, 25, 50}% + store-all | conditional | on | RQ1 / RQ5 |

The answerer, embedding model, dataset and evaluation are identical across rows unless the row names the change.
The primary budget of 10% was fixed before any test run, and no configuration was selected on test results.

## Metrics
See `docs/EXPERIMENTS.md` (definitions) and `src/agentmamba/evaluation/`. Uncertainty is reported as a 95% bootstrap CI
(2,000 resamples over questions) for accuracy and EM. Paired method differences use the exact McNemar test on per-question
correctness. No multiple-comparison correction is applied, so treat p-values as descriptive.

## Leakage controls (all tested)
1. No entity or event shared across splits (`validate_dataset`).
2. `load_annotations` refuses test annotations for training or tuning purposes.
3. Live modules (`pipeline`, `retrieval`, `verification`, `models`, `memory`, `api`) never import the annotation loader.
4. The initial stage cannot reach SQLite or FAISS (no store handle is passed).
5. IDF and PCA statistics are fitted on train events only, without labels.
6. Test labels are used only in `run_experiment.py` to *score* retention and gate AUC after the fact.
