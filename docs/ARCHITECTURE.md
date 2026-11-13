# Architecture

AgentMamba is made of eight separate components. None of them is a renamed version of another.

| # | Component | Module | What it is |
|---|---|---|---|
| 1 | Mamba backbone | `models/mamba_backbone.py` | frozen `state-spaces/mamba-130m-hf` (`MambaForCausalLM`, 129.1M params, 24 layers, d=768, state 16), CPU reference scan path |
| 2 | Learned memory-writing gate | `gate/` | logistic regression (or small MLP ablation) over documented write-time features; trained on train-split labels |
| 3 | Episodic store | `memory/sqlite_store.py` | SQLite, the authoritative store: original text, structured fields, observations with recorded units, provenance, versions, gate decisions, audit |
| 4 | Vector index | `memory/vector_index.py` | FAISS `IndexIDMap2(IndexFlatIP)` on L2-normalized all-MiniLM-L6-v2 vectors (cosine); stable integer IDs = SQLite `vector_id` |
| 5 | Relationship store | `relationships` table + `memory/relationship_store.py` | documented links derived from structured identifiers and recognized raw alarm/trip IDs (`data/event_linker.py`) |
| 6 | Answerability controller | `retrieval/answerability.py`, `retrieval/controller.py` | explicit assessment features → logistic regression P(initial context sufficient) → validation-tuned threshold |
| 7 | Evidence verifier | `verification/` | identity linkage, temporal constraints, supersession, conflicts, process order, causal discipline |
| 8 | Answer generator | `models/answerer.py`, `pipeline/answering.py` | typed candidate spans ranked by Mamba likelihood; templated, provenance-carrying answer text |

## Ingestion and memory writing (`pipeline/ingestion.py`)

```
source file ──► loader (CSV/JSON/JSONL/TXT/UCI) ──► Event (stable id, source text/raw kept, units as recorded, tz as given)
          ──► dedupe / in-place update by content hash
          ──► features: MiniLM embedding (+ Mamba surprisal & pooled state if the gate uses them)
          ──► streaming novelty vs. last 256 records considered
          ──► gate score (trained artifact; missing artifact = error, never a heuristic)
          ──► BudgetedSelector: admit, evict lowest score while bytes > budget  ──► WRITE / SKIP / EVICT logged
          ──► SQLite (record + documented relationships, one transaction) ──► FAISS add ──► index saved
```

- **Budget unit:** UTF-8 bytes of the text plus the serialized observations (`Event.size_bytes`). Record count and
  file sizes are reported as well.
- **Streaming rule:** an evicted record never comes back, and its relationships are removed with it (a link is retained only if
  the record documenting it is retained).
- **Out-of-distribution records:** public UCI records go through the same pipeline under an *explicitly named*
  `store_all(public)` policy. Applying a gate trained on synthetic event types to them would not be justified.
- **Operational logs:** common ISO, day-first numeric and month-name timestamps are normalized without discarding the
  original timestamp or line. Recognized severity levels, event fields, observations, and raw alarm/trip identifiers
  remain available for evidence extraction and documented links; unknown fields stay searchable in the source text.

## Question answering (`pipeline/agent.py`), default mode *conditional*

```
question ──► QueryProcessor: ids (regex), dates, intent (TF-IDF+LR trained on train questions) → AnswerSpec
        ──► INITIAL: candidates from the permitted initial context only ──► Mamba ranks them
        ──► verifier on the initial context only (no DB) ──► assessment features
        ──► controller: ABSTAIN_OR_CLARIFY | RETURN_INITIAL | RETRIEVE
RETRIEVE ──► FAISS top-k·4 (SQLite metadata filters as an IDSelector) ──► resolve ids in SQLite
        ──► documented-link expansion from question/context identifiers (BFS, depth ≤ 3, ≤ 60 nodes, no traversal through hubs)
        ──► answer-type + temporal metadata filters ──► newer versions of superseded hits ──► dedupe ──► rerank ──► top-k + linking records
        ──► Mamba ranks candidates from (context + retrieved) ──► verifier (memory available) ──► answer / abstain
```

- **Initial-stage isolation:** in the initial stage `verify(..., store=None)` runs and the graph is built from the
  context records only. No FAISS or SQLite call happens before the controller decides.
- **Modes:** `conditional` (default), `always` (always retrieve; baseline) and `never` (no external retrieval;
  baseline). An ambiguous question, or one without the identifier its intent needs, gets a clarification request in every mode.
- **Evidence guard:** in conditional mode, complete `supported` verification returns the initial answer without
  retrieval; insufficient, partial, or conflicting evidence always triggers retrieval. The trained controller remains
  available for unresolved/unverified cases, but cannot override a verified evidence result. This avoids applying a
  benchmark-trained sufficiency prediction as a substitute for evidence on uploaded logs. Explicit `always`/`never`
  modes remain retrieval ablations.
- **Decision trace:** each response carries the parse, initial candidates and scores, initial status, assessment
  features, the controller decision with its probability, threshold and reasons, retrieval items with similarity, path and
  explanation, the relationships traversed, per-candidate verification checks, evidence with provenance, timings per stage,
  and the config, gate and controller versions.

## Mamba's role, stated precisely
- **Answer extraction:** Mamba scores each candidate string (taken from the records, typed by the intent) by mean
  token log-probability after a prompt that contains the records and the question. The prompt is prefilled
  once, and each candidate is decoded token by token from a copy of the recurrent state. This matches a full
  forward pass to within 3e-5 per token, and a model test checks it.
- **Optional gate features:** mean and max token NLL plus mean-pooled final hidden state (PCA 16), used in gate H and
  dropped in gate G.
- **Interactive raw-generation baseline:** the comparison page can run greedy Mamba generation directly from the
  shared context, without candidate extraction. Uploaded records are included in the prompt; they do not update or
  fine-tune the frozen model weights. The base checkpoint is not instruction-tuned, and the probe
  (`artifacts/models/mamba_probe.json`) showed that its free generation does not reliably answer record questions.
- **Not a memory system.** Mamba's selective state-space recurrence is unrelated to the external write gate. The gate
  is a separate classifier that sees Mamba only through the optional features.

The staged context-growth benchmark and interactive dashboard default to the candidate-span ranking baseline so its
answers can be scored against extractive answer types. Compound questions are split into individual requirements and
the frozen Mamba ranks candidate spans from the shared context independently for each part; this baseline still has no
retrieval or evidence verification. Raw greedy generation remains an optional diagnostic for inspecting direct
continuation behavior, but the base checkpoint is not instruction-tuned and may produce incomplete answers.

## Verification rules (`verification/evidence_checker.py`)
1. **Identity.** The candidate's record must name the anchor identifier, or reach it through documented entity
   links. Shared groupings (ladle, cast sequence, hot-metal batch, asset, document) can be reached but are never
   traversed *through*. A record naming a *different* heat, slab, coil or work order than the chain is rejected. Among
   linked candidates, the closest link distance wins.
2. **Temporal.** `on_date`, `before_date` and `latest_before_link`. The last uses the time of the record documenting the
   last link (e.g. the heat→ladle assignment) and keeps only the latest record before it.
3. **Currency.** Superseded records are dropped and the highest version per document wins. Without memory access,
   currency is *unverified* and the status is `partial`.
4. **Conflict.** A single-valued slot that receives distinct values from distinct records gets status `conflict`, and
   the system abstains and lists both.
5. **Causal discipline.** A causal answer is allowed only from a record whose own `rca_status` is `confirmed`
   (`documented_by_source`). Otherwise the best linked association is returned as `hypothesis_only`, with
   explicit wording, and the inconclusive RCA records are shown. The system never claims formal causal inference.
6. **Process order.** Records on the same material chain whose timestamps contradict the nominal process order
   downgrade the result to `partial`.
7. **Citations.** Every cited record ID must resolve to a stored record or a supplied context record (checked
   per response).

## Document-grounded answers (`verification/document_evidence.py`)
Uploaded documents become `document_chunk` records whose structured identifier fields are null, so the
typed path above cannot use them. Such questions follow these rules:
- **Identifiers.** Besides the benchmark ID formats, any upper-case hyphenated token containing a digit is
  extracted (e.g. `QA-LADLE-417`). Its kind comes from a keyword inside it (`LADLE`, `HEAT`, `WO`, …) or the preceding
  word ("ladle X"), and is otherwise `identifier`. An explicitly supplied identifier is never asked for again.
- **When to clarify.** A typed question that names no identifier gets one from the initial context only if that context
  contains exactly one identifier of the required kind. Otherwise it asks for clarification.
- **Untrusted intents.** If the intent classifier's confidence is below 0.5 (every benchmark question is ≥ 0.783), the typed
  AnswerSpec is not used and only document evidence can answer.
- **Retrieval.** It also fetches `document_chunk` records whose *text* names a question identifier, checked as an exact
  whole token. Raw alarm/trip identifiers are also retained as documented relationship keys for related event lookup.
- **Verification.** A passage supports an answer only if it names the identifier, satisfies the date (stated in the
  passage, or in a chunk of the same document that also names the identifier), and covers the relevant question
  clause's content using conservative terminology equivalences. Coordinated `when`/`what`/`which` clauses are checked
  separately and combined only when each has supporting evidence. Event timestamps are taken from the record that
  supports that clause, not from a nearby unrelated event. Question sentences inside documents don't count. Passages
  are ranked by MiniLM similarity. The answer is a quotation with provenance. Passages from several documents give
  status `partial`, because their agreement is not checked.
- **Precedence.** Document evidence is tried only when typed verification is `insufficient`. It never overrides a typed
  supported, partial or conflict result.

## Operational logs and untyped records
- **Loading** (`data/document_loader.py`, `load_any`). CSV, JSON and JSONL files in AgentMamba event layout load as
  events. Any other CSV, JSON or JSONL file loads as an operational log (`load_log_records`):
  - Common ISO, day-first numeric and month-name timestamps are normalized. Timestamp, equipment/asset, heat, ladle,
    coil, work-order and severity columns are recognized through normalized names and aliases.
  - Long-format sensor rows (`signal`/`tag` + `value` [+ `unit`]) become an Observation with the recorded unit.
  - The original row and source text are retained; fields not mapped to the structured record remain searchable.
  - Recognized raw alarm/trip identifiers are included in relationship lookup without requiring every file to use the
    same column capitalization or spacing.
  - `event_type` comes from a type column, or from the file name without date parts (`alarm_log_20260314.jsonl` →
    `alarm_log`).
  - Record IDs are derived from the file name and row content, so re-delivered files are counted as duplicates.
  - TXT/MD documents are chunked by paragraph (at most 800 characters).
- **Untyped answering** (`verification/document_evidence.py`). This covers record types that no `AnswerSpec` models
  (document chunks, log rows) and questions whose identifiers do not match a typed intent's anchor kind.
  - `text_candidates`: candidates from structured field names/values and document text. For compound questions,
    candidates are extracted per clause and combined, so a time request does not suppress event/action candidates for
    other clauses. Context-setting fragments before the first question are not treated as an extra answer requirement.
    A specific source field (for example an alarm description, finding, or action) takes precedence over a generic
    `event_type` label when both match. Common timestamp and maintenance-action fields have explicit handling.
  - The candidates are ranked by the same answerer (Mamba).
  - `verify_text` checks:
    - identity (identifier in structured fields or as a whole token in text);
    - date (timestamp or stated date, or the same document);
    - clock time from the event record that supports the clause (not another event's timestamp);
    - coverage of the content requirements of each clause, rather than requiring one log row to answer the whole
      question; each record check carries its question part and verification status, and conflicts in any part remain
      visible instead of being downgraded to generic insufficiency; equivalent wording is recognized conservatively;
    - hedged or negated statements for causal and clearance questions ("under investigation", "pending",
      "not cleared" → abstain);
    - same-time conflicts between sources (status `conflict`).

    Multiple entries at different times give status `partial`; listing ("timeline") questions return every
    matching record in time order.
- **Retrieval channels** (`retrieval.channels`): `vector` (FAISS), `relationship` (documented-link expansion), `text`
  (identifier text match, SQL date filter). For untyped questions, a stated date or time filters every channel. With
  the vector channel off, items are ranked by lexical overlap.

## Baseline Mamba (`pipeline/baseline.py`) and the side-by-side comparison
- **Context.** The baseline receives the selected initial-context records, then the most recent arrivals as whole
  records, within a declared Mamba-token budget (`build_window`). It is question-independent: no similarity search,
  identifier lookup, relationship expansion or memory access. Mamba-130m has no positional context limit; the budget
  is a CPU-cost policy, and the experiment reports whether the evidence was inside it.
- **Answer.** Same candidate extractors and the same Mamba ranking as AgentMamba; the top candidate is returned.
- **No verification, no abstention.** With no candidate it gives no answer (`no_candidate`).
- **Post-hoc audit.** The verifier is run on the baseline's answer for display only; it never changes that answer.
- **Shared context in the comparison.** `AgentService.compare` / `POST /compare` and the dashboard's "Baseline Mamba vs
  AgentMamba" tab give both systems the same question and the same initial context. AgentMamba may then retrieve.
  - Model-loading failures are reported per panel; no other model is substituted.
  - Optional evaluation cases run a benchmark question against its scenario's memory snapshot at a chosen stage.
    Gold is read only after both outputs exist.
- **Confidence fields** (both systems, never merged):
  - `candidate_share`: Mamba softmax share among distinct candidate values; uncalibrated; flagged when trivially 1.0.
  - `evidence_support`: the verification outcome.
  - `retrieval_max_similarity`.
  - `calibrated`: a dev-split logistic calibrator, only if its artifact exists.

  Each response also carries `model_calls` (stage, candidates, cache hit, prompt tokens), `resources` (CPU seconds,
  RSS) and per-stage timings.

## Persistence and recovery
- SQLite is the source of truth. FAISS is validated against the active `vector_id` set on open, and a mismatch or a
  missing file triggers a rebuild that re-embeds the SQLite text (or an actionable error when
  `auto_recover=False`). The index is serialized with a sidecar metadata file (model id, dimension, count, ID
  checksum), and a model or dimension mismatch refuses to load.
- Lifecycle (`memory/lifecycle.py`): supersede, in-place update (re-embeds, rebuilds the record's links, version history
  kept), delete (removes the vector and the links the record documents), and backup and restore of both artifacts.

## Interfaces
- **FastAPI** (`api/main.py`): `/health`, `/status`, `/events/ingest`, `/documents/ingest` (documents and logs; type and
  size checks, client path components stripped, optional explicit `policy=store_all`), `/compare` (Baseline Mamba vs
  AgentMamba), `/events`, `/events/{id}`, `/memory/stats`, `/memory/rebuild-index`, `/query`,
  `/experiments`, `/experiments/{id}` (path-traversal safe).
- **Streamlit** (`app/streamlit_app.py`): status, plant overview with synthetic warnings, event explorer, ingestion with
  WRITE/SKIP decisions, investigation playground with the full trace, and experiments read from result files.
