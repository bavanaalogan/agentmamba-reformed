# Limitations

**Public-data coverage gaps.** No openly licensed dataset I found links several steel production stages through
shared identifiers. The two real datasets used (UCI #851 energy, UCI #198 plate faults) are single-source and have no
heat, ladle or coil identifiers. Every cross-stage result is therefore measured on synthetic data.

**Synthetic cross-stage relationships.** The benchmark's relationships, incidents, warnings and values are generated.
Results describe behaviour on *this generator's* regularities and do not show real-plant performance. In particular:
- Question phrasings are templated, so intent classification is near-trivial and does not reflect free-text robustness.
- Useful records have recognizable surface forms (three warning phrases, a fixed RCA wording). A learned gate can
  exploit this, and real plant text is noisier and more varied.
- Individually unremarkable linking records (heat→ladle assignments) are only "useful" when a question happens to
  target that heat. No write-time feature can predict this, which caps any gate's performance on chain questions.

**Small Mamba model.** `mamba-130m-hf` is a base language model, not instruction-tuned. Its free generation did not
answer the probe question, so it is used only to rank candidate spans that the system extracted itself. Its
contribution to accuracy is measured separately (ablation `E_first_candidate`), and in the full system most of
the work is done by typed candidate extraction and the verifier.

**Rule-based question grammar.** `AnswerSpec`s map each intent to the answer field, record types, anchor kind and
temporal constraint. This is an explicit, inspectable operationalization, and it does not generalize to questions
outside the declared intents. Uploaded-log questions with coordinated `when`/`what`/`which` clauses are now checked
per clause and combined with citations; broader natural-language question decomposition is still unsupported.

**Document-grounded answers are lexical and quoted.** For uploaded documents, support requires the identifier,
the date and the content terms for an individual clause to be present in a record/passage (with field aliases for
common structured-log action and timestamp questions). Synonyms outside those aliases and paraphrased questions can
still abstain even when the answer is present. Answers are quoted sentences or recorded field values, not newly
generated facts. Contradictions between free-text documents are not detected; several supporting documents are only
flagged `partial`. Natural-language dates are parsed in a few common formats (ISO, "20 October 2026", "October 20,
2026", day-first numeric).

**The log benchmark and the verifier share an author.** I wrote the log benchmark's question templates and record
formats while I was writing the text verifier's lexical rules (content-term coverage, hedge lexicon, nearest-time rule).
High AgentMamba scores on the template families partly reflect that fit. The four paraphrase families were written to
measure this, and their results are reported separately. There is no human-written or real-plant question set.

**Observed on the log test split: lexical coverage fails on paraphrases, and one timing bug.** "What work was
carried out on X" and "Which alarm did X trigger" were abstained on in 100% of cases, because the records never say
"work", "carried" or "trigger". Separately, a timestamp-less shift-note passage that mentioned "vibration" was
accepted for a question about a reading at 09:30 (4 rows, one scenario). That is because the nearest-time rule keeps
untimed passages. Neither was fixed after being found on test; see REPORT §14.8.

**Log-line parsing is format-based.** A text file is treated as a log when at least 80% of its non-comment
lines start with a date-time. Fields are split on `|` and `KEY=value`. Logs without timestamps, multi-line
entries such as stack traces, and day-first or other unlisted timestamp formats are not parsed as records.

**The CP-17 protocol is one scenario with five questions.** It demonstrates the stage-1/stage-2 behaviour; it
does not measure it statistically. The questions were written by the assistant, because the user's protocol
file was not available.

**The base Mamba is not instruction-tuned and is a weak free-form answerer.** The interactive comparison defaults to
candidate-span ranking from the shared context, independently for each part of a compound question. It does not retrieve
or verify evidence and may rank an unsupported candidate. Raw greedy generation remains available as a diagnostic;
it can produce incomplete answers and has no candidate-share confidence or source verification. Uploads do not train
or update Mamba weights.

**Context-window loss is a policy, not Mamba forgetting.** The comparison baseline sees the shared recent-arrival
window (commonly 768 or 2048 tokens); Mamba-130m has no hard context limit. A longer window (or a different,
question-dependent selection) changes what the baseline can see and its CPU cost (about 7 s per question at 768 tokens
and about 18 s at 2048, measured). The online upload workflow does not fine-tune or otherwise teach Mamba the records.

**Log ingestion heuristics.** Column roles are recognized by name only. Day-first or ambiguous timestamp formats
are parsed only in the declared formats (others become `timestamp=None`). Units are kept as recorded and never
converted. Rows with identical content in one file collapse into one record.

**Calibration is small-sample and in-distribution.** Calibrators are fitted on about 45 dev questions × 4 stages of
the same synthetic generator. They are not validated on other data, and the dashboard says so wherever it shows them.

**Latency includes contention.** The dev and test context-growth runs ran concurrently (6 torch threads each) on the
same machine, with the dashboard open.

**CPU constraints.** There is no optimized selective-scan kernel on CPU. Mamba feature extraction took about an hour for 17k
records, and a retrieving query takes seconds. Nothing was fine-tuned.

**Memory-gate evidence loss.** Under tight budgets the gate drops evidence that later questions need. The system then
abstains (or, if a distractor survives verification, can answer wrongly). It cannot recover discarded records. The
"evidence lost" analysis in every run summary quantifies this.

**Selective memory can hide conflicts and corrupt "latest" selection (observed).** In the 25%-budget learned run,
6 conflicting-record questions were answered confidently, because the gate had kept only one of the two disagreeing
inspection sheets, and the system cannot know that a conflicting record ever existed. One multi-stage question took an
*older* surviving inspection as "the last inspection before the heat" because the true last one had been dropped.
Verification checks only the evidence that survived, so it cannot detect evidence that was never retained. A deployed
system would need retention guarantees for record classes whose absence changes the answer (e.g. all inspections of
an asset), or explicit "memory is incomplete for this asset/time window" signals.

**Gate behaviour on unseen record types.** The gate gave two uploaded `document_chunk` records (a type absent
from training) scores of 0.91 and 0.98. Its scores on out-of-distribution record types are arbitrary, so public
data is ingested under an explicit store-all policy instead.

**Retrieval failures.** Dense retrieval over short, template-like records is weak at distinguishing identifiers. Most
cross-stage recall comes from documented-link expansion, which depends on the linking records having been retained.

**Incorrect relationship links.** Links are created only from co-recorded identifiers. A wrong identifier in a source
record would create a wrong documented link, and nothing here detects data-entry errors beyond conflicts and
process-order checks.

**Temporal ambiguity.** Timezones are recorded but never converted, and the UCI #851 interval convention is unknown.
"Before the incident date" is evaluated at day granularity.

**Causal overclaiming.** The system only *reports* a cause when a source record itself states a confirmed cause. It
performs no causal inference. A confirmed RCA record could still be wrong, and the system would cite it as documented.

**Dataset leakage.** Splits are by independent campaign with disjoint ID namespaces, and access to test annotations is
enforced in code. All campaigns still come from the same generator, so train and test share templates and
distributions. This is in-distribution generalization, not transfer.

**Statistics.** The test set has 192 synthetic questions, plus 61 public ones. CIs are bootstrap over questions, and p-values
(exact McNemar) are uncorrected for multiple comparisons.

**No real-world validation and no safety certification.** The system has not been evaluated with plant engineers,
real plant data, or under any safety or quality standard, and it must not be used to authorize production or
safety decisions.
