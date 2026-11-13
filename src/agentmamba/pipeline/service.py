"""Long-lived application service used by the FastAPI app, the Streamlit dashboard and the
CLI scripts. Components load lazily; ``status()`` reports what is actually available."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from agentmamba.config import PROJECT_ROOT, get_dotted, resolve_path
from agentmamba.data.document_loader import load_any
from agentmamba.gate.features import FeatureStore
from agentmamba.gate.predictor import GatePredictor, build_scorer
from agentmamba.gate.train import GateArtifact
from agentmamba.memory.episodic_memory import EpisodicMemory
from agentmamba.memory.schema import Event
from agentmamba.pipeline.agent import AgentMamba, QueryRequest
from agentmamba.pipeline.builder import build_agent, build_answerer, gate_path
from agentmamba.pipeline.ingestion import FeatureProvider, IngestionPipeline
from agentmamba.models.answerer import MambaAnswerer
from agentmamba.pipeline.baseline import BaselineMamba


class ServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


class AgentService:
    def __init__(self, cfg: dict, storage_dir: str | Path | None = None, embedder=None, answerer=None,
                 backbone=None, token_counter=None):
        self.cfg = cfg
        self.storage_dir = Path(storage_dir) if storage_dir else resolve_path(cfg, "paths.storage_dir") / "app"
        self._embedder = embedder
        self._answerer = answerer
        self._backbone = backbone
        self._memory = None
        self._agent = None
        self._pipeline = None
        self._lock = threading.RLock()
        self.errors: dict[str, str] = {}
        self._token_counter = token_counter
        self._pipelines: dict[str, IngestionPipeline] = {}
        self._case_agents: dict[tuple, tuple] = {}

    # ---------------------------------------------------------------- components
    @property
    def embedder(self):
        if self._embedder is None:
            from agentmamba.models.embedding_model import SentenceEmbedder

            self._embedder = SentenceEmbedder(self.cfg["models"]["embedding_id"])
        return self._embedder

    @property
    def backbone(self):
        if self._backbone is None:
            from agentmamba.models.mamba_backbone import MambaBackbone

            m = self.cfg["models"]
            self._backbone = MambaBackbone(m["mamba_id"], m["mamba_threads"], m["mamba_max_tokens"],
                                           resolve_path(self.cfg, "paths.artifacts_dir") / "models" / "mamba_score_cache.sqlite3")
        return self._backbone

    @property
    def memory(self) -> EpisodicMemory:
        with self._lock:
            if self._memory is None:
                self._memory = EpisodicMemory(self.storage_dir, self.embedder)
            return self._memory

    @property
    def agent(self) -> AgentMamba:
        with self._lock:
            if self._agent is None:
                try:
                    answerer = self._answerer or build_answerer(self.cfg, self.backbone)
                    self._agent = build_agent(self.cfg, self.memory, answerer)
                except FileNotFoundError as exc:
                    raise ServiceError(str(exc)) from exc
            return self._agent

    def gate_artifact(self) -> GateArtifact | None:
        p = gate_path(self.cfg)
        return GateArtifact.load(p) if p.exists() else None

    def pipeline(self, policy: str | None = None) -> IngestionPipeline:
        """Ingestion pipeline for the configured policy, or an explicitly requested ``store_all``."""
        if policy is not None and policy != self.cfg["memory"]["policy"]:
            if policy != "store_all":
                raise ServiceError(f"policy override {policy!r} not allowed; only 'store_all' may override", 422)
            with self._lock:
                if policy not in self._pipelines:
                    fp = FeatureProvider(self.embedder, FeatureStore(resolve_path(self.cfg, "paths.feature_cache")))
                    self._pipelines[policy] = IngestionPipeline(self.memory, fp, build_scorer("store_all"), "store_all",
                                                                False, self.cfg["gate"]["novelty_window"],
                                                                run_tag="app:store_all")
                return self._pipelines[policy]
        with self._lock:
            if self._pipeline is None:
                policy = self.cfg["memory"]["policy"]
                predictor = None
                if policy == "learned":
                    art = self.gate_artifact()
                    if art is None:
                        raise ServiceError(f"memory.policy=learned but no trained gate at {gate_path(self.cfg)}; "
                                           f"run scripts/train_gate.py (no heuristic fallback is used)")
                    predictor = GatePredictor(art)
                scorer = build_scorer(policy, predictor=predictor, seed=self.cfg["seed"])
                fs = FeatureStore(resolve_path(self.cfg, "paths.feature_cache"))
                needs_mamba = bool(predictor and predictor.needs_mamba)
                fp = FeatureProvider(self.embedder, fs, self.backbone if needs_mamba else None,
                                     self.cfg["models"]["mamba_feature_max_tokens"])
                self._pipeline = IngestionPipeline(self.memory, fp, scorer, policy, needs_mamba,
                                                   self.cfg["gate"]["novelty_window"], run_tag="app")
            return self._pipeline

    # ---------------------------------------------------------------- operations
    def ingest(self, events: list[Event], budget_bytes: int | None = None, policy: str | None = None) -> dict:
        budget = budget_bytes if budget_bytes is not None else get_dotted(self.cfg, "memory.app_budget_bytes")
        with self._lock:
            rep = self.pipeline(policy).ingest(events, None if policy == "store_all" else budget).to_dict()
        rep["received"] = len(events)
        rep["chunks_created"] = sum(e.event_type == "document_chunk" for e in events)
        rep["records_parsed"] = len(events) - rep["chunks_created"]
        rep["already_stored_or_repeated"] = rep["duplicates"]
        return rep

    def ingest_file(self, path: str | Path, stage: str = "MAINTENANCE", source_type: str = "authorized",
                    budget_bytes: int | None = None, policy: str | None = None) -> dict:
        """Parse one uploaded file (documents are chunked, logs normalized) and ingest it. The
        report keeps parsed rows, chunks, records written / skipped / evicted and duplicates apart."""
        events = load_any(path, stage=stage, source_type=source_type)
        rep = self.ingest(events, budget_bytes, policy)
        rep["file"] = Path(path).name
        rep["normalized_preview"] = [e.to_dict() for e in events[:100]]
        rep["preview_rows_shown"] = min(100, len(events))
        return rep

    def load_mamba(self) -> dict:
        try:
            self.backbone.load()
        except Exception as exc:  # noqa: BLE001 - reported to the caller, never hidden
            raise ServiceError(f"Mamba failed to load: {exc}") from exc
        return {"loaded": True, "load_seconds": round(self.backbone.load_seconds or 0.0, 2)}

    def query(self, req: QueryRequest) -> dict:
        return self.agent.query(req)

    # ---------------------------------------------------------------- baseline vs enhanced
    def count_tokens(self, text: str) -> int:
        return self._token_counter(text) if self._token_counter else self.backbone.count_tokens(text)

    def baseline(self, token_budget: int, qp=None) -> BaselineMamba:
        ans = self._answerer or build_answerer(self.cfg, self.backbone)
        if isinstance(ans, MambaAnswerer):
            ans = MambaAnswerer(ans.backbone, max_prompt_records=None, max_tokens=token_budget)
        return BaselineMamba(qp or self.agent.qp, ans, self.count_tokens, token_budget,
                             time_window_min=float(get_dotted(self.cfg, "verification.time_window_minutes", 30)))

    def calibrators(self) -> dict:
        d = resolve_path(self.cfg, "paths.artifacts_dir") / "calibration" / "logbench"
        out = {}
        for name, key in (("baseline", "baseline_mamba_768"), ("enhanced", "agentmamba")):
            p = d / f"{key}.json"
            out[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        return out

    def compare(self, question: str, initial_ids: list[str] | None = None, window_tokens: int = 768,
                mode: str = "conditional", verification: bool = True, case: dict | None = None,
                baseline_mode: str = "candidate_ranking", source: dict | None = None,
                reference: dict | None = None) -> dict:
        """Run Baseline Mamba and AgentMamba on the same question and the same initial context.

        Shared initial context = the explicitly selected records, then (if ``window_tokens`` > 0) the
        most recent arrivals in memory, as whole records, within ``window_tokens`` Mamba tokens.
        The baseline answers from that context only; AgentMamba starts from it and may retrieve.
        ``case`` = {"split", "question_id", "stage", "run"}: evaluate a benchmark question against the
        snapshot of its scenario's memory at that stage; gold is read only after both outputs exist."""
        from agentmamba.evaluation.comparison import apply_calibrator, output_features, score_output
        from agentmamba.pipeline.baseline import ContextWindow, build_window

        if baseline_mode not in {"candidate_ranking", "raw_generation"}:
            raise ValueError("baseline_mode must be 'candidate_ranking' or 'raw_generation'")
        ladle_ann = None
        if source:     # a stored ladle dataset snapshot: {"dataset", "scenario", "stage"}
            memory, agent, history = self._ladle_memory(source["dataset"], source["scenario"], int(source["stage"]))
            ladle_ann = self._ladle_gold(source["dataset"], source["scenario"], question)
        elif case and not case.get("protocol"):
            memory, agent, history = self._case_memory(case)
        else:          # app memory, or a protocol question asked against the CURRENT profile
            memory, agent = self.memory, self.agent
            history = memory.store.recent_events(600) if window_tokens else []
        initial = []
        for i in initial_ids or []:
            ev = memory.store.get_event(i)
            if ev is None:
                raise ServiceError(f"initial-context record {i} not found", 404)
            initial.append(ev)
        budget = window_tokens or 768
        win = build_window(question, initial, history, self.count_tokens, budget) if window_tokens else \
            ContextWindow(initial, 0, budget, 0, 0, [])
        errors = {}
        base = enh = None
        try:
            baseline = self.baseline(budget, agent.qp)
            if baseline_mode == "raw_generation":
                base = baseline.generate(question, win)
            else:
                base = baseline.answer(question, win)
        except Exception as exc:  # noqa: BLE001 - shown in the panel, never replaced by another model
            errors["baseline"] = f"{type(exc).__name__}: {exc}"
        try:
            enh = agent.query(QueryRequest(question, win.records, mode, verification=verification))
        except Exception as exc:  # noqa: BLE001
            errors["enhanced"] = f"{type(exc).__name__}: {exc}"
        cals = self.calibrators()
        for out, key in ((base, "baseline"), (enh, "enhanced")):
            if (out is not None and not out.get("generation_mode") and cals[key]
                    and cals[key].get("available")):
                c = apply_calibrator(cals[key], output_features(out))
                out["confidence"]["calibrated"] = None if c is None else round(c, 4)
                out["confidence"]["calibrated_definition"] = (
                    f"logistic calibrator fitted on {cals[key].get('n_dev')} answered DEV questions of the synthetic "
                    f"log benchmark ({cals[key].get('fitted_on')}); not validated on other data")
        bb = self._backbone
        res = {"question": question, "shared_context": win.to_dict(),
               "context_policy": ("selected records + most recent arrivals within the token window"
                                  if window_tokens else "selected records only"),
               "baseline_mode": baseline_mode,
               "baseline": base, "enhanced": enh, "errors": errors,
               "models": {"mamba_id": self.cfg["models"]["mamba_id"], "answerer": self.cfg["models"]["answerer"]
                          if self._answerer is None else getattr(self._answerer, "name", "?"),
                          "mamba_loaded": bool(bb and bb.loaded), "mamba_load_error": bb.load_error if bb else None},
               "diagnostic_only": "single-question result; not a benchmark metric"}
        n_mem = memory.store.conn.execute("SELECT COUNT(*) FROM events WHERE status='active'").fetchone()[0]
        res["memory_records"] = n_mem
        if ladle_ann is not None:
            ids = win.to_dict()["record_ids"]
            res["evaluation"] = {
                "gold": {"answer_kind": ladle_ann.answer_kind, "expected_answer": ladle_ann.expected_answer,
                         "match": ladle_ann.match, "required_evidence_ids": ladle_ann.required_evidence_ids,
                         "family": ladle_ann.family, "source": f"ladle dataset gold ({source['dataset']})"},
                "baseline": score_output(base, ladle_ann, system="baseline", stage=int(source["stage"]),
                                         context_ids=ids, n_available=n_mem) if base else None,
                "enhanced": score_output(enh, ladle_ann, system="agentmamba", stage=int(source["stage"]),
                                         context_ids=ids, n_available=n_mem) if enh else None}
        elif reference and not case:
            res["evaluation"] = self._score_reference(self._reference_annotation(reference), base, enh, win,
                                                      n_mem)
        if case:
            if case.get("protocol"):
                from agentmamba.data.protocols import annotation

                ann = annotation(case["protocol"], case["question_id"])
                case = {**case, "stage": int(case.get("stage") or 0)}
            else:
                from agentmamba.data.log_benchmark import load_annotations

                ann = load_annotations(self._bench_root(), case["split"])[case["question_id"]]
            ids = win.to_dict()["record_ids"]
            n = memory.store.conn.execute("SELECT COUNT(*) FROM events WHERE status='active'").fetchone()[0]
            res["evaluation"] = {
                "gold": {"answer_kind": ann.answer_kind, "expected_answer": ann.expected_answer, "match": ann.match,
                         "required_evidence_ids": ann.required_evidence_ids, "family": ann.family},
                "baseline": score_output(base, ann, system="baseline", stage=case["stage"], context_ids=ids,
                                         n_available=n) if base else None,
                "enhanced": score_output(enh, ann, system="agentmamba", stage=case["stage"], context_ids=ids,
                                         n_available=n) if enh else None}
        self._append_history(res, source)
        return res

    # ---------------------------------------------------------------- ladle datasets (short / long)
    def ladle_sources(self) -> list[dict]:
        """Ladle dataset snapshots built by scripts/run_ladle_eval.py (or build_ladle_memory)."""
        from agentmamba.data import ladle_benchmark as lb

        out = []
        for ds in lb.DATASETS:
            base = PROJECT_ROOT / "storage" / "ladle" / ds
            if not base.exists():
                continue
            for sc in sorted(p.name for p in base.iterdir() if p.is_dir()):
                stages = sorted(int(p.name[5:]) for p in (base / sc).glob("stage*")
                                if (p / "memory.sqlite3").exists())
                if stages:
                    out.append({"dataset": ds, "scenario": sc, "stages": stages})
        return out

    def ladle_questions(self, dataset: str, scenario: str) -> list[dict]:
        from agentmamba.data import ladle_benchmark as lb

        return [{"question_id": q.question_id, "text": q.text} for q in lb.load_questions(dataset)
                if q.scenario == scenario]

    def _ladle_gold(self, dataset: str, scenario: str, question: str):
        """Gold annotation when ``question`` is one of the dataset's questions (read only for scoring)."""
        from agentmamba.data import ladle_benchmark as lb

        q = next((q for q in lb.load_questions(dataset) if q.scenario == scenario
                  and q.text.strip().lower() == question.strip().lower()), None)
        return lb.load_annotations(dataset)[q.question_id] if q else None

    def _ladle_memory(self, dataset: str, scenario: str, stage: int):
        from agentmamba.data import ladle_benchmark as lb

        key = ("ladle", dataset, scenario, stage)
        if key not in self._case_agents:
            d = PROJECT_ROOT / "storage" / "ladle" / dataset / scenario / f"stage{stage}"
            if not (d / "memory.sqlite3").exists():
                raise ServiceError(f"no memory snapshot at {d}; run scripts/run_ladle_eval.py first", 404)
            mem = EpisodicMemory(d, self.embedder)
            answerer = self._answerer or build_answerer(self.cfg, self.backbone)
            agent = build_agent(self.cfg, mem, answerer, versions={"ladle_memory": f"{dataset}/{scenario}/stage{stage}"})
            history, seen = [], set()
            for s in lb.stages_of(dataset, scenario):          # the same arrival order the evaluation used
                if s > stage:
                    break
                for e in lb.stage_events(dataset, scenario, s):
                    if e.event_id not in seen:
                        seen.add(e.event_id)
                        history.append(e)
            self._case_agents[key] = (mem, agent, history)
        return self._case_agents[key]

    def ladle_memory(self, dataset: str, scenario: str, stage: int) -> EpisodicMemory:
        return self._ladle_memory(dataset, scenario, stage)[0]

    # ---------------------------------------------------------------- user reference + run history
    @staticmethod
    def _reference_annotation(reference: dict):
        from agentmamba.data.log_benchmark import LogAnnotation

        kind = reference.get("answer_kind") or "value"
        if kind not in ("value", "abstain"):
            raise ServiceError("reference.answer_kind must be 'value' or 'abstain'", 422)
        expected = (reference.get("expected_answer") or "").strip() or None
        match = reference.get("match") or "contains"
        if kind == "value" and not expected:
            raise ServiceError("reference.expected_answer is required when answer_kind is 'value'", 422)
        if match not in ("contains", "exact"):
            raise ServiceError("reference.match must be 'contains' or 'exact'", 422)
        return LogAnnotation("user-reference", "app", "app", "user_reference", kind,
                             expected if kind == "value" else None, match if kind == "value" else None, [], [],
                             note="reference answer supplied by the user at question time")

    @staticmethod
    def _score_reference(ann, base: dict | None, enh: dict | None, win, n_available: int) -> dict:
        """Score both outputs against a user reference. Without evidence record IDs, grounding and
        evidence coverage are undefined and reported as None (never guessed)."""
        from agentmamba.evaluation.comparison import score_output

        ids = win.to_dict()["record_ids"]
        undefined = ("grounded_correct", "unsupported", "evidence_in_context", "evidence_covered", "recall_at_k",
                     "precision_at_k", "rr")
        scored = {}
        for key, out, system in (("baseline", base, "baseline"), ("enhanced", enh, "agentmamba")):
            scored[key] = None if out is None else {
                **score_output(out, ann, system=system, stage=0, context_ids=ids, n_available=n_available),
                **{k: None for k in undefined}}
        return {"gold": {"answer_kind": ann.answer_kind, "expected_answer": ann.expected_answer, "match": ann.match,
                         "required_evidence_ids": [], "family": ann.family, "source": "your reference answer"},
                **scored}

    @property
    def history_path(self) -> Path:
        return self.storage_dir / "comparison_history.jsonl"

    def _append_history(self, res: dict, source: dict | None) -> None:
        from datetime import datetime

        ev = res.get("evaluation") or {}
        row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "question": res["question"],
               "memory": (f"{source['dataset']} / {source['scenario']} / stage {source['stage']}" if source
                          else "uploaded logs"), "records": res.get("memory_records")}
        for key, out, e in (("baseline", res["baseline"], ev.get("baseline")),
                            ("agentmamba", res["enhanced"], ev.get("enhanced"))):
            row.update({f"{key}_answer": (out or {}).get("answer_value") or (out or {}).get("answer_status", "failed"),
                        f"{key}_correct": e["correct"] if e else None,
                        f"{key}_s": ((out or {}).get("timings") or {}).get("total_s")})
        try:
            with self._lock:
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                with self.history_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass                    # history is a convenience; never fail an investigation over it

    def comparison_history(self) -> list[dict]:
        p = self.history_path
        if not p.exists():
            return []
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def clear_comparison_history(self) -> None:
        with self._lock:
            self.history_path.unlink(missing_ok=True)

    def protocol_cases(self) -> list[dict]:
        """Questions of every staged protocol (text only; gold stays on disk until scoring)."""
        from agentmamba.data.protocols import list_protocols, load_protocol

        return [{"protocol": n, "question_id": q["question_id"], "text": q["text"], "kind": q["kind"]}
                for n in list_protocols() for q in load_protocol(n)["questions"]]

    def _bench_root(self) -> Path:
        return PROJECT_ROOT / "data" / "log_benchmark" / "v1"

    def benchmark_cases(self) -> list[dict]:
        """Benchmark questions whose scenario has stage snapshots from a finished context-growth run."""
        from agentmamba.data.log_benchmark import load_questions

        base = PROJECT_ROOT / "storage" / "experiments" / "context_growth"
        runs = sorted((p.name for p in base.iterdir() if p.is_dir()), reverse=True) if base.exists() else []
        out = []
        for split in ("dev", "test"):
            try:
                qs = load_questions(self._bench_root(), split)
            except FileNotFoundError:
                continue
            for q in qs:
                run = next((r for r in runs if f"_{split}" in r and (base / r / q.scenario / "stage1").exists()), None)
                if run:
                    out.append({"split": split, "question_id": q.question_id, "scenario": q.scenario,
                                "text": q.text, "run": run})
        return out

    def _case_memory(self, case: dict):
        from agentmamba.data.log_benchmark import stage_files

        split, qid, stage, run = case["split"], case["question_id"], int(case["stage"]), case["run"]
        scen = qid.split("-")[0]
        key = (run, scen, stage)
        if key not in self._case_agents:
            d = PROJECT_ROOT / "storage" / "experiments" / "context_growth" / run / scen / f"stage{stage}"
            if not (d / "memory.sqlite3").exists():
                raise ServiceError(f"no memory snapshot at {d}; run scripts/run_context_growth.py first", 404)
            mem = EpisodicMemory(d, self.embedder)
            answerer = self._answerer or build_answerer(self.cfg, self.backbone)
            agent = build_agent(self.cfg, mem, answerer, versions={"case_memory": f"{run}/{scen}/stage{stage}"})
            root = self._bench_root() / split / scen
            history, seen = [], set()
            for s in range(1, stage + 1):          # the same arrival order the experiment used
                for f in stage_files(root, s):
                    for e in load_any(root / f["file"], stage=f["stage"]):
                        if e.event_id not in seen:
                            seen.add(e.event_id)
                            history.append(e)
            self._case_agents[key] = (mem, agent, history)
        return self._case_agents[key]

    def status(self) -> dict:
        st: dict = {"storage_dir": str(self.storage_dir), "memory_policy": self.cfg["memory"]["policy"],
                    "app_budget_bytes": get_dotted(self.cfg, "memory.app_budget_bytes")}
        gp = gate_path(self.cfg)
        try:
            art = self.gate_artifact()
            st["gate"] = {"path": str(gp), "present": art is not None,
                          "version": art.meta["gate_version"] if art else None,
                          "use_mamba_features": art.meta["use_mamba_features"] if art else None,
                          "val": art.meta["val"] if art else None}
        except Exception as exc:  # noqa: BLE001
            st["gate"] = {"path": str(gp), "present": False, "error": str(exc)}
        paths = AgentMamba.artifact_paths(self.cfg)
        st["controller"] = {"present": paths["controller"].exists(), "intent_classifier": paths["intent"].exists()}
        try:
            mem = self.memory
            s = mem.stats()
            st["sqlite"] = {"ok": True, "records": s["records"], "relationships": s["relationships"],
                            "bytes": s["record_bytes"]}
            st["faiss"] = {"ok": s["index_consistency"]["ok"], "vectors": s["vectors"],
                           "recovery_note": s["recovery_note"]}
        except Exception as exc:  # noqa: BLE001
            st["sqlite"] = {"ok": False, "error": str(exc)}
        bb = self._backbone
        st["mamba"] = {"model_id": self.cfg["models"]["mamba_id"], "loaded": bool(bb and bb.loaded),
                       "load_error": bb.load_error if bb else None, "answerer": self.cfg["models"]["answerer"],
                       "note": "loaded lazily on the first Mamba scoring call that is not already in the score "
                               "cache; use load_mamba() / the dashboard button to load it now"}
        st["embedding_model"] = self.cfg["models"]["embedding_id"]
        st["ready"] = bool(st.get("sqlite", {}).get("ok") and st["controller"]["present"])
        return st

    def experiments(self) -> list[dict]:
        root = resolve_path(self.cfg, "paths.results_dir")
        out = []
        for d in sorted(root.glob("*"), reverse=True) if root.exists() else []:
            if not d.is_dir():
                continue
            s = d / "summary.json"
            item = {"experiment_id": d.name, "has_summary": s.exists(),
                    "status": "complete" if s.exists() else "incomplete_or_failed"}
            if s.exists():
                try:
                    data = json.loads(s.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "answer" in data:
                        item.update({"method": data.get("method"), "n": data.get("n"),
                                     "overall_accuracy": data["answer"]["overall_accuracy"],
                                     "exact_match": data["answer"]["exact_match"]})
                    elif isinstance(data, dict) and "by_condition" in data:
                        am = data["by_condition"].get("agentmamba", {})
                        item.update({"method": f"context_growth ({data.get('split')}, {data.get('answerer')})",
                                     "n": data.get("n_rows"), "overall_accuracy": am.get("accuracy"),
                                     "exact_match": am.get("exact_match")})
                except json.JSONDecodeError:
                    item["status"] = "corrupt_summary"
            out.append(item)
        return out

    def experiment(self, experiment_id: str) -> dict:
        root = resolve_path(self.cfg, "paths.results_dir").resolve()
        d = (root / experiment_id).resolve()
        if root not in d.parents or not d.is_dir():
            raise ServiceError(f"experiment {experiment_id!r} not found", 404)
        out = {"experiment_id": experiment_id, "files": sorted(p.name for p in d.iterdir())}
        for name in ("summary.json", "config.json"):
            if (d / name).exists():
                out[name.split(".")[0]] = json.loads((d / name).read_text(encoding="utf-8"))
        return out
