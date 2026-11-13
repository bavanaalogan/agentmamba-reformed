"""FastAPI application. Every endpoint calls the real pipeline through AgentService.

    uvicorn agentmamba.api.main:app --app-dir src --port 8000
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from agentmamba.api.schemas import CompareIn, EventsIn, QueryIn
from agentmamba.config import load_config
from agentmamba.data.document_loader import SUPPORTED_SUFFIXES
from agentmamba.memory.schema import Event
from agentmamba.pipeline.agent import QueryRequest
from agentmamba.pipeline.service import AgentService, ServiceError

MAX_UPLOAD_BYTES = 20 * 2**20


def create_app(service: AgentService | None = None) -> FastAPI:
    svc = service or AgentService(load_config())
    app = FastAPI(title="AgentMamba", version="0.1.0",
                  description="Research prototype: plant episodic memory + conditional evidence retrieval. "
                              "Not a safety-certified system.")

    def guard(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except ServiceError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, f"{type(exc).__name__}: {exc}") from exc

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/status")
    def status():
        return svc.status()

    @app.post("/models/load")
    def load_models():
        return guard(svc.load_mamba)

    @app.post("/events/ingest")
    def ingest_events(body: EventsIn):
        events = guard(lambda: [Event.from_dict(e) for e in body.events])
        return guard(svc.ingest, events, body.budget_bytes)

    @app.post("/documents/ingest")
    async def ingest_document(file: UploadFile = File(...), stage: str = Form("MAINTENANCE"),
                              source_type: str = Form("authorized"), budget_bytes: int | None = Form(None),
                              policy: str | None = Form(None)):
        """Documents (TXT/MD, chunked) and operational logs (CSV/JSON/JSONL, any columns).
        ``policy=store_all`` explicitly bypasses the memory gate for this upload."""
        name = Path(file.filename or "upload").name          # strip any client path components
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(415, f"unsupported file type {suffix!r}; supported: {sorted(SUPPORTED_SUFFIXES)}")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / name
            p.write_bytes(data)
            report = guard(svc.ingest_file, p, stage, source_type, budget_bytes, policy or None)
        report["normalized_events"] = report.pop("normalized_preview")[:50]
        return report

    @app.get("/events")
    def list_events(stage: str | None = None, event_type: str | None = None, asset_id: str | None = None,
                    heat_id: str | None = None, ladle_id: str | None = None, coil_id: str | None = None,
                    source_type: str | None = None, start: str | None = None, end: str | None = None,
                    q: str | None = None, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
        evs = svc.memory.store.query_events(stage=stage, event_type=event_type, asset_id=asset_id, heat_id=heat_id,
                                            ladle_id=ladle_id, coil_id=coil_id, source_type=source_type, start=start,
                                            end=end, text_like=q, limit=limit, offset=offset)
        return {"count": len(evs), "events": [e.to_dict() for e in evs]}

    @app.get("/events/{event_id}")
    def get_event(event_id: str):
        ev = svc.memory.store.get_event(event_id)
        if ev is None:
            raise HTTPException(404, f"record {event_id} not in memory")
        rels = [r.to_dict() for r in svc.memory.store.relationships_of(event_id)]
        for ent in ev.entity_ids():
            rels += [r.to_dict() for r in svc.memory.store.relationships_of(ent) if r.rel_type != "mentions"]
        return {"event": ev.to_dict(), "relationships": rels, "gate_decisions": svc.memory.store.decisions(event_id=event_id)}

    @app.get("/memory/stats")
    def memory_stats():
        return svc.memory.stats()

    @app.post("/memory/rebuild-index")
    def rebuild_index():
        return guard(svc.memory.rebuild_index)

    @app.post("/query")
    def query(body: QueryIn):
        ctx = [Event.from_dict({**c.model_dump(), "observations": c.observations}) for c in body.initial_context]
        for eid in body.initial_context_ids:
            ev = svc.memory.store.get_event(eid)
            if ev is None:
                raise HTTPException(404, f"initial-context record {eid} not found")
            ctx.append(ev)
        filters = {k: v for k, v in (body.filters.model_dump() if body.filters else {}).items() if v}
        req = QueryRequest(body.question, ctx, body.mode, filters or None, body.verification, top_k=body.top_k)
        return guard(svc.query, req)

    @app.post("/compare")
    def compare(body: CompareIn):
        """Baseline Mamba and AgentMamba on the same question and initial context (single-question
        diagnostic; aggregate results come from scripts/run_context_growth.py)."""
        case = body.case.model_dump() if body.case else None
        return guard(svc.compare, body.question, body.initial_context_ids, body.window_tokens, body.mode,
                     body.verification, case)

    @app.get("/experiments")
    def experiments():
        return svc.experiments()

    @app.get("/experiments/{experiment_id}")
    def experiment(experiment_id: str):
        return guard(svc.experiment, experiment_id)

    app.state.service = svc
    return app


app = create_app()
