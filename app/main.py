"""FastAPI backend for IncidentIQ.

Endpoints:
  GET  /health          -> health check (used by Docker and Render)
  GET  /api/demo        -> generate synthetic demo data, analyse it, return everything
  POST /api/analyze     -> upload a CSV, analyse it, return everything
  GET  /api/incidents   -> incidents from the latest analysis (optional filters)
  GET  /api/model-info  -> how the models were trained and how well they did
  POST /api/ingest      -> real-time streaming telemetry ingestion for live microservices
  GET  /api/stream      -> view recently ingested live telemetry events and diagnoses
  DELETE /api/stream    -> clear the live telemetry event buffer
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.data_gen import generate_demo
from app.intelligence import (GROUP_GAP_MINUTES, LOW_CONFIDENCE, SEVERITY_ORDER, THRESHOLDS,
                              analyze, evaluate_single_event, public_incidents)
from app.ml import load_models
from app.validation import MAX_BYTES, MAX_ROWS, CSVValidationError, parse_csv

STATIC_DIR = Path(__file__).resolve().parent / "static"

# The latest analysis is kept in memory so /api/incidents can return it.
# (Simple on purpose: one shared "current result", fine for a demo, not for many users.)
STATE = {"bundle": None, "result": None, "source": None}

# In-memory buffer for real-time streamed events from microservices
LIVE_STREAM = []
MAX_STREAM_ITEMS = 300


class TelemetryEvent(BaseModel):
    service: str = Field(..., description="Service name (e.g. payment-service)")
    message: str = Field(..., description="Log message text")
    latency_ms: float = Field(..., ge=0, description="Response time in milliseconds")
    error_rate: float = Field(..., ge=0, le=100, description="Error rate percentage (0-100)")
    cpu_pct: float = Field(..., ge=0, le=100, description="CPU usage percentage (0-100)")
    memory_pct: float = Field(..., ge=0, le=100, description="Memory usage percentage (0-100)")
    log_level: str = Field("INFO", description="Log level: INFO, WARN, ERROR, or CRITICAL")
    timestamp: Optional[str] = Field(None, description="Event timestamp (defaults to current UTC)")


def _store(result, source):
    """Remember the latest result and return the JSON-ready version of it."""
    STATE["result"], STATE["source"] = result, source
    return {"source": source, **result, "incidents": public_incidents(result["incidents"])}


@asynccontextmanager
async def lifespan(app):
    # On startup: load (or train) the models, then analyse demo data so the API is usable at once.
    STATE["bundle"] = load_models()
    _store(analyze(generate_demo(), STATE["bundle"]), "demo")
    yield


app = FastAPI(title="IncidentIQ", version="1.0.0", lifespan=lifespan,
              description="AI-powered incident intelligence (anomaly detection + log classification + rules).")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": STATE["bundle"] is not None}


@app.get("/api/demo")
def demo():
    """Create a fresh synthetic dataset (same every time) and analyse it."""
    return _store(analyze(generate_demo(), STATE["bundle"]), "demo")


@app.post("/api/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """Analyse an uploaded CSV file."""
    name = file.filename or ""
    if not name.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a file with a .csv extension.")
    raw = await file.read(MAX_BYTES + 1)  # read one extra byte so we can tell if it is too big
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File is too large. The limit is {MAX_BYTES // (1024 * 1024)} MB.")
    try:
        df, notes = parse_csv(raw)
    except CSVValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        result = analyze(df, STATE["bundle"], notes)
    except Exception:  # unexpected failure inside the analysis; do not leak internals
        raise HTTPException(status_code=500, detail="The analysis failed unexpectedly. Please check your data and try again.")
    return _store(result, f"upload: {name}")


@app.get("/api/incidents")
def incidents(severity: Optional[str] = None, category: Optional[str] = None):
    """Incidents from the latest analysis. Filter with ?severity=HIGH and/or ?category=database."""
    items = public_incidents(STATE["result"]["incidents"])
    if severity:
        if severity.upper() not in SEVERITY_ORDER:
            raise HTTPException(status_code=400, detail=f"severity must be one of {', '.join(SEVERITY_ORDER)}")
        items = [i for i in items if i["severity"] == severity.upper()]
    if category:
        items = [i for i in items if i["category"] == category.lower()]
    return {"source": STATE["source"], "count": len(items), "incidents": items}


@app.get("/api/model-info")
def model_info():
    """Training details, hold-out results and the rule settings."""
    info = STATE["bundle"]["info"]
    return {
        **info,
        "rules": {
            "thresholds": THRESHOLDS,
            "group_gap_minutes": GROUP_GAP_MINUTES,
            "low_confidence_cutoff": LOW_CONFIDENCE,
            "upload_limits": {"max_rows": MAX_ROWS, "max_megabytes": MAX_BYTES // (1024 * 1024)},
        },
        "note": ("Models are trained on synthetic data made from fixed message templates, "
                 "so the hold-out scores are optimistic compared with real-world logs."),
    }


@app.post("/api/ingest")
def ingest_event(payload: Union[TelemetryEvent, List[TelemetryEvent]]):
    """Real-time streaming telemetry ingestion endpoint.

    Accepts a single telemetry event or a batch of events from any live service.
    Runs each through Isolation Forest and TF-IDF classifiers, scores severity,
    and returns immediate diagnostics (anomaly score, category, root cause, action).
    """
    if STATE["bundle"] is None:
        raise HTTPException(status_code=503, detail="Models are still initializing. Please retry in a moment.")

    items = payload if isinstance(payload, list) else [payload]
    results = []

    for item in items:
        event_dict = item.model_dump() if hasattr(item, "model_dump") else item.dict()
        if not event_dict.get("timestamp"):
            event_dict["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        evaluation = evaluate_single_event(event_dict, STATE["bundle"])
        event_record = {
            "id": f"evt-{uuid.uuid4().hex[:8]}",
            **evaluation,
            "raw": event_dict,
        }
        LIVE_STREAM.append(event_record)
        if len(LIVE_STREAM) > MAX_STREAM_ITEMS:
            LIVE_STREAM.pop(0)
        results.append(event_record)

    return results[0] if not isinstance(payload, list) else {"count": len(results), "events": results}


@app.get("/api/stream")
def get_stream(limit: int = 50, service: Optional[str] = None):
    """Retrieve recent real-time ingested events and their live evaluation."""
    events = LIVE_STREAM
    if service:
        events = [e for e in events if e.get("service") == service]
    sliced = events[-limit:]
    return {"total": len(events), "returned": len(sliced), "events": list(reversed(sliced))}


@app.delete("/api/stream")
def clear_stream():
    """Clear the in-memory live stream telemetry buffer."""
    count = len(LIVE_STREAM)
    LIVE_STREAM.clear()
    return {"status": "ok", "cleared_events": count}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
