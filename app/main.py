"""FastAPI backend for IncidentIQ.

Endpoints:
  GET  /health          -> health check (used by Docker and Render)
  GET  /api/demo        -> generate synthetic demo data, analyse it, return everything
  POST /api/analyze     -> upload a CSV, analyse it, return everything
  GET  /api/incidents   -> incidents from the latest analysis (optional filters)
  GET  /api/model-info  -> how the models were trained and how well they did
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.data_gen import generate_demo
from app.intelligence import (GROUP_GAP_MINUTES, LOW_CONFIDENCE, SEVERITY_ORDER, THRESHOLDS,
                              analyze, public_incidents)
from app.ml import load_models
from app.validation import MAX_BYTES, MAX_ROWS, CSVValidationError, parse_csv

STATIC_DIR = Path(__file__).resolve().parent / "static"

# The latest analysis is kept in memory so /api/incidents can return it.
# (Simple on purpose: one shared "current result", fine for a demo, not for many users.)
STATE = {"bundle": None, "result": None, "source": None}


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


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
