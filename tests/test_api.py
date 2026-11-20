"""API tests using FastAPI's built-in test client (no real server needed)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

HEADER = "timestamp,service,log_level,message,latency_ms,error_rate,cpu_pct,memory_pct\n"
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_upload.csv"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs the startup code that loads the models
        yield c


def upload(client, content, name="logs.csv"):
    return client.post("/api/analyze", files={"file": (name, content, "text/csv")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "models_loaded": True}


def test_demo_returns_summary_charts_and_incidents(client):
    body = client.get("/api/demo").json()
    assert body["source"] == "demo"
    assert body["summary"]["total_incidents"] == len(body["incidents"]) > 0
    assert body["timeseries"] and body["thresholds"]
    assert not any(k.startswith("_") for k in body["incidents"][0])  # internal fields are hidden


def test_incidents_endpoint_filters(client):
    client.get("/api/demo")
    everything = client.get("/api/incidents").json()
    critical = client.get("/api/incidents?severity=critical").json()
    assert critical["count"] <= everything["count"]
    assert all(i["severity"] == "CRITICAL" for i in critical["incidents"])
    assert client.get("/api/incidents?severity=nonsense").status_code == 400


def test_model_info(client):
    info = client.get("/api/model-info").json()
    assert info["classes"] == ["application", "database", "deployment", "network", "resource"]
    assert "holdout_accuracy" in info["classifier"]


def test_analyze_valid_csv_is_stored_for_incidents_endpoint(client):
    ok = upload(client, SAMPLE.read_bytes(), "sample.csv")
    assert ok.status_code == 200 and ok.json()["source"] == "upload: sample.csv"
    assert client.get("/api/incidents").json()["source"] == "upload: sample.csv"


def test_analyze_rejects_bad_files(client):
    assert upload(client, b"hello", "notes.txt").status_code == 400            # wrong extension
    assert upload(client, b"", "empty.csv").status_code == 400                 # empty
    assert upload(client, b"a,b\n1,2\n").status_code == 400                    # missing columns
    assert upload(client, HEADER + "bad-date,svc,INFO,hi,10,1,10,10\n").status_code == 400  # no valid rows


def test_analyze_drops_bad_rows_but_keeps_good_ones(client):
    csv = (HEADER
           + "2026-01-01 10:00:00,svc,INFO,all good,100,1,40,50\n"
           + "2026-01-01 10:01:00,svc,INFO,missing number,abc,1,40,50\n"
           + "2026-01-01 10:02:00,svc,INFO,percent too big,100,500,40,50\n")
    response = upload(client, csv.encode())
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["rows_analyzed"] == 1
    assert len(body["data_quality_notes"]) == 2
