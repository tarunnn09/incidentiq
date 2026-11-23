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


def test_ingest_normal_event(client):
    client.delete("/api/stream")
    payload = {
        "service": "order-service",
        "message": "Order 1042 created successfully",
        "latency_ms": 110.5,
        "error_rate": 0.5,
        "cpu_pct": 35.0,
        "memory_pct": 50.0,
        "log_level": "INFO",
    }
    res = client.post("/api/ingest", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "order-service"
    assert data["status"] == "normal"
    assert data["evaluation"]["is_anomaly"] is False
    assert data["diagnosis"]["severity"] == "LOW"


def test_ingest_anomalous_incident_event(client):
    payload = {
        "service": "inventory-db",
        "message": "CRITICAL Connection pool exhausted: all 212 connections in use",
        "latency_ms": 2400.0,
        "error_rate": 15.0,
        "cpu_pct": 85.0,
        "memory_pct": 92.0,
        "log_level": "CRITICAL",
    }
    res = client.post("/api/ingest", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "inventory-db"
    assert data["status"] in ("incident_triggered", "anomaly_detected")
    assert data["evaluation"]["predicted_category"] == "database"
    assert "connection pool" in data["diagnosis"]["probable_cause"].lower()
    assert len(data["diagnosis"]["recommended_steps"]) > 0


def test_ingest_batch_and_stream_endpoints(client):
    client.delete("/api/stream")
    events = [
        {
            "service": "auth-service",
            "message": "User login successful",
            "latency_ms": 95.0,
            "error_rate": 0.1,
            "cpu_pct": 20.0,
            "memory_pct": 40.0,
            "log_level": "INFO",
        },
        {
            "service": "payment-service",
            "message": "TLS handshake failed with bank gateway",
            "latency_ms": 1500.0,
            "error_rate": 20.0,
            "cpu_pct": 60.0,
            "memory_pct": 55.0,
            "log_level": "ERROR",
        },
    ]
    res = client.post("/api/ingest", json=events)
    assert res.status_code == 200
    batch = res.json()
    assert batch["count"] == 2

    # Check stream endpoint
    stream_res = client.get("/api/stream?limit=10")
    assert stream_res.status_code == 200
    stream = stream_res.json()
    assert stream["total"] >= 2
    assert stream["returned"] >= 2

    # Clear stream
    del_res = client.delete("/api/stream")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "ok"
    assert client.get("/api/stream").json()["total"] == 0
