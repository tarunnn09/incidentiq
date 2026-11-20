"""Tests for data generation, the two ML models and the rule engine."""
import pandas as pd
import pytest

from app.data_gen import CATEGORIES, generate_dataset, generate_demo
from app.intelligence import analyze, severity_from_points, score_severity, group_events
from app.ml import classify_messages, detect_anomalies, load_models


@pytest.fixture(scope="module")
def bundle():
    return load_models()  # trains automatically if models/ is empty


def test_dataset_is_reproducible_and_has_required_columns():
    a, b = generate_demo(seed=1), generate_demo(seed=1)
    pd.testing.assert_frame_equal(a, b)  # same seed -> identical data
    assert not generate_demo(seed=1).equals(generate_demo(seed=2))
    for col in ["timestamp", "service", "log_level", "message", "latency_ms", "error_rate", "cpu_pct", "memory_pct"]:
        assert col in a.columns
    assert {"normal", *CATEGORIES} == set(generate_dataset(minutes=400, incidents_per_category=5)["true_category"])


def test_anomaly_model_separates_normal_from_abnormal(bundle):
    rows = pd.DataFrame([
        {"latency_ms": 120, "error_rate": 0.8, "cpu_pct": 40, "memory_pct": 55},    # typical minute
        {"latency_ms": 3000, "error_rate": 20, "cpu_pct": 97, "memory_pct": 96},    # clearly broken
    ])
    scores, flags = detect_anomalies(bundle, rows)
    assert not flags[0] and flags[1]
    assert scores[1] > scores[0]


@pytest.mark.parametrize("message, expected", [
    ("Deadlock detected on table orders, transaction rolled back", "database"),
    ("DNS resolution failed for host db.internal", "network"),
    ("Unhandled exception: NullPointerException in OrderController", "application"),
    ("Rollback triggered after release v2.1.0", "deployment"),
    ("OutOfMemoryError: Java heap space", "resource"),
])
def test_classifier_predicts_expected_category(bundle, message, expected):
    classes, proba = classify_messages(bundle, [message])
    assert classes[proba[0].argmax()] == expected
    assert proba[0].max() > 0.4


def test_classifier_holdout_accuracy_is_reported(bundle):
    assert bundle["info"]["classifier"]["holdout_accuracy"] >= 0.9


def test_severity_points_map_to_levels():
    assert [severity_from_points(p) for p in (0, 3, 4, 6, 8, 10)] == ["LOW", "LOW", "MEDIUM", "HIGH", "CRITICAL", "CRITICAL"]
    peaks = {"latency_ms": 3000, "error_rate": 20, "cpu_pct": 97, "memory_pct": 96}
    severity, reasons = score_severity(peaks, 0.8, 0.6, 12, {"CRITICAL", "ERROR"})
    assert severity == "CRITICAL" and reasons  # every point has a written reason


def test_events_are_grouped_per_service_with_time_gap():
    t0 = pd.Timestamp("2026-01-01 10:00")
    events = pd.DataFrame({
        "service": ["a", "a", "a", "b"],
        "timestamp": [t0, t0 + pd.Timedelta(minutes=2), t0 + pd.Timedelta(minutes=30), t0],
    })
    grouped = group_events(events)
    assert grouped["group"].nunique() == 3  # a (10:00+10:02), a (10:30), b


def test_demo_analysis_finds_planted_incidents_with_honest_wording(bundle):
    result = analyze(generate_demo(), bundle)
    found = {(i["service"], i["category"]) for i in result["incidents"]}
    for expected in [("inventory-db", "database"), ("api-gateway", "network"), ("order-service", "application"),
                     ("payment-service", "deployment"), ("notification-service", "resource")]:
        assert expected in found
    for incident in result["incidents"]:
        assert incident["severity"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert incident["evidence"] and incident["recommendations"]
        assert "not a confirmed cause" in incident["cause_disclaimer"]  # never claim certainty
