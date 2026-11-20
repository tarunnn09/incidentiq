"""Synthetic log + metric generator for IncidentIQ.

Every service writes one log line per minute together with four health numbers
(latency, error rate, CPU, memory). Most minutes are "normal". On top of that we
inject incidents (database, network, application, deployment, resource) where the
numbers drift away from normal and the log message describes the failure.

Everything is driven by a seed, so the same seed always gives the same dataset.

Run `python -m app.data_gen` to write the CSV files into the data/ folder.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SERVICES = [
    "api-gateway",
    "auth-service",
    "payment-service",
    "order-service",
    "inventory-db",
    "notification-service",
]
CATEGORIES = ["database", "network", "application", "deployment", "resource"]
CSV_COLUMNS = [
    "timestamp", "service", "log_level", "message",
    "latency_ms", "error_rate", "cpu_pct", "memory_pct",
]

BASE_TIME = pd.Timestamp("2026-01-15 09:00:00")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Seeds: training data, hold-out test data and the demo dashboard all use different seeds.
TRAIN_SEED = 42
HOLDOUT_SEED = 99
DEMO_SEED = 7

# ----------------------------------------------------------------------------
# Log message templates. {placeholders} are filled with random values.
# ----------------------------------------------------------------------------
NORMAL_INFO = [
    "Request completed successfully in {ms}ms",
    "Health check passed",
    "User login successful for user_{id}",
    "Cache hit for key product_{id}",
    "Scheduled cleanup job finished",
    "Order {id} created",
    "Payment {id} processed",
    "Metrics exported successfully",
    "Configuration reloaded",
    "Heartbeat OK",
]
NORMAL_WARN = [  # harmless warnings: they should NOT become incidents
    "Retrying request, attempt 1 of 3",
    "Slow response from cache, {ms}ms",
    "Deprecated API version used by client",
]

INCIDENT_TEMPLATES = {
    "database": [
        "Connection pool exhausted: all {cnt} connections in use",
        "Deadlock detected on table {table}, transaction rolled back",
        "Slow query detected: SELECT on {table} took {ms}ms",
        "Too many connections to PostgreSQL server",
        "Database query timeout after {ms}ms on table {table}",
        "Replication lag of {sec}s detected on replica",
        "Could not acquire lock on table {table}",
    ],
    "network": [
        "Connection timeout while calling upstream service {svc}",
        "DNS resolution failed for host {host}",
        "Connection reset by peer on port {port}",
        "Packet loss of {loss}% detected on network interface eth0",
        "TLS handshake timeout with {svc}",
        "502 Bad Gateway returned by upstream {svc}",
        "No route to host {host}",
        "Socket read timeout after {ms}ms",
    ],
    "application": [
        "Unhandled exception: NullPointerException in OrderController",
        "HTTP 500 Internal Server Error in handler /api/{ep}",
        "KeyError: '{field}' while parsing request payload",
        "Unhandled TypeError in {mod} module, stack trace follows",
        "Request validation failed unexpectedly for endpoint /{ep}",
        "Uncaught exception in worker thread, request aborted",
        "JSON decode error in response handled by {mod}",
        "Request handler timed out in {mod} module",
    ],
    "deployment": [
        "Deployment {ver} failed health check, rolling back",
        "Rollback triggered after release {ver}",
        "Missing environment variable {var} after deploy",
        "Container failed to start: ImagePullBackOff for image {svc}:{ver}",
        "Config file not found after release {ver}",
        "Readiness probe failed for new version {ver}",
        "Database migration {ver} failed during deployment",
        "Canary release {ver} error rate above threshold",
    ],
    "resource": [
        "OutOfMemoryError: Java heap space",
        "CPU usage above {pct}% for 5 minutes",
        "Disk usage at {pct}% on /var/lib/data",
        "GC overhead limit exceeded, memory almost full",
        "Memory usage critical: {pct}% of limit used",
        "Thread pool saturated, no free worker threads",
        "Container killed: OOMKilled, memory limit reached",
        "File descriptor limit reached: too many open files",
    ],
}

# "Peak" metric values reached at full intensity: (low, high)
INCIDENT_PEAKS = {
    "database":    {"latency_ms": (900, 2600), "error_rate": (5, 15),  "cpu_pct": (55, 80), "memory_pct": (55, 75)},
    "network":     {"latency_ms": (1500, 5000), "error_rate": (6, 25), "cpu_pct": (30, 50), "memory_pct": (50, 60)},
    "application": {"latency_ms": (250, 700),  "error_rate": (8, 30),  "cpu_pct": (40, 65), "memory_pct": (55, 70)},
    "deployment":  {"latency_ms": (300, 1000), "error_rate": (12, 45), "cpu_pct": (30, 60), "memory_pct": (50, 70)},
    "resource":    {"latency_ms": (600, 2500), "error_rate": (3, 12),  "cpu_pct": (91, 99), "memory_pct": (91, 99)},
}

# A fixed incident plan for the demo dashboard: (category, service, start_minute, minutes, intensity)
DEMO_MINUTES = 180
DEMO_PLAN = [
    ("database",    "inventory-db",         35, 9, 1.0),
    ("network",     "api-gateway",          70, 7, 0.8),
    ("application", "order-service",       100, 5, 0.45),   # mild -> lower severity
    ("deployment",  "payment-service",     128, 8, 0.9),
    ("resource",    "auth-service",        150, 6, 0.35),   # mild -> lower severity
    ("resource",    "notification-service", DEMO_MINUTES - 12, 12, 1.0),  # still going at the end
]


def _fill(rng, template, service):
    """Replace {placeholders} in a message template with random realistic values."""
    values = {
        "ms": int(rng.integers(150, 9000)),
        "id": int(rng.integers(1000, 99999)),
        "cnt": int(rng.integers(20, 500)),
        "sec": int(rng.integers(5, 300)),
        "loss": int(rng.integers(5, 60)),
        "pct": int(rng.integers(90, 100)),
        "port": int(rng.choice([443, 5432, 6379, 8080, 9000])),
        "table": str(rng.choice(["orders", "payments", "users", "inventory", "sessions"])),
        "svc": str(rng.choice([s for s in SERVICES if s != service])),
        "host": str(rng.choice(["db.internal", "cache.internal", "payments.partner.com", "auth.corp.net"])),
        "ep": str(rng.choice(["orders", "checkout", "users", "inventory"])),
        "field": str(rng.choice(["user_id", "amount", "sku", "address"])),
        "mod": str(rng.choice(["billing", "cart", "pricing", "search"])),
        "var": str(rng.choice(["DB_URL", "API_KEY_PATH", "REDIS_HOST", "SERVICE_PORT"])),
        "ver": f"v{rng.integers(1, 4)}.{rng.integers(0, 9)}.{rng.integers(0, 9)}",
    }
    text = template.format(**values)
    # Small bit of realistic noise so the classifier cannot rely on exact strings.
    if rng.random() < 0.2:
        text += f" (trace_id={rng.integers(16**7, 16**8):x})"
    if rng.random() < 0.1:
        text = f"[{service}] {text}"
    return text


def _normal_metrics(rng, n):
    """Healthy behaviour: latency ~120ms, errors ~1%, CPU ~40%, memory ~55%."""
    minute = np.arange(n)
    daily_wave = np.sin(minute / 240 * 2 * np.pi)  # gentle slow wave so the charts look alive
    latency = np.clip(rng.normal(120, 22, n) + 8 * daily_wave, 40, None)
    spike = rng.random(n) < 0.02  # occasional harmless spike
    latency[spike] *= rng.uniform(1.5, 2.2, spike.sum())
    error = np.clip(np.abs(rng.normal(0.8, 0.5, n)), 0, 2.5)
    cpu = np.clip(rng.normal(40, 7, n) + 5 * daily_wave, 10, 72)
    mem = np.clip(rng.normal(55, 5, n), 30, 75)
    return latency, error, cpu, mem


def _build(seed, minutes, plan):
    """Create the full dataset: normal behaviour everywhere + injected incidents from `plan`."""
    rng = np.random.default_rng(seed)
    per_service = {}
    for svc in SERVICES:
        lat, err, cpu, mem = _normal_metrics(rng, minutes)
        levels, messages, labels = [], [], []
        for _ in range(minutes):
            if rng.random() < 0.06:
                levels.append("WARN")
                messages.append(_fill(rng, str(rng.choice(NORMAL_WARN)), svc))
            else:
                levels.append("INFO")
                messages.append(_fill(rng, str(rng.choice(NORMAL_INFO)), svc))
            labels.append("normal")
        per_service[svc] = {
            "latency_ms": lat, "error_rate": err, "cpu_pct": cpu, "memory_pct": mem,
            "log_level": levels, "message": messages, "true_category": labels,
        }

    # Overlay incidents: metrics drift towards the category's "peak" values.
    for category, svc, start, length, intensity in plan:
        data = per_service[svc]
        peaks = {m: rng.uniform(*rng_range) for m, rng_range in INCIDENT_PEAKS[category].items()}
        for k in range(length):
            i = start + k
            if i >= minutes:
                break
            ramp = np.sin(np.pi * (k + 0.5) / length) ** 0.5  # rises, peaks in the middle, eases off
            strength = float(np.clip(intensity * (0.4 + 0.6 * ramp) * rng.uniform(0.85, 1.0), 0, 1))
            for metric, peak in peaks.items():
                normal_value = data[metric][i]
                data[metric][i] = normal_value + (peak - normal_value) * strength
            data["log_level"][i] = "CRITICAL" if (strength > 0.8 and rng.random() < 0.5) else (
                "WARN" if (k == 0 and rng.random() < 0.5) else "ERROR")
            data["message"][i] = _fill(rng, str(rng.choice(INCIDENT_TEMPLATES[category])), svc)
            data["true_category"][i] = category

    frames = []
    times = BASE_TIME + pd.to_timedelta(np.arange(minutes), unit="m")
    for svc, data in per_service.items():
        frame = pd.DataFrame(data)
        frame.insert(0, "service", svc)
        frame.insert(0, "timestamp", times)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["timestamp", "service"]).reset_index(drop=True)
    df["latency_ms"] = df["latency_ms"].round(1)
    for col in ("error_rate", "cpu_pct", "memory_pct"):
        df[col] = df[col].round(2)
    return df


def generate_dataset(seed=TRAIN_SEED, minutes=1500, incidents_per_category=25):
    """Big labelled dataset used to train and test the models (has a `true_category` column)."""
    rng = np.random.default_rng(seed + 1000)
    plan = []
    for category in CATEGORIES:
        for _ in range(incidents_per_category):
            plan.append((
                category,
                str(rng.choice(SERVICES)),
                int(rng.integers(0, minutes - 12)),
                int(rng.integers(4, 11)),
                float(rng.uniform(0.4, 1.0)),
            ))
    return _build(seed, minutes, plan)


def generate_demo(seed=DEMO_SEED):
    """Small 3-hour dataset with six known incidents, used by the demo button."""
    return _build(seed, DEMO_MINUTES, DEMO_PLAN)


def save_csvs():
    """Write data/training_data.csv (with labels) and data/sample_upload.csv (upload-ready)."""
    DATA_DIR.mkdir(exist_ok=True)
    training = generate_dataset()
    training.to_csv(DATA_DIR / "training_data.csv", index=False)
    demo = generate_demo()
    demo[CSV_COLUMNS].to_csv(DATA_DIR / "sample_upload.csv", index=False)
    return len(training), len(demo)


if __name__ == "__main__":
    n_train, n_demo = save_csvs()
    print(f"Wrote {n_train} training rows and {n_demo} demo rows to {DATA_DIR}")
