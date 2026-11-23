"""Incident intelligence: combines ML output with simple, readable rules.

What comes from ML (predictions):
  - anomaly score / "unusual or not"   (Isolation Forest)
  - problem category + confidence      (TF-IDF + Logistic Regression)

What comes from rules (plain Python you can read and change):
  - which rows count as "events"
  - grouping events into incidents
  - severity (a points system)
  - probable root cause, evidence and recommended steps

A probable cause is a *suggestion based on evidence*, never a confirmed cause.
"""
import numpy as np
import pandas as pd

from app.ml import FEATURES, classify_messages, detect_anomalies

# ---------------------------------------------------------------- settings you can tune
THRESHOLDS = {"latency_ms": 800.0, "error_rate": 5.0, "cpu_pct": 90.0, "memory_pct": 90.0}
LABELS = {"latency_ms": "Latency", "error_rate": "Error rate", "cpu_pct": "CPU", "memory_pct": "Memory"}
UNITS = {"latency_ms": "ms", "error_rate": "%", "cpu_pct": "%", "memory_pct": "%"}
GROUP_GAP_MINUTES = 10      # events from one service closer than this belong to the same incident
ACTIVE_WINDOW_MINUTES = 15  # an incident that ended this recently is "still active"
LOW_CONFIDENCE = 0.40       # below this the classifier is too unsure to name a category
SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LEVEL_RANK = {"INFO": 0, "WARN": 1, "ERROR": 2, "CRITICAL": 3}
HEALTH_PENALTY = {"CRITICAL": 35, "HIGH": 25, "MEDIUM": 12, "LOW": 5}

DISCLAIMER = ("This is a probable cause suggested by simple rules from the evidence below. "
              "It is not a confirmed cause - verify it before acting.")

# ------------------------------------------------- root-cause rules, one list per category
# Each rule: (keywords to look for in the log text, probable cause, first action to try)
RULES = {
    "database": [
        (("deadlock", "acquire lock"), "Competing transactions are probably blocking each other (lock contention).",
         "Look for long-running transactions and lock waits, and stop the blocking session if it is safe."),
        (("connection pool", "too many connections"), "The database connection pool is probably exhausted, so new requests wait or fail.",
         "Check how many connections are open and which service holds them; raise the pool limit or fix connection leaks."),
        (("slow query", "query timeout"), "One or more slow queries are probably holding the database busy.",
         "Find the slowest queries and check whether the table has the right index."),
        (("replication",), "A database replica is probably falling behind the primary.",
         "Check replica health and network between primary and replica; route reads to the primary meanwhile."),
    ],
    "network": [
        (("dns",), "DNS lookups are probably failing, so the service cannot find its upstream hosts.",
         "Test name resolution from the affected host and check the DNS server or recent DNS changes."),
        (("tls", "handshake"), "Secure (TLS) connections to an upstream are probably failing.",
         "Check certificate expiry and TLS settings on the upstream service."),
        (("packet loss", "reset by peer", "no route"), "The network path between services is probably unstable.",
         "Check the network interface and firewall/route changes; test connectivity with ping or traceroute."),
        (("timeout", "bad gateway"), "An upstream service is probably slow or unreachable.",
         "Check the health of the upstream service and any load balancer in front of it."),
    ],
    "application": [
        (("nullpointer", "keyerror", "typeerror", "exception", "stack trace"),
         "A code bug in a recent request path is probably raising unhandled exceptions.",
         "Read the stack trace, find the failing code path and check what changed in the code recently."),
        (("validation", "json decode"), "The service is probably receiving data in an unexpected format.",
         "Compare the failing request payloads with what the code expects; check the calling service."),
        (("500", "timed out"), "A request handler is probably failing or hanging.",
         "Check the application logs for the failing endpoint and test it manually."),
    ],
    "deployment": [
        (("rolling back", "rollback", "health check", "readiness", "canary"),
         "The newest release probably failed its health checks.",
         "Roll back to the previous version, then compare the two releases."),
        (("environment variable", "config"), "A setting is probably missing or wrong after the release.",
         "Compare the deployed configuration/environment variables with the last working release."),
        (("image", "container failed"), "The container image probably could not be pulled or started.",
         "Check that the image tag exists in the registry and that the registry credentials are valid."),
        (("migration",), "A database migration probably failed during the deployment.",
         "Check the migration logs and whether the database schema is half-updated."),
    ],
    "resource": [
        (("memory", "oom", "heap", "gc overhead"), "Memory is probably running out (possible memory leak or limit set too low).",
         "Check memory trends and restart the service if needed; look for a leak or raise the memory limit."),
        (("cpu",), "The CPU is probably saturated.",
         "Find the busiest processes; scale out or add CPU if the load is genuine."),
        (("disk",), "The disk is probably almost full.",
         "Clean old logs/temp files or expand the disk."),
        (("thread", "file descriptor"), "The service is probably running out of threads or file handles.",
         "Check for leaked connections/files and raise the OS or pool limits if needed."),
    ],
}

DEFAULT_CAUSE = {
    "database": "The database is probably responding slowly or refusing connections.",
    "network": "A network or upstream problem is probably delaying or dropping requests.",
    "application": "The application code is probably failing on some requests.",
    "deployment": "A recent release or configuration change is probably causing failures.",
    "resource": "The service is probably running short of CPU, memory or other resources.",
}

# Predefined troubleshooting steps per category (the matching rule's action is added in front).
STEPS = {
    "database": ["Check database CPU, connections and slow-query log.",
                 "Look at recent schema or query changes.",
                 "Confirm the database host and disk are healthy."],
    "network": ["Check connectivity between the service and its upstream hosts.",
                "Look at recent firewall, DNS or load-balancer changes.",
                "Compare with other services - if they are slow too, the network is likely involved."],
    "application": ["Open the application logs around the incident time.",
                    "Check what code was released recently.",
                    "Reproduce the failing request in a test environment."],
    "deployment": ["Find out which release went out just before the incident.",
                   "Roll back to the last good version if errors continue.",
                   "Check the deployment pipeline logs for failed steps."],
    "resource": ["Check CPU, memory and disk graphs for the affected service.",
                 "Restart or scale the service if the system is starved.",
                 "Look for a memory leak or runaway job."],
    "unclassified": ["Open the raw logs around the incident time.",
                     "Check for recent releases or configuration changes.",
                     "Compare the metrics of this service with other services."],
}


# ---------------------------------------------------------------- severity (points system)
def severity_from_points(total):
    """Turn the points total into a severity label."""
    if total >= 8:
        return "CRITICAL"
    if total >= 6:
        return "HIGH"
    if total >= 4:
        return "MEDIUM"
    return "LOW"


def score_severity(peaks, max_score, score_threshold, duration_minutes, levels):
    """Add up transparent points. Returns (severity, [{points, reason}, ...])."""
    items = []
    if max_score >= score_threshold:
        items.append((1, "The ML model flagged the metrics as unusual"))
    if max_score >= score_threshold + 0.10:
        items.append((1, "The anomaly score is far above the normal range"))
    for feature, limit in THRESHOLDS.items():
        if peaks[feature] >= limit:
            items.append((1, f"{LABELS[feature]} reached {peaks[feature]:.0f}{UNITS[feature]} (limit {limit:.0f}{UNITS[feature]})"))
    if "CRITICAL" in levels:
        items.append((2, "A CRITICAL log line was written"))
    elif "ERROR" in levels:
        items.append((1, "ERROR log lines were written"))
    if duration_minutes >= 10:
        items.append((2, f"The problem lasted {duration_minutes} minutes"))
    elif duration_minutes >= 5:
        items.append((1, f"The problem lasted {duration_minutes} minutes"))
    total = sum(p for p, _ in items)
    return severity_from_points(total), [{"points": p, "reason": r} for p, r in items]


# ---------------------------------------------------------------- root cause + evidence
def explain(category, confidence, runner_up, peaks, max_score, n_logs, text, info):
    """Rule-based explanation. Returns (probable_cause, evidence_list, recommendations)."""
    baseline = info["anomaly_model"]["baseline"]
    threshold = info["anomaly_model"]["score_threshold"]
    evidence = [{"source": "ML", "text": f"Isolation Forest anomaly score {max_score:.2f} "
                                          f"(flagged above {threshold:.2f}; higher = more unusual)."}]

    matched_action = None
    if category == "unclassified":
        cause = ("The metrics look unusual, but the log messages do not clearly match any known problem "
                 "type, so no probable cause is suggested.")
        evidence.append({"source": "ML", "text": f"The log classifier was not confident (best guess only {confidence:.0%})."})
    else:
        text_note = f"The log classifier predicts '{category}' with {confidence:.0%} average confidence over {n_logs} log line(s)."
        if runner_up:
            text_note += f" Second guess: '{runner_up[0]}' ({runner_up[1]:.0%})."
        evidence.append({"source": "ML", "text": text_note})

        cause = DEFAULT_CAUSE[category]
        for keywords, rule_cause, action in RULES[category]:
            hit = next((k for k in keywords if k in text), None)
            if hit:
                cause, matched_action = rule_cause, action
                evidence.append({"source": "Rule", "text": f"The log text contains '{hit}', a typical sign of a {category} problem."})
                break

    # Metric evidence: compare each peak with the normal baseline learned from training data.
    for feature in FEATURES:
        mean, std = baseline[feature]["mean"], baseline[feature]["std"]
        value = peaks[feature]
        breached = value >= THRESHOLDS[feature]
        if breached or value > mean + 3 * std:
            tag = "Rule" if breached else "Data"
            limit = f" - above the {THRESHOLDS[feature]:.0f}{UNITS[feature]} limit" if breached else ""
            evidence.append({"source": tag, "text": f"{LABELS[feature]} peaked at {value:.1f}{UNITS[feature]} "
                                                   f"(normal average is about {mean:.0f}{UNITS[feature]}){limit}."})

    # Cross-checks between the ML category and the raw numbers.
    resource_high = peaks["cpu_pct"] >= THRESHOLDS["cpu_pct"] or peaks["memory_pct"] >= THRESHOLDS["memory_pct"]
    if category not in ("resource", "unclassified") and resource_high:
        evidence.append({"source": "Rule", "text": "CPU or memory was also very high, so resource pressure may be contributing."})
    if category == "resource" and not resource_high:
        evidence.append({"source": "Rule", "text": "Caution: the log says 'resource', but CPU and memory stayed below their limits."})
    if category == "network" and peaks["cpu_pct"] < 60 and peaks["latency_ms"] >= THRESHOLDS["latency_ms"]:
        evidence.append({"source": "Rule", "text": "Latency is high while CPU is normal, which points away from local overload."})

    steps = ([matched_action] if matched_action else []) + STEPS[category]
    return cause, evidence, steps


# ---------------------------------------------------------------- main pipeline
def find_events(df, bundle):
    """Score every row and mark which ones are 'events' worth looking at."""
    df = df.copy()
    scores, flags = detect_anomalies(bundle, df)
    df["anomaly_score"] = scores
    df["is_anomaly"] = flags
    breach = pd.DataFrame({f: df[f] >= limit for f, limit in THRESHOLDS.items()}).any(axis=1)
    df["is_event"] = df["is_anomaly"] | breach | df["log_level"].isin(["ERROR", "CRITICAL"])
    return df


def group_events(events):
    """Group events of the same service that are close in time (default: within 10 minutes)."""
    events = events.sort_values(["service", "timestamp"]).copy()
    gap = events.groupby("service")["timestamp"].diff()
    starts_new = gap.isna() | (gap > pd.Timedelta(minutes=GROUP_GAP_MINUTES))
    events["group"] = starts_new.cumsum()
    return events


def drop_blips(events):
    """Ignore one-off blips: a single odd minute with no ERROR log and no limit breached.

    Real systems have harmless spikes all the time. Showing each as an incident would bury the
    real problems, so a group is kept only if it has 2+ events or at least one serious sign.
    """
    if events.empty:
        return events
    serious = events["log_level"].isin(["ERROR", "CRITICAL"])
    for feature, limit in THRESHOLDS.items():
        serious = serious | (events[feature] >= limit)
    size = events.groupby("group")["group"].transform("size")
    has_serious = serious.groupby(events["group"]).transform("any")
    return events[(size >= 2) | has_serious]


def build_incidents(events, bundle):
    """Turn grouped events into incident dictionaries (the thing the dashboard shows)."""
    if events.empty:
        return []
    info = bundle["info"]
    score_threshold = info["anomaly_model"]["score_threshold"]
    classes, proba = classify_messages(bundle, events["message"])  # ML: log text -> category
    events = events.reset_index(drop=True)
    incidents = []

    for _, idx in events.groupby("group").groups.items():
        g = events.loc[idx]
        p = proba[list(idx)]
        # Prefer WARN/ERROR/CRITICAL lines as evidence for the category.
        loud = g["log_level"].isin(["WARN", "ERROR", "CRITICAL"]).to_numpy()
        avg = (p[loud] if loud.any() else p).mean(axis=0)
        order = np.argsort(avg)[::-1]
        category, confidence = str(classes[order[0]]), float(avg[order[0]])
        runner_up = (str(classes[order[1]]), float(avg[order[1]])) if avg[order[1]] >= 0.2 else None
        if confidence < LOW_CONFIDENCE:
            category = "unclassified"

        start, end = g["timestamp"].min(), g["timestamp"].max()
        duration = int((end - start).total_seconds() // 60) + 1
        peaks = {f: float(g[f].max()) for f in FEATURES}
        max_score = float(g["anomaly_score"].max())
        severity, severity_points = score_severity(
            peaks, max_score, score_threshold, duration, set(g["log_level"]))
        cause, evidence, steps = explain(
            category, confidence, runner_up, peaks, max_score, int(loud.sum() or len(g)),
            " ".join(g["message"].str.lower()), info)
        evidence.append({"source": "Data", "text": f"{len(g)} related event(s) from {g['service'].iloc[0]} "
                                                    f"between {start:%H:%M} and {end:%H:%M} were grouped into this incident."})

        # The "original log message" shown on the page = the most serious line (highest level, then score).
        ranked = g.assign(rank=g["log_level"].map(LEVEL_RANK)).sort_values(["rank", "anomaly_score"], ascending=False)
        main_row = ranked.iloc[0]
        kind = f"a {category} problem" if category != "unclassified" else "a problem it cannot clearly classify"
        summary = (f"{g['service'].iloc[0]} behaved unusually for {duration} minute(s). "
                   f"The system thinks this is probably {kind}, with {severity} severity.")

        event_rows = g.sort_values("timestamp").head(60)
        incidents.append({
            "service": str(g["service"].iloc[0]),
            "start": start.strftime("%Y-%m-%d %H:%M"), "end": end.strftime("%Y-%m-%d %H:%M"),
            "_start": start, "_end": end,
            "duration_minutes": duration, "event_count": int(len(g)),
            "category": category, "category_confidence": round(confidence, 3),
            "runner_up": ({"category": runner_up[0], "confidence": round(runner_up[1], 3)} if runner_up else None),
            "severity": severity, "severity_points": severity_points,
            "peaks": {f: round(v, 1) for f, v in peaks.items()},
            "max_anomaly_score": round(max_score, 3),
            "anomaly_threshold": score_threshold,
            "log_message": str(main_row["message"]), "log_level": str(main_row["log_level"]),
            "log_time": main_row["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "summary": summary, "probable_cause": cause, "cause_disclaimer": DISCLAIMER,
            "evidence": evidence, "recommendations": steps,
            "events": [{
                "timestamp": r["timestamp"].strftime("%Y-%m-%d %H:%M"), "log_level": r["log_level"],
                "message": r["message"], "latency_ms": round(float(r["latency_ms"]), 1),
                "error_rate": round(float(r["error_rate"]), 2), "cpu_pct": round(float(r["cpu_pct"]), 1),
                "memory_pct": round(float(r["memory_pct"]), 1),
                "anomaly_score": round(float(r["anomaly_score"]), 3), "is_anomaly": bool(r["is_anomaly"]),
            } for _, r in event_rows.iterrows()],
        })

    incidents.sort(key=lambda i: i["_start"])
    for number, incident in enumerate(incidents, start=1):
        incident["id"] = f"INC-{number:03d}"
    return incidents


def build_timeseries(scored, max_points=120):
    """Charts show the WORST value among all services in each time bucket (so spikes stand out)."""
    d = scored.set_index("timestamp").sort_index()
    span_minutes = (d.index.max() - d.index.min()).total_seconds() / 60
    step = max(1, int(np.ceil(span_minutes / max_points)))
    d = d.assign(anomalies=d["is_anomaly"].astype(int))
    buckets = d.resample(f"{step}min").agg(
        {"latency_ms": "max", "error_rate": "max", "cpu_pct": "max", "memory_pct": "max", "anomalies": "sum"}
    ).dropna()
    return [{"t": ts.strftime("%H:%M"), "date": ts.strftime("%Y-%m-%d"),
             "latency_ms": round(float(r["latency_ms"]), 1), "error_rate": round(float(r["error_rate"]), 2),
             "cpu_pct": round(float(r["cpu_pct"]), 1), "memory_pct": round(float(r["memory_pct"]), 1),
             "anomalies": int(r["anomalies"])} for ts, r in buckets.iterrows()]


def build_summary(scored, incidents):
    """Numbers for the top of the dashboard: health score, counts, and per-service status."""
    data_end = scored["timestamp"].max()
    cutoff = data_end - pd.Timedelta(minutes=ACTIVE_WINDOW_MINUTES)
    active = [i for i in incidents if i["_end"] >= cutoff]
    score = max(0, 100 - sum(HEALTH_PENALTY[i["severity"]] for i in active))
    status = "Healthy" if score >= 90 else ("Degraded" if score >= 60 else "Critical")

    services = []
    for name, g in scored.groupby("service"):
        worst = max((i["severity"] for i in active if i["service"] == name),
                    key=SEVERITY_ORDER.index, default=None)
        state = "healthy" if worst is None else ("critical" if worst in ("HIGH", "CRITICAL") else "degraded")
        services.append({
            "service": str(name), "status": state,
            "avg_latency_ms": round(float(g["latency_ms"].mean()), 1),
            "avg_error_rate": round(float(g["error_rate"].mean()), 2),
            "avg_cpu_pct": round(float(g["cpu_pct"].mean()), 1),
            "avg_memory_pct": round(float(g["memory_pct"].mean()), 1),
            "incidents": sum(1 for i in incidents if i["service"] == name),
        })
    return {
        "rows_analyzed": int(len(scored)),
        "time_start": scored["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
        "time_end": data_end.strftime("%Y-%m-%d %H:%M"),
        "anomalous_rows": int(scored["is_anomaly"].sum()),
        "total_incidents": len(incidents),
        "active_incidents": len(active),
        "severity_counts": {s: sum(1 for i in incidents if i["severity"] == s) for s in SEVERITY_ORDER},
        "health_score": int(score), "health_status": status,
        "services": services,
    }


def analyze(df, bundle, notes=None):
    """Full pipeline: score rows -> pick events -> group -> build incidents -> summary."""
    scored = find_events(df, bundle)
    events = drop_blips(group_events(scored[scored["is_event"]]))
    incidents = build_incidents(events, bundle)
    result = {
        "summary": build_summary(scored, incidents),
        "timeseries": build_timeseries(scored),
        "thresholds": {f: {"label": LABELS[f], "unit": UNITS[f], "limit": limit} for f, limit in THRESHOLDS.items()},
        "incidents": incidents,
        "data_quality_notes": notes or [],
    }
    return result


def public_incidents(incidents):
    """Remove internal fields (those starting with an underscore) before sending JSON."""
    return [{k: v for k, v in i.items() if not k.startswith("_")} for i in incidents]


def evaluate_single_event(event: dict, bundle: dict) -> dict:
    """Evaluate one live telemetry reading in real time using both ML models and rules."""
    info = bundle["info"]
    score_threshold = float(info["anomaly_model"]["score_threshold"])

    # 1. Prepare numeric features for Isolation Forest
    df_single = pd.DataFrame([{
        "latency_ms": float(event["latency_ms"]),
        "error_rate": float(event["error_rate"]),
        "cpu_pct": float(event["cpu_pct"]),
        "memory_pct": float(event["memory_pct"]),
    }])
    scores, flags = detect_anomalies(bundle, df_single)
    anomaly_score = float(scores[0])
    is_anomaly = bool(flags[0])

    # 2. Log classification with TF-IDF + Logistic Regression
    classes, proba = classify_messages(bundle, [str(event.get("message", ""))])
    order = np.argsort(proba[0])[::-1]
    best_cat, confidence = str(classes[order[0]]), float(proba[0][order[0]])
    runner_up = (str(classes[order[1]]), float(proba[0][order[1]])) if len(classes) > 1 else None

    # 3. Check threshold limits
    threshold_breaches = []
    peaks = {}
    for feat, limit in THRESHOLDS.items():
        val = float(event[feat])
        peaks[feat] = val
        if val >= limit:
            threshold_breaches.append(f"{LABELS[feat]} {val:.1f}{UNITS[feat]} >= {limit:.0f}{UNITS[feat]}")

    log_level = str(event.get("log_level", "INFO")).upper()
    is_event = is_anomaly or len(threshold_breaches) > 0 or log_level in ("ERROR", "CRITICAL")

    if is_event:
        category = "unclassified" if confidence < LOW_CONFIDENCE else best_cat
        severity, points_breakdown = score_severity(
            peaks=peaks,
            max_score=anomaly_score,
            score_threshold=score_threshold,
            duration_minutes=1,
            levels=[log_level],
        )
        cause, evidence, steps = explain(
            category=category,
            confidence=confidence,
            runner_up=runner_up,
            peaks=peaks,
            max_score=anomaly_score,
            n_logs=1,
            text=str(event.get("message", "")),
            info=info,
        )
        status = "incident_triggered" if severity in ("HIGH", "CRITICAL") else "anomaly_detected"
        total_points = sum(p["points"] for p in points_breakdown)
    else:
        category = best_cat
        severity = "LOW"
        total_points = 0
        points_breakdown = []
        cause = "Metrics and logs are within normal operational baselines."
        evidence = [{"source": "ML", "text": f"Anomaly score {anomaly_score:.2f} is within normal range (threshold {score_threshold:.2f})."}]
        steps = ["System operating normally. No intervention needed."]
        status = "normal"

    return {
        "status": status,
        "service": event.get("service", "unknown"),
        "timestamp": event.get("timestamp"),
        "evaluation": {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "score_threshold": round(score_threshold, 4),
            "threshold_breaches": threshold_breaches,
            "predicted_category": category,
            "confidence": round(confidence, 4),
        },
        "diagnosis": {
            "severity": severity,
            "severity_points": total_points,
            "points_breakdown": points_breakdown,
            "probable_cause": cause,
            "evidence": evidence,
            "recommended_steps": steps,
        },
    }
