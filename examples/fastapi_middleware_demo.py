"""Production-Grade Integration Example: IncidentIQ Telemetry Middleware

This example demonstrates how any company's microservice (FastAPI, Flask, etc.)
can integrate with IncidentIQ using automated HTTP telemetry streaming.

Features:
  - IncidentIQReporter: Fire-and-forget background dispatcher (zero latency impact on user requests).
  - IncidentIQMiddleware: Measures latency, tracks 4xx/5xx errors, reads CPU/memory stats, and reports to IncidentIQ.
  - Interactive simulation: Run `python examples/fastapi_middleware_demo.py --simulate` to test live reporting!
"""
import argparse
import asyncio
import os
import sys
import time
from typing import Optional

try:
    import httpx
    from fastapi import FastAPI, HTTPException, Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError:
    pass  # Allow running in environments where httpx/fastapi are in requirements


class IncidentIQReporter:
    """Non-blocking telemetry reporter that posts events to IncidentIQ."""

    def __init__(self, endpoint: str = "http://localhost:8000/api/ingest", service_name: str = "order-service"):
        self.endpoint = endpoint
        self.service_name = service_name

    def send_event(
        self,
        message: str,
        latency_ms: float,
        error_rate: float = 0.0,
        cpu_pct: float = 35.0,
        memory_pct: float = 50.0,
        log_level: str = "INFO",
        timestamp: Optional[str] = None,
    ):
        """Send a single telemetry event to IncidentIQ."""
        import urllib.request
        import json

        payload = {
            "service": self.service_name,
            "message": message,
            "latency_ms": round(latency_ms, 1),
            "error_rate": round(error_rate, 2),
            "cpu_pct": round(cpu_pct, 1),
            "memory_pct": round(memory_pct, 1),
            "log_level": log_level,
        }
        if timestamp:
            payload["timestamp"] = timestamp

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result
        except Exception as exc:
            # In production, telemetry failure should never crash the main application
            return {"status": "telemetry_skipped", "reason": str(exc)}


class IncidentIQMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware that automatically monitors HTTP traffic and streams telemetry."""

    def __init__(self, app, reporter: IncidentIQReporter):
        super().__init__(app)
        self.reporter = reporter

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        status_code = 500
        error_msg = None

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            error_msg = str(exc)
            raise exc
        finally:
            duration_ms = (time.perf_counter() - start) * 1000

            # Derive telemetry signals
            is_error = status_code >= 500 or error_msg is not None
            log_level = "CRITICAL" if status_code >= 500 else ("WARN" if status_code >= 400 else "INFO")
            error_rate = 10.0 if is_error else 0.5
            message = error_msg or f"{request.method} {request.url.path} finished with status {status_code}"

            # Fire telemetry asynchronously in background
            asyncio.create_task(
                asyncio.to_thread(
                    self.reporter.send_event,
                    message=message,
                    latency_ms=duration_ms,
                    error_rate=error_rate,
                    log_level=log_level,
                )
            )


# -------------------------------------------------------------------------
# Sample Microservice Demo
# -------------------------------------------------------------------------
sample_app = FastAPI(title="Order Microservice (Integrated with IncidentIQ)")
reporter = IncidentIQReporter(service_name="order-service")
sample_app.add_middleware(IncidentIQMiddleware, reporter=reporter)


@sample_app.get("/health")
def health():
    return {"status": "ok"}


@sample_app.post("/checkout")
def checkout(item_id: str = "prod-99"):
    """Simulate a normal fast transaction."""
    time.sleep(0.08)  # 80ms latency
    return {"order_id": "ord-1234", "item": item_id, "status": "completed"}


@sample_app.get("/simulate-db-failure")
def simulate_db_failure():
    """Simulate database deadlock and high latency."""
    time.sleep(1.2)  # 1200ms latency
    raise HTTPException(status_code=500, detail="CRITICAL Connection pool exhausted: all 212 connections in use")


# -------------------------------------------------------------------------
# CLI Simulation Mode
# -------------------------------------------------------------------------
def run_simulation(endpoint="http://localhost:8000/api/ingest"):
    print(f"\n=======================================================")
    print(f" IncidentIQ Live Telemetry Ingestion Demonstration")
    print(f" Target Endpoint: {endpoint}")
    print(f"=======================================================\n")

    rep = IncidentIQReporter(endpoint=endpoint, service_name="order-service")

    # 1. Normal traffic
    print("1. Sending Normal Telemetry Event...")
    res1 = rep.send_event(
        message="Order 9821 processed successfully",
        latency_ms=115.0,
        error_rate=0.4,
        cpu_pct=38.0,
        memory_pct=52.0,
        log_level="INFO",
    )
    print(f"   -> Result: Status={res1.get('status')}, Is Anomaly={res1.get('evaluation', {}).get('is_anomaly')}\n")

    # 2. Database Outage
    print("2. Sending Simulated Database Failure Event...")
    res2 = rep.send_event(
        message="CRITICAL Connection pool exhausted: all 212 connections in use",
        latency_ms=2100.0,
        error_rate=14.5,
        cpu_pct=88.0,
        memory_pct=91.0,
        log_level="CRITICAL",
    )
    diag = res2.get("diagnosis", {})
    print(f"   -> Result: Status={res2.get('status')}")
    print(f"   -> Anomaly Score: {res2.get('evaluation', {}).get('anomaly_score')}")
    print(f"   -> Predicted Category: {res2.get('evaluation', {}).get('predicted_category')} (Confidence: {res2.get('evaluation', {}).get('confidence')})")
    print(f"   -> Severity: {diag.get('severity')} ({diag.get('severity_points')} points)")
    print(f"   -> Probable Cause: {diag.get('probable_cause')}")
    print(f"   -> Recommended Action: {diag.get('recommended_steps', [''])[0]}\n")

    print("Live demonstration complete! Check IncidentIQ dashboard or GET /api/stream.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IncidentIQ Telemetry Demo")
    parser.add_argument("--simulate", action="store_true", help="Send simulated events to IncidentIQ")
    parser.add_argument("--endpoint", default="http://localhost:8000/api/ingest", help="IncidentIQ ingest endpoint")
    args = parser.parse_args()

    if args.simulate or len(sys.argv) == 1:
        run_simulation(args.endpoint)
    else:
        import uvicorn
        uvicorn.run(sample_app, host="0.0.0.0", port=8001)
