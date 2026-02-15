import json
import time
from pathlib import Path
from typing import Dict

# Store logs inside artifacts so docker volume mount can persist them
LOG_DIR = Path("artifacts/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
PRED_LOG = LOG_DIR / "predictions.jsonl"

# Simple in-memory counters (fine for this project)
METRICS: Dict[str, float] = {
    "requests_total": 0,
    "errors_total": 0,
    "fraud_flagged_total": 0,
    "latency_ms_sum": 0,
}

APP_START = time.time()

def log_prediction(payload: dict) -> None:
    # JSONL = one JSON per line
    with PRED_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

def record_request(latency_ms: float, is_error: bool, is_fraud: bool) -> None:
    METRICS["requests_total"] += 1
    METRICS["latency_ms_sum"] += latency_ms
    if is_error:
        METRICS["errors_total"] += 1
    if is_fraud:
        METRICS["fraud_flagged_total"] += 1

def uptime_seconds() -> int:
    return int(time.time() - APP_START)

def prometheus_metrics() -> str:
    # minimal Prometheus format
    avg_latency = 0.0
    if METRICS["requests_total"] > 0:
        avg_latency = METRICS["latency_ms_sum"] / METRICS["requests_total"]

    lines = [
        f"requests_total {int(METRICS['requests_total'])}",
        f"errors_total {int(METRICS['errors_total'])}",
        f"fraud_flagged_total {int(METRICS['fraud_flagged_total'])}",
        f"avg_latency_ms {avg_latency:.3f}",
        f"uptime_seconds {uptime_seconds()}",
    ]
    return "\n".join(lines) + "\n"
