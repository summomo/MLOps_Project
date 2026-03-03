from __future__ import annotations

import os
import statistics
import sys
import time
from typing import List

import requests


def _fail(message: str) -> None:
    print(f"GATES FAILED: {message}")
    raise SystemExit(1)


def _p95_seconds(latencies: List[float]) -> float:
    if not latencies:
        return 0.0
    if len(latencies) < 20:
        return max(latencies)
    quantiles = statistics.quantiles(latencies, n=100, method="inclusive")
    return quantiles[94]


def main() -> None:
    staging_url = os.getenv("STAGING_URL", "").strip().rstrip("/")
    if not staging_url:
        _fail("STAGING_URL is missing")

    threshold_seconds = float(os.getenv("GATE_MAX_P95_SECONDS", "1.0"))
    texts = ["bonjour", "merci", "salut"]
    latencies: List[float] = []

    print(f"Running gates against: {staging_url}")
    print(f"Latency threshold (p95): {threshold_seconds:.3f}s")

    endpoint = f"{staging_url}/translate"

    for index, text in enumerate(texts, start=1):
        started_at = time.perf_counter()
        try:
            response = requests.post(endpoint, json={"text": text}, timeout=15)
        except requests.RequestException as exc:
            _fail(f"request {index} failed with network error: {exc}")

        elapsed = time.perf_counter() - started_at
        latencies.append(elapsed)

        print(f"Request {index}: status={response.status_code}, latency_ms={elapsed * 1000:.2f}")

        if response.status_code != 200:
            _fail(f"request {index} returned non-200 status: {response.status_code}")

        try:
            payload = response.json()
        except ValueError:
            _fail(f"request {index} did not return valid JSON")

        translation = payload.get("translation")
        if not isinstance(translation, str) or not translation.strip():
            _fail(f"request {index} returned empty or missing 'translation'")

    p95 = _p95_seconds(latencies)
    print(f"Computed p95 latency: {p95 * 1000:.2f} ms")

    if p95 > threshold_seconds:
        _fail(
            f"latency gate failed: p95={p95 * 1000:.2f} ms exceeds threshold={threshold_seconds * 1000:.2f} ms"
        )

    print("GATES PASSED")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
