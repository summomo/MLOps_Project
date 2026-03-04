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


def _post_with_retries(
    endpoint: str,
    payload: dict,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> requests.Response:
    last_exception: requests.RequestException | None = None
    for attempt in range(1, retries + 1):
        try:
            return requests.post(endpoint, json=payload, timeout=timeout_seconds)
        except requests.RequestException as exc:
            last_exception = exc
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
    if last_exception is None:
        _fail("unexpected retry failure")
    _fail(f"network error after {retries} attempts: {last_exception}")


def main() -> None:
    staging_url = os.getenv("STAGING_URL", "").strip().rstrip("/")
    if not staging_url:
        _fail("STAGING_URL is missing")

    threshold_seconds = float(os.getenv("GATE_MAX_P95_SECONDS", "1.0"))
    warmup_timeout_seconds = float(os.getenv("GATE_WARMUP_TIMEOUT_SECONDS", "60"))
    request_timeout_seconds = float(os.getenv("GATE_REQUEST_TIMEOUT_SECONDS", "25"))
    request_retries = int(os.getenv("GATE_REQUEST_RETRIES", "3"))
    request_backoff_seconds = float(os.getenv("GATE_REQUEST_BACKOFF_SECONDS", "2"))
    texts = ["bonjour", "merci", "salut"]
    latencies: List[float] = []

    print(f"Running gates against: {staging_url}")
    print(f"Latency threshold (p95): {threshold_seconds:.3f}s")

    endpoint = f"{staging_url}/translate"

    print(
        "Warmup request: "
        f"timeout={warmup_timeout_seconds:.1f}s, retries={request_retries}, backoff={request_backoff_seconds:.1f}s"
    )
    warmup_response = _post_with_retries(
        endpoint=endpoint,
        payload={"text": "warmup"},
        timeout_seconds=warmup_timeout_seconds,
        retries=request_retries,
        backoff_seconds=request_backoff_seconds,
    )
    print(f"Warmup: status={warmup_response.status_code}")
    if warmup_response.status_code != 200:
        _fail(f"warmup returned non-200 status: {warmup_response.status_code}")

    for index, text in enumerate(texts, start=1):
        started_at = time.perf_counter()
        response = _post_with_retries(
            endpoint=endpoint,
            payload={"text": text},
            timeout_seconds=request_timeout_seconds,
            retries=request_retries,
            backoff_seconds=request_backoff_seconds,
        )

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
