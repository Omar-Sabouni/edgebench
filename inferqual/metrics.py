"""Small, dependency-light metric helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt


def summarize_latency(samples_ms: Sequence[float]) -> dict[str, Any]:
    """Round raw samples once, then derive their useful summaries."""
    samples = [round(float(value), 6) for value in samples_ms]
    if not samples or any(not np.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("latency samples must be finite and positive")
    p50, p95 = np.percentile(samples, [50, 95])
    return {
        "samples_ms": samples,
        "p50_ms": round(float(p50), 6),
        "p95_ms": round(float(p95), 6),
        "throughput_per_second": round(1000.0 / float(p50), 3),
    }


def fidelity(
    reference: npt.NDArray[np.float32],
    candidate: npt.NDArray[np.float32],
    *,
    precision: str,
) -> dict[str, Any]:
    """Compare candidate outputs with explicit FP32 and INT8 gates."""
    expected = np.asarray(reference, dtype=np.float64).ravel()
    actual = np.asarray(candidate, dtype=np.float64).ravel()
    if expected.shape != actual.shape or expected.size == 0:
        raise ValueError("candidate output shape does not match the reference")
    if not np.isfinite(actual).all():
        raise ValueError("candidate output contains non-finite values")

    delta = np.abs(expected - actual)
    denominator = float(np.linalg.norm(expected) * np.linalg.norm(actual))
    cosine = 1.0 if denominator == 0 and np.array_equal(expected, actual) else 0.0
    if denominator:
        cosine = float(np.dot(expected, actual) / denominator)
    max_limit, cosine_limit = (0.15, 0.98) if precision == "INT8" else (1e-4, 0.99999)
    max_error = float(delta.max())
    mean_error = float(delta.mean())
    return {
        "max_abs_error": round(max_error, 8),
        "mean_abs_error": round(mean_error, 8),
        "cosine_similarity": round(cosine, 8),
        "limits": {
            "max_abs_error": max_limit,
            "min_cosine_similarity": cosine_limit,
        },
        "passed": max_error <= max_limit and cosine >= cosine_limit,
    }


def select_fastest(results: Sequence[dict[str, Any]]) -> str:
    """Select the lowest-median backend among candidates that passed fidelity."""
    eligible = [result for result in results if result.get("status") == "passed"]
    if not eligible:
        raise ValueError("no backend passed the fidelity gate")
    return str(
        min(
            eligible,
            key=lambda result: (
                float(result["latency"]["p50_ms"]),
                float(result["latency"]["p95_ms"]),
                str(result["id"]),
            ),
        )["id"]
    )
