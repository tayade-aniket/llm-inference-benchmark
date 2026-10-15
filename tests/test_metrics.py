"""
Unit tests for metric calculations (TTFT, ITL, throughput, time per output token).
Runs model-free in milliseconds.
"""

import pytest
import numpy as np
from src.inference_bench.metrics import MetricsCalculator, BenchmarkRecord


def test_metrics_calculation_standard_stream():
    # Simulated request:
    # Request launched at t = 10.0
    # Token 1 arrived at t = 10.200 (TTFT = 200 ms)
    # Token 2 arrived at t = 10.250 (ITL 1 = 50 ms)
    # Token 3 arrived at t = 10.300 (ITL 2 = 50 ms)
    # Token 4 arrived at t = 10.360 (ITL 3 = 60 ms)
    # Request end at t = 10.360 (End-to-End = 360 ms)
    req_start = 10.000
    token_timestamps = [10.200, 10.250, 10.300, 10.360]
    req_end = 10.360
    prompt_tokens = 15

    results = MetricsCalculator.compute(
        request_start_s=req_start,
        token_timestamps_s=token_timestamps,
        request_end_s=req_end,
        prompt_token_count=prompt_tokens,
    )

    assert results["prompt_tokens"] == 15
    assert results["generated_tokens"] == 4
    # TTFT: 10.200 - 10.000 = 0.200 s = 200.0 ms
    assert results["ttft_ms"] == pytest.approx(200.0, rel=1e-2)
    # ITL values: [50 ms, 50 ms, 60 ms] -> median = 50.0 ms
    assert results["median_itl_ms"] == pytest.approx(50.0, rel=1e-2)
    assert results["p95_itl_ms"] >= 50.0
    # Time per output token: steady state (10.360 - 10.200) / (4 - 1) = 0.160 / 3 = 53.33 ms
    assert results["time_per_output_token_ms"] == pytest.approx(53.33, rel=1e-2)
    # End-to-end: 360.0 ms
    assert results["end_to_end_ms"] == pytest.approx(360.0, rel=1e-2)
    # Throughput: 4 / 0.360 s = 11.11 tokens/s
    assert results["output_tokens_per_second"] == pytest.approx(11.11, rel=1e-2)


def test_metrics_calculation_single_token():
    # Only 1 token generated (e.g. stop token or classification)
    req_start = 1.000
    token_timestamps = [1.100]
    req_end = 1.100
    prompt_tokens = 8

    results = MetricsCalculator.compute(
        request_start_s=req_start,
        token_timestamps_s=token_timestamps,
        request_end_s=req_end,
        prompt_token_count=prompt_tokens,
    )

    assert results["generated_tokens"] == 1
    assert results["ttft_ms"] == pytest.approx(100.0, rel=1e-2)
    # No inter-token latency for single token
    assert results["median_itl_ms"] is None
    assert results["p95_itl_ms"] is None
    # No steady state output token interval
    assert results["time_per_output_token_ms"] is None
    assert results["output_tokens_per_second"] == pytest.approx(10.0, rel=1e-2)


def test_metrics_calculation_zero_tokens_error():
    # OOM or failed request with 0 tokens generated
    req_start = 1.000
    token_timestamps = []
    req_end = 1.050
    prompt_tokens = 10

    results = MetricsCalculator.compute(
        request_start_s=req_start,
        token_timestamps_s=token_timestamps,
        request_end_s=req_end,
        prompt_token_count=prompt_tokens,
    )

    assert results["generated_tokens"] == 0
    assert results["ttft_ms"] is None
    assert results["median_itl_ms"] is None
    assert results["output_tokens_per_second"] == 0.0
