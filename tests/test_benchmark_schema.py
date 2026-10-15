"""
Tests for Benchmark Schema compliance and serialization.
Verifies all required fields from Section 10 are present and properly formatted.
"""

from src.inference_bench.metrics import BenchmarkRecord


def test_benchmark_record_schema_fields():
    rec = BenchmarkRecord(
        experiment_id="exp_01",
        timestamp_utc="2026-10-15T00:00:00Z",
        engine="transformers",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        quantization="none",
        device="cpu",
        cpu_model="Intel(R) Core(TM) i3-5005U",
        ram_gb=4.0,
        gpu_model=None,
        vram_gb=None,
        prompt_tokens=12,
        generated_tokens=32,
        batch_size=1,
        concurrency=1,
        ttft_ms=120.5,
        median_itl_ms=45.2,
        p95_itl_ms=52.1,
        time_per_output_token_ms=46.0,
        end_to_end_ms=1550.0,
        output_tokens_per_second=20.6,
        process_peak_ram_mb=620.5,
        peak_vram_mb=None,
        cache_enabled=False,
        cache_hit=None,
        status="success",
        notes=None,
    )

    data = rec.to_dict()

    # Verify all mandatory schema keys exist
    expected_keys = [
        "experiment_id",
        "timestamp_utc",
        "engine",
        "model_id",
        "model_revision_or_hash",
        "quantization",
        "device",
        "cpu_model",
        "ram_gb",
        "gpu_model",
        "vram_gb",
        "prompt_tokens",
        "generated_tokens",
        "batch_size",
        "concurrency",
        "ttft_ms",
        "median_itl_ms",
        "p95_itl_ms",
        "time_per_output_token_ms",
        "end_to_end_ms",
        "output_tokens_per_second",
        "process_peak_ram_mb",
        "peak_vram_mb",
        "cache_enabled",
        "cache_hit",
        "status",
        "notes",
    ]

    for k in expected_keys:
        assert k in data, f"Missing required schema field: {k}"

    # Confirm nulls are retained instead of zero for unmeasured GPU
    assert data["gpu_model"] is None
    assert data["vram_gb"] is None
    assert data["peak_vram_mb"] is None
