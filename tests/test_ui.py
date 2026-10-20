"""
Unit tests for LLM Inference UI helpers, data loaders, and analytical modeling.
"""

import os
import pytest
import pandas as pd

from app import (
    load_benchmark_data,
    compute_analytical_model,
    get_live_system_metrics,
)


def test_load_benchmark_data():
    """Validates that benchmark data loader parses real result files without error."""
    raw_df, summary_df, env_audit = load_benchmark_data()
    assert isinstance(raw_df, pd.DataFrame)
    assert isinstance(summary_df, pd.DataFrame)
    assert isinstance(env_audit, dict)

    # If results exist, check critical schema columns
    if not summary_df.empty:
        assert "engine" in summary_df.columns
        assert "quantization" in summary_df.columns
        assert "output_tokens_per_second_median" in summary_df.columns


def test_compute_analytical_model_edge_cpu():
    """Validates roofline and analytical modeling for 0.5B model on dual-core CPU."""
    res = compute_analytical_model(
        param_count_b=0.49,
        quant_bits=4.5,
        bandwidth_gb_s=16.0,
        peak_gflops=64.0,
        prompt_tokens=24,
        output_tokens=32,
    )

    # Active model size = 0.49 * (4.5 / 8) = 0.2756 GB (~282 MB)
    assert 0.27 <= res["active_model_size_gb"] <= 0.28
    assert res["active_model_size_mb"] > 270

    # Arithmetic intensity = 16 / 4.5 = 3.555... FLOPs/byte
    assert 3.5 <= res["intensity"] <= 3.6

    # Roofline knee = 64 / 16 = 4.0 FLOPs/byte
    assert res["roofline_knee"] == 4.0

    # Because intensity (3.56) < knee (4.0), decoding MUST be memory-bandwidth bound
    assert res["is_memory_bound"] is True
    assert res["operational_regime"] == "Memory-Bandwidth Bound"

    # Achievable tokens/s: bandwidth (16 GB/s) / model_size (0.276 GB) ~ 58 tok/s theoretical bus limit
    assert res["achievable_tok_s"] > 0
    assert res["itl_ms"] > 0
    assert res["ttft_ms"] > 0

    # Memory footprint should be safe for 4GB system (<= 850 MB)
    assert res["safe_for_4gb"] is True
    assert res["peak_ram_mb"] < 600.0


def test_compute_analytical_model_large_model_oom():
    """Validates that a 70B model correctly triggers memory limit flag on edge hardware."""
    res = compute_analytical_model(
        param_count_b=70.6,
        quant_bits=16.0,
        bandwidth_gb_s=16.0,
        peak_gflops=64.0,
        prompt_tokens=24,
        output_tokens=32,
    )
    # Model size alone is ~141 GB
    assert res["active_model_size_gb"] > 140
    # Must fail 4GB edge budget
    assert res["safe_for_4gb"] is False


def test_get_live_system_metrics():
    """Validates live psutil system inspection."""
    metrics = get_live_system_metrics()
    assert metrics["total_ram_gb"] > 0
    assert metrics["available_ram_gb"] > 0
    assert metrics["cores_logical"] >= 1
    assert 0 <= metrics["used_ram_percent"] <= 100


def test_ui_static_assets_exist():
    """Verifies that the standalone web showcase files are present and contain key visual proof features."""
    assert os.path.isfile("ui/index.html")
    assert os.path.isfile("ui/style.css")
    assert os.path.isfile("ui/app.js")

    with open("ui/index.html", "r", encoding="utf-8") as f:
        html = f.read()
        assert "LLM Inference Study" in html
        assert "Dynamic Roofline Performance Model" in html or "Roofline Performance Model" in html
        assert "Head-to-Head Token Generation Race" in html
        assert "Prompt & Prefix Caching Visualizer" in html
        assert "Memory Safety & Working RAM Budget" in html
        assert "llama.cpp (Q4_K_M AVX2)" in html

    with open("ui/app.js", "r", encoding="utf-8") as f:
        js = f.read()
        assert "initHeadToHeadRace" in js
        assert "initPromptCachingVisualizer" in js
        assert "updateRooflineChart" in js


def test_streamlit_app_has_visual_proof_features():
    """Verifies that app.py contains the visual proof sections and race simulator."""
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()
        assert "Start Head-to-Head Token Race" in code
        assert "Interactive Prompt & Prefix Caching Visualizer" in code
        assert "Memory Safety & Process Budget" in code
        assert "Dynamic Roofline Model" in code or "Interactive Roofline Model" in code
        assert "Executive Verification Sheet" in code


