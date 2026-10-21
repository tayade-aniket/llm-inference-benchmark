"""
Unit tests for universal system detection and dynamic auto-configuration.
"""

import pytest
from src.inference_bench.system_detector import (
    detect_cpu_brand,
    get_cpu_short_name,
    detect_simd_flags,
    detect_gpu_info,
    compute_safe_process_budget,
    estimate_roofline_defaults,
    get_recommended_model_tier,
    detect_system_specs,
)
from src.inference_bench.config import auto_configure_hardware, load_local_config


def test_detect_cpu_brand():
    brand = detect_cpu_brand()
    assert isinstance(brand, str)
    assert len(brand) > 0


def test_get_cpu_short_name():
    assert get_cpu_short_name("Intel(R) Core(TM) i3-5005U CPU @ 2.00GHz") == "Intel Core i3-5005U"
    assert get_cpu_short_name("AMD Ryzen 7 5800X 8-Core Processor") == "AMD Ryzen 7 5800X 8-Core"
    assert get_cpu_short_name("Apple M2 Pro") == "Apple M2 Pro"


def test_detect_simd_flags():
    flags = detect_simd_flags()
    assert isinstance(flags, list)
    assert len(flags) > 0


def test_detect_gpu_info():
    gpu = detect_gpu_info()
    assert isinstance(gpu, dict)
    assert "has_gpu" in gpu
    assert "label" in gpu
    assert "type" in gpu


def test_compute_safe_process_budget():
    # 4GB system (edge)
    budget_4gb = compute_safe_process_budget(total_ram_gb=3.9, available_ram_gb=1.0)
    assert budget_4gb <= 850

    # 16GB system (laptop)
    budget_16gb = compute_safe_process_budget(total_ram_gb=16.0, available_ram_gb=10.0)
    assert budget_16gb >= 2000

    # 64GB system (workstation/server)
    budget_64gb = compute_safe_process_budget(total_ram_gb=64.0, available_ram_gb=40.0)
    assert budget_64gb >= 8000


def test_estimate_roofline_defaults():
    # Edge CPU
    edge = estimate_roofline_defaults(
        cpu_brand="Intel(R) Core(TM) i3-5005U CPU @ 2.00GHz",
        cores_physical=2,
        total_ram_gb=3.9,
        gpu_info={"has_gpu": False},
    )
    assert edge["bandwidth_gb_s"] == 16.0
    assert edge["peak_gflops"] == 64.0

    # Modern 8-core CPU
    modern = estimate_roofline_defaults(
        cpu_brand="AMD Ryzen 7 5800X",
        cores_physical=8,
        total_ram_gb=16.0,
        gpu_info={"has_gpu": False},
    )
    assert modern["bandwidth_gb_s"] >= 40.0
    assert modern["peak_gflops"] >= 300.0


def test_detect_system_specs():
    specs = detect_system_specs()
    assert "cpu_brand" in specs
    assert "cpu_short_name" in specs
    assert "cores_physical" in specs
    assert "cores_logical" in specs
    assert "total_ram_gb" in specs
    assert "safe_process_budget_mb" in specs
    assert "profile_tier" in specs
    assert "roofline_defaults" in specs
    assert specs["total_ram_gb"] > 0
    assert specs["safe_process_budget_mb"] > 0


def test_auto_configure_hardware():
    hw = auto_configure_hardware()
    assert hw.system_ram_gb > 0
    assert hw.cpu_threads >= 1
    assert hw.max_memory_mb > 0
    assert hw.cpu_brand is not None
