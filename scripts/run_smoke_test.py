"""
Lightweight Smoke Test for LLM Inference Benchmark Suite.
Runs 100% locally and completely model-free (no downloads, zero GPU required).
Validates configs, memory monitoring, latency math, benchmark serialization, and roofline analysis.
"""

import os
import sys
import time

# Ensure src is in python path
sys.path.insert(0, os.path.abspath("."))

from src.inference_bench.config import load_local_config
from src.inference_bench.resource_monitor import ResourceMonitor, PeakMemoryTracker
from src.inference_bench.metrics import MetricsCalculator, BenchmarkRecord
from src.inference_bench.caching import PrefixCacheTracker
from src.inference_bench.engines.base import BaseInferenceEngine
from src.inference_bench.benchmark import BenchmarkRunner
from src.inference_bench.reporting import BenchmarkReporter


class MockEngine(BaseInferenceEngine):
    """Synthetic engine simulating token generation for instant smoke tests."""

    def load_model(self) -> None:
        self.is_loaded = True
        self.memory_before_load_mb = 100.0
        self.memory_after_load_mb = 120.0

    def generate(self, prompt: str, max_new_tokens: int = 8, temperature: float = 0.0):
        t0 = time.perf_counter()
        # Simulate 8 tokens arriving at 10ms intervals
        timestamps = [t0 + 0.05 + (i * 0.010) for i in range(max_new_tokens)]
        return "Simulated response text.", timestamps, len(prompt.split()), max_new_tokens

    def unload_model(self) -> None:
        self.is_loaded = False

    def get_model_size_mb(self) -> float:
        return 0.0


def run_smoke_test():
    print("=" * 60)
    print("  RUNNING LLM INFERENCE BENCHMARK SMOKE TEST")
    print("=" * 60)

    # 1. Configuration Validation
    print("[1/6] Validating configuration files...")
    cfg = load_local_config("configs/local_cpu.yaml")
    assert cfg.hardware.cpu_threads > 0
    assert cfg.hardware.max_memory_mb <= 1500
    print("  -> Configs loaded and validated successfully.")

    # 2. Resource Monitor
    print("[2/6] Validating memory and safety checks...")
    mon = ResourceMonitor(min_available_mb=100)
    mem_stats = mon.get_system_memory()
    print(f"  -> System RAM Total: {mem_stats['total_mb']} MB, Available: {mem_stats['available_mb']} MB")
    assert mem_stats["total_mb"] > 0
    with PeakMemoryTracker() as tracker:
        x = [i for i in range(100000)]
        tracker.update()
    print("  -> Peak memory tracker verified.")

    # 3. Metrics Calculator Math
    print("[3/6] Validating TTFT, ITL, and throughput mathematics...")
    req_start = 100.0
    ts = [100.150, 100.170, 100.190, 100.210]  # TTFT = 150ms, ITLs = 20ms
    m = MetricsCalculator.compute(req_start, ts, 100.210, prompt_token_count=10)
    assert m["ttft_ms"] == 150.0
    assert m["median_itl_ms"] == 20.0
    assert m["output_tokens_per_second"] > 0
    print(f"  -> Metrics verified: TTFT={m['ttft_ms']}ms, ITL={m['median_itl_ms']}ms, Throughput={m['output_tokens_per_second']} tok/s")

    # 4. Benchmark Harness & Mock Engine
    print("[4/6] Validating benchmark runner and JSONL/CSV serialization...")
    engine = MockEngine("mock-model")
    engine.load_model()
    runner = BenchmarkRunner(
        experiment_id="smoke_test",
        engine=engine,
        engine_name="mock_engine",
        model_name="mock-0.5b",
        quantization="q4_k_m",
        cpu_model="Mock CPU",
        warmup_runs=1,
        repetitions=2,
        max_new_tokens=4,
        output_dir="results/raw",
    )
    test_prompts = [{"id": "t1", "prompt": "Hello world"}]
    records = runner.run_suite(test_prompts)
    assert len(records) == 2
    assert os.path.exists("results/raw/smoke_test_mock_engine.jsonl")
    assert os.path.exists("results/raw/smoke_test_mock_engine.csv")
    print("  -> Benchmark runner generated valid JSONL and CSV output.")

    # 5. Caching Simulation
    print("[5/6] Validating prefix cache tracker...")
    cache_tracker = PrefixCacheTracker()
    hit, _, speedup = cache_tracker.query("System instruction: Assistant. Query: Question 1")
    assert not hit
    hit2, _, speedup2 = cache_tracker.query("System instruction: Assistant. Query: Question 2")
    assert hit2
    assert speedup2 > 1.0
    print(f"  -> Prefix cache tracker verified: hit rate={cache_tracker.hit_rate}")

    # 6. Roofline Analysis
    print("[6/6] Validating roofline model calculations...")
    roofline = BenchmarkReporter.compute_roofline_analysis(
        param_count_billions=0.49,
        quant_bits=4.5,
        memory_bandwidth_gb_s=16.0,
        cpu_peak_gflops=64.0,
    )
    assert roofline["operational_regime"] == "Memory-Bandwidth Bound"
    print(f"  -> Roofline verified: Regime={roofline['operational_regime']}, Peak={roofline['theoretical_max_tokens_per_s']} tok/s")

    print("=" * 60)
    print("  ALL SMOKE TESTS PASSED CLEANLY (100% Local, Zero Downloads)")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
