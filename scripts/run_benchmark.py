"""
Benchmark Execution Runner Script.
Orchestrates end-to-end benchmarking across engines (Transformers & llama.cpp),
recording TTFT, ITL, throughput, and memory consumption.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.inference_bench.config import load_local_config
from src.inference_bench.resource_monitor import ResourceMonitor
from src.inference_bench.benchmark import BenchmarkRunner
from src.inference_bench.reporting import BenchmarkReporter
from src.inference_bench.engines.transformers_engine import TransformersEngine
from src.inference_bench.engines.llama_cpp_engine import LlamaCppEngine
import pandas as pd


def load_prompts(dataset_path: str, limit: int = 5):
    prompts = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
                if len(prompts) >= limit:
                    break
    return prompts


def run_benchmark(
    engine_type: str = "llama_cpp",
    model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    quantization: str = "q4_k_m",
    max_new_tokens: int = 32,
    repetitions: int = 3,
    warmup_runs: int = 1,
    experiment_id: str = "exp_local",
    prompt_file: str = "data/prompts/benchmark_prompts.jsonl",
    prompt_limit: int = 3,
):
    print("=" * 65)
    print(f"  STARTING BENCHMARK: Engine={engine_type}, Model={model_path}")
    print("=" * 65)

    # 1. Safety verification
    monitor = ResourceMonitor(min_available_mb=400)
    monitor.check_safety(required_mb=450)
    mem_stats = monitor.get_system_memory()
    print(f"[Resource Check] Available System RAM: {mem_stats['available_mb']} MB (Safety threshold: 400 MB)")

    # 2. Initialize Engine
    cfg = load_local_config("configs/local_cpu.yaml")
    cpu_model = "Intel Core i3-5005U"

    if engine_type == "llama_cpp":
        engine = LlamaCppEngine(
            model_path,
            config={
                "n_threads": cfg.hardware.cpu_threads,
                "n_ctx": 512,
                "n_batch": 64,
                "kv_cache_type_k": "f16",
                "kv_cache_type_v": "f16",
            },
        )
    elif engine_type == "transformers":
        engine = TransformersEngine(
            model_path,
            config={
                "torch_dtype": "float32",
                "cpu_threads": cfg.hardware.cpu_threads,
            },
        )
    else:
        raise ValueError(f"Unknown engine: {engine_type}")

    print(f"[{engine_type}] Loading model into memory...")
    t0 = datetime.now()
    engine.load_model()
    load_time_s = (datetime.now() - t0).total_seconds()
    print(f"[{engine_type}] Model loaded in {load_time_s:.2f}s.")
    print(f"[{engine_type}] Process RSS before: {engine.memory_before_load_mb:.1f} MB, after: {engine.memory_after_load_mb:.1f} MB")

    # 3. Load Prompts
    prompts = load_prompts(prompt_file, limit=prompt_limit)
    print(f"[Prompts] Loaded {len(prompts)} evaluation prompts from {prompt_file}")

    # 4. Execute Benchmark Suite
    runner = BenchmarkRunner(
        experiment_id=experiment_id,
        engine=engine,
        engine_name=engine_type,
        model_name=os.path.basename(model_path),
        quantization=quantization,
        cpu_model=cpu_model,
        ram_gb=cfg.hardware.system_ram_gb,
        warmup_runs=warmup_runs,
        repetitions=repetitions,
        max_new_tokens=max_new_tokens,
        output_dir="results/raw",
    )

    print(f"[Suite] Executing {warmup_runs} warmup + {repetitions} repetitions per prompt...")
    records = runner.run_suite(prompts)
    print(f"[Suite] Benchmark completed. {len(records)} observation records generated.")

    # 5. Cleanup
    engine.unload_model()
    print(f"[{engine_type}] Model unloaded. Memory reclaimed.")

    # 6. Aggregate Summary
    df_raw = pd.DataFrame([r.to_dict() for r in records])
    summary_df = BenchmarkReporter.aggregate_results(df_raw)
    os.makedirs("results/processed", exist_ok=True)
    summary_path = f"results/processed/{experiment_id}_{engine_type}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[Summary] Saved aggregated summary to: {summary_path}")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Inference Benchmark")
    parser.add_argument("--engine", choices=["llama_cpp", "transformers"], default="llama_cpp")
    parser.add_argument("--model-path", type=str, default="models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    parser.add_argument("--quant", type=str, default="q4_k_m")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--exp-id", type=str, default="exp01_local")
    args = parser.parse_args()

    run_benchmark(
        engine_type=args.engine,
        model_path=args.model_path,
        quantization=args.quant,
        max_new_tokens=args.tokens,
        repetitions=args.reps,
        warmup_runs=args.warmup,
        experiment_id=args.exp_id,
        prompt_limit=args.prompts,
    )
