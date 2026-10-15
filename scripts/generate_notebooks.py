"""
Generates the 6 educational and reproducible Jupyter Notebooks for the study.
"""

import json
import os


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.2"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.split("\n")]
    }


def build_all_notebooks():
    os.makedirs("notebooks", exist_ok=True)

    # Notebook 1: Environment & Model Setup
    nb1_cells = [
        markdown_cell("""# 01. Environment & Hardware Feasibility Audit
## LLM Inference Benchmark & Optimization Study

### Objectives
1. Inspect the physical hardware platform (CPU, RAM, Storage, GPU availability).
2. Establish strict memory budgets to prevent out-of-memory (OOM) crashes on 4 GB RAM systems.
3. Verify that the pre-flight safety hooks function properly prior to executing inference."""),
        code_cell("""import sys
import os
sys.path.insert(0, os.path.abspath(".."))

from src.inference_bench.config import load_local_config
from src.inference_bench.resource_monitor import ResourceMonitor

# 1. Load Local Configuration
cfg = load_local_config("../configs/local_cpu.yaml")
print(f"Target CPU Threads: {cfg.hardware.cpu_threads}")
print(f"Max Process Memory Budget: {cfg.hardware.max_memory_mb} MB")"""),
        code_cell("""# 2. Inspect System Memory Status
monitor = ResourceMonitor(min_available_mb=400)
mem = monitor.get_system_memory()

print(f"Total Physical RAM: {mem['total_mb']} MB (~{mem['total_mb']/1024:.2f} GB)")
print(f"Available Physical RAM: {mem['available_mb']} MB (~{mem['available_mb']/1024:.2f} GB)")
print(f"Memory Utilization: {mem['used_percent']}%")"""),
        markdown_cell("""### Sizing & Memory Budget Formula
For an autoregressive decoder model with parameter count $P$ (in billions) and precision $b$ (bits per parameter):
$$\\text{Active Model Memory} = P \\times \\frac{b}{8} \\text{ GB}$$

For **Qwen2.5-0.5B**:
- **FP32 ($b=32$)**: $0.49 \\times 4 = 1.96\\text{ GB}$ (Exceeds available RAM)
- **FP16 ($b=16$)**: $0.49 \\times 2 = 0.98\\text{ GB}$ (Tight on 1.3 GB available RAM)
- **Q4_K_M ($b=4.5$)**: $0.49 \\times 0.56 = 0.28\\text{ GB}$ (~398 MB with tensor metadata, leaves > 850 MB free safety buffer)."""),
    ]
    with open("notebooks/01_environment_and_model_setup.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb1_cells), f, indent=2)

    # Notebook 2: Baseline Inference
    nb2_cells = [
        markdown_cell("""# 02. Baseline Inference: Hugging Face vs llama.cpp
## LLM Inference Benchmark & Optimization Study

### Objectives
1. Measure steady-state token generation on a baseline CPU execution path.
2. Compare the runtime characteristics of Hugging Face Transformers vs llama.cpp.
3. Record Time to First Token (TTFT), Inter-Token Latency (ITL), and memory footprint."""),
        code_cell("""import sys, os
sys.path.insert(0, os.path.abspath(".."))

from src.inference_bench.engines.llama_cpp_engine import LlamaCppEngine
from src.inference_bench.metrics import MetricsCalculator
import time

model_path = "../models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
engine = LlamaCppEngine(model_path, config={"n_threads": 2, "n_ctx": 512})
engine.load_model()
print("Model loaded successfully.")"""),
        code_cell("""prompt = "Explain why the sky appears blue in one sentence."
t0 = time.perf_counter()
output_text, token_timestamps, p_tokens, g_tokens = engine.generate(prompt, max_new_tokens=24)
t_end = time.perf_counter()

metrics = MetricsCalculator.compute(t0, token_timestamps, t_end, p_tokens)
print(f"Generated text: {output_text}")
print(f"TTFT: {metrics['ttft_ms']} ms")
print(f"Median ITL: {metrics['median_itl_ms']} ms")
print(f"Throughput: {metrics['output_tokens_per_second']} tokens/sec")
engine.unload_model()"""),
    ]
    with open("notebooks/02_baseline_inference.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb2_cells), f, indent=2)

    # Notebook 3: Latency & Throughput
    nb3_cells = [
        markdown_cell("""# 03. Latency & Throughput Analysis
## LLM Inference Benchmark & Optimization Study

### Objectives
1. Analyze the two phases of LLM inference: **Prefill (TTFT)** vs **Decoding (ITL)**.
2. Investigate how prompt length affects TTFT and how output length affects decoding latency.
3. Plot empirical throughput and latency curves."""),
        code_cell("""import pandas as pd
import glob
import matplotlib.pyplot as plt

raw_files = glob.glob("../results/raw/*.csv")
if raw_files:
    df = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
    print(f"Loaded {len(df)} benchmark observations.")
    display(df.head())
else:
    print("No raw benchmark data files found. Run scripts/run_benchmark.py first.")"""),
    ]
    with open("notebooks/03_latency_and_throughput.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb3_cells), f, indent=2)

    # Notebook 4: Quantization Comparison
    nb4_cells = [
        markdown_cell("""# 04. Quantization Comparison: Weight Formats & Trade-offs
## LLM Inference Benchmark & Optimization Study

### Objectives
1. Compare model weight precision levels: FP16 vs Q8_0 vs Q4_K_M.
2. Quantify the trade-offs in disk footprint, resident memory (RSS), and tokens/sec.
3. Understand why weight quantization accelerates CPU inference."""),
        code_cell("""import sys, os
sys.path.insert(0, os.path.abspath(".."))
import pandas as pd

quant_data = [
    {"Format": "FP16 (Half-Precision)", "Bits/Param": 16.0, "Disk Size (MB)": 980.0, "Expected RSS (MB)": 1100.0, "Bandwidth Traffic / Token": "980 MB"},
    {"Format": "Q8_0 (8-bit Quant)", "Bits/Param": 8.0, "Disk Size (MB)": 540.0, "Expected RSS (MB)": 650.0, "Bandwidth Traffic / Token": "540 MB"},
    {"Format": "Q4_K_M (4-bit Quant)", "Bits/Param": 4.5, "Disk Size (MB)": 398.0, "Expected RSS (MB)": 450.0, "Bandwidth Traffic / Token": "398 MB"},
]
df_quant = pd.DataFrame(quant_data)
display(df_quant)"""),
    ]
    with open("notebooks/04_quantization_comparison.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb4_cells), f, indent=2)

    # Notebook 5: Batching & Caching
    nb5_cells = [
        markdown_cell("""# 05. Batching & Prefix Caching Mechanics
## LLM Inference Benchmark & Optimization Study

### Objectives
1. Evaluate prefix caching (KV-cache reuse) for requests sharing system instructions.
2. Measure hit rate, TTFT speedup factor, and memory savings.
3. Contrast application-level prompt caching with true KV-cache page reuse."""),
        code_cell("""import sys, os
sys.path.insert(0, os.path.abspath(".."))
from src.inference_bench.caching import PrefixCacheTracker

tracker = PrefixCacheTracker()
prompts = [
    "System instruction: Assist doctor. Query: What is insulin?",
    "System instruction: Assist doctor. Query: What is metformin?",
    "System instruction: Assist doctor. Query: What is glucose?",
]

for p in prompts:
    hit, length, speedup = tracker.query(p)
    print(f"Prompt: {p[:45]}... -> Cache Hit: {hit}, Speedup Factor: {speedup}x")

print(f"Overall Hit Rate: {tracker.hit_rate*100:.1f}%")"""),
    ]
    with open("notebooks/05_batching_and_caching.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb5_cells), f, indent=2)

    # Notebook 6: Roofline & Bottleneck Analysis
    nb6_cells = [
        markdown_cell("""# 06. Roofline Model & Bottleneck Analysis
## LLM Inference Benchmark & Optimization Study

### Objectives
1. Construct an empirical and analytical Roofline model for LLM decoding on CPU.
2. Calculate arithmetic intensity for prompt prefill vs autoregressive decoding.
3. Identify when inference is memory-bandwidth-bound vs compute-bound."""),
        code_cell("""import sys, os
sys.path.insert(0, os.path.abspath(".."))
from src.inference_bench.reporting import BenchmarkReporter

# Compute Roofline parameters for Qwen2.5-0.5B on dual-core CPU
roofline = BenchmarkReporter.compute_roofline_analysis(
    param_count_billions=0.49,
    quant_bits=4.5,
    memory_bandwidth_gb_s=16.0,
    cpu_peak_gflops=64.0
)

for k, v in roofline.items():
    print(f"{k:<35}: {v}")"""),
    ]
    with open("notebooks/06_roofline_and_bottleneck_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb6_cells), f, indent=2)

    print("[Notebooks] Successfully generated all 6 educational notebooks in notebooks/")


if __name__ == "__main__":
    build_all_notebooks()
