# Optional Tier B: vLLM & 8B GPU Inference Guide

This document provides complete instructions for executing the optional **Tier B: 8B Parameter & vLLM Serving Experiment** on a free remote GPU runtime (such as Google Colab or Kaggle Notebooks with a free NVIDIA T4 GPU).

---

## 1. Motivation & Scope

While 0.5B CPU inference demonstrates memory-bandwidth scaling at the edge, production LLM serving architectures (such as OpenAI, Anthropic, or vLLM deployments) operate on discrete server GPUs with continuous batching and PagedAttention.

### Target Environment Requirements
- **Hardware**: NVIDIA T4 (16 GB VRAM) or A10G / L4.
- **Operating System**: Linux x86_64.
- **CUDA**: 12.1+.
- **Software**: `vllm>=0.4.0`, `torch>=2.2.0`, `transformers>=4.40.0`.

---

## 2. Experimental Procedure in Google Colab (Free T4)

### Step 1: Open Free GPU Runtime
In Google Colab, select: `Runtime` -> `Change runtime type` -> `T4 GPU`.

### Step 2: Install vLLM
```bash
!pip install vllm
```

### Step 3: Run Baseline vs vLLM Comparison
```python
from vllm import LLM, SamplingParams
import time

# 1. Initialize vLLM with Meta-Llama-3-8B-Instruct or Qwen2.5-7B
model_id = "Qwen/Qwen2.5-7B-Instruct"

llm = LLM(
    model=model_id,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=2048,
    enable_prefix_caching=True,  # Automatic Prefix Caching
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

prompts = [
    "Explain the concept of PagedAttention in vLLM.",
    "Compare continuous batching with static batching.",
    "Explain why autoregressive decoding is memory-bandwidth bound.",
]

# 2. Batch Inference Benchmark
t0 = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
t_end = time.perf_counter()

total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
elapsed_s = t_end - t0
throughput = total_tokens / elapsed_s

print(f"Total Tokens Generated: {total_tokens}")
print(f"Elapsed Time: {elapsed_s:.2f} s")
print(f"Aggregate GPU Serving Throughput: {throughput:.2f} tokens/s")
```

---

## 3. Comparative Observations: CPU vs GPU Serving
| Dimension | Local Edge (Intel i3-5005U CPU) | Remote Serving (NVIDIA T4 vLLM) |
| :--- | :--- | :--- |
| **Model Evaluated** | Qwen2.5-0.5B (Q4_K_M) | Qwen2.5-7B / Llama-3-8B |
| **Memory Bandwidth** | ~16 GB/s (DDR3L) | ~320 GB/s (GDDR6) |
| **Execution Paradigm** | Sequential, single-request | Continuous iteration batching |
| **Batch Scaling** | Degrades latency at batch > 1 | Scales throughput up to batch 32+ |
| **Prefix Caching** | File-based `--prompt-cache` | Virtual page-table cache eviction |
