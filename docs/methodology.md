# Benchmark Methodology & Experimental Protocol

This document establishes the repeatable experimental protocol for evaluating large language model inference on resource-constrained hardware.

---

## 1. Experimental Design Principles

To ensure that results are technically credible and acceptable to an ML Systems interviewer, the benchmark adheres to these principles:

1. **Strict Engine Parity**: When benchmarking engines (e.g. Hugging Face vs llama.cpp), identical model weights (`Qwen2.5-0.5B-Instruct`), prompt texts, tokenization parameters, and generation lengths are utilized.
2. **Warmup Protocol**: The first iteration on any engine is treated as a warm-up run. Warm-up allows:
   - Dynamic library loading into process space
   - Memory page faulting and kernel buffer allocation
   - Instruction cache warming
   Warm-up metrics are logged separately and discarded from official benchmark distributions.
3. **Sequential Execution**: On a 4 GB RAM machine, running benchmarks concurrently would trigger CPU context switching thrash and memory paging. Experiments run sequentially in isolated passes.
4. **Repeated Measurements**: Every prompt is evaluated across multiple repetitions (default: 3) to compute both central tendency (median) and tail dispersion (p95, standard deviation).
5. **Separation of Loading and Steady-State**:
   - Model loading latency and memory allocation are tracked as distinct setup events.
   - Steady-state generation latency is isolated from initial prompt prefill.

---

## 2. Benchmark Pipeline

```
[Prompt Dataset] ──> [Memory Pre-Check] ──> [Warmup Iteration]
                                                   │
  ┌────────────────────────────────────────────────┘
  ▼
[Repetition 1..N] ──> [Timestamped Stream] ──> [Metrics Calculation]
                                                       │
  ┌────────────────────────────────────────────────────┘
  ▼
[Raw JSONL / CSV] ──> [Statistical Aggregation] ──> [Markdown Report & Plots]
```

---

## 3. Hardware Isolation & Resource Management

- **CPU Affinity & Thread Budget**: For dual-core processors (such as the Intel Core i3-5005U with 2 physical cores / 4 threads), thread pools are bounded to `n_threads = 2`. Over-subscribing threads on low-power mobile architectures causes severe L2/L3 cache line evictions and thermal throttling.
- **Safety Ceiling**: Available system memory is queried prior to every prompt launch. If free system RAM falls below 500 MB, the runner terminates safely rather than allowing the operating system to freeze.
- **Deterministic Sampling**: `temperature = 0.0` (greedy decoding) is set across all runs to guarantee deterministic token outputs and identical decoding trajectories across engines.
