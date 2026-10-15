# Benchmark Metrics Definition & Measurement Methodology

This document specifies the exact formal mathematical definitions, instrumentation boundaries, and measurement procedures used throughout the benchmark harness.

---

## 1. Time to First Token (TTFT)

### Definition
Time to First Token (TTFT) measures the latency required to ingest the prompt (prefill phase) and output the very first generated token.

$$\text{TTFT} = t_{\text{token}_1} - t_{\text{request\_start}}$$

- **Measurement Boundary**: Starts immediately after the prompt is submitted to the engine's generation call and ends when the first output token id is yielded by the token streamer / decoder callback.
- **Components Included**:
  - Tokenization / prompt encoding
  - Memory allocation for the initial KV cache
  - Model prompt evaluation (matrix multiplication across all input tokens)
  - Logit projection and argmax / greedy sampling for token 1.
- **Unit**: Milliseconds (ms).
- **Optimization Target**: Lower is better ($\downarrow$).

---

## 2. Inter-Token Latency (ITL)

### Definition
Inter-token latency (ITL), also termed *time per token* or *inter-token jitter*, is the elapsed time between consecutive generated tokens during autoregressive decoding.

For generated tokens $k \in \{2, 3, \dots, N\}$:

$$\text{ITL}_k = t_{\text{token}_k} - t_{\text{token}_{k-1}}$$

- **Statistical Aggregation**:
  - **Median ITL**: $\text{Median}(\{\text{ITL}_2, \dots, \text{ITL}_N\})$ provides the robust central tendency of token arrival rates resistant to OS scheduling jitter.
  - **p95 ITL**: 95th percentile ITL captures tail latency spikes caused by CPU thread preemption, garbage collection, or memory cache misses.
- **Unit**: Milliseconds (ms).
- **Optimization Target**: Lower is better ($\downarrow$).

---

## 3. Time per Output Token (Steady-State Decoding Latency)

### Definition
The average generation latency strictly during the autoregressive decoding loop, explicitly **excluding** the initial prompt prefill (TTFT) phase:

$$\text{Time Per Output Token} = \frac{t_{\text{token}_N} - t_{\text{token}_1}}{N - 1} \quad (\text{for } N > 1)$$

- **Significance**: Unlike naive `End-to-End / Total Tokens`, this metric isolates memory-bandwidth streaming efficiency from prompt length variation.
- **Unit**: Milliseconds (ms).
- **Optimization Target**: Lower is better ($\downarrow$).

---

## 4. End-to-End Latency

### Definition
Total elapsed wall-clock duration of the request from start to completion:

$$\text{End-to-End Latency} = t_{\text{request\_end}} - t_{\text{request\_start}}$$

- **Unit**: Milliseconds (ms).
- **Optimization Target**: Lower is better ($\downarrow$).

---

## 5. Generation Throughput

### Definition
The rate of completed output tokens generated per unit of time:

$$\text{Throughput} = \frac{N_{\text{generated}}}{t_{\text{request\_end}} - t_{\text{request\_start}}}$$

- **Distinction**:
  - **Single-Stream Throughput**: Measures single-user interactive reading experience (tokens/sec).
  - **Aggregate Serving Throughput**: Measures total tokens across all concurrent streams ($\sum N_i / T_{\text{total}}$).
- **Unit**: Tokens per second ($\text{tokens/s}$).
- **Optimization Target**: Higher is better ($\uparrow$).

---

## 6. Memory Consumption Metrics

| Metric | Tool / Source | Description |
| :--- | :--- | :--- |
| **Process Peak RAM (RSS)** | `psutil.Process().memory_info().rss` | Peak resident set size of the process during inference execution. |
| **Memory Delta ($\Delta$ RAM)** | `RSS_{\text{after}} - RSS_{\text{before}}` | Active memory occupied by the model weights and runtime caches. |
| **Available System RAM** | `psutil.virtual_memory().available` | Unallocated physical RAM available to the OS before paging occurs. |
| **Model Disk Size** | `os.path.getsize()` | Static file size of model weights on disk. |
| **Peak VRAM** | `torch.cuda.max_memory_allocated()` | Discrete GPU memory. Explicitly reported as `null` on CPU-only runs. |

---

## 7. Reliability & Error Codes

Every benchmark observation records an explicit status flag:
- `success`: Prompt prefilled and full output tokens decoded without errors.
- `oom`: Execution aborted due to memory allocation limit or safety threshold breach.
- `failed`: Exception encountered during execution (error logged to `notes`).
