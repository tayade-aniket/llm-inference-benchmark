# Threats to Validity, Hardware Realities, and Study Limitations

A credible Machine Learning Systems study must rigorously document its experimental limitations, boundary conditions, and threats to internal and external validity.

---

## 1. Threats to External Validity (Generalizability)

### A. Non-Equivalence of CPU Edge vs GPU Data-Center Serving
- **Memory Bandwidth Gap**: The dual-channel DDR3L-1600 memory subsystem on this laptop delivers approximately **$12 - 16\text{ GB/s}$** of effective memory bandwidth. In contrast, modern server GPUs (NVIDIA A100/H100) deliver **$2,000 - 3,350\text{ GB/s}$** via HBM2e/HBM3.
- **Compute Throughput Gap**: The dual-core Intel Core i3-5005U peaks at approximately **$64\text{ GFLOPs/s}$**, compared to **$312 - 1,000\text{ TFLOPs/s}$** on modern Tensor Core hardware.
- **Serving Architecture**: While continuous batching (iteration-level scheduling) provides massive 5x–15x throughput gains on GPUs by saturating thousands of CUDA cores, on a 2-core CPU, batch sizes $> 1$ quickly exhaust compute and cache capacity, increasing latency without proportional throughput gains.

### B. Parameter Scaling Discrepancy
- Results measured on a 0.5B parameter model (such as `Qwen2.5-0.5B`) cannot be directly extrapolated to 8B, 70B, or mixture-of-experts (MoE) architectures:
  - KV-cache growth in small models (2 KV heads, 896 hidden dim) is negligible (< 15 MB for 512 tokens).
  - In 70B models, KV caches consume tens of gigabytes, making KV-cache quantization (e.g., FP8/INT4) far more impactful than demonstrated at small scale.

---

## 2. Threats to Internal Validity (Measurement Noise)

### A. Mobile Thermal Throttling
- The Intel Core i3-5005U operates within a 15W thermal design envelope (TDP). Sustained AVX2 matrix operations cause CPU die temperatures to rise, inducing CPU frequency throttling from 2.0 GHz down to 1.6–1.8 GHz.
- **Mitigation**: Benchmark runs include inter-prompt cool-down intervals and discard warm-up runs.

### B. Background OS Jitter on Windows 10
- Windows background services (Windows Update, Antimalware Service Executable) periodically preempt CPU cores.
- **Mitigation**: We report **Median** and **p95** metrics across repeated trials rather than a single arithmetic mean.

---

## 3. Methodological Integrity & Honest Reporting

- Unexecuted configurations (such as local vLLM or 8B local execution) are strictly labeled as **"Unsupported / Not Executed"**.
- Hardware utilization metrics (e.g. memory bandwidth saturation) are computed using analytical Roofline formulas and labelled as **"Theoretical Estimate"** rather than hardware-counter measured.
