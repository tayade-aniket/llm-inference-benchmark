# Technology Compatibility Matrix & Evaluation

This document outlines the detailed technical justifications, OS requirements, hardware prerequisites, and runtime support boundaries for all evaluated inference engines and optimization libraries.

---

## 1. Engine & Optimization Compatibility Matrix

| Technology | Support Status | Target Execution Environment | Prerequisites & Dependencies | Architectural Limitations & Trade-Offs |
| :--- | :--- | :--- | :--- | :--- |
| **Hugging Face Transformers** | **Supported** (Local Baseline) | Windows 10 CPU | Python 3.10+, `torch>=2.0.0+cpu`, `transformers>=4.40.0` | High memory overhead (~1.0 GB for 0.5B model). Pure-Python generation loop introduces per-token dispatch overhead. Batch size > 1 triggers OOM on 4 GB RAM. |
| **llama.cpp (GGUF)** | **Supported** (Local Optimized) | Windows 10 CPU (AVX2) | Clang/MSVC compiled binary or `llama-cpp-python` | Exceptional memory efficiency (~450 MB for Q4_K_M). C++ SIMD vectorization bypasses Python GIL. Ideal for CPU edge inference. |
| **vLLM** | **Unsupported** (Local) / **Supported** (Remote) | Tier B: Linux x86_64 GPU | Linux kernel, NVIDIA GPU ($\ge 12\text{ GB}$ VRAM, CC $\ge 7.0$), CUDA 12.1+ | Strictly incompatible with Windows 10 CPU. Requires PagedAttention CUDA GEMM kernels. Must run on remote GPU runtime (Colab/Kaggle). |
| **GPTQ (AutoGPTQ)** | **Unsupported** (Local) / **Supported** (Remote) | Tier B: Linux x86_64 GPU | CUDA toolkit, NVIDIA GPU, Marlin or ExLlama CUDA kernels | Weights packed in 32-bit integer containers. CPU decompression kernels are severely unoptimized or unsupported on Windows. |
| **AWQ (AutoAWQ)** | **Unsupported** (Local) / **Supported** (Remote) | Tier B: Linux x86_64 GPU | CUDA 12+, NVIDIA Ampere or newer recommended | GEMM/W4A16 kernels are CUDA-only. No performant native Windows CPU execution path. |
| **KV-Cache Quantization** | **Supported** (Local via llama.cpp) | Windows 10 CPU | llama.cpp `-ctk` and `-ctv` flags (`q8_0`, `q4_0`) | Quantizes keys and values in attention cache from FP16 to 8-bit or 4-bit integers. Reduces KV cache growth for extended contexts. |
| **Prompt / Prefix Caching** | **Supported** (Local & Remote) | Windows 10 CPU (`llama.cpp`), Remote (`vLLM`) | llama.cpp `--prompt-cache` or vLLM automatic prefix caching | Eliminates redundant prefill compute for shared system prompts. Reuses computed KV tokens across requests. |

---

## 2. In-Depth Technical Analysis

### Why vLLM cannot run on Windows 10 CPU
vLLM's core innovation—**PagedAttention**—manages attention key-value states through virtual memory page tables. Its implementation relies directly on custom CUDA C++ kernels compiled with `nvcc` for NVIDIA Tensor Cores. On a CPU-only Windows host:
1. The required CUDA runtime (`libcudart`, `libcublas`) does not exist.
2. Even community experimental CPU ports of vLLM require POSIX socket APIs and OpenMP thread schedulers optimized for multi-socket Linux servers, resulting in non-functional execution on Windows 10 laptops.
3. Therefore, vLLM is properly isolated in **Tier B** as an optional remote experiment.

### Why llama.cpp is the premier CPU engine
1. **Zero Runtime Dependencies**: The entire inference pipeline is compiled into a self-contained, statically linked binary without Python or PyTorch DLL overhead.
2. **SIMD Vectorization**: Features hand-tuned AVX, AVX2, and FMA assembly kernels for 4-bit and 8-bit block quantization (e.g., `Q4_K_M`, `Q8_0`).
3. **mmap Weight Streaming**: `llama.cpp` maps GGUF files directly into the virtual address space using `mmap()`, allowing OS page tables to page weights on demand without duplicate heap allocations.
