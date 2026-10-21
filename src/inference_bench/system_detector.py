"""
Universal Hardware and System Detection Module.

Automatically detects host architecture, CPU capabilities, memory capacity,
GPU/accelerators, and computes safe runtime budgets for universal deployment
across mobile edge devices, laptops, workstations, cloud servers, and GPU clusters.
"""

import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional
import psutil


def detect_cpu_brand() -> str:
    """Accurately detects CPU brand name across Windows, Linux, and macOS."""
    # 1. Try py-cpuinfo if installed
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw") or info.get("brand")
        if brand and brand.strip():
            return brand.strip()
    except Exception:
        pass

    # 2. Windows WMIC / PowerShell fallback
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            lines = [l.strip() for l in out.strip().splitlines() if l.strip() and "Name" not in l]
            if lines:
                return lines[0]
        except Exception:
            pass

        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            cleaned = out.strip().splitlines()
            if cleaned and cleaned[0].strip():
                return cleaned[0].strip()
        except Exception:
            pass

    # 3. Linux /proc/cpuinfo
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    # 4. macOS sysctl
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            if out.strip():
                return out.strip()
        except Exception:
            pass

    # 5. Standard fallback
    proc = platform.processor()
    if proc and proc.strip():
        return proc.strip()
    return f"{platform.machine()} Generic CPU"


def get_cpu_short_name(brand: str) -> str:
    """Extracts a clean, human-readable short CPU name."""
    cleaned = brand.replace("(R)", "").replace("(TM)", "").replace("CPU", "").replace("Processor", "").strip()
    # Collapse multiple spaces
    cleaned = " ".join(cleaned.split())
    # Truncate at '@' if present (removes base clock for concise badge)
    if "@" in cleaned:
        cleaned = cleaned.split("@")[0].strip()
    return cleaned if cleaned else brand


def detect_simd_flags() -> List[str]:
    """Detects available SIMD and vector extensions (AVX, AVX2, AVX512, NEON, SSE)."""
    flags: List[str] = []
    try:
        import cpuinfo
        cpu_data = cpuinfo.get_cpu_info()
        available = cpu_data.get("flags", [])
        for f in ["avx512f", "avx512", "avx2", "avx", "fma", "sse4_2", "sse4_1", "neon", "amx"]:
            if f in available and f.upper() not in flags:
                flags.append(f.upper())
    except Exception:
        pass

    if not flags:
        # Generic heuristic based on architecture
        arch = platform.machine().lower()
        if "arm" in arch or "aarch64" in arch:
            flags = ["NEON", "FP16"]
        elif "amd64" in arch or "x86_64" in arch:
            flags = ["AVX2", "FMA", "SSE4.2"]
        else:
            flags = ["STANDARD_ALU"]
    return flags


def detect_gpu_info() -> Dict[str, Any]:
    """Detects discrete GPU or accelerator availability (CUDA, MPS, or CPU)."""
    # 1. PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if count > 0 else "NVIDIA GPU"
            vram_gb = 0.0
            if count > 0:
                vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            return {
                "has_gpu": True,
                "type": "CUDA",
                "name": name,
                "count": count,
                "vram_gb": vram_gb,
                "label": f"{name} ({vram_gb} GB VRAM)" if vram_gb else name,
            }

        # 2. Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "has_gpu": True,
                "type": "MPS",
                "name": "Apple Silicon Neural/GPU Engine",
                "count": 1,
                "vram_gb": "Unified",
                "label": "Apple Silicon MPS (Unified Memory)",
            }
    except Exception:
        pass

    return {
        "has_gpu": False,
        "type": "None",
        "name": "Pure CPU (Zero Discrete VRAM)",
        "count": 0,
        "vram_gb": 0.0,
        "label": "Pure CPU (Zero Discrete VRAM)",
    }


def compute_safe_process_budget(total_ram_gb: float, available_ram_gb: float) -> int:
    """
    Dynamically computes the safe resident process memory ceiling (in MB)
    to prevent OS memory pressure, thrashing, and out-of-memory lockups.
    """
    total_mb = int(total_ram_gb * 1024)
    available_mb = int(available_ram_gb * 1024)

    if total_ram_gb <= 4.5:
        # Edge constrained (e.g., 4GB laptop): reserve ~2.6GB for OS baseline
        return min(850, max(350, int(available_mb * 0.70)))
    elif total_ram_gb <= 8.5:
        # 8GB System: safe budget up to ~2,500 MB
        return min(2500, max(850, int(available_mb * 0.75)))
    elif total_ram_gb <= 16.5:
        # 16GB System: safe budget up to ~6,000 MB
        return min(6000, max(2000, int(available_mb * 0.75)))
    elif total_ram_gb <= 32.5:
        # 32GB System: safe budget up to ~14,000 MB
        return min(14000, max(4000, int(available_mb * 0.80)))
    else:
        # 64GB+ Server/Workstation: up to 60% of total RAM or 80% of available
        return min(int(total_mb * 0.60), max(8000, int(available_mb * 0.80)))


def estimate_roofline_defaults(cpu_brand: str, cores_physical: int, total_ram_gb: float, gpu_info: Dict[str, Any]) -> Dict[str, float]:
    """Estimates appropriate default Memory Bandwidth (GB/s) and Peak Compute (GFLOPs) for the detected system."""
    brand_lower = cpu_brand.lower()

    if gpu_info.get("has_gpu"):
        vram = gpu_info.get("vram_gb", 0)
        # GPU Profile
        if isinstance(vram, (int, float)) and vram >= 16:
            return {"bandwidth_gb_s": 800.0, "peak_gflops": 40000.0, "tier_name": "High-End GPU Accelerator"}
        elif isinstance(vram, (int, float)) and vram >= 8:
            return {"bandwidth_gb_s": 360.0, "peak_gflops": 18000.0, "tier_name": "Mid-Range GPU Accelerator"}
        else:
            return {"bandwidth_gb_s": 150.0, "peak_gflops": 8000.0, "tier_name": "Entry GPU / Integrated"}

    # Apple Silicon Unified Memory
    if "apple" in brand_lower or "m1" in brand_lower or "m2" in brand_lower or "m3" in brand_lower or "m4" in brand_lower:
        if "max" in brand_lower or "ultra" in brand_lower:
            return {"bandwidth_gb_s": 300.0, "peak_gflops": 1200.0, "tier_name": "Apple Silicon (Max/Ultra Tier)"}
        elif "pro" in brand_lower:
            return {"bandwidth_gb_s": 150.0, "peak_gflops": 700.0, "tier_name": "Apple Silicon (Pro Tier)"}
        return {"bandwidth_gb_s": 100.0, "peak_gflops": 450.0, "tier_name": "Apple Silicon (Base Unified)"}

    # x86 CPU profiling
    if total_ram_gb <= 4.5 or cores_physical <= 2:
        # Legacy DDR3 / Dual-Core Mobile (e.g. i3-5005U, Celeron, Athlon)
        return {"bandwidth_gb_s": 16.0, "peak_gflops": 64.0, "tier_name": "Dual-Core Edge CPU (DDR3 ~16 GB/s)"}
    elif total_ram_gb <= 12.0 or cores_physical <= 4:
        # Quad-Core DDR4 Laptop / Desktop
        return {"bandwidth_gb_s": 28.0, "peak_gflops": 140.0, "tier_name": "Quad-Core CPU (DDR4 ~28 GB/s)"}
    elif total_ram_gb <= 24.0 or cores_physical <= 8:
        # 6-8 Core Modern Laptop / Desktop (DDR4/DDR5)
        return {"bandwidth_gb_s": 45.0, "peak_gflops": 320.0, "tier_name": "Multi-Core Modern CPU (DDR4/5 ~45 GB/s)"}
    else:
        # 12+ Core Workstation or Server (DDR5 Dual/Quad-Channel)
        return {"bandwidth_gb_s": 75.0, "peak_gflops": 750.0, "tier_name": "Workstation / Server CPU (DDR5 ~75 GB/s)"}


def get_recommended_model_tier(total_ram_gb: float) -> str:
    """Returns recommended LLM model size tier based on system RAM capacity."""
    if total_ram_gb <= 4.5:
        return "Qwen2.5-0.5B to 1.5B (4-bit Q4_K_M, <= 512 context)"
    elif total_ram_gb <= 8.5:
        return "Qwen2.5-1.5B to 3B (4-bit Q4_K_M / Q8_0, 1024 context)"
    elif total_ram_gb <= 16.5:
        return "Llama-3.2-3B or Qwen2.5-7B (4-bit Q4_K_M, 2048 context)"
    elif total_ram_gb <= 32.5:
        return "Llama-3.1-8B or Mistral-7B (Q4_K_M / Q8_0, 4096 context)"
    else:
        return "Command-R 35B / Llama-3.1-70B (4-bit quant) or 8B FP16"


def detect_system_specs() -> Dict[str, Any]:
    """
    Comprehensive system auto-detection.
    Returns a unified dictionary with live telemetry, hardware specs,
    dynamic process safety limits, and engine configuration recommendations.
    """
    vm = psutil.virtual_memory()
    total_ram_gb = round(vm.total / (1024**3), 2)
    available_ram_gb = round(vm.available / (1024**3), 2)
    used_ram_gb = round((vm.total - vm.available) / (1024**3), 2)
    used_ram_percent = vm.percent

    cores_physical = psutil.cpu_count(logical=False) or 2
    cores_logical = psutil.cpu_count(logical=True) or 4

    cpu_brand = detect_cpu_brand()
    cpu_short = get_cpu_short_name(cpu_brand)
    simd_flags = detect_simd_flags()
    gpu_info = detect_gpu_info()
    safe_budget_mb = compute_safe_process_budget(total_ram_gb, available_ram_gb)
    roofline_est = estimate_roofline_defaults(cpu_brand, cores_physical, total_ram_gb, gpu_info)
    model_tier = get_recommended_model_tier(total_ram_gb)

    # Classify overall system profile
    if gpu_info.get("has_gpu"):
        profile_tier = "GPU Accelerated"
    elif total_ram_gb <= 4.5:
        profile_tier = "Ultra-Constrained Edge (<= 4GB)"
    elif total_ram_gb <= 12.0:
        profile_tier = "Mainstream Personal Device (8-12GB)"
    elif total_ram_gb <= 24.0:
        profile_tier = "Performance Workstation (16-24GB)"
    else:
        profile_tier = "High-Capacity Server / Workstation (32GB+)"

    return {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "os_platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_brand": cpu_brand,
        "cpu_short_name": cpu_short,
        "cores_physical": cores_physical,
        "cores_logical": cores_logical,
        "simd_flags": simd_flags,
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": available_ram_gb,
        "used_ram_gb": used_ram_gb,
        "used_ram_percent": used_ram_percent,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "gpu": gpu_info,
        "safe_process_budget_mb": safe_budget_mb,
        "recommended_model_tier": model_tier,
        "profile_tier": profile_tier,
        "roofline_defaults": roofline_est,
    }


if __name__ == "__main__":
    import json
    specs = detect_system_specs()
    print(json.dumps(specs, indent=2))
