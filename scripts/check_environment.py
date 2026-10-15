"""
Environment and Hardware Diagnostic Script for LLM Inference Benchmarking.

Collects system architecture, CPU details, RAM limits, storage availability,
GPU/CUDA status, and Python package versions. Designed for resource-constrained
systems (e.g., 4GB RAM, CPU-only).
"""

import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone


def get_cpu_info():
    info = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "cores_physical": None,
        "cores_logical": os.cpu_count(),
        "model_name": platform.processor(),
    }
    try:
        import psutil

        info["cores_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass

    try:
        import cpuinfo

        cpu_data = cpuinfo.get_cpu_info()
        info["model_name"] = cpu_data.get("brand_raw", platform.processor())
        info["flags"] = [
            f
            for f in ["avx", "avx2", "avx512", "fma", "sse4_2"]
            if f in cpu_data.get("flags", [])
        ]
    except Exception:
        info["flags"] = "cpuinfo not available or unsupported"

    return info


def get_memory_info():
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total_ram_gb": round(vm.total / (1024**3), 2),
            "available_ram_gb": round(vm.available / (1024**3), 2),
            "free_ram_gb": round(vm.free / (1024**3), 2),
            "used_ram_percent": vm.percent,
            "swap_total_gb": round(psutil.swap_memory().total / (1024**3), 2),
            "swap_free_gb": round(psutil.swap_memory().free / (1024**3), 2),
        }
    except ImportError:
        return {"error": "psutil not installed"}


def get_disk_info(paths=None):
    if paths is None:
        paths = ["C:\\", os.path.abspath(".")]
    disk_data = {}
    for p in set(paths):
        try:
            total, used, free = shutil.disk_usage(p)
            disk_data[p] = {
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
            }
        except Exception as e:
            disk_data[p] = {"error": str(e)}
    return disk_data


def get_gpu_info():
    gpu_data = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "driver_version": None,
    }

    try:
        import torch

        gpu_data["cuda_available"] = torch.cuda.is_available()
        gpu_data["device_count"] = (
            torch.cuda.device_count() if gpu_data["cuda_available"] else 0
        )
        if gpu_data["cuda_available"]:
            for i in range(gpu_data["device_count"]):
                gpu_data["devices"].append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_vram_gb": round(
                            torch.cuda.get_device_properties(i).total_memory
                            / (1024**3),
                            2,
                        ),
                    }
                )
    except ImportError:
        gpu_data["torch_installed"] = False

    # Check for non-CUDA GPUs via WMI on Windows
    if sys.platform == "win32":
        try:
            import wmi

            w = wmi.WMI()
            adapters = []
            for controller in w.Win32_VideoController():
                adapters.append(
                    {
                        "name": controller.Name,
                        "adapter_ram_mb": (
                            round(int(controller.AdapterRAM) / (1024**2), 1)
                            if controller.AdapterRAM
                            else None
                        ),
                        "driver_version": controller.DriverVersion,
                    }
                )
            gpu_data["windows_display_adapters"] = adapters
        except Exception:
            pass

    return gpu_data


def get_python_env():
    import importlib.metadata

    pkgs = [
        "torch",
        "transformers",
        "tokenizers",
        "llama_cpp",
        "psutil",
        "numpy",
        "pandas",
        "pytest",
        "safetensors",
        "vllm",
    ]
    installed_versions = {}
    for pkg in pkgs:
        try:
            ver = importlib.metadata.version(pkg)
            installed_versions[pkg] = ver
        except importlib.metadata.PackageNotFoundError:
            installed_versions[pkg] = "not installed"

    return {
        "python_version": sys.version.split()[0],
        "python_path": sys.executable,
        "packages": installed_versions,
    }


def run_diagnostics(output_json=None):
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "disk": get_disk_info(),
        "gpu": get_gpu_info(),
        "python": get_python_env(),
    }

    print("=" * 65)
    print("  SYSTEM & HARDWARE DIAGNOSTIC REPORT (Phase 1 Audit)")
    print("=" * 65)
    print(f"Timestamp (UTC) : {report['timestamp_utc']}")
    print(
        f"OS              : {report['os']['system']} {report['os']['release']} ({report['cpu']['architecture']})"
    )
    print(f"CPU Model       : {report['cpu']['model_name']}")
    print(
        f"CPU Topology    : {report['cpu']['cores_physical']} Physical Cores, {report['cpu']['cores_logical']} Logical Threads"
    )
    print(f"Instruction Sets: {report['cpu'].get('flags', 'N/A')}")
    print("-" * 65)
    mem = report["memory"]
    print(f"Total RAM       : {mem.get('total_ram_gb', 'N/A')} GB")
    print(
        f"Available RAM   : {mem.get('available_ram_gb', 'N/A')} GB ({mem.get('used_ram_percent', 'N/A')}% in use)"
    )
    print(f"Pagefile/Swap   : {mem.get('swap_total_gb', 'N/A')} GB Total")
    print("-" * 65)
    print("Storage:")
    for path, d in report["disk"].items():
        print(
            f"  {path:<20} Free: {d.get('free_gb', 'N/A')} GB / Total: {d.get('total_gb', 'N/A')} GB"
        )
    print("-" * 65)
    gpu = report["gpu"]
    print(f"CUDA Available  : {gpu['cuda_available']}")
    if "windows_display_adapters" in gpu:
        for adapter in gpu["windows_display_adapters"]:
            print(
                f"Display Adapter : {adapter['name']} (Driver: {adapter['driver_version']})"
            )
    print("-" * 65)
    print("Python & ML Libraries:")
    print(f"  Python Version: {report['python']['python_version']}")
    for pkg, ver in report["python"]["packages"].items():
        print(f"  {pkg:<15}: {ver}")
    print("=" * 65)

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[Saved JSON Diagnostic Report to: {output_json}]")

    return report


if __name__ == "__main__":
    out_file = (
        sys.argv[1] if len(sys.argv) > 1 else "results/raw/environment_audit.json"
    )
    run_diagnostics(out_file)
