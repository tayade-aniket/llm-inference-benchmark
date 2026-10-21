"""
Configuration loading, auto-detection, and validation module.
"""

import os
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

from src.inference_bench.system_detector import detect_system_specs


class HardwareConfig(BaseModel):
    device: str = "cpu"
    cpu_brand: Optional[str] = None
    cpu_threads: int = 2
    max_memory_mb: int = 850
    system_ram_gb: float = 4.0
    gpu_available: bool = False
    profile_tier: Optional[str] = None


class ModelSpec(BaseModel):
    family: str = "Qwen2.5"
    repo_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    parameter_count: str = "0.49B"
    context_length: int = 512
    max_new_tokens: int = 32


class TransformersEngineConfig(BaseModel):
    enabled: bool = True
    torch_dtype: str = "float32"
    device_map: str = "cpu"
    low_cpu_mem_usage: bool = True


class LlamaCppEngineConfig(BaseModel):
    enabled: bool = True
    n_threads: int = 2
    n_ctx: int = 512
    n_batch: int = 64
    use_mmap: bool = True
    kv_cache_type_k: str = "f16"
    kv_cache_type_v: str = "f16"


class LocalConfig(BaseModel):
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    models: Dict[str, Any] = Field(default_factory=dict)
    engines: Dict[str, Any] = Field(default_factory=dict)
    safety: Dict[str, Any] = Field(default_factory=dict)


def auto_configure_hardware() -> HardwareConfig:
    """
    Universally detects host machine specs and generates an optimal HardwareConfig.
    """
    specs = detect_system_specs()
    threads = max(1, specs["cores_physical"])
    device = "cuda" if specs["gpu"].get("has_gpu") and specs["gpu"].get("type") == "CUDA" else "cpu"
    return HardwareConfig(
        device=device,
        cpu_brand=specs["cpu_brand"],
        cpu_threads=threads,
        max_memory_mb=specs["safe_process_budget_mb"],
        system_ram_gb=specs["total_ram_gb"],
        gpu_available=specs["gpu"].get("has_gpu", False),
        profile_tier=specs["profile_tier"],
    )


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_local_config(path: str = "configs/local_cpu.yaml", auto_detect: bool = True) -> LocalConfig:
    """
    Loads configuration. If auto_detect is True and path does not exist,
    or if hardware parameters should match the live running machine,
    automatically configures hardware for the host system.
    """
    if os.path.exists(path):
        data = load_yaml(path)
        cfg = LocalConfig(**data)
        if auto_detect:
            # Augment with auto-detected live specs
            auto_hw = auto_configure_hardware()
            cfg.hardware.cpu_brand = auto_hw.cpu_brand
            cfg.hardware.profile_tier = auto_hw.profile_tier
        return cfg
    elif auto_detect:
        # Generate complete auto-configured config
        hw = auto_configure_hardware()
        return LocalConfig(
            hardware=hw,
            models={
                "primary": {
                    "family": "Qwen2.5",
                    "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
                    "parameter_count": "0.49B",
                    "context_length": 512,
                    "max_new_tokens": 32,
                }
            },
            engines={
                "llama_cpp": {
                    "enabled": True,
                    "n_threads": hw.cpu_threads,
                    "n_ctx": 512,
                    "n_batch": 64,
                    "use_mmap": True,
                }
            },
            safety={
                "enforce_memory_check": True,
                "min_available_ram_mb": 500,
            },
        )
    raise FileNotFoundError(f"Configuration file not found: {path}")
