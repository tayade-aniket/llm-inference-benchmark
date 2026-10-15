"""
Configuration loading and validation module.
"""

import os
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class HardwareConfig(BaseModel):
    device: str = "cpu"
    cpu_threads: int = 2
    max_memory_mb: int = 850
    system_ram_gb: float = 4.0


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


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_local_config(path: str = "configs/local_cpu.yaml") -> LocalConfig:
    data = load_yaml(path)
    return LocalConfig(**data)
