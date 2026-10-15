"""
Resource Monitor module for CPU and Memory tracking on constrained hardware.
"""

import gc
import os
from typing import Dict, Optional, Tuple
import psutil


class MemoryCheckError(RuntimeError):
    """Raised when available memory is below the required safety threshold."""
    pass


class ResourceMonitor:
    def __init__(self, min_available_mb: int = 500):
        self.min_available_mb = min_available_mb
        self.process = psutil.Process(os.getpid())

    @staticmethod
    def get_system_memory() -> Dict[str, float]:
        vm = psutil.virtual_memory()
        return {
            "total_mb": round(vm.total / (1024 * 1024), 1),
            "available_mb": round(vm.available / (1024 * 1024), 1),
            "free_mb": round(vm.free / (1024 * 1024), 1),
            "used_percent": vm.percent,
        }

    def get_process_memory_mb(self) -> float:
        """Returns the resident set size (RSS) of the current process in MB."""
        return round(self.process.memory_info().rss / (1024 * 1024), 2)

    def check_safety(self, required_mb: int = 400) -> bool:
        """Checks if there is enough free system memory before initiating a run."""
        mem = self.get_system_memory()
        available = mem["available_mb"]
        if available < self.min_available_mb or available < required_mb:
            raise MemoryCheckError(
                f"Insufficient memory! Available: {available:.1f} MB, "
                f"Required minimum buffer: {self.min_available_mb} MB (requested: {required_mb} MB). "
                f"Aborting to prevent system lockup/paging thrash."
            )
        return True

    @staticmethod
    def force_clean_memory():
        """Aggressive garbage collection and memory release."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class PeakMemoryTracker:
    """Context manager to measure process peak memory delta during a code block."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_rss_mb: float = 0.0
        self.peak_rss_mb: float = 0.0
        self.end_rss_mb: float = 0.0

    def __enter__(self):
        ResourceMonitor.force_clean_memory()
        self.start_rss_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_rss_mb = self.start_rss_mb
        return self

    def update(self):
        current = self.process.memory_info().rss / (1024 * 1024)
        if current > self.peak_rss_mb:
            self.peak_rss_mb = current

    def __exit__(self, exc_type, exc_val, exc_tb):
        current = self.process.memory_info().rss / (1024 * 1024)
        if current > self.peak_rss_mb:
            self.peak_rss_mb = current
        self.end_rss_mb = current
