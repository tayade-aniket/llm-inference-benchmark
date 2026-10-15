"""
Benchmark Metrics and Schema Definition.
Provides exact calculations for TTFT, ITL, time per output token, throughput,
and standard schema serialization.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class TokenTimestamp:
    token_id: int
    timestamp_s: float


@dataclass
class BenchmarkRecord:
    experiment_id: str
    timestamp_utc: str
    engine: str
    model_id: str
    model_revision_or_hash: Optional[str] = None
    quantization: str = "none"
    device: str = "cpu"
    cpu_model: str = "Unknown"
    ram_gb: float = 4.0
    gpu_model: Optional[str] = None
    vram_gb: Optional[float] = None
    prompt_tokens: int = 0
    generated_tokens: int = 0
    batch_size: int = 1
    concurrency: int = 1
    ttft_ms: Optional[float] = None
    median_itl_ms: Optional[float] = None
    p95_itl_ms: Optional[float] = None
    time_per_output_token_ms: Optional[float] = None
    end_to_end_ms: Optional[float] = None
    output_tokens_per_second: Optional[float] = None
    process_peak_ram_mb: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    cache_enabled: bool = False
    cache_hit: Optional[bool] = None
    status: str = "success"  # success, oom, failed
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCalculator:
    """Calculates latency, ITL, and throughput from raw request timestamps."""

    @staticmethod
    def compute(
        request_start_s: float,
        token_timestamps_s: List[float],
        request_end_s: float,
        prompt_token_count: int,
    ) -> Dict[str, Optional[float]]:
        """
        Computes TTFT, ITL list, median ITL, p95 ITL, time per output token,
        end-to-end latency, and output tokens per second.

        Parameters:
            request_start_s: Timestamp in seconds at request launch.
            token_timestamps_s: Monotonic timestamps in seconds when each token was generated.
            request_end_s: Timestamp in seconds when request completed.
            prompt_token_count: Number of tokens in input prompt.
        """
        num_generated = len(token_timestamps_s)
        total_time_s = max(0.0, request_end_s - request_start_s)
        end_to_end_ms = round(total_time_s * 1000.0, 2)

        if num_generated == 0:
            return {
                "prompt_tokens": prompt_token_count,
                "generated_tokens": 0,
                "ttft_ms": None,
                "median_itl_ms": None,
                "p95_itl_ms": None,
                "time_per_output_token_ms": None,
                "end_to_end_ms": end_to_end_ms,
                "output_tokens_per_second": 0.0,
            }

        # 1. Time to First Token (TTFT): from request start to 1st token
        ttft_s = token_timestamps_s[0] - request_start_s
        ttft_ms = round(max(0.0, ttft_s) * 1000.0, 2)

        # 2. Inter-token latencies (ITL): delta between consecutive tokens
        itls_ms: List[float] = []
        for i in range(1, num_generated):
            delta_ms = (token_timestamps_s[i] - token_timestamps_s[i - 1]) * 1000.0
            if delta_ms >= 0:
                itls_ms.append(delta_ms)

        if itls_ms:
            median_itl = round(float(np.median(itls_ms)), 2)
            p95_itl = round(float(np.percentile(itls_ms, 95)), 2)
        else:
            median_itl = None
            p95_itl = None

        # 3. Time per output token: steady-state decoding time (excluding TTFT)
        if num_generated > 1:
            steady_state_time_s = token_timestamps_s[-1] - token_timestamps_s[0]
            time_per_output_token_ms = round(
                (steady_state_time_s / (num_generated - 1)) * 1000.0, 2
            )
        else:
            time_per_output_token_ms = None

        # 4. Output throughput (tokens/second)
        throughput = (
            round(num_generated / total_time_s, 2) if total_time_s > 0 else 0.0
        )

        return {
            "prompt_tokens": prompt_token_count,
            "generated_tokens": num_generated,
            "ttft_ms": ttft_ms,
            "median_itl_ms": median_itl,
            "p95_itl_ms": p95_itl,
            "time_per_output_token_ms": time_per_output_token_ms,
            "end_to_end_ms": end_to_end_ms,
            "output_tokens_per_second": throughput,
        }
