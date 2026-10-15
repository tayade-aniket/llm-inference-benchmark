"""
Benchmark Orchestration Module.
Runs sequential micro-benchmarks across models, engines, and prompts.
Saves raw observations to JSONL and CSV with standardized schema.
"""

import csv
from datetime import datetime, timezone
import json
import os
import time
from typing import Any, Dict, List, Optional
import psutil

from .engines.base import BaseInferenceEngine
from .metrics import BenchmarkRecord, MetricsCalculator
from .resource_monitor import PeakMemoryTracker, ResourceMonitor


class BenchmarkRunner:
    def __init__(
        self,
        experiment_id: str,
        engine: BaseInferenceEngine,
        engine_name: str,
        model_name: str,
        quantization: str = "none",
        cpu_model: str = "Unknown",
        ram_gb: float = 4.0,
        warmup_runs: int = 1,
        repetitions: int = 3,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        output_dir: str = "results/raw",
    ):
        self.experiment_id = experiment_id
        self.engine = engine
        self.engine_name = engine_name
        self.model_name = model_name
        self.quantization = quantization
        self.cpu_model = cpu_model
        self.ram_gb = ram_gb
        self.warmup_runs = warmup_runs
        self.repetitions = repetitions
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.output_dir = output_dir

        self.records: List[BenchmarkRecord] = []
        os.makedirs(output_dir, exist_ok=True)

    def run_prompt(
        self,
        prompt_id: str,
        prompt_text: str,
        is_warmup: bool = False,
        cache_enabled: bool = False,
        cache_hit: Optional[bool] = None,
    ) -> Optional[BenchmarkRecord]:
        """Runs a single prompt through the inference engine and records metrics."""
        monitor = ResourceMonitor()
        monitor.check_safety(required_mb=400)

        # Monotonic time boundaries
        with PeakMemoryTracker() as mem_tracker:
            req_start_s = time.perf_counter()
            try:
                (
                    output_text,
                    token_timestamps_s,
                    prompt_tokens,
                    generated_tokens,
                ) = self.engine.generate(
                    prompt_text,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                req_end_s = time.perf_counter()
                status = "success"
                notes = None
            except MemoryError as me:
                req_end_s = time.perf_counter()
                status = "oom"
                notes = f"Out of memory: {str(me)}"
                token_timestamps_s = []
                prompt_tokens = len(prompt_text.split())
                generated_tokens = 0
            except Exception as e:
                req_end_s = time.perf_counter()
                status = "failed"
                notes = str(e)
                token_timestamps_s = []
                prompt_tokens = len(prompt_text.split())
                generated_tokens = 0

            mem_tracker.update()

        if is_warmup:
            return None  # Do not record warmup iterations in official metrics

        computed = MetricsCalculator.compute(
            request_start_s=req_start_s,
            token_timestamps_s=token_timestamps_s,
            request_end_s=req_end_s,
            prompt_token_count=prompt_tokens,
        )

        record = BenchmarkRecord(
            experiment_id=self.experiment_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            engine=self.engine_name,
            model_id=self.model_name,
            model_revision_or_hash=None,
            quantization=self.quantization,
            device="cpu",
            cpu_model=self.cpu_model,
            ram_gb=self.ram_gb,
            gpu_model=None,
            vram_gb=None,
            prompt_tokens=computed["prompt_tokens"],
            generated_tokens=computed["generated_tokens"],
            batch_size=1,
            concurrency=1,
            ttft_ms=computed["ttft_ms"],
            median_itl_ms=computed["median_itl_ms"],
            p95_itl_ms=computed["p95_itl_ms"],
            time_per_output_token_ms=computed["time_per_output_token_ms"],
            end_to_end_ms=computed["end_to_end_ms"],
            output_tokens_per_second=computed["output_tokens_per_second"],
            process_peak_ram_mb=round(mem_tracker.peak_rss_mb, 2),
            peak_vram_mb=None,
            cache_enabled=cache_enabled,
            cache_hit=cache_hit,
            status=status,
            notes=notes,
        )
        self.records.append(record)
        return record

    def run_suite(self, prompts: List[Dict[str, str]]) -> List[BenchmarkRecord]:
        """Runs the complete benchmark suite: warmup followed by repetitions."""
        # 1. Warm-up
        if prompts and self.warmup_runs > 0:
            warmup_p = prompts[0]["prompt"]
            for _ in range(self.warmup_runs):
                self.run_prompt("warmup", warmup_p, is_warmup=True)

        # 2. Benchmark iterations
        for rep in range(self.repetitions):
            for p in prompts:
                self.run_prompt(
                    prompt_id=f"{p['id']}_rep{rep+1}",
                    prompt_text=p["prompt"],
                    is_warmup=False,
                )

        self.save_results()
        return self.records

    def save_results(self, filename_prefix: Optional[str] = None):
        prefix = filename_prefix or f"{self.experiment_id}_{self.engine_name}"
        jsonl_path = os.path.join(self.output_dir, f"{prefix}.jsonl")
        csv_path = os.path.join(self.output_dir, f"{prefix}.csv")

        # Write JSONL
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict()) + "\n")

        # Write CSV
        if self.records:
            fieldnames = list(self.records[0].to_dict().keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.records:
                    writer.writerow(r.to_dict())
