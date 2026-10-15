"""
Hugging Face Transformers Baseline Inference Engine for CPU.
Features exact per-token streaming timestamps and memory guards.
"""

import gc
import os
import time
from typing import Dict, List, Optional, Tuple
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

from ..resource_monitor import ResourceMonitor, PeakMemoryTracker
from .base import BaseInferenceEngine


class TimestampStreamer(BaseStreamer):
    """Captures the monotonic arrival timestamp of each generated token."""

    def __init__(self, tokenizer, skip_prompt: bool = True):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.token_timestamps: List[float] = []
        self.generated_token_ids: List[int] = []
        self._prompt_seen = False

    def put(self, value):
        now = time.perf_counter()
        # Convert tensor to 1D list of ints
        if torch.is_tensor(value):
            t_ids = value.squeeze().tolist()
            if isinstance(t_ids, int):
                t_ids = [t_ids]
        elif isinstance(value, int):
            t_ids = [value]
        else:
            t_ids = list(value)

        # In transformers generate(), if skip_prompt is enabled, the first chunk may be prompt
        if self.skip_prompt and not self._prompt_seen:
            self._prompt_seen = True
            return

        for tid in t_ids:
            self.token_timestamps.append(now)
            self.generated_token_ids.append(tid)

    def end(self):
        pass


class TransformersEngine(BaseInferenceEngine):
    def __init__(self, model_id_or_path: str, config: Optional[Dict] = None):
        super().__init__(model_id_or_path, config)
        self.tokenizer = None
        self.model = None
        self.dtype = getattr(
            torch, self.config.get("torch_dtype", "float32"), torch.float32
        )
        self.cpu_threads = self.config.get("cpu_threads", 2)

    def load_model(self) -> None:
        process = psutil.Process(os.getpid())
        self.memory_before_load_mb = process.memory_info().rss / (1024 * 1024)

        torch.set_num_threads(self.cpu_threads)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id_or_path,
            trust_remote_code=True,
            use_fast=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path,
            torch_dtype=self.dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()

        self.memory_after_load_mb = process.memory_info().rss / (1024 * 1024)
        self.is_loaded = True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> Tuple[str, List[float], int, int]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        prompt_tokens = inputs.input_ids.shape[1]

        streamer = TimestampStreamer(self.tokenizer, skip_prompt=True)

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        generated_tokens = len(streamer.token_timestamps)
        # Decode only the generated portion
        output_text = self.tokenizer.decode(
            streamer.generated_token_ids, skip_special_tokens=True
        )

        return (
            output_text,
            streamer.token_timestamps,
            prompt_tokens,
            generated_tokens,
        )

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.is_loaded = False
        ResourceMonitor.force_clean_memory()

    def get_model_size_mb(self) -> float:
        total_size = 0
        if os.path.isdir(self.model_id_or_path):
            for dirpath, _, filenames in os.walk(self.model_id_or_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
        elif os.path.isfile(self.model_id_or_path):
            total_size = os.path.getsize(self.model_id_or_path)
        return round(total_size / (1024 * 1024), 2)
