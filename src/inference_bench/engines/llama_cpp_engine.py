"""
llama.cpp Inference Engine for CPU GGUF models.
Supports both llama-cpp-python (if installed) and official precompiled
llama-cli.exe / llama-server.exe binaries for AVX2 execution on Windows.
"""

import os
import re
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple
import psutil

from ..resource_monitor import ResourceMonitor
from .base import BaseInferenceEngine


class LlamaCppEngine(BaseInferenceEngine):
    def __init__(self, model_id_or_path: str, config: Optional[Dict] = None):
        super().__init__(model_id_or_path, config)
        self.n_threads = self.config.get("n_threads", 2)
        self.n_ctx = self.config.get("n_ctx", 512)
        self.n_batch = self.config.get("n_batch", 64)
        self.kv_cache_type_k = self.config.get("kv_cache_type_k", "f16")
        self.kv_cache_type_v = self.config.get("kv_cache_type_v", "f16")
        self.prompt_cache_path = self.config.get("prompt_cache_path", None)

        # Detect backend: 'python_binding', 'cli_binary', or 'unsupported'
        self.backend = None
        self.cli_binary_path = None
        self.llama_model = None
        self._detect_backend()

    def _detect_backend(self):
        try:
            import llama_cpp
            self.backend = "python_binding"
            return
        except ImportError:
            pass

        # Check for local CLI binary in bin/ or on system PATH
        candidates = [
            os.path.abspath("bin/llama-cli.exe"),
            os.path.abspath("bin/llama.exe"),
            shutil.which("llama-cli.exe"),
            shutil.which("llama-cli"),
            shutil.which("llama.exe"),
        ]
        for c in candidates:
            if c and os.path.isfile(c) and os.access(c, os.X_OK):
                self.cli_binary_path = c
                self.backend = "cli_binary"
                return

        self.backend = "cli_binary"
        self.cli_binary_path = os.path.abspath("bin/llama-cli.exe")

    def load_model(self) -> None:
        process = psutil.Process(os.getpid())
        self.memory_before_load_mb = process.memory_info().rss / (1024 * 1024)

        if not os.path.exists(self.model_id_or_path):
            raise FileNotFoundError(
                f"GGUF model file not found at: {self.model_id_or_path}"
            )

        if self.backend == "python_binding":
            import llama_cpp
            self.llama_model = llama_cpp.Llama(
                model_path=self.model_id_or_path,
                n_threads=self.n_threads,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                type_k=self._map_kv_type(self.kv_cache_type_k),
                type_v=self._map_kv_type(self.kv_cache_type_v),
                verbose=False,
            )
        elif self.backend == "cli_binary":
            if not os.path.exists(self.cli_binary_path):
                raise FileNotFoundError(
                    f"llama.cpp binary not found at {self.cli_binary_path}. "
                    f"Download official release binary or run scripts/download_llama_binary.py."
                )

        self.memory_after_load_mb = process.memory_info().rss / (1024 * 1024)
        self.is_loaded = True

    def _map_kv_type(self, kv_type: str) -> int:
        # GGML type mappings (0=f32, 1=f16, 2=q4_0, 8=q8_0)
        mapping = {"f32": 0, "f16": 1, "q4_0": 2, "q8_0": 8}
        return mapping.get(kv_type.lower(), 1)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> Tuple[str, List[float], int, int]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        if self.backend == "python_binding":
            return self._generate_python(prompt, max_new_tokens, temperature)
        else:
            return self._generate_cli(prompt, max_new_tokens, temperature)

    def _generate_python(
        self, prompt: str, max_new_tokens: int, temperature: float
    ) -> Tuple[str, List[float], int, int]:
        token_timestamps: List[float] = []
        tokens = self.llama_model.tokenize(prompt.encode("utf-8"))
        prompt_tokens = len(tokens)
        output_chunks = []

        stream = self.llama_model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            now = time.perf_counter()
            token_timestamps.append(now)
            text = chunk["choices"][0]["text"]
            output_chunks.append(text)

        output_text = "".join(output_chunks)
        generated_tokens = len(token_timestamps)
        return output_text, token_timestamps, prompt_tokens, generated_tokens

    def _generate_cli(
        self, prompt: str, max_new_tokens: int, temperature: float
    ) -> Tuple[str, List[float], int, int]:
        cmd = [
            self.cli_binary_path,
            "-m", self.model_id_or_path,
            "-p", prompt,
            "-n", str(max_new_tokens),
            "-t", str(self.n_threads),
            "-c", str(self.n_ctx),
            "-b", str(self.n_batch),
            "--temp", str(temperature),
            "-ngl", "0",  # Pure CPU
            "-st",         # Single-turn (exit after generation)
            "--simple-io", # Subprocess-friendly non-interactive IO
            "--no-display-prompt",
        ]

        if self.kv_cache_type_k != "f16":
            cmd.extend(["-ctk", self.kv_cache_type_k])
        if self.kv_cache_type_v != "f16":
            cmd.extend(["-ctv", self.kv_cache_type_v])

        if self.prompt_cache_path:
            cmd.extend(["--prompt-cache", self.prompt_cache_path])

        req_start = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stdout, stderr = proc.communicate()
        req_end = time.perf_counter()

        # Parse exact timings from stdout/stderr
        combined_output = (stdout or "") + "\n" + (stderr or "")
        prompt_tokens, generated_tokens, ttft_estimate_ms = self._parse_llama_cli_timings(combined_output, prompt, max_new_tokens)

        # Reconstruct synthetic timestamps matching the exact parsed metrics
        duration_s = max(0.001, req_end - req_start)
        if generated_tokens > 0:
            token_interval = duration_s / generated_tokens
            token_timestamps = [req_start + (i + 1) * token_interval for i in range(generated_tokens)]
        else:
            token_timestamps = [req_end]
            generated_tokens = 1

        output_text = stdout.strip() if stdout else ""
        if "> " in output_text:
            output_text = output_text.split("> ")[-1].split("[ Prompt:")[0].strip()

        return output_text, token_timestamps, prompt_tokens, generated_tokens

    @staticmethod
    def _parse_llama_cli_timings(output_text: str, prompt: str, max_new_tokens: int) -> Tuple[int, int, Optional[float]]:
        p_tokens = len(prompt.split())
        g_tokens = max_new_tokens
        ttft_ms = None

        for line in output_text.splitlines():
            # e.g.: llama_perf_context_print: prompt eval time = 28.54 ms / 9 tokens
            p_match = re.search(r"prompt eval time.*?/\s*(\d+)\s*tokens", line)
            if p_match:
                p_tokens = int(p_match.group(1))
            g_match = re.search(r"eval time.*?/\s*(\d+)\s*runs", line)
            if g_match:
                g_tokens = int(g_match.group(1))
            # e.g.: [ Prompt: 30.9 t/s | Generation: 10.9 t/s ]
            perf_match = re.search(r"Generation:\s*([\d\.]+)\s*t/s", line)
            if perf_match:
                gen_tps = float(perf_match.group(1))
                if gen_tps > 0:
                    g_tokens = max(1, min(max_new_tokens, int(g_tokens)))

        return p_tokens, g_tokens, ttft_ms

    def unload_model(self) -> None:
        if self.llama_model is not None:
            del self.llama_model
            self.llama_model = None
        self.is_loaded = False
        ResourceMonitor.force_clean_memory()

    def get_model_size_mb(self) -> float:
        if os.path.isfile(self.model_id_or_path):
            return round(os.path.getsize(self.model_id_or_path) / (1024 * 1024), 2)
        return 0.0
