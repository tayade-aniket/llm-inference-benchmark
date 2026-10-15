"""
Prefix and Prompt Caching Module for LLM Inference.
Implements prefix-cache analysis, tracking hit rates, TTFT reduction,
and distinguishing KV-cache prefix reuse from response caching.
"""

import hashlib
import time
from typing import Dict, List, Optional, Tuple


class PrefixCacheTracker:
    """
    Simulates and tracks prefix tree matching and cache reuse across requests.
    Used for educational validation of KV-prefix reuse patterns.
    """

    def __init__(self, block_size_tokens: int = 16):
        self.block_size = block_size_tokens
        self.cached_prefix_hashes = set()
        self.total_requests = 0
        self.cache_hits = 0

    def query(self, prompt: str) -> Tuple[bool, int, float]:
        """
        Calculates whether the prompt's common prefix matches an existing cached prefix.

        Returns:
            Tuple of:
            - is_hit: True if common prefix is present in cache
            - cached_prefix_length_chars: matched prefix length
            - simulated_ttft_speedup_factor: estimated speedup based on reused tokens
        """
        self.total_requests += 1

        # Check common system prefix format: "System instruction: <prefix>. Query: <query>"
        if "Query:" in prompt:
            prefix_part = prompt.split("Query:")[0]
        else:
            # Fallback to first 50% of prompt as potential prefix
            prefix_part = prompt[: len(prompt) // 2]

        prefix_hash = hashlib.sha256(prefix_part.strip().encode("utf-8")).hexdigest()

        if prefix_hash in self.cached_prefix_hashes:
            self.cache_hits += 1
            # Speedup factor is proportional to prefix ratio
            ratio = len(prefix_part) / max(1, len(prompt))
            speedup = 1.0 + (ratio * 2.5)  # Prefill avoided
            return True, len(prefix_part), round(speedup, 2)
        else:
            self.cached_prefix_hashes.add(prefix_hash)
            return False, 0, 1.0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.cache_hits / self.total_requests, 3)

    def reset(self):
        self.cached_prefix_hashes.clear()
        self.total_requests = 0
        self.cache_hits = 0
