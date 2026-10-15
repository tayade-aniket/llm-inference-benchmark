"""
Tests for prefix caching tracking and simulation.
"""

from src.inference_bench.caching import PrefixCacheTracker


def test_prefix_cache_tracker_lifecycle():
    tracker = PrefixCacheTracker(block_size_tokens=16)

    prompt_a = (
        "System instruction: You are a helpful assistant for medical knowledge. Query: What is asthma?"
    )
    prompt_b = (
        "System instruction: You are a helpful assistant for medical knowledge. Query: What is diabetes?"
    )
    prompt_diff = "System instruction: You are a financial expert. Query: What is inflation?"

    # First request: cold cache (miss)
    is_hit, prefix_len, speedup = tracker.query(prompt_a)
    assert not is_hit
    assert speedup == 1.0

    # Second request with identical prefix: warm cache (hit)
    is_hit, prefix_len, speedup = tracker.query(prompt_b)
    assert is_hit
    assert prefix_len > 0
    assert speedup > 1.0

    # Third request with different prefix: cold cache (miss)
    is_hit, prefix_len, speedup = tracker.query(prompt_diff)
    assert not is_hit

    assert tracker.total_requests == 3
    assert tracker.cache_hits == 1
    assert round(tracker.hit_rate, 2) == 0.33
