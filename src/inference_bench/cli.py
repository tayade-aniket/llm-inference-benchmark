"""
Command Line Interface for LLM Inference Benchmarks.
"""

import argparse
import sys
from .config import load_local_config
from .resource_monitor import ResourceMonitor


def main():
    parser = argparse.ArgumentParser(
        description="LLM Inference Benchmark & Optimization Suite"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Audit command
    subparsers.add_parser("audit", help="Run hardware environment audit")

    # Smoke test command
    subparsers.add_parser("smoke-test", help="Run model-free smoke test")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run inference benchmark")
    bench_parser.add_argument(
        "--engine",
        choices=["transformers", "llama_cpp"],
        default="transformers",
        help="Inference engine",
    )
    bench_parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Path or HuggingFace repo ID for model",
    )
    bench_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max output tokens per prompt",
    )
    bench_parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Repetitions per prompt",
    )

    args = parser.parse_args()

    if args.command == "audit":
        from ..scripts.check_environment import run_diagnostics
        run_diagnostics()
    elif args.command == "smoke-test":
        print("[CLI] Running lightweight smoke test...")
        # Will call smoke test runner
    elif args.command == "benchmark":
        print(f"[CLI] Running benchmark on engine: {args.engine}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
