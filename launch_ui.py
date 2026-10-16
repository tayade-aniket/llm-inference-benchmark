"""
Launcher script for LLM Inference Benchmark UI.
Provides a single-command entrypoint to launch either the Streamlit Interactive Dashboard
or open the zero-dependency Standalone Web Showcase.
"""

import argparse
import os
import subprocess
import sys
import webbrowser


def launch_streamlit(port: int = 8501):
    print("=" * 60)
    print("  LAUNCHING STREAMLIT INTERACTIVE BENCHMARK DASHBOARD")
    print("=" * 60)
    print(f"Starting server on http://localhost:{port} ...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        f"--server.port={port}",
        "--server.headless=false",
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[Dashboard] Shutdown requested by user.")


def launch_web_showcase(port: int = 8080):
    print("=" * 60)
    print("  LAUNCHING STANDALONE WEB SHOWCASE")
    print("=" * 60)
    html_path = os.path.abspath("ui/index.html")
    if not os.path.exists(html_path):
        print(f"Error: {html_path} not found.")
        sys.exit(1)

    print(f"Opening showcase in default web browser: {html_path}")
    webbrowser.open(f"file://{html_path}")


def main():
    parser = argparse.ArgumentParser(description="Launch LLM Inference Benchmark UI")
    parser.add_argument(
        "--mode",
        choices=["streamlit", "web", "auto"],
        default="streamlit",
        help="UI mode: 'streamlit' (interactive ML dashboard) or 'web' (zero-install standalone showcase)",
    )
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit server")
    args = parser.parse_args()

    if args.mode == "web":
        launch_web_showcase()
    else:
        # Check if streamlit is available
        try:
            import streamlit
            launch_streamlit(port=args.port)
        except ImportError:
            print("[Warning] Streamlit is not installed. Falling back to standalone web showcase.")
            launch_web_showcase()


if __name__ == "__main__":
    main()
