"""
Model Download Utility.
Downloads compatible lightweight models for Tier A local CPU experiments.
Includes memory safety validation and progress tracking.
"""

import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download

# Ensure src in sys.path
sys.path.insert(0, os.path.abspath("."))
from src.inference_bench.resource_monitor import ResourceMonitor


def download_qwen_gguf(quant: str = "q4_k_m", target_dir: str = "models"):
    os.makedirs(target_dir, exist_ok=True)
    filename = f"qwen2.5-0.5b-instruct-{quant.lower()}.gguf"
    local_path = os.path.join(target_dir, filename)

    if os.path.exists(local_path):
        print(f"[Model] {filename} already exists at {local_path} ({round(os.path.getsize(local_path)/(1024*1024), 1)} MB)")
        return local_path

    ResourceMonitor().check_safety(required_mb=500)
    print(f"[Model] Downloading {filename} from Qwen/Qwen2.5-0.5B-Instruct-GGUF...")

    downloaded = hf_hub_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename=filename,
        local_dir=target_dir,
    )
    print(f"[Model] Saved to {downloaded}")
    return downloaded


def download_qwen_hf(target_dir: str = "models/hf_qwen2.5_0.5b"):
    os.makedirs(target_dir, exist_ok=True)
    safetensors_path = os.path.join(target_dir, "model.safetensors")

    if os.path.exists(safetensors_path):
        print(f"[Model] HF weights already present at {target_dir}")
        return target_dir

    ResourceMonitor().check_safety(required_mb=500)
    print(f"[Model] Downloading Qwen/Qwen2.5-0.5B-Instruct SafeTensors to {target_dir}...")

    downloaded_dir = snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        local_dir=target_dir,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    print(f"[Model] Saved HF snapshot to {downloaded_dir}")
    return downloaded_dir


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gguf_q4"
    if mode == "gguf_q4":
        download_qwen_gguf("q4_k_m")
    elif mode == "gguf_q8":
        download_qwen_gguf("q8_0")
    elif mode == "hf":
        download_qwen_hf()
    elif mode == "all":
        download_qwen_gguf("q4_k_m")
        download_qwen_gguf("q8_0")
        download_qwen_hf()
