"""
Helper script to download official precompiled llama.cpp Windows x64 CPU binary.
Extracts llama-cli.exe and dependencies to bin/
"""

import io
import os
import urllib.request
import zipfile


def download_llama_cpu_binary(target_dir: str = "bin"):
    os.makedirs(target_dir, exist_ok=True)
    cli_path = os.path.join(target_dir, "llama-cli.exe")
    if os.path.exists(cli_path):
        print(f"[llama.cpp] Binary already exists at {cli_path}")
        return cli_path

    url = "https://github.com/ggerganov/llama.cpp/releases/download/b11669/llama-b11669-bin-win-cpu-x64.zip"
    print(f"[llama.cpp] Downloading official CPU binary from {url} (~18.6 MB)...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        zip_bytes = resp.read()

    print("[llama.cpp] Extracting archive...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(target_dir)

    print(f"[llama.cpp] Successfully extracted to {target_dir}/")
    return cli_path


if __name__ == "__main__":
    download_llama_cpu_binary()
