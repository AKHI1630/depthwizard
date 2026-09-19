"""Download model weights at container startup.

HF Spaces can reach huggingface.co even though the dev laptop cannot.
Weights are cached in HF_HOME (default ~/.cache/huggingface) across restarts.
"""
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")

DA_V2_DIR = MODELS_DIR / "depth-anything-v2-small"
DA_V2_FILES = {
    "config.json": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/config.json",
    "model.safetensors": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/model.safetensors",
    "preprocessor_config.json": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/preprocessor_config.json",
}

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_PATH = MODELS_DIR / "sam_vit_b_01ec64.pth"


def download(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [skip] {dest} already exists ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [download] {url} -> {dest}")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  [done] {dest.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    print("=== Downloading model weights ===")

    print("\nDepth Anything V2 Small:")
    for fname, url in DA_V2_FILES.items():
        download(url, DA_V2_DIR / fname)

    print("\nSAM ViT-B:")
    download(SAM_URL, SAM_PATH)

    print("\nAll weights ready.")


if __name__ == "__main__":
    main()
