"""Download model weights at container startup.

Weights are cached in the container's Hugging Face/local model directory.
"""
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")

# Transformers-compatible Depth Anything V2 Small repository.
# The original depth-anything/Depth-Anything-V2-Small repo contains the
# native .pth checkpoint, while this app loads the Transformers format
# (config.json + model.safetensors + preprocessor_config.json).
DA_V2_DIR = MODELS_DIR / "depth-anything-v2-small"
DA_V2_FILES = {
    "config.json": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/config.json",
    "model.safetensors": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/model.safetensors",
    "preprocessor_config.json": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/preprocessor_config.json",
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
