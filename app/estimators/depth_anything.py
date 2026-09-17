import io
import logging
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from ..depth_postprocess import (
    HEIGHT_SCALE,
    OUT_SIZE,
    finalize_heightmap,
    highpass_detrend,
    orient_depth,
)
from ..tiling import tile_and_merge
from .base import HeightEstimator, HeightMetadata

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "depth-anything-v2-small"
TILE_SIZE = 518
TILE_OVERLAP = 96


class DepthAnythingEstimator(HeightEstimator):

    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

        if not MODEL_DIR.exists():
            raise RuntimeError(
                f"Depth Anything V2 model directory not found: {MODEL_DIR}\n"
                "Clone https://github.com/AKHI1630/my-project and copy "
                "model.safetensors, config.json, preprocessor_config.json "
                "into models/depth-anything-v2-small/"
            )

        required = ["model.safetensors", "config.json", "preprocessor_config.json"]
        missing = [f for f in required if not (MODEL_DIR / f).exists()]
        if missing:
            raise RuntimeError(
                f"Missing files in {MODEL_DIR}: {missing}"
            )

        weights_size = (MODEL_DIR / "model.safetensors").stat().st_size
        if weights_size < 1_000_000:
            raise RuntimeError(
                f"model.safetensors is only {weights_size} bytes — "
                "likely a Git LFS pointer. Run 'git lfs pull' in the source repo."
            )

        phys_cores = os.cpu_count() or 4
        torch.set_num_threads(phys_cores)
        logger.info("torch.set_num_threads(%d)", phys_cores)

        logger.info("Loading Depth-Anything-V2-Small from %s…", MODEL_DIR)
        t0 = time.perf_counter()

        self._processor = AutoImageProcessor.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )
        self._model = DepthAnythingForDepthEstimation.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )
        self._model.eval()

        logger.info(
            "Depth-Anything-V2-Small loaded in %.1f s", time.perf_counter() - t0
        )

    def _infer_tile(self, tile_pil: Image.Image) -> np.ndarray:
        import torch

        inputs = self._processor(images=tile_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth

        pred = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(tile_pil.size[1], tile_pil.size[0]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return pred.numpy().astype(np.float32)

    def _infer_batch(self, tile_pils: list[Image.Image]) -> list[np.ndarray]:
        """Run batch inference on multiple tiles in a single forward pass."""
        import torch

        inputs = self._processor(images=tile_pils, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth

        results = []
        for i, tile_pil in enumerate(tile_pils):
            pred = torch.nn.functional.interpolate(
                predicted_depth[i:i+1].unsqueeze(1),
                size=(tile_pil.size[1], tile_pil.size[0]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            results.append(pred.numpy().astype(np.float32))

        return results

    def estimate(
        self, image_bytes: bytes, detrend: bool = True
    ) -> Tuple[np.ndarray, HeightMetadata]:
        try:
            img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot decode image: {e}") from e

        w, h = img_pil.size
        logger.info("Input image: %d×%d", w, h)

        if w < 16 or h < 16:
            raise ValueError(f"Image too small ({w}×{h}). Minimum 16×16.")
        if w > 8192 or h > 8192:
            scale = 8192 / max(w, h)
            img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h = img_pil.size
            logger.info("Resized large image to %d×%d", w, h)

        t0 = time.perf_counter()

        if w <= TILE_SIZE and h <= TILE_SIZE:
            depth_raw = self._infer_tile(img_pil)
        else:
            depth_raw = tile_and_merge(
                img_pil,
                self._infer_tile,
                tile_size=TILE_SIZE,
                overlap=TILE_OVERLAP,
                batch_fn=self._infer_batch,
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "DepthAnything inference %.2f s | raw min=%.4f max=%.4f mean=%.4f std=%.4f",
            elapsed,
            depth_raw.min(),
            depth_raw.max(),
            depth_raw.mean(),
            depth_raw.std(),
        )

        gray = np.array(
            img_pil.convert("L").resize(
                (depth_raw.shape[1], depth_raw.shape[0]), Image.BILINEAR
            ),
            dtype=np.float32,
        )
        depth = orient_depth(depth_raw, gray)

        if detrend:
            depth = highpass_detrend(depth)

        height = finalize_heightmap(depth, img_pil, OUT_SIZE, HEIGHT_SCALE)

        meta = HeightMetadata(
            width=OUT_SIZE,
            height=OUT_SIZE,
            units="relative",
            min_height=0.0,
            max_height=HEIGHT_SCALE,
        )
        return height, meta
