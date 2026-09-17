import io
import logging
import time
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from ..tiling import tile_and_merge
from .base import HeightEstimator, HeightMetadata

logger = logging.getLogger(__name__)

OUT_SIZE = 1024
HEIGHT_SCALE = 150.0
TILE_SIZE = 256
TILE_OVERLAP = 64

torch.hub._check_repo_is_trusted = lambda *a, **k: None


class MidasEstimator(HeightEstimator):

    def __init__(self):
        logger.info("Loading MiDaS_small from torch.hub (weights from GitHub releases)…")
        t0 = time.perf_counter()
        self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        transforms_mod = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self._transform = transforms_mod.small_transform
        self._model.eval()
        logger.info("MiDaS_small loaded in %.1f s", time.perf_counter() - t0)

    def _infer_tile(self, tile_pil: Image.Image) -> np.ndarray:
        """Run MiDaS on a single tile. Returns float32 inverse-depth at tile resolution."""
        tile_np = np.array(tile_pil.convert("RGB"))
        inp = self._transform(tile_np)
        with torch.no_grad():
            pred = self._model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=(tile_pil.size[1], tile_pil.size[0]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return pred.numpy().astype(np.float32)

    def estimate(self, image_bytes: bytes, detrend: bool = True) -> Tuple[np.ndarray, HeightMetadata]:
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
                img_pil, self._infer_tile,
                tile_size=TILE_SIZE, overlap=TILE_OVERLAP,
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "MiDaS inference %.2f s | raw depth min=%.4f max=%.4f mean=%.4f std=%.4f",
            elapsed, depth_raw.min(), depth_raw.max(), depth_raw.mean(), depth_raw.std(),
        )

        # MiDaS outputs inverse depth (near=large). Invert so tall = high.
        depth_inv = depth_raw.max() - depth_raw

        if detrend:
            depth_inv = self._highpass(depth_inv)

        # Resize to output grid
        if depth_inv.shape[0] != OUT_SIZE or depth_inv.shape[1] != OUT_SIZE:
            depth_inv = np.array(
                Image.fromarray(depth_inv).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR),
                dtype=np.float32,
            )

        d_range = float(depth_inv.max() - depth_inv.min()) or 1.0
        height = (depth_inv - depth_inv.min()) / d_range * HEIGHT_SCALE

        meta = HeightMetadata(
            width=OUT_SIZE, height=OUT_SIZE,
            units="relative", min_height=0.0, max_height=HEIGHT_SCALE,
        )
        return height, meta

    @staticmethod
    def _highpass(depth: np.ndarray) -> np.ndarray:
        """Remove bogus low-frequency ramp (MiDaS's ground-level photo prior)."""
        std_before = float(depth.std())

        sigma = max(depth.shape) // 4
        low_freq = gaussian_filter(depth, sigma=sigma)
        residual = depth - low_freq

        std_after = float(residual.std())
        logger.info(
            "Detrend: sigma=%d | std before=%.4f after=%.4f (ratio %.2f)",
            sigma, std_before, std_after,
            std_after / std_before if std_before > 0 else 0,
        )

        residual -= residual.min()
        return residual
