import io
import logging
import time
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from .base import HeightEstimator, HeightMetadata

logger = logging.getLogger(__name__)

OUT_SIZE = 1024
HEIGHT_SCALE = 150.0  # uncalibrated metres

torch.hub._check_repo_is_trusted = lambda *a, **k: None


class MidasEstimator(HeightEstimator):
    """
    Depth estimator using MiDaS_small (intel-isl/MiDaS).
    Weights download from GitHub releases — no HuggingFace CDN required.
    """

    def __init__(self):
        logger.info("Loading MiDaS_small from torch.hub (weights from GitHub releases)…")
        t0 = time.perf_counter()
        self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        transforms_mod = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self._transform = transforms_mod.small_transform
        self._model.eval()
        logger.info("MiDaS_small loaded in %.1f s", time.perf_counter() - t0)

    def estimate(self, image_bytes: bytes) -> Tuple[np.ndarray, HeightMetadata]:
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img_pil)

        inp = self._transform(img_np)

        t0 = time.perf_counter()
        with torch.no_grad():
            depth_tensor = self._model(inp)
            # interpolate to OUT_SIZE
            depth_tensor = torch.nn.functional.interpolate(
                depth_tensor.unsqueeze(1),
                size=(OUT_SIZE, OUT_SIZE),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        elapsed = time.perf_counter() - t0

        depth_raw = depth_tensor.numpy().astype(np.float32)

        raw_min = float(depth_raw.min())
        raw_max = float(depth_raw.max())
        raw_mean = float(depth_raw.mean())
        logger.info(
            "MiDaS inference %.2f s | raw depth min=%.4f  max=%.4f  mean=%.4f",
            elapsed, raw_min, raw_max, raw_mean,
        )

        # MiDaS outputs inverse depth (near=large). Invert so tall buildings = high.
        depth_inv = raw_max - depth_raw

        # Normalise to [0, HEIGHT_SCALE] m (uncalibrated).
        d_range = float(depth_inv.max() - depth_inv.min()) or 1.0
        height = (depth_inv - depth_inv.min()) / d_range * HEIGHT_SCALE

        meta = HeightMetadata(
            width=OUT_SIZE,
            height=OUT_SIZE,
            units="relative",
            min_height=0.0,
            max_height=HEIGHT_SCALE,
        )
        return height, meta
