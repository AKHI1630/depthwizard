import io
import logging
import time
from typing import Tuple

import numpy as np
from PIL import Image

from .base import HeightEstimator, HeightMetadata

logger = logging.getLogger(__name__)

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
OUT_SIZE = 1024
HEIGHT_SCALE = 150.0  # uncalibrated metres — adjust when metric GT is available


class DepthAnythingEstimator(HeightEstimator):
    def __init__(self):
        # Import lazily so a missing torch install gives a clear error at load time,
        # not at first request.
        from transformers import pipeline as hf_pipeline

        logger.info("Loading %s on CPU (first run downloads ~99 MB)…", MODEL_ID)
        t0 = time.perf_counter()
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=MODEL_ID,
            device="cpu",
        )
        logger.info("Model loaded in %.1f s", time.perf_counter() - t0)

    def estimate(self, image_bytes: bytes) -> Tuple[np.ndarray, HeightMetadata]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        t0 = time.perf_counter()
        result = self._pipe(img)
        elapsed = time.perf_counter() - t0

        # result["depth"] is a PIL Image (mode "I" — 32-bit int — or "F")
        depth_pil = result["depth"]
        depth_raw = np.array(depth_pil, dtype=np.float32)

        raw_min = float(depth_raw.min())
        raw_max = float(depth_raw.max())
        raw_mean = float(depth_raw.mean())
        logger.info(
            "Inference done in %.2f s | raw depth — min=%.4f  max=%.4f  mean=%.4f",
            elapsed, raw_min, raw_max, raw_mean,
        )

        # Invert: model outputs near=large, we need tall=large.
        depth_inv = raw_max - depth_raw

        # Resize to OUT_SIZE×OUT_SIZE (detail above 518px is interpolation, not real).
        inv_pil = Image.fromarray(depth_inv).resize(
            (OUT_SIZE, OUT_SIZE), Image.BICUBIC
        )
        height = np.array(inv_pil, dtype=np.float32)

        # Normalise to [0, HEIGHT_SCALE] metres (uncalibrated).
        d_min = float(height.min())
        d_max = float(height.max())
        d_range = d_max - d_min if d_max != d_min else 1.0
        height = (height - d_min) / d_range * HEIGHT_SCALE

        meta = HeightMetadata(
            width=OUT_SIZE,
            height=OUT_SIZE,
            units="relative",
            min_height=0.0,
            max_height=HEIGHT_SCALE,
        )
        return height, meta
