"""
Tiling support for large images that exceed estimator capacity.
Splits image into overlapping tiles, runs estimator on each, blends results.
"""
import logging
from typing import Callable, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def tile_and_merge(
    image: Image.Image,
    estimate_fn: Callable[[Image.Image], np.ndarray],
    tile_size: int = 1024,
    overlap: int = 128,
) -> np.ndarray:
    w, h = image.size

    if w <= tile_size and h <= tile_size:
        return estimate_fn(image)

    logger.info("Tiling %d×%d image into %d×%d tiles with %d overlap", w, h, tile_size, tile_size, overlap)

    step = tile_size - overlap
    nx = max(1, (w - overlap + step - 1) // step)
    ny = max(1, (h - overlap + step - 1) // step)

    result = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    blend = _make_blend_mask(tile_size, tile_size, overlap)

    for iy in range(ny):
        for ix in range(nx):
            x0 = min(ix * step, w - tile_size)
            y0 = min(iy * step, h - tile_size)
            x1 = x0 + tile_size
            y1 = y0 + tile_size

            crop = image.crop((x0, y0, x1, y1))
            tile_h = estimate_fn(crop)

            if tile_h.shape != (tile_size, tile_size):
                tile_img = Image.fromarray(tile_h)
                tile_h = np.array(tile_img.resize((tile_size, tile_size), Image.BILINEAR), dtype=np.float32)

            th, tw = tile_h.shape
            b = blend[:th, :tw]
            result[y0:y1, x0:x1] += tile_h * b
            weight[y0:y1, x0:x1] += b

    weight[weight == 0] = 1.0
    return (result / weight).astype(np.float32)


def _make_blend_mask(h: int, w: int, overlap: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=np.float32)
    if overlap <= 0:
        return mask

    ramp = np.linspace(0, 1, overlap)
    for i in range(overlap):
        mask[i, :] *= ramp[i]
        mask[h - 1 - i, :] *= ramp[i]
        mask[:, i] *= ramp[i]
        mask[:, w - 1 - i] *= ramp[i]

    return mask
