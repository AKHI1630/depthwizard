"""
Tiling support for running a fixed-resolution estimator on large images.
Splits image into overlapping tiles, runs inference per tile, blends with
cosine-feathered weights so seams are invisible.
"""
import logging
from typing import Callable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def tile_and_merge(
    image: Image.Image,
    estimate_fn: Callable[[Image.Image], np.ndarray],
    tile_size: int = 256,
    overlap: int = 64,
) -> np.ndarray:
    w, h = image.size

    if w <= tile_size and h <= tile_size:
        return estimate_fn(image)

    step = tile_size - overlap
    nx = max(1, (w - overlap + step - 1) // step)
    ny = max(1, (h - overlap + step - 1) // step)

    logger.info(
        "Tiling %d×%d into %dx%d grid of %d×%d tiles (overlap %d, step %d)",
        w, h, nx, ny, tile_size, tile_size, overlap, step,
    )

    result = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    blend = _cosine_blend_mask(tile_size, overlap)

    for iy in range(ny):
        for ix in range(nx):
            x0 = min(ix * step, max(0, w - tile_size))
            y0 = min(iy * step, max(0, h - tile_size))
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)

            crop = image.crop((x0, y0, x1, y1))
            tile_h = estimate_fn(crop)

            tw, th = x1 - x0, y1 - y0
            if tile_h.shape[0] != th or tile_h.shape[1] != tw:
                tile_h = np.array(
                    Image.fromarray(tile_h).resize((tw, th), Image.BILINEAR),
                    dtype=np.float32,
                )

            b = blend[:th, :tw]
            result[y0:y1, x0:x1] += tile_h * b
            weight[y0:y1, x0:x1] += b

    weight[weight == 0] = 1.0
    merged = (result / weight).astype(np.float32)
    logger.info("Tiled merge complete: output %d×%d", merged.shape[1], merged.shape[0])
    return merged


def _cosine_blend_mask(size: int, overlap: int) -> np.ndarray:
    """Cosine-feathered blend mask — 1.0 in the interior, smooth 0→1 ramp in overlap."""
    mask = np.ones((size, size), dtype=np.float32)
    if overlap <= 0:
        return mask

    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap))
    for i in range(overlap):
        mask[i, :] *= ramp[i]
        mask[size - 1 - i, :] *= ramp[i]
        mask[:, i] *= ramp[i]
        mask[:, size - 1 - i] *= ramp[i]

    return mask
