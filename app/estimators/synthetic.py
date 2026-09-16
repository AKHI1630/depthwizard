import numpy as np
from typing import Tuple
from .base import HeightEstimator, HeightMetadata

W, H = 1024, 1024


def _gaussian(cx: float, cy: float, sx: float, sy: float, amp: float, grid_x, grid_y) -> np.ndarray:
    return amp * np.exp(-(((grid_x - cx) / sx) ** 2 + ((grid_y - cy) / sy) ** 2) / 2)


class SyntheticEstimator(HeightEstimator):
    """Returns a deterministic synthetic terrain: Gaussian hills + box buildings."""

    def estimate(self, image: bytes) -> Tuple[np.ndarray, HeightMetadata]:
        xs = np.linspace(0, 1, W, dtype=np.float32)
        ys = np.linspace(0, 1, H, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)

        terrain = np.zeros((H, W), dtype=np.float32)

        hills = [
            (0.25, 0.30, 0.18, 0.18, 120.0),
            (0.70, 0.65, 0.22, 0.20, 95.0),
            (0.50, 0.20, 0.12, 0.14, 70.0),
            (0.15, 0.75, 0.16, 0.13, 85.0),
            (0.80, 0.25, 0.10, 0.12, 60.0),
        ]
        for cx, cy, sx, sy, amp in hills:
            terrain += _gaussian(cx, cy, sx, sy, amp, gx, gy)

        # Six rectangular buildings on top of the terrain
        buildings = [
            # (x0, y0, x1, y1, extra_height)
            (0.42, 0.42, 0.50, 0.50, 80.0),
            (0.55, 0.44, 0.61, 0.52, 55.0),
            (0.30, 0.55, 0.37, 0.63, 40.0),
            (0.62, 0.20, 0.68, 0.28, 65.0),
            (0.18, 0.35, 0.23, 0.42, 30.0),
            (0.72, 0.68, 0.79, 0.75, 50.0),
        ]
        px = (gx * W).astype(np.int32)
        py = (gy * H).astype(np.int32)
        for x0, y0, x1, y1, extra in buildings:
            mask = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
            # Building top is terrain surface at its center + extra
            cx_b = int((x0 + x1) / 2 * W)
            cy_b = int((y0 + y1) / 2 * H)
            base = float(terrain[cy_b, cx_b])
            terrain[mask] = base + extra

        min_h = float(terrain.min())
        max_h = float(terrain.max())
        meta = HeightMetadata(
            width=W, height=H, units="meters", min_height=min_h, max_height=max_h
        )
        return terrain, meta
