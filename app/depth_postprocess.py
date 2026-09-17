"""Shared post-processing for monocular depth estimators.

Sign correction, high-pass detrending, edge taper, normalization,
and brightness-height diagnostics. Used by both MiDaS and Depth Anything.
"""
import logging

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

OUT_SIZE = 1024
HEIGHT_SCALE = 150.0


def orient_depth(depth_raw: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Auto-orient depth so bright pixels (rooftops) are HIGH, dark (roads) LOW.

    Computes Pearson r between image brightness and raw depth.
    If negative, flips the sign. MiDaS / Depth Anything output inverse depth
    (near = large). On nadir imagery buildings are bright AND near, so raw
    depth should already correlate positively with brightness.
    """
    corr, _ = pearsonr(gray.ravel(), depth_raw.ravel())

    if corr >= 0:
        logger.info("Sign: raw depth OK (brightness-depth r=%.4f)", corr)
        return depth_raw.copy()

    flipped = depth_raw.max() - depth_raw
    logger.info("Sign: FLIPPED (brightness-depth r=%.4f < 0, inverted)", corr)
    return flipped


def highpass_detrend(depth: np.ndarray) -> np.ndarray:
    """Remove low-freq ramp (ground-level photo prior). Taper edges to kill rim."""
    std_before = float(depth.std())
    sigma = max(depth.shape) // 4

    low_freq = gaussian_filter(depth, sigma=sigma, mode="nearest")
    residual = depth - low_freq

    border = min(sigma, min(depth.shape) // 3)
    if border > 1:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, border))
        taper = np.ones_like(residual)
        for i in range(border):
            w = ramp[i]
            taper[i, :] *= w
            taper[-(i + 1), :] *= w
            taper[:, i] *= w
            taper[:, -(i + 1)] *= w
        residual *= taper

    std_after = float(residual.std())
    logger.info(
        "Detrend: sigma=%d border=%d | std %.4f -> %.4f (ratio %.2f)",
        sigma,
        border,
        std_before,
        std_after,
        std_after / std_before if std_before > 0 else 0,
    )
    residual -= residual.min()
    return residual


def finalize_heightmap(
    depth: np.ndarray,
    img_pil: Image.Image,
    out_size: int = OUT_SIZE,
    height_scale: float = HEIGHT_SCALE,
) -> np.ndarray:
    """Resize, normalize to [0, height_scale], log rooftop/road/Pearson diagnostics."""
    if depth.shape[0] != out_size or depth.shape[1] != out_size:
        depth = np.array(
            Image.fromarray(depth).resize((out_size, out_size), Image.BILINEAR),
            dtype=np.float32,
        )

    d_range = float(depth.max() - depth.min()) or 1.0
    height = (depth - depth.min()) / d_range * height_scale

    gray = np.array(
        img_pil.convert("L").resize((out_size, out_size), Image.BILINEAR),
        dtype=np.float32,
    )

    p80 = np.percentile(gray, 80)
    p20 = np.percentile(gray, 20)
    bright_mask = gray >= p80
    dark_mask = gray <= p20

    h_bright = float(height[bright_mask].mean()) if bright_mask.any() else 0.0
    h_dark = float(height[dark_mask].mean()) if dark_mask.any() else 0.0
    corr, _ = pearsonr(gray.ravel(), height.ravel())

    logger.info(
        "Final heights: rooftop(bright)=%.2f m  road(dark)=%.2f m  "
        "Pearson r=%.4f  %s",
        h_bright,
        h_dark,
        corr,
        "OK" if h_bright > h_dark else "INVERTED - buildings below roads!",
    )

    return height
