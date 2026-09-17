"""
Metric calibration pipeline.

Workflow:
1. Detect if uploaded file is a GeoTIFF (has CRS + transform).
2. If GeoTIFF with elevation band: treat as reference DSM — return directly.
3. If regular image: run depth estimator, produce *relative* heights.
4. If SRTM/reference DSM is provided alongside: calibrate with RANSAC
   to map relative depths → absolute metres.
"""
import io
import json
import logging
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    heights: np.ndarray
    width: int
    height: int
    min_height: float
    max_height: float
    units: str  # "metres" or "relative"
    crs: Optional[str] = None
    transform: Optional[list] = None
    provenance: str = "relative"  # "relative", "geotiff-direct", "srtm-calibrated"
    scale: Optional[float] = None
    offset: Optional[float] = None

    def to_meta_dict(self) -> dict:
        d = {
            "width": self.width,
            "height": self.height,
            "units": self.units,
            "min_height": float(self.min_height),
            "max_height": float(self.max_height),
            "provenance": self.provenance,
        }
        if self.crs:
            d["crs"] = self.crs
        if self.transform:
            d["transform"] = self.transform
        if self.scale is not None:
            d["calibration_scale"] = float(self.scale)
            d["calibration_offset"] = float(self.offset)
        return d


def detect_geotiff(file_bytes: bytes) -> Optional[dict]:
    """Check if file_bytes is a GeoTIFF. Returns metadata dict or None."""
    try:
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError:
        logger.debug("rasterio not installed — GeoTIFF detection disabled")
        return None

    try:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as ds:
                if ds.crs is None:
                    return None
                return {
                    "crs": str(ds.crs),
                    "transform": list(ds.transform)[:6],
                    "width": ds.width,
                    "height": ds.height,
                    "count": ds.count,
                    "dtype": str(ds.dtypes[0]),
                    "nodata": ds.nodata,
                }
    except Exception:
        return None


def read_geotiff_elevation(file_bytes: bytes, target_size: int = 1024) -> Optional[CalibrationResult]:
    """Read elevation band from a GeoTIFF, resized to target_size×target_size."""
    try:
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError:
        return None

    try:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as ds:
                if ds.crs is None:
                    return None
                band = ds.read(1).astype(np.float32)
                nodata = ds.nodata
                if nodata is not None:
                    band[band == nodata] = np.nan

                valid = band[~np.isnan(band)]
                if valid.size == 0:
                    return None

                from PIL import Image
                img = Image.fromarray(band)
                img = img.resize((target_size, target_size), Image.BILINEAR)
                heights = np.array(img, dtype=np.float32)
                heights = np.nan_to_num(heights, nan=0.0)

                return CalibrationResult(
                    heights=heights,
                    width=target_size,
                    height=target_size,
                    min_height=float(valid.min()),
                    max_height=float(valid.max()),
                    units="metres",
                    crs=str(ds.crs),
                    transform=list(ds.transform)[:6],
                    provenance="geotiff-direct",
                )
    except Exception as e:
        logger.error("GeoTIFF read failed: %s", e)
        return None


def ransac_calibrate(
    predicted: np.ndarray,
    reference: np.ndarray,
    n_samples: int = 500,
    n_iter: int = 1000,
    inlier_threshold: float = 5.0,
) -> Tuple[float, float, float]:
    """
    RANSAC linear calibration: reference = scale * predicted + offset.
    Returns (scale, offset, inlier_fraction).
    """
    try:
        from sklearn.linear_model import RANSACRegressor
    except ImportError:
        logger.warning("scikit-learn not available — falling back to least squares")
        mask = np.isfinite(predicted) & np.isfinite(reference)
        p, r = predicted[mask].flatten(), reference[mask].flatten()
        if p.size < 10:
            return 1.0, 0.0, 0.0
        A = np.vstack([p, np.ones_like(p)]).T
        result = np.linalg.lstsq(A, r, rcond=None)
        scale, offset = result[0]
        return float(scale), float(offset), 1.0

    mask = np.isfinite(predicted) & np.isfinite(reference)
    p = predicted[mask].flatten()
    r = reference[mask].flatten()

    if p.size < 50:
        logger.warning("Too few valid points (%d) for RANSAC", p.size)
        return 1.0, 0.0, 0.0

    idx = np.random.choice(p.size, min(n_samples, p.size), replace=False)
    p_sub, r_sub = p[idx], r[idx]

    ransac = RANSACRegressor(
        max_trials=n_iter,
        residual_threshold=inlier_threshold,
        random_state=42,
    )
    ransac.fit(p_sub.reshape(-1, 1), r_sub)

    scale = float(ransac.estimator_.coef_[0])
    offset = float(ransac.estimator_.intercept_)
    inlier_mask = ransac.inlier_mask_
    inlier_frac = float(inlier_mask.sum()) / len(inlier_mask)

    logger.info(
        "RANSAC calibration: scale=%.4f offset=%.2f inliers=%.1f%%",
        scale, offset, inlier_frac * 100,
    )
    return scale, offset, inlier_frac


def calibrate_with_reference(
    predicted_heights: np.ndarray,
    reference_bytes: bytes,
    target_size: int = 1024,
) -> Optional[CalibrationResult]:
    """
    Given predicted (relative) heights and a reference GeoTIFF,
    RANSAC-calibrate to absolute metres.
    """
    ref = read_geotiff_elevation(reference_bytes, target_size)
    if ref is None:
        logger.warning("Reference is not a valid elevation GeoTIFF")
        return None

    from PIL import Image
    pred_img = Image.fromarray(predicted_heights)
    pred_resized = np.array(
        pred_img.resize((ref.width, ref.height), Image.BILINEAR),
        dtype=np.float32,
    )

    scale, offset, inlier_frac = ransac_calibrate(pred_resized, ref.heights)

    if inlier_frac < 0.1:
        logger.warning("RANSAC inlier fraction too low (%.1f%%) — calibration unreliable", inlier_frac * 100)

    calibrated = predicted_heights * scale + offset

    return CalibrationResult(
        heights=calibrated,
        width=predicted_heights.shape[1],
        height=predicted_heights.shape[0],
        min_height=float(calibrated.min()),
        max_height=float(calibrated.max()),
        units="metres",
        crs=ref.crs,
        transform=ref.transform,
        provenance="srtm-calibrated",
        scale=scale,
        offset=offset,
    )


def export_dsm_geotiff(
    heights: np.ndarray,
    crs: str,
    transform: list,
    output_path: str,
) -> None:
    """Export calibrated height array as GeoTIFF."""
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError:
        raise RuntimeError("rasterio required for GeoTIFF export")

    h, w = heights.shape
    affine = Affine(*transform[:6])

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        width=w, height=h, count=1,
        dtype="float32",
        crs=crs,
        transform=affine,
    ) as dst:
        dst.write(heights.astype(np.float32), 1)

    logger.info("Exported DSM GeoTIFF to %s", output_path)
