"""Multi-cue height estimation fusion for satellite building reconstruction."""
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.sam_segmentation import InstanceMask
from app.shadow_detection import ShadowBlob, measure_shadow_length
from app.sun_geometry import SunPosition

logger = logging.getLogger(__name__)


@dataclass
class HeightEstimate:
    height_m: float
    uncertainty_m: float
    method: str  # "shadow", "facade", "lean", "dav2_calibrated", "neighbour_prior"


@dataclass
class FusedHeight:
    height_m: float
    uncertainty_m: float
    methods: list[str]
    estimates: list[HeightEstimate]
    source: str  # "measured" or "inferred"
    low_confidence_flag: bool


_MEASURED_METHODS = {"shadow", "facade", "lean"}

_SUN_DELTA_BY_SOURCE = {
    "metadata": 1.0,
    "computed": 0.5,
    "estimated": 5.0,
    "user": 10.0,
}


def _sun_elevation_delta(sun: SunPosition) -> float:
    return _SUN_DELTA_BY_SOURCE.get(sun.source, 5.0)


def _sanity_check(h: float, lo: float = 1.0, hi: float = 300.0) -> bool:
    return lo <= h <= hi


# ---------------------------------------------------------------------------
# A. Shadow-based height
# ---------------------------------------------------------------------------

def estimate_shadow_height(
    inst: InstanceMask,
    shadow_blobs: list[ShadowBlob],
    building_masks: list[np.ndarray],
    sun: SunPosition,
    gsd_m: float,
) -> Optional[HeightEstimate]:
    matched_blobs = [
        b for b in shadow_blobs
        if b.building_idx is not None
        and b.building_idx < len(building_masks)
        and np.array_equal(building_masks[b.building_idx], inst.mask)
    ]
    if not matched_blobs:
        logger.debug("shadow: no matched blobs for building at bbox=%s", inst.bbox)
        return None

    lengths_px = []
    for blob in matched_blobs:
        length = measure_shadow_length(blob, inst.mask, sun.azimuth_deg)
        if length > 0:
            lengths_px.append(length)

    if not lengths_px:
        logger.debug("shadow: all shadow lengths zero for bbox=%s", inst.bbox)
        return None

    median_length_px = float(np.median(lengths_px))
    length_m = median_length_px * gsd_m

    elev_rad = math.radians(sun.elevation_deg)
    h = length_m * math.tan(elev_rad)

    # Propagate sun elevation uncertainty (multiplicative)
    delta_deg = _sun_elevation_delta(sun)
    h_lo = length_m * math.tan(math.radians(max(sun.elevation_deg - delta_deg, 0.1)))
    h_hi = length_m * math.tan(math.radians(min(sun.elevation_deg + delta_deg, 89.9)))
    sun_unc = (h_hi - h_lo) / 2.0

    # Shadow length uncertainty ~ 15% of measured length
    length_unc = h * 0.15

    combined_unc = math.sqrt(length_unc**2 + sun_unc**2)

    if not _sanity_check(h):
        logger.debug("shadow: rejected h=%.1fm (sanity) for bbox=%s", h, inst.bbox)
        return None

    logger.info(
        "shadow: h=%.1fm unc=%.1fm (shadow_L=%.1fpx, elev=%.1f°, delta=%.1f°) bbox=%s",
        h, combined_unc, median_length_px, sun.elevation_deg, delta_deg, inst.bbox,
    )
    return HeightEstimate(height_m=h, uncertainty_m=combined_unc, method="shadow")


# ---------------------------------------------------------------------------
# B. Facade (visible wall) height
# ---------------------------------------------------------------------------

def estimate_facade_height(
    inst: InstanceMask,
    image_rgb: np.ndarray,
    sun: SunPosition,
    gsd_m: float,
) -> Optional[HeightEstimate]:
    h_img, w_img = inst.mask.shape[:2]

    # Sun-opposite direction: facade is illuminated on the side away from shadow
    shadow_dir_rad = math.radians((sun.azimuth_deg + 180) % 360)
    dx = math.sin(shadow_dir_rad)
    dy = -math.cos(shadow_dir_rad)

    # Build a 10px directional strip outside the mask boundary
    mask_u8 = inst.mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    boundary_pts = contours[0].reshape(-1, 2)
    centroid_x = float(np.mean(boundary_pts[:, 0]))
    centroid_y = float(np.mean(boundary_pts[:, 1]))

    # Select boundary points on the sun-opposite side
    dot_products = (boundary_pts[:, 0] - centroid_x) * dx + (boundary_pts[:, 1] - centroid_y) * dy
    sun_side_pts = boundary_pts[dot_products > 0]
    if len(sun_side_pts) < 3:
        return None

    # Create strip mask: 10px outward from those boundary points
    strip_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for px, py in sun_side_pts:
        for step in range(1, 11):
            sx = int(round(px + dx * step))
            sy = int(round(py + dy * step))
            if 0 <= sx < w_img and 0 <= sy < h_img:
                strip_mask[sy, sx] = 255

    # Exclude pixels inside the building mask
    strip_mask[inst.mask] = 0
    strip_pixels = strip_mask > 0
    if np.count_nonzero(strip_pixels) < 5:
        return None

    # Check brightness in LAB: facade should be brighter than shadow threshold
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32) / 255.0

    strip_L = l_channel[strip_pixels]
    building_L = l_channel[inst.mask]

    # Facade must be bright (L > 0.5) and brighter than building mean
    if np.median(strip_L) <= 0.5 or np.median(strip_L) <= np.mean(building_L):
        return None

    # Measure perpendicular extent of the coherent bright strip
    bright_in_strip = strip_L > 0.5
    coherence = np.count_nonzero(bright_in_strip) / len(strip_L)
    if coherence < 0.4:
        return None

    # Perpendicular extent: max distance from boundary along the strip direction
    strip_coords = np.argwhere(strip_pixels)  # (row, col)
    if len(strip_coords) == 0:
        return None
    projections = strip_coords[:, 1] * dx + strip_coords[:, 0] * dy
    strip_width_px = float(projections.max() - projections.min())
    if strip_width_px < 1:
        return None

    height = strip_width_px * gsd_m
    unc = height * 0.4

    if not _sanity_check(height):
        return None

    logger.info("facade: h=%.1fm unc=%.1fm (strip=%.1fpx) bbox=%s", height, unc, strip_width_px, inst.bbox)
    return HeightEstimate(height_m=height, uncertainty_m=unc, method="facade")


# ---------------------------------------------------------------------------
# C. Lean (off-nadir displacement) height
# ---------------------------------------------------------------------------

def estimate_lean_height(
    inst: InstanceMask,
    image_center: tuple[float, float],
    gsd_m: float,
    off_nadir_deg: Optional[float] = None,
) -> Optional[HeightEstimate]:
    if off_nadir_deg is None or off_nadir_deg < 5.0:
        return None

    ys, xs = np.where(inst.mask)
    if len(xs) == 0:
        return None

    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    icx, icy = image_center

    # Direction from image center to building centroid
    dir_x, dir_y = cx - icx, cy - icy
    radial_dist = math.sqrt(dir_x**2 + dir_y**2)
    if radial_dist < 1:
        return None

    # Asymmetry: compare mask centroid to centroid of ground-facing boundary pixels
    mask_u8 = inst.mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    boundary_pts = contours[0].reshape(-1, 2).astype(np.float64)

    # Ground-facing side: boundary points closer to image center
    unit_dx, unit_dy = -dir_x / radial_dist, -dir_y / radial_dist
    dots = (boundary_pts[:, 0] - cx) * unit_dx + (boundary_pts[:, 1] - cy) * unit_dy
    ground_pts = boundary_pts[dots > 0]
    if len(ground_pts) < 3:
        return None

    ground_cx = float(np.mean(ground_pts[:, 0]))
    ground_cy = float(np.mean(ground_pts[:, 1]))

    displacement_px = math.sqrt((cx - ground_cx)**2 + (cy - ground_cy)**2)
    off_nadir_rad = math.radians(off_nadir_deg)

    if math.tan(off_nadir_rad) < 1e-6:
        return None

    height = displacement_px * gsd_m / math.tan(off_nadir_rad)
    unc = height * 0.5

    if not _sanity_check(height):
        return None

    logger.info(
        "lean: h=%.1fm unc=%.1fm (disp=%.1fpx, off_nadir=%.1f°) bbox=%s",
        height, unc, displacement_px, off_nadir_deg, inst.bbox,
    )
    return HeightEstimate(height_m=height, uncertainty_m=unc, method="lean")


# ---------------------------------------------------------------------------
# D. DAv2 depth calibration + per-building estimate
# ---------------------------------------------------------------------------

def calibrate_depth_to_height(
    building_instances: list[InstanceMask],
    depth_map: np.ndarray,
    min_points: int = 3,
) -> Optional[tuple[float, float, float]]:
    dav2_vals = []
    meas_vals = []

    for inst in building_instances:
        if inst.height_m is None:
            continue
        if inst.height_source not in ("measured",) and inst.height_confidence not in ("high", "medium"):
            continue

        # Resample mask to depth_map shape
        mask_resized = cv2.resize(
            inst.mask.astype(np.uint8), (depth_map.shape[1], depth_map.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        depth_vals = depth_map[mask_resized]
        if len(depth_vals) < 10:
            continue

        dav2_vals.append(float(np.median(depth_vals)))
        meas_vals.append(inst.height_m)

    if len(dav2_vals) < min_points:
        logger.warning("calibrate: only %d/%d anchor points, need %d", len(dav2_vals), min_points, min_points)
        return None

    dav2_arr = np.array(dav2_vals)
    meas_arr = np.array(meas_vals)

    a, b = np.polyfit(dav2_arr, meas_arr, 1)

    predicted = a * dav2_arr + b
    ss_res = float(np.sum((meas_arr - predicted) ** 2))
    ss_tot = float(np.sum((meas_arr - np.mean(meas_arr)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    residual_std = float(np.std(meas_arr - predicted))

    logger.info("calibrate: a=%.4f b=%.2f R²=%.3f residual_std=%.2fm n=%d", a, b, r_squared, residual_std, len(dav2_vals))

    if r_squared < 0.3:
        logger.warning("calibrate: R²=%.3f < 0.3, rejecting calibration", r_squared)
        return None

    return (a, b, residual_std)


def estimate_dav2_calibrated(
    inst: InstanceMask,
    depth_map: np.ndarray,
    calibration: tuple[float, float],
    calibration_residual: float,
) -> Optional[HeightEstimate]:
    a, b = calibration

    mask_resized = cv2.resize(
        inst.mask.astype(np.uint8), (depth_map.shape[1], depth_map.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    depth_vals = depth_map[mask_resized]
    if len(depth_vals) < 5:
        return None

    median_depth = float(np.median(depth_vals))
    h = a * median_depth + b
    unc = calibration_residual * 1.5

    if not _sanity_check(h):
        logger.debug("dav2_calibrated: rejected h=%.1fm (sanity) bbox=%s", h, inst.bbox)
        return None

    logger.info("dav2_calibrated: h=%.1fm unc=%.1fm (depth=%.4f) bbox=%s", h, unc, median_depth, inst.bbox)
    return HeightEstimate(height_m=h, uncertainty_m=unc, method="dav2_calibrated")


# ---------------------------------------------------------------------------
# E. Neighbour prior
# ---------------------------------------------------------------------------

def estimate_neighbour_prior(
    inst: InstanceMask,
    all_building_instances: list[InstanceMask],
    max_radius_px: float = 200,
) -> Optional[HeightEstimate]:
    ys, xs = np.where(inst.mask)
    if len(xs) == 0:
        return None
    cx, cy = float(np.mean(xs)), float(np.mean(ys))

    measured = []
    for other in all_building_instances:
        if other is inst or other.height_m is None:
            continue
        if other.height_source != "measured":
            continue
        oys, oxs = np.where(other.mask)
        if len(oxs) == 0:
            continue
        ocx, ocy = float(np.mean(oxs)), float(np.mean(oys))
        dist = math.sqrt((cx - ocx)**2 + (cy - ocy)**2)
        measured.append((dist, other.height_m))

    nearby = [(d, h) for d, h in measured if d <= max_radius_px]

    if not nearby:
        # Fall back to global median of all measured buildings
        all_heights = [h for _, h in measured]
        if not all_heights:
            return None
        h = float(np.median(all_heights))
        unc = max(float(np.std(all_heights)) if len(all_heights) > 1 else h * 0.5, h * 0.3, 3.0)
        logger.info("neighbour_prior: global median h=%.1fm unc=%.1fm (n=%d) bbox=%s", h, unc, len(all_heights), inst.bbox)
        return HeightEstimate(height_m=h, uncertainty_m=unc, method="neighbour_prior")

    weights = np.array([1.0 / max(d, 10.0) for d, _ in nearby])
    heights = np.array([h for _, h in nearby])

    h = float(np.average(heights, weights=weights))
    std = float(np.std(heights)) if len(heights) > 1 else h * 0.5
    unc = max(std, h * 0.3, 3.0)

    logger.info("neighbour_prior: h=%.1fm unc=%.1fm (n=%d neighbours) bbox=%s", h, unc, len(nearby), inst.bbox)
    return HeightEstimate(height_m=h, uncertainty_m=unc, method="neighbour_prior")


# ---------------------------------------------------------------------------
# Inverse-variance weighted fusion
# ---------------------------------------------------------------------------

def fuse_heights(estimates: list[Optional[HeightEstimate]]) -> Optional[FusedHeight]:
    valid = [e for e in estimates if e is not None]
    if not valid:
        return None

    # Floor uncertainty at 0.5m
    for e in valid:
        e.uncertainty_m = max(e.uncertainty_m, 0.5)

    weights = np.array([1.0 / (e.uncertainty_m ** 2) for e in valid])
    heights = np.array([e.height_m for e in valid])

    w_sum = float(np.sum(weights))
    h_fused = float(np.sum(weights * heights)) / w_sum
    sigma_fused = 1.0 / math.sqrt(w_sum)

    # Disagreement check: pairwise
    low_confidence = False
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            delta = abs(valid[i].height_m - valid[j].height_m)
            combined_sigma = math.sqrt(valid[i].uncertainty_m**2 + valid[j].uncertainty_m**2)
            if delta > combined_sigma:
                low_confidence = True
                logger.warning(
                    "DISAGREE: %s=%.1fm vs %s=%.1fm (delta=%.1fm > combined_sigma=%.1fm)",
                    valid[i].method, valid[i].height_m,
                    valid[j].method, valid[j].height_m,
                    delta, combined_sigma,
                )

    source = "measured" if any(e.method in _MEASURED_METHODS for e in valid) else "inferred"
    methods = [e.method for e in valid]

    logger.info(
        "fused: h=%.1fm sigma=%.1fm methods=%s source=%s low_conf=%s",
        h_fused, sigma_fused, methods, source, low_confidence,
    )

    return FusedHeight(
        height_m=h_fused,
        uncertainty_m=sigma_fused,
        methods=methods,
        estimates=valid,
        source=source,
        low_confidence_flag=low_confidence,
    )
