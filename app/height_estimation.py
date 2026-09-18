"""
Shadow-based metric height estimation pipeline.

Ties together: SAM segmentation → shadow detection → sun geometry →
per-building height computation with confidence and error bars.

h = L_shadow * tan(θ_sun)

where L_shadow is the median shadow length in metres and θ_sun
is the sun elevation angle.
"""
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .sam_segmentation import InstanceMask, segment_and_classify
from .shadow_detection import (
    ShadowBlob,
    detect_shadows,
    match_shadows_to_buildings,
    measure_shadow_length,
)
from .sun_geometry import SunPosition, sun_from_datetime, sun_from_metadata, estimate_sun_from_shadows

logger = logging.getLogger(__name__)


@dataclass
class HeightResult:
    instances: list[InstanceMask]
    sun: SunPosition
    shadow_blobs: list[ShadowBlob]
    gsd_m: float                # ground sampling distance in metres/pixel
    timing: dict
    coverage: dict              # {total_buildings, with_height, interpolated, no_height}


def estimate_heights(
    image_rgb: np.ndarray,
    instances: list[InstanceMask],
    sun: Optional[SunPosition] = None,
    image_bytes: Optional[bytes] = None,
    gsd_m: float = 0.5,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    capture_dt: Optional[datetime] = None,
) -> HeightResult:
    """
    Full height estimation pipeline.

    Args:
        image_rgb: H×W×3 uint8
        instances: pre-computed SAM instances with classification
        sun: sun position (if known). If None, attempts metadata → computed → estimated.
        image_bytes: original file bytes for metadata extraction
        gsd_m: ground sampling distance in metres/pixel (default 0.5m for high-res sat)
        lat, lon: geographic coordinates for sun computation
        capture_dt: capture datetime for sun computation

    Returns:
        HeightResult with per-building heights, confidence, and coverage stats
    """
    timing = {}
    t_total = time.perf_counter()

    # 1. Determine sun position
    t0 = time.perf_counter()
    if sun is None:
        if image_bytes:
            sun = sun_from_metadata(image_bytes)
        if sun is None and lat is not None and lon is not None and capture_dt is not None:
            sun = sun_from_datetime(lat, lon, capture_dt)

    # 2. Detect shadows
    building_instances = [inst for inst in instances if inst.label == "building"]
    building_masks = [inst.mask for inst in building_instances]

    t_shadow = time.perf_counter()
    shadow_blobs = detect_shadows(image_rgb, building_masks)
    shadow_blobs = match_shadows_to_buildings(shadow_blobs, building_masks)
    timing["shadow_detection"] = round(time.perf_counter() - t_shadow, 2)

    # 3. If no sun position yet, estimate from shadows
    if sun is None:
        pairs = [(b.building_idx, i) for i, b in enumerate(shadow_blobs) if b.building_idx is not None]
        shadow_masks_list = [b.mask for b in shadow_blobs]
        sun = estimate_sun_from_shadows(image_rgb, building_masks, shadow_masks_list, pairs)

    timing["sun_geometry"] = round(time.perf_counter() - t0, 2)
    logger.info("Sun: elevation=%.1f°, azimuth=%.1f°, source=%s, confidence=%s",
                sun.elevation_deg, sun.azimuth_deg, sun.source, sun.confidence)

    # 4. Measure shadow lengths and compute heights
    t_height = time.perf_counter()

    if sun.elevation_deg <= 0:
        logger.warning("Sun below horizon (%.1f°) — no shadow heights possible", sun.elevation_deg)
        for inst in building_instances:
            inst.height_m = None
            inst.height_confidence = "unavailable"
            inst.height_uncertainty_m = None
    else:
        tan_elev = np.tan(np.radians(sun.elevation_deg))

        heights_computed = []
        for inst in building_instances:
            matched_shadows = [b for b in shadow_blobs if b.building_idx is not None
                               and building_masks[b.building_idx] is inst.mask]
            if not matched_shadows:
                inst.height_m = None
                inst.height_confidence = "no_shadow"
                inst.height_uncertainty_m = None
                continue

            shadow_lengths_px = []
            for sb in matched_shadows:
                length = measure_shadow_length(sb, inst.mask, sun.azimuth_deg)
                if length > 0:
                    sb.shadow_length_px = length
                    shadow_lengths_px.append(length)

            if not shadow_lengths_px:
                inst.height_m = None
                inst.height_confidence = "no_measurable_shadow"
                inst.height_uncertainty_m = None
                continue

            median_length_px = np.median(shadow_lengths_px)
            shadow_length_m = median_length_px * gsd_m
            height_m = shadow_length_m * tan_elev

            # Uncertainty from shadow length variance + sun angle uncertainty
            if len(shadow_lengths_px) > 1:
                length_std_px = np.std(shadow_lengths_px)
                length_std_m = length_std_px * gsd_m
            else:
                length_std_m = 1.0 * gsd_m  # assume 1px uncertainty

            # Propagate sun elevation uncertainty (±5° if estimated)
            elev_uncertainty = 5.0 if sun.confidence != "high" else 2.0
            elev_lo = max(1, sun.elevation_deg - elev_uncertainty)
            elev_hi = sun.elevation_deg + elev_uncertainty
            h_lo = shadow_length_m * np.tan(np.radians(elev_lo))
            h_hi = shadow_length_m * np.tan(np.radians(elev_hi))
            height_uncertainty = max(abs(h_hi - height_m), abs(height_m - h_lo), length_std_m * tan_elev)

            # Sanity clamp
            if height_m < 1.0:
                inst.height_m = None
                inst.height_confidence = "too_short"
                inst.height_uncertainty_m = None
                continue
            if height_m > 300:
                inst.height_m = None
                inst.height_confidence = "unreasonable"
                inst.height_uncertainty_m = None
                continue

            inst.height_m = round(height_m, 2)
            inst.height_uncertainty_m = round(height_uncertainty, 2)
            inst.height_confidence = "high" if sun.confidence == "high" and len(shadow_lengths_px) > 1 else "medium"
            heights_computed.append(height_m)

        # 5. Interpolate missing heights from neighbourhood median
        if heights_computed:
            neighbourhood_median = float(np.median(heights_computed))
            for inst in building_instances:
                if inst.height_m is None and inst.height_confidence in ("no_shadow", "no_measurable_shadow"):
                    inst.height_m = round(neighbourhood_median, 2)
                    inst.height_confidence = "interpolated"
                    inst.height_uncertainty_m = round(float(np.std(heights_computed)) if len(heights_computed) > 1 else neighbourhood_median * 0.3, 2)

    timing["height_computation"] = round(time.perf_counter() - t_height, 2)
    timing["total"] = round(time.perf_counter() - t_total, 2)

    # 6. Coverage stats
    total_bldg = len(building_instances)
    with_height = sum(1 for i in building_instances if i.height_m is not None and i.height_confidence not in ("interpolated",))
    interpolated = sum(1 for i in building_instances if i.height_confidence == "interpolated")
    no_height = sum(1 for i in building_instances if i.height_m is None)

    coverage = {
        "total_buildings": total_bldg,
        "with_confident_height": with_height,
        "interpolated": interpolated,
        "no_height": no_height,
        "coverage_pct": round(100 * (with_height + interpolated) / max(total_bldg, 1), 1),
    }

    logger.info("Height coverage: %d/%d confident, %d interpolated, %d no height (%.1f%% coverage)",
                with_height, total_bldg, interpolated, no_height, coverage["coverage_pct"])

    return HeightResult(
        instances=instances,
        sun=sun,
        shadow_blobs=shadow_blobs,
        gsd_m=gsd_m,
        timing=timing,
        coverage=coverage,
    )


def height_result_to_json(result: HeightResult) -> dict:
    """Convert HeightResult to JSON-serialisable dict."""
    from .sam_segmentation import instances_to_json

    buildings = [i for i in result.instances if i.label == "building"]
    heights = [i.height_m for i in buildings if i.height_m is not None]

    return {
        "sun": {
            "elevation_deg": round(result.sun.elevation_deg, 1),
            "azimuth_deg": round(result.sun.azimuth_deg, 1),
            "source": result.sun.source,
            "confidence": result.sun.confidence,
        },
        "gsd_m": result.gsd_m,
        "coverage": result.coverage,
        "height_stats": {
            "min_m": round(min(heights), 2) if heights else None,
            "max_m": round(max(heights), 2) if heights else None,
            "mean_m": round(float(np.mean(heights)), 2) if heights else None,
            "median_m": round(float(np.median(heights)), 2) if heights else None,
            "std_m": round(float(np.std(heights)), 2) if len(heights) > 1 else None,
        },
        "timing": result.timing,
        "instances": instances_to_json(result.instances),
    }
