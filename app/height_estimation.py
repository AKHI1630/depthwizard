"""
Multi-cue metric height estimation pipeline.

Ties together: SAM segmentation → shadow detection → sun geometry →
5 independent height methods → inverse-variance fusion → honest coverage.

Methods:
  A. Shadow length: h = L_shadow × tan(θ_sun)
  B. Facade (visible wall) height
  C. Building lean / relief displacement
  D. DAv2 depth calibrated against shadow anchors
  E. Neighbourhood prior (inverse-distance weighted)
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
from .sun_geometry import (
    SunPosition,
    SunSourceLog,
    cross_check_sun_sources,
    cross_check_sun_sources_pair,
    estimate_sun_from_shadows,
    sun_from_datetime,
    sun_from_metadata,
)
from .height_fusion import (
    HeightEstimate,
    FusedHeight,
    estimate_shadow_height,
    estimate_facade_height,
    estimate_lean_height,
    calibrate_depth_to_height,
    estimate_dav2_calibrated,
    estimate_neighbour_prior,
    fuse_heights,
)

logger = logging.getLogger(__name__)


@dataclass
class HeightResult:
    instances: list[InstanceMask]
    sun: SunPosition
    shadow_blobs: list[ShadowBlob]
    gsd_m: float
    timing: dict
    coverage: dict
    sun_sources: Optional[SunSourceLog] = None


def estimate_heights(
    image_rgb: np.ndarray,
    instances: list[InstanceMask],
    sun: Optional[SunPosition] = None,
    image_bytes: Optional[bytes] = None,
    gsd_m: float = 0.5,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    capture_dt: Optional[datetime] = None,
    depth_array: Optional[np.ndarray] = None,
    off_nadir_deg: Optional[float] = None,
    sidecar_sun: Optional[SunPosition] = None,
) -> HeightResult:
    """
    Full multi-cue height estimation pipeline.

    Args:
        image_rgb: H×W×3 uint8
        instances: pre-computed SAM instances with classification
        sun: sun position override (user slider). If None, attempts metadata → computed → estimated.
        image_bytes: original file bytes for metadata extraction
        gsd_m: ground sampling distance in metres/pixel
        lat, lon: geographic coordinates for sun computation
        capture_dt: capture datetime for sun computation
        depth_array: DAv2 depth map (run inline, not from cross-request state)
        off_nadir_deg: off-nadir angle for lean method
        sidecar_sun: sun position from sidecar metadata file
    """
    timing = {}
    t_total = time.perf_counter()
    sun_log = SunSourceLog()

    # 1. Determine sun position via priority chain
    t0 = time.perf_counter()

    # Track user slider if provided
    if sun is not None and sun.source == "user":
        sun_log.slider = {"elevation": sun.elevation_deg, "azimuth": sun.azimuth_deg}

    # Priority 1: sidecar metadata
    resolved_sun = None
    if sidecar_sun is not None:
        resolved_sun = sidecar_sun
        sun_log.metadata = {
            "elevation": sidecar_sun.elevation_deg,
            "azimuth": sidecar_sun.azimuth_deg,
            "format": "sidecar",
        }

    # Priority 1b: EXIF metadata
    if resolved_sun is None and image_bytes:
        exif_sun = sun_from_metadata(image_bytes)
        if exif_sun is not None:
            resolved_sun = exif_sun
            sun_log.metadata = {
                "elevation": exif_sun.elevation_deg,
                "azimuth": exif_sun.azimuth_deg,
                "format": "exif",
            }

    # Priority 2: computed from date/time/location
    if lat is not None and lon is not None and capture_dt is not None:
        computed = sun_from_datetime(lat, lon, capture_dt)
        sun_log.computed = {"elevation": computed.elevation_deg, "azimuth": computed.azimuth_deg}
        if resolved_sun is None:
            resolved_sun = computed

    # 2. Detect shadows
    building_instances = [inst for inst in instances if inst.label == "building"]
    building_masks = [inst.mask for inst in building_instances]

    t_shadow = time.perf_counter()
    shadow_blobs = detect_shadows(image_rgb, building_masks)
    shadow_blobs = match_shadows_to_buildings(shadow_blobs, building_masks)
    timing["shadow_detection"] = round(time.perf_counter() - t_shadow, 2)

    # Priority 3: measured azimuth from shadows (always run for cross-check)
    pairs = [(b.building_idx, i) for i, b in enumerate(shadow_blobs) if b.building_idx is not None]
    shadow_masks_list = [b.mask for b in shadow_blobs]
    measured_sun = estimate_sun_from_shadows(image_rgb, building_masks, shadow_masks_list, pairs)
    if measured_sun.source != "default":
        sun_log.measured_azimuth = measured_sun.azimuth_deg
    sun_log.measured_coherence_R = measured_sun.coherence_R
    sun_log.measured_n_pairs = measured_sun.n_shadow_pairs

    # Priority 3 gate: measured azimuth wins ONLY if BOTH conditions hold:
    #   (a) coherence R >= 0.5  (not random noise)
    #   (b) delta to any higher-priority azimuth <= 20 deg  (not a consistent artefact)
    # High R on a wrong direction (e.g. R=0.85 but delta=132 deg) is a coherent
    # artefact, not real shadows — must not override computed/metadata.
    measured_usable = False
    measured_reject_reason = None
    measured_delta_to_ref = None
    if measured_sun.source != "default" and measured_sun.coherence_R is not None:
        R = measured_sun.coherence_R

        # Always find the reference azimuth and compute delta (needed for
        # both the standard gate and the borderline combined test).
        ref_az = None
        ref_src = None
        if sun_log.metadata:
            ref_az = sun_log.metadata["azimuth"]
            ref_src = "metadata"
        elif sun_log.computed:
            ref_az = sun_log.computed["azimuth"]
            ref_src = "computed"
        elif sun is not None:
            ref_az = sun.azimuth_deg
            ref_src = "user"

        if ref_az is not None:
            delta = abs(measured_sun.azimuth_deg - ref_az)
            if delta > 180:
                delta = 360 - delta
            measured_delta_to_ref = delta

        if R < 0.5:
            measured_reject_reason = f"R={R:.2f} < 0.5"
        elif measured_delta_to_ref is not None and measured_delta_to_ref > 20:
            measured_reject_reason = (
                f"R={R:.2f} OK but measured az={measured_sun.azimuth_deg:.1f} vs "
                f"{ref_src}={ref_az:.1f} (delta={measured_delta_to_ref:.1f} > 20 deg)"
            )
        elif measured_delta_to_ref is not None:
            measured_usable = True
        else:
            measured_usable = True

    if measured_reject_reason:
        logger.warning("Measured shadow azimuth REJECTED: %s", measured_reject_reason)
    elif measured_usable:
        logger.info("Measured shadow azimuth ACCEPTED: az=%.1f R=%.2f",
                     measured_sun.azimuth_deg, measured_sun.coherence_R or 0)

    # Elevation inheritance: shadow measurement gives azimuth only, never
    # elevation. If measured azimuth wins, inherit elevation from the best
    # available source — never fall back to a 45 deg default.
    best_elevation = None
    best_elev_src = None
    if sun_log.metadata:
        best_elevation = sun_log.metadata["elevation"]
        best_elev_src = "metadata"
    elif sun_log.computed:
        best_elevation = sun_log.computed["elevation"]
        best_elev_src = "computed"
    elif sun is not None:
        best_elevation = sun.elevation_deg
        best_elev_src = "user"

    if resolved_sun is None:
        if measured_usable:
            elev = best_elevation if best_elevation is not None else 45.0
            resolved_sun = SunPosition(
                elevation_deg=elev,
                azimuth_deg=measured_sun.azimuth_deg,
                source="estimated",
                confidence=measured_sun.confidence,
                coherence_R=measured_sun.coherence_R,
                n_shadow_pairs=measured_sun.n_shadow_pairs,
            )
            if best_elevation is not None:
                logger.info("Sun: measured az=%.1f + %s elevation=%.1f",
                            measured_sun.azimuth_deg, best_elev_src, elev)
            else:
                logger.warning("Sun: measured az=%.1f but NO elevation source — using 45 deg default",
                               measured_sun.azimuth_deg)
        elif sun is not None:
            resolved_sun = sun
            logger.info("Sun resolved from user slider (measured rejected: %s)",
                        measured_reject_reason or "no valid pairs")
        else:
            resolved_sun = SunPosition(
                elevation_deg=45.0, azimuth_deg=180.0,
                source="default", confidence="low",
            )
            logger.warning("No usable sun source — using default 45 deg / 180 deg")

    # Cross-check: compare independent azimuth sources pairwise.
    # Collect all available azimuths with their provenance.
    available_azimuths: list[tuple[str, float]] = []
    if sun_log.metadata:
        available_azimuths.append(("metadata", sun_log.metadata["azimuth"]))
    if sun_log.computed:
        available_azimuths.append(("computed", sun_log.computed["azimuth"]))
    if sun_log.measured_azimuth is not None:
        available_azimuths.append(("measured", sun_log.measured_azimuth))
    if sun_log.slider:
        available_azimuths.append(("slider", sun_log.slider["azimuth"]))

    if len(available_azimuths) >= 2:
        # Pick the two highest-priority independent sources for the headline check.
        # Priority: metadata > computed > measured > slider
        src_a, az_a = available_azimuths[0]
        src_b, az_b = available_azimuths[1]
        sun_log.cross_check = cross_check_sun_sources_pair(src_a, az_a, src_b, az_b)
    elif len(available_azimuths) == 1:
        src_only, _ = available_azimuths[0]
        sun_log.cross_check = {
            "azimuth_delta_deg": None,
            "consistent": None,
            "note": f"single source ({src_only}) — no independent validation",
        }
        logger.info("Sun cross-check: only %s azimuth available — no independent validation", src_only)

    timing["sun_geometry"] = round(time.perf_counter() - t0, 2)
    logger.info("Sun: elevation=%.1f°, azimuth=%.1f°, source=%s, confidence=%s",
                resolved_sun.elevation_deg, resolved_sun.azimuth_deg,
                resolved_sun.source, resolved_sun.confidence)

    # Log all available sun sources
    sources_str = []
    if sun_log.metadata:
        sources_str.append(f"metadata={sun_log.metadata['elevation']:.1f}°/{sun_log.metadata['azimuth']:.1f}°")
    if sun_log.computed:
        sources_str.append(f"computed={sun_log.computed['elevation']:.1f}°/{sun_log.computed['azimuth']:.1f}°")
    if sun_log.measured_azimuth is not None:
        sources_str.append(f"measured_az={sun_log.measured_azimuth:.1f}°")
    if sun_log.slider:
        sources_str.append(f"slider={sun_log.slider['elevation']:.1f}°/{sun_log.slider['azimuth']:.1f}°")
    if sources_str:
        logger.info("Sun sources: %s", ", ".join(sources_str))

    # 3. Multi-cue height estimation
    t_height = time.perf_counter()
    image_center = (image_rgb.shape[1] / 2.0, image_rgb.shape[0] / 2.0)

    # Shadow feasibility pre-check: at high sun elevation, shadows are too
    # short to measure. h_min = min_offset_px * gsd * tan(elevation).
    shadow_feasibility = None
    if resolved_sun.elevation_deg > 0:
        from .shadow_detection import MIN_OFFSET_PX
        h_min = MIN_OFFSET_PX * gsd_m * np.tan(np.radians(resolved_sun.elevation_deg))
        shadow_feasibility = {
            "h_min_m": round(float(h_min), 1),
            "sun_elevation_deg": round(float(resolved_sun.elevation_deg), 1),
            "gsd_m": gsd_m,
            "min_offset_px": MIN_OFFSET_PX,
        }
        if resolved_sun.elevation_deg > 60:
            logger.warning(
                "Sun elevation %.1f° > 60° — shadows are very short. "
                "Minimum measurable building height: %.1f m at %.2f m GSD. "
                "Imagery may be unsuitable for shadow photogrammetry.",
                resolved_sun.elevation_deg, h_min, gsd_m,
            )
            shadow_feasibility["warning"] = (
                f"Sun elevation {resolved_sun.elevation_deg:.0f}° is too high for reliable "
                f"shadow photogrammetry. Buildings shorter than {h_min:.1f} m cannot be measured."
            )
        else:
            logger.info(
                "Shadow feasibility: h_min=%.1f m at %.1f° sun, %.2f m GSD",
                h_min, resolved_sun.elevation_deg, gsd_m,
            )

    # Quality gate — two tiers:
    #
    # STANDARD:   R_weighted >= 0.5              -> "measured"
    # BORDERLINE: p_combined < 0.01 AND delta <= 10 deg  -> "measured (borderline)"
    #
    # Combined Rayleigh-Azimuth test.  Under H0 (uniform random shadow
    # directions), the mean resultant length R and the circular mean
    # direction are independent.  Their p-values multiply:
    #   p_R     = exp(-n * R^2)        Rayleigh test for non-uniformity
    #   p_delta = 2 * delta / 360      prob mean falls within delta of ref
    #   p_combined = p_R * p_delta
    # The delta <= 10 deg hard gate prevents high-n artefacts (coherent
    # but wrong direction) from sneaking through on p_R alone.
    shadow_coherence_R = measured_sun.coherence_R
    shadow_n_pairs = measured_sun.n_shadow_pairs
    shadow_reliable = (shadow_coherence_R is not None
                       and shadow_coherence_R >= 0.5
                       and measured_delta_to_ref is not None
                       and measured_delta_to_ref <= 20.0)
    shadow_borderline = False
    shadow_p_combined = None

    if (not shadow_reliable
            and shadow_coherence_R is not None
            and shadow_n_pairs is not None
            and shadow_n_pairs >= 3
            and measured_delta_to_ref is not None):
        import math as _math
        p_R = _math.exp(-shadow_n_pairs * shadow_coherence_R ** 2)
        p_delta = min(1.0, 2.0 * measured_delta_to_ref / 360.0)
        shadow_p_combined = p_R * p_delta
        if shadow_p_combined < 0.01 and measured_delta_to_ref <= 10.0:
            shadow_borderline = True
            logger.info(
                "Shadow coherence BORDERLINE: R=%.3f, delta=%.1f deg, "
                "n=%d, p_combined=%.6f < 0.01 — physically consistent, "
                "coherence marginal",
                shadow_coherence_R, measured_delta_to_ref,
                shadow_n_pairs, shadow_p_combined,
            )

    if not shadow_reliable and not shadow_borderline:
        R_str = f"{shadow_coherence_R:.2f}" if shadow_coherence_R is not None else "N/A"
        logger.warning(
            "Shadow coherence R=%s — shadow detection unreliable for this image. "
            "Shadow-derived heights will be labelled RELATIVE, not metric.",
            R_str,
        )

    if resolved_sun.elevation_deg <= 0:
        logger.warning("Sun below horizon (%.1f°) — no shadow heights possible", resolved_sun.elevation_deg)
        for inst in building_instances:
            inst.height_m = None
            inst.height_confidence = "unavailable"
            inst.height_uncertainty_m = None
            inst.height_methods = []
            inst.height_source = None
    else:
        # Phase A: shadow heights for all buildings
        for inst in building_instances:
            est = estimate_shadow_height(inst, shadow_blobs, building_masks, resolved_sun, gsd_m)
            if est is not None:
                inst.height_m = round(est.height_m, 2)
                inst.height_uncertainty_m = round(est.uncertainty_m, 2)
                inst.shadow_length_px = round(est.height_m / (gsd_m * np.tan(np.radians(resolved_sun.elevation_deg))), 1) if gsd_m > 0 else None
                if shadow_reliable:
                    inst.height_confidence = "high" if resolved_sun.confidence == "high" else "medium"
                    inst.height_source = "measured"
                elif shadow_borderline:
                    inst.height_confidence = "borderline"
                    inst.height_source = "measured"
                else:
                    inst.height_confidence = "low"
                    inst.height_source = "relative"
                inst.height_methods = ["shadow"]

        # Phase D: DAv2 calibration using shadow anchors
        dav2_calibration = None
        if depth_array is not None and (shadow_reliable or shadow_borderline):
            dav2_calibration = calibrate_depth_to_height(building_instances, depth_array)

        # Phase B/C/D/E: additional methods per building
        for inst in building_instances:
            estimates: list[Optional[HeightEstimate]] = []

            # A: shadow (include in fusion only if coherent)
            if inst.height_m is not None and inst.height_source in ("measured", "relative"):
                shadow_est = HeightEstimate(
                    height_m=inst.height_m,
                    uncertainty_m=inst.height_uncertainty_m or 1.0,
                    method="shadow",
                )
                if shadow_borderline:
                    shadow_est.uncertainty_m = max(shadow_est.uncertainty_m, inst.height_m * 0.3)
                elif not shadow_reliable:
                    shadow_est.uncertainty_m = max(shadow_est.uncertainty_m, inst.height_m * 0.5)
                estimates.append(shadow_est)

            # B: facade
            facade_est = estimate_facade_height(inst, image_rgb, resolved_sun, gsd_m)
            if facade_est is not None:
                estimates.append(facade_est)

            # C: lean
            lean_est = estimate_lean_height(inst, image_center, gsd_m, off_nadir_deg)
            if lean_est is not None:
                estimates.append(lean_est)

            # D: DAv2 calibrated (only if shadow anchors were reliable)
            if depth_array is not None and dav2_calibration is not None:
                a, b, residual_std = dav2_calibration
                dav2_est = estimate_dav2_calibrated(inst, depth_array, (a, b), residual_std)
                if dav2_est is not None:
                    estimates.append(dav2_est)

            # E: neighbour prior (only if no direct measurement)
            has_measured = any(e.method in ("shadow", "facade", "lean") for e in estimates if e is not None)
            if not has_measured:
                nbr_est = estimate_neighbour_prior(inst, building_instances)
                if nbr_est is not None:
                    estimates.append(nbr_est)

            # Fuse all estimates
            if estimates:
                fused = fuse_heights(estimates)
                if fused is not None:
                    inst.height_m = round(fused.height_m, 2)
                    inst.height_uncertainty_m = round(fused.uncertainty_m, 2)
                    inst.height_methods = fused.methods
                    if not shadow_reliable and not shadow_borderline:
                        inst.height_source = "relative"
                        inst.height_confidence = "low"
                    elif shadow_borderline:
                        inst.height_source = fused.source
                        inst.height_confidence = "borderline"
                    else:
                        inst.height_source = fused.source
                        if fused.low_confidence_flag:
                            inst.height_confidence = "low"
                        elif fused.source == "measured":
                            inst.height_confidence = "high" if resolved_sun.confidence == "high" else "medium"
                        else:
                            inst.height_confidence = "low"
            else:
                inst.height_m = None
                inst.height_confidence = "no_measurement"
                inst.height_uncertainty_m = None
                inst.height_methods = []
                inst.height_source = None

    # Check for suspicious patterns
    shadow_pxs = [i.shadow_length_px for i in building_instances
                  if i.shadow_length_px is not None]
    if shadow_pxs:
        if all(v == 0 for v in shadow_pxs):
            logger.warning("ALL shadow lengths are ZERO — heights are fallbacks, not measurements")
        elif len(set(round(v, 0) for v in shadow_pxs)) == 1 and len(shadow_pxs) > 2:
            logger.warning(
                "ALL %d shadow lengths are identical (%.1fpx) — likely detection artefact",
                len(shadow_pxs), shadow_pxs[0],
            )

    timing["height_computation"] = round(time.perf_counter() - t_height, 2)
    timing["total"] = round(time.perf_counter() - t_total, 2)

    # 4. Honest coverage stats
    total_bldg = len(building_instances)
    directly_measured = sum(1 for i in building_instances
                           if i.height_source == "measured" and i.height_confidence != "borderline")
    borderline_measured = sum(1 for i in building_instances
                             if i.height_source == "measured" and i.height_confidence == "borderline")
    relative_only = sum(1 for i in building_instances if i.height_source == "relative")
    inferred = sum(1 for i in building_instances if i.height_source == "inferred")
    failed = sum(1 for i in building_instances if i.height_m is None)

    building_areas = sorted([i.area for i in building_instances])
    area_hist = None
    if building_areas:
        pcts = np.percentile(building_areas, [10, 25, 50, 75, 90])
        area_hist = {
            "p10": int(pcts[0]), "p25": int(pcts[1]), "p50": int(pcts[2]),
            "p75": int(pcts[3]), "p90": int(pcts[4]),
            "min": building_areas[0], "max": building_areas[-1],
        }

    coverage = {
        "total_buildings": total_bldg,
        "directly_measured": directly_measured,
        "borderline": borderline_measured,
        "relative": relative_only,
        "inferred": inferred,
        "failed": failed,
        "shadow_coherence_R": round(float(shadow_coherence_R), 3) if shadow_coherence_R is not None else None,
        "shadow_reliable": bool(shadow_reliable),
        "shadow_borderline": bool(shadow_borderline),
        "shadow_p_combined": round(float(shadow_p_combined), 6) if shadow_p_combined is not None else None,
        "area_histogram_px": area_hist,
        "shadow_feasibility": shadow_feasibility,
    }

    if shadow_reliable:
        logger.info("Height coverage: %d/%d measured, %d inferred, %d failed (R=%.2f, shadows reliable)",
                     directly_measured, total_bldg, inferred, failed, shadow_coherence_R or 0)
    elif shadow_borderline:
        logger.info(
            "Height coverage: %d/%d borderline measured, %d inferred, %d failed "
            "(R=%.3f, p_combined=%.6f, shadows borderline)",
            borderline_measured, total_bldg, inferred, failed,
            shadow_coherence_R or 0, shadow_p_combined or 0,
        )
    else:
        logger.warning(
            "Height coverage: %d relative, %d inferred, %d failed of %d buildings — "
            "shadow detection UNRELIABLE (R=%.2f), heights are NOT metric",
            relative_only, inferred, failed, total_bldg, shadow_coherence_R or 0,
        )

    # Per-building table log
    header = f"{'id':>3} | {'area_px':>7} | {'shadow':>7} | {'dav2_cal':>8} | {'nbr_pri':>7} | {'fused':>6} | {'±unc':>5} | {'methods':<20} | {'source':<10}"
    logger.info("Per-building table:\n%s\n%s", header, "-" * len(header))
    for idx, inst in enumerate(building_instances):
        methods_str = "+".join(inst.height_methods) if inst.height_methods else "--"
        h_str = f"{inst.height_m:.1f}" if inst.height_m is not None else "--"
        u_str = f"{inst.height_uncertainty_m:.1f}" if inst.height_uncertainty_m is not None else "--"
        src_str = inst.height_source or "--"
        logger.info(
            "%3d | %7d | %7s | %8s | %7s | %6s | %5s | %-20s | %-10s",
            idx + 1, inst.area, "--", "--", "--",
            h_str, u_str, methods_str, src_str,
        )

    return HeightResult(
        instances=instances,
        sun=resolved_sun,
        shadow_blobs=shadow_blobs,
        gsd_m=gsd_m,
        timing=timing,
        coverage=coverage,
        sun_sources=sun_log,
    )


def height_result_to_json(result: HeightResult, image_width: int = 512, image_height: int = 512) -> dict:
    """Convert HeightResult to JSON-serialisable dict."""
    from .sam_segmentation import instances_to_json

    buildings = [i for i in result.instances if i.label == "building"]
    heights = [i.height_m for i in buildings if i.height_m is not None]

    sun_dict = {
        "elevation_deg": round(float(result.sun.elevation_deg), 1),
        "azimuth_deg": round(float(result.sun.azimuth_deg), 1),
        "source": result.sun.source,
        "confidence": result.sun.confidence,
    }

    if result.sun_sources:
        sl = result.sun_sources
        sun_dict["all_sources"] = {}
        if sl.metadata:
            sun_dict["all_sources"]["metadata"] = sl.metadata
        if sl.computed:
            sun_dict["all_sources"]["computed"] = sl.computed
        if sl.measured_azimuth is not None:
            sun_dict["all_sources"]["measured_azimuth"] = {
                "azimuth": float(sl.measured_azimuth),
                "coherence_R": float(sl.measured_coherence_R) if sl.measured_coherence_R is not None else None,
                "n_pairs": int(sl.measured_n_pairs) if sl.measured_n_pairs is not None else None,
            }
        if sl.slider:
            sun_dict["all_sources"]["slider"] = sl.slider
        if sl.cross_check:
            sun_dict["cross_check"] = sl.cross_check

    return {
        "sun": sun_dict,
        "gsd_m": result.gsd_m,
        "coverage": result.coverage,
        "image_width": image_width,
        "image_height": image_height,
        "height_stats": {
            "min_m": round(min(heights), 2) if heights else None,
            "max_m": round(max(heights), 2) if heights else None,
            "mean_m": round(float(np.mean(heights)), 2) if heights else None,
            "median_m": round(float(np.median(heights)), 2) if heights else None,
            "std_m": round(float(np.std(heights)), 2) if len(heights) > 1 else None,
        },
        "timing": result.timing,
        "instances": instances_to_json(result.instances, image_shape=(image_height, image_width)),
    }
