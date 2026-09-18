"""
Sun geometry for shadow-based height estimation.

Priority chain:
1. Product metadata (.IMD/.XML/.MTL/EXIF sidecar)
2. Computed via pysolar from date/time/location
3. Measured azimuth from shadow direction in image
4. User-supplied slider (last resort)
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SunPosition:
    elevation_deg: float      # angle above horizon (0-90)
    azimuth_deg: float        # compass bearing (0=N, 90=E, 180=S, 270=W)
    source: str               # "metadata" | "computed" | "estimated" | "user"
    confidence: str           # "high" | "medium" | "low"
    coherence_R: Optional[float] = None  # mean resultant length of shadow azimuths (0=random, 1=perfect)
    n_shadow_pairs: Optional[int] = None  # usable pairs after filtering


@dataclass
class SunSourceLog:
    """All available sun sources for cross-check logging."""
    metadata: Optional[dict] = None      # {"elevation": float, "azimuth": float, "format": str}
    computed: Optional[dict] = None      # {"elevation": float, "azimuth": float}
    measured_azimuth: Optional[float] = None
    measured_coherence_R: Optional[float] = None  # shadow direction coherence
    measured_n_pairs: Optional[int] = None
    slider: Optional[dict] = None        # {"elevation": float, "azimuth": float}
    cross_check: Optional[dict] = None   # {"azimuth_delta_deg": float, "consistent": bool}


def sun_from_datetime(
    lat: float, lon: float,
    dt: datetime,
) -> SunPosition:
    """Compute sun position from geographic coordinates and datetime."""
    from pysolar.solar import get_altitude, get_azimuth

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    elevation = get_altitude(lat, lon, dt)
    azimuth = get_azimuth(lat, lon, dt)

    # pysolar azimuth: 0=S, positive=W; convert to geographic: 0=N, 90=E
    geo_azimuth = (azimuth + 180) % 360

    logger.info(
        "Sun position from pysolar: elevation=%.1f°, azimuth=%.1f° (geo) "
        "for lat=%.4f, lon=%.4f at %s",
        elevation, geo_azimuth, lat, lon, dt.isoformat(),
    )

    return SunPosition(
        elevation_deg=elevation,
        azimuth_deg=geo_azimuth,
        source="computed",
        confidence="high",
    )


# ---------------------------------------------------------------------------
# Sidecar metadata parsing (.IMD / .XML / .MTL)
# ---------------------------------------------------------------------------

def _parse_imd(text: str) -> Optional[tuple[float, float]]:
    """Parse DigitalGlobe/Maxar .IMD file for meanSunEl/meanSunAz."""
    el_match = re.search(r'meanSunEl\s*=\s*([\d.+-]+)', text, re.IGNORECASE)
    az_match = re.search(r'meanSunAz\s*=\s*([\d.+-]+)', text, re.IGNORECASE)
    if el_match and az_match:
        return float(el_match.group(1)), float(az_match.group(1))
    return None


def _parse_mtl(text: str) -> Optional[tuple[float, float]]:
    """Parse Landsat .MTL file for SUN_ELEVATION/SUN_AZIMUTH."""
    el_match = re.search(r'SUN_ELEVATION\s*=\s*([\d.+-]+)', text, re.IGNORECASE)
    az_match = re.search(r'SUN_AZIMUTH\s*=\s*([\d.+-]+)', text, re.IGNORECASE)
    if el_match and az_match:
        return float(el_match.group(1)), float(az_match.group(1))
    return None


def _parse_xml(text: str) -> Optional[tuple[float, float]]:
    """Parse XML sidecar for sun elevation/azimuth tags."""
    el_tags = ['SUNEL', 'SUN_ELEVATION', 'meanSunEl', 'SunElevation']
    az_tags = ['SUNAZ', 'SUN_AZIMUTH', 'meanSunAz', 'SunAzimuth']

    el_val = None
    az_val = None
    for tag in el_tags:
        m = re.search(rf'<{tag}[^>]*>\s*([\d.+-]+)\s*</{tag}>', text, re.IGNORECASE)
        if m:
            el_val = float(m.group(1))
            break
    for tag in az_tags:
        m = re.search(rf'<{tag}[^>]*>\s*([\d.+-]+)\s*</{tag}>', text, re.IGNORECASE)
        if m:
            az_val = float(m.group(1))
            break
    if el_val is not None and az_val is not None:
        return el_val, az_val
    return None


def parse_sidecar_metadata(file_bytes: bytes, filename: str) -> Optional[SunPosition]:
    """
    Parse sun elevation/azimuth from a sidecar metadata file.
    Detects format from filename extension: .IMD, .XML, .MTL
    """
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    try:
        text = file_bytes.decode('utf-8', errors='replace')
    except Exception:
        return None

    result = None
    fmt = ext

    if ext == 'imd':
        result = _parse_imd(text)
    elif ext == 'mtl':
        result = _parse_mtl(text)
    elif ext == 'xml':
        result = _parse_xml(text)
    else:
        for parser, name in [(_parse_imd, 'imd'), (_parse_xml, 'xml'), (_parse_mtl, 'mtl')]:
            result = parser(text)
            if result:
                fmt = name
                break

    if result is None:
        logger.debug("No sun metadata found in %s", filename)
        return None

    el, az = result
    logger.info("Sun from sidecar (%s): elevation=%.1f°, azimuth=%.1f° [%s]", fmt, el, az, filename)
    return SunPosition(
        elevation_deg=el,
        azimuth_deg=az,
        source="metadata",
        confidence="high",
    )


def sun_from_metadata(image_bytes: bytes) -> Optional[SunPosition]:
    """
    Try to extract sun geometry from image EXIF metadata.
    Checks GPS + DateTime → pysolar computation.
    Returns None if no relevant metadata found.
    """
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes))
        exif = img.getexif()
        if not exif:
            return None

        gps_info = exif.get(34853)  # GPSInfo tag
        datetime_str = exif.get(36867) or exif.get(306)  # DateTimeOriginal or DateTime

        if gps_info and datetime_str:
            lat_ref = gps_info.get(1, 'N')
            lat_dms = gps_info.get(2, (0, 0, 0))
            lon_ref = gps_info.get(3, 'E')
            lon_dms = gps_info.get(4, (0, 0, 0))

            lat = lat_dms[0] + lat_dms[1] / 60 + lat_dms[2] / 3600
            if lat_ref == 'S':
                lat = -lat
            lon = lon_dms[0] + lon_dms[1] / 60 + lon_dms[2] / 3600
            if lon_ref == 'W':
                lon = -lon

            dt = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)

            pos = sun_from_datetime(lat, lon, dt)
            pos.source = "metadata"
            return pos

    except Exception as e:
        logger.debug("No usable sun metadata: %s", e)

    return None


def estimate_sun_from_shadows(
    image_rgb: np.ndarray,
    building_masks: list[np.ndarray],
    shadow_masks: list[np.ndarray],
    building_shadow_pairs: list[tuple[int, int]],
    min_offset_px: float = 5.0,
) -> SunPosition:
    """
    Estimate sun azimuth from shadow directions across multiple buildings.
    Sun elevation cannot be determined from direction alone — returns
    a default with low confidence.

    Pairs where the building-shadow centroid offset is below min_offset_px
    are rejected (sub-pixel offsets produce random bearings).
    """
    azimuths = []
    rejected = 0

    for bldg_idx, shadow_idx in building_shadow_pairs:
        bldg_mask = building_masks[bldg_idx]
        shadow_mask = shadow_masks[shadow_idx]

        bldg_coords = np.argwhere(bldg_mask)
        shadow_coords = np.argwhere(shadow_mask)

        if len(bldg_coords) == 0 or len(shadow_coords) == 0:
            continue

        bldg_center = bldg_coords.mean(axis=0)
        shadow_center = shadow_coords.mean(axis=0)

        dy = shadow_center[0] - bldg_center[0]
        dx = shadow_center[1] - bldg_center[1]

        offset = np.sqrt(dx * dx + dy * dy)
        if offset < min_offset_px:
            rejected += 1
            continue

        shadow_azimuth = np.degrees(np.arctan2(dx, -dy)) % 360
        sun_azimuth = (shadow_azimuth + 180) % 360
        azimuths.append(sun_azimuth)

    if rejected > 0:
        logger.info("Sun estimation: rejected %d/%d pairs with centroid offset < %.0fpx",
                     rejected, rejected + len(azimuths), min_offset_px)

    if not azimuths:
        logger.warning("No usable building-shadow pairs for sun estimation, using default")
        return SunPosition(
            elevation_deg=45.0,
            azimuth_deg=180.0,
            source="default",
            confidence="low",
        )

    az_rad = np.radians(azimuths)
    mean_sin = np.mean(np.sin(az_rad))
    mean_cos = np.mean(np.cos(az_rad))
    mean_azimuth = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360

    R = np.sqrt(mean_sin**2 + mean_cos**2)
    confidence = "medium" if R > 0.7 else "low"

    logger.info(
        "Estimated sun azimuth: %.1f° from %d pairs (R=%.2f, %s, %d rejected)",
        mean_azimuth, len(azimuths), R, confidence, rejected,
    )

    return SunPosition(
        elevation_deg=45.0,
        azimuth_deg=mean_azimuth,
        source="estimated",
        coherence_R=round(float(R), 3),
        n_shadow_pairs=len(azimuths),
        confidence=confidence,
    )


def cross_check_sun_sources(
    primary: SunPosition,
    measured_azimuth: Optional[float],
    threshold_deg: float = 15.0,
) -> dict:
    """Compare primary sun source against measured azimuth from shadows."""
    if measured_azimuth is None:
        return {"azimuth_delta_deg": None, "consistent": True}

    delta = abs(primary.azimuth_deg - measured_azimuth)
    if delta > 180:
        delta = 360 - delta

    consistent = delta <= threshold_deg

    if consistent:
        logger.info(
            "Sun cross-check: %s=%.1f° vs measured=%.1f° (Δ=%.1f°, consistent)",
            primary.source, primary.azimuth_deg, measured_azimuth, delta,
        )
    else:
        logger.warning(
            "Sun cross-check WARNING: %s=%.1f° vs measured=%.1f° (Δ=%.1f° > %.0f°)",
            primary.source, primary.azimuth_deg, measured_azimuth, delta, threshold_deg,
        )

    return {"azimuth_delta_deg": round(float(delta), 1), "consistent": bool(consistent)}


def cross_check_sun_sources_pair(
    src_a: str, az_a: float,
    src_b: str, az_b: float,
    threshold_deg: float = 15.0,
) -> dict:
    """Compare two named azimuth sources. Returns delta and consistency."""
    delta = abs(az_a - az_b)
    if delta > 180:
        delta = 360 - delta

    consistent = delta <= threshold_deg

    if consistent:
        logger.info(
            "Sun cross-check: %s=%.1f° vs %s=%.1f° (Δ=%.1f°, consistent)",
            src_a, az_a, src_b, az_b, delta,
        )
    else:
        logger.warning(
            "Sun cross-check WARNING: %s=%.1f° vs %s=%.1f° (Δ=%.1f° > %.0f°)",
            src_a, az_a, src_b, az_b, delta, threshold_deg,
        )

    return {
        "sources": [src_a, src_b],
        "azimuth_delta_deg": round(float(delta), 1),
        "consistent": bool(consistent),
    }
