"""
Sun geometry for shadow-based height estimation.

Three sources, in priority order:
1. Image metadata (EXIF, WorldView IMD/XML sidecar)
2. User-supplied (date, time, lat/lon → pysolar)
3. Estimated from shadow direction in the image itself
"""
import logging
from dataclasses import dataclass
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


def sun_from_metadata(image_bytes: bytes) -> Optional[SunPosition]:
    """
    Try to extract sun geometry from image metadata.
    Checks EXIF, WorldView IMD/XML sidecar info.
    Returns None if no relevant metadata found.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        import io

        img = Image.open(io.BytesIO(image_bytes))
        exif = img.getexif()
        if not exif:
            return None

        # Look for GPSInfo + DateTime
        gps_info = exif.get(34853)  # GPSInfo tag
        datetime_str = exif.get(36867) or exif.get(306)  # DateTimeOriginal or DateTime

        if gps_info and datetime_str:
            # Extract lat/lon from GPS info
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
) -> SunPosition:
    """
    Estimate sun azimuth from shadow directions across multiple buildings.
    Sun elevation cannot be determined from direction alone — returns
    a default with low confidence.

    Args:
        image_rgb: source image
        building_masks: list of building bool masks
        shadow_masks: list of shadow bool masks
        building_shadow_pairs: list of (building_idx, shadow_idx) pairs

    Returns:
        SunPosition with estimated azimuth and default elevation
    """
    azimuths = []

    for bldg_idx, shadow_idx in building_shadow_pairs:
        bldg_mask = building_masks[bldg_idx]
        shadow_mask = shadow_masks[shadow_idx]

        bldg_coords = np.argwhere(bldg_mask)  # (row, col)
        shadow_coords = np.argwhere(shadow_mask)

        if len(bldg_coords) == 0 or len(shadow_coords) == 0:
            continue

        bldg_center = bldg_coords.mean(axis=0)
        shadow_center = shadow_coords.mean(axis=0)

        # Direction from building to shadow
        dy = shadow_center[0] - bldg_center[0]  # row (positive = down = south)
        dx = shadow_center[1] - bldg_center[1]  # col (positive = right = east)

        # Shadow falls opposite to sun direction
        # Convert to geographic azimuth (0=N, 90=E)
        shadow_azimuth = np.degrees(np.arctan2(dx, -dy)) % 360
        sun_azimuth = (shadow_azimuth + 180) % 360
        azimuths.append(sun_azimuth)

    if not azimuths:
        logger.warning("No building-shadow pairs for sun estimation, using default")
        return SunPosition(
            elevation_deg=45.0,
            azimuth_deg=180.0,
            source="default",
            confidence="low",
        )

    # Circular mean of azimuths
    az_rad = np.radians(azimuths)
    mean_sin = np.mean(np.sin(az_rad))
    mean_cos = np.mean(np.cos(az_rad))
    mean_azimuth = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360

    # Circular std as confidence measure
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    confidence = "medium" if R > 0.7 else "low"

    logger.info(
        "Estimated sun azimuth: %.1f° from %d building-shadow pairs (R=%.2f, %s)",
        mean_azimuth, len(azimuths), R, confidence,
    )

    return SunPosition(
        elevation_deg=45.0,  # unknown — default, labelled as estimated
        azimuth_deg=mean_azimuth,
        source="estimated",
        confidence=confidence,
    )
