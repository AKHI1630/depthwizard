"""
Shadow detection for satellite imagery.

Detects cast shadows as dark, low-saturation regions adjacent to
building footprints. Shadows are matched to their casting buildings
for height computation.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShadowBlob:
    mask: np.ndarray         # bool H×W
    area_px: int
    centroid: tuple          # (x, y)
    building_idx: Optional[int] = None
    shadow_length_px: Optional[float] = None


def detect_shadows(
    image_rgb: np.ndarray,
    building_masks: list[np.ndarray],
    l_thresh: float = 0.35,
    s_thresh: float = 0.25,
    min_area_px: int = 50,
) -> list[ShadowBlob]:
    """
    Detect shadow regions via thresholding in LAB colour space.

    Method:
    1. Convert to LAB, threshold on low L (dark) and low saturation (desaturated)
    2. Morphological cleanup to remove noise
    3. Filter: must be adjacent to a building mask (within dilation distance)
    4. Reject dark roofs and dark pavement by checking adjacency

    Args:
        image_rgb: H×W×3 uint8 RGB image
        building_masks: list of bool masks, one per building
        l_thresh: lightness threshold (0-1 scale, fraction of 255)
        s_thresh: saturation threshold in HSV (fraction of 255)
        min_area_px: minimum shadow blob area

    Returns:
        List of ShadowBlob objects matched to buildings
    """
    h, w = image_rgb.shape[:2]

    # Convert to LAB for lightness, HSV for saturation
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    l_channel = lab[:, :, 0].astype(np.float32) / 255.0
    s_channel = hsv[:, :, 1].astype(np.float32) / 255.0

    # Dark AND desaturated = shadow candidate
    shadow_candidate = (l_channel < l_thresh) & (s_channel < s_thresh)

    # Exclude building interiors — shadows are OUTSIDE buildings
    all_buildings = np.zeros((h, w), dtype=bool)
    for bm in building_masks:
        all_buildings |= bm
    shadow_candidate &= ~all_buildings

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_u8 = shadow_candidate.astype(np.uint8) * 255
    shadow_u8 = cv2.morphologyEx(shadow_u8, cv2.MORPH_CLOSE, kernel)
    shadow_u8 = cv2.morphologyEx(shadow_u8, cv2.MORPH_OPEN, kernel)

    # Adjacency check: dilate building masks, shadow must overlap with dilation
    building_dilated = np.zeros((h, w), dtype=np.uint8)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    for bm in building_masks:
        bm_u8 = bm.astype(np.uint8) * 255
        dilated = cv2.dilate(bm_u8, dilate_kernel)
        building_dilated = cv2.bitwise_or(building_dilated, dilated)

    # Shadow must be adjacent to at least one building
    shadow_u8 = cv2.bitwise_and(shadow_u8, building_dilated)

    # Connected components
    num_labels, labels = cv2.connectedComponents(shadow_u8)

    blobs = []
    for label_id in range(1, num_labels):
        blob_mask = labels == label_id
        area = int(blob_mask.sum())
        if area < min_area_px:
            continue

        coords = np.argwhere(blob_mask)
        cy, cx = coords.mean(axis=0)

        blob = ShadowBlob(
            mask=blob_mask,
            area_px=area,
            centroid=(float(cx), float(cy)),
        )
        blobs.append(blob)

    logger.info("Shadow detection: %d blobs from %d candidates", len(blobs), num_labels - 1)
    return blobs


def match_shadows_to_buildings(
    shadow_blobs: list[ShadowBlob],
    building_masks: list[np.ndarray],
    max_distance_px: int = 50,
) -> list[ShadowBlob]:
    """
    Match each shadow blob to its casting building.

    Uses nearest-building heuristic: for each shadow blob, find the
    building whose dilated mask has maximum overlap with the shadow.
    """
    if not shadow_blobs or not building_masks:
        return shadow_blobs

    h, w = building_masks[0].shape
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_distance_px, max_distance_px))

    for blob in shadow_blobs:
        best_overlap = 0
        best_idx = None

        for i, bm in enumerate(building_masks):
            bm_dilated = cv2.dilate(bm.astype(np.uint8) * 255, dilate_kernel) > 0
            overlap = (blob.mask & bm_dilated).sum()
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        blob.building_idx = best_idx

    matched = sum(1 for b in shadow_blobs if b.building_idx is not None)
    logger.info("Shadow matching: %d/%d blobs matched to buildings", matched, len(shadow_blobs))
    return shadow_blobs


def measure_shadow_length(
    shadow_blob: ShadowBlob,
    building_mask: np.ndarray,
    sun_azimuth_deg: float,
) -> float:
    """
    Measure shadow length along the solar azimuth direction.

    For each row of the shadow (perpendicular to azimuth), measures
    the distance from the building edge to the shadow tip. Returns
    the median of these measurements (robust to partial occlusion).

    Args:
        shadow_blob: The shadow to measure
        building_mask: The casting building's mask
        sun_azimuth_deg: Sun azimuth in degrees (0=N, 90=E, 180=S, 270=W)

    Returns:
        Shadow length in pixels (median of per-scanline measurements)
    """
    if shadow_blob.mask is None or building_mask is None:
        return 0.0

    # Shadow direction: shadows fall opposite to sun azimuth
    shadow_dir_rad = np.radians((sun_azimuth_deg + 180) % 360)
    dx = np.sin(shadow_dir_rad)
    dy = -np.cos(shadow_dir_rad)  # image y is inverted

    # Find building edge closest to shadow
    bldg_coords = np.argwhere(building_mask)  # (row, col)
    shadow_coords = np.argwhere(shadow_blob.mask)

    if len(bldg_coords) == 0 or len(shadow_coords) == 0:
        return 0.0

    # Project all points onto the shadow direction
    # shadow_dir vector in (row, col) space
    dir_vec = np.array([dy, dx])  # (row_component, col_component)

    bldg_proj = bldg_coords @ dir_vec
    shadow_proj = shadow_coords @ dir_vec

    # Shadow extends further along the projection than building
    bldg_edge = np.max(bldg_proj)
    shadow_tip = np.max(shadow_proj)
    shadow_base = np.min(shadow_proj)

    # The shadow length is the extent beyond the building edge
    length = shadow_tip - bldg_edge

    if length <= 0:
        # Try from the other side (depends on sun position)
        bldg_edge_min = np.min(bldg_proj)
        length = bldg_edge_min - shadow_base
        length = abs(length)

    # Also compute per-scanline lengths for robustness
    # Rotate shadow coordinates to align shadow direction with x-axis
    cos_a = np.cos(-shadow_dir_rad)
    sin_a = np.sin(-shadow_dir_rad)
    rot_shadow = shadow_coords.copy().astype(np.float64)
    rot_shadow[:, 0] = shadow_coords[:, 0] * cos_a - shadow_coords[:, 1] * sin_a
    rot_shadow[:, 1] = shadow_coords[:, 0] * sin_a + shadow_coords[:, 1] * cos_a

    rot_bldg = bldg_coords.copy().astype(np.float64)
    rot_bldg[:, 0] = bldg_coords[:, 0] * cos_a - bldg_coords[:, 1] * sin_a
    rot_bldg[:, 1] = bldg_coords[:, 0] * sin_a + bldg_coords[:, 1] * cos_a

    # Bin by perpendicular coordinate (row in rotated space)
    perp_min = min(rot_shadow[:, 0].min(), rot_bldg[:, 0].min())
    perp_max = max(rot_shadow[:, 0].max(), rot_bldg[:, 0].max())
    n_bins = max(1, int((perp_max - perp_min) / 3))

    scanline_lengths = []
    for bin_i in range(n_bins):
        lo = perp_min + bin_i * (perp_max - perp_min) / n_bins
        hi = lo + (perp_max - perp_min) / n_bins

        s_in_bin = rot_shadow[(rot_shadow[:, 0] >= lo) & (rot_shadow[:, 0] < hi)]
        b_in_bin = rot_bldg[(rot_bldg[:, 0] >= lo) & (rot_bldg[:, 0] < hi)]

        if len(s_in_bin) == 0 or len(b_in_bin) == 0:
            continue

        s_extent = s_in_bin[:, 1].max()
        b_extent = b_in_bin[:, 1].max()
        sl = abs(s_extent - b_extent)
        if sl > 1:
            scanline_lengths.append(sl)

    if scanline_lengths:
        return float(np.median(scanline_lengths))

    return max(0.0, float(length))


def render_shadow_overlay(
    image_shape: tuple,
    shadow_blobs: list[ShadowBlob],
) -> np.ndarray:
    """Render shadow detection as RGBA overlay."""
    h, w = image_shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for blob in shadow_blobs:
        if blob.building_idx is not None:
            overlay[blob.mask] = (40, 40, 180, 150)  # blue for matched shadows
        else:
            overlay[blob.mask] = (80, 80, 80, 100)   # gray for unmatched
    return overlay
