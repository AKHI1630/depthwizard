"""Structure-aware DSM: impose flat-roof / flat-ground priors via superpixel segmentation.

Pipeline:
1. SLIC superpixels on the RGB image
2. Classify each region: BUILDING, ROAD, VEGETATION, GROUND
3. Assign heights:
   - BUILDING: constant median depth (flat roofs, sharp walls)
   - ROAD + GROUND: common base elevation (flat terrain)
   - VEGETATION: median height, mild roughness retained
4. Compose final heightmap
"""
import logging
import time

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.segmentation import slic

logger = logging.getLogger(__name__)

BUILDING = 0
ROAD = 1
VEGETATION = 2
GROUND = 3
CLASS_NAMES = {BUILDING: "BUILDING", ROAD: "ROAD", VEGETATION: "VEGETATION", GROUND: "GROUND"}


def _classify_vectorized(image_rgb: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Classify superpixels using vectorized scipy.ndimage operations."""
    n_segs = segments.max() + 1
    seg_ids = np.arange(n_segs)
    classes = np.full(n_segs, GROUND, dtype=np.int32)

    r = image_rgb[:, :, 0]
    g = image_rgb[:, :, 1]
    b = image_rgb[:, :, 2]
    gray = (r + g + b) / 3.0

    mean_r = ndimage.mean(r, segments, seg_ids)
    mean_g = ndimage.mean(g, segments, seg_ids)
    mean_b = ndimage.mean(b, segments, seg_ids)
    mean_brightness = (np.array(mean_r) + np.array(mean_g) + np.array(mean_b)) / 3.0
    texture_std = np.array(ndimage.standard_deviation(gray, segments, seg_ids))

    rgb_sum = np.array(mean_r) + np.array(mean_g) + np.array(mean_b)
    rgb_sum[rgb_sum == 0] = 1.0
    green_excess = (2 * np.array(mean_g) - np.array(mean_r) - np.array(mean_b)) / rgb_sum

    is_veg = (green_excess > 0.08) & (texture_std > 15)
    is_bldg = (~is_veg) & (mean_brightness > 140) & (texture_std < 35)
    is_road = (~is_veg) & (~is_bldg) & (mean_brightness < 100) & (texture_std < 25)

    classes[is_veg] = VEGETATION
    classes[is_bldg] = BUILDING
    classes[is_road] = ROAD

    return classes


def structure_aware_dsm(
    depth: np.ndarray,
    image_pil: Image.Image,
    n_segments: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply structural priors to a raw depth map.

    Returns:
        structured_depth: float32 array with flat roofs, flat terrain
        class_map: int32 array with per-pixel class labels
        stats: dict with per-class region counts and mean heights
    """
    t0 = time.perf_counter()

    h, w = depth.shape
    image_rgb = np.array(
        image_pil.convert("RGB").resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    )

    t_slic = time.perf_counter()
    segments = slic(
        image_rgb / 255.0,
        n_segments=n_segments,
        compactness=20,
        start_label=0,
        channel_axis=2,
    )
    t_slic_done = time.perf_counter()
    logger.info("SLIC: %d superpixels in %.2f s", segments.max() + 1, t_slic_done - t_slic)

    t_cls = time.perf_counter()
    classes = _classify_vectorized(image_rgb, segments)
    t_cls_done = time.perf_counter()
    logger.info("Classification: %.2f s", t_cls_done - t_cls)

    class_map = classes[segments]

    ground_mask = (class_map == ROAD) | (class_map == GROUND)
    base_height = float(np.median(depth[ground_mask])) if ground_mask.any() else float(np.percentile(depth, 20))

    t_compose = time.perf_counter()
    structured = np.copy(depth)
    n_segs = segments.max() + 1
    counts = {BUILDING: 0, ROAD: 0, VEGETATION: 0, GROUND: 0}
    height_sums = {BUILDING: [], ROAD: [], VEGETATION: [], GROUND: []}

    for seg_id in range(n_segs):
        cls = classes[seg_id]
        mask = segments == seg_id
        counts[cls] += 1

        if cls == BUILDING:
            med = float(np.median(depth[mask]))
            structured[mask] = med
            height_sums[cls].append(med)
        elif cls in (ROAD, GROUND):
            structured[mask] = base_height
            height_sums[cls].append(base_height)
        else:
            med = float(np.median(depth[mask]))
            roughness = depth[mask] - med
            structured[mask] = med + roughness * 0.3
            height_sums[cls].append(med)

    t_compose_done = time.perf_counter()

    stats = {}
    for cls in (BUILDING, ROAD, VEGETATION, GROUND):
        name = CLASS_NAMES[cls]
        cnt = counts[cls]
        mean_h = float(np.mean(height_sums[cls])) if height_sums[cls] else 0.0
        stats[name] = {"count": cnt, "mean_height": round(mean_h, 2)}
        logger.info("  %s: %d regions, mean height=%.2f", name, cnt, mean_h)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Structure DSM: %.2f s total (SLIC %.2f + classify %.2f + compose %.2f)",
        elapsed, t_slic_done - t_slic, t_cls_done - t_cls, t_compose_done - t_compose,
    )

    return structured.astype(np.float32), class_map.astype(np.int32), stats
