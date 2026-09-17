"""Structure-aware DSM: impose flat-roof / flat-ground priors via superpixel segmentation.

Pipeline:
1. SLIC superpixels on the RGB image
2. Classify each region: BUILDING, ROAD, VEGETATION, GROUND
3. Assign heights:
   - BUILDING → constant median depth (flat roofs, sharp walls)
   - ROAD + GROUND → common base elevation (flat terrain)
   - VEGETATION → median height, mild roughness retained
4. Compose final heightmap
"""
import logging
import time

import numpy as np
from PIL import Image
from skimage.segmentation import slic
from skimage.measure import regionprops, label as sk_label

logger = logging.getLogger(__name__)

# Class labels
BUILDING = 0
ROAD = 1
VEGETATION = 2
GROUND = 3
CLASS_NAMES = {BUILDING: "BUILDING", ROAD: "ROAD", VEGETATION: "VEGETATION", GROUND: "GROUND"}


def classify_regions(
    image_rgb: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    """Classify each superpixel region from simple features.

    Features per region:
    - mean RGB
    - interior texture std (grayscale std within region)
    - compactness (area / perimeter^2, higher = more compact/square)
    - green excess index (2*G - R - B) / (R + G + B)
    """
    gray = np.mean(image_rgb, axis=2)
    n_segments = segments.max() + 1
    classes = np.full(n_segments, GROUND, dtype=np.int32)

    label_img = sk_label(segments + 1)
    props = regionprops(label_img)

    for region in props:
        seg_id = region.label - 1
        if seg_id < 0 or seg_id >= n_segments:
            continue

        mask = segments == seg_id
        pixels = image_rgb[mask]
        gray_pixels = gray[mask]

        if len(pixels) < 4:
            continue

        mean_r = float(pixels[:, 0].mean())
        mean_g = float(pixels[:, 1].mean())
        mean_b = float(pixels[:, 2].mean())
        mean_brightness = (mean_r + mean_g + mean_b) / 3.0
        texture_std = float(gray_pixels.std())

        rgb_sum = mean_r + mean_g + mean_b
        green_excess = (2 * mean_g - mean_r - mean_b) / rgb_sum if rgb_sum > 0 else 0.0

        area = float(region.area)
        perimeter = float(region.perimeter) if region.perimeter > 0 else 1.0
        compactness = area / (perimeter * perimeter)

        bbox = region.bbox
        bbox_h = bbox[2] - bbox[0]
        bbox_w = bbox[3] - bbox[1]
        aspect = max(bbox_h, bbox_w) / (min(bbox_h, bbox_w) + 1e-6)

        if green_excess > 0.08 and texture_std > 15:
            classes[seg_id] = VEGETATION
        elif mean_brightness > 140 and compactness > 0.02 and texture_std < 35:
            classes[seg_id] = BUILDING
        elif mean_brightness < 100 and texture_std < 25:
            classes[seg_id] = ROAD
        else:
            classes[seg_id] = GROUND

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
    logger.info("SLIC: %d superpixels in %.2f s", segments.max() + 1, time.perf_counter() - t_slic)

    t_cls = time.perf_counter()
    classes = classify_regions(image_rgb, segments)
    logger.info("Classification done in %.2f s", time.perf_counter() - t_cls)

    class_map = classes[segments]

    ground_mask = (class_map == ROAD) | (class_map == GROUND)
    if ground_mask.any():
        base_height = float(np.median(depth[ground_mask]))
    else:
        base_height = float(np.percentile(depth, 20))

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

    stats = {}
    for cls in (BUILDING, ROAD, VEGETATION, GROUND):
        name = CLASS_NAMES[cls]
        cnt = counts[cls]
        mean_h = float(np.mean(height_sums[cls])) if height_sums[cls] else 0.0
        stats[name] = {"count": cnt, "mean_height": round(mean_h, 2)}
        logger.info("  %s: %d regions, mean height=%.2f", name, cnt, mean_h)

    elapsed = time.perf_counter() - t0
    logger.info("Structure-aware DSM: %.2f s total", elapsed)

    return structured.astype(np.float32), class_map.astype(np.int32), stats
