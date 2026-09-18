"""Structure-aware DSM: impose flat-roof / flat-ground priors via superpixel segmentation.

Pipeline:
1. Adaptive SLIC superpixels on the RGB image
2. Classify each region using SHAPE AND CONTEXT ONLY:
   BUILDING, ROAD, VEGETATION, GROUND
   (brightness and depth are NOT used for classification)
3. Merge adjacent building superpixels into unified rooftops
4. Assign heights with outlier clamping:
   - BUILDING: constant height = clipped median depth (flat roofs, sharp walls)
   - ROAD + GROUND: common base elevation (flat terrain)
   - VEGETATION: median height, mild roughness retained
5. Height sanity clamping: buildings outside [base+2, base+100] demoted to GROUND
6. Compose final heightmap + assert flat-roof invariant
"""
import logging
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.segmentation import slic

logger = logging.getLogger(__name__)

BUILDING = 0
ROAD = 1
VEGETATION = 2
GROUND = 3
CLASS_NAMES = {BUILDING: "BUILDING", ROAD: "ROAD", VEGETATION: "VEGETATION", GROUND: "GROUND"}


def _compute_shape_features(
    segments: np.ndarray, n_segs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute shape features per superpixel using vectorized bounding boxes.

    Returns (solidity, elongation, border_count, span_fraction).
    """
    h, w = segments.shape
    seg_flat = segments.ravel()
    areas = np.bincount(seg_flat, minlength=n_segs)

    ys, xs = np.mgrid[:h, :w]
    ys_flat = ys.ravel().astype(np.float64)
    xs_flat = xs.ravel().astype(np.float64)

    min_y = np.full(n_segs, h, dtype=np.float64)
    max_y = np.full(n_segs, 0, dtype=np.float64)
    min_x = np.full(n_segs, w, dtype=np.float64)
    max_x = np.full(n_segs, 0, dtype=np.float64)

    np.minimum.at(min_y, seg_flat, ys_flat)
    np.maximum.at(max_y, seg_flat, ys_flat)
    np.minimum.at(min_x, seg_flat, xs_flat)
    np.maximum.at(max_x, seg_flat, xs_flat)

    extent_y = max_y - min_y + 1
    extent_x = max_x - min_x + 1
    bbox_area = np.maximum(extent_y * extent_x, 1)
    solidity = (areas / bbox_area).astype(np.float32)

    short = np.minimum(extent_y, extent_x)
    long = np.maximum(extent_y, extent_x)
    elongation = (long / np.maximum(short, 1)).astype(np.float32)

    border_count = (
        (min_y == 0).astype(np.int32)
        + (max_y == h - 1).astype(np.int32)
        + (min_x == 0).astype(np.int32)
        + (max_x == w - 1).astype(np.int32)
    )

    span_fraction = np.maximum(extent_x / w, extent_y / h).astype(np.float32)

    return solidity, elongation, border_count, span_fraction


def _classify_vectorized(
    image_rgb: np.ndarray,
    segments: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """Classify superpixels using SHAPE AND CONTEXT ONLY.

    Brightness and depth are NOT used for class decisions.
    Depth is only used later for height assignment.
    """
    h, w = segments.shape
    n_segs = segments.max() + 1
    seg_ids = np.arange(n_segs)
    classes = np.full(n_segs, GROUND, dtype=np.int32)

    r = image_rgb[:, :, 0]
    g = image_rgb[:, :, 1]
    b = image_rgb[:, :, 2]
    gray = (r + g + b) / 3.0

    mean_r = np.array(ndimage.mean(r, segments, seg_ids))
    mean_g = np.array(ndimage.mean(g, segments, seg_ids))
    mean_b = np.array(ndimage.mean(b, segments, seg_ids))
    texture_std = np.array(ndimage.standard_deviation(gray, segments, seg_ids))

    rgb_sum = mean_r + mean_g + mean_b
    rgb_sum[rgb_sum == 0] = 1.0
    green_excess = (2 * mean_g - mean_r - mean_b) / rgb_sum

    seg_areas = np.bincount(segments.ravel(), minlength=n_segs).astype(np.float32)
    min_bldg_area = 200.0 * (w * h) / (1024.0 * 1024.0)

    solidity, elongation, border_count, span_fraction = _compute_shape_features(segments, n_segs)

    # --- Vegetation: green excess + texture (color IS valid for vegetation) ---
    is_veg = (
        (green_excess > 0.05) & (texture_std > 10)
    ) | (
        (green_excess > 0.12)
    )

    # --- Road rejection: SHAPE AND CONTEXT ONLY (applied BEFORE building check) ---
    is_road = (
        (~is_veg)
        & (
            (elongation > 4.0)
            | ((border_count >= 2) & (elongation > 2.0))
            | (span_fraction > 0.6)
            | ((elongation > 3.0) & (solidity < 0.4))
        )
    )

    # --- Building confirmation: ALL required, NO brightness, NO depth ---
    is_bldg = (
        (~is_veg)
        & (~is_road)
        & (solidity > 0.65)
        & (elongation < 4.0)
        & (seg_areas >= min_bldg_area)
        & (border_count < 2)
        & (texture_std < 40)
    )

    classes[is_veg] = VEGETATION
    classes[is_bldg] = BUILDING
    classes[is_road] = ROAD

    return classes


def _merge_adjacent_buildings(
    segments: np.ndarray, classes: np.ndarray, depth: np.ndarray, n_segs: int
) -> np.ndarray:
    """Merge adjacent building superpixels that share similar depth into unified rooftops.

    Returns a merged_labels array: all building superpixels belonging to the same
    physical rooftop get the same label.
    """
    merged = np.arange(n_segs, dtype=np.int32)

    bldg_ids = set(np.where(classes == BUILDING)[0])
    if not bldg_ids:
        return merged

    bldg_depths = {}
    for sid in bldg_ids:
        mask = segments == sid
        vals = depth[mask]
        p10, p90 = np.percentile(vals, [10, 90])
        clipped = vals[(vals >= p10) & (vals <= p90)]
        bldg_depths[sid] = float(np.median(clipped)) if len(clipped) > 0 else float(np.median(vals))

    # Find adjacency by checking neighboring pixels
    padded = np.pad(segments, 1, mode='edge')
    shifts = [(0, 1), (1, 0)]
    adj = set()
    for dy, dx in shifts:
        shifted = padded[1+dy:padded.shape[0]-1+dy, 1+dx:padded.shape[1]-1+dx]
        diff_mask = segments != shifted
        pairs = set(zip(segments[diff_mask].ravel(), shifted[diff_mask].ravel()))
        for a, b in pairs:
            if a in bldg_ids and b in bldg_ids:
                adj.add((min(a, b), max(a, b)))

    # Union-find merge for adjacent buildings with similar depth
    def find(x):
        while merged[x] != x:
            merged[x] = merged[merged[x]]
            x = merged[x]
        return x

    depth_threshold = 8.0
    for a, b in adj:
        if abs(bldg_depths[a] - bldg_depths[b]) < depth_threshold:
            ra, rb = find(a), find(b)
            if ra != rb:
                merged[rb] = ra

    for i in range(n_segs):
        merged[i] = find(i)

    return merged


def extract_building_footprints(
    image_pil: Image.Image,
    depth: np.ndarray,
    min_area: int = 500,
    max_aspect: float = 6.0,
    min_solidity: float = 0.4,
) -> tuple[list[dict], np.ndarray]:
    """Extract building footprints using Canny edges + morphological closing.

    Returns (footprints_list, outline_mask).
    """
    h, w = depth.shape
    gray = np.array(
        image_pil.convert("L").resize((w, h), Image.BILINEAR),
        dtype=np.uint8,
    )

    median_val = int(np.median(gray))
    low_thresh = max(0, int(0.5 * median_val))
    high_thresh = min(255, int(1.5 * median_val))
    edges = cv2.Canny(gray, low_thresh, high_thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Flood fill from corners to find enclosed regions
    filled = closed.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if filled[seed[1], seed[0]] == 0:
            cv2.floodFill(filled, flood_mask, seed, 255)
    filled_inv = cv2.bitwise_not(filled)
    regions = closed | filled_inv

    n_labels, labels = cv2.connectedComponents(regions)
    outline_mask = np.zeros((h, w), dtype=np.uint8)
    footprints = []

    for label_id in range(1, n_labels):
        component_mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, (rw, rh), _) = rect
        if min(rw, rh) < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect > max_aspect:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0

        mean_height = float(np.median(depth[component_mask > 0]))

        cv2.drawContours(outline_mask, [approx], -1, 255, 2)

        footprints.append({
            "polygon": approx.reshape(-1, 2).tolist(),
            "area": float(area),
            "centroid": [cx, cy],
            "mean_height": round(mean_height, 2),
            "solidity": round(solidity, 3),
            "aspect_ratio": round(aspect, 2),
        })

    logger.info("Footprint extraction: %d buildings from %d components", len(footprints), n_labels - 1)
    return footprints, outline_mask


def structure_aware_dsm(
    depth: np.ndarray,
    image_pil: Image.Image,
    n_segments: int | None = None,
    footprint_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply structural priors to a raw depth map.

    Returns:
        structured_depth: float32 array with flat roofs, flat terrain
        class_map: int32 array with per-pixel class labels
        stats: dict with per-class region counts and mean heights
    """
    t0 = time.perf_counter()

    h, w = depth.shape

    if n_segments is None:
        n_segments = max(200, min(1200, (w * h) // 4000))
    logger.info("Adaptive superpixel count: %d (for %dx%d)", n_segments, w, h)

    image_rgb = np.array(
        image_pil.convert("RGB").resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    )

    t_slic = time.perf_counter()
    segments = slic(
        image_rgb / 255.0,
        n_segments=n_segments,
        compactness=20,
        max_num_iter=5,
        start_label=0,
        channel_axis=2,
    )
    t_slic_done = time.perf_counter()
    actual_segs = segments.max() + 1
    logger.info("SLIC: %d superpixels in %.2f s", actual_segs, t_slic_done - t_slic)

    t_cls = time.perf_counter()
    classes = _classify_vectorized(image_rgb, segments, depth)

    old_bldg_count = int((classes == BUILDING).sum())

    merged_labels = _merge_adjacent_buildings(segments, classes, depth, actual_segs)
    t_cls_done = time.perf_counter()

    logger.info(
        "Classification: %.2f s — %d building, %d road, %d veg, %d ground",
        t_cls_done - t_cls,
        int((classes == BUILDING).sum()),
        int((classes == ROAD).sum()),
        int((classes == VEGETATION).sum()),
        int((classes == GROUND).sum()),
    )

    class_map = classes[segments]

    ground_mask = (class_map == ROAD) | (class_map == GROUND)
    base_height = float(np.median(depth[ground_mask])) if ground_mask.any() else float(np.percentile(depth, 20))

    t_compose = time.perf_counter()
    structured = np.copy(depth)
    n_segs = actual_segs
    counts = {BUILDING: 0, ROAD: 0, VEGETATION: 0, GROUND: 0}
    height_sums = {BUILDING: [], ROAD: [], VEGETATION: [], GROUND: []}

    # For merged buildings: compute ONE height per merged group
    merged_bldg_heights = {}
    for seg_id in range(n_segs):
        if classes[seg_id] != BUILDING:
            continue
        root = merged_labels[seg_id]
        if root not in merged_bldg_heights:
            merged_bldg_heights[root] = []

    for seg_id in range(n_segs):
        if classes[seg_id] != BUILDING:
            continue
        root = merged_labels[seg_id]
        mask = segments == seg_id
        vals = depth[mask]
        merged_bldg_heights[root].append(vals)

    # Compute clipped median per merged rooftop
    rooftop_heights = {}
    for root, val_list in merged_bldg_heights.items():
        all_vals = np.concatenate(val_list)
        p10, p90 = np.percentile(all_vals, [10, 90])
        clipped = all_vals[(all_vals >= p10) & (all_vals <= p90)]
        rooftop_heights[root] = float(np.median(clipped)) if len(clipped) > 0 else float(np.median(all_vals))

    # Height sanity clamping: building must be in [base+2, base+100]
    demoted_roots = set()
    for root, roof_h in list(rooftop_heights.items()):
        if roof_h < base_height + 2.0 or roof_h > base_height + 100.0:
            demoted_roots.add(root)
            del rooftop_heights[root]
    if demoted_roots:
        for seg_id in range(actual_segs):
            if classes[seg_id] == BUILDING and merged_labels[seg_id] in demoted_roots:
                classes[seg_id] = GROUND
        class_map = classes[segments]
        logger.info("Height clamping: demoted %d rooftop groups to GROUND", len(demoted_roots))

    unique_buildings = len(rooftop_heights)
    logger.info("Merged %d building superpixels into %d rooftops", old_bldg_count, unique_buildings)

    for seg_id in range(n_segs):
        cls = classes[seg_id]
        mask = segments == seg_id
        counts[cls] += 1

        if cls == BUILDING:
            roof_h = rooftop_heights[merged_labels[seg_id]]
            structured[mask] = roof_h
            height_sums[cls].append(roof_h)
        elif cls in (ROAD, GROUND):
            structured[mask] = base_height
            height_sums[cls].append(base_height)
        else:
            vals = depth[mask]
            p10, p90 = np.percentile(vals, [10, 90])
            clipped = vals[(vals >= p10) & (vals <= p90)]
            med = float(np.median(clipped)) if len(clipped) > 0 else float(np.median(vals))
            roughness = depth[mask] - med
            structured[mask] = med + roughness * 0.3
            height_sums[cls].append(med)

    t_compose_done = time.perf_counter()

    # P3 SANITY CHECK: every building region must have within-region std < 0.01
    bad_regions = 0
    for seg_id in range(n_segs):
        if classes[seg_id] != BUILDING:
            continue
        mask = segments == seg_id
        region_std = float(np.std(structured[mask]))
        if region_std >= 0.01:
            bad_regions += 1
            logger.warning(
                "FLAT-ROOF VIOLATION: seg=%d std=%.6f (expected <0.01)",
                seg_id, region_std,
            )
    if bad_regions == 0:
        logger.info("Flat-roof check: all %d building regions have std < 0.01", int((classes == BUILDING).sum()))

    stats = {}
    for cls in (BUILDING, ROAD, VEGETATION, GROUND):
        name = CLASS_NAMES[cls]
        cnt = counts[cls]
        mean_h = float(np.mean(height_sums[cls])) if height_sums[cls] else 0.0
        stats[name] = {"count": cnt, "mean_height": round(mean_h, 2)}
        logger.info("  %s: %d regions, mean height=%.2f", name, cnt, mean_h)

    stats["merged_rooftops"] = unique_buildings

    # Separation and correlation
    bldg_mean = float(stats["BUILDING"]["mean_height"])
    ground_mean = float(base_height)
    separation = bldg_mean - ground_mean
    gray_flat = ((image_rgb[:, :, 0] + image_rgb[:, :, 1] + image_rgb[:, :, 2]) / 3.0).ravel()
    from scipy.stats import pearsonr
    corr, _ = pearsonr(gray_flat, structured.ravel())
    logger.info(
        "Building mean=%.1f, ground=%.1f, separation=%.1f, brightness-height r=%.4f",
        bldg_mean, ground_mean, separation, corr,
    )
    stats["separation"] = round(float(separation), 2)
    stats["pearson_r"] = round(float(corr), 4)

    if footprint_mode:
        t_fp = time.perf_counter()
        footprints, outline_mask = extract_building_footprints(image_pil, depth)
        t_fp_done = time.perf_counter()
        stats["footprints"] = footprints
        stats["footprint_count"] = len(footprints)
        stats["_outline_mask"] = outline_mask
        logger.info("Footprint extraction: %d buildings in %.2f s", len(footprints), t_fp_done - t_fp)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Structure DSM: %.2f s total (SLIC %.2f + classify %.2f + compose %.2f)",
        elapsed, t_slic_done - t_slic, t_cls_done - t_cls, t_compose_done - t_compose,
    )

    return structured.astype(np.float32), class_map.astype(np.int32), stats
