"""Structure-aware DSM: impose flat-roof / flat-ground priors via superpixel segmentation.

Pipeline:
1. Adaptive SLIC superpixels on the RGB image
2. Classify each region using shape + color + texture features:
   BUILDING, ROAD, VEGETATION, GROUND
3. Merge adjacent building superpixels into unified rooftops
4. Assign heights with outlier clamping:
   - BUILDING: constant height = clipped median depth (flat roofs, sharp walls)
   - ROAD + GROUND: common base elevation (flat terrain)
   - VEGETATION: median height, mild roughness retained
5. Compose final heightmap + assert flat-roof invariant
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


def _compute_shape_features(segments: np.ndarray, n_segs: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute solidity and elongation per superpixel using vectorized ops."""
    solidity = np.ones(n_segs, dtype=np.float32)
    elongation = np.zeros(n_segs, dtype=np.float32)

    for seg_id in range(n_segs):
        ys, xs = np.where(segments == seg_id)
        if len(ys) < 4:
            continue
        area = len(ys)
        min_r, max_r = ys.min(), ys.max()
        min_c, max_c = xs.min(), xs.max()
        bbox_area = max((max_r - min_r + 1) * (max_c - min_c + 1), 1)
        solidity[seg_id] = area / bbox_area

        extent_r = max_r - min_r + 1
        extent_c = max_c - min_c + 1
        short = min(extent_r, extent_c)
        long = max(extent_r, extent_c)
        elongation[seg_id] = long / max(short, 1)

    return solidity, elongation


def _classify_vectorized(
    image_rgb: np.ndarray,
    segments: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Classify superpixels using color, texture, shape, and depth features.

    Returns class array and per-region diagnostic dicts.
    """
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
    mean_brightness = (mean_r + mean_g + mean_b) / 3.0
    texture_std = np.array(ndimage.standard_deviation(gray, segments, seg_ids))

    rgb_sum = mean_r + mean_g + mean_b
    rgb_sum[rgb_sum == 0] = 1.0
    green_excess = (2 * mean_g - mean_r - mean_b) / rgb_sum

    mean_depth = np.array(ndimage.mean(depth, segments, seg_ids))
    depth_std = np.array(ndimage.standard_deviation(depth, segments, seg_ids))

    seg_areas = np.bincount(segments.ravel(), minlength=n_segs).astype(np.float32)

    solidity, elongation = _compute_shape_features(segments, n_segs)

    # --- Vegetation: green excess + high texture (catches dark tree crowns) ---
    is_veg = (
        (green_excess > 0.05) & (texture_std > 10)
    ) | (
        (green_excess > 0.12)
    )

    # --- Building: compact shape + moderate-high brightness + low internal texture ---
    is_bldg = (
        (~is_veg)
        & (solidity > 0.55)
        & (elongation < 4.0)
        & (texture_std < 40)
        & (depth_std < mean_depth * 0.3 + 1.0)
        & (mean_brightness > 100)
    )

    # --- Road: elongated or low-texture dark regions ---
    is_road = (
        (~is_veg) & (~is_bldg)
        & (
            (elongation > 3.0)
            | ((mean_brightness < 110) & (texture_std < 25))
            | ((solidity < 0.45) & (texture_std < 20))
        )
    )

    classes[is_veg] = VEGETATION
    classes[is_bldg] = BUILDING
    classes[is_road] = ROAD

    diags = []
    for i in range(n_segs):
        diags.append({
            "seg": i,
            "class": CLASS_NAMES[classes[i]],
            "area": int(seg_areas[i]),
            "solidity": round(float(solidity[i]), 3),
            "elongation": round(float(elongation[i]), 2),
            "brightness": round(float(mean_brightness[i]), 1),
            "texture_std": round(float(texture_std[i]), 1),
            "green_excess": round(float(green_excess[i]), 3),
            "depth_mean": round(float(mean_depth[i]), 2),
            "depth_std": round(float(depth_std[i]), 2),
            "height": 0.0,
        })

    return classes, diags


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


def structure_aware_dsm(
    depth: np.ndarray,
    image_pil: Image.Image,
    n_segments: int | None = None,
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
        compactness=15,
        start_label=0,
        channel_axis=2,
    )
    t_slic_done = time.perf_counter()
    actual_segs = segments.max() + 1
    logger.info("SLIC: %d superpixels in %.2f s", actual_segs, t_slic_done - t_slic)

    t_cls = time.perf_counter()
    classes, diags = _classify_vectorized(image_rgb, segments, depth)

    old_bldg_count = int((classes == BUILDING).sum())

    merged_labels = _merge_adjacent_buildings(segments, classes, depth, actual_segs)
    t_cls_done = time.perf_counter()

    veg_from_bldg = int((classes == VEGETATION).sum()) - 0
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

    unique_buildings = len(rooftop_heights)
    logger.info("Merged %d building superpixels into %d rooftops", old_bldg_count, unique_buildings)

    bldg_stds = []
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

    # Log per-region diagnostics (first 15)
    for d in diags[:15]:
        seg_id = d["seg"]
        if classes[seg_id] == BUILDING:
            d["height"] = rooftop_heights.get(merged_labels[seg_id], 0.0)
        elif classes[seg_id] in (ROAD, GROUND):
            d["height"] = base_height
        logger.info(
            "  seg=%d class=%-10s area=%5d sol=%.2f elong=%.1f bright=%.0f tex=%.1f "
            "green=%.3f depth_m=%.1f depth_std=%.1f → h=%.1f",
            d["seg"], d["class"], d["area"], d["solidity"], d["elongation"],
            d["brightness"], d["texture_std"], d["green_excess"],
            d["depth_mean"], d["depth_std"], d["height"],
        )

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

    elapsed = time.perf_counter() - t0
    logger.info(
        "Structure DSM: %.2f s total (SLIC %.2f + classify %.2f + compose %.2f)",
        elapsed, t_slic_done - t_slic, t_cls_done - t_cls, t_compose_done - t_compose,
    )

    return structured.astype(np.float32), class_map.astype(np.int32), stats
