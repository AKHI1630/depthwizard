"""
SAM-based instance segmentation for satellite imagery.

Produces per-instance masks from nadir satellite tiles using
Segment Anything (ViT-B) in automatic mask generation mode.
Each mask is classified as building/tree/road/ground based on
shape and colour features (no depth information used).
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
SAM_CHECKPOINT = MODELS_DIR / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

# Classification thresholds — single config dict per the brief
CLASS_CONFIG = {
    "building": {
        "min_rectangularity": 0.55,
        "max_rectangularity": 1.0,
        "min_solidity": 0.7,
        "max_exg": 0.05,
        "min_area_px": 200,
        "max_area_px": 200000,
        "max_aspect_ratio": 6.0,
        "max_texture_var": 800,
    },
    "tree": {
        "min_exg": 0.03,
        "min_texture_var": 100,
        "max_rectangularity": 0.65,
        "min_area_px": 50,
        "max_area_px": 50000,
    },
    "road": {
        "min_aspect_ratio": 4.0,
        "max_exg": 0.02,
        "max_texture_var": 300,
        "min_area_px": 500,
    },
}


@dataclass
class InstanceMask:
    mask: np.ndarray          # bool H×W
    area: int
    bbox: tuple               # (x, y, w, h)
    label: str                # building | tree | road | ground
    confidence: float         # 0-1
    rectangularity: float
    solidity: float
    aspect_ratio: float
    exg: float                # excess green index
    texture_var: float
    mean_rgb: tuple
    contour: Optional[np.ndarray] = None
    regularised_polygon: Optional[np.ndarray] = None
    height_m: Optional[float] = None
    height_confidence: Optional[str] = None
    height_uncertainty_m: Optional[float] = None
    height_methods: list[str] = field(default_factory=list)
    height_source: Optional[str] = None
    shadow_length_px: Optional[float] = None


def _load_sam_model():
    """Load SAM model weights (once). Returns the sam model."""
    import torch
    from segment_anything import sam_model_registry

    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM weights not found at {SAM_CHECKPOINT}. "
            "Download from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading SAM %s on %s from %s", SAM_MODEL_TYPE, device, SAM_CHECKPOINT)
    t0 = time.perf_counter()

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)

    logger.info("SAM loaded in %.1f s", time.perf_counter() - t0)
    return sam


_sam_model = None
_mask_generator = None
_generator_params: dict = {}


def get_mask_generator(points_per_side: int = 12, min_mask_region_area: int = 30):
    global _sam_model, _mask_generator, _generator_params
    from segment_anything import SamAutomaticMaskGenerator

    if _sam_model is None:
        _sam_model = _load_sam_model()

    requested = {"points_per_side": points_per_side, "min_mask_region_area": min_mask_region_area}
    if requested != _generator_params:
        _mask_generator = SamAutomaticMaskGenerator(
            model=_sam_model,
            points_per_side=points_per_side,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=0,
            min_mask_region_area=min_mask_region_area,
        )
        _generator_params = requested
        logger.info("SAM generator configured: points_per_side=%d, min_mask_region_area=%d",
                    points_per_side, min_mask_region_area)
    return _mask_generator


def split_large_masks(sam_masks: list, image_rgb: np.ndarray, area_factor: float = 3.0) -> list:
    """Split oversized masks using watershed with distance-transform markers."""
    if len(sam_masks) < 3:
        return sam_masks

    areas = [m["area"] for m in sam_masks]
    median_area = float(np.median(areas))
    if median_area < 50:
        return sam_masks

    result = []
    split_count = 0

    for m in sam_masks:
        if m["area"] <= area_factor * median_area:
            result.append(m)
            continue

        mask_u8 = m["segmentation"].astype(np.uint8) * 255
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        thresh = 0.5 * dist.max()
        _, peaks = cv2.threshold(dist, thresh, 255, cv2.THRESH_BINARY)
        peaks_u8 = peaks.astype(np.uint8)
        n_labels, markers = cv2.connectedComponents(peaks_u8)

        if n_labels <= 2:
            result.append(m)
            continue

        markers = markers + 1
        markers[mask_u8 == 0] = 0

        h, w = image_rgb.shape[:2]
        mh, mw = mask_u8.shape
        if mh == h and mw == w:
            ws_img = image_rgb.copy()
        else:
            ws_img = np.stack([mask_u8, mask_u8, mask_u8], axis=2)

        markers_ws = cv2.watershed(ws_img, markers.astype(np.int32))

        for lbl in range(2, n_labels + 1):
            sub_mask = (markers_ws == lbl)
            sub_area = int(sub_mask.sum())
            if sub_area < 100:
                continue
            ys, xs = np.where(sub_mask)
            bx, by = int(xs.min()), int(ys.min())
            bw, bh = int(xs.max()) - bx, int(ys.max()) - by
            result.append({
                "segmentation": sub_mask,
                "area": sub_area,
                "bbox": [bx, by, bw, bh],
                "predicted_iou": m.get("predicted_iou", 0.8),
                "stability_score": m.get("stability_score", 0.8),
            })
            split_count += 1

        if split_count == 0:
            result.append(m)

    if split_count > 0:
        logger.info("Watershed split: %d large masks → %d total sub-masks", split_count, len(result) - len(sam_masks) + split_count)

    return result


def compute_mask_features(mask_bool: np.ndarray, image_rgb: np.ndarray) -> dict:
    """Compute shape and colour features for a single binary mask."""
    h, w = mask_bool.shape
    coords = np.argwhere(mask_bool)
    if len(coords) < 10:
        return None

    area = int(mask_bool.sum())

    # Contour for shape analysis
    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)
    if contour_area < 10:
        return None

    # Bounding rect and min-area rect
    x, y, bw, bh = cv2.boundingRect(contour)
    min_rect = cv2.minAreaRect(contour)
    (_, (rect_w, rect_h), _) = min_rect
    if rect_w < 1 or rect_h < 1:
        return None

    rect_area = rect_w * rect_h
    rectangularity = contour_area / rect_area if rect_area > 0 else 0

    # Convex hull solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / hull_area if hull_area > 0 else 0

    # Aspect ratio (min-area rect)
    long_side = max(rect_w, rect_h)
    short_side = min(rect_w, rect_h)
    aspect_ratio = long_side / short_side if short_side > 0 else 999

    # Colour features — mean RGB within mask
    pixels = image_rgb[mask_bool]
    mean_rgb = tuple(pixels.mean(axis=0).astype(int).tolist())
    r, g, b = mean_rgb
    denom = r + g + b + 1e-6
    exg = (2 * g - r - b) / denom

    # Texture variance — grayscale std within mask
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    texture_var = float(gray[mask_bool].var())

    return {
        "area": area,
        "bbox": (x, y, bw, bh),
        "rectangularity": round(rectangularity, 3),
        "solidity": round(solidity, 3),
        "aspect_ratio": round(aspect_ratio, 2),
        "exg": round(exg, 4),
        "texture_var": round(texture_var, 1),
        "mean_rgb": mean_rgb,
        "contour": contour,
    }


def classify_mask(feats: dict) -> tuple[str, float]:
    """Classify a mask as building/tree/road/ground from shape+colour features."""
    cfg = CLASS_CONFIG
    bc = cfg["building"]
    tc = cfg["tree"]
    rc = cfg["road"]

    rect = feats["rectangularity"]
    sol = feats["solidity"]
    ar = feats["aspect_ratio"]
    exg = feats["exg"]
    tvar = feats["texture_var"]
    area = feats["area"]

    # Tree: green, textured, non-rectangular
    tree_score = 0.0
    if exg > tc["min_exg"]:
        tree_score += 0.4
    if tvar > tc["min_texture_var"]:
        tree_score += 0.3
    if rect < tc["max_rectangularity"]:
        tree_score += 0.2
    if tc["min_area_px"] <= area <= tc["max_area_px"]:
        tree_score += 0.1
    if tree_score >= 0.7 and exg > tc["min_exg"]:
        return "tree", min(tree_score, 1.0)

    # Road: elongated, not green, low texture
    road_score = 0.0
    if ar >= rc["min_aspect_ratio"]:
        road_score += 0.5
    if exg < rc["max_exg"]:
        road_score += 0.2
    if tvar < rc["max_texture_var"]:
        road_score += 0.2
    if area >= rc["min_area_px"]:
        road_score += 0.1
    if road_score >= 0.7 and ar >= rc["min_aspect_ratio"]:
        return "road", min(road_score, 1.0)

    # Building: rectangular, solid, not green, not too textured
    bldg_score = 0.0
    if rect >= bc["min_rectangularity"]:
        bldg_score += 0.3
    if sol >= bc["min_solidity"]:
        bldg_score += 0.25
    if exg < bc["max_exg"]:
        bldg_score += 0.15
    if tvar < bc["max_texture_var"]:
        bldg_score += 0.15
    if bc["min_area_px"] <= area <= bc["max_area_px"]:
        bldg_score += 0.1
    if ar <= bc["max_aspect_ratio"]:
        bldg_score += 0.05
    if bldg_score >= 0.65 and rect >= bc["min_rectangularity"] and sol >= bc["min_solidity"]:
        return "building", min(bldg_score, 1.0)

    return "ground", 0.5


def segment_and_classify(
    image: Image.Image,
    sam_max_dim: int = 512,
    points_per_side: int = 12,
    min_mask_region_area: int = 30,
    watershed_area_factor: float = 2.0,
) -> list[InstanceMask]:
    """Run SAM automatic mask generation + shape/colour classification."""
    img_array = np.array(image)
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    h, w = img_array.shape[:2]
    logger.info("SAM segmentation on %d×%d image (max_dim=%d, pts=%d)", w, h, sam_max_dim, points_per_side)
    t0 = time.perf_counter()

    scale_factor = 1.0
    sam_input = img_array
    if max(h, w) > sam_max_dim:
        scale_factor = sam_max_dim / max(h, w)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        sam_input = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info("Downscaled %d×%d → %d×%d (factor %.2f) for SAM", w, h, new_w, new_h, scale_factor)

    generator = get_mask_generator(points_per_side=points_per_side, min_mask_region_area=min_mask_region_area)
    sam_masks = generator.generate(sam_input)

    # Upscale masks back to original resolution if downscaled
    if scale_factor < 1.0:
        for m in sam_masks:
            small_mask = m["segmentation"]
            m["segmentation"] = cv2.resize(
                small_mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            m["area"] = int(m["segmentation"].sum())
            if "bbox" in m:
                bx, by, bw, bh = m["bbox"]
                inv = 1.0 / scale_factor
                m["bbox"] = [int(bx * inv), int(by * inv), int(bw * inv), int(bh * inv)]

    t_sam = time.perf_counter() - t0
    logger.info("SAM generated %d masks in %.1f s", len(sam_masks), t_sam)

    # Sort by area descending
    sam_masks.sort(key=lambda m: m["area"], reverse=True)

    # Split oversized masks (touching buildings)
    sam_masks = split_large_masks(sam_masks, img_array, area_factor=watershed_area_factor)

    instances = []
    t_classify = time.perf_counter()

    for i, sam_mask in enumerate(sam_masks):
        mask_bool = sam_mask["segmentation"]
        feats = compute_mask_features(mask_bool, img_array)
        if feats is None:
            continue

        label, confidence = classify_mask(feats)

        reg_poly = None
        if label == "building" and feats["contour"] is not None:
            reg_poly = regularise_polygon(feats["contour"], image_shape=img_array.shape[:2])

        instance = InstanceMask(
            mask=mask_bool,
            area=feats["area"],
            bbox=feats["bbox"],
            label=label,
            confidence=confidence,
            rectangularity=feats["rectangularity"],
            solidity=feats["solidity"],
            aspect_ratio=feats["aspect_ratio"],
            exg=feats["exg"],
            texture_var=feats["texture_var"],
            mean_rgb=feats["mean_rgb"],
            contour=feats["contour"],
            regularised_polygon=reg_poly,
        )
        instances.append(instance)

    t_cls = time.perf_counter() - t_classify
    counts = {}
    for inst in instances:
        counts[inst.label] = counts.get(inst.label, 0) + 1
    logger.info(
        "Classification: %.2f s — %s",
        t_cls,
        ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
    )

    building_areas = sorted(
        [inst.area for inst in instances if inst.label == "building"]
    )
    if building_areas:
        pcts = np.percentile(building_areas, [10, 25, 50, 75, 90])
        logger.info(
            "Building area histogram (px): p10=%d p25=%d p50=%d p75=%d p90=%d  n=%d  range=[%d, %d]",
            int(pcts[0]), int(pcts[1]), int(pcts[2]), int(pcts[3]), int(pcts[4]),
            len(building_areas), building_areas[0], building_areas[-1],
        )

    return instances


def render_segmentation_overlay(
    image: Image.Image, instances: list[InstanceMask]
) -> np.ndarray:
    """Render classification overlay as RGBA image."""
    w, h = image.size
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    colors = {
        "building": (220, 50, 50, 160),
        "tree": (50, 180, 50, 160),
        "road": (140, 140, 140, 160),
        "ground": (160, 120, 80, 80),
    }

    for inst in instances:
        color = colors.get(inst.label, (100, 100, 100, 80))
        overlay[inst.mask] = color

    return overlay


def regularise_polygon(contour: np.ndarray, epsilon_frac: float = 0.02,
                       image_shape: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Douglas-Peucker simplification + edge snapping to dominant orientations."""
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_frac * perimeter
    simplified = cv2.approxPolyDP(contour, epsilon, True)

    if len(simplified) < 4:
        return simplified

    pts = simplified.reshape(-1, 2).astype(np.float64)
    n = len(pts)

    # Compute edge angles (mod 180° since direction doesn't matter)
    edges = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    angles = np.arctan2(edges[:, 1], edges[:, 0]) % np.pi
    lengths = np.linalg.norm(edges, axis=1)

    # Find dominant orientation via length-weighted histogram
    n_bins = 36
    hist = np.zeros(n_bins)
    for a, l in zip(angles, lengths):
        bin_idx = int(a / np.pi * n_bins) % n_bins
        hist[bin_idx] += l

    # Two dominant angles (typically 90° apart for rectilinear buildings)
    peak1 = np.argmax(hist)
    # Suppress neighbourhood of first peak
    suppressed = hist.copy()
    for offset in range(-3, 4):
        suppressed[(peak1 + offset) % n_bins] = 0
    peak2 = np.argmax(suppressed)

    dom_angles = [peak1 * np.pi / n_bins, peak2 * np.pi / n_bins]

    # Snap each edge to the nearest dominant angle
    snapped = pts.copy()
    for i in range(n):
        j = (i + 1) % n
        edge = pts[j] - pts[i]
        edge_angle = np.arctan2(edge[1], edge[0]) % np.pi
        edge_len = np.linalg.norm(edge)
        if edge_len < 1:
            continue

        # Find nearest dominant angle
        diffs = [min(abs(edge_angle - da), np.pi - abs(edge_angle - da)) for da in dom_angles]
        nearest = dom_angles[np.argmin(diffs)]

        # Only snap if within 15° of a dominant angle
        if min(diffs) < np.radians(15):
            direction = np.array([np.cos(nearest), np.sin(nearest)])
            projected = np.dot(edge, direction)
            snapped[j] = snapped[i] + direction * projected

    if image_shape is not None:
        h_img, w_img = image_shape
        before = snapped.copy()
        snapped[:, 0] = np.clip(snapped[:, 0], 0, w_img - 1)
        snapped[:, 1] = np.clip(snapped[:, 1], 0, h_img - 1)
        oob = np.any(before != snapped)
        if oob:
            logger.warning("Polygon vertex clamped to image bounds (%dx%d)", w_img, h_img)

    return snapped.reshape(-1, 1, 2).astype(np.int32)


def instances_to_json(instances: list[InstanceMask], image_shape: Optional[tuple[int, int]] = None) -> list[dict]:
    """Convert instance list to JSON-serialisable format."""
    result = []
    for inst in instances:
        bx, by, bw, bh = inst.bbox
        if image_shape is not None:
            ih, iw = image_shape
            bx = max(0, bx)
            by = max(0, by)
            bw = min(bw, iw - bx)
            bh = min(bh, ih - by)

        d = {
            "label": inst.label,
            "confidence": round(inst.confidence, 3),
            "area_px": inst.area,
            "bbox": [bx, by, bw, bh],
            "rectangularity": inst.rectangularity,
            "solidity": inst.solidity,
            "aspect_ratio": inst.aspect_ratio,
            "exg": round(inst.exg, 4),
            "texture_var": inst.texture_var,
            "mean_rgb": list(inst.mean_rgb),
        }
        if inst.height_m is not None:
            d["height_m"] = round(inst.height_m, 2)
            d["height_confidence"] = inst.height_confidence
            d["height_uncertainty_m"] = round(inst.height_uncertainty_m, 2) if inst.height_uncertainty_m else None
        elif inst.height_confidence:
            d["height_m"] = None
            d["height_confidence"] = inst.height_confidence
        if inst.shadow_length_px is not None:
            d["shadow_length_px"] = round(inst.shadow_length_px, 1)
        if inst.height_methods:
            d["height_methods"] = inst.height_methods
        if inst.height_source:
            d["height_source"] = inst.height_source
        if inst.regularised_polygon is not None:
            poly = inst.regularised_polygon.reshape(-1, 2).tolist()
            if image_shape is not None:
                ih, iw = image_shape
                poly = [[max(0, min(x, iw - 1)), max(0, min(y, ih - 1))] for x, y in poly]
            d["polygon"] = poly
        result.append(d)
    return result
