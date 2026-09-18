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


def _load_sam():
    """Load SAM model. Returns (sam, SamAutomaticMaskGenerator)."""
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

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

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=100,
    )

    logger.info("SAM loaded in %.1f s", time.perf_counter() - t0)
    return sam, mask_generator


_sam_model = None
_mask_generator = None


def get_mask_generator():
    global _sam_model, _mask_generator
    if _mask_generator is None:
        _sam_model, _mask_generator = _load_sam()
    return _mask_generator


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


def segment_and_classify(image: Image.Image) -> list[InstanceMask]:
    """
    Run SAM automatic mask generation + shape/colour classification.

    Args:
        image: PIL RGB image of the satellite tile

    Returns:
        List of InstanceMask objects, one per detected instance
    """
    img_array = np.array(image)
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    h, w = img_array.shape[:2]
    logger.info("SAM segmentation on %d×%d image", w, h)
    t0 = time.perf_counter()

    generator = get_mask_generator()
    sam_masks = generator.generate(img_array)

    t_sam = time.perf_counter() - t0
    logger.info("SAM generated %d masks in %.1f s", len(sam_masks), t_sam)

    # Sort by area descending
    sam_masks.sort(key=lambda m: m["area"], reverse=True)

    instances = []
    t_classify = time.perf_counter()

    for i, sam_mask in enumerate(sam_masks):
        mask_bool = sam_mask["segmentation"]
        feats = compute_mask_features(mask_bool, img_array)
        if feats is None:
            continue

        label, confidence = classify_mask(feats)

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


def instances_to_json(instances: list[InstanceMask]) -> list[dict]:
    """Convert instance list to JSON-serialisable format."""
    result = []
    for inst in instances:
        d = {
            "label": inst.label,
            "confidence": round(inst.confidence, 3),
            "area_px": inst.area,
            "bbox": list(inst.bbox),
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
        if inst.regularised_polygon is not None:
            d["polygon"] = inst.regularised_polygon.reshape(-1, 2).tolist()
        result.append(d)
    return result
