"""
GAMUS height evaluation for DepthWizard.

Loads GAMUS .h5 image/AGL pairs, runs the pipeline, and computes metrics.
Designed to run the moment paired tiles are available.

Dataset structure (from EarthNets/RSI-MMSegmentation):
  root/images/{split}/{base}IMG.h5  — RGB satellite image, key 'image'
  root/classes/{split}/{base}CLS.h5 — 7-class segmentation, key 'image'
  root/heights/{split}/{base}AGL.h5 — Above Ground Level (m), key 'image'

Class labels: 0=others, 1=ground, 2=low_veg, 3=buildings, 4=water, 5=road, 6=tree

Two evaluation protocols:
  1. Frozen global affine — fit scale+offset on val, apply unchanged to test (headline)
  2. Per-crop oracle — fit affine per crop (ceiling only, never presented as deployment accuracy)

Usage:
  python eval/gamus.py --root /path/to/gamus --split test
  python eval/gamus.py --sanity  # run sanity checks with synthetic data
"""
import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)

NODATA = -9999.0
BUILDING_CLASS = 3
HEIGHT_BANDS = [(0, 3), (3, 10), (10, 20), (20, 50), (50, float("inf"))]
BAND_LABELS = ["0-3m", "3-10m", "10-20m", "20-50m", "50m+"]


def load_h5(path: str) -> np.ndarray:
    """Load a GAMUS .h5 file. Returns the 'image' array."""
    with h5py.File(path, "r") as f:
        return f["image"][()]


def find_triplets(root: str, split: str) -> list[dict]:
    """Find matching IMG/CLS/AGL triplets in a GAMUS directory."""
    img_dir = os.path.join(root, "images", split)
    cls_dir = os.path.join(root, "classes", split)
    hgt_dir = os.path.join(root, "heights", split)

    if not os.path.isdir(img_dir):
        logger.warning("Image directory not found: %s", img_dir)
        return []

    triplets = []
    for img_path in sorted(glob.glob(os.path.join(img_dir, "*IMG.h5"))):
        base = os.path.basename(img_path).replace("IMG.h5", "")
        cls_path = os.path.join(cls_dir, f"{base}CLS.h5")
        agl_path = os.path.join(hgt_dir, f"{base}AGL.h5")

        if not os.path.exists(agl_path):
            logger.warning("Missing AGL for %s", base)
            continue

        triplets.append({
            "base": base,
            "img": img_path,
            "cls": cls_path if os.path.exists(cls_path) else None,
            "agl": agl_path,
        })

    return triplets


def valid_mask(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Boolean mask where both ref and pred are valid (not nodata, not nan)."""
    mask = np.isfinite(ref) & np.isfinite(pred)
    mask &= (ref != NODATA) & (pred != NODATA)
    mask &= (ref > -100) & (ref < 1000)
    return mask


def compute_metrics(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    """MAE, RMSE, R^2, bias over valid pixels."""
    if mask.sum() < 10:
        return {"n": int(mask.sum()), "mae": None, "rmse": None, "r2": None, "bias": None}

    r = ref[mask].astype(np.float64)
    p = pred[mask].astype(np.float64)
    err = p - r

    n = int(mask.sum())
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else None

    return {"n": n, "mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 4) if r2 is not None else None, "bias": round(bias, 3)}


def stratify_by_height(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    """Metrics per height band."""
    results = {}
    for (lo, hi), label in zip(HEIGHT_BANDS, BAND_LABELS):
        band_mask = mask & (ref >= lo) & (ref < hi)
        results[label] = compute_metrics(ref, pred, band_mask)
    return results


def fit_affine(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Fit pred_calibrated = scale * pred + offset via least-squares."""
    if mask.sum() < 20:
        return 1.0, 0.0
    r = ref[mask].astype(np.float64)
    p = pred[mask].astype(np.float64)
    A = np.vstack([p, np.ones_like(p)]).T
    result = np.linalg.lstsq(A, r, rcond=None)
    scale, offset = result[0]
    return float(scale), float(offset)


def apply_affine(pred: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Apply affine calibration."""
    return pred * scale + offset


def run_pipeline_on_image(image: np.ndarray, gsd: float = 0.5) -> Optional[np.ndarray]:
    """Run DepthWizard pipeline on an image and return predicted height map."""
    import io
    import requests
    from PIL import Image

    img = Image.fromarray(image.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    try:
        resp = requests.post(
            "http://127.0.0.1:8002/estimate-heights",
            params={"gsd": gsd, "sam_points": 12},
            files={"file": ("tile.png", buf, "image/png")},
            timeout=120,
        )
        if resp.status_code != 200:
            logger.error("Pipeline failed: HTTP %d", resp.status_code)
            return None

        data = resp.json()
        instances = data.get("instances", [])
        h, w = image.shape[:2]
        height_map = np.zeros((h, w), dtype=np.float32)

        for inst in instances:
            if inst.get("label") != "building" or inst.get("height_m") is None:
                continue
            poly = inst.get("polygon")
            if not poly or len(poly) < 3:
                continue

            import cv2
            pts = np.array(poly, dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            height_map[mask > 0] = inst["height_m"]

        return height_map

    except Exception as e:
        logger.error("Pipeline exception: %s", e)
        return None


def sanity_checks():
    """Verify evaluation framework correctness with synthetic data."""
    print("SANITY CHECKS")
    print("=" * 60)

    # Test 1: reference = prediction -> zero error
    ref = np.random.uniform(0, 30, size=(100, 100)).astype(np.float32)
    pred = ref.copy()
    mask = valid_mask(ref, pred)
    m = compute_metrics(ref, pred, mask)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["bias"] == 0.0, f"FAIL: identical arrays should give 0 error, got {m}"
    assert m["r2"] == 1.0, f"FAIL: identical arrays should give R2=1.0, got {m['r2']}"
    print("[PASS] reference = prediction -> MAE=0, RMSE=0, R2=1.0")

    # Test 2: constant prediction -> hand-checkable baseline
    ref = np.tile(np.array([0, 5, 10, 15, 20], dtype=np.float32), 10)
    pred = np.full_like(ref, 10.0)
    mask = valid_mask(ref, pred)
    m = compute_metrics(ref, pred, mask)
    expected_mae = np.mean(np.abs(pred - ref))
    expected_rmse = np.sqrt(np.mean((pred - ref) ** 2))
    expected_bias = np.mean(pred - ref)
    assert abs(m["mae"] - expected_mae) < 0.001, f"FAIL: MAE {m['mae']} != {expected_mae}"
    assert abs(m["rmse"] - expected_rmse) < 0.001, f"FAIL: RMSE {m['rmse']} != {expected_rmse}"
    assert abs(m["bias"] - expected_bias) < 0.001, f"FAIL: bias {m['bias']} != {expected_bias}"
    print(f"[PASS] constant pred=10 vs ref=[0,5,10,15,20] -> MAE={m['mae']:.1f}, RMSE={m['rmse']:.1f}, bias={m['bias']:.1f}")

    # Test 3: nodata masking
    ref = np.tile(np.array([5.0, NODATA, 10.0, np.nan, 15.0], dtype=np.float32), 10)
    pred = np.tile(np.array([5.0, 8.0, 10.0, 12.0, 15.0], dtype=np.float32), 10)
    mask = valid_mask(ref, pred)
    assert mask.sum() == 30, f"FAIL: nodata mask should keep 30 pixels (3 valid × 10 tiles), got {mask.sum()}"
    m = compute_metrics(ref, pred, mask)
    assert m["mae"] == 0.0, f"FAIL: valid pixels are identical, MAE should be 0, got {m['mae']}"
    print("[PASS] nodata pixels correctly masked out")

    # Test 4: affine calibration recovers known transform
    true_scale, true_offset = 2.5, 3.0
    ref = np.random.uniform(5, 25, size=(1000,)).astype(np.float32)
    raw_pred = (ref - true_offset) / true_scale + np.random.normal(0, 0.01, size=ref.shape).astype(np.float32)
    mask = np.ones(len(ref), dtype=bool)
    scale, offset = fit_affine(ref.reshape(-1), raw_pred.reshape(-1), mask)
    assert abs(scale - true_scale) < 0.1, f"FAIL: recovered scale {scale} != {true_scale}"
    assert abs(offset - true_offset) < 0.5, f"FAIL: recovered offset {offset} != {true_offset}"
    calibrated = apply_affine(raw_pred, scale, offset)
    m_after = compute_metrics(ref, calibrated, mask)
    assert m_after["mae"] < 0.1, f"FAIL: calibrated MAE should be near 0, got {m_after['mae']}"
    print(f"[PASS] affine calibration recovers scale={scale:.2f} offset={offset:.2f} (true: {true_scale}, {true_offset})")

    # Test 5: height band stratification
    ref = np.tile(np.array([1, 2, 5, 12, 25, 60], dtype=np.float32), 10)
    pred = ref + 1.0
    mask = np.ones(len(ref), dtype=bool)
    bands = stratify_by_height(ref, pred, mask)
    assert bands["0-3m"]["n"] == 20
    assert bands["3-10m"]["n"] == 10
    assert bands["10-20m"]["n"] == 10
    assert bands["20-50m"]["n"] == 10
    assert bands["50m+"]["n"] == 10
    print("[PASS] height band stratification correct")

    print("\nAll sanity checks passed.")


def evaluate(root: str, split: str, val_split: str = "val", gsd: float = 0.5, out_dir: str = "eval/results"):
    """Full GAMUS evaluation with both protocols."""
    triplets = find_triplets(root, split)
    if not triplets:
        print(f"No GAMUS triplets found in {root}/{split}")
        return

    val_triplets = find_triplets(root, val_split) if split != val_split else None
    print(f"GAMUS Evaluation — {len(triplets)} crops in {split}")
    if val_triplets:
        print(f"  Calibration set: {len(val_triplets)} crops in {val_split}")
    print()

    # Protocol 1: Fit global affine on val, apply to test
    global_scale, global_offset = 1.0, 0.0
    if val_triplets:
        print("Fitting global affine on validation set...")
        all_ref, all_pred = [], []
        for t in val_triplets:
            ref = load_h5(t["agl"]).astype(np.float32)
            img = load_h5(t["img"])
            pred = run_pipeline_on_image(img, gsd=gsd)
            if pred is None:
                continue
            if pred.shape != ref.shape:
                from PIL import Image
                pred = np.array(Image.fromarray(pred).resize((ref.shape[1], ref.shape[0])))
            mask = valid_mask(ref, pred) & (ref > 0)
            all_ref.append(ref[mask])
            all_pred.append(pred[mask])

        if all_ref:
            cat_ref = np.concatenate(all_ref)
            cat_pred = np.concatenate(all_pred)
            global_scale, global_offset = fit_affine(cat_ref, cat_pred, np.ones(len(cat_ref), dtype=bool))
            print(f"  Global affine: scale={global_scale:.4f}, offset={global_offset:.2f}")
        print()

    # Evaluate test set
    all_results = []
    for t in triplets:
        base = t["base"]
        print(f"--- {base} ---")

        ref = load_h5(t["agl"]).astype(np.float32)
        img = load_h5(t["img"])
        cls = load_h5(t["cls"]) if t["cls"] else None

        pred_raw = run_pipeline_on_image(img, gsd=gsd)
        if pred_raw is None:
            print("  Pipeline failed, skipping")
            continue

        if pred_raw.shape != ref.shape:
            from PIL import Image as PILImage
            pred_raw = np.array(PILImage.fromarray(pred_raw).resize(
                (ref.shape[1], ref.shape[0]), PILImage.NEAREST))

        mask_all = valid_mask(ref, pred_raw) & (ref > 0)
        mask_bldg = mask_all & (cls == BUILDING_CLASS) if cls is not None else mask_all

        # Protocol 1: frozen global affine
        pred_global = apply_affine(pred_raw, global_scale, global_offset)
        m_global_all = compute_metrics(ref, pred_global, mask_all)
        m_global_bldg = compute_metrics(ref, pred_global, mask_bldg)
        bands_global = stratify_by_height(ref, pred_global, mask_bldg)

        # Protocol 2: per-crop oracle affine
        oracle_scale, oracle_offset = fit_affine(ref, pred_raw, mask_all)
        pred_oracle = apply_affine(pred_raw, oracle_scale, oracle_offset)
        m_oracle_all = compute_metrics(ref, pred_oracle, mask_all)
        m_oracle_bldg = compute_metrics(ref, pred_oracle, mask_bldg)

        result = {
            "base": base,
            "shape": list(ref.shape),
            "valid_px": int(mask_all.sum()),
            "building_px": int(mask_bldg.sum()),
            "global_affine": {"scale": global_scale, "offset": global_offset},
            "oracle_affine": {"scale": oracle_scale, "offset": oracle_offset},
            "global_all": m_global_all,
            "global_buildings": m_global_bldg,
            "global_bands": bands_global,
            "oracle_all": m_oracle_all,
            "oracle_buildings": m_oracle_bldg,
        }
        all_results.append(result)

        print(f"  Global affine (headline): MAE={m_global_bldg['mae']}m RMSE={m_global_bldg['rmse']}m R2={m_global_bldg['r2']} bias={m_global_bldg['bias']}m")
        print(f"  Oracle affine (ceiling):  MAE={m_oracle_bldg['mae']}m RMSE={m_oracle_bldg['rmse']}m R2={m_oracle_bldg['r2']} bias={m_oracle_bldg['bias']}m")
        for bl, bm in bands_global.items():
            if bm["n"] and bm["n"] > 0:
                print(f"    {bl}: n={bm['n']} MAE={bm['mae']}m RMSE={bm['rmse']}m")
        print()

    # Summary
    if all_results:
        print("=" * 60)
        print("SUMMARY — Global affine on buildings (headline)")
        print("=" * 60)
        hdr = f"{'Crop':<20} {'n':>8} {'MAE':>7} {'RMSE':>7} {'R2':>7} {'Bias':>7}"
        print(hdr)
        print("-" * len(hdr))
        for r in all_results:
            m = r["global_buildings"]
            print(f"{r['base']:<20} {m['n']:>8} {m['mae'] or '--':>7} {m['rmse'] or '--':>7} {m['r2'] or '--':>7} {m['bias'] or '--':>7}")

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "gamus_evaluation.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="GAMUS height evaluation")
    parser.add_argument("--root", type=str, default="data/gamus", help="GAMUS dataset root")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate")
    parser.add_argument("--val-split", type=str, default="val", help="Split for affine calibration")
    parser.add_argument("--gsd", type=float, default=0.5, help="Ground sampling distance")
    parser.add_argument("--sanity", action="store_true", help="Run sanity checks only")
    parser.add_argument("--out-dir", type=str, default="eval/results")
    args = parser.parse_args()

    if args.sanity:
        sanity_checks()
        return

    evaluate(args.root, args.split, args.val_split, args.gsd, args.out_dir)


if __name__ == "__main__":
    main()
