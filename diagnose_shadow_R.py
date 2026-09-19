"""Diagnose shadow coherence R: per-pair data, weighted R, Rayleigh test."""
import glob
import math
import os
import sys

import cv2
import numpy as np
import rasterio

sys.path.insert(0, ".")
from app.sam_segmentation import segment_and_classify
from app.shadow_detection import detect_shadows, match_shadows_to_buildings, MIN_OFFSET_PX
from app.sun_geometry import estimate_sun_from_shadows

COMPUTED_AZ = 162.0
COMPUTED_EL = 28.3
GSD = 0.305

crops = sorted(glob.glob("data/crops/crop*_antakya_pre*.tif"))
print(f"Diagnosing {len(crops)} crops — computed sun az={COMPUTED_AZ}, el={COMPUTED_EL}\n")


def analyse_pairs(building_masks, shadow_blobs, min_offset=MIN_OFFSET_PX):
    """Extract per-pair diagnostics and compute multiple R variants."""
    pairs = [(b.building_idx, i) for i, b in enumerate(shadow_blobs) if b.building_idx is not None]
    shadow_masks = [b.mask for b in shadow_blobs]

    # Dedup: largest shadow per building
    best = {}
    for bldg_idx, shad_idx in pairs:
        area = int(shadow_masks[shad_idx].sum())
        if bldg_idx not in best or area > best[bldg_idx][1]:
            best[bldg_idx] = (shad_idx, area)
    dedup = [(bi, si) for bi, (si, _) in best.items()]

    rows = []
    for bldg_idx, shad_idx in dedup:
        bm = building_masks[bldg_idx]
        sm = shadow_masks[shad_idx]
        bc = np.argwhere(bm)
        sc = np.argwhere(sm)
        if len(bc) == 0 or len(sc) == 0:
            continue
        b_cen = bc.mean(axis=0)
        s_cen = sc.mean(axis=0)
        dy = s_cen[0] - b_cen[0]
        dx = s_cen[1] - b_cen[1]
        offset = math.sqrt(dx*dx + dy*dy)
        shadow_az = math.degrees(math.atan2(dx, -dy)) % 360
        sun_az = (shadow_az + 180) % 360
        bldg_area = int(bm.sum())
        shad_area = int(sm.sum())
        rows.append({
            "bldg": bldg_idx, "offset": offset,
            "sun_az": sun_az, "shadow_az": shadow_az,
            "bldg_area": bldg_area, "shad_area": shad_area,
        })

    # Filter by min offset
    kept = [r for r in rows if r["offset"] >= min_offset]
    rejected = [r for r in rows if r["offset"] < min_offset]

    if not kept:
        return None

    azimuths = np.array([r["sun_az"] for r in kept])
    offsets = np.array([r["offset"] for r in kept])
    n = len(azimuths)
    az_rad = np.radians(azimuths)

    # --- Scheme 1: Unweighted R (current) ---
    ms = np.mean(np.sin(az_rad))
    mc = np.mean(np.cos(az_rad))
    R_unw = math.sqrt(ms**2 + mc**2)
    mean_az_unw = math.degrees(math.atan2(ms, mc)) % 360

    # --- Scheme 2: Offset-weighted R ---
    w = offsets
    w_sum = w.sum()
    wms = np.sum(w * np.sin(az_rad)) / w_sum
    wmc = np.sum(w * np.cos(az_rad)) / w_sum
    R_wt = math.sqrt(wms**2 + wmc**2)
    mean_az_wt = math.degrees(math.atan2(wms, wmc)) % 360

    # --- Scheme 3: Offset²-weighted R ---
    w2 = offsets ** 2
    w2s = w2.sum()
    w2ms = np.sum(w2 * np.sin(az_rad)) / w2s
    w2mc = np.sum(w2 * np.cos(az_rad)) / w2s
    R_wt2 = math.sqrt(w2ms**2 + w2mc**2)
    mean_az_wt2 = math.degrees(math.atan2(w2ms, w2mc)) % 360

    # --- Scheme 4: Drop short pairs (offset < 10px) ---
    long_mask = offsets >= 10.0
    if long_mask.sum() >= 3:
        az_long = az_rad[long_mask]
        ms_l = np.mean(np.sin(az_long))
        mc_l = np.mean(np.cos(az_long))
        R_long = math.sqrt(ms_l**2 + mc_l**2)
        mean_az_long = math.degrees(math.atan2(ms_l, mc_l)) % 360
        n_long = int(long_mask.sum())
    else:
        R_long = None
        mean_az_long = None
        n_long = int(long_mask.sum())

    # --- Scheme 5: Drop short pairs (offset < 15px) ---
    long15_mask = offsets >= 15.0
    if long15_mask.sum() >= 3:
        az_l15 = az_rad[long15_mask]
        ms_15 = np.mean(np.sin(az_l15))
        mc_15 = np.mean(np.cos(az_l15))
        R_long15 = math.sqrt(ms_15**2 + mc_15**2)
        mean_az_l15 = math.degrees(math.atan2(ms_15, mc_15)) % 360
        n_long15 = int(long15_mask.sum())
    else:
        R_long15 = None
        mean_az_l15 = None
        n_long15 = int(long15_mask.sum())

    # Rayleigh test p-value (approximate: p = exp(-n * R²))
    rayleigh_p_unw = math.exp(-n * R_unw**2)
    rayleigh_p_wt = math.exp(-n * R_wt**2)  # approximate, n is effective sample size

    # Deviation from computed sun
    def delta_from_computed(az):
        d = abs(az - COMPUTED_AZ)
        return min(d, 360 - d)

    return {
        "n_raw": len(rows),
        "n_rejected": len(rejected),
        "n_kept": n,
        "offsets": offsets,
        "azimuths": azimuths,
        "pairs": kept,
        # Current scheme
        "R_unweighted": R_unw,
        "mean_az_unw": mean_az_unw,
        "delta_unw": delta_from_computed(mean_az_unw),
        # Offset-weighted
        "R_offset_wt": R_wt,
        "mean_az_wt": mean_az_wt,
        "delta_wt": delta_from_computed(mean_az_wt),
        # Offset²-weighted
        "R_offset2_wt": R_wt2,
        "mean_az_wt2": mean_az_wt2,
        "delta_wt2": delta_from_computed(mean_az_wt2),
        # Long-only (>=10px)
        "R_long10": R_long,
        "mean_az_long10": mean_az_long,
        "delta_long10": delta_from_computed(mean_az_long) if mean_az_long else None,
        "n_long10": n_long,
        # Long-only (>=15px)
        "R_long15": R_long15,
        "mean_az_long15": mean_az_l15,
        "delta_long15": delta_from_computed(mean_az_l15) if mean_az_l15 else None,
        "n_long15": n_long15,
        # Rayleigh
        "rayleigh_p_unw": rayleigh_p_unw,
        "rayleigh_p_wt": rayleigh_p_wt,
    }


for crop_path in crops:
    name = os.path.basename(crop_path).replace(".tif", "")
    print(f"{'='*70}")
    print(f" {name}")
    print(f"{'='*70}")

    with rasterio.open(crop_path) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)

    instances = segment_and_classify(img, sam_max_dim=384, points_per_side=8)
    building_masks = [inst.mask for inst in instances if inst.label == "building"]
    print(f"  Buildings detected: {len(building_masks)}")

    shadow_blobs = detect_shadows(img, building_masks)
    shadow_blobs = match_shadows_to_buildings(shadow_blobs, building_masks)
    matched = sum(1 for b in shadow_blobs if b.building_idx is not None)
    print(f"  Shadow blobs: {len(shadow_blobs)}, matched: {matched}")

    result = analyse_pairs(building_masks, shadow_blobs)
    if result is None:
        print("  NO usable pairs after dedup + offset filter\n")
        continue

    print(f"\n  Raw pairs: {result['n_raw']}, rejected (offset<{MIN_OFFSET_PX}px): {result['n_rejected']}, kept: {result['n_kept']}")

    # Per-pair table
    print(f"\n  {'#':>3} {'bldg':>5} {'offset':>8} {'sun_az':>8} {'dev':>7} {'bldg_a':>7} {'shad_a':>7}")
    print(f"  {'-'*52}")
    for i, p in enumerate(sorted(result["pairs"], key=lambda x: -x["offset"])):
        dev = abs(p["sun_az"] - COMPUTED_AZ)
        if dev > 180:
            dev = 360 - dev
        marker = " " if dev <= 30 else " *"  # flag outliers
        print(f"  {i+1:>3} {p['bldg']:>5} {p['offset']:>8.1f} {p['sun_az']:>8.1f} {dev:>6.1f}°{marker} {p['bldg_area']:>7} {p['shad_area']:>7}")

    print(f"\n  Offset stats: min={result['offsets'].min():.1f} median={np.median(result['offsets']):.1f} max={result['offsets'].max():.1f} px")

    print(f"\n  {'Scheme':<25} {'R':>6} {'n':>4} {'mean_az':>8} {'delta':>7} {'pass_R05':>8} {'pass_both':>9}")
    print(f"  {'-'*70}")

    schemes = [
        ("Unweighted (current)", result["R_unweighted"], result["n_kept"], result["mean_az_unw"], result["delta_unw"]),
        ("Offset-weighted", result["R_offset_wt"], result["n_kept"], result["mean_az_wt"], result["delta_wt"]),
        ("Offset²-weighted", result["R_offset2_wt"], result["n_kept"], result["mean_az_wt2"], result["delta_wt2"]),
    ]
    if result["R_long10"] is not None:
        schemes.append(("Offset >= 10px", result["R_long10"], result["n_long10"], result["mean_az_long10"], result["delta_long10"]))
    if result["R_long15"] is not None:
        schemes.append(("Offset >= 15px", result["R_long15"], result["n_long15"], result["mean_az_long15"], result["delta_long15"]))

    for label, R, n, az, delta in schemes:
        pass_r = "YES" if R >= 0.5 else "no"
        pass_both = "YES" if R >= 0.5 and delta <= 20 else "no"
        print(f"  {label:<25} {R:>6.3f} {n:>4} {az:>8.1f} {delta:>6.1f}° {pass_r:>8} {pass_both:>9}")

    print(f"\n  Rayleigh test (uniformity rejected = directional signal):")
    print(f"    Unweighted: p = {result['rayleigh_p_unw']:.4f} {'*** SIGNIFICANT' if result['rayleigh_p_unw'] < 0.05 else '(not significant)'}")
    print()
