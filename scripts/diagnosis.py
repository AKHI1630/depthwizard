"""
Section 0 Diagnosis: Does DAv2 correlate disparity with texture variance
rather than actual scene structure on nadir satellite imagery?

Creates a synthetic satellite-like tile with:
- Flat uniform building rooftops (low texture variance)
- High-texture vegetation patches (tree canopy simulation)
- Gray uniform roads
- Mixed-texture ground

Runs DAv2 on it and checks whether raised regions in the disparity map
correspond to vegetation (high texture) rather than buildings (low texture).
"""
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = Path(__file__).parent.parent / "docs"
OUT_DIR.mkdir(exist_ok=True)

SERVER = "http://127.0.0.1:8002"


def make_satellite_tile(size=512):
    """
    Create a synthetic nadir satellite tile with realistic texture properties:
    - Buildings: flat, low-variance rooftops (light gray, slight concrete texture)
    - Vegetation: high-variance, green, organic shapes (simulated canopy)
    - Roads: dark gray, low-variance, elongated
    - Ground: medium texture, brownish
    """
    rng = np.random.RandomState(42)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Ground base: brownish-gray with light noise
    ground_base = np.array([160, 150, 130], dtype=np.uint8)
    noise = rng.randint(-15, 16, (size, size, 3)).astype(np.int16)
    img = np.clip(ground_base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Roads: dark gray, low variance, grid pattern
    road_color = np.array([80, 80, 85])
    road_noise = rng.randint(-5, 6, (size, size, 3)).astype(np.int16)
    road_mask = np.zeros((size, size), dtype=bool)
    # Horizontal roads
    for y in [100, 200, 310, 420]:
        road_mask[y:y+12, :] = True
    # Vertical roads
    for x in [80, 190, 300, 400]:
        road_mask[:, x:x+12] = True
    img[road_mask] = np.clip(road_color + road_noise[road_mask], 0, 255).astype(np.uint8)

    # Buildings: flat, low-variance rooftops (concrete-like)
    buildings = [
        # (x, y, w, h, color_base)
        (20, 20, 50, 60, [200, 195, 190]),    # light concrete
        (120, 25, 55, 50, [185, 180, 175]),    # medium concrete
        (210, 30, 70, 65, [195, 190, 185]),    # large building
        (320, 20, 60, 55, [210, 205, 200]),    # white roof
        (430, 25, 50, 55, [190, 185, 180]),
        (25, 130, 45, 50, [205, 200, 195]),
        (130, 125, 40, 55, [180, 175, 170]),
        (215, 120, 65, 70, [200, 195, 190]),   # warehouse
        (320, 130, 55, 50, [195, 190, 185]),
        (425, 120, 55, 65, [185, 180, 175]),
        (20, 230, 50, 55, [210, 205, 200]),
        (120, 225, 60, 65, [190, 185, 180]),
        (210, 225, 55, 55, [200, 195, 190]),
        (320, 230, 50, 60, [185, 185, 180]),
        (420, 225, 60, 55, [195, 190, 185]),
        (25, 340, 55, 55, [205, 200, 195]),
        (125, 335, 50, 60, [190, 185, 180]),
        (215, 340, 60, 50, [200, 195, 190]),
        (325, 335, 55, 60, [185, 180, 175]),
        (430, 340, 50, 55, [210, 205, 200]),
        (20, 440, 55, 50, [195, 190, 185]),
        (130, 435, 45, 55, [200, 195, 190]),
        (320, 440, 60, 50, [190, 185, 180]),
        (430, 435, 55, 55, [185, 180, 175]),
    ]
    bldg_mask = np.zeros((size, size), dtype=bool)
    for (bx, by, bw, bh, base_col) in buildings:
        # Very low variance — flat concrete rooftop
        bldg_noise = rng.randint(-3, 4, (bh, bw, 3)).astype(np.int16)
        patch = np.clip(np.array(base_col, dtype=np.int16) + bldg_noise, 0, 255).astype(np.uint8)
        img[by:by+bh, bx:bx+bw] = patch
        bldg_mask[by:by+bh, bx:bx+bw] = True

    # Vegetation: high-variance green blobs with organic texture
    veg_centers = [
        (260, 260, 35), (370, 260, 25), (460, 260, 30),
        (55, 280, 20), (170, 280, 25),
        (100, 380, 30), (250, 390, 28), (380, 380, 22),
        (460, 390, 25), (30, 480, 20), (230, 470, 30),
        (350, 470, 22), (170, 170, 18), (440, 170, 20),
        (90, 50, 15), (350, 90, 18),
    ]
    veg_mask = np.zeros((size, size), dtype=bool)
    for (cx, cy, r) in veg_centers:
        yy, xx = np.ogrid[max(0,cy-r):min(size,cy+r), max(0,cx-r):min(size,cx+r)]
        # Irregular blob shape
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        angle = np.arctan2(yy - cy, xx - cx)
        r_var = r * (0.7 + 0.3 * np.sin(3 * angle + rng.uniform(0, 6.28)))
        blob = dist < r_var
        # High-variance green canopy texture
        patch_h = blob.shape[0]
        patch_w = blob.shape[1]
        veg_r = rng.randint(30, 90, (patch_h, patch_w)).astype(np.uint8)
        veg_g = rng.randint(80, 170, (patch_h, patch_w)).astype(np.uint8)
        veg_b = rng.randint(20, 70, (patch_h, patch_w)).astype(np.uint8)
        veg_patch = np.stack([veg_r, veg_g, veg_b], axis=-1)
        y_start = max(0, cy - r)
        x_start = max(0, cx - r)
        img[y_start:y_start+patch_h, x_start:x_start+patch_w][blob] = veg_patch[blob]
        veg_mask[y_start:y_start+patch_h, x_start:x_start+patch_w][blob] = True

    # Ensure buildings override vegetation where they overlap
    for (bx, by, bw, bh, base_col) in buildings:
        bldg_noise = rng.randint(-3, 4, (bh, bw, 3)).astype(np.int16)
        patch = np.clip(np.array(base_col, dtype=np.int16) + bldg_noise, 0, 255).astype(np.uint8)
        img[by:by+bh, bx:bx+bw] = patch
        veg_mask[by:by+bh, bx:bx+bw] = False

    return img, bldg_mask, veg_mask, road_mask


def upload_and_get_raw_disparity(img_array, estimator="depth_anything"):
    """Upload image to server and get raw heightmap back."""
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    resp = requests.post(
        f"{SERVER}/upload",
        params={"estimator": estimator, "detrend": "false", "structure": "false"},
        files={"file": ("diag.png", buf, "image/png")},
    )
    resp.raise_for_status()

    data = resp.content
    meta_len = struct.unpack("<I", data[:4])[0]
    meta = json.loads(data[4:4 + meta_len])
    floats = np.frombuffer(data[4 + meta_len:], dtype=np.float32)
    h = w = int(np.sqrt(len(floats)))
    height_map = floats.reshape(h, w)
    return height_map, meta


def run_diagnosis():
    print("=== Section 0 Diagnosis: DAv2 on nadir satellite imagery ===\n")

    # 1. Create synthetic satellite tile
    print("Creating synthetic satellite tile...")
    tile, bldg_mask, veg_mask, road_mask = make_satellite_tile(512)
    tile_path = OUT_DIR / "diag_input_tile.png"
    Image.fromarray(tile).save(tile_path)
    print(f"  Saved: {tile_path}")

    # Compute texture variance per region for reference
    from scipy.ndimage import uniform_filter
    gray = np.mean(tile.astype(np.float32), axis=2)
    local_mean = uniform_filter(gray, size=7)
    local_sq_mean = uniform_filter(gray**2, size=7)
    local_var = local_sq_mean - local_mean**2

    bldg_var = np.mean(local_var[bldg_mask]) if bldg_mask.any() else 0
    veg_var = np.mean(local_var[veg_mask]) if veg_mask.any() else 0
    road_var = np.mean(local_var[road_mask]) if road_mask.any() else 0
    ground_mask = ~(bldg_mask | veg_mask | road_mask)
    ground_var = np.mean(local_var[ground_mask]) if ground_mask.any() else 0

    print(f"  Texture variance — bldg: {bldg_var:.1f}, veg: {veg_var:.1f}, "
          f"road: {road_var:.1f}, ground: {ground_var:.1f}")

    # 2. Run DAv2
    print("\nRunning Depth Anything V2...")
    try:
        height_map, meta = upload_and_get_raw_disparity(tile, "depth_anything")
    except Exception as e:
        print(f"  DAv2 failed: {e}")
        print("  Falling back to synthetic estimator for comparison...")
        height_map, meta = upload_and_get_raw_disparity(tile, "synthetic")

    print(f"  Height map: {height_map.shape}, range [{height_map.min():.2f}, {height_map.max():.2f}]")
    print(f"  Meta: estimator={meta.get('estimator', 'unknown')}")

    # 3. Resample masks to height_map size
    from PIL import Image as PILImage
    hh, hw = height_map.shape
    def resize_mask(m, target_h, target_w):
        return np.array(PILImage.fromarray(m.astype(np.uint8) * 255).resize(
            (target_w, target_h), PILImage.NEAREST)) > 127

    bm = resize_mask(bldg_mask, hh, hw)
    vm = resize_mask(veg_mask, hh, hw)
    rm = resize_mask(road_mask, hh, hw)
    gm = ~(bm | vm | rm)

    # 4. Measure mean disparity per region
    bldg_disp = np.mean(height_map[bm]) if bm.any() else 0
    veg_disp = np.mean(height_map[vm]) if vm.any() else 0
    road_disp = np.mean(height_map[rm]) if rm.any() else 0
    ground_disp = np.mean(height_map[gm]) if gm.any() else 0

    bldg_std = np.std(height_map[bm]) if bm.any() else 0
    veg_std = np.std(height_map[vm]) if vm.any() else 0

    print(f"\n  Mean disparity — bldg: {bldg_disp:.2f} (std {bldg_std:.2f}), "
          f"veg: {veg_disp:.2f} (std {veg_std:.2f}), "
          f"road: {road_disp:.2f}, ground: {ground_disp:.2f}")

    veg_higher = veg_disp > bldg_disp
    print(f"\n  Vegetation disparity {'>' if veg_higher else '<='} building disparity: "
          f"{'CONFIRMS' if veg_higher else 'CONTRADICTS'} hypothesis")

    # 5. Correlation: texture variance vs disparity
    from scipy.stats import pearsonr
    var_flat = local_var.flatten()
    # Resize height_map to tile size for correlation
    hm_resized = np.array(PILImage.fromarray(
        ((height_map - height_map.min()) / (height_map.max() - height_map.min() + 1e-8) * 255).astype(np.uint8)
    ).resize((512, 512), PILImage.BILINEAR)).astype(np.float32)
    r_val, p_val = pearsonr(var_flat, hm_resized.flatten())
    print(f"  Pearson r(texture_variance, disparity) = {r_val:.4f} (p={p_val:.2e})")

    # 6. Save disparity map visualization
    disp_norm = (height_map - height_map.min()) / (height_map.max() - height_map.min() + 1e-8)
    disp_img = (disp_norm * 255).astype(np.uint8)
    disp_path = OUT_DIR / "diag_dav2_disparity.png"
    PILImage.fromarray(disp_img).save(disp_path)
    print(f"  Saved disparity: {disp_path}")

    # 7. Create overlay: disparity heatmap on source tile
    import colorsys
    disp_colored = np.zeros((*disp_img.shape, 3), dtype=np.uint8)
    for i in range(disp_img.shape[0]):
        for j in range(disp_img.shape[1]):
            v = disp_norm[i, j]
            r, g, b = colorsys.hsv_to_rgb(0.66 * (1 - v), 0.8, 0.9)
            disp_colored[i, j] = [int(r*255), int(g*255), int(b*255)]

    overlay_tile = np.array(PILImage.fromarray(tile).resize((hh, hw), PILImage.BILINEAR))
    overlay = (overlay_tile.astype(np.float32) * 0.5 + disp_colored.astype(np.float32) * 0.5).astype(np.uint8)
    overlay_path = OUT_DIR / "diag_overlay.png"
    PILImage.fromarray(overlay).save(overlay_path)
    print(f"  Saved overlay: {overlay_path}")

    # 8. Write findings
    findings = {
        "texture_variance": {
            "building": round(float(bldg_var), 1),
            "vegetation": round(float(veg_var), 1),
            "road": round(float(road_var), 1),
            "ground": round(float(ground_var), 1),
        },
        "mean_disparity": {
            "building": round(float(bldg_disp), 2),
            "vegetation": round(float(veg_disp), 2),
            "road": round(float(road_disp), 2),
            "ground": round(float(ground_disp), 2),
        },
        "disparity_std": {
            "building": round(float(bldg_std), 2),
            "vegetation": round(float(veg_std), 2),
        },
        "texture_disparity_correlation": round(float(r_val), 4),
        "vegetation_higher_than_buildings": bool(veg_higher),
        "estimator": meta.get("estimator", "unknown"),
    }

    return findings, r_val, veg_higher


if __name__ == "__main__":
    findings, r_val, veg_higher = run_diagnosis()
    print("\n" + json.dumps(findings, indent=2))
