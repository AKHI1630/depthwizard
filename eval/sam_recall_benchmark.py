"""
P1 Benchmark: SAM detection recall vs OSM footprints.

Tests 3 SAM configurations on crop4_antakya_pre.tif (169 OSM buildings).
Reports: building count, recall, precision, runtime, area histogram.
"""
import math
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon

sys.path.insert(0, ".")

CROP = "data/crops/crop4_antakya_pre.tif"
GSD = 0.305
SUN_EL = 28.3
SUN_AZ = 162.0
API = "http://127.0.0.1:8002/estimate-heights"

CONFIGS = [
    {"label": "384/8",  "sam_max_dim": 384, "sam_points": 8},
    {"label": "512/12", "sam_max_dim": 512, "sam_points": 12},
    {"label": "768/16", "sam_max_dim": 768, "sam_points": 16},
]


def get_osm_footprints_px():
    """Fetch all OSM building footprints for crop4, return as pixel-coord Polygons."""
    with rasterio.open(CROP) as src:
        crs = src.crs
        tf = src.transform
        h, w = src.height, src.width
        bounds = src.bounds

    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv_tf = ~tf

    lon_min, lat_min = to_wgs84.transform(bounds.left, bounds.bottom)
    lon_max, lat_max = to_wgs84.transform(bounds.right, bounds.top)

    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={lon_min:.6f},{lat_min:.6f},{lon_max:.6f},{lat_max:.6f}"
    result = subprocess.run(["curl", "-s", url], capture_output=True, timeout=30)
    xml_str = result.stdout.decode("utf-8", errors="replace")
    root = ET.fromstring(xml_str)

    nodes = {}
    for nd in root.findall("node"):
        nodes[nd.get("id")] = (float(nd.get("lat")), float(nd.get("lon")))

    osm_polys = []
    for way in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        if "building" not in tags:
            continue
        nd_refs = [nd.get("ref") for nd in way.findall("nd")]
        coords_px = []
        for ref in nd_refs:
            if ref in nodes:
                lat, lon = nodes[ref]
                x_utm, y_utm = to_crs.transform(lon, lat)
                col, row = inv_tf * (x_utm, y_utm)
                coords_px.append((col, row))
        if len(coords_px) < 4:
            continue
        poly = Polygon(coords_px)
        if poly.is_valid and poly.area > 10:
            img_box = Polygon([(0, 0), (w, 0), (w, h), (0, h)])
            if poly.intersects(img_box):
                osm_polys.append(poly)

    return osm_polys


def run_config(cfg, osm_polys):
    """Run one SAM config via API, compute recall/precision vs OSM."""
    import requests

    t0 = time.time()
    with open(CROP, "rb") as f:
        resp = requests.post(
            API,
            params={
                "gsd": GSD,
                "sun_elevation": SUN_EL,
                "sun_azimuth": SUN_AZ,
                "sam_max_dim": cfg["sam_max_dim"],
                "sam_points": cfg["sam_points"],
            },
            files={"file": (os.path.basename(CROP), f, "image/tiff")},
        )
    wall_time = time.time() - t0

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}", "wall_s": wall_time}

    data = resp.json()
    timing = data.get("stage_timing", {})
    sam_time = timing.get("sam_segmentation_s", "?")

    instances = data.get("instances", [])
    buildings = [i for i in instances if i.get("label") == "building"]
    all_masks = len(instances)

    # Build predicted polygons
    pred_polys = []
    for b in buildings:
        poly_data = b.get("polygon")
        if poly_data and len(poly_data) >= 3:
            p = Polygon([(pt[0], pt[1]) for pt in poly_data])
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_valid and p.area > 0:
                pred_polys.append(p)
                continue
        bbox = b.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            pred_polys.append(Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)]))

    # Match: for each OSM polygon, find best overlapping predicted polygon
    osm_matched = 0
    pred_matched_set = set()
    for osm_p in osm_polys:
        best_iou = 0
        best_idx = -1
        for j, pred_p in enumerate(pred_polys):
            if osm_p.centroid.distance(pred_p.centroid) > 200:
                continue
            try:
                inter = osm_p.intersection(pred_p).area
                union = osm_p.union(pred_p).area
                iou = inter / union if union > 0 else 0
            except Exception:
                iou = 0
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= 0.3:
            osm_matched += 1
            pred_matched_set.add(best_idx)

    recall = osm_matched / len(osm_polys) if osm_polys else 0
    precision = len(pred_matched_set) / len(pred_polys) if pred_polys else 0

    # Area histogram
    areas = sorted([b.get("area_px", 0) for b in buildings])
    if areas:
        pcts = np.percentile(areas, [10, 25, 50, 75, 90])
        area_hist = {
            "p10": int(pcts[0]), "p25": int(pcts[1]), "p50": int(pcts[2]),
            "p75": int(pcts[3]), "p90": int(pcts[4]),
            "min": areas[0], "max": areas[-1],
        }
    else:
        area_hist = {}

    return {
        "label": cfg["label"],
        "all_masks": all_masks,
        "buildings": len(buildings),
        "osm_total": len(osm_polys),
        "osm_matched": osm_matched,
        "recall": recall,
        "precision": precision,
        "wall_s": round(wall_time, 1),
        "sam_s": sam_time,
        "area_hist": area_hist,
    }


def main():
    print("P1 — SAM Detection Recall Benchmark")
    print("=" * 60)
    print(f"Crop: {CROP}")
    print(f"Sun: el={SUN_EL}, az={SUN_AZ}, GSD={GSD}m")
    print()

    print("Fetching OSM footprints...")
    osm_polys = get_osm_footprints_px()
    print(f"OSM buildings in tile: {len(osm_polys)}")
    print()

    results = []
    for cfg in CONFIGS:
        print(f"--- Config: {cfg['label']} (max_dim={cfg['sam_max_dim']}, pts={cfg['sam_points']}) ---")
        r = run_config(cfg, osm_polys)
        results.append(r)

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        print(f"  Total masks: {r['all_masks']}")
        print(f"  Buildings detected: {r['buildings']}")
        print(f"  OSM matched (IoU>=0.3): {r['osm_matched']} / {r['osm_total']}")
        print(f"  Recall:    {r['recall']:.1%}")
        print(f"  Precision: {r['precision']:.1%}")
        print(f"  Wall time: {r['wall_s']}s  (SAM: {r['sam_s']}s)")
        ah = r.get("area_hist", {})
        if ah:
            print(f"  Area histogram: p10={ah['p10']} p25={ah['p25']} p50={ah['p50']} p75={ah['p75']} p90={ah['p90']} range=[{ah['min']},{ah['max']}]")
        print()

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    hdr = f"{'Config':<10} {'Masks':>6} {'Bldgs':>6} {'Match':>6} {'Recall':>8} {'Prec':>8} {'Wall':>6} {'SAM':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r.get('label','?'):<10} ERROR")
            continue
        print(
            f"{r['label']:<10} {r['all_masks']:>6} {r['buildings']:>6} "
            f"{r['osm_matched']:>6} {r['recall']:>7.1%} {r['precision']:>7.1%} "
            f"{r['wall_s']:>5.0f}s {r['sam_s']:>5}s"
        )

    best = None
    for r in results:
        if "error" in r:
            continue
        if r["wall_s"] <= 60 and (best is None or r["recall"] > best["recall"]):
            best = r
    if best:
        print(f"\nRECOMMENDED: {best['label']} — {best['recall']:.1%} recall in {best['wall_s']}s")
    else:
        print("\nNo config fits within 60s budget.")


if __name__ == "__main__":
    main()
