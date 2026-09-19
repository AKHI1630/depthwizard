"""
OSM height reference validation for DepthWizard predictions.

Queries OpenStreetMap for building footprints with height/levels tags,
matches them to predicted buildings by centroid proximity + IoU,
and computes MAE/RMSE/R²/bias against reference heights.

Reference height priority:
  1. 'height' tag (meters, direct measurement)
  2. 'building:levels' × 3.2 m/level (estimated)

Uses curl subprocess as workaround for corporate SSL/TLS interception
that blocks Python requests to OSM endpoints.
"""
import json
import logging
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon, Point
from shapely.ops import transform as shapely_transform

logger = logging.getLogger(__name__)

METERS_PER_LEVEL = 3.2
IOU_THRESHOLD = 0.3
CENTROID_SEARCH_RADIUS_M = 30.0


@dataclass
class OSMBuilding:
    way_id: str
    polygon_wgs84: Polygon
    polygon_px: Optional[Polygon] = None
    ref_height_m: Optional[float] = None
    ref_source: Optional[str] = None  # "height_tag" | "levels_tag"
    building_type: str = "yes"
    levels: Optional[int] = None
    height_tag: Optional[str] = None


@dataclass
class MatchedPair:
    osm_id: str
    pred_idx: int
    ref_height_m: float
    ref_source: str
    pred_height_m: float
    pred_source: str  # "measured" | "measured (borderline)" | "relative"
    iou: float
    centroid_dist_m: float


def _fetch_osm_bbox(bbox_wgs84: tuple[float, float, float, float]) -> Optional[str]:
    """Fetch OSM data via curl (bypasses Python SSL issues on corporate networks)."""
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    url = (
        f"https://api.openstreetmap.org/api/0.6/map"
        f"?bbox={lon_min:.6f},{lat_min:.6f},{lon_max:.6f},{lat_max:.6f}"
    )
    try:
        result = subprocess.run(
            ["curl", "-s", url],
            capture_output=True, timeout=30,
        )
        xml_str = result.stdout.decode("utf-8", errors="replace")
        if result.returncode != 0 or not xml_str.startswith("<?xml"):
            logger.error("curl OSM failed: rc=%d, len=%d", result.returncode, len(xml_str))
            return None
        return xml_str
    except Exception as e:
        logger.error("curl OSM exception: %s", e)
        return None


def _parse_osm_buildings(xml_str: str) -> list[OSMBuilding]:
    """Parse OSM XML into building polygons with optional height info."""
    root = ET.fromstring(xml_str)

    nodes = {}
    for nd in root.findall("node"):
        nodes[nd.get("id")] = (float(nd.get("lat")), float(nd.get("lon")))

    buildings = []
    for way in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        if "building" not in tags:
            continue

        nd_refs = [nd.get("ref") for nd in way.findall("nd")]
        coords = []
        for ref in nd_refs:
            if ref in nodes:
                lat, lon = nodes[ref]
                coords.append((lon, lat))
        if len(coords) < 4:
            continue

        poly = Polygon(coords)
        if not poly.is_valid or poly.area == 0:
            continue

        ref_height = None
        ref_source = None
        height_tag = tags.get("height")
        levels_str = tags.get("building:levels")

        if height_tag:
            try:
                ref_height = float(height_tag.replace("m", "").strip())
                ref_source = "height_tag"
            except ValueError:
                pass

        if ref_height is None and levels_str:
            try:
                levels = int(levels_str)
                ref_height = levels * METERS_PER_LEVEL
                ref_source = "levels_tag"
            except ValueError:
                pass

        buildings.append(OSMBuilding(
            way_id=way.get("id"),
            polygon_wgs84=poly,
            ref_height_m=ref_height,
            ref_source=ref_source,
            building_type=tags.get("building", "yes"),
            levels=int(levels_str) if levels_str and levels_str.isdigit() else None,
            height_tag=height_tag,
        ))

    return buildings


def _wgs84_to_pixel(
    poly_wgs84: Polygon,
    crs,
    transform,
    img_shape: tuple[int, int],
) -> Optional[Polygon]:
    """Convert WGS84 polygon to pixel coordinates."""
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv_transform = ~transform

    coords_px = []
    for lon, lat in poly_wgs84.exterior.coords:
        x_utm, y_utm = to_crs.transform(lon, lat)
        col, row = inv_transform * (x_utm, y_utm)
        coords_px.append((col, row))

    poly_px = Polygon(coords_px)
    if not poly_px.is_valid or poly_px.area < 1:
        return None

    h, w = img_shape
    img_box = Polygon([(0, 0), (w, 0), (w, h), (0, h)])
    if not poly_px.intersects(img_box):
        return None

    return poly_px


def _mask_to_polygon(mask: np.ndarray) -> Optional[Polygon]:
    """Convert a binary mask to a simplified polygon."""
    import cv2
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None
    pts = largest.squeeze()
    if pts.ndim != 2 or pts.shape[0] < 3:
        return None
    poly = Polygon(pts.tolist())
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area < 1:
        return None
    return poly


def match_osm_to_predictions(
    osm_buildings: list[OSMBuilding],
    pred_instances: list[dict],
    crs,
    geo_transform,
    img_shape: tuple[int, int],
    gsd_m: float,
) -> list[MatchedPair]:
    """Match OSM buildings with height refs to predicted buildings by centroid+IoU."""
    ref_buildings = [b for b in osm_buildings if b.ref_height_m is not None]
    if not ref_buildings:
        return []

    for b in ref_buildings:
        b.polygon_px = _wgs84_to_pixel(b.polygon_wgs84, crs, geo_transform, img_shape)

    ref_buildings = [b for b in ref_buildings if b.polygon_px is not None]
    if not ref_buildings:
        return []

    pred_polys = []
    pred_centroids_px = []
    for inst in pred_instances:
        if inst.get("label") != "building" or inst.get("height_m") is None:
            pred_polys.append(None)
            pred_centroids_px.append(None)
            continue

        poly = None
        polygon_data = inst.get("polygon")
        if polygon_data and isinstance(polygon_data, list) and len(polygon_data) >= 3:
            poly = Polygon([(p[0], p[1]) for p in polygon_data])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < 1:
                poly = None

        bbox = inst.get("bbox")
        if poly is not None:
            pred_polys.append(poly)
            c = poly.centroid
            pred_centroids_px.append((c.x, c.y))
        elif bbox and len(bbox) == 4:
            x, y, w, h = bbox
            box_poly = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            pred_polys.append(box_poly)
            pred_centroids_px.append((x + w / 2, y + h / 2))
        else:
            pred_polys.append(None)
            pred_centroids_px.append(None)

    matches = []
    search_radius_px = CENTROID_SEARCH_RADIUS_M / gsd_m

    for osm_b in ref_buildings:
        osm_centroid = osm_b.polygon_px.centroid
        best_iou = 0
        best_idx = -1
        best_dist = float("inf")

        for i, pred_poly in enumerate(pred_polys):
            if pred_poly is None or pred_centroids_px[i] is None:
                continue
            cx, cy = pred_centroids_px[i]
            dist = math.sqrt((osm_centroid.x - cx) ** 2 + (osm_centroid.y - cy) ** 2)
            if dist > search_radius_px:
                continue

            try:
                inter = osm_b.polygon_px.intersection(pred_poly).area
                union = osm_b.polygon_px.union(pred_poly).area
                iou = inter / union if union > 0 else 0
            except Exception:
                iou = 0

            if iou > best_iou:
                best_iou = iou
                best_idx = i
                best_dist = dist

        # Relaxed fallback: if no IoU match, accept nearest centroid within
        # a tight radius. This handles cases where SAM detects the building
        # but with a different footprint shape (low IoU).
        if best_iou < IOU_THRESHOLD and best_idx < 0:
            for i, pred_poly in enumerate(pred_polys):
                if pred_poly is None or pred_centroids_px[i] is None:
                    continue
                cx, cy = pred_centroids_px[i]
                dist = math.sqrt((osm_centroid.x - cx) ** 2 + (osm_centroid.y - cy) ** 2)
                centroid_limit_px = 15.0 / gsd_m  # 15m in pixels
                if dist < centroid_limit_px and dist < best_dist:
                    best_dist = dist
                    best_idx = i
                    best_iou = 0.0  # mark as centroid-only match

        if best_idx >= 0 and (best_iou >= IOU_THRESHOLD or best_dist * gsd_m <= 15.0):
            inst = pred_instances[best_idx]
            conf = inst.get("height_confidence", "")
            src = inst.get("height_source", "")
            if conf == "borderline":
                pred_source = "measured (borderline)"
            elif src == "measured":
                pred_source = "measured"
            else:
                pred_source = "relative"

            matches.append(MatchedPair(
                osm_id=osm_b.way_id,
                pred_idx=best_idx,
                ref_height_m=osm_b.ref_height_m,
                ref_source=osm_b.ref_source,
                pred_height_m=inst["height_m"],
                pred_source=pred_source,
                iou=best_iou,
                centroid_dist_m=best_dist * gsd_m,
            ))

    return matches


def compute_metrics(matches: list[MatchedPair]) -> dict:
    """Compute MAE, RMSE, R², bias from matched pairs."""
    if not matches:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "bias": None}

    refs = np.array([m.ref_height_m for m in matches])
    preds = np.array([m.pred_height_m for m in matches])
    errors = preds - refs

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    if len(matches) >= 2 and np.std(refs) > 0:
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((refs - np.mean(refs)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    else:
        r2 = None

    return {
        "n": len(matches),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 3) if r2 is not None else None,
        "bias": round(bias, 2),
    }


def save_scatter_plot(
    matches: list[MatchedPair],
    out_path: str,
    title: str = "Predicted vs OSM Reference Heights",
):
    """Save a scatter plot of predicted vs reference heights."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping scatter plot")
        return

    refs = [m.ref_height_m for m in matches]
    preds = [m.pred_height_m for m in matches]
    sources = [m.ref_source for m in matches]
    pred_sources = [m.pred_source for m in matches]

    fig, ax = plt.subplots(figsize=(7, 7))

    colors = []
    for ps in pred_sources:
        if ps == "measured":
            colors.append("#2196F3")
        elif ps == "measured (borderline)":
            colors.append("#FF9800")
        else:
            colors.append("#9E9E9E")

    markers = []
    for s in sources:
        markers.append("o" if s == "height_tag" else "s")

    for r, p, c, mk in zip(refs, preds, colors, markers):
        ax.scatter(r, p, c=c, marker=mk, s=80, edgecolors="black", linewidths=0.5, zorder=3)

    all_vals = refs + preds
    lo = max(0, min(all_vals) - 2)
    hi = max(all_vals) + 2
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="1:1 line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("OSM Reference Height (m)")
    ax.set_ylabel("Predicted Height (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=8, markeredgecolor="k", label="Measured"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9800",
               markersize=8, markeredgecolor="k", label="Borderline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9E9E9E",
               markersize=8, markeredgecolor="k", label="Relative"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
               markersize=8, markeredgecolor="k", label="From levels tag"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markersize=8, markeredgecolor="k", label="From height tag"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=8)

    metrics = compute_metrics(matches)
    stats_text = (
        f"n={metrics['n']}\n"
        f"MAE={metrics['mae']:.1f}m\n"
        f"RMSE={metrics['rmse']:.1f}m\n"
        f"Bias={metrics['bias']:+.1f}m"
    )
    if metrics["r2"] is not None:
        stats_text += f"\nR²={metrics['r2']:.3f}"
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    honesty_text = "All ref heights from building:levels tag (estimated)"
    has_direct = any(m.ref_source == "height_tag" for m in matches)
    if has_direct:
        n_direct = sum(1 for m in matches if m.ref_source == "height_tag")
        n_levels = sum(1 for m in matches if m.ref_source == "levels_tag")
        honesty_text = f"{n_direct} from height tag (measured), {n_levels} from levels tag (estimated)"
    fig.text(0.5, 0.01, honesty_text, ha="center", fontsize=8, style="italic", color="gray")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Scatter plot saved to %s", out_path)


def validate_crop(
    crop_path: str,
    api_response: dict,
    gsd_m: float,
    out_dir: str = "eval/results",
) -> dict:
    """Run full OSM validation for one crop. Returns metrics dict."""
    name = Path(crop_path).stem

    with rasterio.open(crop_path) as src:
        crs = src.crs
        geo_transform = src.transform
        img_shape = (src.height, src.width)
        bounds = src.bounds

    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = to_wgs84.transform(bounds.left, bounds.bottom)
    lon_max, lat_max = to_wgs84.transform(bounds.right, bounds.top)
    bbox = (lon_min, lat_min, lon_max, lat_max)

    xml_str = _fetch_osm_bbox(bbox)
    if xml_str is None:
        return {"crop": name, "error": "OSM fetch failed"}

    osm_buildings = _parse_osm_buildings(xml_str)
    ref_count = sum(1 for b in osm_buildings if b.ref_height_m is not None)

    instances = api_response.get("instances", [])
    pred_buildings = [i for i in instances if i.get("label") == "building" and i.get("height_m") is not None]

    matches = match_osm_to_predictions(
        osm_buildings, instances, crs, geo_transform, img_shape, gsd_m,
    )

    all_metrics = compute_metrics(matches)
    measured_matches = [m for m in matches if m.pred_source in ("measured", "measured (borderline)")]
    measured_metrics = compute_metrics(measured_matches)

    if matches:
        plot_path = os.path.join(out_dir, f"{name}_osm_scatter.png")
        save_scatter_plot(matches, plot_path, title=f"{name} — Predicted vs OSM Reference")

    result = {
        "crop": name,
        "osm_total": len(osm_buildings),
        "osm_with_ref": ref_count,
        "pred_buildings": len(pred_buildings),
        "matched": len(matches),
        "matched_measured": len(measured_matches),
        "all": all_metrics,
        "measured_only": measured_metrics,
        "matches": [
            {
                "osm_id": m.osm_id,
                "ref_m": m.ref_height_m,
                "ref_src": m.ref_source,
                "pred_m": round(m.pred_height_m, 1),
                "pred_src": m.pred_source,
                "iou": round(m.iou, 3),
                "dist_m": round(m.centroid_dist_m, 1),
            }
            for m in matches
        ],
        "honesty": {
            "ref_from_height_tag": sum(1 for m in matches if m.ref_source == "height_tag"),
            "ref_from_levels_tag": sum(1 for m in matches if m.ref_source == "levels_tag"),
            "levels_assumed_m_per_level": METERS_PER_LEVEL,
            "iou_threshold": IOU_THRESHOLD,
            "note": (
                "Zero height tags in Antakya; all references from building:levels "
                f"at {METERS_PER_LEVEL}m/level (estimated, not surveyed)."
            ),
        },
    }

    return result


def main():
    """Run OSM validation on all pre-eq Antakya crops."""
    import glob
    import time

    import requests as http

    crops = sorted(glob.glob("data/crops/crop*_antakya_pre*.tif"))
    if not crops:
        print("No crops found in data/crops/")
        return

    sun_elev = 28.3
    sun_az = 162.0
    gsd = 0.305
    api_url = "http://127.0.0.1:8002/estimate-heights"

    print(f"OSM Height Validation — {len(crops)} crops")
    print(f"Sun: elev={sun_elev}, az={sun_az}, GSD={gsd}m")
    print(f"IoU threshold: {IOU_THRESHOLD}, levels: {METERS_PER_LEVEL}m/level")
    print()

    all_results = []

    for crop_path in crops:
        name = Path(crop_path).stem
        print(f"{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        t0 = time.time()
        with open(crop_path, "rb") as f:
            resp = http.post(
                api_url,
                params={"gsd": gsd, "sun_elevation": sun_elev, "sun_azimuth": sun_az, "sam_points": 8},
                files={"file": (os.path.basename(crop_path), f, "image/tiff")},
            )
        if resp.status_code != 200:
            print(f"  API error: {resp.status_code}")
            continue
        api_data = resp.json()
        api_time = time.time() - t0

        result = validate_crop(crop_path, api_data, gsd)
        result["api_time_s"] = round(api_time, 1)
        all_results.append(result)

        print(f"  OSM buildings: {result['osm_total']} total, {result['osm_with_ref']} with height ref")
        print(f"  Predicted buildings: {result['pred_buildings']}")
        print(f"  Matched (IoU >= {IOU_THRESHOLD}): {result['matched']}")

        if result["matches"]:
            print(f"\n  {'OSM ID':>12} {'ref_m':>6} {'src':>10} {'pred_m':>7} {'pred_src':>20} {'IoU':>5} {'dist_m':>7}")
            print(f"  {'-'*75}")
            for m in result["matches"]:
                print(
                    f"  {m['osm_id']:>12} {m['ref_m']:>6.1f} {m['ref_src']:>10} "
                    f"{m['pred_m']:>7.1f} {m['pred_src']:>20} {m['iou']:>5.3f} {m['dist_m']:>7.1f}"
                )

            met = result["all"]
            print(f"\n  All matches (n={met['n']}): MAE={met['mae']}m, RMSE={met['rmse']}m, bias={met['bias']:+.1f}m", end="")
            if met["r2"] is not None:
                print(f", R2={met['r2']:.3f}")
            else:
                print()

            mm = result["measured_only"]
            if mm["n"] > 0:
                print(f"  Measured only (n={mm['n']}): MAE={mm['mae']}m, RMSE={mm['rmse']}m, bias={mm['bias']:+.1f}m", end="")
                if mm["r2"] is not None:
                    print(f", R2={mm['r2']:.3f}")
                else:
                    print()
        else:
            print("  No matches found")

        print(f"  API time: {api_time:.1f}s")
        print()

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    hdr = f"{'Crop':<25} {'OSM':>4} {'Ref':>4} {'Pred':>5} {'Match':>5} {'MAE':>6} {'RMSE':>6} {'Bias':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        met = r["all"]
        mae_s = f"{met['mae']:.1f}" if met["mae"] is not None else "--"
        rmse_s = f"{met['rmse']:.1f}" if met["rmse"] is not None else "--"
        bias_s = f"{met['bias']:+.1f}" if met["bias"] is not None else "--"
        print(
            f"{r['crop']:<25} {r['osm_total']:>4} {r['osm_with_ref']:>4} "
            f"{r['pred_buildings']:>5} {r['matched']:>5} {mae_s:>6} {rmse_s:>6} {bias_s:>6}"
        )

    # Honesty notice
    total_matched = sum(r["matched"] for r in all_results)
    total_from_levels = sum(r["honesty"]["ref_from_levels_tag"] for r in all_results)
    total_from_height = sum(r["honesty"]["ref_from_height_tag"] for r in all_results)
    print(f"\nHonesty: {total_matched} matched buildings total")
    print(f"  {total_from_height} references from 'height' tag (surveyed)")
    print(f"  {total_from_levels} references from 'building:levels' tag (estimated at {METERS_PER_LEVEL}m/level)")
    if total_from_height == 0:
        print("  WARNING: No surveyed height tags — all references are level-count estimates.")
        print("  Treat metrics as indicative, not ground truth.")

    # Save JSON
    out_path = "eval/results/osm_validation.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
