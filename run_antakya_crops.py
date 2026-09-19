"""Run full pipeline on Antakya WV02 crops — low sun, dense urban."""
import glob
import json
import math
import os
import time

import requests

crops = sorted(glob.glob("data/crops/crop*_antakya*.tif"))
print(f"Found {len(crops)} Antakya crops")

# Sun from STAC + pysolar (agree to 0.1 deg)
sun_elev = 38.0
sun_az = 162.1
gsd = 0.305

print(f"Sun: elevation={sun_elev} deg, azimuth={sun_az} deg (SSE)")
print(f"GSD: {gsd} m")
print(f"h_min = 5 * {gsd} * tan({sun_elev}) = {5 * gsd * math.tan(math.radians(sun_elev)):.2f} m")
print(f"Scene: 10300100E18CB600 (WV02), 2023-02-11 08:52 UTC")
print(f"Location: Antakya (Hatay), Turkey (36.22N, 36.15E)")
print(f"pysolar verified: 37.9/162.2 vs STAC 38.0/162.1 (delta 0.1 deg)")
print()

results = []

for crop_path in crops:
    name = os.path.basename(crop_path).replace(".tif", "")
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Processing: {name}")
    print(sep)

    t0 = time.time()
    with open(crop_path, "rb") as f:
        resp = requests.post(
            "http://127.0.0.1:8002/estimate-heights",
            params={
                "gsd": gsd,
                "sun_elevation": sun_elev,
                "sun_azimuth": sun_az,
                "sam_points": 8,
            },
            files={"file": (os.path.basename(crop_path), f, "image/tiff")},
        )
    wall_time = time.time() - t0

    if resp.status_code != 200:
        print(f"  ERROR: HTTP {resp.status_code} - {resp.text[:200]}")
        continue

    d = resp.json()
    cov = d.get("coverage", {})
    sun_resp = d.get("sun", {})
    timing = d.get("stage_timing", {})

    R = cov.get("shadow_coherence_R")
    shadow_ok = cov.get("shadow_reliable")

    all_sources = sun_resp.get("all_sources", {})
    measured_az = None
    if "measured" in all_sources:
        measured_az = all_sources["measured"]

    cross = sun_resp.get("cross_check", {})
    delta = cross.get("azimuth_delta_deg")
    consistent = cross.get("consistent")

    total = cov.get("total_buildings", 0)
    measured_ct = cov.get("directly_measured", 0)
    inferred_ct = cov.get("inferred", 0)
    failed_ct = cov.get("failed", 0)
    relative_ct = cov.get("relative", 0)

    buildings = [
        i for i in d.get("instances", [])
        if i.get("label") == "building" and i.get("height_m") is not None
    ]
    heights = [b["height_m"] for b in buildings]

    result = {
        "name": name,
        "R": R,
        "shadow_ok": shadow_ok,
        "computed_az": sun_az,
        "measured_az": measured_az,
        "delta_deg": delta,
        "consistent": consistent,
        "total": total,
        "measured": measured_ct,
        "relative": relative_ct,
        "inferred": inferred_ct,
        "failed": failed_ct,
        "heights": heights,
        "mean_h": round(sum(heights) / len(heights), 1) if heights else None,
        "min_h": round(min(heights), 1) if heights else None,
        "max_h": round(max(heights), 1) if heights else None,
        "wall_s": round(wall_time, 1),
        "timing": timing,
        "area_hist": cov.get("area_histogram_px"),
        "sun_source": sun_resp.get("source"),
        "sun_elev": sun_resp.get("elevation_deg"),
        "sun_az_used": sun_resp.get("azimuth_deg"),
    }
    results.append(result)

    print(f"  Shadow coherence R:   {R}")
    print(f"  Shadow reliable:      {shadow_ok}")
    print(f"  STAC sun az:          {sun_az} deg")
    print(f"  Measured shadow az:   {measured_az}")
    print(f"  Azimuth delta:        {delta} deg")
    print(f"  Consistent:           {consistent}")
    print(f"  Sun used:             elev={sun_resp.get('elevation_deg')} az={sun_resp.get('azimuth_deg')} source={sun_resp.get('source')}")
    print(f"  Buildings:            {total} total")
    print(f"    Measured:           {measured_ct}")
    print(f"    Relative:           {relative_ct}")
    print(f"    Inferred:           {inferred_ct}")
    print(f"    Failed:             {failed_ct}")
    if heights:
        print(f"  Heights:              mean={result['mean_h']}m  range=[{result['min_h']}, {result['max_h']}]m")
    else:
        print("  Heights:              no heights computed")
    feas = cov.get("shadow_feasibility")
    if feas:
        print(f"  Shadow feasibility:   h_min={feas.get('h_min_m')}m at {feas.get('sun_elevation_deg')} deg")
        if feas.get("warning"):
            print(f"    WARNING: {feas['warning']}")
    print(f"  Area histogram:       {cov.get('area_histogram_px')}")
    print(f"  Wall time:            {wall_time:.1f}s")
    sam_t = timing.get("sam_segmentation_s", "?")
    dav2_t = timing.get("dav2_depth_s", "?")
    h_t = timing.get("height_estimation_s", "?")
    print(f"  Stage timing:         SAM={sam_t}s  DAv2={dav2_t}s  Heights={h_t}s")

sep = "=" * 60
print(f"\n{sep}")
print("SUMMARY TABLE")
print(sep)
hdr = f"{'Crop':<25} {'R':>6} {'OK':>4} {'Delta':>7} {'Bldg':>5} {'Meas':>5} {'Rel':>5} {'Inf':>5} {'Fail':>5} {'Mean_h':>7} {'Time':>6}"
print(hdr)
print("-" * len(hdr))
for r in results:
    d_str = f"{r['delta_deg']:.1f}" if r["delta_deg"] is not None else "?"
    h_str = f"{r['mean_h']:.1f}m" if r["mean_h"] is not None else "--"
    r_str = f"{r['R']:.3f}" if r["R"] is not None else "?"
    ok_str = "YES" if r["shadow_ok"] else "NO"
    print(
        f"{r['name']:<25} {r_str:>6} {ok_str:>4} {d_str:>7} "
        f"{r['total']:>5} {r['measured']:>5} {r['relative']:>5} {r['inferred']:>5} {r['failed']:>5} "
        f"{h_str:>7} {r['wall_s']:>5.0f}s"
    )

# Cross-crop consistency check
print(f"\n{sep}")
print("CROSS-CROP CONSISTENCY CHECK")
print(sep)
crop_azimuths = [(r["name"], r["measured_az"]) for r in results if r["measured_az"] is not None]
if len(crop_azimuths) >= 2:
    azs = [az for _, az in crop_azimuths]
    sin_sum = sum(math.sin(math.radians(a)) for a in azs)
    cos_sum = sum(math.cos(math.radians(a)) for a in azs)
    mean_az = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
    R_cross = math.sqrt(sin_sum**2 + cos_sum**2) / len(azs)

    max_delta = 0
    max_pair = ("", "")
    for i in range(len(crop_azimuths)):
        for j in range(i + 1, len(crop_azimuths)):
            d = abs(crop_azimuths[i][1] - crop_azimuths[j][1])
            if d > 180:
                d = 360 - d
            if d > max_delta:
                max_delta = d
                max_pair = (crop_azimuths[i][0], crop_azimuths[j][0])

    for name_c, az in crop_azimuths:
        d_from_stac = abs(az - sun_az)
        if d_from_stac > 180:
            d_from_stac = 360 - d_from_stac
        print(f"  {name_c}: measured_az = {az:.1f} deg  (delta from STAC: {d_from_stac:.1f} deg)")
    print(f"  Circular mean:       {mean_az:.1f} deg  (STAC: {sun_az} deg)")
    print(f"  Cross-crop R:        {R_cross:.3f}")
    print(f"  Max pairwise delta:  {max_delta:.1f} deg ({max_pair[0]} vs {max_pair[1]})")

    if max_delta > 20:
        print(f"  VERDICT: UNRELIABLE — max delta {max_delta:.1f} > 20 deg")
    else:
        print(f"  VERDICT: CONSISTENT — all crops agree within 20 deg")
        mean_delta = abs(mean_az - sun_az)
        if mean_delta > 180:
            mean_delta = 360 - mean_delta
        if mean_delta <= 20:
            print(f"  Shadow-derived azimuth VALIDATES computed/STAC sun (delta {mean_delta:.1f} deg)")
        else:
            print(f"  WARNING: cross-crop mean {mean_az:.1f} disagrees with STAC {sun_az} by {mean_delta:.1f} deg")
else:
    print(f"  Only {len(crop_azimuths)} crop(s) with measured azimuth")

print(f"\n{sep}")
print("SUN SOURCE SUMMARY")
print(sep)
for r in results:
    print(f"  {r['name']}: source={r['sun_source']}, elev={r['sun_elev']}, az={r['sun_az_used']}")

print(f"\nSTAC sun: elev={sun_elev} az={sun_az} (pysolar: 37.9/162.2)")
print(f"Scene: 10300100E18CB600 (WV02), 2023-02-11 08:52 UTC, Antakya, Turkey")
print(f"GSD: {gsd} m, h_min: {5 * gsd * math.tan(math.radians(sun_elev)):.2f} m")
print(f"\nExpected: Antakya old city is dense low-rise (2-5 storeys = 6-15m).")
print(f"Post-earthquake damage visible — some collapsed structures.")
