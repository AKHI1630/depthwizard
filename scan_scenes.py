"""Scan Maxar Open Data scene collections for acquisition dates and compute sun elevation."""
import json
import math
import sys
import urllib.request

BASE = "https://maxar-opendata.s3.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23/ard/acquisition_collections"

with open(
    r"C:\Users\rtadi001\.claude\projects\c--Users-rtadi001-ClaudeAI-depthwizard"
    r"\dac5a73c-fd0e-49f3-aef8-927fdedf4060\tool-results\webfetch-1789761365604-akcxii.bin",
    "rb",
) as f:
    catalog = json.load(f)

children = [l["href"] for l in catalog.get("links", []) if l.get("rel") == "child"]
scene_ids = [c.split("/")[-1].replace("_collection.json", "") for c in children]

def sun_elevation(lat, lon, dt_str):
    """Quick solar elevation estimate from datetime string."""
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    doy = dt.timetuple().tm_yday
    # Solar declination (Spencer formula simplified)
    B = math.radians((360 / 365) * (doy - 81))
    decl = math.radians(23.45 * math.sin(B))
    # Hour angle
    hour_frac = dt.hour + dt.minute / 60 + dt.second / 3600
    # Equation of time approximation (minutes)
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    # Solar noon in UTC for this longitude
    solar_noon_utc = 12 - lon / 15 - eot / 60
    hour_angle = math.radians(15 * (hour_frac - solar_noon_utc))
    lat_r = math.radians(lat)
    elev = math.asin(
        math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle)
    )
    return math.degrees(elev)

print(f"Scanning {len(scene_ids)} scenes...")
print(f"{'Scene ID':<25} {'Date':>12} {'Time(UTC)':>10} {'Lat':>7} {'Lon':>7} {'SunElev':>8}")
print("-" * 80)

results = []
for sid in scene_ids:
    url = f"{BASE}/{sid}_collection.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            d = json.load(resp)
        temporal = d.get("extent", {}).get("temporal", {}).get("interval", [])
        spatial = d.get("extent", {}).get("spatial", {}).get("bbox", [])
        if not temporal or not temporal[0]:
            continue
        dt_str = temporal[0][0]
        date_part = dt_str[:10]
        time_part = dt_str[11:19]
        # Use center of first bbox
        if spatial:
            bb = spatial[0]
            lat = (bb[1] + bb[3]) / 2
            lon = (bb[0] + bb[2]) / 2
        else:
            lat, lon = 37.5, 37.5
        se = sun_elevation(lat, lon, dt_str)
        results.append((sid, date_part, time_part, lat, lon, se))
        print(f"{sid:<25} {date_part:>12} {time_part:>10} {lat:>7.2f} {lon:>7.2f} {se:>7.1f}°")
    except Exception as e:
        print(f"{sid:<25} ERROR: {e}")

# Filter for Feb 2023 with good sun angle
print("\n\n=== BEST CANDIDATES (Feb 2023, sun elev 25-50 deg) ===")
for sid, dt, tm, lat, lon, se in sorted(results, key=lambda x: x[5]):
    if "2023-02" in dt and 25 <= se <= 50:
        print(f"  {sid}  {dt} {tm}  lat={lat:.2f} lon={lon:.2f}  sun_elev={se:.1f}")
