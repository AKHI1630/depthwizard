"""Pre-compute demo scene results for instant loading.

Runs the full pipeline on each demo scene, saves:
  static/demo/{scene_id}.json   — complete API response
  static/demo/{scene_id}.jpg    — ground texture (JPEG, for roof UV mapping)

These are committed to the repo so the frontend can load them without
running SAM/shadow/height inference.
"""
import io
import json
import os
import sys
import time

import requests
from PIL import Image

API = "http://127.0.0.1:8002"
OUT_DIR = "static/demo"

SCENES = [
    {
        "id": "crop2_antakya",
        "file": "data/crops/crop2_antakya_pre.tif",
        "gsd": 0.305,
        "sun_elevation": 28.3,
        "sun_azimuth": 162.0,
        "label": "Antakya crop2 (measured)",
    },
    {
        "id": "crop1_antakya",
        "file": "data/crops/crop1_antakya_pre.tif",
        "gsd": 0.305,
        "sun_elevation": 28.3,
        "sun_azimuth": 162.0,
        "label": "Antakya crop1 (borderline)",
    },
    {
        "id": "crop1_kathmandu",
        "file": "data/crops/crop1_kathmandu_NE.tif",
        "gsd": 0.5,
        "sun_elevation": 45.0,
        "sun_azimuth": 180.0,
        "label": "Kathmandu NE (rejected)",
    },
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Check server
    try:
        r = requests.get(f"{API}/health", timeout=5)
        status = r.json()
        if status.get("model_status") != "ready":
            print(f"Server not ready: {status}")
            sys.exit(1)
    except Exception as e:
        print(f"Cannot reach server: {e}")
        sys.exit(1)

    for scene in SCENES:
        sid = scene["id"]
        fpath = scene["file"]
        if not os.path.exists(fpath):
            print(f"SKIP {sid}: file not found ({fpath})")
            continue

        print(f"\n=== {sid}: {scene['label']} ===")
        print(f"  File: {fpath}")

        # Run pipeline
        t0 = time.time()
        with open(fpath, "rb") as f:
            resp = requests.post(
                f"{API}/estimate-heights",
                params={
                    "gsd": scene["gsd"],
                    "sun_elevation": scene["sun_elevation"],
                    "sun_azimuth": scene["sun_azimuth"],
                    "sam_max_dim": 512,
                    "sam_points": 12,
                },
                files={"file": (os.path.basename(fpath), f, "image/tiff")},
                timeout=300,
            )
        elapsed = time.time() - t0

        if resp.status_code != 200:
            print(f"  FAILED: HTTP {resp.status_code}")
            continue

        data = resp.json()
        buildings = [i for i in data.get("instances", []) if i.get("label") == "building"]
        cov = data.get("coverage", {})
        print(f"  Buildings: {len(buildings)}")
        print(f"  Coverage: {json.dumps(cov)}")
        print(f"  Pipeline time: {elapsed:.1f}s")

        # Add metadata
        data["_precomputed"] = True
        data["_scene_id"] = sid
        data["_scene_label"] = scene["label"]
        data["_pipeline_time_s"] = round(elapsed, 1)
        data["_precomputed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Save JSON
        json_path = os.path.join(OUT_DIR, f"{sid}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        json_size = os.path.getsize(json_path) / 1024
        print(f"  Saved: {json_path} ({json_size:.0f} KB)")

        # Save ground texture as JPEG
        try:
            img = Image.open(fpath)
            rgb = img.convert("RGB")
            jpg_path = os.path.join(OUT_DIR, f"{sid}.jpg")
            rgb.save(jpg_path, "JPEG", quality=85)
            jpg_size = os.path.getsize(jpg_path) / 1024
            print(f"  Saved: {jpg_path} ({jpg_size:.0f} KB)")
        except Exception as e:
            print(f"  Texture save failed: {e}")

    print("\n=== Done ===")
    # Write manifest
    manifest = []
    for scene in SCENES:
        sid = scene["id"]
        jp = os.path.join(OUT_DIR, f"{sid}.json")
        if os.path.exists(jp):
            manifest.append({
                "id": sid,
                "label": scene["label"],
                "json": f"demo/{sid}.json",
                "texture": f"demo/{sid}.jpg",
                "gsd": scene["gsd"],
                "sun_elevation": scene["sun_elevation"],
                "sun_azimuth": scene["sun_azimuth"],
                "default": sid == "crop2_antakya",
            })
    mpath = os.path.join(OUT_DIR, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {mpath} ({len(manifest)} scenes)")


if __name__ == "__main__":
    main()
