"""Re-precompute crop2 only. Run after server is ready on port 8002."""
import json, os, time, requests
from PIL import Image

API = "http://127.0.0.1:8002"
OUT = "static/demo"
SCENE = {
    "id": "crop2_antakya",
    "file": "data/crops/crop2_antakya_pre.tif",
    "gsd": 0.305,
    "sun_elevation": 28.1,
    "sun_azimuth": 141.6,
    "label": "Antakya crop2 (measured)",
}

r = requests.get(f"{API}/health", timeout=5)
print("Health:", r.json())

sid = SCENE["id"]
t0 = time.time()
with open(SCENE["file"], "rb") as f:
    resp = requests.post(
        f"{API}/estimate-heights",
        params={
            "gsd": SCENE["gsd"],
            "sun_elevation": SCENE["sun_elevation"],
            "sun_azimuth": SCENE["sun_azimuth"],
            "sam_max_dim": 384,
            "sam_points": 8,
        },
        files={"file": (os.path.basename(SCENE["file"]), f, "image/tiff")},
        timeout=300,
    )
elapsed = time.time() - t0

print(f"HTTP {resp.status_code} in {elapsed:.1f}s")
if resp.status_code != 200:
    print("BODY:", resp.text[:500])
    raise SystemExit(1)

data = resp.json()
cov = data.get("coverage", {})
print(f"Coverage: {json.dumps(cov, indent=2)}")

data["_precomputed"] = True
data["_scene_id"] = sid
data["_scene_label"] = SCENE["label"]
data["_pipeline_time_s"] = round(elapsed, 1)
data["_precomputed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

jp = os.path.join(OUT, f"{sid}.json")
with open(jp, "w") as f:
    json.dump(data, f, separators=(",", ":"))
print(f"Saved {jp} ({os.path.getsize(jp)/1024:.0f} KB)")

# Texture
img = Image.open(SCENE["file"]).convert("RGB")
jpg = os.path.join(OUT, f"{sid}.jpg")
img.save(jpg, "JPEG", quality=85)
print(f"Saved {jpg} ({os.path.getsize(jpg)/1024:.0f} KB)")

# Update manifest — rebuild from all 3 scenes
SCENES = [
    {"id": "crop2_antakya", "label": "Antakya crop2 (measured)", "gsd": 0.305,
     "sun_elevation": 28.1, "sun_azimuth": 141.6, "default": True},
    {"id": "crop1_antakya", "label": "Antakya crop1 (borderline)", "gsd": 0.305,
     "sun_elevation": 28.1, "sun_azimuth": 141.6, "default": False},
    {"id": "crop1_kathmandu", "label": "Kathmandu NE (rejected)", "gsd": 0.5,
     "sun_elevation": 45.0, "sun_azimuth": 180.0, "default": False},
]
manifest = []
for s in SCENES:
    if os.path.exists(os.path.join(OUT, f"{s['id']}.json")):
        manifest.append({
            "id": s["id"], "label": s["label"],
            "json": f"demo/{s['id']}.json", "texture": f"demo/{s['id']}.jpg",
            "gsd": s["gsd"], "sun_elevation": s["sun_elevation"],
            "sun_azimuth": s["sun_azimuth"], "default": s.get("default", False),
        })
mp = os.path.join(OUT, "manifest.json")
with open(mp, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest: {len(manifest)} scenes")
print("DONE")
