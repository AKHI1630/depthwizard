# DepthWizard Handover — Shadow-Height Rearchitecture

**Date:** 2026-09-18
**Branch:** `rearchitect/shadow-height`
**Commits:** 9 (e1704d4 → 61b31e2)
**Lines added:** ~2,600 across 17 files

---

## What was done

### Completed items (in order)

| Item | Title | Status | Commit |
|------|-------|--------|--------|
| S0 | DAv2 diagnosis on nadir imagery | Done | e1704d4 |
| P0-1 | SAM ViT-B instance segmentation | Done | 7d0341c |
| P0-2 | Shape + colour classification | Done | 7d0341c |
| P0-3 | Polygon regularisation | Done | 4c6907e |
| P0-4 | Footprint IoU validation | **Skipped** — no real tile | — |
| P0-5 | Shadow detection (LAB/HSV) | Done | 4cf6f6d |
| P0-6 | Sun geometry (3-source hierarchy) | Done | 14e3c88 |
| P0-7 | Shadow ↔ building matching | Done | 3d2c983 |
| P0-8 | Shadow length measurement | Done | 3d2c983 |
| P0-9 | Per-building height + confidence | Done | 837c424 |
| P0-10 | Server endpoints (SAM pipeline) | Done | 837c424 |
| P0-11 | LOD1 Three.js extrusion | Done | 837c424 |
| P1-12 | Flood inundation model | Done | 91f5d18 |
| P1-13 | Water surface + slider | Done | 91f5d18 |
| P1-14 | CSV export | Done | 91f5d18 |
| P1-15 | Landing page with orbital intro | Done | 61b31e2 |

### Not attempted

| Item | Title | Reason |
|------|-------|--------|
| P2-16 | Learned height refinement | Lower priority; all P0/P1 items done first |
| P2-17 | Multi-tile stitching | Lower priority |
| eval/run.py | Evaluation harness | No benchmark dataset downloaded |

---

## Architecture overview

```
User uploads satellite tile
         │
         ▼
┌─────────────────────┐
│ SAM ViT-B           │  ~48s on CPU (16 points, 0 crop layers)
│ Automatic masks     │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ classify_mask()     │  rectangularity, solidity, ExG, texture
│ → building/tree/road│  NO brightness, NO depth thresholds
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ regularise_polygon()│  Douglas-Peucker + angle snapping
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ detect_shadows()    │  LAB L<0.35, HSV S<0.25, adjacency
│ match_to_buildings()│  dilated mask overlap
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ sun_geometry        │  metadata → pysolar → shadow estimate
│ measure_shadow_len()│  median per-scanline shadow length
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ h = L·tan(θ_sun)    │  with uncertainty propagation
│ clamp [1m, 300m]    │  interpolate missing from neighbours
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ LOD1 Three.js       │  ExtrudeGeometry from polygons
│ Flood model         │  per-building inundation at water level W
│ CSV export          │  freeboard, status, area
└─────────────────────┘
```

---

## Files created or modified

### New Python modules
- `app/sam_segmentation.py` — SAM loading, feature computation, classification, polygon regularisation
- `app/shadow_detection.py` — shadow segmentation + building matching
- `app/sun_geometry.py` — pysolar integration + shadow-based estimation
- `app/height_estimation.py` — full pipeline: SAM → shadows → heights
- `app/flood_model.py` — per-building flood inundation

### Modified
- `app/main.py` — new endpoints (`/segment`, `/estimate-heights`, `/shadow-overlay`, `/flood`, root redirect)
- `static/index.html` — LOD1 renderer, flood UI, building-rise animation
- `.gitignore` — added `*.pth`, `*.ckpt`

### New static files
- `static/landing.html` — orbital intro (Three.js Earth sphere)
- `static/assets/earth_2k.jpg` — NASA Blue Marble texture (2.5 MB)

### Documentation
- `docs/DIAGNOSIS.md` — S0 findings
- `docs/WORKLOG.md` — per-item log
- `docs/BLOCKERS.md` — P0-4 deferral
- `scripts/diagnosis.py` — DAv2 validation script

---

## How to run

```bash
cd C:\Users\rtadi001\ClaudeAI\depthwizard
.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8002
```

Open `http://127.0.0.1:8002` — landing page → "Launch Workspace".

SAM weights (`models/sam_vit_b_01ec64.pth`, 375 MB) load in a background thread. The `/segment` endpoint returns 503 until loading completes (~15s). Check `GET /health` for status.

Skip the landing page with `http://127.0.0.1:8002/index.html` or `?fast=1`.

---

## Known limitations

1. **No real satellite tile tested.** All pipeline validation used synthetic imagery. Classification thresholds, shadow detection parameters, and height estimates need validation on real nadir tiles.

2. **SAM is slow on CPU.** ~48s for a 512×512 tile with current settings. With GPU: ~3s. Consider reducing `points_per_side` further or switching to MobileSAM for CPU deployment.

3. **No benchmark dataset.** The handoff brief references IEEE GRSS DFC2019/DFC2018 and ISPRS Vaihingen/Potsdam. None were downloaded. The evaluation harness (`eval/run.py`) was not built.

4. **Shadow detection tuned for synthetic imagery.** The LAB L < 0.35 and HSV S < 0.25 thresholds may need adjustment for different lighting conditions, image histograms, and atmospheric effects.

5. **Flood model assumes flat terrain.** Ground elevation is a single scalar (`ground_elev_m = 0.0`). Real applications need a DEM or per-pixel ground elevation.

6. **Port 8001 still running old server.** The original DAv2-only server may still be listening on port 8001 from a prior session.

---

## What to do next

1. **Upload a real satellite tile** and run the full pipeline (`POST /segment` → `POST /estimate-heights`). Validate building detection, shadow matching, and height estimates against known building heights.

2. **Download benchmark datasets** (Section 3.1 of the handoff brief) and build `eval/run.py`.

3. **P2-16: Learned height refinement** — train a small regression network on (shadow features, shape features) → height for cases where shadow detection fails.

4. **P2-17: Multi-tile stitching** — extend the pipeline to handle images larger than SAM's input size by tiling with overlap and deduplicating instances at tile boundaries.

5. **Push to remote** — all commits are local. Run `git push -u origin rearchitect/shadow-height` when ready.
