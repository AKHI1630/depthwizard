---
title: DepthWizard
emoji: 🏗️
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# DepthWizard — SIH 26175

Single-view metric height estimation and 3D flythrough from satellite imagery.

## Quick Start

```bash
# 1. Clone and enter
git clone <repo-url> && cd depthwizard

# 2. Create venv and install
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Run (models download automatically on first start)
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
# Open http://localhost:7860
```

### Docker

```bash
docker build -t depthwizard .
docker run -p 7860:7860 depthwizard
# Health check: curl http://localhost:7860/health
```

## Pipeline

1. **Upload** satellite/aerial image with GSD and sun elevation
2. **SAM segmentation** detects building footprints (ViT-B, 512px/12pts default)
3. **Shadow detection** measures cast shadow length per building
4. **Height estimation** via shadow formula: `h = shadow_length * GSD * tan(sun_elevation)`
5. **Multi-cue fusion** combines shadow, facade, lean, depth-calibrated, and neighbour prior estimates
6. **3D LOD1 scene** with extruded buildings, trees, source image roof texture, orbit camera

## Results

Tested on 4 pre-earthquake Antakya crops (148 buildings total, ~1024×1024px tiles).

### Detection

| Metric | 384/8 | 512/12 (default) |
|--------|-------|-------------------|
| Centroid recall (≤30m) | — | **96%** |
| Mask IoU recall (≥0.3) | 6.9% | **9.4%** |
| Buildings detected | 24 | 37 |
| Pipeline time | 35.7s | 47.7s |

We locate nearly every building (96% centroid recall); boundary precision is limited in dense fabric (9.4% mask IoU) because SAM merges adjacent buildings into single masks. This is a mask boundary quality problem, not a detection failure. The identified fix is footprint regularisation (minAreaRect replacement where area/rect_area > 0.6).

### Heights

| Metric | Value |
|--------|-------|
| Median building height | 3.4 m (systematic underestimate — see below) |
| Coverage | 37 borderline + 109 relative + 2 inferred + 19 failed of 167 |
| Shadow coherence R (best crop) | 0.681 |

### Per-stage Timing (crop2 at 512/12, warm)

| Stage | Time |
|-------|------|
| SAM segmentation | 29.3s |
| DAv2 depth inference | 13.2s |
| Height estimation | 5.1s |
| **Total** | **47.7s** |

### Honest Limitations

- **Shadow truncation in dense fabric**: `shadow_candidate &= ~all_buildings` (shadow_detection.py:71) excludes shadow pixels falling on neighbouring building roofs. In dense old city fabric (5-10m building spacing), a 10m building casts a 61px shadow but only ~20px falls on open ground — the rest lands on the next building's roof and is masked out. Measured shadow length is shorter than true length, producing ~3m instead of ~10m. This is the dominant error source.
- **SAM merges adjacent buildings**: 96% centroid recall vs 9.4% mask IoU — we find buildings but don't separate touching ones. Footprint regularisation is the identified fix.
- **Sun elevation is multiplicative**: All shadow-based heights scale as `tan(sun_elevation)`. A wrong elevation biases every height by the same factor.
- **No ground truth available**: Antakya OSM has 324 buildings but only 8 with height/levels tags. GAMUS evaluator is ready but needs paired .h5 tiles.

## Resource Requirements

- ~1.8 GB peak RAM (measured at 512/12)
- CPU-only, no GPU required
- Disk: ~600 MB for model weights

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model status (200 when ready, 503 during load) |
| `/upload` | POST | Depth estimation (heightfield + structure DSM) |
| `/estimate-heights` | POST | Full LOD1 pipeline (SAM + shadow + heights) |
| `/structure-map` | GET | Segmentation overlay PNG |
| `/calibrate` | POST | SRTM RANSAC calibration with reference GeoTIFF |
| `/validate` | POST | Error metrics vs reference |
| `/flood` | POST | Flood impact analysis |
| `/demo/list` | GET | Available demo examples |
| `/demo/{file}` | GET | Serve demo image |

## Architecture

```
app/
  main.py                 FastAPI server
  sam_segmentation.py     SAM ViT-B building detection
  shadow_detection.py     Shadow mask + length measurement
  sun_geometry.py         Sun position (metadata/pysolar/measured/slider)
  height_estimation.py    Per-building height pipeline
  height_fusion.py        Multi-cue inverse-variance fusion
  structure_dsm.py        SLIC superpixels + classification
  calibration.py          GeoTIFF + SRTM calibration
  validation.py           Error metrics
  estimators/             Depth model backends (DA-V2, MiDaS, synthetic)

static/
  landing.html            Landing page with demo scene preload
  index.html              3D viewer (Three.js r134, no build step)

eval/
  sam_recall_benchmark.py SAM recall vs OSM footprints
  gamus.py                GAMUS .h5 height validation
  osm_reference.py        OSM height reference matching
```

## Tech Stack

- Python 3.12, FastAPI, uvicorn
- Depth Anything V2 Small (local weights, CPU)
- Segment Anything ViT-B (CPU)
- Three.js r134 (CDN, no build step)
- pysolar for sun geometry
- OpenCV, scikit-image, scipy for image processing
