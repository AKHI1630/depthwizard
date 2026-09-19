# DepthWizard — SIH 26175

Single-view metric height estimation and 3D flythrough from satellite imagery.

## Quick Start

```bash
# 1. Clone and enter
git clone <repo-url> && cd depthwizard

# 2. Create venv and install
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Download model weights (one-time)
# Depth Anything V2 Small → models/depth-anything-v2-small/
# SAM ViT-B → models/sam_vit_b_01ec64.pth

# 4. Run
uvicorn app.main:app --host 127.0.0.1 --port 8001
# Open http://localhost:8001
```

### Docker

```bash
docker build -t depthwizard .
docker run -p 8001:8001 depthwizard
# Health check: curl http://localhost:8001/health
```

## Pipeline

1. **Upload** satellite/aerial image with GSD and sun elevation
2. **SAM segmentation** detects building footprints (ViT-B, 512px/12pts default)
3. **Shadow detection** measures cast shadow length per building
4. **Height estimation** via shadow formula: `h = shadow_length * GSD * tan(sun_elevation)`
5. **Multi-cue fusion** combines shadow, facade, lean, depth-calibrated, and neighbour prior estimates
6. **3D LOD1 scene** with extruded buildings, trees, source image roof texture, orbit camera

## Results

Tested on 4 pre-earthquake Antakya crops (148 buildings total).

| Metric | Value | Notes |
|--------|-------|-------|
| SAM building recall (IoU >= 0.3) | 9.4% (512/12) | Low due to SAM merging adjacent buildings |
| SAM centroid recall (<= 30m) | 95.9% (768/16) | SAM finds buildings but doesn't separate them |
| Median building height | 3.4 m | Systematic underestimate in dense urban fabric |
| Height coverage | 37 borderline + 109 relative + 2 inferred + 19 failed of 167 | No silent fallbacks |
| Pipeline time (512px) | ~5.4s | CPU-only, no GPU |

### Honest Limitations

- **Shadow truncation**: In dense urban areas (5-10m building spacing), shadows fall ON neighboring building roofs and are excluded by the shadow mask. This underestimates heights by 2-3x. Root cause: `shadow_candidate &= ~all_buildings` in shadow detection.
- **SAM merges adjacent buildings**: IoU-based recall is 10-17% because SAM treats touching buildings as one mask. Centroid-based recall is much higher (96%).
- **Sun elevation is multiplicative**: All shadow-based heights scale as `tan(sun_elevation)`. A wrong elevation biases every height by the same factor.
- **No ground truth available**: Antakya OSM has 324 buildings but only 8 with height/levels tags. GAMUS evaluator is ready but needs paired .h5 tiles.

## Resource Requirements

- ~4 GB RAM (SAM ViT-B 375 MB + Depth Anything V2 99 MB + PyTorch overhead)
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
  landing.html            Animated landing page with demo loader
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
