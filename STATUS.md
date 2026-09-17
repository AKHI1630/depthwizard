# DepthWizard — Status Report (2026-09-17)

## SIH 26175 Tasks (Steps 1–4)

### STEP 1 — Fix Inverted Depth Sign ✅
- **Commit**: `9dd7824`
- Pearson correlation between brightness and depth: auto-flip when r < 0
- DA V2: r=0.38 raw → 0.64 final; MiDaS: r=0.06 raw → 0.17 final
- gaussian_filter `mode='nearest'` + cosine edge taper fixes detrend rim artifact
- Shared post-processing in `app/depth_postprocess.py`

### STEP 2 — Local Depth Anything V2 ✅
- **Commit**: `79e2274`
- Local weights from github.com/AKHI1630/my-project → `models/depth-anything-v2-small/`
- Uses `DepthAnythingForDepthEstimation` + `AutoImageProcessor`, `local_files_only=True`
- 518×518 tiles, 128px overlap (vs MiDaS 256×256, 64px)
- DA V2 is default; MiDaS kept as secondary; fallback: DA → MiDaS → synthetic
- Load: 0.4s local, no network

### STEP 3 — Structure-Aware DSM ✅
- **Commit**: `fd2e096`
- SLIC superpixels (scikit-image, classical only, no model downloads)
- Classes: BUILDING (flat roof), ROAD (flat terrain), VEGETATION (mild roughness), GROUND (base)
- UI toggle, building count display, color-coded segmentation overlay
- `/structure-map` endpoint returns RGBA PNG

### STEP 4 — Efficiency ✅
- Vectorized classification: scipy.ndimage replaces regionprops (2.19s → 0.21s, 10× speedup)
- Per-stage timing in response metadata (`timing` key in JSON)
- Timing displayed in UI status bar
- Total pipeline: ~5s for 512px image (well under 30s budget)

## Performance (512×512 test image, DA V2 + structure)
| Stage | Time |
|-------|------|
| Depth inference (DA V2) | 1.33s |
| Post-processing (orient + detrend + finalize) | 0.69s |
| Structure DSM total | 2.95s |
|  — SLIC superpixels | 2.11s |
|  — Classification (vectorized) | 0.21s |
|  — Composition | 0.58s |
| **Pipeline total** | **4.98s** |

## Prior Tasks

### TASK 0 — Depth Model ✅
- **MiDaS_small** via `torch.hub` (GitHub releases CDN). Trust bypass applied.
- **Depth Anything V2 Small** from local weights (no HuggingFace network).

### TASK 1 — Metric Calibration ✅
- `app/calibration.py`: GeoTIFF detection via rasterio, SRTM RANSAC calibration.
- `POST /calibrate` endpoint. Provenance badges: RELATIVE / GeoTIFF DIRECT / SRTM CALIBRATED.

### TASK 2 — Enhanced Viewer ✅
- Color palettes: Viridis, Cividis, Terrain.
- Overlays: Slope, Hillshade, Segmentation.
- Camera modes: Orbit, First-Person, Aerial, Waypoint Flythrough.
- Click-to-measure, Cross-section profile.

### TASK 3 — Validation Panel ✅
- `POST /validate`: MAE, RMSE, R², bias, signed error map, scatter plot.

### TASK 4 — Robustness ✅
- Input validation, 50MB limit, tiling for large images, proper error propagation.

## Architecture
```
app/
├── main.py                FastAPI server, dual model loading, timing
├── depth_postprocess.py   orient_depth, highpass_detrend, finalize_heightmap
├── structure_dsm.py       SLIC superpixels, vectorized classification
├── calibration.py         GeoTIFF detection, RANSAC calibration, DSM export
├── validation.py          Error metrics, error map, scatter data
├── tiling.py              Overlapping tile merge with cosine blending
└── estimators/
    ├── base.py            HeightEstimator ABC + HeightMetadata
    ├── depth_anything.py  Depth-Anything-V2-Small (local, 518px tiles)
    ├── midas.py           MiDaS_small (torch.hub, 256px tiles)
    └── synthetic.py       Gaussian hills + box buildings

static/
└── index.html             Three.js 3D viewer with all controls

models/
└── depth-anything-v2-small/   Local weights (gitignored)
```

## Known Limitations
- Output is uncalibrated (0–150m relative) without SRTM reference.
- CPU-only inference — no GPU, no batching.
- SLIC is the bottleneck (~2s for 1024×1024); could be replaced with SEEDS or watershed for speed.
- Structure classification uses fixed thresholds — may not generalize to all satellite imagery.
