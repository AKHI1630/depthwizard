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

## P1-P4 Accuracy + Speed Fixes ✅

### P1 — Fix Spiky Rooftops + Height Consistency ✅
- **Commit**: `924c558`
- Diagnosis: within-building std was <0.00002 — flat assignment works. Spikes from adjacent
  superpixels of same building getting different medians (too few superpixels).
- Adaptive superpixel count: (w*h)/4000, clamped 200-1200
- Shape-based classification: solidity + elongation + depth variance (not brightness alone)
- Outlier clamping: 10th-90th percentile clip before taking median
- Adjacent building superpixels merged into unified rooftops (union-find + depth similarity)
- Pearson r improved: 0.64 → 0.87

### P2 — Sharp Walls + Road Classification ✅
- NearestFilter confirmed on height texture (lines 886-887 of index.html)
- No smoothing or blending applied to composed structure-aware heightmap
- Roads classified by elongation + low texture, not brightness alone
- Buildings require high solidity (>0.55) + low elongation (<4) + compact shape

### P3 — Flat-Roof Assertion ✅
- After composing, asserts every building region has within-std < 0.01
- Logs loud warning on violation; all 29 regions pass

### P4 — Speed ✅
- Vectorized shape features (np.minimum.at/np.maximum.at instead of per-segment np.where)
- SLIC max_num_iter=5 (3.4s → 1.0s, no visible quality loss)
- Batch tile inference (stacked forward pass for multi-tile images)
- torch.set_num_threads(cpu_count)
- Tile overlap reduced: 128 → 96px
- Diagnostic dict loop removed (was building 200+ dicts per request)

## Performance (512×512 test image, DA V2 + structure, warm)
| Stage | Before | After P4 | Speedup |
|-------|--------|----------|---------|
| SLIC superpixels | 3.37s | 1.01s | 3.3× |
| Classification | 0.70s | 0.39s | 1.8× |
| Composition | 0.94s | 0.53s | 1.8× |
| **Structure total** | **5.32s** | **2.14s** | **2.5×** |
| Depth inference (DA V2) | 3.07s | 3.22s | — |
| **Pipeline total** | **8.43s** | **5.40s** | **1.6×** |

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
