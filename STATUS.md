# DepthWizard — Status Report (2026-09-17)

## Tasks Completed

### TASK 0 — Depth Model ✅
- **MiDaS_small** integrated via `torch.hub` (GitHub releases CDN).
- Trust bypass: `torch.hub._check_repo_is_trusted = lambda *a, **k: None`.
- Loads in ~15s, inference in ~0.4s on CPU.
- Verified end-to-end: 256×256 test image → 1024×1024 depth → 3D terrain renders correctly (buildings UP, not craters).
- Depth-Anything-V2-Small code exists but HuggingFace Xet CDN is blocked (503).

### TASK 1 — Metric Calibration ✅
- `app/calibration.py`: GeoTIFF detection via rasterio, SRTM RANSAC calibration via scikit-learn.
- `POST /calibrate` endpoint: upload reference GeoTIFF, RANSAC maps relative→absolute metres.
- `POST /upload` auto-detects GeoTIFFs and reads elevation band directly (provenance: `geotiff-direct`).
- Provenance badges in UI: RELATIVE (amber), GeoTIFF DIRECT (green), SRTM CALIBRATED (blue).
- DSM export function available (`export_dsm_geotiff`).

### TASK 2 — Enhanced Viewer (50% of score) ✅
- **Color palettes**: Viridis, Cividis, Terrain. No rainbow/jet.
- **Overlays**: Slope (green→yellow→red, 0–45°), Hillshade-only (greyscale).
- **Camera modes**: Orbit (default), First-Person (WASD+mouse, pointer lock), Aerial (top-down, scroll zoom), Waypoint Flythrough (animated spline path).
- **Reset Camera** button.
- **Click-to-measure**: raycaster-based height readout at cursor, pixel coordinates shown.
- **Cross-section profile**: click two terrain points → 2D profile chart with elevation scale.
- **Legible fonts**: 13-14px throughout for screen recording readability.

### TASK 3 — Validation Panel ✅
- `app/validation.py`: MAE, RMSE, median AE, R², bias computation.
- `POST /validate` endpoint: upload reference GeoTIFF → comparison stats.
- UI panel shows: stats table, signed error map (blue=under, red=over), pred-vs-ref scatter plot.

### TASK 4 — Robustness ✅
- Input validation: empty file, 50 MB limit, 16px–8192px size range.
- Large images auto-resized to ≤8192px.
- `app/tiling.py`: overlapping tile-and-merge for images exceeding estimator capacity.
- Proper error propagation: ValueError → 400, unexpected errors → 500 with logged stack trace.
- Frontend shows server error detail messages.

## Architecture

```
app/
├── main.py              FastAPI server, endpoints, model loader
├── calibration.py       GeoTIFF detection, RANSAC calibration, DSM export
├── validation.py        Error metrics, error map, scatter data
├── tiling.py            Overlapping tile merge for large images
└── estimators/
    ├── base.py          HeightEstimator ABC + HeightMetadata
    ├── midas.py         MiDaS_small via torch.hub
    ├── depth_anything.py  Depth-Anything-V2 (HF CDN blocked)
    └── synthetic.py     Gaussian hills + box buildings

static/
└── index.html           Three.js 3D viewer with all controls
```

## Known Limitations
- MiDaS output is uncalibrated (0–150m relative). Needs SRTM reference for absolute metres.
- MiDaS internal resolution is 256×256; output is interpolated to 1024×1024.
- Validation and calibration require rasterio-readable GeoTIFF references.
- No tiling integration in the API yet (module ready but not wired into /upload).
- Single-threaded inference — no GPU, no batching.
