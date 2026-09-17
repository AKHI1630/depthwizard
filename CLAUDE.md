# DepthWizard — SIH 26175

## Project
Single-view height estimation and 3D flythrough.

## Working Directory
Fixed to `C:\Users\rtadi001\ClaudeAI\depthwizard`. Do not create or reference files outside this folder.

## Frontend
- Single `index.html` — no bundler, no build step.
- Three.js r134 and OrbitControls load from cdnjs.cloudflare.com / cdn.jsdelivr.net CDN.

## Architecture
- Height generation is behind a `HeightEstimator` interface (`estimate(image) -> (array, metadata)`).
- Swap implementations in `estimators/` without touching the frontend.
- Post-processing: `app/depth_postprocess.py` — shared orient_depth, highpass_detrend, finalize_heightmap.
- Structure DSM: `app/structure_dsm.py` — SLIC superpixels + vectorized classification.
- Calibration: `app/calibration.py` — GeoTIFF detection, SRTM RANSAC calibration, DSM export.
- Validation: `app/validation.py` — MAE, RMSE, R², error map, scatter plot.
- Tiling: `app/tiling.py` — overlapping tile merge for large images.

## Commands
```bash
# Start backend (from repo root, inside venv) — NO --reload flag
.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8001
```
Server starts immediately; models load in background. Watch the log for:
`DepthWizard ready — Depth-Anything-V2 loaded, accepting requests.`
Poll `GET /health` to check model status programmatically.

## API Endpoints
- `GET /health` — per-model status (`depth_anything`, `midas`)
- `POST /upload?estimator=depth_anything|midas|synthetic&detrend=true|false&structure=true|false` — returns binary heightmap + timing metadata
- `GET /structure-map` — color-coded segmentation PNG (red=building, gray=road, green=veg, brown=ground)
- `POST /calibrate` — upload reference GeoTIFF to RANSAC-calibrate last prediction
- `POST /validate` — upload reference GeoTIFF to compute error stats vs last prediction

## ⚠ Environment constraints — READ BEFORE TOUCHING TEXTURES
- **This machine has no `OES_texture_float` WebGL support.** Float textures silently clamp to 0 → completely flat terrain. This cost a full debug session.
- All height data **must** travel as RG-packed 16-bit in `THREE.RGBAFormat + THREE.UnsignedByteType` with `NearestFilter` on both min and mag filters. The unpack formula in the vertex shader is: `h = (t.r * 255.0 * 256.0 + t.g * 255.0) / 65535.0`.
- **Never reintroduce `FloatType`, `RedFormat`, or `HalfFloatType` textures.**

## ⚠ Network constraints
- **HuggingFace Xet CDN (us.aws.cdn.hf.co) returns 503** on this corporate network. `HF_HUB_DISABLE_XET=1` is set in main.py but may not help.
- **Depth Anything V2 loads from local weights** in `models/depth-anything-v2-small/` — NEVER touch HuggingFace network. `local_files_only=True`.
- **MiDaS via torch.hub works** — GitHub releases CDN is reachable. Trust bypass: `torch.hub._check_repo_is_trusted = lambda *a, **k: None`.
- Do NOT use `--reload` with uvicorn — it watches `.venv/` and restarts during pip installs.

## What's working (as of 2026-09-17)
- **Depth Anything V2 Small** — primary estimator, local weights (0.4s load, 1.3s inference for 512px).
- **MiDaS_small** — secondary estimator via torch.hub (5s load).
- **Sign correction**: Pearson correlation between brightness and depth auto-detects/flips inverted depth.
- **Detrending**: gaussian_filter mode='nearest' + cosine edge taper removes tile rim artifacts.
- **Structure-aware DSM**: SLIC superpixels + vectorized classification → flat roofs, flat terrain, vegetation roughness.
- **Per-stage timing** in response metadata and UI status bar.
- RG-packed 16-bit heightmap rendering — hills and buildings visible.
- **Viewer features**: viridis/cividis/terrain palettes, slope overlay, hillshade-only mode, segmentation overlay.
- **Camera modes**: orbit, first-person (WASD+mouse), aerial (top-down), waypoint flythrough.
- **Click-to-measure**: real-time height readout at cursor position.
- **Cross-section profile**: click two points, see 2D elevation profile.
- **Calibration pipeline**: GeoTIFF auto-detection, SRTM RANSAC calibration, provenance badges.
- **Validation panel**: MAE/RMSE/R²/bias stats, signed error map, pred-vs-ref scatter plot.
- **Robustness**: 50MB upload limit, image size validation (16px–8192px), proper error messages, tiling module.
- Synthetic fallback always available. Estimator dropdown in UI.

## Estimator notes
- **Depth Anything V2 Small** (default): 99.2 MB local weights in `models/depth-anything-v2-small/`. Uses `DepthAnythingForDepthEstimation` (NOT DPTForDepthEstimation). 518×518 tiles, 128px overlap.
- **MiDaS_small**: 81.8 MB from GitHub releases. 256×256 tiles, 64px overlap.
- **Tiling**: images >tile_size are split into overlapping tiles, inferred individually, merged with cosine-feathered blending.
- **Detrending**: High-pass filter (large-sigma Gaussian subtraction) removes low-frequency ramps. `mode='nearest'` + cosine taper to prevent edge artifacts.
- **Sign correction**: Pearson r between image brightness and raw depth; auto-flip if negative.
- Output interpolated to 1024×1024. Uncalibrated 0–150m range.
- Synthetic: deterministic Gaussian hills + box buildings, 1024×1024.

## Structure-aware DSM
- SLIC superpixels (n=200, compactness=20) from scikit-image.
- Vectorized classification using scipy.ndimage.mean/standard_deviation (not regionprops).
- Classes: BUILDING (flat roof=median depth), ROAD/GROUND (base elevation), VEGETATION (median + 30% roughness).
- Thresholds: brightness, texture std, green excess.
- Performance: ~3s for 1024×1024 (SLIC ~2s, classify ~0.2s, compose ~0.6s).

## Performance (512px test image)
- Depth inference (DA V2): ~1.3s
- Structure DSM: ~2.9s (SLIC 2.1 + classify 0.2 + compose 0.6)
- Total: ~5s — well under 30s budget

## Conventions
- Python 3.12.10, venv at `.venv/`, transformers==4.49.0, scikit-image==0.26.0
- Static frontend at `static/`
- Model weights in `models/` (gitignored)
- Do not auto-commit without asking.
- Never embed GitHub tokens in URLs or command lines.
